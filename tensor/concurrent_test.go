package tensor_test

import (
	"context"
	"errors"
	"math"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/gsoultan/mimir/nn/loss"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
)

// trainSeparate trains a two-layer model of its own, drawn from seed, for
// steps steps of SGD with momentum, and returns its weights.
func trainSeparate(ctx context.Context, seed uint64, steps int) ([][]float32, error) {
	w1 := tensor.NewParam(tensor.Randn(seed, 16, 8))
	w2 := tensor.NewParam(tensor.Randn(seed+1, 8, 1))
	x, y := tensor.Randn(seed+2, 32, 16), tensor.Randn(seed+3, 32, 1)
	if err := tensor.Realize(ctx, x, y); err != nil {
		return nil, err
	}
	params := []*tensor.Param{w1, w2}
	opt := optim.SGD(params, optim.LR(0.01), optim.Momentum(0.9))
	for range steps {
		l := loss.MSE(x.MatMul(w1.Value()).Tanh().MatMul(w2.Value()), y)
		if err := l.Backward(ctx); err != nil {
			return nil, err
		}
		if err := opt.Step(ctx); err != nil {
			return nil, err
		}
		opt.ZeroGrad()
	}
	var out [][]float32
	for _, p := range params {
		v, err := p.Value().Data[float32](ctx)
		if err != nil {
			return nil, err
		}
		out = append(out, v)
	}
	return out, nil
}

// near reports whether got and want agree to a relative tolerance: the
// kernels of schedules that run at once may split their loops otherwise,
// and so round sums otherwise.
func nearAll(got, want []float32) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if d := math.Abs(float64(got[i] - want[i])); d > 1e-4*(1+math.Abs(float64(want[i]))) {
			return false
		}
	}
	return true
}

// Models trained on separate goroutines at once, forward, backward and
// optimizer steps, end with the weights they have when trained one after
// another.
func TestConcurrentSeparateModels(t *testing.T) {
	const models, steps = 4, 8
	ctx := context.Background()
	want := make([][][]float32, models)
	for i := range models {
		w, err := trainSeparate(ctx, uint64(10*i+1), steps)
		if err != nil {
			t.Fatal(err)
		}
		want[i] = w
	}
	got := make([][][]float32, models)
	errs := make([]error, models)
	var wg sync.WaitGroup
	for i := range models {
		wg.Go(func() { got[i], errs[i] = trainSeparate(ctx, uint64(10*i+1), steps) })
	}
	wg.Wait()
	for i := range models {
		if errs[i] != nil {
			t.Fatalf("model %d: %v", i, errs[i])
		}
		for j := range want[i] {
			if !nearAll(got[i][j], want[i][j]) {
				t.Errorf("model %d, weight %d: trained at once %v, one after another %v", i, j, got[i][j][:4], want[i][j][:4])
			}
		}
	}
}

// Goroutines whose graphs share an input, a realized one that requires
// gradients and one not computed yet, compute their own losses and
// gradients, and add to the shared input's gradient one after another.
func TestConcurrentSharedInput(t *testing.T) {
	const workers, reps, rows, in, out = 4, 3, 16, 8, 4
	ctx := context.Background()
	x := tensor.Randn(1, rows, in).RequiresGrad()
	if err := x.Realize(ctx); err != nil {
		t.Fatal(err)
	}
	// lazy is computed by whichever schedule reaches it first; the
	// reference is the same values in a graph of its own.
	lazy := tensor.Randn(2, rows, in).Mul(tensor.Full(tensor.Float32, 2, rows, in))
	ref := tensor.Randn(2, rows, in).Mul(tensor.Full(tensor.Float32, 2, rows, in))
	ws := make([]*tensor.Param, workers)
	for i := range ws {
		ws[i] = tensor.NewParam(tensor.Randn(uint64(10+i), in, out))
	}
	xv, err := x.Detach().Data[float32](ctx)
	if err != nil {
		t.Fatal(err)
	}
	rv, err := ref.Data[float32](ctx)
	if err != nil {
		t.Fatal(err)
	}
	wv := make([][]float32, workers)
	for i, w := range ws {
		if wv[i], err = w.Value().Data[float32](ctx); err != nil {
			t.Fatal(err)
		}
	}

	// The loss of worker i is the sum of (x+lazy)·w_i. Its gradient for
	// w_i[k][o] is the sum over rows of (x+lazy)[r][k], and for x[r][k]
	// the sum over o of w_i[k][o].
	colSum := make([]float32, in)
	var total float64
	for r := range rows {
		for k := range in {
			colSum[k] += xv[r*in+k] + rv[r*in+k]
		}
	}
	wantLoss := make([]float32, workers)
	wantW := make([][]float32, workers)
	wantX := make([]float32, rows*in)
	for i := range workers {
		wantW[i] = make([]float32, in*out)
		for k := range in {
			var rowSum float32
			for o := range out {
				wantW[i][k*out+o] = reps * colSum[k]
				rowSum += wv[i][k*out+o]
				total += float64(colSum[k] * wv[i][k*out+o])
			}
			for r := range rows {
				wantX[r*in+k] += reps * rowSum
			}
		}
		wantLoss[i] = float32(total)
		total = 0
	}

	errs := make([]error, workers)
	var wg sync.WaitGroup
	for i := range workers {
		wg.Go(func() {
			for range reps {
				l := x.Add(lazy).MatMul(ws[i].Value()).Sum()
				if err := l.Backward(ctx); err != nil {
					errs[i] = err
					return
				}
				v, err := l.Data[float32](ctx)
				if err != nil {
					errs[i] = err
					return
				}
				if !nearAll(v, wantLoss[i:i+1]) {
					t.Errorf("worker %d: loss %v, want %v", i, v[0], wantLoss[i])
				}
				// A forward pass alone reads the shared input too.
				y, err := x.Detach().MatMul(ws[i].Value().Detach()).Data[float32](ctx)
				if err != nil {
					errs[i] = err
					return
				}
				if len(y) != rows*out {
					t.Errorf("worker %d: %d elements, want %d", i, len(y), rows*out)
				}
			}
		})
	}
	wg.Wait()
	for i, err := range errs {
		if err != nil {
			t.Fatalf("worker %d: %v", i, err)
		}
	}
	for i, w := range ws {
		g, err := w.Grad().Data[float32](ctx)
		if err != nil {
			t.Fatal(err)
		}
		if !nearAll(g, wantW[i]) {
			t.Errorf("gradient of w%d: %v, want %v", i, g[:4], wantW[i][:4])
		}
	}
	g, err := x.Grad().Data[float32](ctx)
	if err != nil {
		t.Fatal(err)
	}
	if !nearAll(g, wantX) {
		t.Errorf("gradient of the shared input: %v, want %v", g[:4], wantX[:4])
	}
}

// Readers of a tensor that a writer keeps writing over in place see each
// of its values whole, or ErrOverwritten for one already written over:
// a schedule does not read memory while another writes it.
func TestConcurrentWriteOver(t *testing.T) {
	const n, writes, readers = 1 << 20, 100, 3
	ctx := context.Background()
	var mu sync.Mutex
	counter := tensor.FromSlice(make([]float32, n), n)
	if err := counter.Realize(ctx); err != nil {
		t.Fatal(err)
	}
	current := func() *tensor.Tensor {
		mu.Lock()
		defer mu.Unlock()
		return counter
	}
	var done atomic.Bool
	var wg sync.WaitGroup
	wg.Go(func() {
		defer done.Store(true)
		for range writes {
			mu.Lock()
			next := counter.Add(tensor.Full(tensor.Float32, 1, n))
			err := tensor.RealizeInto(ctx, []*tensor.Tensor{counter}, []*tensor.Tensor{next})
			if err == nil {
				counter = next
			}
			mu.Unlock()
			if err != nil {
				t.Error(err)
				return
			}
		}
	})
	for range readers {
		wg.Go(func() {
			for !done.Load() {
				v, err := current().Mul(tensor.Full(tensor.Float32, 2, n)).Data[float32](ctx)
				if errors.Is(err, tensor.ErrOverwritten) {
					continue
				}
				if err != nil {
					t.Error(err)
					return
				}
				for i := range v {
					if v[i] != v[0] {
						t.Errorf("read a tensor being written: element %d is %v, element 0 %v", i, v[i], v[0])
						return
					}
				}
			}
		})
	}
	wg.Wait()
	v, err := current().Data[float32](ctx)
	if err != nil {
		t.Fatal(err)
	}
	if v[0] != writes || v[n-1] != writes {
		t.Errorf("after %d writes: %v, %v", writes, v[0], v[n-1])
	}
}
