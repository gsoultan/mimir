package tensor_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/backend/cpu"
	"github.com/gsoultan/mimir/tensor"
)

// mlp is a block of two linear layers around a ReLU, with its weights
// held as parameters, as a layer holds them.
type mlp struct {
	w1, b1, w2, b2 *tensor.Param
}

func newMLP(seed uint64, in, hidden, out int) *mlp {
	p := func(seed uint64, shape ...int) *tensor.Param {
		return tensor.NewParam(tensor.Randn(seed, shape...).Mul(tensor.Scalar[float32](0.5)))
	}
	return &mlp{w1: p(seed+1, hidden, in), b1: p(seed+2, hidden), w2: p(seed+3, out, hidden), b2: p(seed+4, out)}
}

func (m *mlp) params() []*tensor.Param { return []*tensor.Param{m.w1, m.b1, m.w2, m.b2} }

func (m *mlp) forward(in ...*tensor.Tensor) *tensor.Tensor {
	h := in[0].MatMul(m.w1.Value().Transpose(0, 1)).Add(m.b1.Value()).ReLU()
	return h.MatMul(m.w2.Value().Transpose(0, 1)).Add(m.b2.Value())
}

// gradients runs loss(x) through Backward and returns the loss value and
// the gradients of x, if it has one, then of each of ps, which it zeroes.
func gradients(t *testing.T, loss func(x *tensor.Tensor) *tensor.Tensor, x *tensor.Tensor, ps ...*tensor.Param) (float32, [][]float32) {
	t.Helper()
	l := loss(x)
	if err := l.Backward(t.Context()); err != nil {
		t.Fatal(err)
	}
	v, err := l.Item[float32](t.Context())
	if err != nil {
		t.Fatal(err)
	}
	var gs [][]float32
	if x.Tracked() {
		if x.Grad() == nil {
			t.Fatal("input has no gradient")
		}
		gs = append(gs, data[float32](t, x.Grad()))
		x.ZeroGrad()
	}
	for i, p := range ps {
		if p.Grad() == nil {
			t.Fatalf("parameter %d has no gradient", i)
		}
		gs = append(gs, data[float32](t, p.Grad()))
		p.ZeroGrad()
	}
	return v, gs
}

// sameGradients checks that a checkpointed computation gives the loss and
// gradients of the plain one. The same kernels run on the same values in
// the same order, so the results are equal bit for bit.
func sameGradients(t *testing.T, plain, checkpointed func(x *tensor.Tensor) *tensor.Tensor, x *tensor.Tensor, ps ...*tensor.Param) {
	t.Helper()
	wantLoss, want := gradients(t, plain, x, ps...)
	gotLoss, got := gradients(t, checkpointed, x, ps...)
	if gotLoss != wantLoss {
		t.Errorf("loss %v, want %v", gotLoss, wantLoss)
	}
	for i := range want {
		if !slices.Equal(got[i], want[i]) {
			t.Errorf("gradient %d differs:\n got %v\nwant %v", i, got[i], want[i])
		}
	}
}

func TestCheckpointGradientsMatchPlain(t *testing.T) {
	m := newMLP(1, 4, 6, 3)
	x := tensor.Randn(7, 5, 4)
	loss := func(f func(in ...*tensor.Tensor) *tensor.Tensor) func(x *tensor.Tensor) *tensor.Tensor {
		return func(x *tensor.Tensor) *tensor.Tensor { return f(x).Mul(f(x).Sigmoid()).Sum() }
	}
	ckpt := func(in ...*tensor.Tensor) *tensor.Tensor { return tensor.Checkpoint(m.forward, in...) }
	t.Run("tracked input", func(t *testing.T) {
		sameGradients(t, loss(m.forward), loss(ckpt), x.RequiresGrad(), m.params()...)
	})
	t.Run("untracked input", func(t *testing.T) {
		// The parameters are read only inside the checkpointed function,
		// so the checkpoint has no tracked input.
		sameGradients(t, loss(m.forward), loss(ckpt), x, m.params()...)
	})
	t.Run("no input", func(t *testing.T) {
		plain := func(*tensor.Tensor) *tensor.Tensor { return m.forward(x).Sum() }
		ckpt := func(*tensor.Tensor) *tensor.Tensor {
			return tensor.Checkpoint(func(...*tensor.Tensor) *tensor.Tensor { return m.forward(x) }).Sum()
		}
		sameGradients(t, plain, ckpt, x, m.params()...)
	})
	t.Run("nested", func(t *testing.T) {
		inner := newMLP(2, 3, 5, 4)
		plain := func(x *tensor.Tensor) *tensor.Tensor { return m.forward(inner.forward(x)).Sum() }
		ckpt := func(x *tensor.Tensor) *tensor.Tensor {
			return tensor.Checkpoint(func(in ...*tensor.Tensor) *tensor.Tensor {
				return m.forward(tensor.Checkpoint(inner.forward, in[0]))
			}, x).Sum()
		}
		x := tensor.Randn(8, 5, 3).RequiresGrad()
		sameGradients(t, plain, ckpt, x, slices.Concat(inner.params(), m.params())...)
	})
	t.Run("nested reading the outer input", func(t *testing.T) {
		// The inner function reads the outer one's input from its
		// closure: a leaf while the outer one is recomputed, whose
		// gradient must reach the outer input.
		inner := newMLP(5, 4, 5, 4)
		plain := func(x *tensor.Tensor) *tensor.Tensor { return inner.forward(x).Mul(x).Sum() }
		ckpt := func(x *tensor.Tensor) *tensor.Tensor {
			return tensor.Checkpoint(func(in ...*tensor.Tensor) *tensor.Tensor {
				return tensor.Checkpoint(func(j ...*tensor.Tensor) *tensor.Tensor { return inner.forward(j[0]).Mul(in[0]) }, in[0])
			}, x).Sum()
		}
		sameGradients(t, plain, ckpt, x.RequiresGrad(), inner.params()...)
	})
	t.Run("shared parameter", func(t *testing.T) {
		// Two checkpoints of the same block, so its parameters get a
		// gradient from each recompute.
		sq := newMLP(3, 4, 6, 4)
		plain := func(x *tensor.Tensor) *tensor.Tensor { return sq.forward(sq.forward(x)).Sum() }
		ckpt := func(x *tensor.Tensor) *tensor.Tensor {
			return tensor.Checkpoint(sq.forward, tensor.Checkpoint(sq.forward, x)).Sum()
		}
		sameGradients(t, plain, ckpt, x.RequiresGrad(), sq.params()...)
	})
	t.Run("parameter shared with the outer graph", func(t *testing.T) {
		plain := func(x *tensor.Tensor) *tensor.Tensor { return m.forward(x).Sum().Add(m.w1.Value().Sum()) }
		ckpt := func(x *tensor.Tensor) *tensor.Tensor { return ckpt(x).Sum().Add(m.w1.Value().Sum()) }
		sameGradients(t, plain, ckpt, x.RequiresGrad(), m.params()...)
	})
	t.Run("output used twice", func(t *testing.T) {
		use := func(y *tensor.Tensor) *tensor.Tensor { return y.Mul(y).Sum().Add(y.Exp().Mean()) }
		plain := func(x *tensor.Tensor) *tensor.Tensor { return use(m.forward(x)) }
		ckpt := func(x *tensor.Tensor) *tensor.Tensor { return use(ckpt(x)) }
		sameGradients(t, plain, ckpt, x.RequiresGrad(), m.params()...)
	})
	t.Run("input passed twice", func(t *testing.T) {
		sq := newMLP(4, 3, 4, 3)
		f := func(in ...*tensor.Tensor) *tensor.Tensor { return in[0].Mul(in[1]).Add(sq.forward(in[0])) }
		plain := func(x *tensor.Tensor) *tensor.Tensor { return f(x, x).Sum() }
		ckpt := func(x *tensor.Tensor) *tensor.Tensor { return tensor.Checkpoint(f, x, x).Sum() }
		sameGradients(t, plain, ckpt, tensor.Randn(9, 5, 3).RequiresGrad(), sq.params()...)
	})
}

func TestCheckpointRecomputesOncePerBackward(t *testing.T) {
	w := tensor.NewParam(tensor.FromSlice([]float32{1, 2, 3}))
	calls := 0
	f := func(in ...*tensor.Tensor) *tensor.Tensor {
		calls++
		return in[0].Mul(w.Value())
	}
	x := tensor.FromSlice([]float32{4, 5, 6}).RequiresGrad()
	y := tensor.Checkpoint(f, x)
	if calls != 1 {
		t.Fatalf("Checkpoint called f %d times, want 1", calls)
	}
	// Two uses of the result sum their gradients before the one recompute.
	if err := y.Sum().Add(y.Mul(y).Sum()).Backward(t.Context()); err != nil {
		t.Fatal(err)
	}
	if calls != 2 {
		t.Errorf("Backward called f %d more times, want 1", calls-1)
	}
	// d/dx (xw + (xw)²) = w + 2xw²; d/dw = x + 2wx².
	if got := data[float32](t, x.Grad()); !slices.Equal(got, []float32{9, 42, 111}) {
		t.Errorf("x gradient %v, want [9 42 111]", got)
	}
	if got := data[float32](t, w.Grad()); !slices.Equal(got, []float32{36, 105, 222}) {
		t.Errorf("w gradient %v, want [36 105 222]", got)
	}
	if err := y.Sum().Backward(t.Context()); !errors.Is(err, tensor.ErrNoGrad) {
		t.Errorf("second Backward: %v, want ErrNoGrad", err)
	}
}

func TestCheckpointErrors(t *testing.T) {
	ctx := t.Context()
	x := tensor.FromSlice([]float32{1, 2}).RequiresGrad()
	h := x.Mul(x) // tracked, and not a leaf
	t.Run("captured at the call", func(t *testing.T) {
		y := tensor.Checkpoint(func(in ...*tensor.Tensor) *tensor.Tensor { return in[0].Add(h) }, x)
		if err := y.Err(); !errors.Is(err, tensor.ErrCheckpoint) {
			t.Errorf("Checkpoint: %v, want ErrCheckpoint", err)
		}
		if err := y.Sum().Backward(ctx); !errors.Is(err, tensor.ErrCheckpoint) {
			t.Errorf("Backward: %v, want ErrCheckpoint", err)
		}
	})
	t.Run("captured in the recompute", func(t *testing.T) {
		// A function that is not deterministic: it reads h only the
		// second time, which Backward sees.
		calls := 0
		f := func(in ...*tensor.Tensor) *tensor.Tensor {
			if calls++; calls > 1 {
				return in[0].Add(h)
			}
			return in[0].Add(h.Detach())
		}
		y := tensor.Checkpoint(f, x)
		if err := y.Err(); err != nil {
			t.Fatal(err)
		}
		if err := y.Sum().Backward(ctx); !errors.Is(err, tensor.ErrCheckpoint) {
			t.Errorf("Backward: %v, want ErrCheckpoint", err)
		}
		if x.Grad() != nil {
			t.Error("a failed Backward changed a gradient")
		}
	})
	t.Run("captured leaf", func(t *testing.T) {
		// A leaf is a parameter as far as Checkpoint knows: allowed.
		y := tensor.Checkpoint(func(in ...*tensor.Tensor) *tensor.Tensor { return in[0].Mul(x) }, h.Detach())
		if err := y.Sum().Backward(ctx); err != nil {
			t.Fatal(err)
		}
		if got := data[float32](t, x.Grad()); !slices.Equal(got, []float32{1, 4}) {
			t.Errorf("gradient %v, want x² = [1 4]", got)
		}
		x.ZeroGrad()
	})
	t.Run("failed function", func(t *testing.T) {
		bad := func(in ...*tensor.Tensor) *tensor.Tensor { return in[0].Reshape(3) }
		if err := tensor.Checkpoint(bad, x).Err(); !errors.Is(err, tensor.ErrShape) {
			t.Errorf("Checkpoint of a failing function: %v, want ErrShape", err)
		}
		var null *tensor.Tensor
		if err := tensor.Checkpoint(bad, null).Err(); !errors.Is(err, tensor.ErrNilTensor) {
			t.Errorf("Checkpoint of nil: %v, want ErrNilTensor", err)
		}
		if err := tensor.Checkpoint(nil, x).Err(); !errors.Is(err, tensor.ErrInternal) {
			t.Errorf("Checkpoint(nil): %v, want ErrInternal", err)
		}
		if err := tensor.Checkpoint(func(...*tensor.Tensor) *tensor.Tensor { return nil }, x).Err(); !errors.Is(err, tensor.ErrNilTensor) {
			t.Errorf("Checkpoint of a function returning nil: %v, want ErrNilTensor", err)
		}
	})
	t.Run("untracked", func(t *testing.T) {
		y := tensor.Checkpoint(func(in ...*tensor.Tensor) *tensor.Tensor { return in[0].Exp() }, h.Detach())
		if y.Tracked() {
			t.Error("checkpoint of untracked inputs is tracked")
		}
		if got := data[float32](t, y); len(got) != 2 {
			t.Errorf("values %v", got)
		}
	})
}

// within returns what fn returns, or fails the test if fn has not
// returned after a minute: a Backward that deadlocks would otherwise hang
// until the test binary times out. The goroutine of a deadlocked fn is
// left behind, blocked, which a failed test can afford.
func within(t *testing.T, fn func() error) error {
	t.Helper()
	done := make(chan error, 1)
	go func() { done <- fn() }()
	select {
	case err := <-done:
		return err
	case <-time.After(time.Minute):
		t.Fatal("still running after a minute: deadlocked")
		return nil
	}
}

func TestCheckpointedFunctionMayUseGradients(t *testing.T) {
	ctx := t.Context()
	for _, tc := range []struct {
		name string
		use  func(t *testing.T, w, other *tensor.Param)
	}{
		{"Grad", func(_ *testing.T, w, _ *tensor.Param) { _ = w.Grad() }},
		{"ZeroGrad", func(_ *testing.T, _, other *tensor.Param) { other.ZeroGrad() }},
		{"Param.To", func(t *testing.T, _, other *tensor.Param) {
			if err := other.To(tensor.CPU()); err != nil {
				t.Error(err)
			}
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			w := tensor.NewParam(tensor.FromSlice([]float32{1, 2, 3}))
			other := tensor.NewParam(tensor.FromSlice([]float32{7, 8}))
			if err := other.Value().Sum().Backward(ctx); err != nil {
				t.Fatal(err)
			}
			// f does what a module's forward pass may: here in the
			// forward call and again in the recompute, under Backward.
			f := func(in ...*tensor.Tensor) *tensor.Tensor {
				tc.use(t, w, other)
				return in[0].Mul(w.Value())
			}
			x := tensor.FromSlice([]float32{4, 5, 6}).RequiresGrad()
			if err := within(t, func() error { return tensor.Checkpoint(f, x).Sum().Backward(ctx) }); err != nil {
				t.Fatal(err)
			}
			if got := data[float32](t, x.Grad()); !slices.Equal(got, []float32{1, 2, 3}) {
				t.Errorf("x gradient %v, want w = [1 2 3]", got)
			}
			if got := data[float32](t, w.Grad()); !slices.Equal(got, []float32{4, 5, 6}) {
				t.Errorf("w gradient %v, want x = [4 5 6]", got)
			}
		})
	}
}

func TestCheckpointBackwardsOnManyGoroutinesTakeTurns(t *testing.T) {
	// Backwards on several goroutines, each recomputing a checkpoint, take
	// turns rather than deadlock.
	ctx := t.Context()
	w := tensor.NewParam(tensor.Ones(tensor.Float32, 3))
	f := func(in ...*tensor.Tensor) *tensor.Tensor { return in[0].Mul(w.Value()) }
	const goroutines, rounds = 4, 25
	var wg sync.WaitGroup
	for range goroutines {
		wg.Go(func() {
			for range rounds {
				x := tensor.FromSlice([]float32{1, 2, 3})
				if err := tensor.Checkpoint(f, x).Sum().Backward(ctx); err != nil {
					t.Error(err)
					return
				}
			}
		})
	}
	wg.Wait()
	const n = goroutines * rounds
	if got := data[float32](t, w.Grad()); !slices.Equal(got, []float32{n, 2 * n, 3 * n}) {
		t.Errorf("w gradient %v, want %d x = [%d %d %d]", got, n, n, 2*n, 3*n)
	}
}

func TestCheckpointFailsOnAParameterChangedBeforeBackward(t *testing.T) {
	ctx := t.Context()
	f := func(w *tensor.Param) func(in ...*tensor.Tensor) *tensor.Tensor {
		return func(in ...*tensor.Tensor) *tensor.Tensor { return in[0].Mul(w.Value()) }
	}
	for _, tc := range []struct {
		name string
		loss func(w *tensor.Param, x *tensor.Tensor) *tensor.Tensor
	}{
		{"plain", func(w *tensor.Param, x *tensor.Tensor) *tensor.Tensor { return f(w)(x).Sum() }},
		{"checkpointed", func(w *tensor.Param, x *tensor.Tensor) *tensor.Tensor {
			return tensor.Checkpoint(f(w), x).Sum()
		}},
		{"nested", func(w *tensor.Param, x *tensor.Tensor) *tensor.Tensor {
			// The outer recompute calls the inner Checkpoint again, after
			// the change: the outer checkpoint holds what the inner one
			// read in the forward pass.
			return tensor.Checkpoint(func(in ...*tensor.Tensor) *tensor.Tensor {
				return tensor.Checkpoint(f(w), in[0])
			}, x).Sum()
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			w := tensor.NewParam(tensor.FromSlice([]float32{1, 2, 3}))
			x := tensor.FromSlice([]float32{4, 5, 6}).RequiresGrad()
			l := tc.loss(w, x)
			if _, err := l.Item[float32](ctx); err != nil {
				t.Fatal(err)
			}
			// An optimizer step: the new value written over the old one.
			cur := w.Value().Detach()
			next := cur.Mul(tensor.Scalar[float32](0.5))
			if err := tensor.RealizeInto(ctx, []*tensor.Tensor{cur}, []*tensor.Tensor{next}); err != nil {
				t.Fatal(err)
			}
			if err := w.Set(next); err != nil {
				t.Fatal(err)
			}
			if err := l.Backward(ctx); !errors.Is(err, tensor.ErrOverwritten) {
				t.Errorf("Backward after the step: %v, want ErrOverwritten", err)
			}
			if w.Grad() != nil || x.Grad() != nil {
				t.Error("a failed Backward changed a gradient")
			}
		})
	}
}

// peakMeter is a backend that records the most memory its buffers held
// at once. Buffers go back to the scheduler's pool, not to the backend,
// so what the backend has allocated and not freed is the most the pool
// and the live buffers held together: within one schedule, the peak of
// the memory in use.
type peakMeter struct {
	*cpu.Backend
	mu          sync.Mutex
	inUse, peak int // bytes
}

func (m *peakMeter) Alloc(dt backend.DType, n int) (backend.Buffer, error) {
	buf, err := m.Backend.Alloc(dt, n)
	if err == nil {
		m.mu.Lock()
		m.inUse += n * dt.Size()
		m.peak = max(m.peak, m.inUse)
		m.mu.Unlock()
	}
	return buf, err
}

func (m *peakMeter) Free(buf backend.Buffer) {
	m.mu.Lock()
	m.inUse -= buf.Len() * buf.DType().Size()
	m.mu.Unlock()
	m.Backend.Free(buf)
}

func TestCheckpointLowersPeakMemory(t *testing.T) {
	if tensor.Eager() {
		t.Skip("eager mode computes every operation into memory of its own")
	}
	// A chain of layers y = sin(x * w) over [rows, cols] values, with w a
	// vector over the columns. The backward pass of a layer reads two of
	// its activations: its input, for the gradient of w, and the product,
	// for the gradient of the sine. Without checkpointing all of them are
	// kept from the forward pass to the backward pass, since the gradient
	// rules of the layers capture them: 2*layers buffers of size bytes
	// are alive when the forward pass ends. With the chain cut into
	// checkpoints of segment layers, the forward pass keeps only the
	// input of each checkpoint, and the backward pass recomputes one
	// checkpoint at a time and frees its activations before the next:
	// layers/segment + 2*segment buffers, plus a few for the gradients
	// flowing between the layers. For 30 layers in checkpoints of 5 that
	// is about 60 buffers against 6 + 10 + 3, a third; measured, 61
	// buffers against 16. The buffers are 1 MiB, so the vectors of w's
	// gradient do not count.
	const layers, segment = 30, 5
	const rows, cols = 256, 1024
	const size = rows * cols * 4
	run := func(checkpoint bool) (peak int, grad []float32) {
		be := &peakMeter{Backend: cpu.New()}
		defer tensor.SetDevice(be)()
		x := tensor.Randn(1, rows, cols)
		if err := x.Realize(t.Context()); err != nil {
			t.Fatal(err)
		}
		w := tensor.NewParam(tensor.Rand(2, cols).Add(tensor.Scalar[float32](0.5)))
		block := func(in ...*tensor.Tensor) *tensor.Tensor {
			h := in[0]
			for range segment {
				h = h.Mul(w.Value()).Sin()
			}
			return h
		}
		h := x
		for range layers / segment {
			if checkpoint {
				h = tensor.Checkpoint(block, h)
			} else {
				h = block(h)
			}
		}
		if err := h.Sum().Backward(t.Context()); err != nil {
			t.Fatal(err)
		}
		return be.peak, data[float32](t, w.Grad())
	}
	plain, want := run(false)
	with, got := run(true)
	t.Logf("peak memory: %d MiB without checkpointing, %d MiB with (%d buffers of %d MiB against %d)",
		plain>>20, with>>20, plain/size, size>>20, with/size)
	if plain < 2*layers*size {
		t.Errorf("peak without checkpointing is %d buffers, want at least %d: the test no longer measures what it should", plain/size, 2*layers)
	}
	if with > plain*6/10 {
		t.Errorf("peak with checkpointing is %d bytes, want under 0.6 of %d", with, plain)
	}
	// The recompute runs the same elementwise kernels, fused differently.
	for i := range want {
		if d := math.Abs(float64(got[i] - want[i])); d > 1e-5*math.Abs(float64(want[i]))+1e-6 {
			t.Fatalf("gradient %d: %v with checkpointing, %v without", i, got[i], want[i])
		}
	}
}

func ExampleCheckpoint() {
	ctx := context.Background()
	w := tensor.NewParam(tensor.FromSlice([]float32{1, 2, 3}))
	x := tensor.FromSlice([]float32{4, 5, 6}).RequiresGrad()
	// Backward recomputes block to get the gradients instead of keeping
	// the product until then.
	block := func(in ...*tensor.Tensor) *tensor.Tensor { return in[0].Mul(w.Value()) }
	loss := tensor.Checkpoint(block, x).Sum()
	if err := loss.Backward(ctx); err != nil {
		fmt.Println(err)
		return
	}
	dx, _ := x.Grad().Data[float32](ctx)
	dw, _ := w.Grad().Data[float32](ctx)
	fmt.Println(dx, dw)
	// Output: [1 2 3] [4 5 6]
}
