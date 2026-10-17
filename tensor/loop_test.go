package tensor_test

import (
	"errors"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/tensor"
)

// rnnStep is a recurrent step h' = tanh(x + h W), returning h' as the
// output too.
func rnnStep(w *tensor.Tensor) tensor.ScanStep {
	return func(carry, x []*tensor.Tensor) (next, y []*tensor.Tensor) {
		h := x[0].Add(carry[0].MatMul(w)).Tanh()
		return []*tensor.Tensor{h}, []*tensor.Tensor{h}
	}
}

func TestScanRunsTheLoop(t *testing.T) {
	// A running sum with a running maximum as a second carry, over xs
	// laid out as [batch, time, features] and scanned along time.
	r := rand.New(rand.NewPCG(3, 4))
	const b, n, d = 2, 5, 3
	xv := vals(r, b*n*d, -2, 2)
	x := tensor.FromSlice(xv, b, n, d)
	step := func(carry, x []*tensor.Tensor) (next, y []*tensor.Tensor) {
		s := carry[0].Add(x[0])
		m := carry[1].Maximum(x[0])
		return []*tensor.Tensor{s, m}, []*tensor.Tensor{s, s.Sub(m)}
	}
	for _, reverse := range []bool{false, true} {
		var opts []tensor.ScanOption
		if reverse {
			opts = append(opts, tensor.ScanReverse())
		}
		init := []*tensor.Tensor{tensor.Zeros(tensor.Float32, b, d), tensor.Full(tensor.Float32, -100, b, d)}
		carry, ys := tensor.Scan(step, init, []*tensor.Tensor{x}, 1, opts...)
		if len(carry) != 2 || len(ys) != 2 {
			t.Fatalf("reverse %v: %d carries and %d outputs", reverse, len(carry), len(ys))
		}
		if !slices.Equal(ys[0].Shape(), []int{b, n, d}) {
			t.Fatalf("reverse %v: output shape %v", reverse, ys[0].Shape())
		}
		want := make([]float32, b*n*d)
		wantS := make([]float32, b*d)
		for i := range b {
			for j := range d {
				s := float32(0)
				for k := range n {
					step := k
					if reverse {
						step = n - 1 - k
					}
					s += xv[(i*n+step)*d+j]
					want[(i*n+step)*d+j] = s
				}
				wantS[i*d+j] = s
			}
		}
		if got := data[float32](t, ys[0]); !close(got, want) {
			t.Errorf("reverse %v: sums %v, want %v", reverse, got, want)
		}
		if got := data[float32](t, carry[0]); !close(got, wantS) {
			t.Errorf("reverse %v: final sum %v, want %v", reverse, got, wantS)
		}
	}
}

// close reports whether a and b agree to float32 rounding.
func close(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if d := a[i] - b[i]; d > 1e-5 || d < -1e-5 {
			return false
		}
	}
	return true
}

func scanGradCases() []gradCase {
	r := rand.New(rand.NewPCG(5, 6))
	in := func(shape ...int) []float32 {
		n := 1
		for _, d := range shape {
			n *= d
		}
		return vals(r, n, -1, 1)
	}
	// x [5, 2, 3] scanned along 0, h0 [2, 3], w [3, 3]; the result joins
	// every output and the final carry.
	rnn := func(opts ...tensor.ScanOption) func(x []*tensor.Tensor) *tensor.Tensor {
		return func(x []*tensor.Tensor) *tensor.Tensor {
			carry, ys := tensor.Scan(rnnStep(x[2]), x[1:2], x[:1], 0, opts...)
			return tensor.Cat(0, ys[0].Reshape(-1), carry[0].Reshape(-1))
		}
	}
	dims := [][]int{{5, 2, 3}, {2, 3}, {3, 3}}
	rnnCase := func(name string, opts ...tensor.ScanOption) gradCase {
		return gradCase{name: name, in: [][]float32{in(5, 2, 3), in(2, 3), in(3, 3)}, dims: dims, f: rnn(opts...)}
	}
	return []gradCase{
		rnnCase("scan"),
		rnnCase("scan_reverse", tensor.ScanReverse()),
		rnnCase("scan_checkpoint", tensor.ScanCheckpoint(2)),
		rnnCase("scan_checkpoint_reverse", tensor.ScanCheckpoint(3), tensor.ScanReverse()),
		{
			// Two scanned tensors along axis 1, two carries, two outputs,
			// and a loss that reads the last carry only.
			name: "scan_two_carries",
			in:   [][]float32{in(2, 4, 3), in(2, 4), in(2, 3), in(2, 3)},
			dims: [][]int{{2, 4, 3}, {2, 4}, {2, 3}, {2, 3}},
			f: func(x []*tensor.Tensor) *tensor.Tensor {
				step := func(carry, x []*tensor.Tensor) (next, y []*tensor.Tensor) {
					a := carry[0].Mul(x[0].Sigmoid()).Add(carry[1])
					b := carry[1].Mul(x[1].Unsqueeze(-1)).Tanh()
					return []*tensor.Tensor{a, b}, []*tensor.Tensor{a.Sum(-1), b}
				}
				carry, _ := tensor.Scan(step, x[2:], x[:2], 1)
				return carry[0].Add(carry[1])
			},
		},
	}
}

func TestScanGradientsMatchFiniteDifferences(t *testing.T) {
	for _, c := range scanGradCases() {
		t.Run(c.name, func(t *testing.T) { checkFiniteDifferences(t, c) })
	}
}

func TestScanCheckpointMatchesScan(t *testing.T) {
	ctx := t.Context()
	r := rand.New(rand.NewPCG(7, 8))
	xv, hv, wv := vals(r, 9*2*3, -1, 1), vals(r, 2*3, -1, 1), vals(r, 3*3, -1, 1)
	run := func(opts ...tensor.ScanOption) (out []float32, grads [3][]float32) {
		x := tensor.FromSlice(xv, 9, 2, 3).RequiresGrad()
		h := tensor.FromSlice(hv, 2, 3).RequiresGrad()
		w := tensor.FromSlice(wv, 3, 3).RequiresGrad()
		carry, ys := tensor.Scan(rnnStep(w), []*tensor.Tensor{h}, []*tensor.Tensor{x}, 0, opts...)
		loss := ys[0].Mul(ys[0]).Sum().Add(carry[0].Sum())
		if err := loss.Backward(ctx); err != nil {
			t.Fatal(err)
		}
		for i, v := range []*tensor.Tensor{x, h, w} {
			grads[i] = data[float32](t, v.Grad())
		}
		return data[float32](t, ys[0]), grads
	}
	want, wantGrads := run()
	for _, k := range []int{1, 2, 4, 9, 20} {
		got, grads := run(tensor.ScanCheckpoint(k))
		if !close(got, want) {
			t.Errorf("checkpoint every %d: outputs differ", k)
		}
		for i := range grads {
			if !close(grads[i], wantGrads[i]) {
				t.Errorf("checkpoint every %d: gradient %d differs", k, i)
			}
		}
	}
}

func TestScanErrors(t *testing.T) {
	x := tensor.Zeros(tensor.Float32, 4, 2, 3)
	h := tensor.Zeros(tensor.Float32, 2, 3)
	w := tensor.Zeros(tensor.Float32, 3, 3)
	step := rnnStep(w)
	tests := map[string]struct {
		carry, ys []*tensor.Tensor
		want      error
	}{}
	add := func(name string, want error, step tensor.ScanStep, init, xs []*tensor.Tensor, axis int, opts ...tensor.ScanOption) {
		c, y := tensor.Scan(step, init, xs, axis, opts...)
		tests[name] = struct {
			carry, ys []*tensor.Tensor
			want      error
		}{c, y, want}
	}
	one := []*tensor.Tensor{h}
	add("lengths", tensor.ErrShape, step, one, []*tensor.Tensor{x, tensor.Zeros(tensor.Float32, 3, 2, 3)}, 0)
	add("axis", tensor.ErrShape, step, one, []*tensor.Tensor{x}, 3)
	add("no xs", tensor.ErrShape, func(carry, _ []*tensor.Tensor) ([]*tensor.Tensor, []*tensor.Tensor) {
		return carry, carry
	}, one, nil, 0)
	add("failed init", tensor.ErrDType, step, []*tensor.Tensor{tensor.Failed(tensor.ErrDType)}, []*tensor.Tensor{x}, 0)
	add("carry shape", tensor.ErrShape, step, []*tensor.Tensor{tensor.Zeros(tensor.Float32, 1, 3)}, []*tensor.Tensor{x}, 0)
	add("carry dtype", tensor.ErrShape, func(carry, x []*tensor.Tensor) ([]*tensor.Tensor, []*tensor.Tensor) {
		return []*tensor.Tensor{carry[0].Cast(tensor.BFloat16)}, x
	}, one, []*tensor.Tensor{x}, 0)
	add("checkpoint", tensor.ErrShape, step, one, []*tensor.Tensor{x}, 0, tensor.ScanCheckpoint(-1))
	add("checkpoint dtypes", tensor.ErrDType, func(carry, x []*tensor.Tensor) ([]*tensor.Tensor, []*tensor.Tensor) {
		return carry, []*tensor.Tensor{x[0].Cast(tensor.BFloat16)}
	}, one, []*tensor.Tensor{x}, 0, tensor.ScanCheckpoint(2))
	add("output stacked past rank", tensor.ErrShape, func(carry, x []*tensor.Tensor) ([]*tensor.Tensor, []*tensor.Tensor) {
		return carry, []*tensor.Tensor{x[0].Sum()}
	}, one, []*tensor.Tensor{tensor.Zeros(tensor.Float32, 2, 3, 4)}, 2)
	for name, tt := range tests {
		if len(tt.carry) != 1 || len(tt.ys) != 1 {
			t.Errorf("%s: %d carries and %d outputs, want 1 and 1", name, len(tt.carry), len(tt.ys))
			continue
		}
		for _, r := range append(tt.carry, tt.ys...) {
			if err := r.Err(); !errors.Is(err, tt.want) {
				t.Errorf("%s: %v, want %v", name, err, tt.want)
			}
		}
	}
	if c, y := tensor.Scan(nil, []*tensor.Tensor{h}, []*tensor.Tensor{x}, 0); len(c) != 1 || !errors.Is(c[0].Err(), tensor.ErrInternal) || len(y) != 0 {
		t.Errorf("nil step: %v, %v", c, y)
	}
}

func TestScanOfNoSteps(t *testing.T) {
	h := tensor.Ones(tensor.Float32, 2, 3)
	carry, ys := tensor.Scan(rnnStep(tensor.Zeros(tensor.Float32, 3, 3)), []*tensor.Tensor{h}, []*tensor.Tensor{tensor.Zeros(tensor.Float32, 2, 0, 3)}, 1)
	if carry[0] != h {
		t.Error("carry of no steps is not init")
	}
	if !slices.Equal(ys[0].Shape(), []int{2, 0, 3}) || ys[0].Err() != nil {
		t.Errorf("outputs of no steps: %v", ys[0])
	}
}
