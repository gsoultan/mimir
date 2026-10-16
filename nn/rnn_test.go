package nn_test

import (
	"errors"
	"fmt"
	"maps"
	"math"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/internal/goldentest"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/tensor"
)

// rnnConfig mirrors RNN_CASES in tools/golden/gen.py.
type rnnConfig struct {
	kind                                   string
	in, hidden, layers                     int
	bidir, batchFirst, relu, bias, batched bool
}

var rnnCases = map[string]rnnConfig{
	"rnn_tanh":                     {"rnn", 3, 4, 1, false, false, false, true, true},
	"rnn_relu_2layer_bidir":        {"rnn", 3, 4, 2, true, false, true, true, true},
	"lstm":                         {"lstm", 3, 4, 1, false, false, false, true, true},
	"lstm_2layer_bidir":            {"lstm", 3, 4, 2, true, false, false, true, true},
	"lstm_batch_first_unbiased":    {"lstm", 3, 4, 1, false, true, false, false, true},
	"lstm_unbatched":               {"lstm", 3, 4, 1, false, false, false, true, false},
	"gru":                          {"gru", 3, 4, 1, false, false, false, true, true},
	"gru_2layer_bidir_batch_first": {"gru", 3, 4, 2, true, true, false, true, true},
}

func (c rnnConfig) options() []nn.Option {
	opts := []nn.Option{nn.NumLayers(c.layers)}
	if c.bidir {
		opts = append(opts, nn.Bidirectional())
	}
	if c.batchFirst {
		opts = append(opts, nn.BatchFirst())
	}
	if c.relu {
		opts = append(opts, nn.RNNReLU())
	}
	if !c.bias {
		opts = append(opts, nn.NoBias())
	}
	return opts
}

// paramNames returns the parameter names of c in torch's order.
func (c rnnConfig) paramNames() []string {
	var names []string
	dirs := []string{""}
	if c.bidir {
		dirs = append(dirs, "_reverse")
	}
	for l := range c.layers {
		for _, d := range dirs {
			sfx := "_l" + string(rune('0'+l)) + d
			names = append(names, "weight_ih"+sfx, "weight_hh"+sfx)
			if c.bias {
				names = append(names, "bias_ih"+sfx, "bias_hh"+sfx)
			}
		}
	}
	return names
}

// recurrentLayer is what the tests use of RNN, LSTM and GRU.
type recurrentLayer interface {
	nn.Module
	run(x *tensor.Tensor, states []*tensor.Tensor) (out *tensor.Tensor, final []*tensor.Tensor)
	params() map[string]*tensor.Param
}

type rnnLayer struct{ *nn.RNN }

func (l rnnLayer) run(x *tensor.Tensor, s []*tensor.Tensor) (*tensor.Tensor, []*tensor.Tensor) {
	out, h := l.Run(x, s[0])
	return out, []*tensor.Tensor{h}
}
func (l rnnLayer) params() map[string]*tensor.Param { return l.Params }

type lstmLayer struct{ *nn.LSTM }

func (l lstmLayer) run(x *tensor.Tensor, s []*tensor.Tensor) (*tensor.Tensor, []*tensor.Tensor) {
	out, h, c := l.Run(x, s[0], s[1])
	return out, []*tensor.Tensor{h, c}
}
func (l lstmLayer) params() map[string]*tensor.Param { return l.Params }

type gruLayer struct{ *nn.GRU }

func (l gruLayer) run(x *tensor.Tensor, s []*tensor.Tensor) (*tensor.Tensor, []*tensor.Tensor) {
	out, h := l.Run(x, s[0])
	return out, []*tensor.Tensor{h}
}
func (l gruLayer) params() map[string]*tensor.Param { return l.Params }

func (c rnnConfig) build(extra ...nn.Option) recurrentLayer {
	opts := append(c.options(), extra...)
	switch c.kind {
	case "lstm":
		return lstmLayer{nn.NewLSTM(c.in, c.hidden, opts...)}
	case "gru":
		return gruLayer{nn.NewGRU(c.in, c.hidden, opts...)}
	}
	return rnnLayer{nn.NewRNN(c.in, c.hidden, opts...)}
}

// input returns a random sequence of steps steps for c, in its layout,
// in batches of 2 if c is batched.
func (c rnnConfig) input(seed uint64, steps int) *tensor.Tensor {
	switch {
	case !c.batched:
		return tensor.Randn(seed, steps, c.in)
	case c.batchFirst:
		return tensor.Randn(seed, 2, steps, c.in)
	}
	return tensor.Randn(seed, steps, 2, c.in)
}

// loss runs l on x from zero states and returns a weighted sum of its
// outputs and final states, which every element of them changes.
func (c rnnConfig) loss(l recurrentLayer, x *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor) {
	out, final := l.run(x, make([]*tensor.Tensor, c.states()))
	parts := []*tensor.Tensor{out.Reshape(-1)}
	for _, f := range final {
		parts = append(parts, f.Reshape(-1))
	}
	all := tensor.Cat(0, parts...)
	return all, all.Mul(tensor.Randn(99, all.Shape()...)).Sum()
}

func (c rnnConfig) states() int {
	if c.kind == "lstm" {
		return 2
	}
	return 1
}

func TestRecurrentGolden(t *testing.T) {
	cases, meta := goldentest.Load(t, "rnn")
	t.Logf("golden oracle: %s", meta["oracle"])
	for _, name := range slices.Sorted(maps.Keys(rnnCases)) {
		if _, ok := cases[name]; !ok {
			t.Errorf("case %q has no golden data; regenerate with tools/golden/gen.py", name)
		}
	}
	tol := goldentest.Tol{Atol: 1e-5, Rtol: 1e-4}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			cfg, ok := rnnCases[name]
			if !ok {
				t.Fatalf("golden case %q has no Go counterpart", name)
			}
			in := goldentest.Inputs(t, c)
			layer := cfg.build()
			ns := cfg.states()
			grads := map[int]goldentest.Grader{}
			for i := range 1 + ns {
				grads[i] = in[i]
			}
			names := cfg.paramNames()
			if len(names) != len(layer.params()) || len(in) != 1+ns+len(names) {
				t.Fatalf("%d parameters and %d inputs for %d names", len(layer.params()), len(in), len(names))
			}
			for i, n := range names {
				p := layer.params()[n]
				if p == nil {
					t.Fatalf("no parameter %s", n)
				}
				set(t, p, in[1+ns+i])
				grads[1+ns+i] = p
			}
			out, final := layer.run(in[0], in[1:1+ns])
			joined := []*tensor.Tensor{out.Reshape(-1)}
			extra := map[string]*tensor.Tensor{"output": out, "h_n": final[0]}
			for i, f := range final {
				joined = append(joined, f.Reshape(-1))
				if i == 1 {
					extra["c_n"] = f
				}
			}
			goldentest.CheckGradients(t, c, tensor.Cat(0, joined...), grads, tol)
			for part, want := range c.Extra {
				t.Run(part, func(t *testing.T) { goldentest.CompareTensor(t, want, extra[part], tol) })
			}
		})
	}
}

func TestRecurrentParameterNames(t *testing.T) {
	l := nn.NewLSTM(3, 4, nn.NumLayers(2), nn.Bidirectional())
	var got []string
	for name := range nn.NamedParams(nn.NewSequential(l)) {
		got = append(got, name)
	}
	want := rnnConfig{kind: "lstm", layers: 2, bidir: true, bias: true}.paramNames()
	for i, n := range want {
		want[i] = "Layers.0." + n
	}
	slices.Sort(want)
	slices.Sort(got)
	if !slices.Equal(got, want) {
		t.Errorf("parameters %v, want %v", got, want)
	}
	if s := l.Params["weight_ih_l1_reverse"].Value().Shape(); !slices.Equal(s, []int{16, 8}) {
		t.Errorf("weight_ih_l1_reverse shape %v, want [16 8]", s)
	}
}

func TestRecurrentShapesAndErrors(t *testing.T) {
	x := tensor.Randn(1, 5, 2, 3)
	out, h, c := nn.NewLSTM(3, 4, nn.NumLayers(2), nn.Bidirectional()).Run(x, nil, nil)
	for _, tt := range []struct {
		got  *tensor.Tensor
		want []int
	}{{out, []int{5, 2, 8}}, {h, []int{4, 2, 4}}, {c, []int{4, 2, 4}}} {
		if err := tt.got.Err(); err != nil || !slices.Equal(tt.got.Shape(), tt.want) {
			t.Errorf("shape %v (%v), want %v", tt.got.Shape(), err, tt.want)
		}
	}
	if s := nn.NewGRU(3, 4, nn.BatchFirst()).Forward(x).Shape(); !slices.Equal(s, []int{5, 2, 4}) {
		t.Errorf("batch-first GRU output %v, want [5 2 4]", s)
	}
	for name, got := range map[string]*tensor.Tensor{
		"features":    nn.NewRNN(4, 4).Forward(x),
		"rank":        nn.NewRNN(3, 4).Forward(tensor.Randn(1, 3)),
		"state shape": func() *tensor.Tensor { _, h := nn.NewGRU(3, 4).Run(x, tensor.Zeros(tensor.Float32, 2, 2, 4)); return h }(),
		"no layers":   nn.NewLSTM(3, 4, nn.NumLayers(0)).Forward(x),
	} {
		if err := got.Err(); !errors.Is(err, tensor.ErrShape) {
			t.Errorf("%s: %v, want ErrShape", name, err)
		}
	}
}

// near fails t at the first element of got further than tol from want,
// relative to the larger of 1 and want.
func near(t *testing.T, what string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", what, len(got), len(want))
	}
	for i := range got {
		if d := math.Abs(float64(got[i] - want[i])); d > tol*max(1, math.Abs(float64(want[i]))) {
			t.Fatalf("%s[%d] = %v, want %v", what, i, got[i], want[i])
		}
	}
}

func TestRecurrentStepCheckpointMatchesPlainRun(t *testing.T) {
	for _, name := range slices.Sorted(maps.Keys(rnnCases)) {
		c := rnnCases[name]
		t.Run(name, func(t *testing.T) {
			x := c.input(1, 7)
			nn.ManualSeed(5)
			plain := c.build()
			nn.ManualSeed(5)
			ck := c.build(nn.StepCheckpoint(3))
			want, wantLoss := c.loss(plain, x)
			got, gotLoss := c.loss(ck, x)
			for _, l := range []*tensor.Tensor{wantLoss, gotLoss} {
				if err := l.Backward(t.Context()); err != nil {
					t.Fatal(err)
				}
			}
			near(t, "outputs", data(t, got), data(t, want), 1e-6)
			for _, p := range slices.Sorted(maps.Keys(plain.params())) {
				near(t, p+" gradient", data(t, ck.params()[p].Grad()), data(t, plain.params()[p].Grad()), 1e-5)
			}
		})
	}
}

// TestRecurrentHiddenWeightGradient checks the gradient of each W_hh,
// computed after the loop from every step's states, against finite
// differences, for sequences of one step and of several.
func TestRecurrentHiddenWeightGradient(t *testing.T) {
	for _, kind := range []string{"rnn", "lstm", "gru"} {
		for _, steps := range []int{1, 4} {
			c := rnnConfig{kind, 3, 4, 1, true, true, false, true, true}
			t.Run(fmt.Sprintf("%s/%d", kind, steps), func(t *testing.T) {
				ctx := t.Context()
				nn.ManualSeed(7)
				l := c.build()
				x := c.input(2, steps)
				_, loss := c.loss(l, x)
				if err := loss.Backward(ctx); err != nil {
					t.Fatal(err)
				}
				for _, name := range []string{"weight_hh_l0", "weight_hh_l0_reverse"} {
					p := l.params()[name]
					w, grad := data(t, p.Value()), data(t, p.Grad())
					const eps = 1e-2
					for _, i := range []int{0, 5, len(w) - 1} {
						at := func(v float32) float64 {
							moved := slices.Clone(w)
							moved[i] = v
							set(t, p, tensor.FromSlice(moved, p.Value().Shape()...))
							_, loss := c.loss(l, x)
							f, err := loss.Item[float32](ctx)
							if err != nil {
								t.Fatal(err)
							}
							return float64(f)
						}
						want := (at(w[i]+eps) - at(w[i]-eps)) / (2 * eps)
						set(t, p, tensor.FromSlice(w, p.Value().Shape()...))
						if d := math.Abs(float64(grad[i]) - want); d > 2e-3*max(1, math.Abs(want)) {
							t.Errorf("%s[%d] gradient %v, finite differences %v", name, i, grad[i], want)
						}
					}
				}
			})
		}
	}
}

func TestRecurrentLayerDropout(t *testing.T) {
	x := tensor.Randn(1, 6, 2, 3)
	run := func(layers int, train bool, opts ...nn.Option) []float32 {
		nn.ManualSeed(3)
		m := nn.NewLSTM(3, 8, append([]nn.Option{nn.NumLayers(layers)}, opts...)...)
		if !train {
			nn.Eval(m)
		}
		return data(t, m.Forward(x))
	}
	none := run(2, true)
	near(t, "eval", run(2, false, nn.LayerDropout(0.5)), none, 0)
	if dropped := run(2, true, nn.LayerDropout(0.5)); slices.Equal(dropped, none) {
		t.Error("dropout between layers changed nothing in training")
	}
	// A single layer has no layer after it whose input to drop.
	near(t, "one layer", run(1, true, nn.LayerDropout(0.5)), run(1, true), 0)
}
