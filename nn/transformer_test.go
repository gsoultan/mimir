package nn_test

import (
	"errors"
	"maps"
	"math"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/internal/goldentest"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/tensor"
)

type graders = map[int]goldentest.Grader

// attentionCases build each case of the attention golden file from its
// inputs and return the output and where each input's gradient lands.
var attentionCases = map[string]func(t *testing.T, in []*tensor.Tensor) (*tensor.Tensor, graders){
	"sdpa":        sdpaCase(),
	"sdpa_causal": sdpaCase(tensor.Causal()),
	"sdpa_mask": func(t *testing.T, in []*tensor.Tensor) (*tensor.Tensor, graders) {
		return tensor.ScaledDotProductAttention(in[0], in[1], in[2], tensor.AdditiveMask(in[3])), graders{0: in[0], 1: in[1], 2: in[2]}
	},
	"sdpa_bool_mask": func(t *testing.T, in []*tensor.Tensor) (*tensor.Tensor, graders) {
		return tensor.ScaledDotProductAttention(in[0], in[1], in[2], tensor.BoolMask(in[3])), graders{0: in[0], 1: in[1], 2: in[2]}
	},
	"sdpa_gqa":           sdpaCase(tensor.CausalOffset(2)),
	"sdpa_scale":         sdpaCase(tensor.AttentionScale(0.3)),
	"sdpa_causal_offset": sdpaCase(tensor.CausalOffset(4)),
	"multihead_attention": func(t *testing.T, in []*tensor.Tensor) (*tensor.Tensor, graders) {
		m := nn.NewMultiheadAttention(8, 2)
		setAll(t, in[1:], m.InProjWeight, m.InProjBias, m.OutProj.Weight, m.OutProj.Bias)
		return m.Forward(in[0]), graders{0: in[0], 1: m.InProjWeight, 2: m.InProjBias, 3: m.OutProj.Weight, 4: m.OutProj.Bias}
	},
	"multihead_attention_causal": func(t *testing.T, in []*tensor.Tensor) (*tensor.Tensor, graders) {
		m := nn.NewMultiheadAttention(8, 4)
		setAll(t, in[3:], m.InProjWeight, m.InProjBias, m.OutProj.Weight, m.OutProj.Bias)
		return m.Attend(in[0], in[1], in[2], tensor.Causal()),
			graders{0: in[0], 1: in[1], 2: in[2], 3: m.InProjWeight, 4: m.InProjBias, 5: m.OutProj.Weight, 6: m.OutProj.Bias}
	},
	"rms_norm": func(t *testing.T, in []*tensor.Tensor) (*tensor.Tensor, graders) {
		r := nn.NewRMSNorm(8, nn.Eps(1e-6))
		set(t, r.Weight, in[1])
		return r.Forward(in[0]), graders{0: in[0], 1: r.Weight}
	},
	"rms_norm_default_eps": func(t *testing.T, in []*tensor.Tensor) (*tensor.Tensor, graders) {
		r := nn.NewRMSNorm(6)
		set(t, r.Weight, in[1])
		return r.Forward(in[0]), graders{0: in[0], 1: r.Weight}
	},
	"rope_neox": func(t *testing.T, in []*tensor.Tensor) (*tensor.Tensor, graders) {
		return nn.NewRotaryEmbedding(8, 10000).Apply(in[0], 3), graders{0: in[0]}
	},
	"rope_interleaved": func(t *testing.T, in []*tensor.Tensor) (*tensor.Tensor, graders) {
		r := nn.NewRotaryEmbedding(8, 10000)
		r.Interleaved = true
		return r.Apply(in[0], 3), graders{0: in[0]}
	},
	"rope_partial_scaled": func(t *testing.T, in []*tensor.Tensor) (*tensor.Tensor, graders) {
		r := nn.NewRotaryEmbedding(8, 10000)
		r.Scaling = nn.Llama3RoPEScaling(8, 1, 4, 64)
		return r.Apply(in[0], 2), graders{0: in[0]}
	},
	"decoder_block_gpt2": func(t *testing.T, in []*tensor.Tensor) (*tensor.Tensor, graders) {
		b := nn.NewGPT2Block(8, 2)
		n1, n2, mlp := b.Norm1.(*nn.LayerNorm), b.Norm2.(*nn.LayerNorm), b.MLP.(*nn.MLP)
		ps := []*tensor.Param{n1.Weight, n1.Bias, b.Attn.QKV.Weight, b.Attn.QKV.Bias, b.Attn.Out.Weight, b.Attn.Out.Bias,
			n2.Weight, n2.Bias, mlp.Up.Weight, mlp.Up.Bias, mlp.Down.Weight, mlp.Down.Bias}
		setAll(t, in[1:], ps...)
		return b.Forward(in[0]), paramGraders(in[0], ps)
	},
	"decoder_block_llama": func(t *testing.T, in []*tensor.Tensor) (*tensor.Tensor, graders) {
		b := nn.NewLlamaBlock(16, 4, 2, 24, nn.NewRotaryEmbedding(4, 10000))
		n1, n2, mlp := b.Norm1.(*nn.RMSNorm), b.Norm2.(*nn.RMSNorm), b.MLP.(*nn.GatedMLP)
		ps := []*tensor.Param{n1.Weight, b.Attn.QKV.Weight, b.Attn.Out.Weight, n2.Weight, mlp.Gate.Weight, mlp.Up.Weight, mlp.Down.Weight}
		setAll(t, in[1:], ps...)
		return b.Forward(in[0]), paramGraders(in[0], ps)
	},
	"transformer_encoder_layer": func(t *testing.T, in []*tensor.Tensor) (*tensor.Tensor, graders) {
		l := nn.NewTransformerEncoderLayer(8, 2, 16)
		nn.Eval(l)
		ps := []*tensor.Param{l.SelfAttn.InProjWeight, l.SelfAttn.InProjBias, l.SelfAttn.OutProj.Weight, l.SelfAttn.OutProj.Bias,
			l.Linear1.Weight, l.Linear1.Bias, l.Linear2.Weight, l.Linear2.Bias, l.Norm1.Weight, l.Norm1.Bias, l.Norm2.Weight, l.Norm2.Bias}
		setAll(t, in[1:], ps...)
		return l.Forward(in[0]), paramGraders(in[0], ps)
	},
}

func sdpaCase(opts ...tensor.AttentionOption) func(t *testing.T, in []*tensor.Tensor) (*tensor.Tensor, graders) {
	return func(t *testing.T, in []*tensor.Tensor) (*tensor.Tensor, graders) {
		return tensor.ScaledDotProductAttention(in[0], in[1], in[2], opts...), graders{0: in[0], 1: in[1], 2: in[2]}
	}
}

func setAll(t *testing.T, vals []*tensor.Tensor, ps ...*tensor.Param) {
	t.Helper()
	if len(vals) != len(ps) {
		t.Fatalf("%d values for %d parameters", len(vals), len(ps))
	}
	for i, p := range ps {
		set(t, p, vals[i])
	}
}

// paramGraders returns the graders of an input x followed by ps.
func paramGraders(x *tensor.Tensor, ps []*tensor.Param) graders {
	g := graders{0: x}
	for i, p := range ps {
		g[i+1] = p
	}
	return g
}

// TestAttentionGolden checks attention, its layers and transformer blocks,
// with their gradients, against PyTorch or NumPy (see tools/golden/gen.py).
func TestAttentionGolden(t *testing.T) {
	cases, meta := goldentest.Load(t, "attention")
	t.Logf("golden oracle: %s", meta["oracle"])
	for _, name := range slices.Sorted(maps.Keys(attentionCases)) {
		if _, ok := cases[name]; !ok {
			t.Errorf("case %q has no golden data; regenerate with tools/golden/gen.py", name)
		}
	}
	tol := goldentest.Tol{Atol: 1e-5, Rtol: 1e-4}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			build, ok := attentionCases[name]
			if !ok {
				t.Fatalf("golden case %q has no Go counterpart", name)
			}
			out, grads := build(t, goldentest.Inputs(t, c))
			goldentest.CheckGradients(t, c, out, grads, tol)
		})
	}
}

func near(t *testing.T, what string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", what, len(got), len(want))
	}
	for i := range want {
		if d := math.Abs(float64(got[i] - want[i])); !(d <= tol*math.Max(1, math.Abs(float64(want[i])))) {
			t.Fatalf("%s[%d] = %v, want %v", what, i, got[i], want[i])
		}
	}
}

// TestDecodingWithACacheMatchesForward runs a sequence through blocks at
// once and in steps of several sizes with a cache, which must agree.
func TestDecodingWithACacheMatchesForward(t *testing.T) {
	ctx := t.Context()
	rope := nn.NewRotaryEmbedding(4, 10000)
	for name, b := range map[string]*nn.DecoderBlock{
		"gpt2":  nn.NewGPT2Block(16, 4, nn.Seed(1)),
		"llama": nn.NewLlamaBlock(16, 4, 2, 24, rope, nn.Seed(2)),
	} {
		t.Run(name, func(t *testing.T) {
			x := tensor.Randn(3, 2, 7, 16)
			want := data(t, b.Forward(x))
			cache := nn.NewKVCache(2, b.Attn.KVHeads, 9, 4, tensor.Float32)
			for range 2 { // the second time after a Reset
				var got []*tensor.Tensor
				pos := 0
				for _, n := range []int{3, 1, 1, 2} {
					step := x.Shrink([][2]int{{0, 2}, {pos, pos + n}, {0, 16}})
					y, err := b.Decode(ctx, step, cache)
					if err != nil {
						t.Fatal(err)
					}
					// The next step writes over the cache this one reads.
					if err := y.Realize(ctx); err != nil {
						t.Fatal(err)
					}
					got = append(got, y)
					pos += n
				}
				if cache.Len() != 7 {
					t.Errorf("cache holds %d positions, want 7", cache.Len())
				}
				near(t, "decoded", data(t, tensor.Cat(1, got...)), want, 1e-5)
				cache.Reset()
			}
		})
	}
}

func TestKVCacheErrors(t *testing.T) {
	ctx := t.Context()
	cache := nn.NewKVCache(1, 2, 4, 3, tensor.Float32)
	kv := func(n int) *tensor.Tensor { return tensor.Ones(tensor.Float32, 1, 2, n, 3) }
	if _, _, err := cache.Append(ctx, kv(5), kv(5)); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("past capacity: got %v, want ErrShape", err)
	}
	if _, _, err := cache.Append(ctx, tensor.Ones(tensor.Float32, 1, 3, 1, 3), tensor.Ones(tensor.Float32, 1, 3, 1, 3)); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("wrong heads: got %v, want ErrShape", err)
	}
	if _, _, err := cache.Append(ctx, kv(1), kv(2)); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("keys and values differ: got %v, want ErrShape", err)
	}
	if _, _, err := cache.Append(ctx, tensor.Failed(tensor.ErrDType), kv(1)); !errors.Is(err, tensor.ErrDType) {
		t.Errorf("failed keys: got %v, want ErrDType", err)
	}
	keys, values, err := cache.Append(ctx, kv(3), kv(3).Mul(tensor.Scalar[float32](2)))
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(keys.Shape(), []int{1, 2, 3, 3}) || cache.Len() != 3 || cache.Cap() != 4 {
		t.Errorf("keys %v, length %d, capacity %d", keys.Shape(), cache.Len(), cache.Cap())
	}
	if v := data(t, values); v[0] != 2 {
		t.Errorf("values %v", v)
	}
	if _, _, err := cache.Append(ctx, kv(2), kv(2)); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("overflow: got %v, want ErrShape", err)
	}
	cache.Truncate(1)
	keys, values, err = cache.Append(ctx, kv(2).Mul(tensor.Scalar[float32](3)), kv(2))
	if err != nil {
		t.Fatal(err)
	}
	if got := data(t, keys); cache.Len() != 3 || got[0] != 1 || got[3] != 3 {
		t.Errorf("after truncating to 1 and appending 2: length %d, keys %v", cache.Len(), got)
	}
	if v := data(t, values); v[0] != 2 || v[3] != 1 {
		t.Errorf("values %v", v)
	}
}

func TestRMSNormKeepsTheInputDType(t *testing.T) {
	x := tensor.Randn(1, 2, 8).Cast(tensor.BFloat16)
	r := nn.NewRMSNorm(8)
	if err := nn.Cast(r, tensor.BFloat16); err != nil {
		t.Fatal(err)
	}
	y := r.Forward(x)
	if y.DType() != tensor.BFloat16 {
		t.Fatalf("dtype %s, want bfloat16", y.DType())
	}
	near(t, "bfloat16", data(t, y.Cast(tensor.Float32)), data(t, r.Forward(x.Cast(tensor.Float32))), 1e-2)
}

func TestRotaryEmbeddingErrors(t *testing.T) {
	x := tensor.Ones(tensor.Float32, 2, 3, 6)
	for _, r := range []*nn.RotaryEmbedding{
		nn.NewRotaryEmbedding(5, 10000), nn.NewRotaryEmbedding(8, 10000), nn.NewRotaryEmbedding(0, 10000), nn.NewRotaryEmbedding(4, 0),
	} {
		if err := r.Apply(x, 0).Err(); !errors.Is(err, tensor.ErrShape) {
			t.Errorf("%d features with base %v: got %v, want ErrShape", r.Dim, r.Base, err)
		}
	}
	if err := nn.NewRotaryEmbedding(4, 10000).Apply(x, -1).Err(); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("negative offset: got %v, want ErrShape", err)
	}
	// Positions past the table built so far grow it.
	r := nn.NewRotaryEmbedding(2, 10000)
	y := r.Apply(tensor.FromSlice([]float32{1, 0}, 1, 2), 1000)
	got := data(t, y)
	near(t, "rotated", got, []float32{float32(math.Cos(1000)), float32(math.Sin(1000))}, 1e-6)
}

func TestMultiheadAttentionShapes(t *testing.T) {
	m := nn.NewMultiheadAttention(8, 2, nn.Seed(3))
	x := tensor.Randn(4, 5, 8)
	near(t, "unbatched", data(t, m.Forward(x)), data(t, m.Forward(x.Unsqueeze(0)).Squeeze(0)), 1e-6)
	if err := nn.NewMultiheadAttention(8, 3).Forward(tensor.Randn(4, 1, 5, 8)).Err(); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("3 heads of 8 features: got %v, want ErrShape", err)
	}
	if err := m.Forward(tensor.Randn(4, 2, 2, 5, 8)).Err(); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("rank 4: got %v, want ErrShape", err)
	}
	if nn.NewMultiheadAttention(8, 2, nn.NoBias()).OutProj.Bias != nil {
		t.Error("NoBias kept a bias")
	}
}

func TestTransformerBlocksNameTheirParameters(t *testing.T) {
	var names []string
	for name := range nn.NamedParams(nn.NewLlamaBlock(8, 2, 1, 16, nn.NewRotaryEmbedding(4, 10000))) {
		names = append(names, name)
	}
	want := []string{"Norm1.Weight", "Attn.QKV.Weight", "Attn.Out.Weight", "Norm2.Weight", "MLP.Gate.Weight", "MLP.Up.Weight", "MLP.Down.Weight"}
	if !slices.Equal(names, want) {
		t.Errorf("parameters %v, want %v", names, want)
	}
}

func TestEncoderLayerNormFirst(t *testing.T) {
	l := nn.NewTransformerEncoderLayer(8, 2, 16, nn.Seed(4))
	nn.Eval(l)
	l.NormFirst = true
	x := tensor.Randn(5, 2, 3, 8)
	h := x.Add(l.SelfAttn.Forward(l.Norm1.Forward(x)))
	want := h.Add(l.Linear2.Forward(l.Linear1.Forward(l.Norm2.Forward(h)).ReLU()))
	near(t, "pre-norm", data(t, l.Forward(x)), data(t, want), 1e-6)
}
