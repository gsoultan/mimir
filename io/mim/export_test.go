package mim_test

import (
	"bytes"
	"context"
	"errors"
	"math/rand/v2"
	"strings"
	"testing"

	mimfile "github.com/gsoultan/mim"
	"github.com/gsoultan/mim/interp"
	"github.com/gsoultan/mim/ref"

	"github.com/gsoultan/mimir/internal/quanttest"
	"github.com/gsoultan/mimir/io/mim"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/tensor"
)

// checkExport runs the exported file data on args, with mimir and with
// the format's reference interpreter, and compares the outputs of both
// with wants: within tol of the largest of each, and within refTol for
// the reference, whose float64 arithmetic differs from mimir's float32,
// unless refTol is negative.
func checkExport(t *testing.T, data []byte, args, wants []*tensor.Tensor, tol, refTol float64) {
	t.Helper()
	ctx := context.Background()
	model, err := mim.Parse(data)
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	f, err := mimfile.Parse(data)
	if err != nil {
		t.Fatal(err)
	}
	p, err := interp.Load(f)
	if err != nil {
		t.Fatal(err)
	}
	in, refIn := map[string]*tensor.Tensor{}, map[string]*ref.Tensor{}
	for i, info := range model.Inputs() {
		in[info.Name] = args[i]
		r, err := ref.New(info.Type, args[i].Shape())
		if err != nil {
			t.Fatal(err)
		}
		b, err := args[i].Bytes(ctx)
		if err != nil {
			t.Fatal(err)
		}
		copy(r.Data, b)
		refIn[info.Name] = r
	}
	out, err := model.Run(ctx, in)
	if err != nil {
		t.Fatal(err)
	}
	var refOut map[string]*ref.Tensor
	if refTol >= 0 {
		if refOut, err = p.NewSession().Run(refIn); err != nil {
			t.Fatalf("reference: %v", err)
		}
	}
	for k, want := range wants {
		name := model.Outputs()[k].Name
		w, err := want.Cast(tensor.Float32).Data[float32](ctx)
		if err != nil {
			t.Fatal(err)
		}
		got, err := out[name].Cast(tensor.Float32).Data[float32](ctx)
		if err != nil {
			t.Fatal(err)
		}
		near(t, name, got, w, tol)
		if refOut == nil {
			continue
		}
		r := float64s(t, refOut[name])
		rf := make([]float32, len(r))
		for i, x := range r {
			rf[i] = float32(x)
		}
		near(t, name+" in the reference", rf, w, refTol)
	}
}

// roundTrip exports m for x and checks the file's output.
func roundTrip(t *testing.T, m nn.Module, x *tensor.Tensor, tol, refTol float64) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := mim.Export(context.Background(), &buf, m, x, nil); err != nil {
		t.Fatal(err)
	}
	checkExport(t, buf.Bytes(), []*tensor.Tensor{x}, []*tensor.Tensor{m.Forward(x)}, tol, refTol)
	return buf.Bytes()
}

func randn(seed uint64, shape ...int) *tensor.Tensor {
	return tensor.Randn(seed, shape...)
}

func TestExportRoundTrips(t *testing.T) {
	ids := tensor.FromSlice([]int32{3, 0, 9, 4, 4, 1}, 2, 3)
	rope := nn.NewRotaryEmbedding(4, 10000)
	for _, tc := range []struct {
		name string
		m    nn.Module
		x    *tensor.Tensor
		tol  float64
	}{
		{"mlp", nn.NewSequential(nn.NewLinear(4, 8), nn.ReLU(), nn.NewLinear(8, 3), nn.Softmax(-1)), randn(1, 2, 4), 1e-6},
		{"activations", nn.NewSequential(nn.NewLinear(5, 6), nn.GELU(), nn.NewLinear(6, 6), nn.Tanh(), nn.SiLU(), nn.Sigmoid(), nn.LogSoftmax(-1)), randn(2, 3, 5), 1e-5},
		{"cnn", nn.NewSequential(nn.NewConv2d(2, 3, 3, nn.Padding(1)), nn.ReLU(), nn.MaxPool2d(2), nn.NewConv2d(3, 4, 3, nn.Stride(2)), nn.AvgPool2d(1), nn.Flatten(), nn.NewLinear(4, 2)), randn(3, 2, 2, 8, 8), 1e-5},
		{"dilated conv", nn.NewConv2d(2, 2, 3, nn.Dilation(2), nn.Groups(2)), randn(4, 1, 2, 9, 9), 1e-5},
		{"embedding", nn.NewSequential(nn.NewEmbedding(10, 4), nn.NewLayerNorm(4)), ids, 1e-5},
		{"batch norm", nn.NewSequential(nn.NewBatchNorm2d(3), nn.Flatten()), randn(5, 2, 3, 2, 2), 1e-5},
		{"rms norm", nn.NewRMSNorm(8), randn(6, 2, 8), 1e-5},
		{"gpt-2 block", nn.NewGPT2Block(8, 2), randn(7, 1, 5, 8), 1e-5},
		{"llama block", nn.NewLlamaBlock(8, 2, 1, 16, rope), randn(8, 2, 3, 8), 1e-5},
		{"encoder layer", nn.NewTransformerEncoderLayer(8, 2, 16), randn(9, 2, 4, 8), 1e-5},
	} {
		t.Run(tc.name, func(t *testing.T) {
			nn.Eval(tc.m)
			roundTrip(t, tc.m, tc.x, tc.tol, 1e-5)
		})
	}
}

func TestExportQuantizedLinear(t *testing.T) {
	for _, dt := range []tensor.DType{tensor.Q8_0, tensor.Q4_K, tensor.Q6_K} {
		t.Run(dt.String(), func(t *testing.T) {
			raw := quanttest.Random(rand.New(rand.NewPCG(10, uint64(dt))), dt, 4*256)
			// FromBytes keeps the blocks on the CPU; on the input's device,
			// the module and the file run the same kernel.
			w := tensor.FromBytes(dt, raw, 4, 256).To(tensor.DefaultDevice())
			l := &nn.Linear{Weight: tensor.NewBuffer(w)}
			x := randn(11, 3, 256)
			// The file holds the blocks, which a run reads as they are, so
			// the product is mimir's own; the reference rounds the
			// activations as it likes, within the composite's tolerance.
			data := roundTrip(t, l, x, 1e-6, 2e-2)
			f, err := mimfile.Parse(data)
			if err != nil {
				t.Fatal(err)
			}
			if w, ok := f.Tensor("Weight"); !ok || !strings.EqualFold(w.DType.String(), dt.String()) {
				t.Errorf("weight %+v, want one of %s", w, dt)
			}
		})
	}
}

func TestExportNamesAndMetadata(t *testing.T) {
	ctx := context.Background()
	m := nn.NewSequential(nn.NewLinear(4, 3))
	var buf bytes.Buffer
	opts := &mim.ExportOptions{Inputs: []string{"features"}, Outputs: []string{"logits"}, Metadata: map[string]any{"general.name": "tiny"}}
	if err := mim.Export(ctx, &buf, m, randn(12, 1, 4), opts); err != nil {
		t.Fatal(err)
	}
	model, err := mim.Parse(buf.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	if in, out := model.Inputs(), model.Outputs(); len(in) != 1 || in[0].String() != "features: f32[1 4]" || len(out) != 1 || out[0].String() != "logits: f32[1 3]" {
		t.Errorf("inputs %v, outputs %v", in, out)
	}
	md, err := model.Metadata()
	if err != nil || md["general.name"] != "tiny" {
		t.Errorf("metadata %v, %v", md, err)
	}
}

// checkFunc exports f for args and checks the file's outputs.
func checkFunc(t *testing.T, f func(x ...*tensor.Tensor) []*tensor.Tensor, args []*tensor.Tensor, tol, refTol float64) {
	t.Helper()
	var buf bytes.Buffer
	if err := mim.ExportFunc(context.Background(), &buf, f, args, nil); err != nil {
		t.Fatal(err)
	}
	checkExport(t, buf.Bytes(), args, f(args...), tol, refTol)
}

func TestExportFunc(t *testing.T) {
	a := tensor.FromSlice([]float32{1, 5, 2, -3, 0, 7}, 2, 3)
	b := tensor.FromSlice([]float32{0.5, 0.5, 0.5, 1, 1, 1}, 2, 3)
	ab := []*tensor.Tensor{a, b}
	idx := tensor.FromSlice([]int32{2, 0, 1, 1, 1, 0}, 2, 3)
	q, k, v := randn(20, 1, 2, 3, 4), randn(21, 1, 1, 5, 4), randn(22, 1, 1, 5, 4)
	mask := randn(23, 1, 2, 3, 5)
	for _, tc := range []struct {
		name string
		f    func(x ...*tensor.Tensor) []*tensor.Tensor
		args []*tensor.Tensor
		// noRef marks a function of a composite that the reference
		// interpreter does not run.
		noRef bool
	}{
		{"several results", func(x ...*tensor.Tensor) []*tensor.Tensor {
			s := x[0].Add(x[1])
			return []*tensor.Tensor{s, x[0], s, s.Flip(1).CumSum(1)}
		}, ab, false},
		{"comparisons", func(x ...*tensor.Tensor) []*tensor.Tensor {
			return []*tensor.Tensor{tensor.Where(x[0].Less(x[1]), x[0], x[1].Neg()), x[0].Equal(x[1]).Cast(tensor.Int32)}
		}, ab, false},
		{"sorting", func(x ...*tensor.Tensor) []*tensor.Tensor {
			vals, ids := x[0].TopK(2, 1, true)
			lows, _ := x[0].TopK(2, 1, false)
			return []*tensor.Tensor{x[0].Argsort(1, true), x[0].Argsort(0, false), vals, ids, lows}
		}, ab, true},
		{"gather", func(x ...*tensor.Tensor) []*tensor.Tensor {
			return []*tensor.Tensor{x[0].Gather(1, x[1]), x[0].Gather(0, x[1].Shrink([][2]int{{0, 1}, {0, 3}}).Fmod(tensor.Full(tensor.Int32, 2, 1, 3)))}
		}, []*tensor.Tensor{a, idx}, false},
		{"erf", func(x ...*tensor.Tensor) []*tensor.Tensor { return []*tensor.Tensor{x[0].Erf()} }, ab[:1], false},
		{"attention", func(x ...*tensor.Tensor) []*tensor.Tensor {
			return []*tensor.Tensor{
				tensor.ScaledDotProductAttention(x[0], x[1], x[2]),
				tensor.ScaledDotProductAttention(x[0], x[1], x[2], tensor.CausalOffset(2)),
				tensor.ScaledDotProductAttention(x[0], x[1], x[2], tensor.Causal(), tensor.AdditiveMask(x[3])),
				tensor.ScaledDotProductAttention(x[0].Reshape(2, 3, 4), x[1].Reshape(1, 5, 4).Expand(2, 5, 4), x[2].Reshape(1, 5, 4).Expand(2, 5, 4)),
			}
		}, []*tensor.Tensor{q, k, v, mask}, false},
		{"constant folding", func(x ...*tensor.Tensor) []*tensor.Tensor {
			spd := tensor.FromSlice([]float32{4, 2, 2, 3}, 2, 2)
			return []*tensor.Tensor{x[0].MatMul(spd.Cholesky())}
		}, []*tensor.Tensor{randn(24, 3, 2)}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			refTol := 1e-5
			if tc.noRef {
				refTol = -1
			}
			checkFunc(t, tc.f, tc.args, 2e-6, refTol)
		})
	}
}

func TestExportRejectsWhatTheFormatCannotExpress(t *testing.T) {
	f := func(x ...*tensor.Tensor) []*tensor.Tensor {
		return []*tensor.Tensor{x[0].MatMul(x[0].Transpose(0, 1)).Cholesky()}
	}
	err := mim.ExportFunc(context.Background(), &bytes.Buffer{}, f, []*tensor.Tensor{randn(25, 2, 2)}, nil)
	var u *mim.UnsupportedError
	if !errors.As(err, &u) || u.Op != "cholesky" {
		t.Fatalf("export of a Cholesky factorization: %v, want an UnsupportedError naming it", err)
	}
}

func TestExportQuantizedEmbedding(t *testing.T) {
	raw := quanttest.Random(rand.New(rand.NewPCG(26, 27)), tensor.Q8_0, 10*32)
	e := &nn.Embedding{Weight: tensor.NewBuffer(tensor.FromBytes(tensor.Q8_0, raw, 10, 32))}
	roundTrip(t, e, tensor.FromSlice([]int32{3, 9, 0}, 3), 0, 0)
}

func TestExportRejectsDropoutInTraining(t *testing.T) {
	m := nn.NewSequential(nn.NewLinear(4, 4), nn.NewDropout(0.5))
	err := mim.Export(context.Background(), &bytes.Buffer{}, m, randn(13, 2, 4), nil)
	if !errors.Is(err, mim.ErrUnsupported) {
		t.Fatalf("export in training mode: %v, want ErrUnsupported", err)
	}
}
