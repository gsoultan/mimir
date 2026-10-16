package nn_test

import (
	"bytes"
	"encoding/binary"
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/io/safetensors"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/nn/loss"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
)

// floats returns x's elements, through float32 for a half tensor.
func floats(t *testing.T, x *tensor.Tensor) []float32 {
	t.Helper()
	return data(t, x.Cast(tensor.Float32))
}

func TestDTypeNames(t *testing.T) {
	for dt, name := range map[tensor.DType]string{tensor.Float32: "F32", tensor.Int32: "I32", tensor.BFloat16: "BF16", tensor.Float16: "F16"} {
		if got := nn.DTypeName(dt); got != name {
			t.Errorf("DTypeName(%s) = %q, want %q", dt, got, name)
		}
		if got, ok := nn.DTypeOf(name); !ok || got != dt {
			t.Errorf("DTypeOf(%q) = %s, %v; want %s", name, got, ok, dt)
		}
	}
	if _, ok := nn.DTypeOf("F64"); ok {
		t.Error("DTypeOf(F64) found a dtype")
	}
}

func TestCastConvertsTrainableParameters(t *testing.T) {
	m := nn.NewSequential(nn.NewConv2d(1, 2, 3), nn.NewBatchNorm2d(2), nn.Flatten(), nn.NewLinear(8, 3))
	x := tensor.Randn(5, 3, 1, 4, 4)
	if err := m.Forward(x).Sum().Backward(t.Context()); err != nil {
		t.Fatal(err)
	}
	before := map[string][]float32{}
	for name, p := range nn.NamedParams(m) {
		before[name] = data(t, p.Value())
	}
	if err := nn.Cast(m, tensor.BFloat16); err != nil {
		t.Fatal(err)
	}
	for name, p := range nn.NamedParams(m) {
		want := tensor.Float32 // a buffer keeps its dtype
		if p.Trainable() {
			want = tensor.BFloat16
		}
		if got := p.Value().DType(); got != want {
			t.Errorf("%s is %s after the cast, want %s", name, got, want)
		}
		if got, want := floats(t, p.Value()), floats(t, tensor.FromSlice(before[name]).Cast(want)); !slices.Equal(got, want) {
			t.Errorf("%s = %v after the cast, want %v", name, got, want)
		}
		if p.Trainable() && p.Grad() == nil {
			t.Errorf("%s lost its gradient", name)
		}
	}
	// The cast model trains: its float32 activations and gradients reach
	// the half parameters.
	nn.Train(m)
	if err := m.Forward(x).Sum().Backward(t.Context()); err != nil {
		t.Fatal(err)
	}
	if err := nn.Cast(m, tensor.Int32); !errors.Is(err, tensor.ErrDType) {
		t.Errorf("cast to int32: %v, want ErrDType", err)
	}
	if err := nn.Cast(m, tensor.Float32); err != nil {
		t.Fatal(err)
	}
	for name, p := range nn.NamedParams(m) {
		if got := p.Value().DType(); got != tensor.Float32 {
			t.Errorf("%s is %s after a cast back, want float32", name, got)
		}
	}
}

func TestSaveWritesHalfTensors(t *testing.T) {
	ctx := t.Context()
	for _, dt := range []tensor.DType{tensor.BFloat16, tensor.Float16} {
		nn.ManualSeed(3)
		a := nn.NewLinear(4, 2)
		if err := nn.Cast(a, dt); err != nil {
			t.Fatal(err)
		}
		var buf bytes.Buffer
		if err := nn.Save(ctx, &buf, a); err != nil {
			t.Fatal(err)
		}
		f, err := safetensors.Parse(buf.Bytes())
		if err != nil {
			t.Fatal(err)
		}
		for name, p := range nn.NamedParams(a) {
			st, ok := f.Tensor(name)
			n, _ := p.Value().Shape(), 0
			if !ok || st.DType != nn.DTypeName(dt) || len(st.Data) != 2*len(floats(t, p.Value())) {
				t.Errorf("%s saved as %s%v of %d bytes, want %s%v", name, st.DType, st.Shape, len(st.Data), nn.DTypeName(dt), n)
			}
			// The file holds the elements as they are, so a decoded tensor
			// encodes to the same bytes.
			v, err := nn.Decode(st)
			if err != nil {
				t.Fatal(err)
			}
			enc, err := nn.Encode(ctx, v)
			if err != nil || v.DType() != dt || !bytes.Equal(enc.Data, st.Data) {
				t.Errorf("%s decoded as %s and encoded as %x, %v; want %x", name, v.DType(), enc.Data, err, st.Data)
			}
		}
		// A model of the file's dtype takes the values as they are.
		nn.ManualSeed(4)
		b := nn.NewLinear(4, 2)
		if err := nn.Cast(b, dt); err != nil {
			t.Fatal(err)
		}
		if err := nn.Load(f, b); err != nil {
			t.Fatal(err)
		}
		x := tensor.Randn(1, 3, 4)
		if b.Weight.Value().DType() != dt || !slices.Equal(data(t, a.Forward(x)), data(t, b.Forward(x))) {
			t.Errorf("%s model loaded as %s computes differently", dt, b.Weight.Value().DType())
		}
		// A float32 model keeps its dtype and takes the values widened.
		c := nn.NewLinear(4, 2)
		if err := nn.Load(f, c); err != nil {
			t.Fatal(err)
		}
		if c.Weight.Value().DType() != tensor.Float32 || !slices.Equal(floats(t, c.Weight.Value()), floats(t, a.Weight.Value().Cast(tensor.Float32))) {
			t.Errorf("%s model loaded into a float32 one: dtype %s, weights %v, want %v", dt, c.Weight.Value().DType(), floats(t, c.Weight.Value()), floats(t, a.Weight.Value().Cast(tensor.Float32)))
		}
	}
}

// TestHalfEncodingMatchesCast checks the safetensors conversions against
// the tensor's Cast: every element decodes to the value the devices give
// it, and any float32 rounds to the element they would store.
func TestHalfEncodingMatchesCast(t *testing.T) {
	ctx := t.Context()
	for _, dt := range []tensor.DType{tensor.BFloat16, tensor.Float16} {
		all := make([]byte, 2*65536)
		for i := range 65536 {
			binary.LittleEndian.PutUint16(all[2*i:], uint16(i))
		}
		st := safetensors.Tensor{DType: nn.DTypeName(dt), Shape: []int{65536}, Data: all}
		x, err := nn.Decode(st)
		if err != nil {
			t.Fatal(err)
		}
		decoded, _ := st.Float32s()
		enc, err := nn.Encode(ctx, x)
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range floats(t, x) {
			if v != decoded[i] && !(math.IsNaN(float64(v)) && math.IsNaN(float64(decoded[i]))) {
				t.Fatalf("%s %#04x decodes to %v, the devices give %v", dt, i, decoded[i], v)
			}
			if got := binary.LittleEndian.Uint16(enc.Data[2*i:]); got != uint16(i) && !math.IsNaN(float64(v)) {
				t.Fatalf("%s %#04x encoded as %#04x", dt, i, got)
			}
		}
		// Around every pair of neighbouring elements: their midpoint, a
		// tie, and the float32s on either side of it.
		var tie []float32
		for h := range 65535 {
			lo, hi := decoded[h], decoded[h+1]
			if math.IsInf(float64(lo), 0) || math.IsInf(float64(hi), 0) || math.IsNaN(float64(lo)) || math.IsNaN(float64(hi)) {
				continue
			}
			mid := float32((float64(lo) + float64(hi)) / 2)
			tie = append(tie, mid, math.Nextafter32(mid, float32(math.Inf(1))), math.Nextafter32(mid, float32(math.Inf(-1))))
		}
		rounded := floats(t, tensor.FromSlice(tie).Cast(dt))
		var file safetensors.Tensor
		if dt == tensor.BFloat16 {
			file = safetensors.BFloat16Tensor([]int{len(tie)}, tie)
		} else {
			file = safetensors.Float16Tensor([]int{len(tie)}, tie)
		}
		got, _ := file.Float32s()
		for i := range tie {
			if got[i] != rounded[i] {
				t.Fatalf("%s: %v (%#08x) written as %v, the devices store %v", dt, tie[i], math.Float32bits(tie[i]), got[i], rounded[i])
			}
		}
	}
}

// TestHalfLinearTrains fits a bfloat16 Linear layer with master weights
// as closely as its float32 twin, and without them still fits.
func TestHalfLinearTrains(t *testing.T) {
	ctx := t.Context()
	x := tensor.Randn(7, 64, 4)
	y := x.MatMul(tensor.FromSlice([]float32{1.1, -2.3, 0.7, 3.3}, 4, 1)).Add(tensor.Scalar(float32(0.45)))
	train := func(dt tensor.DType, opts ...optim.Option) []float32 {
		nn.ManualSeed(1)
		m := nn.NewLinear(4, 1)
		if err := nn.Cast(m, dt); err != nil {
			t.Fatal(err)
		}
		opt := optim.SGD(nn.Parameters(m), append([]optim.Option{optim.LR(0.1)}, opts...)...)
		var losses []float32
		for range 60 {
			l := loss.MSE(m.Forward(x), y)
			losses = append(losses, data(t, l)[0])
			if err := l.Backward(ctx); err != nil {
				t.Fatal(err)
			}
			if err := opt.Step(ctx); err != nil {
				t.Fatal(err)
			}
			opt.ZeroGrad()
		}
		if got := m.Weight.Value().DType(); got != dt {
			t.Fatalf("weights are %s after training, want %s", got, dt)
		}
		return losses
	}
	full := train(tensor.Float32)
	master := train(tensor.BFloat16, optim.MasterWeights())
	plain := train(tensor.BFloat16)
	last := len(full) - 1
	t.Logf("losses after %d steps: float32 %v, bfloat16 with master weights %v, bfloat16 %v", last+1, full[last], master[last], plain[last])
	if full[last] > full[0]/100 {
		t.Fatalf("float32 loss went from %v to %v", full[0], full[last])
	}
	if math.Abs(float64(master[0]-full[0])) > 1e-3*float64(full[0]) {
		t.Errorf("first losses differ: bfloat16 %v, float32 %v", master[0], full[0])
	}
	if math.Abs(float64(master[last]-full[last])) > 1e-2 {
		t.Errorf("bfloat16 loss %v after training with master weights, float32 %v", master[last], full[last])
	}
	if !(plain[last] < plain[0]/2) {
		t.Errorf("bfloat16 loss went from %v to %v without master weights", plain[0], plain[last])
	}
}
