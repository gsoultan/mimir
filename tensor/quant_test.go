package tensor_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/gsoultan/mimir/internal/goldentest"
	"github.com/gsoultan/mimir/internal/quanttest"
	"github.com/gsoultan/mimir/tensor"
)

var quantTypes = []tensor.DType{tensor.Q8_0, tensor.Q4_K, tensor.Q6_K}

// quantized returns a random [n, k] tensor of dt and its values.
func quantized(rng *rand.Rand, dt tensor.DType, n, k int) (*tensor.Tensor, []float32) {
	raw := quanttest.Random(rng, dt, n*k)
	return tensor.FromBytes(dt, raw, n, k), quanttest.Dequantize(dt, raw)
}

func TestQuantizedTensorsDequantizeLikeGGML(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	ctx := context.Background()
	for _, dt := range quantTypes {
		w, want := quantized(rng, dt, 3, 2*dt.BlockSize())
		if w.DType() != dt || w.Size() != len(want) {
			t.Fatalf("%s: got %v", dt, w)
		}
		got, err := w.Cast(tensor.Float32).Data[float32](ctx)
		if err != nil {
			t.Fatal(err)
		}
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("%s: element %d = %v, want %v", dt, i, got[i], want[i])
			}
		}
		// A cast that dequantizes is not fused into what reads it.
		y, err := w.Cast(tensor.Float32).Mul(tensor.Scalar[float32](2)).Data[float32](ctx)
		if err != nil {
			t.Fatal(err)
		}
		for i := range want {
			if y[i] != 2*want[i] {
				t.Fatalf("%s: twice element %d = %v, want %v", dt, i, y[i], 2*want[i])
			}
		}
		half, err := w.Cast(tensor.BFloat16).Cast(tensor.Float32).Data[float32](ctx)
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(float64(half[5]-want[5])) > math.Abs(float64(want[5]))/128 {
			t.Errorf("%s: bfloat16 element 5 = %v, want about %v", dt, half[5], want[5])
		}
	}
}

func TestQuantizedTensorsRefuseOtherOperations(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	w, _ := quantized(rng, tensor.Q4_K, 2, 256)
	ops := map[string]func() *tensor.Tensor{
		"add":       func() *tensor.Tensor { return w.Add(w) },
		"neg":       func() *tensor.Tensor { return w.Neg() },
		"reshape":   func() *tensor.Tensor { return w.Reshape(4, 128) },
		"transpose": func() *tensor.Tensor { return w.Transpose(0, 1) },
		"sum":       func() *tensor.Tensor { return w.Sum() },
		"matmul":    func() *tensor.Tensor { return w.MatMul(tensor.Zeros(tensor.Float32, 256, 3)) },
		"to int":    func() *tensor.Tensor { return w.Cast(tensor.Int32) },
		"clone":     func() *tensor.Tensor { return w.Clone() },
		"grad":      func() *tensor.Tensor { return w.RequiresGrad() },
		"by int":    func() *tensor.Tensor { return tensor.Zeros(tensor.Int32, 256).MatMulT(w) },
	}
	for name, op := range ops {
		if err := op().Err(); !errors.Is(err, tensor.ErrDType) {
			t.Errorf("%s: %v, want ErrDType", name, err)
		}
	}
	if _, err := w.Data[float32](context.Background()); !errors.Is(err, tensor.ErrDType) {
		t.Errorf("Data: %v, want ErrDType", err)
	}
	if err := tensor.NewParam(w).Value().Err(); !errors.Is(err, tensor.ErrDType) {
		t.Errorf("NewParam: %v, want ErrDType", err)
	}
	shapes := map[string]func() *tensor.Tensor{
		"part of a block": func() *tensor.Tensor { return tensor.FromBytes(tensor.Q4_K, make([]byte, 144), 2, 128) },
		"short":           func() *tensor.Tensor { return tensor.FromBytes(tensor.Q4_K, make([]byte, 143)) },
		"too many":        func() *tensor.Tensor { return tensor.FromBytes(tensor.Q8_0, make([]byte, 34), 64) },
		"other width":     func() *tensor.Tensor { return tensor.Zeros(tensor.Float32, 128).MatMulT(w) },
	}
	for name, op := range shapes {
		if err := op().Err(); !errors.Is(err, tensor.ErrShape) {
			t.Errorf("%s: %v, want ErrShape", name, err)
		}
	}
}

// productBound returns x·deqᵀ in float64 for x [m, k] and deq [n, k], and
// a bound on the error of a product that rounds each block of bs values
// of x to int8.
func productBound(x, deq []float32, m, n, k, bs int) (want, bound []float64) {
	want, bound = make([]float64, m*n), make([]float64, m*n)
	for i := range m {
		for j := range n {
			var s, e, mag float64
			for p := range k {
				amax := 0.0
				for _, v := range x[i*k+p/bs*bs : i*k+(p/bs+1)*bs] {
					amax = math.Max(amax, math.Abs(float64(v)))
				}
				w := float64(deq[j*k+p])
				s += float64(x[i*k+p]) * w
				e += math.Abs(w) * amax / 254 * 1.0001
				mag += math.Abs(w * float64(x[i*k+p]))
			}
			want[i*n+j], bound[i*n+j] = s, e+1e-5*mag+1e-30
		}
	}
	return want, bound
}

func TestMatMulTByQuantizedWeights(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	ctx := context.Background()
	for _, dt := range quantTypes {
		bs := 256
		if dt == tensor.Q8_0 {
			bs = 32
		}
		m, n, k := 6, 5, 512
		w, deq := quantized(rng, dt, n, k)
		x := make([]float32, m*k)
		for i := range x {
			x[i] = rng.Float32()*2 - 1
		}
		want, bound := productBound(x, deq, m, n, k, bs)
		t.Run(dt.String(), func(t *testing.T) {
			got, err := tensor.FromSlice(x, 2, 3, k).MatMulT(w).Data[float32](ctx)
			if err != nil {
				t.Fatal(err)
			}
			for i, g := range got {
				if math.Abs(float64(g)-want[i]) > bound[i] {
					t.Fatalf("element %d = %v, want %v ± %v", i, g, want[i], bound[i])
				}
			}
			// A vector, and a float16 input rounded to float16 first.
			v, err := tensor.FromSlice(x[:k]).MatMulT(w).Data[float32](ctx)
			if err != nil || len(v) != n || v[2] != got[2] {
				t.Fatalf("vector product = %v, %v; want %v first", v, err, got[:n])
			}
			h := tensor.FromSlice(x[:k]).Cast(tensor.Float16).MatMulT(w)
			if h.DType() != tensor.Float16 {
				t.Fatalf("float16 product is %s", h.DType())
			}
			hv, err := h.Cast(tensor.Float32).Data[float32](ctx)
			if err != nil || math.Abs(float64(hv[1])-want[1]) > 2*bound[1]+math.Abs(want[1])/256 {
				t.Fatalf("float16 product = %v, %v; want about %v", hv, err, want[:n])
			}
		})
	}
}

func TestMatMulTByFloatWeightsIsMatMulByTheTranspose(t *testing.T) {
	ctx := context.Background()
	x, w := tensor.Randn(1, 2, 3, 4), tensor.Randn(2, 5, 4)
	a, err := x.MatMulT(w).Data[float32](ctx)
	if err != nil {
		t.Fatal(err)
	}
	b, err := x.MatMul(w.Transpose(0, 1)).Data[float32](ctx)
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprint(a) != fmt.Sprint(b) {
		t.Fatalf("MatMulT = %v, MatMul by the transpose = %v", a, b)
	}
	if err := x.MatMulT(tensor.Randn(3, 4)).Err(); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("product with the transpose of a vector: %v, want ErrShape", err)
	}
}

// TestMatMulTGradientIsTheDequantizedProducts checks that the gradient
// with respect to x is the gradient of the product with the dequantized
// weights, which central differences of that product give.
func TestMatMulTGradientIsTheDequantizedProducts(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	ctx := context.Background()
	m, n, k := 3, 4, 256
	w, deq := quantized(rng, tensor.Q6_K, n, k)
	x, gout := make([]float32, m*k), make([]float32, m*n)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}
	for i := range gout {
		gout[i] = rng.Float32()*2 - 1
	}
	xt := tensor.FromSlice(x, m, k).RequiresGrad()
	loss := xt.MatMulT(w).Mul(tensor.FromSlice(gout, m, n)).Sum()
	if err := loss.Backward(ctx); err != nil {
		t.Fatal(err)
	}
	got, err := xt.Grad().Data[float32](ctx)
	if err != nil {
		t.Fatal(err)
	}
	f := func(x []float64) float64 {
		var s float64
		for i := range m {
			for j := range n {
				var y float64
				for p := range k {
					y += x[i*k+p] * float64(deq[j*k+p])
				}
				s += y * float64(gout[i*n+j])
			}
		}
		return s
	}
	wide := make([]float64, len(x))
	for i, v := range x {
		wide[i] = float64(v)
	}
	for _, i := range []int{0, 7, 300, m*k - 1} {
		h := 1e-3
		up, down := slicesClone(wide), slicesClone(wide)
		up[i] += h
		down[i] -= h
		want := (f(up) - f(down)) / (2 * h)
		if math.Abs(float64(got[i])-want) > 1e-4*math.Max(1, math.Abs(want)) {
			t.Errorf("gradient %d = %v, want %v", i, got[i], want)
		}
	}
}

func slicesClone(s []float64) []float64 { return append([]float64(nil), s...) }

func TestGatherFromQuantizedTable(t *testing.T) {
	rng := rand.New(rand.NewPCG(9, 10))
	table, deq := quantized(rng, tensor.Q8_0, 10, 64)
	idx := tensor.FromSlice([]int32{3, 9, 3}, 3, 1).Expand(3, 64)
	rows := table.Gather(0, idx)
	if rows.DType() != tensor.Float32 {
		t.Fatalf("rows are %s", rows.DType())
	}
	got, err := rows.Data[float32](context.Background())
	if err != nil {
		t.Fatal(err)
	}
	for i, r := range []int{3, 9, 3} {
		for j := range 64 {
			if got[i*64+j] != deq[r*64+j] {
				t.Fatalf("row %d element %d = %v, want %v", i, j, got[i*64+j], deq[r*64+j])
			}
		}
	}
}

func TestBufferHoldsQuantizedWeightsInPlace(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 12))
	raw := quanttest.Random(rng, tensor.Q4_K, 512)
	p := tensor.NewBuffer(tensor.FromBytes(tensor.Q4_K, raw, 2, 256))
	if err := p.Value().Err(); err != nil {
		t.Fatal(err)
	}
	before, err := p.Value().Cast(tensor.Float32).Data[float32](context.Background())
	if err != nil {
		t.Fatal(err)
	}
	raw[16] ^= 0xff // a quant of the first element
	after, err := p.Value().Cast(tensor.Float32).Data[float32](context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if before[0] == after[0] {
		t.Error("the buffer holds a copy of the weights")
	}
}

// TestQuantGolden checks quantized weights against tools/golden: their
// dequantization bit for bit, as ggml rounds it, and products by them
// with their gradients. The golden activations are ones the CPU
// backend rounds to int8 exactly, so its products differ from the
// oracle's only in the order of their sums.
func TestQuantGolden(t *testing.T) {
	cases, _ := goldentest.Load(t, "quant")
	dtypes := map[string]tensor.DType{"q8_0": tensor.Q8_0, "q4_K": tensor.Q4_K, "q6_K": tensor.Q6_K}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			cut := strings.LastIndexByte(name, '_')
			kind, op := name[:max(cut, 0)], name[cut+1:]
			dt, ok := dtypes[kind]
			raw := c.Inputs[len(c.Inputs)-1]
			if !ok || raw.DType != "U8" || len(raw.Shape) != 2 {
				t.Fatalf("weights of %s %s%v", kind, raw.DType, raw.Shape)
			}
			rows, k := raw.Shape[0], raw.Shape[1]/dt.BlockBytes()*dt.BlockSize()
			w := tensor.FromBytes(dt, raw.Data, rows, k)
			switch op {
			case "dequantize":
				goldentest.CompareTensor(t, c.Want, w.Cast(tensor.Float32), goldentest.Tol{})
			case "matmul":
				x := goldentest.Tensor(t, c.Inputs[0]).RequiresGrad()
				goldentest.CheckGradients(t, c, x.MatMulT(w), map[int]goldentest.Grader{0: x}, goldentest.Tol{Atol: 1e-5, Rtol: 1e-5})
			default:
				t.Fatalf("unknown case %s", name)
			}
		})
	}
	if len(cases) != 6 {
		t.Errorf("%d golden cases, want 6", len(cases))
	}
}
