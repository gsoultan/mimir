package cpu

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/backend"
)

// refGEMM computes a·b in float64, along with the sum of |a||b| over each
// dot product, which bounds the rounding error of a float32 result.
func refGEMM(a, b operand, m, n, k int) (want, scale []float64) {
	want, scale = make([]float64, m*n), make([]float64, m*n)
	for i := range m {
		for j := range n {
			var s, abs float64
			for p := range k {
				x := float64(a.data[a.off+i*a.rs+p*a.cs])
				y := float64(b.data[b.off+p*b.rs+j*b.cs])
				s += x * y
				abs += math.Abs(x * y)
			}
			want[i*n+j], scale[i*n+j] = s, abs
		}
	}
	return want, scale
}

// layout returns an operand holding a rows×cols matrix of random values
// stored in the named layout.
func layout(rng *rand.Rand, kind string, rows, cols int) operand {
	fill := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(rng.NormFloat64())
		}
		return s
	}
	switch kind {
	case "row-major":
		return operand{data: fill(rows*cols + 3), off: 3, rs: cols, cs: 1}
	case "column-major":
		return operand{data: fill(rows * cols), rs: 1, cs: rows}
	case "padded":
		return operand{data: fill(rows*(cols+5) + 2), off: 2, rs: cols + 5, cs: 1}
	case "strided":
		return operand{data: fill(2*rows*cols + 1), off: 1, rs: 2 * cols, cs: 2}
	case "broadcast rows":
		return operand{data: fill(cols), rs: 0, cs: 1}
	default:
		panic(kind)
	}
}

func TestGEMMKernelsMatchReference(t *testing.T) {
	be := New()
	for _, g := range gemmKernels {
		t.Run(g.name, func(t *testing.T) {
			rng := rand.New(rand.NewPCG(1, 2))
			mr, nr, kc, mc := g.mr, g.nr, g.kc, g.mc
			shapes := [][3]int{
				{1, 1, 1}, {mr, nr, 1}, {mr - 1, nr + 1, 7}, {mr + 1, nr - 1, kc},
				{2*mr + 3, 3*nr + 5, kc + 1}, {mc + mr + 1, nr, 2*kc + 3}, {3, 2 * nr, 0},
				{37, 1, 19}, {1, 53, 300}, {67, 45, 31},
			}
			layouts := []string{"row-major", "column-major", "padded", "strided", "broadcast rows"}
			for _, s := range shapes {
				m, n, k := s[0], s[1], s[2]
				for i, la := range layouts {
					lb := layouts[(i+2)%len(layouts)]
					a, b := layout(rng, la, m, k), layout(rng, lb, k, n)
					out := make([]float32, m*n)
					for i := range out {
						out[i] = float32(math.NaN()) // must be overwritten
					}
					if err := be.sgemm(t.Context(), g, out, a, []int{a.off}, b, []int{b.off}, m, n, k, nil); err != nil {
						t.Fatal(err)
					}
					want, scale := refGEMM(a, b, m, n, k)
					for e, got := range out {
						if diff := math.Abs(float64(got) - want[e]); !(diff <= 1e-5*scale[e]+1e-7) {
							t.Fatalf("%dx%dx%d, a %s, b %s: element %d = %v, want %v", m, n, k, la, lb, e, got, want[e])
						}
					}
				}
			}
		})
	}
}

func TestGEMMBatchesAndSplits(t *testing.T) {
	// Big enough products are split between workers, by batch item or
	// within an item; every split must produce the same result as
	// computing each matrix whole.
	rng := rand.New(rand.NewPCG(3, 4))
	shapes := []struct{ batch, m, n, k int }{
		{3, 70, 90, 40},    // split by item
		{2, 150, 300, 600}, // split by rows, with several blocks of k
		{1, 6, 2100, 300},  // split by columns, with several blocks of n
		{2, 37, 0, 5},      // empty
		{2, 400, 1000, 0},  // zero depth
	}
	for _, s := range shapes {
		a := layout(rng, "row-major", s.batch*s.m, s.k)
		b := layout(rng, "column-major", s.k, s.batch*s.n)
		offA, offB := make([]int, s.batch), make([]int, s.batch)
		for i := range s.batch {
			offA[i], offB[i] = a.off+i*s.m*s.k, i*s.n*s.k
		}
		whole := New()
		whole.workers = 1
		for _, g := range gemmKernels {
			want := make([]float32, s.batch*s.m*s.n)
			if err := whole.sgemm(t.Context(), g, want, a, offA, b, offB, s.m, s.n, s.k, nil); err != nil {
				t.Fatal(err)
			}
			for _, workers := range []int{2, 3, 4, 7, 16} {
				be := New()
				be.workers = workers
				got := make([]float32, len(want))
				for i := range got {
					got[i] = float32(math.NaN()) // must be overwritten
				}
				if err := be.sgemm(t.Context(), g, got, a, offA, b, offB, s.m, s.n, s.k, nil); err != nil {
					t.Fatal(err)
				}
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("%+v, %s with %d workers: element %d = %v, want %v", s, g.name, workers, i, got[i], want[i])
					}
				}
			}
		}
	}
}

func TestGEMMTileAccumulates(t *testing.T) {
	for _, g := range gemmKernels {
		const kc = 3
		a := make([]float32, kc*g.mr)
		b := make([]float32, kc*g.nr)
		for i := range a {
			a[i] = float32(i%5) - 2
		}
		for i := range b {
			b[i] = float32(i%3) + 0.5
		}
		ldc := g.nr + 3
		c := make([]float32, g.mr*ldc)
		for i := range c {
			c[i] = 100
		}
		g.tile(kc, &a[0], &b[0], &c[0], ldc, true)
		for i := range g.mr {
			for j := range ldc {
				want := float32(100)
				if j < g.nr {
					for p := range kc {
						want += a[p*g.mr+i] * b[p*g.nr+j]
					}
				}
				if got := c[i*ldc+j]; got != want {
					t.Fatalf("%s: c[%d][%d] = %v, want %v", g.name, i, j, got, want)
				}
			}
		}
	}
}

func TestPlanGEMMGivesEveryWorkerAShare(t *testing.T) {
	g := &gemmKernel{mr: 6, nr: 16}
	tests := []struct {
		m, n, k, workers int
		want             gemmPlan
	}{
		{8, 8, 8, 4, gemmPlan{mStep: 8, nStep: 8, mParts: 1, nParts: 1}},           // too small to split
		{768, 768, 768, 4, gemmPlan{mStep: 192, nStep: 768, mParts: 4, nParts: 1}}, // rows first
		{6, 2048, 512, 4, gemmPlan{mStep: 6, nStep: 512, mParts: 1, nParts: 4}},    // then columns
		{768, 768, 768, 3, gemmPlan{mStep: 258, nStep: 768, mParts: 3, nParts: 1}}, // uneven
	}
	for _, tt := range tests {
		if got := planGEMM(g, tt.m, tt.n, tt.k, tt.workers); got != tt.want {
			t.Errorf("planGEMM(%d, %d, %d, %d workers) = %+v, want %+v", tt.m, tt.n, tt.k, tt.workers, got, tt.want)
		}
	}
}

func BenchmarkSGEMM(b *testing.B) {
	for _, g := range gemmKernels {
		for _, size := range []int{64, 256, 512, 1024, 2048} {
			b.Run(fmt.Sprintf("%s/%d", g.name, size), func(b *testing.B) {
				rng := rand.New(rand.NewPCG(1, 1))
				x, y := layout(rng, "row-major", size, size), layout(rng, "row-major", size, size)
				out := make([]float32, size*size)
				be := New()
				for b.Loop() {
					if err := be.sgemm(b.Context(), g, out, x, []int{x.off}, y, []int{y.off}, size, size, size, nil); err != nil {
						b.Fatal(err)
					}
				}
				b.ReportMetric(2*float64(size*size*size)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOP/s")
			})
		}
	}
}

func TestMatMulEpilogueMatchesSeparateKernels(t *testing.T) {
	prog := []backend.Instr{
		{Op: backend.OpLoad, DType: backend.Float32, Args: []int{0}}, // the product
		{Op: backend.OpLoad, DType: backend.Float32, Args: []int{1}}, // a bias, broadcast
		{Op: backend.OpLoad, DType: backend.Float32, Args: []int{2}}, // a residual
		{Op: backend.OpAdd, DType: backend.Float32, Args: []int{0, 1}},
		{Op: backend.OpAdd, DType: backend.Float32, Args: []int{3, 2}},
		{Op: backend.OpConst, DType: backend.Float32},
		{Op: backend.OpMax, DType: backend.Float32, Args: []int{4, 5}},
	}
	rng := rand.New(rand.NewPCG(5, 6))
	for _, s := range []struct{ batch, m, n, k, workers int }{
		{6, 7, 33, 20, 3},     // an item per worker
		{1, 150, 300, 600, 4}, // one item shared by every worker
		{1, 6, 2100, 300, 4},  // shared, split by columns
		{2, 4, 5, 0, 2},       // no inner dimension
	} {
		name := fmt.Sprintf("%+v", s)
		be := New()
		be.workers = s.workers
		a := layout(rng, "row-major", s.batch*s.m, s.k)
		b := layout(rng, "column-major", s.k, s.n)
		size := s.batch * s.m * s.n
		in := []backend.View{
			{Shape: []int{s.batch, s.m, s.k}, Strides: []int{s.m * a.rs, a.rs, a.cs}, Offset: a.off},
			{Shape: []int{s.batch, s.k, s.n}, Strides: []int{0, b.rs, b.cs}},
		}
		ins := []backend.Buffer{&buffer[float32]{data: a.data}, &buffer[float32]{data: b.data}}
		// The epilogue sees the product as [batch*m, n].
		eshape := []int{s.batch * s.m, s.n}
		bias, residual := make([]float32, s.n), make([]float32, size)
		for i := range bias {
			bias[i] = rng.Float32() - 0.5
		}
		for i := range residual {
			residual[i] = rng.Float32() - 0.5
		}
		views := []backend.View{{Shape: eshape, Strides: []int{0, 1}}, backend.Contiguous(eshape)}
		bufs := []backend.Buffer{f32buf(bias...), f32buf(residual...)}

		product, _ := be.Alloc(backend.Float32, size)
		plain := &backend.Kernel{Op: backend.OpMatMul, DType: backend.Float32, Shape: []int{s.batch, s.m, s.n}, In: in}
		if err := be.Run(t.Context(), plain, []backend.Buffer{product}, ins); err != nil {
			t.Fatal(err)
		}
		want, _ := be.Alloc(backend.Float32, size)
		fused := &backend.Kernel{Op: backend.OpFused, DType: backend.Float32, Shape: eshape, In: append([]backend.View{backend.Contiguous(eshape)}, views...), Attrs: backend.Attrs{Program: prog}}
		if err := be.Run(t.Context(), fused, []backend.Buffer{want}, append([]backend.Buffer{product}, bufs...)); err != nil {
			t.Fatal(err)
		}

		got, _ := be.Alloc(backend.Float32, size)
		withEpilogue := &backend.Kernel{Op: backend.OpMatMul, DType: backend.Float32, Shape: plain.Shape, In: append(slices.Clone(in), views...), Attrs: backend.Attrs{Program: prog}}
		if err := be.Run(t.Context(), withEpilogue, []backend.Buffer{got}, append(slices.Clone(ins), bufs...)); err != nil {
			t.Fatal(err)
		}
		g, _ := data[float32](got)
		w, _ := data[float32](want)
		if !slices.Equal(g, w) {
			t.Errorf("%s: product with epilogue differs from the separate kernels", name)
		}
	}
}

func TestInt32MatMulEpilogue(t *testing.T) {
	prog := []backend.Instr{
		{Op: backend.OpLoad, DType: backend.Int32, Args: []int{0}},
		{Op: backend.OpConst, DType: backend.Int32, Value: 3},
		{Op: backend.OpMul, DType: backend.Int32, Args: []int{0, 1}},
	}
	k := &backend.Kernel{
		Op: backend.OpMatMul, DType: backend.Int32, Shape: []int{2, 2},
		In:    []backend.View{backend.Contiguous([]int{2, 3}), backend.Contiguous([]int{3, 2})},
		Attrs: backend.Attrs{Program: prog},
	}
	got, err := run[int32](t, k, i32buf(1, 2, 3, 4, 5, 6), i32buf(1, 0, 0, 1, 1, 1))
	if want := []int32{12, 15, 30, 33}; err != nil || !slices.Equal(got, want) {
		t.Fatalf("got %v, %v; want %v", got, err, want)
	}
}
