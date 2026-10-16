package cpu

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/backend"
)

// dense is a float64 matrix for checking results.
type dense struct {
	r, c int
	v    []float64
}

func denseOf(v []float32, r, c int) dense {
	d := dense{r, c, make([]float64, r*c)}
	for i, x := range v[:r*c] {
		d.v[i] = float64(x)
	}
	return d
}

func (a dense) at(i, j int) float64 { return a.v[i*a.c+j] }

func (a dense) mul(b dense) dense {
	out := dense{a.r, b.c, make([]float64, a.r*b.c)}
	for i := range a.r {
		for p := range a.c {
			x := a.at(i, p)
			for j := range b.c {
				out.v[i*b.c+j] += x * b.at(p, j)
			}
		}
	}
	return out
}

func (a dense) t() dense {
	out := dense{a.c, a.r, make([]float64, len(a.v))}
	for i := range a.r {
		for j := range a.c {
			out.v[j*a.r+i] = a.at(i, j)
		}
	}
	return out
}

// near fails unless a and b agree to tol relative to b's largest element.
func near(t *testing.T, what string, a, b dense, tol float64) {
	t.Helper()
	if a.r != b.r || a.c != b.c {
		t.Fatalf("%s: %dx%d, want %dx%d", what, a.r, a.c, b.r, b.c)
	}
	scale := 1.0
	for _, x := range b.v {
		scale = max(scale, math.Abs(x))
	}
	for i := range a.v {
		if d := math.Abs(a.v[i] - b.v[i]); !(d <= tol*scale) {
			t.Fatalf("%s: element (%d, %d) is %v, want %v", what, i/a.c, i%a.c, a.v[i], b.v[i])
		}
	}
}

// orthonormalColumns fails unless aᵀa = I.
func orthonormalColumns(t *testing.T, what string, a dense, tol float64) {
	t.Helper()
	g := a.t().mul(a)
	near(t, what+"ᵀ"+what, g, identity(a.c), tol)
}

func identity(n int) dense {
	d := dense{n, n, make([]float64, n*n)}
	for i := range n {
		d.v[i*n+i] = 1
	}
	return d
}

func randMatrix(rng *rand.Rand, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(rng.NormFloat64())
	}
	return v
}

// spd returns b symmetric positive definite n×n matrices.
func spd(rng *rand.Rand, b, n int) []float32 {
	out := make([]float32, 0, b*n*n)
	for range b {
		m := denseOf(randMatrix(rng, n*n), n, n)
		a := m.mul(m.t())
		for i := range n {
			a.v[i*n+i] += float64(n)
		}
		for _, x := range a.v {
			out = append(out, float32(x))
		}
	}
	return out
}

func linalgKernel(op backend.Op, shape []int, ins ...backend.View) *backend.Kernel {
	k := &backend.Kernel{Op: op, DType: backend.Float32, Shape: shape, In: ins}
	if op != backend.OpCholesky && op != backend.OpTriangularSolve {
		_, size, _ := backend.FactorParts(op, ins[0].Shape)
		k.Shape = []int{size}
	}
	return k
}

var linalgWorkers = []int{1, 4}

func TestCholeskyReconstructs(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for _, s := range []struct{ b, n int }{{1, 1}, {2, 3}, {1, 17}, {3, 64}, {1, 65}, {1, 200}} {
		x := spd(rng, s.b, s.n)
		for _, w := range linalgWorkers {
			t.Run(fmt.Sprintf("%dx%d workers %d", s.b, s.n, w), func(t *testing.T) {
				shape := []int{s.b, s.n, s.n}
				got := runOn[float32](t, w, linalgKernel(backend.OpCholesky, shape, backend.Contiguous(shape)), f32buf(x...))
				nn := s.n * s.n
				for i := range s.b {
					l := denseOf(got[i*nn:], s.n, s.n)
					for r := range s.n {
						for c := r + 1; c < s.n; c++ {
							if l.at(r, c) != 0 {
								t.Fatalf("L[%d, %d] = %v above the diagonal", r, c, l.at(r, c))
							}
						}
					}
					near(t, "L·Lᵀ", l.mul(l.t()), denseOf(x[i*nn:], s.n, s.n), 1e-5)
				}
			})
		}
	}
}

func TestCholeskyOfIndefiniteIsSingular(t *testing.T) {
	shape := []int{2, 2}
	_, err := run[float32](t, linalgKernel(backend.OpCholesky, shape, backend.Contiguous(shape)), f32buf(1, 2, 2, 1))
	if !errors.Is(err, backend.ErrSingular) {
		t.Fatalf("got %v, want ErrSingular", err)
	}
}

func TestLUReconstructs(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	for _, s := range []struct{ b, n int }{{1, 1}, {2, 4}, {1, 70}, {2, 150}} {
		x := randMatrix(rng, s.b*s.n*s.n)
		for _, w := range linalgWorkers {
			t.Run(fmt.Sprintf("%dx%d workers %d", s.b, s.n, w), func(t *testing.T) {
				shape := []int{s.b, s.n, s.n}
				got := runOn[float32](t, w, linalgKernel(backend.OpLU, shape, backend.Contiguous(shape)), f32buf(x...))
				n, nn := s.n, s.n*s.n
				for i := range s.b {
					lu := denseOf(got[i*nn:], n, n)
					perm := got[s.b*nn+i*n:][:n]
					sign := got[s.b*nn+s.b*n+i]
					l, u := identity(n), dense{n, n, make([]float64, nn)}
					for r := range n {
						for c := range n {
							if c < r {
								l.v[r*n+c] = lu.at(r, c)
							} else {
								u.v[r*n+c] = lu.at(r, c)
							}
						}
					}
					pa := dense{n, n, make([]float64, nn)}
					seen := make([]bool, n)
					for r, p := range perm {
						seen[int(p)] = true
						copy(pa.v[r*n:(r+1)*n], denseOf(x[i*nn+int(p)*n:], 1, n).v)
					}
					if slices.Contains(seen, false) {
						t.Fatalf("perm %v is not a permutation", perm)
					}
					near(t, "L·U", l.mul(u), pa, 1e-5)
					if want := permSign(perm); sign != want {
						t.Fatalf("sign %v, want %v", sign, want)
					}
				}
			})
		}
	}
}

func permSign(perm []float32) float32 {
	p := make([]int, len(perm))
	for i, v := range perm {
		p[i] = int(v)
	}
	s := float32(1)
	for i := range p {
		for p[i] != i {
			j := p[i]
			p[i], p[j] = p[j], p[i]
			s = -s
		}
	}
	return s
}

func TestTriangularSolve(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	for _, n := range []int{1, 5, 64, 130} {
		for _, k := range []int{1, 3, 70} {
			for _, upper := range []bool{false, true} {
				for _, unit := range []bool{false, true} {
					for _, transposed := range []bool{false, true} {
						name := fmt.Sprintf("n %d k %d upper %v unit %v transposed %v", n, k, upper, unit, transposed)
						t.Run(name, func(t *testing.T) {
							a := randMatrix(rng, n*n)
							for i := range n {
								a[i*n+i] = 2 + float32(rng.Float64()) // well conditioned
							}
							for i := range a {
								if r, c := i/n, i%n; r != c {
									a[i] *= 1 / float32(n)
								}
							}
							bm := randMatrix(rng, n*k)
							av := backend.Contiguous([]int{n, n})
							if transposed {
								av = av.Permute([]int{1, 0})
							}
							kk := linalgKernel(backend.OpTriangularSolve, []int{n, k}, av, backend.Contiguous([]int{n, k}))
							kk.Attrs.Upper, kk.Attrs.Unit = upper, unit
							for _, w := range linalgWorkers {
								got := runOn[float32](t, w, kk, f32buf(a...), f32buf(bm...))
								// The triangle the kernel reads, as a dense matrix.
								tri := dense{n, n, make([]float64, n*n)}
								for r := range n {
									for c := range n {
										v := float64(a[offsetOf(av, []int{r, c})])
										switch {
										case r == c && unit:
											tri.v[r*n+c] = 1
										case r == c || upper && c > r || !upper && c < r:
											tri.v[r*n+c] = v
										}
									}
								}
								near(t, "A·X", tri.mul(denseOf(got, n, k)), denseOf(bm, n, k), 1e-5)
							}
						})
					}
				}
			}
		}
	}
}

func TestTriangularSolveOfZeroDiagonalIsSingular(t *testing.T) {
	k := linalgKernel(backend.OpTriangularSolve, []int{2, 1}, backend.Contiguous([]int{2, 2}), backend.Contiguous([]int{2, 1}))
	_, err := run[float32](t, k, f32buf(1, 0, 3, 0), f32buf(1, 1))
	if !errors.Is(err, backend.ErrSingular) {
		t.Fatalf("got %v, want ErrSingular", err)
	}
}

func TestQRReconstructs(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	for _, s := range []struct{ b, m, n int }{{1, 1, 1}, {2, 5, 3}, {1, 3, 5}, {1, 4, 4}, {1, 130, 70}, {1, 70, 130}, {1, 200, 200}} {
		x := randMatrix(rng, s.b*s.m*s.n)
		for _, w := range linalgWorkers {
			t.Run(fmt.Sprintf("%dx%dx%d workers %d", s.b, s.m, s.n, w), func(t *testing.T) {
				shape := []int{s.b, s.m, s.n}
				got := runOn[float32](t, w, linalgKernel(backend.OpQR, shape, backend.Contiguous(shape)), f32buf(x...))
				m, n, k := s.m, s.n, min(s.m, s.n)
				for i := range s.b {
					q := denseOf(got[i*m*k:], m, k)
					r := denseOf(got[s.b*m*k+i*k*n:], k, n)
					for row := range k {
						for c := range row {
							if r.at(row, c) != 0 {
								t.Fatalf("R[%d, %d] = %v below the diagonal", row, c, r.at(row, c))
							}
						}
					}
					orthonormalColumns(t, "Q", q, 1e-5)
					near(t, "Q·R", q.mul(r), denseOf(x[i*m*n:], m, n), 1e-5)
				}
			})
		}
	}
}

// checkSVD checks u, s and vh against the m×n a.
func checkSVD(t *testing.T, a []float32, got []float32, m, n int, tol float64) {
	t.Helper()
	k := min(m, n)
	u := denseOf(got, m, k)
	s := got[m*k:][:k]
	vh := denseOf(got[m*k+k:], k, n)
	for i, v := range s {
		if v < 0 || i > 0 && v > s[i-1] {
			t.Fatalf("singular values %v are not descending and non-negative", s)
		}
	}
	orthonormalColumns(t, "U", u, tol)
	orthonormalColumns(t, "V", vh.t(), tol)
	us := dense{m, k, slices.Clone(u.v)}
	for i := range m {
		for j := range k {
			us.v[i*k+j] *= float64(s[j])
		}
	}
	near(t, "U·S·Vh", us.mul(vh), denseOf(a, m, n), tol)
}

func TestSVDReconstructs(t *testing.T) {
	rng := rand.New(rand.NewPCG(9, 10))
	for _, s := range []struct{ m, n int }{{1, 1}, {5, 3}, {3, 5}, {4, 4}, {64, 64}, {150, 100}, {100, 150}} {
		x := randMatrix(rng, s.m*s.n)
		for _, w := range linalgWorkers {
			t.Run(fmt.Sprintf("%dx%d workers %d", s.m, s.n, w), func(t *testing.T) {
				shape := []int{s.m, s.n}
				got := runOn[float32](t, w, linalgKernel(backend.OpSVD, shape, backend.Contiguous(shape)), f32buf(x...))
				checkSVD(t, x, got, s.m, s.n, 2e-5)
			})
		}
	}
}

func TestSVDOfRankDeficient(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 12))
	m, n := 6, 4
	x := randMatrix(rng, m*n)
	for i := range m {
		x[i*n+2] = x[i*n] // a repeated column
		x[i*n+3] = 0      // and a zero one
		x[i*n+1] *= 1e-3  // a small one
	}
	shape := []int{m, n}
	got := runOn[float32](t, 1, linalgKernel(backend.OpSVD, shape, backend.Contiguous(shape)), f32buf(x...))
	checkSVD(t, x, got, m, n, 2e-5)
	if s := got[m*n:][:n]; s[3] != 0 || s[2] > 1e-5 {
		t.Fatalf("singular values %v, want the last two zero", s)
	}
}

func TestEighReconstructs(t *testing.T) {
	rng := rand.New(rand.NewPCG(13, 14))
	diag := func(vals ...float32) []float32 {
		n := len(vals)
		out := make([]float32, n*n)
		for i, v := range vals {
			out[i*n+i] = v
		}
		return out
	}
	cases := []struct {
		name string
		n    int
		x    []float32
	}{
		{"1", 1, []float32{3}},
		{"identity", 5, diag(1, 1, 1, 1, 1)},
		{"repeated", 4, diag(2, -1, 2, 2)},
		{"zero", 3, make([]float32, 9)},
	}
	for _, n := range []int{2, 7, 64, 65, 200} {
		m := denseOf(randMatrix(rng, n*n), n, n)
		a := m.mul(identity(n)).v
		x := make([]float32, n*n)
		for i := range n {
			for j := range n {
				x[i*n+j] = float32(a[i*n+j] + a[j*n+i])
			}
		}
		cases = append(cases, struct {
			name string
			n    int
			x    []float32
		}{fmt.Sprint(n), n, x})
	}
	// A matrix with a repeated eigenvalue that is not diagonal: Q·D·Qᵀ.
	{
		n := 6
		q := denseOf(randMatrix(rng, n*n), n, n)
		qr := runOn[float32](t, 1, linalgKernel(backend.OpQR, []int{n, n}, backend.Contiguous([]int{n, n})), f32buf(q.v32()...))
		qq := denseOf(qr, n, n)
		d := denseOf(diag(1, 1, 1, 2, 3, 3), n, n)
		a := qq.mul(d).mul(qq.t())
		x := make([]float32, n*n)
		for i := range a.v {
			x[i] = float32(a.v[i])
		}
		for i := range n {
			for j := range i {
				x[i*n+j] = x[j*n+i]
			}
		}
		cases = append(cases, struct {
			name string
			n    int
			x    []float32
		}{"clustered", n, x})
	}
	for _, c := range cases {
		for _, w := range linalgWorkers {
			t.Run(fmt.Sprintf("%s workers %d", c.name, w), func(t *testing.T) {
				n := c.n
				// Garbage above the diagonal, which eigh must not read.
				x := slices.Clone(c.x)
				for i := range n {
					for j := i + 1; j < n; j++ {
						x[i*n+j] = 99
					}
				}
				shape := []int{n, n}
				got := runOn[float32](t, w, linalgKernel(backend.OpEigh, shape, backend.Contiguous(shape)), f32buf(x...))
				vals := got[:n]
				if !slices.IsSorted(vals) {
					t.Fatalf("eigenvalues %v are not ascending", vals)
				}
				v := denseOf(got[n:], n, n)
				orthonormalColumns(t, "V", v, 2e-5)
				vd := dense{n, n, slices.Clone(v.v)}
				for i := range n {
					for j := range n {
						vd.v[i*n+j] *= float64(vals[j])
					}
				}
				near(t, "V·W·Vᵀ", vd.mul(v.t()), denseOf(c.x, n, n), 2e-5)
			})
		}
	}
}

func (a dense) v32() []float32 {
	out := make([]float32, len(a.v))
	for i, x := range a.v {
		out[i] = float32(x)
	}
	return out
}
