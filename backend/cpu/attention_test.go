package cpu

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/backend"
)

// place lays out the row-major values of a tensor of shape with its
// dimensions stored in the order perm, and returns a buffer of dt holding
// them and the view that reads them back in the order of shape.
func place(t *testing.T, dt backend.DType, vals []float64, shape, perm []int) (backend.Buffer, backend.View) {
	t.Helper()
	stored := make([]int, len(shape))
	inv := make([]int, len(shape))
	for i, p := range perm {
		stored[i] = shape[p]
		inv[p] = i
	}
	v := backend.Contiguous(stored).Permute(inv)
	f := make([]float32, len(vals))
	c := newCursor(shape, []backend.View{v}, 0)
	for _, x := range vals {
		f[c.offs[0]] = float32(x)
		c.next()
	}
	buf := f32buf(f...)
	if dt != backend.Float32 {
		h, err := New().Alloc(dt, len(f))
		if err != nil {
			t.Fatal(err)
		}
		k := &backend.Kernel{Op: backend.OpCast, DType: dt, Shape: []int{len(f)}, In: []backend.View{backend.Contiguous([]int{len(f)})}}
		if err := New().Run(t.Context(), k, []backend.Buffer{h}, []backend.Buffer{buf}); err != nil {
			t.Fatal(err)
		}
		buf = h
	}
	return buf, v
}

// rounded returns vals rounded to dt.
func rounded(dt backend.DType, vals []float64) []float64 {
	out := slices.Clone(vals)
	if dt.Size() == 2 {
		for i, v := range out {
			out[i] = float64(halfToF32(dt, f32ToHalf(dt, float32(v))))
		}
	}
	return out
}

type attnCase struct {
	name            string
	batch           []int
	h, hk, q, k     int
	d, dv           int
	causal          bool
	offset          int
	mask            bool // broadcast over heads
	strided         bool // queries, keys and values stored [B..., S, H, D]
	dt              backend.DType
	workers         int
	scale           float64
	fullyMaskedRows bool
	hidden          int // keys at the end that the mask hides from every query, as an empty part of a cache
}

// attnRef computes an attention in float64: the result, the logsumexp
// and, for the result gradient gout, the gradients of q, k and v.
func attnRef(c attnCase, q, k, v, mask, gout []float64) (o, lse, dq, dk, dv []float64) {
	nb := 1
	for _, d := range c.batch {
		nb *= d
	}
	g := c.h / c.hk
	o, lse = make([]float64, nb*c.h*c.q*c.dv), make([]float64, nb*c.h*c.q)
	dq, dk, dv = make([]float64, nb*c.h*c.q*c.d), make([]float64, nb*c.hk*c.k*c.d), make([]float64, nb*c.hk*c.k*c.dv)
	for b := range nb {
		for h := range c.h {
			kh := b*c.hk + h/g
			for i := range c.q {
				qi := (b*c.h+h)*c.q + i
				s := make([]float64, c.k)
				m := math.Inf(-1)
				for j := range c.k {
					var dot float64
					for x := range c.d {
						dot += q[qi*c.d+x] * k[(kh*c.k+j)*c.d+x]
					}
					s[j] = c.scale * dot
					if mask != nil {
						s[j] += mask[(b*c.q+i)*c.k+j]
					}
					if c.causal && j > i+c.offset {
						s[j] = math.Inf(-1)
					}
					m = max(m, s[j])
				}
				if math.IsInf(m, -1) {
					lse[qi] = m
					continue
				}
				var l float64
				for j := range s {
					s[j] = math.Exp(s[j] - m)
					l += s[j]
				}
				lse[qi] = m + math.Log(l)
				dp := make([]float64, c.k)
				var dsum float64
				for j := range s {
					s[j] /= l
					for x := range c.dv {
						o[qi*c.dv+x] += s[j] * v[(kh*c.k+j)*c.dv+x]
						dv[(kh*c.k+j)*c.dv+x] += s[j] * gout[qi*c.dv+x]
						dp[j] += gout[qi*c.dv+x] * v[(kh*c.k+j)*c.dv+x]
					}
					dsum += s[j] * dp[j]
				}
				for j := range s {
					ds := s[j] * (dp[j] - dsum) * c.scale
					for x := range c.d {
						dq[qi*c.d+x] += ds * k[(kh*c.k+j)*c.d+x]
						dk[(kh*c.k+j)*c.d+x] += ds * q[qi*c.d+x]
					}
				}
			}
		}
	}
	return o, lse, dq, dk, dv
}

func TestAttentionMatchesReference(t *testing.T) {
	cases := []attnCase{
		{name: "small", batch: []int{2}, h: 2, hk: 2, q: 5, k: 7, d: 4, dv: 3},
		{name: "tiles and blocks", batch: []int{}, h: 4, hk: 2, q: 70, k: 300, d: 16, dv: 24, causal: true, offset: 230},
		{name: "causal from the start", batch: []int{1}, h: 3, hk: 1, q: 130, k: 130, d: 32, dv: 32, causal: true},
		{name: "masked", batch: []int{2, 3}, h: 2, hk: 1, q: 9, k: 33, d: 8, dv: 8, mask: true, causal: true},
		{name: "strided", batch: []int{2}, h: 4, hk: 4, q: 17, k: 19, d: 64, dv: 64, strided: true},
		{name: "no visible keys", batch: []int{1}, h: 2, hk: 1, q: 6, k: 6, d: 5, dv: 5, causal: true, offset: -3, fullyMaskedRows: true},
		{name: "split between workers", batch: []int{1}, h: 1, hk: 1, q: 200, k: 600, d: 8, dv: 8, workers: 8},
		{name: "decoding", batch: []int{1}, h: 8, hk: 2, q: 1, k: 50, d: 16, dv: 16, causal: true, offset: 49},
		{name: "bfloat16", batch: []int{2}, h: 2, hk: 1, q: 12, k: 20, d: 8, dv: 8, causal: true, dt: backend.BFloat16},
		{name: "float16", batch: []int{1}, h: 2, hk: 2, q: 12, k: 20, d: 8, dv: 8, mask: true, dt: backend.Float16},
		{name: "decoding from a cache", batch: []int{1}, h: 8, hk: 2, q: 1, k: 70, d: 16, dv: 16, mask: true, hidden: 61},
		{name: "decoding from a half cache", batch: []int{1}, h: 8, hk: 2, q: 2, k: 70, d: 16, dv: 12, mask: true, hidden: 55, dt: backend.Float16},
		{name: "decoding from a strided half cache", batch: []int{2}, h: 4, hk: 4, q: 1, k: 40, d: 8, dv: 8, mask: true, hidden: 33, strided: true, dt: backend.BFloat16},
		{name: "decoding from an empty cache", batch: []int{1}, h: 2, hk: 1, q: 1, k: 9, d: 4, dv: 4, mask: true, hidden: 9, dt: backend.Float16, fullyMaskedRows: true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if c.dt == backend.Invalid {
				c.dt = backend.Float32
			}
			if c.scale == 0 {
				c.scale = 1 / math.Sqrt(float64(c.d))
			}
			r := rand.New(rand.NewPCG(5, uint64(len(c.name))))
			rnd := func(n int, a float64) []float64 {
				out := make([]float64, n)
				for i := range out {
					out[i] = (r.Float64()*2 - 1) * a
				}
				return out
			}
			qs := slices.Concat(c.batch, []int{c.h, c.q, c.d})
			ks := slices.Concat(c.batch, []int{c.hk, c.k, c.d})
			vs := slices.Concat(c.batch, []int{c.hk, c.k, c.dv})
			n := func(s []int) int { x, _ := backend.NumElements(s); return x }
			q, k, v := rounded(c.dt, rnd(n(qs), 2)), rounded(c.dt, rnd(n(ks), 2)), rounded(c.dt, rnd(n(vs), 1))
			gout := rnd(n(slices.Concat(c.batch, []int{c.h, c.q, c.dv})), 1)
			perm := func(s []int) []int {
				p := make([]int, len(s))
				for i := range p {
					p[i] = i
				}
				if c.strided {
					r := len(s)
					p[r-3], p[r-2] = r-2, r-3
				}
				return p
			}
			qb, qv := place(t, c.dt, q, qs, perm(qs))
			kb, kv := place(t, c.dt, k, ks, perm(ks))
			vb, vv := place(t, c.dt, v, vs, perm(vs))
			ins, views := []backend.Buffer{qb, kb, vb}, []backend.View{qv, kv, vv}
			var mask []float64
			var mb backend.Buffer
			var mv backend.View
			if c.mask {
				// [B..., 1, Q, K], broadcast over the heads.
				ms := slices.Concat(c.batch, []int{1, c.q, c.k})
				mask = rounded(c.dt, rnd(n(ms), 3))
				for i := range mask {
					if i%c.k >= c.k-c.hidden {
						mask[i] = math.Inf(-1)
					}
				}
				mb, mv = place(t, c.dt, mask, ms, perm(ms))
				mv = mv.Expand(slices.Concat(c.batch, []int{c.h, c.q, c.k}))
				ins, views = append(ins, mb), append(views, mv)
			}
			attrs := backend.Attrs{Scale: c.scale, Causal: c.causal, CausalOffset: c.offset}
			s, err := backend.NewAttentionShape(qs, ks, vs, nil)
			if err != nil {
				t.Fatal(err)
			}
			be := New()
			if c.workers > 0 {
				be.workers = c.workers
			}
			fwd := &backend.Kernel{Op: backend.OpAttention, DType: backend.Float32, Shape: []int{s.Size()}, In: views, Attrs: attrs}
			got, err := runWith[float32](t, be, fwd, ins...)
			if err != nil {
				t.Fatal(err)
			}
			wantO, wantL, wantDQ, wantDK, wantDV := attnRef(c, q, k, v, mask, gout)
			tol := 2e-5
			if c.dt != backend.Float32 {
				tol = 2e-3 // the result gradient and the forward results are rounded
			}
			check := func(what string, got []float32, want []float64) {
				t.Helper()
				for i, w := range want {
					g := float64(got[i])
					if math.IsInf(w, -1) && math.IsInf(g, -1) {
						continue
					}
					if math.Abs(g-w) > tol*math.Max(1, math.Abs(w)) || math.IsNaN(g) {
						t.Fatalf("%s[%d] = %v, want %v", what, i, g, w)
					}
				}
			}
			no := len(wantO)
			check("result", got[:no], wantO)
			check("logsumexp", got[no:], wantL)

			// The gradient kernel reads the result and logsumexp from the
			// forward output.
			out := f32buf(got...)
			gb, gv := place(t, c.dt, gout, s.OutShape(), perm(s.OutShape()))
			bins := []backend.Buffer{qb, kb, vb, out, out, gb}
			bviews := []backend.View{qv, kv, vv, backend.Contiguous(s.OutShape()), backend.Contiguous(s.LSEShape()), gv}
			bviews[4].Offset = no
			if c.mask {
				bins, bviews = append(bins, mb), append(bviews, mv)
			}
			if c.dt != backend.Float32 {
				// The gradient of the result is rounded too.
				gout = rounded(c.dt, gout)
				_, _, wantDQ, wantDK, wantDV = attnRef(c, q, k, v, mask, gout)
			}
			bwd := &backend.Kernel{Op: backend.OpAttentionGrad, DType: backend.Float32, Shape: []int{s.GradSize()}, In: bviews, Attrs: attrs}
			grads, err := runWith[float32](t, be, bwd, bins...)
			if err != nil {
				t.Fatal(err)
			}
			nq, nk := len(wantDQ), len(wantDK)
			check("dq", grads[:nq], wantDQ)
			check("dk", grads[nq:nq+nk], wantDK)
			check("dv", grads[nq+nk:], wantDV)
		})
	}
}

func TestSliceScatter(t *testing.T) {
	x := f32buf(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
	src := f32buf(-1, -2, -3, -4)
	k := &backend.Kernel{Op: backend.OpSliceScatter, DType: backend.Float32, Shape: []int{3, 4},
		In:    []backend.View{backend.Contiguous([]int{4, 3}).Permute([]int{1, 0}), backend.Contiguous([]int{2, 2})},
		Attrs: backend.Attrs{Ranges: [][2]int{{1, 3}, {2, 4}}}}
	got, err := run[float32](t, k, x, src)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{0, 3, 6, 9, 1, 4, -1, -2, 2, 5, -3, -4}
	if !slices.Equal(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
	// In place, only the block is written.
	k.In[0] = backend.Contiguous([]int{3, 4})
	if err := New().Run(t.Context(), k, []backend.Buffer{x}, []backend.Buffer{x, src}); err != nil {
		t.Fatal(err)
	}
	want = []float32{0, 1, 2, 3, 4, 5, -1, -2, 8, 9, -3, -4}
	if d, _ := data[float32](x); !slices.Equal(d, want) {
		t.Errorf("in place: got %v, want %v", d, want)
	}
	// Any other kernel refuses to write over its input.
	k.Op = backend.OpGather
	if err := New().Run(t.Context(), k, []backend.Buffer{x}, []backend.Buffer{x, src}); err == nil {
		t.Error(fmt.Errorf("gather over its input: no error"))
	}
}

// BenchmarkAttention runs the attention kernels at the shapes of GPT-2
// small, causal, reporting the useful work of the visible scores.
func BenchmarkAttention(b *testing.B) {
	for _, seq := range []int{256, 1024} {
		shape := []int{8, 12, seq, 64}
		s, err := backend.NewAttentionShape(shape, shape, shape, nil)
		if err != nil {
			b.Fatal(err)
		}
		be := New()
		in := make([]backend.Buffer, 3)
		for i := range in {
			in[i], _ = be.Alloc(backend.Float32, s.Batches()*s.Heads*seq*64)
			d, _ := data[float32](in[i])
			for j := range d {
				d[j] = float32(j%17)/17 - 0.5
			}
		}
		c := backend.Contiguous(shape)
		attrs := backend.Attrs{Scale: 0.125, Causal: true}
		fwd := &backend.Kernel{Op: backend.OpAttention, DType: backend.Float32, Shape: []int{s.Size()}, In: []backend.View{c, c, c}, Attrs: attrs}
		out, _ := be.Alloc(backend.Float32, s.Size())
		visible := float64(8*12) * float64(seq) * float64(seq+1) / 2
		b.Run(fmt.Sprintf("forward/seq%d", seq), func(b *testing.B) {
			for b.Loop() {
				if err := be.Run(b.Context(), fwd, []backend.Buffer{out}, in); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(4*visible*64*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOP/s")
		})
		lse := backend.Contiguous(s.LSEShape())
		lse.Offset = s.Size() - s.Batches()*s.Heads*seq
		bwd := &backend.Kernel{Op: backend.OpAttentionGrad, DType: backend.Float32, Shape: []int{s.GradSize()}, In: []backend.View{c, c, c, c, lse, c}, Attrs: attrs}
		grads, _ := be.Alloc(backend.Float32, s.GradSize())
		b.Run(fmt.Sprintf("backward/seq%d", seq), func(b *testing.B) {
			for b.Loop() {
				if err := be.Run(b.Context(), bwd, []backend.Buffer{grads}, []backend.Buffer{in[0], in[1], in[2], out, out, in[0]}); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(10*visible*64*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOP/s")
		})
	}
}
