package cpu

import (
	"context"
	"fmt"
	"math"
	"math/bits"
	"math/cmplx"
	"slices"
	"sync"

	"github.com/gsoultan/mimir/backend"
)

// A Fourier transform runs along lines, the complex elements that differ
// only in their position along the axis, which workers take whole. Each
// line is read into complex128 values, transformed in float64 and rounded
// once as it is stored, like the pocketfft that NumPy and PyTorch use.
//
// A line whose length has no prime factor above fftLargestRadix is
// transformed by mixed-radix Cooley-Tukey, as KissFFT does: the length
// is factored into radices, fours first, the transform recurses into the
// p subsequences of every p-th element for the first radix p, and
// butterflies combine their transforms, specialized for radices 2, 3 and
// 4. Any other length uses Bluestein's algorithm, which writes the
// transform as a convolution with a chirp and computes that with
// transforms of a power of two at least twice as long. The inverse
// direction conjugates before and after the forward transform.

// fftLargestRadix is the largest prime factor of a length that the
// mixed-radix transform takes, with a butterfly whose cost grows with it.
const fftLargestRadix = 31

// fftPlan holds what transforms of length n need: its radices and the
// twiddle factors e^(-2πik/n), or for Bluestein's algorithm the chirp
// e^(-πij²/n), the transform of its filter, divided by m, and the plan of
// length m.
type fftPlan struct {
	n       int
	radices []int
	twiddle []complex128
	chirp   []complex128
	filter  []complex128
	inner   *fftPlan
}

// fftPlans caches plans by length, up to fftCached of them.
var fftPlans struct {
	sync.Mutex
	m map[int]*fftPlan
}

const fftCached = 64

func planFFT(n int) *fftPlan {
	fftPlans.Lock()
	defer fftPlans.Unlock()
	if p, ok := fftPlans.m[n]; ok {
		return p
	}
	p := newFFTPlan(n)
	if fftPlans.m == nil || len(fftPlans.m) >= fftCached {
		fftPlans.m = map[int]*fftPlan{}
	}
	fftPlans.m[n] = p
	return p
}

// radices returns the factors of n, fours first, then twos and odd
// primes in increasing order, or nil if one exceeds fftLargestRadix.
func radices(n int) []int {
	var out []int
	for n%4 == 0 {
		out, n = append(out, 4), n/4
	}
	for p := 2; n > 1; p++ {
		if p > fftLargestRadix {
			return nil
		}
		for n%p == 0 {
			out, n = append(out, p), n/p
		}
	}
	return out
}

func newFFTPlan(n int) *fftPlan {
	p := &fftPlan{n: n, radices: radices(n)}
	if n <= 1 || p.radices != nil {
		p.twiddle = make([]complex128, n)
		for k := range p.twiddle {
			s, c := math.Sincos(2 * math.Pi * float64(k) / float64(n))
			p.twiddle[k] = complex(c, -s)
		}
		return p
	}
	m := 1 << bits.Len(uint(2*n-2))
	p.inner = newFFTPlan(m)
	p.chirp = make([]complex128, n)
	for j := range p.chirp {
		// j² mod 2n, exactly, gives the angle πj²/n.
		sq := uint64(j) * uint64(j) % uint64(2*n)
		s, c := math.Sincos(math.Pi * float64(sq) / float64(n))
		p.chirp[j] = complex(c, -s)
	}
	p.filter = make([]complex128, m)
	p.filter[0] = cmplx.Conj(p.chirp[0])
	for j := 1; j < n; j++ {
		p.filter[j] = cmplx.Conj(p.chirp[j])
		p.filter[m-j] = p.filter[j]
	}
	work := make([]complex128, m)
	p.inner.transform(p.filter, work)
	scale := complex(1/float64(m), 0)
	for i := range p.filter {
		p.filter[i] *= scale
	}
	return p
}

// transform replaces x, of the plan's length, with its forward transform,
// using work, which grows as needed, as scratch. It returns work.
func (p *fftPlan) transform(x, work []complex128) []complex128 {
	n := p.n
	switch {
	case n <= 1:
	case p.radices != nil:
		work = grow(work, n)
		p.mixed(work[:n], x, 1, p.radices, n)
		copy(x, work[:n])
	default:
		m := p.inner.n
		work = grow(work, 2*m)
		a, scratch := work[:m], work[m:2*m]
		for j, v := range x {
			a[j] = v * p.chirp[j]
		}
		clear(a[n:])
		p.inner.mixed(scratch, a, 1, p.inner.radices, m)
		for i, f := range p.filter {
			a[i] = cmplx.Conj(scratch[i] * f)
		}
		p.inner.mixed(scratch, a, 1, p.inner.radices, m) // the inverse, conjugated
		for k := range x {
			x[k] = cmplx.Conj(scratch[k]) * p.chirp[k]
		}
	}
	return work
}

// transformParallel is transform with the workers sharing the
// transforms of the first radix's subsequences and their butterflies.
func (p *fftPlan) transformParallel(ctx context.Context, b *Backend, x, work []complex128) ([]complex128, error) {
	n := p.n
	switch {
	case n <= 1:
	case p.radices != nil:
		work = grow(work, n)
		if err := p.mixedParallel(ctx, b, work[:n], x); err != nil {
			return work, err
		}
		copy(x, work[:n])
	default:
		m := p.inner.n
		work = grow(work, 2*m)
		a, scratch := work[:m], work[m:2*m]
		for j, v := range x {
			a[j] = v * p.chirp[j]
		}
		clear(a[n:])
		if err := p.inner.mixedParallel(ctx, b, scratch, a); err != nil {
			return work, err
		}
		for i, f := range p.filter {
			a[i] = cmplx.Conj(scratch[i] * f)
		}
		if err := p.inner.mixedParallel(ctx, b, scratch, a); err != nil {
			return work, err
		}
		for k := range x {
			x[k] = cmplx.Conj(scratch[k]) * p.chirp[k]
		}
	}
	return work, nil
}

// mixed sets out, of length n, to the transform of the n elements of in
// stride apart, whose length factors into rs.
func (p *fftPlan) mixed(out, in []complex128, stride int, rs []int, n int) {
	r := rs[0]
	m := n / r
	if m == 1 {
		for q := range r {
			out[q] = in[q*stride]
		}
	} else {
		for q := range r {
			p.mixed(out[q*m:(q+1)*m], in[q*stride:], stride*r, rs[1:], m)
		}
	}
	p.butterflies(out, r, m, p.n/n, 0, m)
}

// mixedParallel is mixed of the plan's whole length with the workers
// sharing the transforms of the subsequences of the first radix and the
// butterflies that combine them.
func (p *fftPlan) mixedParallel(ctx context.Context, b *Backend, out, in []complex128) error {
	r := p.radices[0]
	m := p.n / r
	if m == 1 {
		p.mixed(out, in, 1, p.radices, p.n)
		return nil
	}
	err := b.parallelFor(ctx, r, 1, func(lo, hi int) error {
		for q := lo; q < hi; q++ {
			p.mixed(out[q*m:(q+1)*m], in[q:], r, p.radices[1:], m)
		}
		return nil
	})
	if err != nil {
		return err
	}
	return b.parallelFor(ctx, m, 1<<12, func(lo, hi int) error {
		p.butterflies(out, r, m, 1, lo, hi)
		return nil
	})
}

// butterflies combines the transforms of the r subsequences of a
// transform of length r·m, out[q*m+k] for subsequence q, into
// out[k+o*m], for k in [k0, k1). The transform's twiddles are those of
// the plan step apart.
func (p *fftPlan) butterflies(out []complex128, r, m, step, k0, k1 int) {
	tw := p.twiddle
	switch r {
	case 2:
		lo, hi := out[:m], out[m:2*m]
		for k := k0; k < k1; k++ {
			t := hi[k] * tw[k*step]
			hi[k] = lo[k] - t
			lo[k] += t
		}
	case 3:
		const s3 = 0.86602540378443864676 // sin(2π/3)
		a, b, c := out[:m], out[m:2*m], out[2*m:3*m]
		for k := k0; k < k1; k++ {
			s1, s2 := b[k]*tw[k*step], c[k]*tw[2*k*step]
			sum, diff := s1+s2, s1-s2
			mid := a[k] - sum*0.5
			rot := complex(imag(diff)*s3, -real(diff)*s3) // -i·sin(2π/3)·diff
			a[k] += sum
			b[k] = mid + rot
			c[k] = mid - rot
		}
	case 4:
		a, b, c, d := out[:m], out[m:2*m], out[2*m:3*m], out[3*m:4*m]
		for k := k0; k < k1; k++ {
			s0, s1, s2 := b[k]*tw[k*step], c[k]*tw[2*k*step], d[k]*tw[3*k*step]
			e, f := a[k]+s1, a[k]-s1
			g, h := s0+s2, s0-s2
			ih := complex(imag(h), -real(h)) // -i·h
			a[k], c[k] = e+g, e-g
			b[k], d[k] = f+ih, f-ih
		}
	default:
		var buf [fftLargestRadix]complex128
		t := buf[:r]
		pstep := p.n / r
		for k := k0; k < k1; k++ {
			t[0] = out[k]
			for q := 1; q < r; q++ {
				t[q] = out[q*m+k] * tw[q*k*step]
			}
			for o := range r {
				s := t[0]
				for q, e := 1, o; q < r; q++ {
					s += t[q] * tw[e*pstep]
					if e += o; e >= r {
						e -= r
					}
				}
				out[o*m+k] = s
			}
		}
	}
}

// runFFT runs a Fourier transform of the complex Float32 in into out.
func runFFT(ctx context.Context, b *Backend, k *backend.Kernel, outBuf, inBuf backend.Buffer) error {
	out, err := data[float32](outBuf)
	if err != nil {
		return err
	}
	in, err := data[float32](inBuf)
	if err != nil {
		return err
	}
	axis, v, r := k.Attrs.Axis, k.In[0], len(k.Shape)
	if k.DType != backend.Float32 || r < 2 || k.Shape[r-1] != 2 || axis < 0 || axis >= r-1 || !slices.Equal(v.Shape, k.Shape) {
		return fmt.Errorf("fft of %v along axis %d to %v: %w", v.Shape, axis, k.Shape, backend.ErrInternal)
	}
	if len(out) == 0 {
		return nil
	}
	n := k.Shape[axis]
	plan := planFFT(n)
	outStrides := backend.ContiguousStrides(k.Shape)
	drop := func(s []int) []int { return slices.Delete(slices.Clone(s[:r-1]), axis, axis+1) }
	rest, views := backend.Coalesce(drop(k.Shape), []backend.View{
		{Strides: drop(v.Strides), Offset: v.Offset},
		{Strides: drop(outStrides)},
	})
	lines, _ := backend.NumElements(rest)
	inStep, inImag, outStep := v.Strides[axis], v.Strides[r-1], outStrides[axis]
	conj := k.Attrs.Inverse
	load := func(line []complex128, off int) {
		for j := range line {
			re, im := float64(in[off]), float64(in[off+inImag])
			if conj {
				im = -im
			}
			line[j] = complex(re, im)
			off += inStep
		}
	}
	store := func(line []complex128, off int) {
		for _, z := range line {
			im := imag(z)
			if conj {
				im = -im
			}
			out[off], out[off+1] = float32(real(z)), float32(im)
			off += outStep
		}
	}
	if lines < b.workers && n >= fftSplit {
		// Too few lines to share: the workers share each.
		c := newCursor(rest, views, 0)
		line := make([]complex128, n)
		var work []complex128
		for range lines {
			load(line, c.offs[0])
			if work, err = plan.transformParallel(ctx, b, line, work); err != nil {
				return err
			}
			store(line, c.offs[1])
			c.next()
		}
		return nil
	}
	return b.parallelFor(ctx, lines, max(1, elementGrain/(8*n)), func(lo, hi int) error {
		c := newCursor(rest, views, lo)
		line := make([]complex128, n)
		var work []complex128
		for range hi - lo {
			load(line, c.offs[0])
			work = plan.transform(line, work)
			store(line, c.offs[1])
			c.next()
		}
		return nil
	})
}

// fftSplit is the shortest line whose transform the workers share when
// there are too few lines to go round.
const fftSplit = 1 << 15
