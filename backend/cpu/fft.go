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
// A line whose length is a power of two is transformed by iterative
// radix-2 Cooley-Tukey; a short line of another length by the definition;
// and a long one by Bluestein's algorithm, which writes the transform as
// a convolution with a chirp and computes that with transforms of a
// power of two at least twice as long. The inverse direction conjugates
// before and after the forward transform.

// fftDirect is the longest line of a length other than a power of two
// that is transformed by the definition.
const fftDirect = 16

// fftPlan holds what transforms of length n need: the twiddle factors
// e^(-2πik/n), and for Bluestein's algorithm the chirp e^(-πij²/n), the
// transform of its filter, divided by m, and the plan of length m.
type fftPlan struct {
	n       int
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

func newFFTPlan(n int) *fftPlan {
	p := &fftPlan{n: n}
	pow2 := n&(n-1) == 0
	switch {
	case pow2 || n <= fftDirect:
		size := n
		if pow2 {
			size = n / 2
		}
		p.twiddle = make([]complex128, size)
		for k := range p.twiddle {
			p.twiddle[k] = rootOfUnity(k, n)
		}
	default:
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
		p.inner.transform(p.filter, nil)
		scale := complex(1/float64(m), 0)
		for i := range p.filter {
			p.filter[i] *= scale
		}
	}
	return p
}

// rootOfUnity returns e^(-2πik/n).
func rootOfUnity(k, n int) complex128 {
	s, c := math.Sincos(2 * math.Pi * float64(k%n) / float64(n))
	return complex(c, -s)
}

// transform replaces x, of the plan's length, with its forward transform,
// using work, which grows as needed, as scratch. It returns work.
func (p *fftPlan) transform(x, work []complex128) []complex128 {
	n := p.n
	switch {
	case n <= 1:
	case n&(n-1) == 0:
		p.radix2(x)
	case p.chirp == nil:
		work = grow(work, n)
		for k := range n {
			var s complex128
			for j, v := range x {
				s += v * p.twiddle[j*k%n]
			}
			work[k] = s
		}
		copy(x, work)
	default:
		m := p.inner.n
		work = grow(work, m)
		for j, v := range x {
			work[j] = v * p.chirp[j]
		}
		clear(work[n:])
		p.inner.radix2(work)
		for i, f := range p.filter {
			work[i] = cmplx.Conj(work[i] * f)
		}
		p.inner.radix2(work) // the inverse, conjugated
		for k := range x {
			x[k] = cmplx.Conj(work[k]) * p.chirp[k]
		}
	}
	return work
}

// radix2 transforms x, whose length is the plan's power of two, in place.
func (p *fftPlan) radix2(x []complex128) {
	n := len(x)
	shift := 64 - bits.Len(uint(n-1))
	for i := range x {
		if j := int(bits.Reverse64(uint64(i)) >> shift); i < j {
			x[i], x[j] = x[j], x[i]
		}
	}
	for half := 1; half < n; half *= 2 {
		step := n / (2 * half)
		for start := 0; start < n; start += 2 * half {
			lo := x[start : start+half]
			hi := x[start+half : start+2*half]
			for j := range lo {
				t := hi[j] * p.twiddle[j*step]
				hi[j] = lo[j] - t
				lo[j] += t
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
	return b.parallelFor(ctx, lines, max(1, elementGrain/(8*n)), func(lo, hi int) error {
		c := newCursor(rest, views, lo)
		line := make([]complex128, n)
		var work []complex128
		for range hi - lo {
			off := c.offs[0]
			for j := range line {
				re, im := float64(in[off]), float64(in[off+inImag])
				if conj {
					im = -im
				}
				line[j] = complex(re, im)
				off += inStep
			}
			work = plan.transform(line, work)
			off = c.offs[1]
			for _, z := range line {
				im := imag(z)
				if conj {
					im = -im
				}
				out[off], out[off+1] = float32(real(z)), float32(im)
				off += outStep
			}
			c.next()
		}
		return nil
	})
}
