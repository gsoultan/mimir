package tensor

import (
	"fmt"
	"math"
	"math/bits"
	"math/cmplx"
	"slices"

	"github.com/gsoultan/mimir/internal/graph"
)

// Complex tensors are Float32 tensors whose last dimension, of length 2,
// holds the real and imaginary parts of each element, as
// torch.view_as_real shows a complex tensor: a complex tensor of shape
// [..., n] is a real one of shape [..., n, 2]. The axis of a Fourier
// transform counts the complex dimensions, those before the last, so
// that -1 is the last of them. Transforms of any length run in
// O(n log n), and are unnormalized forward and scaled by 1/n inverse,
// like torch.fft's default norm="backward". Half inputs are computed in
// Float32. Their gradients are those of torch.fft for the real views.

// FFT returns the discrete Fourier transform of the complex t along
// axis, X[k] = Σ x[j]·e^(-2πi·jk/n), like torch.fft.fft.
func (t *Tensor) FFT(axis int) *Tensor {
	x, a, err := t.complexAxis(axis)
	if err != nil {
		return fail(err)
	}
	return x.dft(a, false)
}

// IFFT returns the inverse discrete Fourier transform of the complex t
// along axis, x[j] = Σ X[k]·e^(2πi·jk/n) / n, like torch.fft.ifft.
func (t *Tensor) IFFT(axis int) *Tensor {
	x, a, err := t.complexAxis(axis)
	if err != nil {
		return fail(err)
	}
	return x.dft(a, true).Mul(Scalar(float32(1 / float64(x.node.Shape[a]))))
}

// RFFT returns the n/2+1 non-negative frequencies of the discrete Fourier
// transform of the real t along axis, n its length there, as a complex
// tensor, like torch.fft.rfft: the others are their conjugates.
func (t *Tensor) RFFT(axis int) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	r := len(t.node.Shape)
	if !t.node.DType.IsFloat() {
		return fail(fmt.Errorf("tensor: rfft of %s: %w", t.node.DType, ErrDType))
	}
	a, err := normAxis(axis, r)
	if err != nil {
		return fail(err)
	}
	x := t.Cast(Float32).Unsqueeze(-1).Pad(append(make([][2]int, r), [2]int{0, 1}), 0)
	n := t.node.Shape[a]
	return x.dft(a, false).shrink(axisRange(x.node.Shape, a, 0, n/2+1))
}

// IRFFT returns the real signal of length n whose RFFT along axis is the
// complex t, like torch.fft.irfft(t, n, dim=axis): t holds frequencies 0
// to n/2, cut or padded with zeros to n/2+1, and the imaginary parts of
// frequency 0 and, for an even n, n/2 are ignored. n ≤ 0 means 2(m-1)
// for m frequencies along axis.
func (t *Tensor) IRFFT(axis, n int) *Tensor {
	x, a, err := t.complexAxis(axis)
	if err != nil {
		return fail(err)
	}
	m := x.node.Shape[a]
	if n <= 0 {
		n = 2 * (m - 1)
	}
	if n <= 0 {
		return fail(fmt.Errorf("tensor: irfft of %d frequencies: %w", m, ErrShape))
	}
	// The whole spectrum: frequencies 0 to n/2, then the conjugates of
	// those from n-n/2-1 down to 1.
	h := n/2 + 1
	if m >= h {
		x = x.shrink(axisRange(x.node.Shape, a, 0, h))
	} else {
		pads := make([][2]int, len(x.node.Shape))
		pads[a] = [2]int{0, h - m}
		x = x.Pad(pads, 0)
	}
	full := x
	if rest := n - h; rest > 0 {
		mirror := x.shrink(axisRange(x.node.Shape, a, 1, rest+1)).flip([]int{a})
		last := len(x.node.Shape) - 1
		re := mirror.shrink(axisRange(mirror.node.Shape, last, 0, 1))
		im := mirror.shrink(axisRange(mirror.node.Shape, last, 1, 2))
		full = Cat(a, x, Cat(last, re, im.Neg()))
	}
	y := full.dft(a, true).Mul(Scalar(float32(1 / float64(n))))
	re := axisRange(y.node.Shape, len(y.node.Shape)-1, 0, 1)
	return y.shrink(re).Squeeze(-1)
}

// complexAxis returns t as a Float32 complex tensor and axis, which
// counts its complex dimensions, as one of its own.
func (t *Tensor) complexAxis(axis int) (*Tensor, int, error) {
	if err := firstErr(t); err != nil {
		return nil, 0, err
	}
	r := len(t.node.Shape)
	if r < 2 || t.node.Shape[r-1] != 2 {
		return nil, 0, fmt.Errorf("tensor: Fourier transform of %v, not complex: %w", t.node.Shape, ErrShape)
	}
	if !t.node.DType.IsFloat() {
		return nil, 0, fmt.Errorf("tensor: Fourier transform of %s: %w", t.node.DType, ErrDType)
	}
	a, err := normAxis(axis, r-1)
	if err != nil {
		return nil, 0, err
	}
	return t.Cast(Float32), a, nil
}

// dft is the unnormalized discrete Fourier transform of the complex
// Float32 t along the normalized axis, inverse with the sign of the
// exponent positive. Its gradient is the transform the other way: the
// adjoint of the transform matrix is the other's.
func (t *Tensor) dft(axis int, inverse bool) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	if n := t.node.Shape[axis]; n > gpuDirectDFT && n&(n-1) != 0 && t.Device().backend() != cpuBackend {
		return t.bluestein(axis, inverse)
	}
	var out *Tensor
	if l := batchLevel(t); l != 0 {
		out = batchFFT(l, t, axis, inverse)
	} else {
		out = wrap(graph.FFT(t.node, axis, inverse))
	}
	return record(out, rule{
		vjp: func(g *Tensor, _ []*Tensor, _ *Tensor) []*Tensor { return grads(g.Cast(Float32).dft(axis, !inverse)) },
		// The transform is linear: the tangent's transform is the
		// result's tangent.
		jvp: func(t, _ []*Tensor, _ *Tensor) *Tensor { return t[0].Cast(Float32).dft(axis, inverse) },
	}, t)
}

// gpuDirectDFT is the longest transform of a length that is not a power
// of two that a GPU computes by its definition, in O(n²); longer ones
// are built from power-of-two transforms.
const gpuDirectDFT = 64

// bluestein is dft for any length n from transforms of a power of two m
// ≥ 2n-1, Bluestein's algorithm: with w[j] = e^(∓πi·j²/n), X[k] =
// w[k]·Σ (x[j]·w[j])·conj(w[k-j]), a circular convolution of length m,
// which multiplies the transforms. The transform of the convolution's
// filter is computed on the host.
func (t *Tensor) bluestein(axis int, inverse bool) *Tensor {
	n := t.node.Shape[axis]
	m := 1 << bits.Len(uint(2*n-2))
	sign := -1.0
	if inverse {
		sign = 1
	}
	w := make([]complex128, n)
	for j := range w {
		sq := int64(j) * int64(j) % int64(2*n)
		w[j] = cmplx.Rect(1, sign*math.Pi*float64(sq)/float64(n))
	}
	filter := make([]complex128, m)
	filter[0] = cmplx.Conj(w[0])
	for j := 1; j < n; j++ {
		filter[j], filter[m-j] = cmplx.Conj(w[j]), cmplx.Conj(w[j])
	}
	hostFFT(filter)
	d := t.Device()
	// constant returns the complex values c as a tensor that broadcasts
	// along axis, on t's device.
	constant := func(c []complex128) *Tensor {
		v := make([]float32, 0, 2*len(c))
		for _, x := range c {
			v = append(v, float32(real(x)), float32(imag(x)))
		}
		shape := slices.Repeat([]int{1}, len(t.node.Shape))
		shape[axis], shape[len(shape)-1] = len(c), 2
		return FromSlice(v, shape...).To(d)
	}
	chirp := constant(w)
	pads := make([][2]int, len(t.node.Shape))
	pads[axis] = [2]int{0, m - n}
	a := complexMul(t, chirp).Pad(pads, 0).dft(axis, false)
	conv := complexMul(a, constant(filter)).dft(axis, true).Mul(Scalar(float32(1 / float64(m))))
	return complexMul(conv.shrink(axisRange(conv.node.Shape, axis, 0, n)), chirp)
}

// complexMul returns the products of the complex tensors x and y, which
// broadcast.
func complexMul(x, y *Tensor) *Tensor {
	last := len(x.node.Shape) - 1
	part := func(t *Tensor, i int) *Tensor { return t.shrink(axisRange(t.node.Shape, last, i, i+1)) }
	xr, xi, yr, yi := part(x, 0), part(x, 1), part(y, 0), part(y, 1)
	return Cat(last, xr.Mul(yr).Sub(xi.Mul(yi)), xr.Mul(yi).Add(xi.Mul(yr)))
}

// hostFFT replaces x, whose length is a power of two, with its discrete
// Fourier transform, by radix-2 butterflies.
func hostFFT(x []complex128) {
	n := len(x)
	for i, j := 1, 0; i < n; i++ {
		bit := n >> 1
		for ; j&bit != 0; bit >>= 1 {
			j ^= bit
		}
		j ^= bit
		if i < j {
			x[i], x[j] = x[j], x[i]
		}
	}
	for size := 2; size <= n; size <<= 1 {
		for k := range size / 2 {
			w := cmplx.Rect(1, -2*math.Pi*float64(k)/float64(size))
			for s := 0; s < n; s += size {
				a, b := x[s+k], x[s+k+size/2]*w
				x[s+k], x[s+k+size/2] = a+b, a-b
			}
		}
	}
}

// axisRange returns the ranges of a shrink of shape to [lo, hi) along
// axis.
func axisRange(shape []int, axis, lo, hi int) [][2]int {
	r := make([][2]int, len(shape))
	for i, d := range shape {
		r[i] = [2]int{0, d}
	}
	r[axis] = [2]int{lo, hi}
	return r
}
