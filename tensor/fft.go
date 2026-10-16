package tensor

import (
	"fmt"

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
	out := wrap(graph.FFT(t.node, axis, inverse))
	return track(out, func(g *Tensor) []*Tensor { return grads(g.Cast(Float32).dft(axis, !inverse)) }, t)
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
