package tensor_test

import (
	"context"
	"math"
	"testing"

	"github.com/gsoultan/mimir/tensor"
)

func convData(t *testing.T, x *tensor.Tensor) []float32 {
	t.Helper()
	v, err := x.Data[float32](context.Background())
	if err != nil {
		t.Fatal(err)
	}
	return v
}

func convNear(t *testing.T, what string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", what, len(got), len(want))
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > tol*(1+math.Abs(float64(want[i]))) {
			t.Fatalf("%s: [%d] = %v, want %v", what, i, got[i], want[i])
		}
	}
}

// naiveConv1d is the convolution of x [N, C, L] by w [O, C/g, K] as
// torch defines it.
func naiveConv1d(x, w []float32, n, c, l, o, k, stride, pad, dil, g int) ([]float32, int) {
	ol := (l+2*pad-dil*(k-1)-1)/stride + 1
	out := make([]float32, n*o*ol)
	cg, og := c/g, o/g
	for b := range n {
		for oc := range o {
			grp := oc / og
			for p := range ol {
				var s float32
				for ic := range cg {
					for kk := range k {
						at := p*stride - pad + kk*dil
						if at < 0 || at >= l {
							continue
						}
						s += x[(b*c+grp*cg+ic)*l+at] * w[(oc*cg+ic)*k+kk]
					}
				}
				out[(b*o+oc)*ol+p] = s
			}
		}
	}
	return out, ol
}

func TestConv1d(t *testing.T) {
	for _, tc := range []struct{ stride, pad, dil, groups int }{{1, 0, 1, 1}, {2, 1, 1, 1}, {1, 2, 2, 2}, {3, 1, 2, 4}} {
		const n, c, l, o, k = 2, 4, 11, 8, 3
		x, w := tensor.Randn(1, n, c, l), tensor.Randn(2, o, c/tc.groups, k)
		b := tensor.Randn(3, o)
		got := x.Conv1d(w, b, tensor.ConvOptions{Stride: tc.stride, Padding: tc.pad, Dilation: tc.dil, Groups: tc.groups})
		want, ol := naiveConv1d(convData(t, x), convData(t, w), n, c, l, o, k, tc.stride, tc.pad, tc.dil, tc.groups)
		bias := convData(t, b)
		for i := range want {
			want[i] += bias[i/ol%o]
		}
		if s := got.Shape(); len(s) != 3 || s[2] != ol {
			t.Fatalf("%+v: shape %v, want length %d", tc, s, ol)
		}
		convNear(t, "conv1d", convData(t, got), want, 1e-5)
	}
}

// A 3-D convolution whose kernel is 1 deep is a 2-D one on each slice.
func TestConv3dMatchesConv2dPerSlice(t *testing.T) {
	const n, c, d, h, w, o = 2, 3, 4, 7, 6, 5
	x := tensor.Randn(1, n, c, d, h, w)
	k := tensor.Randn(2, o, c, 1, 3, 3)
	got := x.Conv3d(k, nil, tensor.ConvOptions{Padding: 0})
	// With no padding the depth stays and the slices shrink by 2.
	if s := got.Shape(); s[2] != d || s[3] != h-2 || s[4] != w-2 {
		t.Fatalf("shape %v", s)
	}
	k2 := k.Reshape(o, c, 3, 3)
	for z := range d {
		slice := x.Shrink([][2]int{{0, n}, {0, c}, {z, z + 1}, {0, h}, {0, w}}).Reshape(n, c, h, w)
		want := slice.Conv2d(k2, nil, tensor.ConvOptions{})
		have := got.Shrink([][2]int{{0, n}, {0, o}, {z, z + 1}, {0, h - 2}, {0, w - 2}}).Reshape(n, o, h-2, w-2)
		convNear(t, "conv3d slice", convData(t, have), convData(t, want), 1e-5)
	}
}

func dot(t *testing.T, a, b *tensor.Tensor) float64 {
	t.Helper()
	av, bv := convData(t, a), convData(t, b)
	var s float64
	for i := range av {
		s += float64(av[i]) * float64(bv[i])
	}
	return s
}

// A transposed convolution is the adjoint of the convolution with the
// same kernel: <conv(x), r> = <x, convT(r)>. A conv kernel [O, C/g, K...]
// read as a transposed one is [in, out/g, K...] with in O and out/g C/g,
// so the same tensor serves both.
func TestConvTransposeIsTheAdjoint(t *testing.T) {
	for _, tc := range []struct{ stride, pad, dil, groups int }{{1, 0, 1, 1}, {2, 1, 1, 1}, {2, 0, 2, 2}, {3, 2, 1, 1}} {
		opts := tensor.ConvOptions{Stride: tc.stride, Padding: tc.pad, Dilation: tc.dil, Groups: tc.groups}
		// The rows a strided conv drops come back as output padding.
		outPad := func(size, k int) int { return (size + 2*tc.pad - tc.dil*(k-1) - 1) % tc.stride }
		const n, c, o, h, w = 2, 4, 6, 10, 9
		x := tensor.Randn(1, n, c, h, w)
		k := tensor.Randn(2, o, c/tc.groups, 3, 3)
		y := x.Conv2d(k, nil, opts)
		r := tensor.Randn(3, y.Shape()...)
		if outPad(h, 3) != outPad(w, 3) {
			// One output padding serves both sides only when they agree.
			w := h
			x = tensor.Randn(1, n, c, h, w)
			y = x.Conv2d(k, nil, opts)
			r = tensor.Randn(3, y.Shape()...)
		}
		back := r.ConvTranspose2d(k, nil, opts, outPad(h, 3))
		if bs, xs := back.Shape(), x.Shape(); bs[2] != xs[2] || bs[3] != xs[3] {
			t.Fatalf("%+v: transposed shape %v, want %v", tc, bs, xs)
		}
		if lhs, rhs := dot(t, y, r), dot(t, x, back); math.Abs(lhs-rhs) > 1e-3*(1+math.Abs(lhs)) {
			t.Errorf("%+v: <conv x, r> = %v, <x, convT r> = %v", tc, lhs, rhs)
		}

		const l = 13
		x1 := tensor.Randn(4, n, c, l)
		k1 := tensor.Randn(5, o, c/tc.groups, 3)
		y1 := x1.Conv1d(k1, nil, opts)
		r1 := tensor.Randn(6, y1.Shape()...)
		b1 := r1.ConvTranspose1d(k1, nil, opts, outPad(l, 3))
		if b1.Shape()[2] != l {
			t.Fatalf("%+v: transposed 1-D shape %v", tc, b1.Shape())
		}
		if lhs, rhs := dot(t, y1, r1), dot(t, x1, b1); math.Abs(lhs-rhs) > 1e-3*(1+math.Abs(lhs)) {
			t.Errorf("1-D %+v: <conv x, r> = %v, <x, convT r> = %v", tc, lhs, rhs)
		}
	}
}

func TestConvTransposeShape(t *testing.T) {
	// The usual 2× upsampling: kernel 4, stride 2, padding 1.
	x := tensor.Randn(1, 1, 3, 5, 7)
	k := tensor.Randn(2, 3, 2, 4, 4)
	if s := x.ConvTranspose2d(k, tensor.Randn(3, 2), tensor.ConvOptions{Stride: 2, Padding: 1}, 0).Shape(); s[1] != 2 || s[2] != 10 || s[3] != 14 {
		t.Errorf("shape %v, want [1 2 10 14]", s)
	}
	if err := x.ConvTranspose2d(k, nil, tensor.ConvOptions{Stride: 2}, 2).Err(); err == nil {
		t.Error("output padding 2 at stride 2 is accepted")
	}
}
