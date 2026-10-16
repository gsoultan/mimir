package onnx

import (
	"fmt"
	"math"
	"slices"

	"github.com/gsoultan/mimir/tensor"
)

func softmax(log bool) opFunc {
	return func(c *call) ([]*value, error) {
		if err := c.need(1); err != nil {
			return nil, err
		}
		x := c.in[0]
		t := c.t(0)
		f := (*tensor.Tensor).Softmax
		if log {
			f = (*tensor.Tensor).LogSoftmax
		}
		if c.m.opset >= 13 {
			axis, err := normAxis(c.int("axis", -1), len(x.shape))
			if err != nil {
				return nil, err
			}
			return one1(out(x.typ, f(t, axis))), nil
		}
		// Before opset 13 the input is flattened to a matrix at axis.
		axis, err := normAxis(c.int("axis", 1), len(x.shape))
		if err != nil {
			return nil, err
		}
		inner := 1
		for _, d := range x.shape[axis:] {
			inner *= d
		}
		return one1(out(x.typ, f(t.Reshape(-1, inner), 1).Reshape(x.shape...))), nil
	}
}

// window is the geometry of a convolution or a pooling over the spatial
// axes of an [N, C, spatial...] tensor.
type window struct {
	kernel, stride, dilation []int
	pads                     [][2]int // before and after each spatial axis
}

// effective returns the span of the kernel along spatial axis i.
func (w window) effective(i int) int { return (w.kernel[i]-1)*w.dilation[i] + 1 }

// window reads the window attributes of a node over spatial dimensions
// in, for a kernel of the given shape. With autoPad, auto_pad sets the
// pads of a forward convolution or pooling.
func (c *call) window(in, kernel []int, autoPad bool) (window, error) {
	n := len(in)
	w := window{kernel: kernel, stride: slices.Repeat([]int{1}, n), dilation: slices.Repeat([]int{1}, n), pads: make([][2]int, n)}
	list := func(name string, dst []int, lo int64) error {
		v, ok := c.ints(name)
		if !ok {
			return nil
		}
		if len(v) != len(dst) {
			return fmt.Errorf("onnx: %s %v for %d spatial axes: %w", name, v, n, tensor.ErrShape)
		}
		for i, x := range v {
			if x < lo || x > math.MaxInt32 {
				return fmt.Errorf("onnx: %s %v: %w", name, v, tensor.ErrShape)
			}
			dst[i] = int(x)
		}
		return nil
	}
	for _, k := range kernel {
		if k < 1 {
			return w, fmt.Errorf("onnx: kernel %v: %w", kernel, tensor.ErrShape)
		}
	}
	if err := list("strides", w.stride, 1); err != nil {
		return w, err
	}
	if err := list("dilations", w.dilation, 1); err != nil {
		return w, err
	}
	pads := make([]int, 2*n)
	if err := list("pads", pads, 0); err != nil {
		return w, err
	}
	for i := range n {
		w.pads[i] = [2]int{pads[i], pads[i+n]}
	}
	auto := "NOTSET"
	if autoPad {
		auto = c.str("auto_pad", auto)
	}
	switch auto {
	case "NOTSET":
	case "VALID":
		clear(w.pads)
	case "SAME_UPPER", "SAME_LOWER":
		for i, d := range in {
			o := (d + w.stride[i] - 1) / w.stride[i]
			total := max(0, (o-1)*w.stride[i]+w.effective(i)-d)
			lo := total / 2
			if auto == "SAME_LOWER" {
				lo = total - total/2
			}
			w.pads[i] = [2]int{lo, total - lo}
		}
	default:
		return w, c.unsupported("auto_pad %q", auto)
	}
	return w, nil
}

// square reports whether w is the same along both axes of a 2-D window,
// with the padding p on every side, as tensor.ConvOptions describes.
func (w window) square() (p int, ok bool) {
	if len(w.kernel) != 2 || w.kernel[0] != w.kernel[1] || w.stride[0] != w.stride[1] || w.dilation[0] != w.dilation[1] {
		return 0, false
	}
	p = w.pads[0][0]
	for _, q := range w.pads {
		if q[0] != p || q[1] != p {
			return p, false
		}
	}
	return p, true
}

// padSpatial pads the spatial axes of t by pads, filled with fill.
func padSpatial(t *tensor.Tensor, pads [][2]int, fill float64) *tensor.Tensor {
	if !slices.ContainsFunc(pads, func(p [2]int) bool { return p != [2]int{} }) {
		return t
	}
	return t.Pad(append(make([][2]int, t.Rank()-len(pads)), pads...), fill)
}

// slide returns the windows of w over t, [N, C, spatial...] and padded
// already, as [N, C, out..., kernel...].
func slide(t *tensor.Tensor, w window) *tensor.Tensor {
	n := len(w.kernel)
	for i := range n {
		t = t.Unfold(2+i, w.effective(i), w.stride[i])
	}
	for i, d := range w.dilation {
		if d == 1 {
			continue
		}
		// Keep every d-th element of the window's span: padded to k·d
		// elements, the span is k rows of d, whose first column is kept.
		a, k := 2+n+i, w.kernel[i]
		shape := t.Shape()
		p := make([][2]int, len(shape))
		p[a][1] = k*d - shape[a]
		t = t.Pad(p, 0)
		split := slices.Concat(shape[:a], []int{k, d}, shape[a+1:])
		r := whole(split)
		r[a+1] = [2]int{0, 1}
		t = t.Reshape(split...).Shrink(r).Reshape(slices.Concat(shape[:a], []int{k}, shape[a+1:])...)
	}
	return t
}

// convolve returns the convolution of x, [N, C, spatial...], with w,
// [M, C/groups, kernel...], over the window wn.
func convolve(x, w *tensor.Tensor, wn window, groups int) *tensor.Tensor {
	if p, ok := wn.square(); ok {
		return x.Conv2d(w, nil, tensor.ConvOptions{Stride: wn.stride[0], Padding: p, Dilation: wn.dilation[0], Groups: groups})
	}
	x = padSpatial(x, wn.pads, 0)
	if len(wn.kernel) == 2 && wn.kernel[0] == wn.kernel[1] && wn.stride[0] == wn.stride[1] && wn.dilation[0] == wn.dilation[1] {
		return x.Conv2d(w, nil, tensor.ConvOptions{Stride: wn.stride[0], Dilation: wn.dilation[0], Groups: groups})
	}
	// Any other window multiplies its unfolded columns by the weights.
	xs, ws := x.Shape(), w.Shape()
	if len(xs) < 3 || len(ws) != len(xs) || groups < 1 || ws[0]%groups != 0 || ws[1]*groups != xs[1] {
		return tensor.Failed(fmt.Errorf("onnx: convolution of %v with kernel %v in %d groups: %w", xs, ws, groups, tensor.ErrShape))
	}
	ns := len(wn.kernel)
	win := slide(x, wn) // [N, C, out..., kernel...]
	if win.Err() != nil {
		return win
	}
	perm := []int{0, 1}
	for i := range ns {
		perm = append(perm, 2+ns+i)
	}
	for i := range ns {
		perm = append(perm, 2+i)
	}
	outShape := win.Shape()[2 : 2+ns]
	k, o := 1, 1
	for i := range ns {
		k *= wn.kernel[i]
		o *= outShape[i]
	}
	cols := win.Permute(perm...).Reshape(xs[0], groups, xs[1]/groups*k, o)
	y := w.Reshape(groups, ws[0]/groups, ws[1]*k).MatMul(cols)
	return y.Reshape(slices.Concat([]int{xs[0], ws[0]}, outShape)...)
}

// kernelShape returns the kernel_shape attribute, or the given default.
func (c *call) kernelShape(def []int) ([]int, error) {
	ks, ok := c.ints("kernel_shape")
	if !ok {
		if def == nil {
			return nil, fmt.Errorf("%w: %s without kernel_shape", ErrFormat, c.n.op)
		}
		return def, nil
	}
	if def != nil && len(ks) != len(def) {
		return nil, fmt.Errorf("onnx: kernel_shape %v for a kernel %v: %w", ks, def, tensor.ErrShape)
	}
	k := make([]int, len(ks))
	for i, d := range ks {
		if d < 1 || d > math.MaxInt32 || def != nil && int(d) != def[i] {
			return nil, fmt.Errorf("onnx: kernel_shape %v: %w", ks, tensor.ErrShape)
		}
		k[i] = int(d)
	}
	return k, nil
}

// addChannelBias adds bias, [C], to y, [N, C, spatial...].
func addChannelBias(y, bias *tensor.Tensor) *tensor.Tensor {
	shape := slices.Repeat([]int{1}, y.Rank()-1)
	shape[0] = -1
	return y.Add(bias.Reshape(shape...))
}

func conv(c *call) ([]*value, error) {
	if err := c.need(2); err != nil {
		return nil, err
	}
	x, w := c.t(0), c.t(1)
	xs, ws := x.Shape(), w.Shape()
	if len(xs) < 3 || len(ws) != len(xs) {
		return nil, fmt.Errorf("onnx: convolution of %v with kernel %v: %w", xs, ws, tensor.ErrShape)
	}
	kernel, err := c.kernelShape(ws[2:])
	if err != nil {
		return nil, err
	}
	wn, err := c.window(xs[2:], kernel, true)
	if err != nil {
		return nil, err
	}
	y := convolve(x, w, wn, int(min(max(c.int("group", 1), 0), math.MaxInt32)))
	if c.arg(2) != nil {
		y = addChannelBias(y, c.t(2))
	}
	return one1(out(c.in[0].typ, y)), nil
}

func convTranspose(c *call) ([]*value, error) {
	if err := c.need(2); err != nil {
		return nil, err
	}
	x, w := c.t(0), c.t(1)
	xs, ws := x.Shape(), w.Shape()
	g := int(min(max(c.int("group", 1), 1), math.MaxInt32))
	if len(xs) < 3 || len(ws) != len(xs) || ws[0] != xs[1] || ws[0]%g != 0 {
		return nil, fmt.Errorf("onnx: transposed convolution of %v with kernel %v in %d groups: %w", xs, ws, g, tensor.ErrShape)
	}
	ns := len(xs) - 2
	kernel, err := c.kernelShape(ws[2:])
	if err != nil {
		return nil, err
	}
	auto := c.str("auto_pad", "NOTSET")
	if auto != "NOTSET" && auto != "VALID" && auto != "SAME_UPPER" && auto != "SAME_LOWER" {
		return nil, c.unsupported("auto_pad %q", auto)
	}
	// The pads are those of the transposed convolution, which auto_pad
	// sets below.
	wn, err := c.window(xs[2:], kernel, false)
	if err != nil {
		return nil, err
	}
	outPad := make([]int, ns)
	if v, ok := c.ints("output_padding"); ok {
		if len(v) != ns {
			return nil, fmt.Errorf("onnx: output_padding %v: %w", v, tensor.ErrShape)
		}
		for i, p := range v {
			if p < 0 || p >= int64(max(wn.stride[i], wn.dilation[i])) {
				return nil, fmt.Errorf("onnx: output_padding %v: %w", v, tensor.ErrShape)
			}
			outPad[i] = int(p)
		}
	}
	outShape, given := c.ints("output_shape")
	if given && len(outShape) != ns {
		outShape = outShape[len(outShape)-ns:]
		if len(outShape) != ns {
			return nil, fmt.Errorf("onnx: output_shape %v: %w", outShape, tensor.ErrShape)
		}
	}
	for i := range ns {
		full := wn.stride[i]*(xs[2+i]-1) + outPad[i] + wn.effective(i)
		switch {
		case given || auto == "SAME_UPPER" || auto == "SAME_LOWER":
			want := int64(xs[2+i]) * int64(wn.stride[i])
			if given {
				want = outShape[i]
			}
			total := int64(full) - want
			if total < 0 && !given {
				total = 0
			}
			if total < 0 || total > int64(full) {
				return nil, fmt.Errorf("onnx: output_shape %v: %w", outShape, tensor.ErrShape)
			}
			lo := total / 2
			if auto != "SAME_UPPER" {
				lo = total - total/2
			}
			wn.pads[i] = [2]int{int(lo), int(total - lo)}
		case auto == "VALID":
			wn.pads[i] = [2]int{}
		}
	}
	// Spread the input's elements stride apart, and pad it so that a
	// convolution with the flipped kernel, stride 1, gathers every
	// product.
	t := x
	for i := range ns {
		s := wn.stride[i]
		if s == 1 {
			continue
		}
		a := 2 + i
		shape := t.Shape()
		d := shape[a]
		split := slices.Concat(shape[:a+1], []int{1}, shape[a+1:])
		p := make([][2]int, len(split))
		p[a+1][1] = s - 1
		t = t.Reshape(split...).Pad(p, 0).Reshape(slices.Concat(shape[:a], []int{d * s}, shape[a+1:])...)
		r := whole(t.Shape())
		r[a][1] = (d-1)*s + 1
		t = t.Shrink(r)
	}
	for i := range ns {
		e := wn.effective(i) - 1
		lo, hi := e-wn.pads[i][0], e-wn.pads[i][1]+outPad[i]
		shape := t.Shape()
		a := 2 + i
		if lo < 0 || hi < 0 {
			r := whole(shape)
			r[a] = [2]int{max(-lo, 0), shape[a] - max(-hi, 0)}
			if r[a][1] < r[a][0] {
				return nil, fmt.Errorf("onnx: transposed convolution pads %v: %w", wn.pads, tensor.ErrShape)
			}
			t = t.Shrink(r)
		}
		p := make([][2]int, len(shape))
		p[a] = [2]int{max(lo, 0), max(hi, 0)}
		t = t.Pad(p, 0)
	}
	// [C, M/g, k...] -> [g, C/g, M/g, k...] -> [g, M/g, C/g, k...] ->
	// [M, C/g, k...], flipped.
	cg, mg := ws[0]/g, ws[1]
	perm := []int{0, 2, 1}
	flip := make([]int, ns)
	for i := range ns {
		perm = append(perm, 3+i)
		flip[i] = 2 + i
	}
	wt := w.Reshape(slices.Concat([]int{g, cg, mg}, kernel)...).Permute(perm...).
		Reshape(slices.Concat([]int{g * mg, cg}, kernel)...).Flip(flip...)
	unit := window{kernel: kernel, stride: slices.Repeat([]int{1}, ns), dilation: wn.dilation, pads: make([][2]int, ns)}
	y := convolve(t, wt, unit, g)
	if c.arg(2) != nil {
		y = addChannelBias(y, c.t(2))
	}
	return one1(out(c.in[0].typ, y)), nil
}

// pool returns MaxPool, or AveragePool if not max.
func pool(isMax bool) opFunc {
	return func(c *call) ([]*value, error) {
		if err := c.need(1); err != nil {
			return nil, err
		}
		if isMax && len(c.n.outputs) > 1 && c.n.outputs[1] != "" {
			return nil, c.unsupported("indices output")
		}
		x := c.t(0)
		xs := x.Shape()
		if len(xs) < 3 {
			return nil, fmt.Errorf("onnx: pooling of %v: %w", xs, tensor.ErrShape)
		}
		kernel, err := c.kernelShape(nil)
		if err != nil {
			return nil, err
		}
		if len(kernel) != len(xs)-2 {
			return nil, fmt.Errorf("onnx: kernel_shape %v for %v: %w", kernel, xs, tensor.ErrShape)
		}
		wn, err := c.window(xs[2:], kernel, true)
		if err != nil {
			return nil, err
		}
		if !isMax && slices.ContainsFunc(wn.dilation, func(d int) bool { return d != 1 }) {
			return nil, c.unsupported("dilations")
		}
		includePad := c.int("count_include_pad", 0) == 1
		// ceil_mode adds a partial window at the end of an axis when one
		// starts within the input or its padding before.
		extra := make([]int, len(kernel))
		ceil := c.int("ceil_mode", 0) == 1
		for i, d := range xs[2:] {
			span := d + wn.pads[i][0] + wn.pads[i][1] - wn.effective(i)
			if span < 0 {
				return nil, fmt.Errorf("onnx: pooling window %v over %v: %w", kernel, xs, tensor.ErrShape)
			}
			o := span/wn.stride[i] + 1
			if ceil {
				o = (span+wn.stride[i]-1)/wn.stride[i] + 1
				if (o-1)*wn.stride[i] >= d+wn.pads[i][0] {
					o--
				}
			}
			extra[i] = max(0, (o-1)*wn.stride[i]+wn.effective(i)-(d+wn.pads[i][0]+wn.pads[i][1]))
		}
		noExtra := !slices.ContainsFunc(extra, func(e int) bool { return e != 0 })
		if p, ok := wn.square(); ok && noExtra && p <= kernel[0]/2 && (isMax || p == 0 || includePad) {
			o := tensor.ConvOptions{Stride: wn.stride[0], Padding: p, Dilation: wn.dilation[0]}
			if isMax {
				return one1(out(c.in[0].typ, x.MaxPool2d(kernel[0], o))), nil
			}
			return one1(out(c.in[0].typ, x.AvgPool2d(kernel[0], o))), nil
		}
		fill := 0.0
		if isMax {
			fill = math.Inf(-1)
		}
		padded := slices.Clone(wn.pads)
		for i, e := range extra {
			padded[i][1] += e
		}
		win := slide(padSpatial(x, padded, fill), wn)
		ns := len(kernel)
		axes := make([]int, ns)
		for i := range axes {
			axes[i] = 2 + ns + i
		}
		if isMax {
			return one1(out(c.in[0].typ, win.Max(axes...))), nil
		}
		y := win.Sum(axes...)
		// Divide each window's sum by the number of elements it covers:
		// per axis, those within the input, or within its padding too.
		outShape := y.Shape()[2:]
		for i := range ns {
			counts := make([]float64, outShape[i])
			lo, hi := 0, xs[2+i]
			if includePad {
				lo, hi = -wn.pads[i][0], xs[2+i]+wn.pads[i][1]
			}
			for o := range counts {
				start := o*wn.stride[i] - wn.pads[i][0]
				counts[o] = float64(max(0, min(start+kernel[i], hi)-max(start, lo)))
			}
			shape := slices.Repeat([]int{1}, 2+ns)
			shape[2+i] = outShape[i]
			cv := hostFloats(Float, shape, counts)
			cv.key = c.key("counts")
			ct, err := c.tensor(cv)
			if err != nil {
				return nil, err
			}
			y = y.Div(ct)
		}
		return one1(out(c.in[0].typ, y)), nil
	}
}

func globalPool(isMax bool) opFunc {
	return func(c *call) ([]*value, error) {
		if err := c.need(1); err != nil {
			return nil, err
		}
		x := c.t(0)
		xs := x.Shape()
		if len(xs) < 3 {
			return nil, fmt.Errorf("onnx: global pooling of %v: %w", xs, tensor.ErrShape)
		}
		axes := make([]int, len(xs)-2)
		for i := range axes {
			axes[i] = 2 + i
		}
		var y *tensor.Tensor
		if isMax {
			y = x.Max(axes...)
		} else {
			y = x.Mean(axes...)
		}
		return one1(out(c.in[0].typ, y.Reshape(keptShape(xs, axes, true)...))), nil
	}
}

// batchNorm normalizes x, [N, C, ...], with running statistics, as a·x +
// b with a and b for each channel, computed on the host when the
// statistics are there.
func batchNorm(c *call) ([]*value, error) {
	if err := c.need(5); err != nil {
		return nil, err
	}
	if c.int("training_mode", 0) == 1 {
		return nil, c.unsupported("training_mode")
	}
	x := c.t(0)
	if x.Rank() < 2 {
		return nil, fmt.Errorf("onnx: batch normalization of %v: %w", x.Shape(), tensor.ErrShape)
	}
	eps := c.float("epsilon", 1e-5)
	var a, b *tensor.Tensor
	scale, bias, mean, vr := c.in[1], c.in[2], c.in[3], c.in[4]
	if scale.host && bias.host && mean.host && vr.host && scale.size() == bias.size() && mean.size() == scale.size() && vr.size() == scale.size() {
		// y = x·s/√(v+ε) + (b - m·s/√(v+ε)), with the factors of each
		// channel computed once.
		n := scale.size()
		av, bv := make([]float64, n), make([]float64, n)
		for i := range n {
			av[i] = scale.float(i) / math.Sqrt(vr.float(i)+eps)
			bv[i] = bias.float(i) - mean.float(i)*av[i]
		}
		ah, bh := hostFloats(Float, []int{n}, av), hostFloats(Float, []int{n}, bv)
		ah.key, bh.key = c.key("scale"), c.key("shift")
		constant := scale.constant && bias.constant && mean.constant && vr.constant
		ah.constant, bh.constant = constant, constant
		var err error
		if a, err = c.tensor(ah); err != nil {
			return nil, err
		}
		if b, err = c.tensor(bh); err != nil {
			return nil, err
		}
	} else {
		a = c.t(1).Div(c.t(4).Add(scalarOf(c.t(4), eps)).Sqrt())
		b = c.t(2).Sub(c.t(3).Mul(a))
	}
	if dt := x.DType(); dt != tensor.Float32 {
		a, b = a.Cast(dt), b.Cast(dt)
	}
	shape := slices.Repeat([]int{1}, x.Rank()-1)
	shape[0] = -1
	return one1(out(c.in[0].typ, x.Mul(a.Reshape(shape...)).Add(b.Reshape(shape...)))), nil
}

func instanceNorm(c *call) ([]*value, error) {
	if err := c.need(3); err != nil {
		return nil, err
	}
	x := c.t(0)
	xs := x.Shape()
	if len(xs) < 3 {
		return nil, fmt.Errorf("onnx: instance normalization of %v: %w", xs, tensor.ErrShape)
	}
	axes := make([]int, len(xs)-2)
	for i := range axes {
		axes[i] = 2 + i
	}
	y := normalize(x, axes, c.float("epsilon", 1e-5))
	shape := slices.Repeat([]int{1}, len(xs)-1)
	shape[0] = -1
	y = y.Mul(c.t(1).Reshape(shape...)).Add(c.t(2).Reshape(shape...))
	return one1(out(c.in[0].typ, y)), nil
}

// normalize returns x with zero mean and unit variance over axes,
// computed in float32.
func normalize(x *tensor.Tensor, axes []int, eps float64) *tensor.Tensor {
	y, _, _ := moments(x, axes, eps)
	return y
}

// moments returns x normalized over axes, and the mean and reciprocal
// standard deviation it was normalized with, which keep the axes.
func moments(x *tensor.Tensor, axes []int, eps float64) (y, mean, inv *tensor.Tensor) {
	dt := x.DType()
	if dt != tensor.Float32 {
		x = x.Cast(tensor.Float32)
	}
	keep := keptShape(x.Shape(), axes, true)
	mean = x.Mean(axes...).Reshape(keep...)
	d := x.Sub(mean)
	v := d.Mul(d).Mean(axes...).Reshape(keep...)
	inv = v.Add(scalarOf(v, eps)).Sqrt().Recip()
	y = d.Mul(inv)
	if dt != tensor.Float32 {
		y = y.Cast(dt)
	}
	return y, mean, inv
}

func layerNorm(c *call) ([]*value, error) {
	if err := c.need(2); err != nil {
		return nil, err
	}
	x := c.t(0)
	r := x.Rank()
	axis, err := normAxis(c.int("axis", -1), r)
	if err != nil {
		return nil, err
	}
	axes := make([]int, 0, r-axis)
	for d := axis; d < r; d++ {
		axes = append(axes, d)
	}
	y, mean, inv := moments(x, axes, c.float("epsilon", 1e-5))
	y = y.Mul(c.t(1))
	if c.arg(2) != nil {
		y = y.Add(c.t(2))
	}
	return []*value{out(c.in[0].typ, y), out(Float, mean), out(Float, inv)}, nil
}

func resize(c *call) ([]*value, error) {
	if err := c.need(1); err != nil {
		return nil, err
	}
	x := c.in[0]
	r := len(x.shape)
	mode := c.str("mode", "nearest")
	if mode != "nearest" && mode != "linear" {
		return nil, c.unsupported("mode %q", mode)
	}
	if c.int("antialias", 0) != 0 {
		return nil, c.unsupported("antialias")
	}
	if p := c.str("keep_aspect_ratio_policy", "stretch"); p != "stretch" {
		return nil, c.unsupported("keep_aspect_ratio_policy %q", p)
	}
	coord := c.str("coordinate_transformation_mode", "half_pixel")
	nearest := c.str("nearest_mode", "round_prefer_floor")
	axes := make([]int, r)
	for i := range axes {
		axes[i] = i
	}
	if a, ok := c.ints("axes"); ok {
		var err error
		if axes, err = normAxes(a, r); err != nil {
			return nil, err
		}
	}
	scales := make([]float64, r)
	sizes := slices.Clone(x.shape)
	for i := range scales {
		scales[i] = 1
	}
	sz, haveSizes, err := c.intList(3)
	if err != nil {
		return nil, err
	}
	if haveSizes && len(sz) > 0 {
		if len(sz) != len(axes) {
			return nil, fmt.Errorf("onnx: resize sizes %v for %d axes: %w", sz, len(axes), tensor.ErrShape)
		}
		for i, a := range axes {
			if sz[i] < 0 || sz[i] > math.MaxInt32 {
				return nil, fmt.Errorf("onnx: resize sizes %v: %w", sz, tensor.ErrShape)
			}
			sizes[a] = int(sz[i])
			scales[a] = float64(sz[i]) / float64(x.shape[a])
		}
	} else {
		sc, err := c.host(2)
		if err != nil {
			return nil, err
		}
		if sc.size() != len(axes) {
			return nil, fmt.Errorf("onnx: resize scales of %v for %d axes: %w", sc.shape, len(axes), tensor.ErrShape)
		}
		for i, a := range axes {
			s := sc.float(i)
			o := math.Floor(float64(x.shape[a]) * s)
			if !(s > 0) || o > math.MaxInt32 {
				return nil, fmt.Errorf("onnx: resize scale %v: %w", s, tensor.ErrShape)
			}
			scales[a], sizes[a] = s, int(o)
		}
	}
	for _, o := range sizes {
		if o > maxIndices {
			return nil, fmt.Errorf("onnx: resize %v to %v: %w", x.shape, sizes, tensor.ErrShape)
		}
	}
	v := x
	for a := range r {
		in, o := x.shape[a], sizes[a]
		if in == o && scales[a] == 1 {
			continue
		}
		if in == 0 || o == 0 {
			return nil, fmt.Errorf("onnx: resize %v to %v: %w", x.shape, sizes, tensor.ErrShape)
		}
		src := make([]float64, o)
		for i := range src {
			f, err := sourceCoord(c, coord, float64(i), scales[a], in, o)
			if err != nil {
				return nil, err
			}
			src[i] = f
		}
		if mode == "nearest" {
			idx := make([]int, o)
			for i, f := range src {
				var j float64
				switch nearest {
				case "round_prefer_floor":
					if f == math.Floor(f)+0.5 {
						j = math.Floor(f)
					} else {
						j = math.Round(f)
					}
				case "round_prefer_ceil":
					j = math.Floor(f + 0.5)
				case "floor":
					j = math.Floor(f)
				case "ceil":
					j = math.Ceil(f)
				default:
					return nil, c.unsupported("nearest_mode %q", nearest)
				}
				idx[i] = int(min(max(j, 0), float64(in-1)))
			}
			if v, err = repeatOrTake(c, v, a, idx); err != nil {
				return nil, err
			}
			continue
		}
		lo, hi, w := make([]int, o), make([]int, o), make([]float64, o)
		for i, f := range src {
			f = min(max(f, 0), float64(in-1))
			lo[i] = int(math.Floor(f))
			hi[i] = min(lo[i]+1, in-1)
			w[i] = f - float64(lo[i])
		}
		a0, err := take(c, v, a, lo)
		if err != nil {
			return nil, err
		}
		a1, err := take(c, v, a, hi)
		if err != nil {
			return nil, err
		}
		shape := slices.Repeat([]int{1}, r)
		shape[a] = o
		wv := hostFloats(Float, shape, w)
		wv.key = c.key("weights")
		wt, err := c.tensor(wv)
		if err != nil {
			return nil, err
		}
		t0, t1 := c.tensorOf(a0), c.tensorOf(a1)
		v = out(x.typ, t0.Add(t1.Sub(t0).Mul(wt)))
	}
	return one1(v), nil
}

// sourceCoord returns the coordinate in the input of output coordinate o
// of a resize with the given scale, from in to out elements.
func sourceCoord(c *call, mode string, o, scale float64, in, out int) (float64, error) {
	switch mode {
	case "half_pixel":
		return (o+0.5)/scale - 0.5, nil
	case "pytorch_half_pixel":
		if out > 1 {
			return (o+0.5)/scale - 0.5, nil
		}
		return 0, nil
	case "align_corners":
		if out == 1 {
			return 0, nil
		}
		return o * float64(in-1) / float64(out-1), nil
	case "asymmetric":
		return o / scale, nil
	case "tf_half_pixel_for_nn":
		return (o + 0.5) / scale, nil
	}
	return 0, c.unsupported("coordinate_transformation_mode %q", mode)
}

// repeatOrTake returns x's elements at idx along axis: each repeated f
// times, a view expanded, when idx is i/f.
func repeatOrTake(c *call, x *value, axis int, idx []int) (*value, error) {
	in := x.shape[axis]
	f := len(idx) / in
	repeat := f > 1 && f*in == len(idx) && !x.host
	for i := 0; repeat && i < len(idx); i++ {
		repeat = idx[i] == i/f
	}
	if !repeat {
		return take(c, x, axis, idx)
	}
	t := c.tensorOf(x)
	shape := t.Shape()
	split := slices.Concat(shape[:axis+1], []int{1}, shape[axis+1:])
	wide := slices.Concat(shape[:axis+1], []int{f}, shape[axis+1:])
	final := slices.Clone(shape)
	final[axis] = in * f
	return out(x.typ, t.Reshape(split...).Expand(wide...).Reshape(final...)), nil
}
