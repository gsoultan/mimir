package mim

import (
	"fmt"
	"math"
	"slices"

	mimfile "github.com/gsoultan/mimir-model"

	"github.com/gsoultan/mimir/tensor"
)

// composites holds the standard-library composites mimir runs, by
// name.vN. Each runs on mimir's kernels within the composite's tolerance
// (spec/stdlib.md of the format): float inputs are computed in Float32,
// and outputs are cast to their declared dtypes.
var composites map[string]func(c *call) ([]*tensor.Tensor, error)

func init() {
	composites = map[string]func(c *call) ([]*tensor.Tensor, error){
		"matmul.v1":            matmul,
		"linear.v1":            linear,
		"dequant_matmul.v1":    dequantMatmul(false),
		"dequant_matmul_a8.v1": dequantMatmul(true),
		"dequant_matmul_a8.v2": dequantMatmul(true),
		"conv2d.v1":            conv2d,
		"sigmoid.v1":           unary((*tensor.Tensor).Sigmoid),
		"tanh.v1":              unary((*tensor.Tensor).Tanh),
		"erf.v1":               unary((*tensor.Tensor).Erf),
		"exp.v1":               unary((*tensor.Tensor).Exp),
		"log.v1":               unary((*tensor.Tensor).Log),
		"cos.v1":               unary((*tensor.Tensor).Cos),
		"rsqrt.v1":             unary(func(x *tensor.Tensor) *tensor.Tensor { return x.Sqrt().Recip() }),
		"silu.v1":              unary((*tensor.Tensor).SiLU),
		"gelu_erf.v1":          unary(geluErf),
		"gelu_tanh.v1":         unary((*tensor.Tensor).GELU),
		"swiglu.v1":            swiglu,
		"geglu.v1":             geglu,
		"softmax.v1":           softmax(false),
		"log_softmax.v1":       softmax(true),
		"layer_norm.v1":        layerNorm,
		"rms_norm.v1":          rmsNorm,
		"mean.v1":              mean,
		"variance.v1":          variance,
		"cumsum.v1":            cumsum,
		"rope.v1":              rope,
		"attention.v1":         attention(false),
		"attention.v2":         attention(true),
		"embedding.v1":         embedding,
		"top_k.v1":             topK,
		"argsort.v1":           argsort,
	}
}

// composite runs a call: of the standard library, or of a composite of
// the model's own, whose body is a subgraph.
func (c *call) composite() ([]*tensor.Tensor, error) {
	name := opName(c.op)[len("call "):]
	if f, ok := composites[name]; ok {
		return f(c)
	}
	si, ok := c.r.m.locals[name]
	if !ok {
		return nil, c.unsupported("")
	}
	for _, a := range c.op.Attrs {
		if a.Key != "name" && a.Key != "version" {
			return nil, fmt.Errorf("%w: attribute %s of a model-local composite", ErrFormat, a.Key)
		}
	}
	return c.r.subgraph(&c.r.m.g.Subgraphs[si], c.in)
}

// f32 returns input i in its computation dtype: Float32 for the float
// types, which a composite computes in.
func (c *call) f32(i int) *tensor.Tensor {
	if c.in[i].DType().IsFloat() {
		return c.in[i].Cast(tensor.Float32)
	}
	return c.in[i]
}

// need checks that the call has from to to inputs.
func (c *call) need(from, to int) error {
	if n := len(c.in); n < from || n > to {
		return fmt.Errorf("%w: %d inputs, want %d to %d", ErrFormat, n, from, to)
	}
	return nil
}

func unary(f func(*tensor.Tensor) *tensor.Tensor) func(*call) ([]*tensor.Tensor, error) {
	return func(c *call) ([]*tensor.Tensor, error) {
		if err := c.need(1, 1); err != nil {
			return nil, err
		}
		return one(f(c.f32(0)))
	}
}

// geluErf returns x·(1 + erf(x/√2))/2.
func geluErf(x *tensor.Tensor) *tensor.Tensor {
	k := func(v float64) *tensor.Tensor {
		return tensor.Full(x.DType(), v, slices.Repeat([]int{1}, x.Rank())...).To(x.Device())
	}
	return x.Mul(k(0.5)).Mul(x.Mul(k(math.Sqrt2 / 2)).Erf().Add(k(1)))
}

func matmul(c *call) ([]*tensor.Tensor, error) {
	if err := c.need(2, 2); err != nil {
		return nil, err
	}
	if !isFloat(c.dt(0)) {
		return nil, c.unsupported("products of " + c.dt(0).String())
	}
	return one(c.f32(0).MatMul(c.f32(1)))
}

func linear(c *call) ([]*tensor.Tensor, error) {
	if err := c.need(2, 3); err != nil {
		return nil, err
	}
	x := c.f32(0)
	return one(c.adapt(c.bias(x.MatMulT(c.f32(1)), 2), x))
}

// bias adds input i, of shape [N], to the last axis of y, if the call has
// it.
func (c *call) bias(y *tensor.Tensor, i int) *tensor.Tensor {
	if len(c.in) <= i || y.Err() != nil {
		return y
	}
	b := c.f32(i)
	if y.Rank() == 0 {
		return tensor.Failed(fmt.Errorf("%w: a bias for a scalar", ErrFormat))
	}
	return y.Add(b.Reshape(append(slices.Repeat([]int{1}, y.Rank()-1), b.Size())...))
}

// dequantMatmul returns the product with a quantized weight. mimir's
// product with block-quantized weights rounds the activations to 8 bits
// in blocks of the weights' block size, as the a8 composites allow when
// their blocks are no shorter; otherwise, and for dequant_matmul.v1, the
// weights are decoded first.
func dequantMatmul(a8 bool) func(c *call) ([]*tensor.Tensor, error) {
	return func(c *call) ([]*tensor.Tensor, error) {
		if err := c.need(2, 3); err != nil {
			return nil, err
		}
		block := c.int("block", 32)
		if block != 32 && block != 256 {
			return nil, fmt.Errorf("%w: activation block %d, want 32 or 256", ErrFormat, block)
		}
		w := c.in[1]
		if w.DType().IsQuantized() {
			wblock := int64(256)
			if w.DType() == tensor.Q8_0 {
				wblock = 32
			}
			if !a8 || wblock > block {
				w = w.Cast(tensor.Float32)
			}
		}
		x := c.f32(0)
		return one(c.adapt(c.bias(x.MatMulT(w), 2), x))
	}
}

func conv2d(c *call) ([]*tensor.Tensor, error) {
	if err := c.need(2, 3); err != nil {
		return nil, err
	}
	x, w := c.f32(0), c.f32(1)
	var b *tensor.Tensor
	if len(c.in) == 3 {
		b = c.f32(2)
	}
	strides := c.ints("strides", []int64{1, 1})
	dilations := c.ints("dilations", []int64{1, 1})
	pads := c.ints("pads", []int64{0, 0, 0, 0})
	groups := c.int("groups", 1)
	if len(strides) != 2 || len(dilations) != 2 || len(pads) != 4 || w.Rank() != 4 {
		return nil, fmt.Errorf("%w: conv2d with strides %v, dilations %v, pads %v", ErrFormat, strides, dilations, pads)
	}
	if dilations[0] != dilations[1] {
		return nil, c.unsupported(fmt.Sprintf("dilations %v", dilations))
	}
	d := int(dilations[0])
	// mimir's kernels are square: a kernel shorter along one axis is
	// padded with zeros at its end, and the input with as many dilated
	// elements, which leaves the output as it was.
	ws := w.Shape()
	kh, kw := ws[2], ws[3]
	k := max(kh, kw)
	top, bottom, left, right := int(pads[0]), int(pads[1]), int(pads[2]), int(pads[3])
	if kh != kw {
		w = w.Pad([][2]int{{0, 0}, {0, 0}, {0, k - kh}, {0, k - kw}}, 0)
		bottom += d * (k - kh)
		right += d * (k - kw)
	}
	x = x.Pad([][2]int{{0, 0}, {0, 0}, {top, bottom}, {left, right}}, 0)
	// Different strides along the axes run with stride 1, then keep every
	// stride-th row and column.
	s := int(strides[0])
	if strides[0] != strides[1] {
		s = 1
	}
	y := x.Conv2d(w, b, tensor.ConvOptions{Stride: s, Dilation: d, Groups: int(groups)})
	if s == 1 && (strides[0] != 1 || strides[1] != 1) {
		y = every(every(y, 2, int(strides[0])), 3, int(strides[1]))
	}
	return one(y)
}

// every keeps elements 0, step, 2·step, ... of t along axis.
func every(t *tensor.Tensor, axis, step int) *tensor.Tensor {
	if step == 1 || t.Err() != nil {
		return t
	}
	shape := t.Shape()
	n := (shape[axis] + step - 1) / step
	p := make([][2]int, len(shape))
	p[axis][1] = n*step - shape[axis]
	t = t.Pad(p, 0)
	split := slices.Concat(shape[:axis], []int{n, step}, shape[axis+1:])
	keep := whole(split)
	keep[axis+1] = [2]int{0, 1}
	shape[axis] = n
	return t.Reshape(split...).Shrink(keep).Reshape(shape...)
}

func swiglu(c *call) ([]*tensor.Tensor, error) {
	if err := c.need(2, 2); err != nil {
		return nil, err
	}
	return one(c.f32(0).SiLU().Mul(c.f32(1)))
}

func geglu(c *call) ([]*tensor.Tensor, error) {
	if err := c.need(2, 2); err != nil {
		return nil, err
	}
	switch a := c.str("approximate", "none"); a {
	case "none":
		return one(geluErf(c.f32(0)).Mul(c.f32(1)))
	case "tanh":
		return one(c.f32(0).GELU().Mul(c.f32(1)))
	default:
		return nil, fmt.Errorf("%w: geglu approximation %q", ErrFormat, a)
	}
}

func softmax(log bool) func(c *call) ([]*tensor.Tensor, error) {
	return func(c *call) ([]*tensor.Tensor, error) {
		if err := c.need(1, 1); err != nil {
			return nil, err
		}
		x := c.f32(0)
		a := c.axis("axis", -1, x.Rank())
		if c.err != nil {
			return nil, c.err
		}
		if log {
			return one(x.LogSoftmax(a))
		}
		return one(x.Softmax(a))
	}
}

// tail returns the axes from a to the last of a tensor of rank r.
func tail(a, r int) []int {
	axes := make([]int, 0, r-a)
	for i := a; i < r; i++ {
		axes = append(axes, i)
	}
	return axes
}

// meanKeep returns the mean of x over axes, which stay with size 1.
func meanKeep(x *tensor.Tensor, axes []int) *tensor.Tensor {
	if x.Err() != nil {
		return x
	}
	keep := x.Shape()
	for _, a := range axes {
		keep[a] = 1
	}
	return x.Mean(axes...).Reshape(keep...)
}

// scaled multiplies y by input i, the shape of y's axes from a on, if the
// call has it, and adds input i+1 likewise.
func (c *call) scaled(y *tensor.Tensor, a, i int) *tensor.Tensor {
	lead := slices.Repeat([]int{1}, a)
	for j := i; j < min(i+2, len(c.in)); j++ {
		p := c.f32(j)
		p = p.Reshape(slices.Concat(lead, p.Shape())...)
		if j == i {
			y = y.Mul(p)
		} else {
			y = y.Add(p)
		}
	}
	return y
}

func layerNorm(c *call) ([]*tensor.Tensor, error) {
	if err := c.need(2, 3); err != nil {
		return nil, err
	}
	x := c.f32(0)
	a := c.axis("axis", -1, x.Rank())
	if c.err != nil {
		return nil, c.err
	}
	axes := tail(a, x.Rank())
	d := x.Sub(meanKeep(x, axes))
	v := meanKeep(d.Mul(d), axes)
	n := d.Mul(v.Add(c.scalar(v, c.float("epsilon", 1e-5))).Sqrt().Recip())
	// The normalized values are rounded to the input's dtype before the
	// scale and bias, as the composite's body does.
	if dt := c.in[0].DType(); dt != tensor.Float32 {
		n = n.Cast(dt).Cast(tensor.Float32)
	}
	return one(c.scaled(n, a, 1))
}

func rmsNorm(c *call) ([]*tensor.Tensor, error) {
	if err := c.need(1, 2); err != nil {
		return nil, err
	}
	x := c.f32(0)
	a := c.axis("axis", -1, x.Rank())
	if c.err != nil {
		return nil, c.err
	}
	ms := meanKeep(x.Mul(x), tail(a, x.Rank()))
	y := x.Mul(ms.Add(c.scalar(ms, c.float("epsilon", 1e-6))).Sqrt().Recip())
	if len(c.in) == 2 {
		s := c.f32(1)
		y = y.Mul(s.Reshape(slices.Concat(slices.Repeat([]int{1}, a), s.Shape())...))
	}
	return one(y)
}

// axesOf returns the axes attribute of a reduction composite, normalized
// for rank r.
func (c *call) axesOf(r int) []int {
	axes := toInts(c.ints("axes", nil))
	for i, a := range axes {
		if a < 0 {
			a += r
		}
		if a < 0 || a >= r {
			c.fail(fmt.Errorf("%w: axis %d of rank %d", ErrFormat, axes[i], r))
			return nil
		}
		axes[i] = a
	}
	return axes
}

func mean(c *call) ([]*tensor.Tensor, error) {
	if err := c.need(1, 1); err != nil {
		return nil, err
	}
	x := c.f32(0)
	axes := c.axesOf(x.Rank())
	if c.err != nil {
		return nil, c.err
	}
	if len(axes) == 0 {
		return one(x)
	}
	return one(meanKeep(x, axes))
}

func variance(c *call) ([]*tensor.Tensor, error) {
	if err := c.need(1, 1); err != nil {
		return nil, err
	}
	x := c.f32(0)
	axes := c.axesOf(x.Rank())
	if c.err != nil {
		return nil, c.err
	}
	keep, n := x.Shape(), 1
	for _, a := range axes {
		n *= keep[a]
		keep[a] = 1
	}
	d := x.Sub(meanKeep(x, axes))
	s := d.Mul(d).Sum(axes...).Reshape(keep...)
	return one(s.Div(c.scalar(s, float64(n)-float64(c.int("correction", 0)))))
}

func cumsum(c *call) ([]*tensor.Tensor, error) {
	if err := c.need(1, 1); err != nil {
		return nil, err
	}
	x := c.f32(0)
	a := c.axis("axis", 0, x.Rank())
	if c.err != nil {
		return nil, c.err
	}
	reverse := c.bool("reverse", false)
	if reverse {
		x = x.Flip(a)
	}
	if c.bool("exclusive", false) {
		// The sums of the elements before each: shift by one, then sum.
		shape := x.Shape()
		p := make([][2]int, len(shape))
		p[a][0] = 1
		keep := whole(shape)
		x = x.Pad(p, 0).Shrink(keep)
	}
	y := x.CumSum(a)
	if reverse {
		y = y.Flip(a)
	}
	return one(y)
}

func rope(c *call) ([]*tensor.Tensor, error) {
	if err := c.need(2, 2); err != nil {
		return nil, err
	}
	x := c.f32(0)
	shape := x.Shape()
	if len(shape) != 4 || c.in[1].Rank() != 2 {
		return nil, fmt.Errorf("%w: rope of %v at positions %v", ErrFormat, shape, c.in[1].Shape())
	}
	b, h, t, d := shape[0], shape[1], shape[2], shape[3]
	theta := c.float("theta", 10000)
	rd := int(c.int("rotary_dims", int64(d)))
	if rd%2 != 0 || rd <= 0 || rd > d {
		return nil, fmt.Errorf("%w: rotary_dims %d of %d", ErrFormat, rd, d)
	}
	half := rd / 2
	factor := c.float("yarn_factor", 1)
	ms := 1.0
	freq := make([]float32, half)
	f := make([]float64, half)
	for i := range f {
		f[i] = math.Pow(theta, -2*float64(i)/float64(rd))
		freq[i] = float32(f[i])
	}
	if factor != 1 {
		orig := float64(c.int("yarn_original_context", 0))
		if orig <= 0 {
			return nil, fmt.Errorf("%w: rope with yarn_factor %g and no yarn_original_context", ErrFormat, factor)
		}
		corr := func(n float64) float64 {
			return float64(rd) * math.Log(orig/(n*2*math.Pi)) / (2 * math.Log(theta))
		}
		lo := math.Max(math.Floor(corr(c.float("yarn_beta_fast", 32))), 0)
		hi := math.Min(math.Ceil(corr(c.float("yarn_beta_slow", 1))), float64(rd-1))
		if lo == hi {
			hi += 0.001
		}
		for i := range f {
			ramp := math.Min(math.Max((float64(i)-lo)/(hi-lo), 0), 1)
			freq[i] = float32(f[i]/factor*ramp + f[i]*(1-ramp))
		}
		ms = c.float("yarn_attn_factor", 0.1*math.Log(factor)+1)
	}
	pos := c.in[1].Cast(tensor.Float32)
	ang := pos.Reshape(pos.Shape()[0], 1, t, 1).Mul(tensor.FromSlice(freq, 1, 1, 1, half).To(c.r.dev))
	cos, sin := ang.Cos(), ang.Sin()
	if ms != 1 {
		cos, sin = cos.Mul(c.scalar(cos, ms)), sin.Mul(c.scalar(sin, ms))
	}
	xr := x
	if rd < d {
		xr = x.Shrink([][2]int{{0, b}, {0, h}, {0, t}, {0, rd}})
	}
	var x1, x2 *tensor.Tensor
	interleaved := c.bool("interleaved", false)
	if interleaved {
		xi := xr.Reshape(b, h, t, half, 2)
		x1 = xi.Shrink([][2]int{{0, b}, {0, h}, {0, t}, {0, half}, {0, 1}}).Reshape(b, h, t, half)
		x2 = xi.Shrink([][2]int{{0, b}, {0, h}, {0, t}, {0, half}, {1, 2}}).Reshape(b, h, t, half)
	} else {
		x1 = xr.Shrink([][2]int{{0, b}, {0, h}, {0, t}, {0, half}})
		x2 = xr.Shrink([][2]int{{0, b}, {0, h}, {0, t}, {half, rd}})
	}
	y1 := x1.Mul(cos).Sub(x2.Mul(sin))
	y2 := x2.Mul(cos).Add(x1.Mul(sin))
	var y *tensor.Tensor
	if interleaved {
		y = tensor.Stack(4, y1, y2).Reshape(b, h, t, rd)
	} else {
		y = tensor.Cat(3, y1, y2)
	}
	if rd < d {
		y = tensor.Cat(3, y, x.Shrink([][2]int{{0, b}, {0, h}, {0, t}, {rd, d}}))
	}
	return one(y)
}

func attention(v2 bool) func(c *call) ([]*tensor.Tensor, error) {
	return func(c *call) ([]*tensor.Tensor, error) {
		if err := c.need(3, 5); err != nil {
			return nil, err
		}
		// The kernel reads keys and values of any float dtype as they
		// are, as a half key-value cache, rather than a Float32 copy of
		// the whole cache made every run.
		q, k, v := c.f32(0), c.in[1], c.in[2]
		if !v2 && (c.dt(1) != c.dt(0) || c.dt(2) != c.dt(0)) {
			return nil, fmt.Errorf("%w: attention.v1 of %s, %s and %s", ErrFormat, c.dt(0), c.dt(1), c.dt(2))
		}
		qs, ks := q.Shape(), k.Shape()
		if len(qs) != 4 || len(ks) != 4 || v.Rank() != 4 {
			return nil, fmt.Errorf("%w: attention of %v, %v and %v", ErrFormat, qs, ks, v.Shape())
		}
		if hkv := c.int("num_kv_heads", int64(qs[1])); hkv != int64(ks[1]) || hkv <= 0 || int64(qs[1])%hkv != 0 {
			return nil, fmt.Errorf("%w: %d query heads, %d key heads and num_kv_heads %d", ErrFormat, qs[1], ks[1], hkv)
		}
		scale := c.float("scale", 1/math.Sqrt(float64(qs[3])))
		causal := c.bool("causal", false)
		softcap := c.float("softcap", 0)
		var mask, sinks *tensor.Tensor
		if len(c.in) > 3 {
			mask = c.in[3]
		}
		if len(c.in) > 4 {
			sinks = c.f32(4)
		}
		if softcap > 0 || sinks != nil {
			return one(c.attentionBody(q, c.f32(1), c.f32(2), mask, sinks, scale, softcap, causal))
		}
		opts := []tensor.AttentionOption{tensor.AttentionScale(scale)}
		if causal {
			opts = append(opts, tensor.CausalOffset(ks[2]-qs[2]))
		}
		if mask != nil {
			if c.dt(3) == mimfile.Bool {
				opts = append(opts, tensor.BoolMask(mask))
			} else {
				opts = append(opts, tensor.AdditiveMask(mask.Cast(tensor.Float32)))
			}
		}
		return one(tensor.ScaledDotProductAttention(q, k, v, opts...))
	}
}

// attentionBody computes attention as the composite's body does, for the
// soft cap and the sinks that mimir's kernel lacks.
func (c *call) attentionBody(q, k, v, mask, sinks *tensor.Tensor, scale, softcap float64, causal bool) *tensor.Tensor {
	qs, ks, vs := q.Shape(), k.Shape(), v.Shape()
	b, hq, tq, hkv, tk := qs[0], qs[1], qs[2], ks[1], ks[2]
	g := hq / hkv
	rep := func(t *tensor.Tensor, d int) *tensor.Tensor {
		return t.Reshape(b, hkv, 1, tk, d).Expand(b, hkv, g, tk, d).Reshape(b, hq, tk, d)
	}
	s := q.MatMul(rep(k, ks[3]).Transpose(2, 3))
	s = s.Mul(c.scalar(s, scale))
	if softcap > 0 {
		s = s.Mul(c.scalar(s, 1/softcap)).Tanh().Mul(c.scalar(s, softcap))
	}
	ninf := c.scalar(s, math.Inf(-1))
	if mask != nil {
		if c.dt(3) == mimfile.Bool {
			s = s.Add(tensor.Where(mask, c.scalar(s, 0), ninf))
		} else {
			s = s.Add(mask.Cast(tensor.Float32))
		}
	}
	if causal {
		i := tensor.Arange[int32](0, int32(tq), 1).To(c.r.dev).Reshape(1, 1, tq, 1)
		j := tensor.Arange[int32](0, int32(tk), 1).To(c.r.dev).Reshape(1, 1, 1, tk)
		hidden := i.Add(c.scalar(i, float64(tk-tq))).Less(j)
		s = s.Add(tensor.Where(hidden, ninf, c.scalar(s, 0)))
	}
	var p *tensor.Tensor
	if sinks != nil {
		z := sinks.Reshape(1, hq, 1, 1)
		m := s.Max(3).Reshape(b, hq, tq, 1).Maximum(z)
		e := s.Sub(m).Exp()
		p = e.Div(e.Sum(3).Reshape(b, hq, tq, 1).Add(z.Sub(m).Exp()))
	} else {
		p = s.Softmax(3)
	}
	return p.MatMul(rep(v, vs[3]))
}

func embedding(c *call) ([]*tensor.Tensor, error) {
	if err := c.need(2, 2); err != nil {
		return nil, err
	}
	if c.in[0].Rank() != 2 {
		return nil, fmt.Errorf("%w: embedding table of shape %v", ErrFormat, c.in[0].Shape())
	}
	return one(c.take(c.in[0], c.in[1], 0))
}

func topK(c *call) ([]*tensor.Tensor, error) {
	if err := c.need(1, 1); err != nil {
		return nil, err
	}
	x := c.f32(0)
	a := c.axis("axis", -1, x.Rank())
	if c.err != nil {
		return nil, c.err
	}
	v, i := x.TopK(int(c.int("k", 0)), a, c.bool("largest", true))
	return []*tensor.Tensor{v, i}, cmpErr(v, i)
}

func argsort(c *call) ([]*tensor.Tensor, error) {
	if err := c.need(1, 1); err != nil {
		return nil, err
	}
	x := c.f32(0)
	a := c.axis("axis", -1, x.Rank())
	if c.err != nil {
		return nil, c.err
	}
	return one(x.Argsort(a, c.bool("descending", false)))
}

func cmpErr(ts ...*tensor.Tensor) error {
	for _, t := range ts {
		if err := t.Err(); err != nil {
			return err
		}
	}
	return nil
}
