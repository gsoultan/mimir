package onnx

import (
	"context"
	"math"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/tensor"
)

// convRef computes a convolution of x, [N, C, spatial...], with w,
// [M, C/groups, kernel...], directly from its definition.
func convRef(x, w *arr, b []float64, stride, dil []int, pads [][2]int, groups int) *arr {
	ns := len(x.shape) - 2
	n, cin, m := x.shape[0], x.shape[1], w.shape[0]
	outSp := make([]int, ns)
	for i := range ns {
		outSp[i] = (x.shape[2+i]+pads[i][0]+pads[i][1]-(w.shape[2+i]-1)*dil[i]-1)/stride[i] + 1
	}
	y := newArr(slices.Concat([]int{n, m}, outSp)...)
	cg, mg := cin/groups, m/groups
	opos := make([]int, ns)
	kpos := make([]int, ns)
	for i := range y.data {
		// Decode i into n, m and the output position.
		r := i
		for d := ns - 1; d >= 0; d-- {
			opos[d] = r % outSp[d]
			r /= outSp[d]
		}
		oc, bn := r%m, r/m
		g := oc / mg
		s := 0.0
		ksize := 1
		for _, k := range w.shape[2:] {
			ksize *= k
		}
		for ic := range cg {
			for kk := range ksize {
				q := kk
				for d := ns - 1; d >= 0; d-- {
					kpos[d] = q % w.shape[2+d]
					q /= w.shape[2+d]
				}
				off := bn*cin + g*cg + ic
				inside := true
				for d := range ns {
					p := opos[d]*stride[d] - pads[d][0] + kpos[d]*dil[d]
					if p < 0 || p >= x.shape[2+d] {
						inside = false
						break
					}
					off = off*x.shape[2+d] + p
				}
				if !inside {
					continue
				}
				woff := oc*cg + ic
				for d := range ns {
					woff = woff*w.shape[2+d] + kpos[d]
				}
				s += x.data[off] * w.data[woff]
			}
		}
		if b != nil {
			s += b[oc]
		}
		y.data[i] = s
	}
	return y
}

func ints64(v ...int) []int64 {
	out := make([]int64, len(v))
	for i, x := range v {
		out[i] = int64(x)
	}
	return out
}

func TestConv(t *testing.T) {
	cases := []struct {
		name          string
		x, w          []int
		stride, dil   []int
		pads          [][2]int
		groups        int
		autoPad       string
		bias          bool
		noKernelShape bool
	}{
		{name: "square", x: []int{2, 4, 7, 7}, w: []int{6, 2, 3, 3}, stride: []int{2, 2}, dil: []int{1, 1}, pads: [][2]int{{1, 1}, {1, 1}}, groups: 2, bias: true},
		{name: "dilated", x: []int{1, 3, 8, 8}, w: []int{4, 3, 3, 3}, stride: []int{1, 1}, dil: []int{2, 2}, pads: [][2]int{{2, 2}, {2, 2}}, groups: 1},
		{name: "asymmetric pads", x: []int{1, 2, 6, 5}, w: []int{3, 2, 3, 3}, stride: []int{1, 1}, dil: []int{1, 1}, pads: [][2]int{{0, 1}, {2, 1}}, groups: 1, bias: true},
		{name: "rectangular", x: []int{2, 3, 7, 9}, w: []int{4, 3, 2, 3}, stride: []int{2, 1}, dil: []int{1, 2}, pads: [][2]int{{1, 0}, {1, 2}}, groups: 1, bias: true},
		{name: "1-D", x: []int{2, 4, 11}, w: []int{4, 1, 3}, stride: []int{2}, dil: []int{1}, pads: [][2]int{{1, 1}}, groups: 4, bias: true},
		{name: "3-D", x: []int{1, 2, 4, 5, 5}, w: []int{3, 2, 2, 3, 3}, stride: []int{1, 2, 1}, dil: []int{1, 1, 1}, pads: [][2]int{{0, 0}, {1, 1}, {1, 1}}, groups: 1},
		{name: "same upper", x: []int{1, 2, 7, 7}, w: []int{3, 2, 4, 4}, stride: []int{2, 2}, dil: []int{1, 1}, pads: [][2]int{{1, 2}, {1, 2}}, groups: 1, autoPad: "SAME_UPPER", noKernelShape: true},
	}
	for i, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			x, w, b := random(uint64(20+i), tc.x...), random(uint64(40+i), tc.w...), random(uint64(60+i), tc.w[0])
			attrs := []*pb{aInts("strides", ints64(tc.stride...)...), aInts("dilations", ints64(tc.dil...)...), aInt("group", int64(tc.groups))}
			if !tc.noKernelShape {
				attrs = append(attrs, aInts("kernel_shape", ints64(tc.w[2:]...)...))
			}
			if tc.autoPad != "" {
				attrs = append(attrs, aString("auto_pad", tc.autoPad))
			} else {
				var p []int
				for _, q := range tc.pads {
					p = append(p, q[0])
				}
				for _, q := range tc.pads {
					p = append(p, q[1])
				}
				attrs = append(attrs, aInts("pads", ints64(p...)...))
			}
			io := []string{"x", "w", "->", "y"}
			var bias []float64
			g := newGraph().input("x", Float, anyDims(tc.x)...).init("w", tc.w, w.f32())
			if tc.bias {
				io = []string{"x", "w", "b", "->", "y"}
				g.init("b", []int{tc.w[0]}, b.f32())
				bias = b.data
			}
			g.node("Conv", io, attrs...).output("y")
			got, shape := runOne(t, g, map[string]*tensor.Tensor{"x": x.tensor()}, "y")
			expect(t, got, shape, convRef(x, w, bias, tc.stride, tc.dil, tc.pads, tc.groups), 1e-5)
		})
	}
}

func anyDims(d []int) []any {
	out := make([]any, len(d))
	for i, x := range d {
		out[i] = x
	}
	return out
}

// convTransposeRef computes a transposed convolution by scattering each
// input element's products.
func convTransposeRef(x, w *arr, stride, dil []int, pads [][2]int, outPad []int, groups int) *arr {
	n, cin := x.shape[0], x.shape[1]
	cg, mg := cin/groups, w.shape[1]
	m := mg * groups
	h, wd := x.shape[2], x.shape[3]
	kh, kw := w.shape[2], w.shape[3]
	oh := stride[0]*(h-1) + outPad[0] + (kh-1)*dil[0] + 1 - pads[0][0] - pads[0][1]
	ow := stride[1]*(wd-1) + outPad[1] + (kw-1)*dil[1] + 1 - pads[1][0] - pads[1][1]
	y := newArr(n, m, oh, ow)
	for b := range n {
		for ic := range cin {
			g := ic / cg
			for i := range h {
				for j := range wd {
					v := x.at(b, ic, i, j)
					for oc := range mg {
						for p := range kh {
							for q := range kw {
								oi := i*stride[0] + p*dil[0] - pads[0][0]
								oj := j*stride[1] + q*dil[1] - pads[1][0]
								if oi < 0 || oj < 0 || oi >= oh || oj >= ow {
									continue
								}
								y.data[y.off(b, g*mg+oc, oi, oj)] += v * w.at(ic, oc, p, q)
							}
						}
					}
				}
			}
		}
	}
	return y
}

func TestConvTranspose(t *testing.T) {
	cases := []struct {
		name        string
		x, w        []int
		stride, dil []int
		pads        [][2]int
		outPad      []int
		groups      int
	}{
		{"stride 2", []int{1, 2, 3, 4}, []int{2, 3, 3, 3}, []int{2, 2}, []int{1, 1}, [][2]int{{1, 1}, {1, 1}}, []int{1, 1}, 1},
		{"groups", []int{2, 4, 3, 3}, []int{4, 1, 2, 2}, []int{2, 2}, []int{1, 1}, [][2]int{{0, 0}, {0, 0}}, []int{0, 0}, 2},
		{"rectangular", []int{1, 2, 3, 3}, []int{2, 2, 3, 2}, []int{1, 3}, []int{2, 1}, [][2]int{{1, 0}, {0, 1}}, []int{1, 2}, 1},
	}
	for i, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			x, w := random(uint64(80+i), tc.x...), random(uint64(90+i), tc.w...)
			p := []int{tc.pads[0][0], tc.pads[1][0], tc.pads[0][1], tc.pads[1][1]}
			g := newGraph().input("x", Float, anyDims(tc.x)...).init("w", tc.w, w.f32()).
				node("ConvTranspose", []string{"x", "w", "->", "y"}, aInts("strides", ints64(tc.stride...)...), aInts("dilations", ints64(tc.dil...)...),
					aInts("pads", ints64(p...)...), aInts("output_padding", ints64(tc.outPad...)...), aInt("group", int64(tc.groups))).output("y")
			got, shape := runOne(t, g, map[string]*tensor.Tensor{"x": x.tensor()}, "y")
			expect(t, got, shape, convTransposeRef(x, w, tc.stride, tc.dil, tc.pads, tc.outPad, tc.groups), 1e-5)
		})
	}
}

// poolRef pools x, [N, C, H, W], over windows k with stride s and pads,
// taking the maximum or the mean of the elements a window covers within
// [lo, hi) of each axis, which the padding extends when it counts.
func poolRef(x *arr, k, s []int, pads [][2]int, out []int, isMax, includePad bool) *arr {
	n, c, h, w := x.shape[0], x.shape[1], x.shape[2], x.shape[3]
	y := newArr(n, c, out[0], out[1])
	for b := range n {
		for ch := range c {
			for i := range out[0] {
				for j := range out[1] {
					acc, cnt := math.Inf(-1), 0
					if !isMax {
						acc = 0
					}
					for p := range k[0] {
						for q := range k[1] {
							r, cc := i*s[0]-pads[0][0]+p, j*s[1]-pads[1][0]+q
							inPad := r >= -pads[0][0] && r < h+pads[0][1] && cc >= -pads[1][0] && cc < w+pads[1][1]
							if r < 0 || cc < 0 || r >= h || cc >= w {
								if includePad && inPad {
									cnt++
								}
								continue
							}
							cnt++
							if isMax {
								acc = max(acc, x.at(b, ch, r, cc))
							} else {
								acc += x.at(b, ch, r, cc)
							}
						}
					}
					if !isMax {
						acc /= float64(cnt)
					}
					y.data[y.off(b, ch, i, j)] = acc
				}
			}
		}
	}
	return y
}

func TestPool(t *testing.T) {
	x := random(100, 2, 3, 7, 6)
	cases := []struct {
		name       string
		op         string
		k, s       []int
		pads       [][2]int
		out        []int
		ceil       bool
		includePad bool
	}{
		{"max", "MaxPool", []int{2, 2}, []int{2, 2}, [][2]int{{0, 0}, {0, 0}}, []int{3, 3}, false, false},
		{"max padded", "MaxPool", []int{3, 3}, []int{2, 2}, [][2]int{{1, 1}, {1, 1}}, []int{4, 3}, false, false},
		{"max ceil", "MaxPool", []int{3, 2}, []int{2, 2}, [][2]int{{0, 1}, {0, 0}}, []int{4, 3}, true, false},
		{"average", "AveragePool", []int{3, 3}, []int{1, 1}, [][2]int{{1, 1}, {1, 1}}, []int{7, 6}, false, false},
		{"average with pads", "AveragePool", []int{3, 3}, []int{2, 2}, [][2]int{{1, 1}, {1, 1}}, []int{4, 3}, false, true},
		{"average ceil", "AveragePool", []int{2, 2}, []int{2, 2}, [][2]int{{0, 0}, {0, 0}}, []int{4, 3}, true, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			attrs := []*pb{aInts("kernel_shape", ints64(tc.k...)...), aInts("strides", ints64(tc.s...)...),
				aInts("pads", ints64(tc.pads[0][0], tc.pads[1][0], tc.pads[0][1], tc.pads[1][1])...)}
			if tc.ceil {
				attrs = append(attrs, aInt("ceil_mode", 1))
			}
			if tc.includePad {
				attrs = append(attrs, aInt("count_include_pad", 1))
			}
			g := newGraph().input("x", Float, 2, 3, 7, 6).node(tc.op, []string{"x", "->", "y"}, attrs...).output("y")
			got, shape := runOne(t, g, map[string]*tensor.Tensor{"x": x.tensor()}, "y")
			expect(t, got, shape, poolRef(x, tc.k, tc.s, tc.pads, tc.out, tc.op == "MaxPool", tc.includePad), 1e-5)
		})
	}
	t.Run("global", func(t *testing.T) {
		g := newGraph().input("x", Float, 2, 3, 7, 6).node("GlobalAveragePool", []string{"x", "->", "y"}).
			node("GlobalMaxPool", []string{"x", "->", "z"}).output("y").output("z")
		got, shape := runOne(t, g, map[string]*tensor.Tensor{"x": x.tensor()}, "y")
		expect(t, got, shape, poolRef(x, []int{7, 6}, []int{1, 1}, make([][2]int, 2), []int{1, 1}, false, false), 1e-5)
		got, shape = runOne(t, g, map[string]*tensor.Tensor{"x": x.tensor()}, "z")
		expect(t, got, shape, poolRef(x, []int{7, 6}, []int{1, 1}, make([][2]int, 2), []int{1, 1}, true, false), 1e-6)
	})
}

func TestNormalization(t *testing.T) {
	x := random(110, 2, 3, 4, 5)
	scale, bias, mean := random(111, 3), random(112, 3), random(113, 3)
	vr := mapArr(random(114, 3), func(v float64) float64 { return v*v + 0.5 })
	t.Run("BatchNormalization", func(t *testing.T) {
		g := newGraph().input("x", Float, 2, 3, 4, 5).init("s", []int{3}, scale.f32()).init("b", []int{3}, bias.f32()).
			init("m", []int{3}, mean.f32()).init("v", []int{3}, vr.f32()).
			node("BatchNormalization", []string{"x", "s", "b", "m", "v", "->", "y"}, aFloat("epsilon", 1e-3)).output("y")
		got, shape := runOne(t, g, map[string]*tensor.Tensor{"x": x.tensor()}, "y")
		want := newArr(x.shape...)
		for i := range want.data {
			c := i / 20 % 3
			want.data[i] = (x.data[i]-mean.data[c])/math.Sqrt(vr.data[c]+float64(float32(1e-3)))*scale.data[c] + bias.data[c]
		}
		expect(t, got, shape, want, 1e-5)
	})
	t.Run("InstanceNormalization", func(t *testing.T) {
		g := newGraph().input("x", Float, 2, 3, 4, 5).init("s", []int{3}, scale.f32()).init("b", []int{3}, bias.f32()).
			node("InstanceNormalization", []string{"x", "s", "b", "->", "y"}).output("y")
		got, shape := runOne(t, g, map[string]*tensor.Tensor{"x": x.tensor()}, "y")
		want := newArr(x.shape...)
		for nc := range 6 {
			seg := x.data[nc*20 : nc*20+20]
			m, v := meanVar(seg)
			for j, e := range seg {
				want.data[nc*20+j] = (e-m)/math.Sqrt(v+float64(float32(1e-5)))*scale.data[nc%3] + bias.data[nc%3]
			}
		}
		expect(t, got, shape, want, 1e-4)
	})
	t.Run("LayerNormalization", func(t *testing.T) {
		w, b := random(115, 4, 5), random(116, 4, 5)
		g := newGraph().input("x", Float, 2, 3, 4, 5).init("w", []int{4, 5}, w.f32()).init("b", []int{4, 5}, b.f32()).
			node("LayerNormalization", []string{"x", "w", "b", "->", "y"}, aInt("axis", -2)).output("y")
		got, shape := runOne(t, g, map[string]*tensor.Tensor{"x": x.tensor()}, "y")
		want := newArr(x.shape...)
		for r := range 6 {
			seg := x.data[r*20 : r*20+20]
			m, v := meanVar(seg)
			for j, e := range seg {
				want.data[r*20+j] = (e-m)/math.Sqrt(v+float64(float32(1e-5)))*w.data[j] + b.data[j]
			}
		}
		expect(t, got, shape, want, 1e-4)
	})
}

func meanVar(s []float64) (m, v float64) {
	for _, x := range s {
		m += x
	}
	m /= float64(len(s))
	for _, x := range s {
		v += (x - m) * (x - m)
	}
	return m, v / float64(len(s))
}

func TestSoftmax(t *testing.T) {
	x := random(120, 2, 3, 4)
	for _, op := range []string{"Softmax", "LogSoftmax"} {
		for _, opset := range []int64{13, 11} {
			g := newGraph().input("x", Float, 2, 3, 4).node(op, []string{"x", "->", "y"}, aInt("axis", 1)).output("y")
			g.opset = opset
			got, shape := runOne(t, g, map[string]*tensor.Tensor{"x": x.tensor()}, "y")
			// Opset 13 normalizes axis 1 alone, opset 11 axes 1 and 2
			// together.
			want := newArr(x.shape...)
			for b := range 2 {
				for k := range 4 {
					var idx []int
					if opset == 13 {
						for j := range 3 {
							idx = append(idx, x.off(b, j, k))
						}
					} else if k == 0 {
						for j := range 12 {
							idx = append(idx, b*12+j)
						}
					}
					s := 0.0
					for _, i := range idx {
						s += math.Exp(x.data[i])
					}
					for _, i := range idx {
						want.data[i] = math.Exp(x.data[i]) / s
						if op == "LogSoftmax" {
							want.data[i] = x.data[i] - math.Log(s)
						}
					}
				}
			}
			expect(t, got, shape, want, 1e-5)
		}
	}
}

func TestReduce(t *testing.T) {
	x := random(130, 2, 3, 4)
	type red struct {
		op string
		f  func(s []float64) float64
	}
	sum := func(s []float64) float64 {
		t := 0.0
		for _, v := range s {
			t += v
		}
		return t
	}
	for _, r := range []red{
		{"ReduceSum", sum},
		{"ReduceMean", func(s []float64) float64 { return sum(s) / float64(len(s)) }},
		{"ReduceMax", func(s []float64) float64 { return slices.Max(s) }},
		{"ReduceMin", func(s []float64) float64 { return slices.Min(s) }},
		{"ReduceSumSquare", func(s []float64) float64 { return sum(mapSlice(s, func(v float64) float64 { return v * v })) }},
		{"ReduceL1", func(s []float64) float64 { return sum(mapSlice(s, math.Abs)) }},
		{"ReduceL2", func(s []float64) float64 {
			return math.Sqrt(sum(mapSlice(s, func(v float64) float64 { return v * v })))
		}},
		{"ReduceLogSumExp", func(s []float64) float64 { return math.Log(sum(mapSlice(s, math.Exp))) }},
	} {
		for _, keep := range []int64{0, 1} {
			g := newGraph().input("x", Float, 2, 3, 4).initInts("axes", []int{2}, []int64{0, -1}).
				node(r.op, []string{"x", "axes", "->", "y"}, aInt("keepdims", keep)).output("y")
			g.opset = 18
			got, shape := runOne(t, g, map[string]*tensor.Tensor{"x": x.tensor()}, "y")
			want := newArr(3)
			if keep == 1 {
				want = newArr(1, 3, 1)
			}
			for j := range 3 {
				var s []float64
				for b := range 2 {
					for k := range 4 {
						s = append(s, x.at(b, j, k))
					}
				}
				want.data[j] = r.f(s)
			}
			expect(t, got, shape, want, 1e-5)
		}
	}
	// ReduceMean of opset 13 takes its axes from an attribute.
	g := newGraph().input("x", Float, 2, 3, 4).node("ReduceMean", []string{"x", "->", "y"}, aInts("axes", 1), aInt("keepdims", 0)).output("y")
	g.opset = 13
	got, shape := runOne(t, g, map[string]*tensor.Tensor{"x": x.tensor()}, "y")
	want := newArr(2, 4)
	for b := range 2 {
		for k := range 4 {
			want.data[b*4+k] = (x.at(b, 0, k) + x.at(b, 1, k) + x.at(b, 2, k)) / 3
		}
	}
	expect(t, got, shape, want, 1e-6)
	// ArgMax and ArgMin.
	for _, op := range []string{"ArgMax", "ArgMin"} {
		g := newGraph().input("x", Float, 2, 3, 4).node(op, []string{"x", "->", "y"}, aInt("axis", 2), aInt("keepdims", 0)).output("y")
		got, shape := runOne(t, g, map[string]*tensor.Tensor{"x": x.tensor()}, "y")
		want := newArr(2, 3)
		for i := range 6 {
			s := x.data[i*4 : i*4+4]
			if op == "ArgMax" {
				want.data[i] = float64(slices.Index(s, slices.Max(s)))
			} else {
				want.data[i] = float64(slices.Index(s, slices.Min(s)))
			}
		}
		expect(t, got, shape, want, 0)
	}
}

func TestMovement(t *testing.T) {
	x := random(140, 2, 3, 4)
	in := map[string]*tensor.Tensor{"x": x.tensor()}
	g := func() *graph { return newGraph().input("x", Float, 2, 3, 4) }
	t.Run("Transpose", func(t *testing.T) {
		got, shape := runOne(t, g().node("Transpose", []string{"x", "->", "y"}, aInts("perm", 1, 2, 0)).output("y"), in, "y")
		want := newArr(3, 4, 2)
		for i := range 3 {
			for j := range 4 {
				for k := range 2 {
					want.data[want.off(i, j, k)] = x.at(k, i, j)
				}
			}
		}
		expect(t, got, shape, want, 0)
	})
	t.Run("Slice", func(t *testing.T) {
		gr := g().initInts("st", []int{2}, ints64(-1, 0)).initInts("en", []int{2}, []int64{-100, math.MaxInt64}).
			initInts("ax", []int{2}, ints64(2, 1)).initInts("sp", []int{2}, ints64(-2, 2)).
			node("Slice", []string{"x", "st", "en", "ax", "sp", "->", "y"}).output("y")
		got, shape := runOne(t, gr, in, "y")
		want := newArr(2, 2, 2)
		for b := range 2 {
			for j, jj := range []int{0, 2} {
				for k, kk := range []int{3, 1} {
					want.data[want.off(b, j, k)] = x.at(b, jj, kk)
				}
			}
		}
		expect(t, got, shape, want, 0)
	})
	t.Run("Gather", func(t *testing.T) {
		gr := g().initInts("i", []int{2, 2}, ints64(-1, 0, 2, 2)).node("Gather", []string{"x", "i", "->", "y"}, aInt("axis", 2)).output("y")
		got, shape := runOne(t, gr, in, "y")
		want := newArr(2, 3, 2, 2)
		for b := range 2 {
			for j := range 3 {
				for p, k := range []int{3, 0, 2, 2} {
					want.data[want.off(b, j, p/2, p%2)] = x.at(b, j, k)
				}
			}
		}
		expect(t, got, shape, want, 0)
	})
	t.Run("Gather device indices", func(t *testing.T) {
		gr := newGraph().input("x", Float, 2, 3, 4).input("i", Int64, 3).node("Gather", []string{"x", "i", "->", "y"}, aInt("axis", 1)).output("y")
		got, shape := runOne(t, gr, map[string]*tensor.Tensor{"x": x.tensor(), "i": tensor.FromSlice([]int32{2, -3, 1}, 3)}, "y")
		want := newArr(2, 3, 4)
		for b := range 2 {
			for p, j := range []int{2, 0, 1} {
				for k := range 4 {
					want.data[want.off(b, p, k)] = x.at(b, j, k)
				}
			}
		}
		expect(t, got, shape, want, 0)
	})
	t.Run("GatherElements", func(t *testing.T) {
		gr := newGraph().input("x", Float, 2, 3, 4).initInts("i", []int{2, 1, 4}, ints64(0, 2, -1, 1, 1, 0, 0, 2)).
			node("GatherElements", []string{"x", "i", "->", "y"}, aInt("axis", 1)).output("y")
		got, shape := runOne(t, gr, in, "y")
		idx := []int{0, 2, 2, 1, 1, 0, 0, 2}
		want := newArr(2, 1, 4)
		for b := range 2 {
			for k := range 4 {
				want.data[b*4+k] = x.at(b, idx[b*4+k], k)
			}
		}
		expect(t, got, shape, want, 0)
	})
	t.Run("Split and Concat", func(t *testing.T) {
		gr := g().initInts("sp", []int{2}, ints64(1, 3)).
			node("Split", []string{"x", "sp", "->", "a", "b"}, aInt("axis", 2)).
			node("Concat", []string{"b", "a", "->", "y"}, aInt("axis", -1)).output("y")
		got, shape := runOne(t, gr, in, "y")
		want := newArr(2, 3, 4)
		for i := range 6 {
			copy(want.data[i*4:], x.data[i*4+1:i*4+4])
			want.data[i*4+3] = x.data[i*4]
		}
		expect(t, got, shape, want, 0)
		eq := g().node("Split", []string{"x", "->", "a", "b", "c"}, aInt("axis", 1)).output("c")
		got, shape = runOne(t, eq, in, "c")
		expect(t, got, shape, &arr{shape: []int{2, 1, 4}, data: slices.Concat(x.data[8:12], x.data[20:24])}, 0)
	})
	t.Run("Squeeze and Unsqueeze", func(t *testing.T) {
		gr := g().initInts("a", []int{2}, ints64(0, -1)).initInts("s", []int{1}, ints64(4)).
			node("Unsqueeze", []string{"x", "a", "->", "u"}).node("Squeeze", []string{"u", "s", "->", "y"}).output("y")
		got, shape := runOne(t, gr, in, "y")
		expect(t, got, shape, &arr{shape: []int{1, 2, 3, 4}, data: x.data}, 0)
		got, shape = runOne(t, g().node("Flatten", []string{"x", "->", "y"}, aInt("axis", 2)).output("y"), in, "y")
		expect(t, got, shape, &arr{shape: []int{6, 4}, data: x.data}, 0)
	})
	t.Run("Expand and Tile", func(t *testing.T) {
		v := random(141, 3, 1)
		gr := newGraph().input("v", Float, 3, 1).initInts("s", []int{3}, ints64(2, 1, 4)).initInts("r", []int{2}, ints64(2, 3)).
			node("Expand", []string{"v", "s", "->", "e"}).node("Tile", []string{"v", "r", "->", "t"}).output("e").output("t")
		got, shape := runOne(t, gr, map[string]*tensor.Tensor{"v": v.tensor()}, "e")
		want := newArr(2, 3, 4)
		for i := range want.data {
			want.data[i] = v.data[i/4%3]
		}
		expect(t, got, shape, want, 0)
		got, shape = runOne(t, gr, map[string]*tensor.Tensor{"v": v.tensor()}, "t")
		want = newArr(6, 3)
		for i := range 6 {
			for j := range 3 {
				want.data[i*3+j] = v.data[i%3]
			}
		}
		expect(t, got, shape, want, 0)
	})
	t.Run("Pad", func(t *testing.T) {
		for _, mode := range []string{"constant", "edge", "reflect"} {
			gr := g().initInts("p", []int{6}, ints64(0, 1, 2, 0, -1, 1)).init("v", []int{}, []float32{7}).
				node("Pad", []string{"x", "p", "v", "->", "y"}, aString("mode", mode)).output("y")
			got, shape := runOne(t, gr, in, "y")
			want := newArr(2, 3, 7)
			for b := range 2 {
				for j := range 3 {
					for k := range 7 {
						jj, kk := j-1, k-2
						switch mode {
						case "constant":
							if jj < 0 || kk < 0 || kk >= 4 {
								want.data[want.off(b, j, k)] = 7
								continue
							}
						case "edge":
							jj, kk = max(jj, 0), min(max(kk, 0), 3)
						case "reflect":
							jj, kk = abs(jj), abs(kk)
							if kk > 3 {
								kk = 6 - kk
							}
						}
						want.data[want.off(b, j, k)] = x.at(b, jj, kk)
					}
				}
			}
			expect(t, got, shape, want, 0)
		}
	})
	t.Run("Where", func(t *testing.T) {
		y := random(142, 3, 4)
		gr := g().input("y", Float, 3, 4).init("zero", []int{}, []float32{0}).
			node("Greater", []string{"x", "zero", "->", "c"}).node("Where", []string{"c", "x", "y", "->", "z"}).output("z")
		got, shape := runOne(t, gr, map[string]*tensor.Tensor{"x": x.tensor(), "y": y.tensor()}, "z")
		want := newArr(2, 3, 4)
		for i := range want.data {
			want.data[i] = y.data[i%12]
			if x.data[i] > 0 {
				want.data[i] = x.data[i]
			}
		}
		expect(t, got, shape, want, 0)
	})
	t.Run("Trilu", func(t *testing.T) {
		for _, upper := range []int64{0, 1} {
			gr := g().initInts("k", []int{}, ints64(1)).node("Trilu", []string{"x", "k", "->", "y"}, aInt("upper", upper)).output("y")
			got, shape := runOne(t, gr, in, "y")
			want := newArr(2, 3, 4)
			for i := range want.data {
				r, c := i/4%3, i%4
				if upper == 1 && c-r >= 1 || upper == 0 && c-r <= 1 {
					want.data[i] = x.data[i]
				}
			}
			expect(t, got, shape, want, 0)
		}
	})
	t.Run("CumSum", func(t *testing.T) {
		for _, mode := range [][2]int64{{0, 0}, {1, 0}, {0, 1}, {1, 1}} {
			gr := g().initInts("a", []int{}, ints64(-1)).node("CumSum", []string{"x", "a", "->", "y"}, aInt("exclusive", mode[0]), aInt("reverse", mode[1])).output("y")
			got, shape := runOne(t, gr, in, "y")
			want := newArr(2, 3, 4)
			for r := range 6 {
				for k := range 4 {
					s := 0.0
					for j := range 4 {
						if (mode[1] == 0 && j < k || mode[1] == 1 && j > k) || mode[0] == 0 && j == k {
							s += x.data[r*4+j]
						}
					}
					want.data[r*4+k] = s
				}
			}
			expect(t, got, shape, want, 1e-6)
		}
	})
}

func abs(x int) int { return max(x, -x) }

// TestHostShapes checks operators on host values of the kind a graph
// computes for shapes: Range, ConstantOfShape, Cast, Equal, Where, Mul
// and Expand in the manner of an exported causal mask.
func TestHostShapes(t *testing.T) {
	gr := newGraph().input("x", Float, "n", 4).
		initInts("zero", []int{}, ints64(0)).initInts("one", []int{}, ints64(1)).initInts("one1", []int{1}, ints64(1)).
		node("Shape", []string{"x", "->", "s"}).
		node("Gather", []string{"s", "zero", "->", "n"}).
		node("Range", []string{"zero", "n", "one", "->", "r"}).
		node("Unsqueeze", []string{"r", "one1", "->", "col"}).
		node("Less", []string{"col", "r", "->", "lt"}).
		node("Cast", []string{"lt", "->", "ltf"}, aInt("to", int64(Float))).
		node("ConstantOfShape", []string{"s", "->", "fill"}, aTensor("value", tensorFloats("", []int{1}, []float32{-2}))).
		node("Mul", []string{"ltf", "fill", "->", "m"}).
		node("Add", []string{"x", "m", "->", "y"}).output("y")
	x := random(150, 4, 4)
	got, shape := runOne(t, gr, map[string]*tensor.Tensor{"x": x.tensor()}, "y")
	want := newArr(4, 4)
	for i := range 4 {
		for j := range 4 {
			// Unsqueeze makes r a column, so the mask is i < j.
			want.data[i*4+j] = x.at(i, j)
			if i < j {
				want.data[i*4+j] -= 2
			}
		}
	}
	expect(t, got, shape, want, 1e-6)
}

func TestCast(t *testing.T) {
	x := &arr{shape: []int{5}, data: []float64{-2.7, -0.5, 0, 0.5, 3.9}}
	gr := newGraph().input("x", Float, 5).
		node("Cast", []string{"x", "->", "i"}, aInt("to", int64(Int64))).
		node("Cast", []string{"x", "->", "b"}, aInt("to", int64(Bool))).
		node("Cast", []string{"b", "->", "bf"}, aInt("to", int64(Float))).
		node("Not", []string{"b", "->", "nb"}).
		node("Cast", []string{"x", "->", "h"}, aInt("to", int64(Float16))).
		output("i").output("bf").output("nb").output("h")
	m := open(t, gr)
	res, err := m.Run(context.Background(), map[string]*tensor.Tensor{"x": x.tensor()})
	if err != nil {
		t.Fatal(err)
	}
	ints := func(name string) []int32 {
		d, err := res[name].Data[int32](context.Background())
		if err != nil {
			t.Fatal(err)
		}
		return d
	}
	if got := ints("i"); !slices.Equal(got, []int32{-2, 0, 0, 0, 3}) {
		t.Errorf("int64 cast %v", got)
	}
	if got := ints("nb"); !slices.Equal(got, []int32{0, 0, 1, 0, 0}) {
		t.Errorf("not %v", got)
	}
	bf, _ := res["bf"].Data[float32](context.Background())
	if !slices.Equal(bf, []float32{1, 1, 0, 1, 1}) {
		t.Errorf("bool cast %v", bf)
	}
	if res["h"].DType() != tensor.Float16 {
		t.Errorf("float16 cast to %v", res["h"].DType())
	}
}

func TestResize(t *testing.T) {
	x := random(160, 1, 2, 3, 4)
	cases := []struct {
		name, mode, coord, nearest string
		sizes                      []int
		ref                        func(o, scale float64, in, out int) float64 // source coordinate
	}{
		{"nearest 2x", "nearest", "asymmetric", "floor", []int{6, 8}, func(o, s float64, _, _ int) float64 { return o / s }},
		{"nearest half pixel", "nearest", "half_pixel", "round_prefer_floor", []int{5, 7}, func(o, s float64, _, _ int) float64 { return (o+0.5)/s - 0.5 }},
		{"linear half pixel", "linear", "half_pixel", "", []int{6, 7}, func(o, s float64, _, _ int) float64 { return (o+0.5)/s - 0.5 }},
		{"linear align corners", "linear", "align_corners", "", []int{5, 6}, func(o, _ float64, in, out int) float64 {
			return o * float64(in-1) / float64(out-1)
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			attrs := []*pb{aString("mode", tc.mode), aString("coordinate_transformation_mode", tc.coord)}
			if tc.nearest != "" {
				attrs = append(attrs, aString("nearest_mode", tc.nearest))
			}
			gr := newGraph().input("x", Float, 1, 2, 3, 4).initInts("sz", []int{4}, ints64(1, 2, tc.sizes[0], tc.sizes[1])).
				node("Resize", []string{"x", "", "", "sz", "->", "y"}, attrs...).output("y")
			got, shape := runOne(t, gr, map[string]*tensor.Tensor{"x": x.tensor()}, "y")
			oh, ow := tc.sizes[0], tc.sizes[1]
			src := func(o, in, out int) (lo, hi int, w float64) {
				f := tc.ref(float64(o), float64(out)/float64(in), in, out)
				if tc.mode == "nearest" {
					var j float64
					if tc.nearest == "floor" {
						j = math.Floor(f)
					} else if f == math.Floor(f)+0.5 {
						j = math.Floor(f)
					} else {
						j = math.Round(f)
					}
					j = min(max(j, 0), float64(in-1))
					return int(j), int(j), 0
				}
				f = min(max(f, 0), float64(in-1))
				lo = int(math.Floor(f))
				return lo, min(lo+1, in-1), f - float64(lo)
			}
			want := newArr(1, 2, oh, ow)
			for c := range 2 {
				for i := range oh {
					i0, i1, wi := src(i, 3, oh)
					for j := range ow {
						j0, j1, wj := src(j, 4, ow)
						top := x.at(0, c, i0, j0)*(1-wj) + x.at(0, c, i0, j1)*wj
						bot := x.at(0, c, i1, j0)*(1-wj) + x.at(0, c, i1, j1)*wj
						want.data[want.off(0, c, i, j)] = top*(1-wi) + bot*wi
					}
				}
			}
			expect(t, got, shape, want, 1e-5)
		})
	}
}

// TestRunTwice runs a model twice, the second time from the schedule
// cache, with other inputs.
func TestRunTwice(t *testing.T) {
	w := random(170, 4, 3)
	gr := newGraph().input("x", Float, "n", 4).init("w", []int{4, 3}, w.f32()).
		node("MatMul", []string{"x", "w", "->", "h"}).node("Relu", []string{"h", "->", "y"}).output("y")
	m := open(t, gr)
	for i := range 3 {
		x := random(uint64(171+i), 2, 4)
		res, err := m.Run(context.Background(), map[string]*tensor.Tensor{"x": x.tensor()})
		if err != nil {
			t.Fatal(err)
		}
		got, err := res["y"].Data[float32](context.Background())
		if err != nil {
			t.Fatal(err)
		}
		want := newArr(2, 3)
		for r := range 2 {
			for c := range 3 {
				s := 0.0
				for k := range 4 {
					s += x.at(r, k) * w.at(k, c)
				}
				want.data[r*3+c] = max(s, 0)
			}
		}
		expect(t, got, res["y"].Shape(), want, 1e-6)
	}
}

func mapSlice(s []float64, f func(float64) float64) []float64 {
	out := make([]float64, len(s))
	for i, x := range s {
		out[i] = f(x)
	}
	return out
}
