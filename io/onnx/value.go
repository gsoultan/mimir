package onnx

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"slices"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/tensor"
)

// hostLimit bounds the elements of a value that is computed on the host.
// Shapes, indices and the scalars of a graph are computed there, once per
// run without touching a device; larger values are lazy tensors.
const hostLimit = 1 << 16

// value is a value of a graph: computed on the host, a lazy tensor, or
// both, as a constant on the host with a copy on a device.
type value struct {
	typ   DataType
	shape []int
	host  bool
	// On the host, the elements in row-major order: ints for the integer
	// types and bool, floats for the float types.
	ints   []int64
	floats []float64
	t      *tensor.Tensor // on a device; nil until needed for a host value
	// key names the value in the model's cache of device copies of host
	// values, "" for a value that is not cached.
	key      string
	constant bool // an initializer or a folded constant, which never changes
}

func hostInts(typ DataType, shape []int, ints []int64) *value {
	return &value{typ: typ, shape: shape, host: true, ints: ints}
}

func hostFloats(typ DataType, shape []int, floats []float64) *value {
	if typ != Double {
		for i, f := range floats {
			floats[i] = float64(float32(f))
		}
	}
	return &value{typ: typ, shape: shape, host: true, floats: floats}
}

func devValue(typ DataType, t *tensor.Tensor) *value {
	return &value{typ: typ, shape: t.Shape(), t: t}
}

// size returns the number of elements of v.
func (v *value) size() int {
	n := 1
	for _, d := range v.shape {
		n *= d
	}
	return n
}

// float returns element i of a host value as a float64.
func (v *value) float(i int) float64 {
	if v.typ.IsFloat() {
		return v.floats[i]
	}
	return float64(v.ints[i])
}

// int returns element i of a host value as an int64, truncating floats.
func (v *value) int(i int) int64 {
	if v.typ.IsFloat() {
		f := v.floats[i]
		if math.IsNaN(f) || f >= math.MaxInt64 || f <= math.MinInt64 {
			return 0
		}
		return int64(f)
	}
	return v.ints[i]
}

// uniform reports whether every element of a non-empty host value is the
// same, and returns it.
func (v *value) uniform() (float64, bool) {
	n := v.size()
	if n == 0 {
		return 0, false
	}
	if v.typ.IsFloat() {
		x := v.floats[0]
		for _, y := range v.floats[1:] {
			if y != x && !(math.IsNaN(x) && math.IsNaN(y)) {
				return 0, false
			}
		}
		return x, true
	}
	x := v.ints[0]
	for _, y := range v.ints[1:] {
		if y != x {
			return 0, false
		}
	}
	return float64(x), true
}

// sameHost reports whether host values a and b are equal, element for
// element, in type and shape.
func sameHost(a, b *value) bool {
	return a.typ == b.typ && slices.Equal(a.shape, b.shape) && slices.Equal(a.ints, b.ints) &&
		slices.EqualFunc(a.floats, b.floats, func(x, y float64) bool { return x == y || math.IsNaN(x) && math.IsNaN(y) })
}

// cacheKey names a device copy of a host value.
type cacheKey struct {
	name string
	dev  tensor.Device
}

// cached is a device copy of a host value.
type cached struct {
	v *value // the host value
	t *tensor.Tensor
}

// run evaluates the nodes of a graph on one device.
type run struct {
	ctx  context.Context
	m    *Model
	dev  tensor.Device
	env  map[string]*value
	fold bool // folding constants at load: every input is a constant
}

// get returns the value named name, or nil for "", the name of an absent
// optional input.
func (r *run) get(name string) (*value, error) {
	if name == "" {
		return nil, nil
	}
	if v, ok := r.env[name]; ok {
		return v, nil
	}
	if v, ok := r.m.consts[name]; ok {
		return v, nil
	}
	return nil, fmt.Errorf("%w: value %q is not defined", ErrFormat, name)
}

// tensor returns v as a tensor on r's device. A host value becomes a
// constant, which takes no memory, when its elements are all equal, and
// otherwise a tensor that the model keeps from one run to the next for
// as long as the value is the same.
func (r *run) tensor(v *value) (*tensor.Tensor, error) {
	if v.t != nil {
		if !v.constant || r.fold || v.t.Device() == r.dev {
			return v.t, v.t.Err()
		}
		return r.moved(v)
	}
	dt, ok := v.typ.DType()
	if !ok {
		return nil, fmt.Errorf("onnx: %s values: %w", v.typ, ErrUnsupported)
	}
	if x, ok := v.uniform(); ok {
		if !dt.IsFloat() && (x > math.MaxInt32 || x < math.MinInt32) {
			return nil, fmt.Errorf("onnx: %s value %v does not fit in an int32: %w", v.typ, x, tensor.ErrDType)
		}
		return tensor.Full(dt, x, v.shape...), nil
	}
	key := cacheKey{v.key, r.dev}
	if c, ok := r.m.cache[key]; ok && v.key != "" && (v.constant || sameHost(c.v, v)) {
		return c.t, nil
	}
	t, err := upload(r.dev, dt, v)
	if err != nil {
		return nil, err
	}
	if v.key != "" {
		r.m.cache[key] = &cached{v: v, t: t}
	}
	return t, nil
}

// moved returns a copy of a constant tensor v on r's device, made once.
func (r *run) moved(v *value) (*tensor.Tensor, error) {
	key := cacheKey{v.key, r.dev}
	if c, ok := r.m.cache[key]; ok {
		return c.t, nil
	}
	t := v.t.To(r.dev)
	if err := t.Realize(r.ctx); err != nil {
		return nil, err
	}
	r.m.cache[key] = &cached{v: v, t: t}
	return t, nil
}

// upload returns host value v as a realized tensor of dt on dev.
func upload(dev tensor.Device, dt tensor.DType, v *value) (*tensor.Tensor, error) {
	n := v.size()
	b := make([]byte, 4*n)
	if dt.IsFloat() {
		for i, f := range v.floats {
			binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(float32(f)))
		}
	} else {
		for i, x := range v.ints {
			if x > math.MaxInt32 || x < math.MinInt32 {
				return nil, fmt.Errorf("onnx: %s value %d does not fit in an int32: %w", v.typ, x, tensor.ErrDType)
			}
			binary.LittleEndian.PutUint32(b[4*i:], uint32(int32(x)))
		}
	}
	src := backend.Float32
	if !dt.IsFloat() {
		src = backend.Int32
	}
	t := tensor.FromBytes(src, b, v.shape...).To(dev)
	if dt != src {
		t = t.Cast(dt)
	}
	return t, t.Realize(context.Background())
}

// download returns the elements of a value as a host value, computing a
// tensor first. Values that steer the shapes of a graph are computed on
// the host as a rule; download serves graphs that compute them on a
// device all the same.
func (r *run) download(v *value) (*value, error) {
	if v.host {
		return v, nil
	}
	if err := v.t.Realize(r.ctx); err != nil {
		return nil, err
	}
	if v.typ.IsFloat() {
		f, err := v.t.Cast(tensor.Float32).Data[float32](r.ctx)
		if err != nil {
			return nil, err
		}
		out := make([]float64, len(f))
		for i, x := range f {
			out[i] = float64(x)
		}
		return &value{typ: v.typ, shape: v.shape, host: true, floats: out, t: v.t}, nil
	}
	d, err := v.t.Data[int32](r.ctx)
	if err != nil {
		return nil, err
	}
	out := make([]int64, len(d))
	for i, x := range d {
		out[i] = int64(x)
	}
	return &value{typ: v.typ, shape: v.shape, host: true, ints: out, t: v.t}, nil
}

// broadcastShape returns the NumPy broadcast of shapes.
func broadcastShape(shapes ...[]int) ([]int, error) {
	rank := 0
	for _, s := range shapes {
		rank = max(rank, len(s))
	}
	out := make([]int, rank)
	for i := range out {
		out[i] = 1
	}
	for _, s := range shapes {
		for i, d := range s {
			o := &out[rank-len(s)+i]
			switch {
			case d == *o || d == 1:
			case *o == 1:
				*o = d
			default:
				return nil, fmt.Errorf("onnx: shapes %v do not broadcast: %w", shapes, tensor.ErrShape)
			}
		}
	}
	return out, nil
}

// broadcastIndex returns, for each element of shape out in row-major
// order, the index of the element of in, a shape that broadcasts to out,
// that it reads.
func broadcastIndex(in, out []int) []int {
	n := 1
	for _, d := range out {
		n *= d
	}
	idx := make([]int, n)
	strides := make([]int, len(out))
	s := 1
	for i := len(in) - 1; i >= 0; i-- {
		if in[i] != 1 {
			strides[len(out)-len(in)+i] = s
		}
		s *= in[i]
	}
	pos := make([]int, len(out))
	off := 0
	for i := range idx {
		idx[i] = off
		for d := len(out) - 1; d >= 0; d-- {
			pos[d]++
			off += strides[d]
			if pos[d] < out[d] {
				break
			}
			off -= strides[d] * pos[d]
			pos[d] = 0
		}
	}
	return idx
}

// numElements returns the elements of shape, or false if a dimension is
// negative or the count passes limit.
func numElements(shape []int, limit int) (int, bool) {
	n := 1
	for _, d := range shape {
		if d < 0 {
			return 0, false
		}
		if d != 0 && n > limit/d {
			return 0, false
		}
		n *= d
	}
	return n, n <= limit
}
