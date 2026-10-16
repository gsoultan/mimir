package nn

import (
	"fmt"
	"reflect"
	"strings"
	"unsafe"

	"github.com/gsoultan/mimir/tensor"
)

// Replicate returns a copy of m on the device d, with parameters and
// buffers of its own holding the values of m's, as data-parallel training
// replicates a model onto each device of a group. The copy has m's
// structure: its structs, slices, maps and pointers are copied, sharing
// between them included, so the copy's NamedParams yields its parameters
// in the order and under the names that m's does. Tensors that m holds
// other than in parameters, such as a mask, are moved to d; the replica's
// layers start in m's mode and with its seeds, so its dropout draws the
// masks m's would. Gradients are not copied.
//
// Functions are shared, not copied, so a Func that holds a layer of m
// keeps using m's layer: build such modules from layers, as Sequential
// does, rather than closures over them.
func Replicate(m Module, d tensor.Device) (Module, error) {
	if m == nil {
		return nil, fmt.Errorf("nn: replicate a nil module: %w", tensor.ErrShape)
	}
	c := copier{dev: d, done: map[pointer]reflect.Value{}}
	src := reflect.ValueOf(m)
	dst := reflect.New(src.Type()).Elem()
	c.copy(dst, src)
	if c.err != nil {
		return nil, fmt.Errorf("nn: replicate on %v: %w", d, c.err)
	}
	return dst.Interface().(Module), nil
}

var tensorType = reflect.TypeFor[*tensor.Tensor]()

// copier deep-copies a module onto a device.
type copier struct {
	dev  tensor.Device
	done map[pointer]reflect.Value // the copy of each pointer copied so far
	err  error
}

// pointer is a pointer and its type: a struct and its first field share
// an address.
type pointer struct {
	at unsafe.Pointer
	t  reflect.Type
}

// copy sets dst, which is settable, to a copy of src of the same type.
func (c *copier) copy(dst, src reflect.Value) {
	if c.err != nil {
		return
	}
	switch src.Kind() {
	case reflect.Pointer:
		if src.IsNil() {
			return
		}
		ptr := pointer{src.UnsafePointer(), src.Type()}
		if v, ok := c.done[ptr]; ok {
			dst.Set(v)
			return
		}
		p := reflect.NewAt(src.Type().Elem(), ptr.at)
		var out reflect.Value
		switch {
		case p.Type() == paramType:
			out = reflect.ValueOf(c.param(p.Interface().(*tensor.Param)))
		case p.Type() == tensorType:
			t := p.Interface().(*tensor.Tensor)
			out = reflect.ValueOf(t.To(c.dev))
			if err := t.Err(); err == nil {
				c.err = out.Interface().(*tensor.Tensor).Err()
			}
		case fromTensor(p.Type().Elem()):
			// Other values of package tensor, such as a sparse Pattern,
			// serve every device.
			out = p
		default:
			out = reflect.New(p.Type().Elem())
			c.done[ptr] = out
			c.copy(out.Elem(), p.Elem())
		}
		c.done[ptr] = out
		dst.Set(out)
	case reflect.Interface:
		if src.IsNil() {
			return
		}
		e := src.Elem()
		v := reflect.New(e.Type()).Elem()
		c.copy(v, e)
		dst.Set(v)
	case reflect.Struct:
		if fromTensor(src.Type()) {
			dst.Set(src)
			return
		}
		if !src.CanAddr() {
			// Fields of a struct that is not addressable, such as a map's
			// value, can only be read through a copy that is.
			v := reflect.New(src.Type()).Elem()
			v.Set(src)
			src = v
		}
		for i := range src.NumField() {
			c.copy(field(dst, i), field(src, i))
		}
	case reflect.Slice:
		if src.IsNil() {
			return
		}
		// Slices of numbers, such as a graph's edges, are shared: no
		// module writes into them.
		if k := src.Type().Elem().Kind(); k <= reflect.Complex128 || k == reflect.String {
			dst.Set(src)
			return
		}
		v := reflect.MakeSlice(src.Type(), src.Len(), src.Len())
		for i := range src.Len() {
			c.copy(v.Index(i), src.Index(i))
		}
		dst.Set(v)
	case reflect.Array:
		for i := range src.Len() {
			c.copy(dst.Index(i), src.Index(i))
		}
	case reflect.Map:
		if src.IsNil() {
			return
		}
		v := reflect.MakeMapWithSize(src.Type(), src.Len())
		for it := src.MapRange(); it.Next(); {
			e := reflect.New(src.Type().Elem()).Elem()
			c.copy(e, it.Value())
			v.SetMapIndex(it.Key(), e)
		}
		dst.Set(v)
	default:
		// Numbers, strings, functions and channels.
		dst.Set(src)
	}
}

// param returns a copy of p on the device, trainable if p is.
func (c *copier) param(p *tensor.Param) *tensor.Param {
	v := p.Value().Detach().To(c.dev)
	if err := v.Err(); err != nil {
		c.err = err
	}
	np := tensor.NewBuffer(v)
	if p.Trainable() {
		np = tensor.NewParam(v)
	}
	if err := np.Value().Err(); err != nil && c.err == nil {
		c.err = err
	}
	return np
}

// field returns field i of the struct v, which is addressable, settable
// even if the field is unexported.
func field(v reflect.Value, i int) reflect.Value {
	f := v.Field(i)
	return reflect.NewAt(f.Type(), unsafe.Pointer(f.UnsafeAddr())).Elem()
}

// fromTensor reports whether t is a type of package tensor.
func fromTensor(t reflect.Type) bool {
	return strings.HasSuffix(t.PkgPath(), "/mimir/tensor")
}
