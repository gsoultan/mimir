// Package nn builds neural networks from layers, like torch.nn.
//
// A Module maps a tensor to a tensor. Layers with weights, such as Linear
// and Conv2d, hold them in exported *tensor.Param fields; stateless layers,
// such as ReLU and MaxPool2d, are Funcs. Custom modules are ordinary
// structs: Parameters finds every *tensor.Param a module holds, in its
// fields, slices and maps and in the modules it contains, whether the
// fields are exported or not.
//
//	model := nn.NewSequential(
//		nn.NewConv2d(1, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
//		nn.Flatten(), nn.NewLinear(32*13*13, 10),
//	)
//	logits := model.Forward(x)
//
// New layers start in training mode; Eval switches dropout and batch
// normalization to inference behavior and Train switches them back.
// Weights are initialized like PyTorch's defaults from a seeded
// generator, so a program builds the same model on every run; ManualSeed
// changes the seed.
package nn

import (
	"cmp"
	"context"
	"fmt"
	"io"
	"iter"
	"reflect"
	"slices"
	"strconv"
	"sync/atomic"
	"unsafe"

	"github.com/gsoultan/mimir/io/safetensors"
	"github.com/gsoultan/mimir/tensor"
)

// Module is a layer or a network of layers.
type Module interface {
	Forward(x *tensor.Tensor) *tensor.Tensor
}

// Func adapts a function to a Module. Layers without state are Funcs.
type Func func(x *tensor.Tensor) *tensor.Tensor

// Forward returns f(x).
func (f Func) Forward(x *tensor.Tensor) *tensor.Tensor { return f(x) }

// Parameters returns the trainable parameters of m, in the order
// NamedParams visits them.
func Parameters(m Module) []*tensor.Param {
	var ps []*tensor.Param
	for _, p := range NamedParams(m) {
		if p.Trainable() {
			ps = append(ps, p)
		}
	}
	return ps
}

// NamedParams yields every parameter and buffer of m with its path, such
// as "Layers.0.Weight". A parameter reachable by several paths is yielded
// once, under the first.
func NamedParams(m Module) iter.Seq2[string, *tensor.Param] {
	return func(yield func(string, *tensor.Param) bool) {
		w := walker{seen: map[unsafe.Pointer]bool{}}
		w.visit(reflect.ValueOf(m), "", func(name string, v reflect.Value) bool {
			if p, ok := v.Interface().(*tensor.Param); ok {
				return yield(name, p)
			}
			return true
		})
	}
}

// trainer is a module whose behavior differs between training and
// inference.
type trainer interface {
	SetTraining(on bool)
}

// Train puts every module in m in training mode.
func Train(m Module) { setTraining(m, true) }

// Eval puts every module in m in inference mode: dropout passes inputs
// through and batch normalization uses its running statistics.
func Eval(m Module) { setTraining(m, false) }

func setTraining(m Module, on bool) {
	w := walker{seen: map[unsafe.Pointer]bool{}}
	w.visit(reflect.ValueOf(m), "", func(_ string, v reflect.Value) bool {
		if t, ok := v.Interface().(trainer); ok {
			t.SetTraining(on)
		}
		return true
	})
}

var (
	paramType   = reflect.TypeFor[*tensor.Param]()
	trainerType = reflect.TypeFor[trainer]()
)

// walker visits the pointers reachable from a module through struct fields,
// slices, arrays, maps and interfaces.
type walker struct {
	seen map[unsafe.Pointer]bool
}

// visit calls fn with each non-nil pointer that is a *tensor.Param or
// implements trainer, and its path. It stops when fn returns false.
func (w *walker) visit(v reflect.Value, path string, fn func(string, reflect.Value) bool) bool {
	switch v.Kind() {
	case reflect.Interface:
		if v.IsNil() {
			return true
		}
		return w.visit(v.Elem(), path, fn)
	case reflect.Pointer:
		if v.IsNil() {
			return true
		}
		ptr := v.UnsafePointer()
		if w.seen[ptr] {
			return true
		}
		w.seen[ptr] = true
		// Rebuild the pointer so values reached through unexported fields
		// can be used.
		v = reflect.NewAt(v.Type().Elem(), ptr)
		if v.Type() == paramType {
			return fn(path, v)
		}
		if v.Type().Implements(trainerType) && !fn(path, v) {
			return false
		}
		return w.visit(v.Elem(), path, fn)
	case reflect.Struct:
		for i := range v.NumField() {
			f := v.Field(i)
			if v.Type().Field(i).Tag.Get("nn") == "-" {
				continue
			}
			if !f.CanInterface() {
				if !f.CanAddr() {
					continue
				}
				f = reflect.NewAt(f.Type(), unsafe.Pointer(f.UnsafeAddr())).Elem()
			}
			if !w.visit(f, join(path, v.Type().Field(i).Name), fn) {
				return false
			}
		}
	case reflect.Slice, reflect.Array:
		for i := range v.Len() {
			if !w.visit(v.Index(i), join(path, strconv.Itoa(i)), fn) {
				return false
			}
		}
	case reflect.Map:
		if v.Type().Key().Kind() != reflect.String {
			return true
		}
		keys := v.MapKeys()
		slices.SortFunc(keys, func(a, b reflect.Value) int { return cmp.Compare(a.String(), b.String()) })
		for _, k := range keys {
			if !w.visit(v.MapIndex(k), join(path, k.String()), fn) {
				return false
			}
		}
	}
	return true
}

func join(path, name string) string {
	if path == "" {
		return name
	}
	return path + "." + name
}

// Save writes every parameter and buffer of m to w in the safetensors
// format, named by NamedParams.
func Save(ctx context.Context, w io.Writer, m Module) error {
	ts := map[string]safetensors.Tensor{}
	for name, p := range NamedParams(m) {
		t, err := Encode(ctx, p.Value())
		if err != nil {
			return fmt.Errorf("nn: save %s: %w", name, err)
		}
		ts[name] = t
	}
	return safetensors.Write(w, ts, nil)
}

// Load sets every parameter and buffer of m from f, which must hold each
// of them with the same shape and dtype, as Save writes them.
func Load(f *safetensors.File, m Module) error {
	return LoadPrefixed(f, "", m)
}

// LoadPrefixed is Load for files that name m's parameters prefix+name.
func LoadPrefixed(f *safetensors.File, prefix string, m Module) error {
	for name, p := range NamedParams(m) {
		st, ok := f.Tensor(prefix + name)
		if !ok {
			return fmt.Errorf("nn: load: no tensor %q: %w", prefix+name, tensor.ErrShape)
		}
		v, err := Decode(st)
		if err != nil {
			return fmt.Errorf("nn: load %s: %w", name, err)
		}
		if err := p.Set(v); err != nil {
			return fmt.Errorf("nn: load %s: %w", name, err)
		}
	}
	return nil
}

// Encode computes t and returns it as a safetensors tensor.
func Encode(ctx context.Context, t *tensor.Tensor) (safetensors.Tensor, error) {
	switch t.DType() {
	case tensor.Float32:
		v, err := t.Data[float32](ctx)
		return safetensors.Float32Tensor(t.Shape(), v), err
	case tensor.Int32:
		v, err := t.Data[int32](ctx)
		return safetensors.Int32Tensor(t.Shape(), v), err
	}
	return safetensors.Tensor{}, t.Err()
}

// Decode returns a safetensors tensor of dtype F32 or I32 as a tensor.
func Decode(st safetensors.Tensor) (*tensor.Tensor, error) {
	switch st.DType {
	case "F32":
		v, err := st.Float32s()
		if err != nil {
			return nil, err
		}
		return tensor.FromSlice(v, st.Shape...), nil
	case "I32":
		v, err := st.Int32s()
		if err != nil {
			return nil, err
		}
		return tensor.FromSlice(v, st.Shape...), nil
	}
	return nil, fmt.Errorf("nn: dtype %s: %w", st.DType, tensor.ErrDType)
}

// seeds hands out initialization seeds: a base set by ManualSeed and a
// counter of seeds taken since.
var seeds struct {
	base, next atomic.Uint64
}

// ManualSeed restarts the sequence of seeds that new layers initialize
// their weights from, like torch.manual_seed. Layers built in the same
// order after the same ManualSeed have the same weights.
func ManualSeed(seed uint64) {
	seeds.base.Store(seed)
	seeds.next.Store(0)
}

// nextSeed returns the next initialization seed.
func nextSeed() uint64 {
	return splitmix(seeds.base.Load() ^ splitmix(seeds.next.Add(1)))
}

// splitmix scrambles x so that nearby seeds give unrelated streams.
func splitmix(x uint64) uint64 {
	x += 0x9E3779B97F4A7C15
	x = (x ^ x>>30) * 0xBF58476D1CE4E5B9
	x = (x ^ x>>27) * 0x94D049BB133111EB
	return x ^ x>>31
}

// Option configures a layer. Options a layer does not use are ignored.
type Option func(*options)

type options struct {
	conv     tensor.ConvOptions
	noBias   bool
	eps      float64
	momentum float64
	seed     *uint64
}

func resolve(opts []Option, base options) options {
	for _, o := range opts {
		o(&base)
	}
	return base
}

// NoBias leaves out a Linear or Conv2d layer's bias, or a normalization
// layer's learned scale and shift.
func NoBias() Option { return func(o *options) { o.noBias = true } }

// Stride sets the step between windows of a convolution or pooling layer.
func Stride(n int) Option { return func(o *options) { o.conv.Stride = n } }

// Padding pads both spatial dimensions of a convolution or pooling layer's
// input by n on every side.
func Padding(n int) Option { return func(o *options) { o.conv.Padding = n } }

// Dilation sets the spacing between a convolution kernel's elements.
func Dilation(n int) Option { return func(o *options) { o.conv.Dilation = n } }

// Groups splits a convolution's channels into n independent groups.
func Groups(n int) Option { return func(o *options) { o.conv.Groups = n } }

// Eps sets the value a normalization layer adds to the variance.
func Eps(eps float64) Option { return func(o *options) { o.eps = eps } }

// Momentum sets how fast batch normalization's running statistics follow
// each batch: new = (1-m)*old + m*batch.
func Momentum(m float64) Option { return func(o *options) { o.momentum = m } }

// Seed initializes a layer's weights from seed instead of the next seed
// after ManualSeed.
func Seed(seed uint64) Option { return func(o *options) { o.seed = &seed } }

func (o options) nextSeed() uint64 {
	if o.seed != nil {
		*o.seed = splitmix(*o.seed)
		return *o.seed
	}
	return nextSeed()
}

// uniform returns a parameter of shape drawn uniformly from [-bound, bound).
func uniform(o options, bound float64, shape ...int) *tensor.Param {
	r := tensor.Rand(o.nextSeed(), shape...)
	return tensor.NewParam(r.Mul(scalar(2 * bound)).Sub(scalar(bound)))
}

func scalar(v float64) *tensor.Tensor { return tensor.Full(tensor.Float32, v) }
