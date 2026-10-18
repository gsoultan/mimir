package mim

import (
	"encoding/binary"
	"fmt"
	"math"

	mimfile "github.com/gsoultan/mim"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/tensor"
)

// storage returns the mimir dtype that holds values of dt: the float and
// block types mimir computes with hold themselves, the 8-bit floats and
// the other block types are held decoded as Float32, and the integer
// types and bool as Int32. It reports false for f64.
func storage(dt mimfile.DType) (tensor.DType, bool) {
	switch dt {
	case mimfile.F32, mimfile.F8E4M3, mimfile.F8E5M2, mimfile.Q4_0, mimfile.MXFP4, mimfile.Affine4:
		return tensor.Float32, true
	case mimfile.F16:
		return tensor.Float16, true
	case mimfile.BF16:
		return tensor.BFloat16, true
	case mimfile.Q8_0:
		return tensor.Q8_0, true
	case mimfile.Q4_K:
		return tensor.Q4_K, true
	case mimfile.Q6_K:
		return tensor.Q6_K, true
	case mimfile.I8, mimfile.U8, mimfile.I16, mimfile.U16, mimfile.I32, mimfile.U32, mimfile.I64, mimfile.U64, mimfile.Bool:
		return tensor.Int32, true
	}
	return backend.Invalid, false
}

func isFloat(dt mimfile.DType) bool {
	switch dt {
	case mimfile.F32, mimfile.F16, mimfile.BF16, mimfile.F64, mimfile.F8E4M3, mimfile.F8E5M2:
		return true
	}
	return dt.IsBlock()
}

func isInt(dt mimfile.DType) bool { return !isFloat(dt) && dt != mimfile.Bool }

// intRange returns the smallest and largest values of the integer type dt.
func intRange(dt mimfile.DType) (lo, hi float64) {
	switch dt {
	case mimfile.I8:
		return math.MinInt8, math.MaxInt8
	case mimfile.U8:
		return 0, math.MaxUint8
	case mimfile.I16:
		return math.MinInt16, math.MaxInt16
	case mimfile.U16:
		return 0, math.MaxUint16
	case mimfile.U32, mimfile.U64:
		return 0, math.MaxInt32
	default: // i32, i64: held as Int32
		return math.MinInt32, math.MaxInt32
	}
}

// weight returns the tensor that a value of kind weight, const or
// variable starts from, on the CPU.
func (m *Model) weight(v mimfile.Value) (*tensor.Tensor, error) {
	if int(v.Ref) >= m.f.NumTensors() {
		return nil, fmt.Errorf("%w: tensor %d of %d", ErrFormat, v.Ref, m.f.NumTensors())
	}
	info := m.f.TensorAt(int(v.Ref))
	if info.DType != v.DType || len(info.Shape) != len(v.Shape) {
		return nil, fmt.Errorf("%w: tensor %q is %s%v, the value %s of rank %d", ErrFormat, info.Name, info.DType, info.Shape, v.DType, len(v.Shape))
	}
	shape := make([]int, len(info.Shape))
	n := 1
	for i, d := range info.Shape {
		if d > math.MaxInt32 || (!v.Shape[i].Symbolic() && uint64(v.Shape[i]) != d) {
			return nil, fmt.Errorf("%w: tensor %q of shape %v for a value of shape %v", ErrFormat, info.Name, info.Shape, v.Shape)
		}
		shape[i] = int(d)
		n *= int(d)
	}
	raw := m.f.TensorData(info)
	switch v.DType {
	case mimfile.F32, mimfile.F16, mimfile.BF16, mimfile.Q8_0, mimfile.Q4_K, mimfile.Q6_K, mimfile.I32:
		dt, _ := storage(v.DType)
		return scalarShape(tensor.FromBytesFunc(dt, raw, m.hold(), shape...), shape)
	case mimfile.F8E4M3, mimfile.F8E5M2, mimfile.Q4_0, mimfile.MXFP4, mimfile.Affine4:
		f := make([]float32, n)
		if err := mimfile.Dequantize(v.DType, raw, f); err != nil {
			return nil, fmt.Errorf("%w: tensor %q: %w", ErrFormat, info.Name, err)
		}
		return scalarShape(tensor.FromSlice(f, shape...), shape)
	}
	ints, err := decodeInts(v.DType, raw, n)
	if err != nil {
		return nil, fmt.Errorf("tensor %q: %w", info.Name, err)
	}
	return scalarShape(tensor.FromSlice(ints, shape...), shape)
}

// scalarShape returns t, made with shape, as a scalar when shape is
// empty: the constructors read no shape as that of a vector.
func scalarShape(t *tensor.Tensor, shape []int) (*tensor.Tensor, error) {
	if len(shape) == 0 {
		t = t.Reshape()
	}
	return t, t.Err()
}

// decodeInts returns the n integers or bools of dtype dt in raw as int32,
// which must hold them.
func decodeInts(dt mimfile.DType, raw []byte, n int) ([]int32, error) {
	size := len(raw) / max(n, 1)
	if n == 0 {
		return []int32{}, nil
	}
	if size*n != len(raw) {
		return nil, fmt.Errorf("%w: %d bytes for %d elements of %s", ErrFormat, len(raw), n, dt)
	}
	out := make([]int32, n)
	for i := range out {
		b := raw[i*size:]
		var x int64
		switch dt {
		case mimfile.I8:
			x = int64(int8(b[0]))
		case mimfile.U8, mimfile.Bool:
			x = int64(b[0])
		case mimfile.I16:
			x = int64(int16(binary.LittleEndian.Uint16(b)))
		case mimfile.U16:
			x = int64(binary.LittleEndian.Uint16(b))
		case mimfile.U32:
			x = int64(binary.LittleEndian.Uint32(b))
		case mimfile.I64:
			x = int64(binary.LittleEndian.Uint64(b))
		case mimfile.U64:
			u := binary.LittleEndian.Uint64(b)
			if u > math.MaxInt32 {
				return nil, &UnsupportedError{Op: "dtype", Detail: fmt.Sprintf("%s value %d does not fit in an int32", dt, u)}
			}
			x = int64(u)
		default:
			return nil, &UnsupportedError{Op: "dtype", Detail: dt.String()}
		}
		if x < math.MinInt32 || x > math.MaxInt32 {
			return nil, &UnsupportedError{Op: "dtype", Detail: fmt.Sprintf("%s value %d does not fit in an int32", dt, x)}
		}
		if dt == mimfile.Bool && x > 1 {
			return nil, fmt.Errorf("%w: bool value %d", ErrFormat, x)
		}
		out[i] = int32(x)
	}
	return out, nil
}
