package tensor

import (
	"encoding/binary"
	"fmt"
	"math"
	"unsafe"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/backend/cpu"
	"github.com/gsoultan/mimir/internal/graph"
)

// hostIsLittleEndian reports whether the host stores integers least
// significant byte first, so that little-endian bytes are in its order
// already. binary.NativeEndian and binary.LittleEndian have different
// types, so they are compared by what they decode.
var hostIsLittleEndian = binary.NativeEndian.Uint32([]byte{1, 2, 3, 4}) == binary.LittleEndian.Uint32([]byte{1, 2, 3, 4})

// FromBytes returns a tensor on the CPU whose elements of dt are the
// little-endian bytes of data, as a safetensors file stores them. With no
// shape it is one-dimensional; otherwise the shape must hold exactly
// len(data)/dt.Size() elements.
//
// The tensor reads data in place, without copying, when data starts at an
// address that is a multiple of the element size and the host is
// little-endian; otherwise it holds a copy, as FromSlice does. In place,
// the caller keeps data valid for as long as the tensor, or any view of
// it, is in use: data may be a mapped file, such as one safetensors.Open
// maps. An operation that writes over the tensor's memory, as an optimizer
// step does over a parameter set from it (see Param.Set), writes the
// bytes.
//
// The tensor lives on the CPU whatever the default device; To moves it.
func FromBytes(dt DType, data []byte, shape ...int) *Tensor {
	if defaultErr != nil {
		return fail(defaultErr)
	}
	size := dt.Size()
	if size == 0 {
		return fail(fmt.Errorf("tensor: FromBytes of %s: %w", dt, ErrDType))
	}
	if len(shape) == 0 {
		shape = []int{len(data) / size}
	}
	n, ok := backend.NumElements(shape)
	if !ok || n > math.MaxInt/size || n*size != len(data) {
		return fail(fmt.Errorf("tensor: %d bytes of %s for shape %v: %w", len(data), dt, shape, ErrShape))
	}
	if hostIsLittleEndian && aligned(data, size) {
		buf, err := cpu.Wrap(dt, data)
		if err != nil {
			return fail(err)
		}
		return wrap(graph.Load(cpuBackend, buf, shape))
	}
	return upload(cpuBackend, dt, hostOrder(size, data), shape)
}

// aligned reports whether data starts at a multiple of size bytes.
func aligned(data []byte, size int) bool {
	return len(data) == 0 || uintptr(unsafe.Pointer(&data[0]))%uintptr(size) == 0
}

// hostOrder returns the little-endian elements of size bytes in data in
// the host's byte order: data itself on a little-endian host, and a
// converted copy elsewhere.
func hostOrder(size int, data []byte) []byte {
	if hostIsLittleEndian {
		return data
	}
	out := make([]byte, len(data))
	for i := 0; i+size <= len(data); i += size {
		switch size {
		case 2:
			binary.NativeEndian.PutUint16(out[i:], binary.LittleEndian.Uint16(data[i:]))
		case 4:
			binary.NativeEndian.PutUint32(out[i:], binary.LittleEndian.Uint32(data[i:]))
		case 8:
			binary.NativeEndian.PutUint64(out[i:], binary.LittleEndian.Uint64(data[i:]))
		default:
			copy(out[i:i+size], data[i:i+size])
		}
	}
	return out
}
