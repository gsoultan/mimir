package tensor

import (
	"context"
	"encoding/binary"
	"fmt"
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
// little-endian bytes of data, as a Mimir Model file stores them. With no
// shape it is one-dimensional; otherwise the shape must hold exactly the
// elements of data: len(data)/dt.Size() of them, or for a block-quantized
// dt the elements of len(data)/dt.BlockBytes() blocks, which fill whole
// rows of the last dimension.
//
// The tensor reads data in place, without copying, when data starts at an
// address that is a multiple of the element size and the host is
// little-endian, and always for a block-quantized dt; otherwise it holds
// a copy, as FromSlice does. In place, the caller keeps data valid for as
// long as the tensor, or any view of it, is in use; FromBytesFunc tells
// the caller when that is. An operation that writes over the tensor's
// memory, as an optimizer step does over a parameter set from it (see
// Param.Set), writes the bytes.
//
// The tensor lives on the CPU whatever the default device; To moves it.
func FromBytes(dt DType, data []byte, shape ...int) *Tensor {
	return FromBytesFunc(dt, data, nil, shape...)
}

// FromBytesFunc is FromBytes for bytes that the tensor holds until it no
// longer reads them, such as a mapped file whose owner counts its
// readers, as io/mim's models do: release runs once, after the
// tensor's memory, which every view and every tensor that took it over
// share, becomes garbage or is freed. When the tensor holds a copy, or
// FromBytesFunc fails, release runs before it returns. A nil release
// makes it FromBytes.
func FromBytesFunc(dt DType, data []byte, release func(), shape ...int) *Tensor {
	done := func() {
		if release != nil {
			release()
		}
	}
	if defaultErr != nil {
		done()
		return fail(defaultErr)
	}
	bs, bb := dt.BlockSize(), dt.BlockBytes()
	if bb == 0 {
		done()
		return fail(fmt.Errorf("tensor: FromBytes of %s: %w", dt, ErrDType))
	}
	if len(shape) == 0 {
		shape = []int{len(data) / bb * bs}
	}
	n, ok := backend.NumElements(shape)
	size, whole := dt.Bytes(n)
	if !ok || !whole || size != len(data) {
		done()
		return fail(fmt.Errorf("tensor: %d bytes of %s for shape %v: %w", len(data), dt, shape, ErrShape))
	}
	if dt.IsQuantized() || hostIsLittleEndian && aligned(data, bb) {
		buf, err := cpu.WrapFunc(dt, data, release)
		if err != nil {
			done()
			return fail(err)
		}
		return wrap(graph.Load(cpuBackend, buf, shape))
	}
	defer done() // upload copies data
	return upload(cpuBackend, dt, hostOrder(bb, data), shape)
}

// Bytes computes t and returns its elements as FromBytes reads them:
// little-endian, and for a block-quantized dtype in its blocks.
func (t *Tensor) Bytes(ctx context.Context) ([]byte, error) {
	if err := firstErr(t); err != nil {
		return nil, err
	}
	if err := computable(t); err != nil {
		return nil, err
	}
	data, err := hostBytes(ctx, t.node)
	if err != nil || t.node.DType.IsQuantized() {
		return data, err
	}
	return hostOrder(t.node.DType.BlockBytes(), data), nil
}

// ReadBytes computes t and copies its elements into dst as Bytes returns
// them. dst must hold exactly t's bytes. It allocates nothing, so a loop
// that reads many tensors, as a file writer does, can reuse one buffer.
func (t *Tensor) ReadBytes(ctx context.Context, dst []byte) error {
	if err := firstErr(t); err != nil {
		return err
	}
	if err := computable(t); err != nil {
		return err
	}
	if size, ok := t.node.DType.Bytes(t.node.Size()); !ok || size != len(dst) {
		return fmt.Errorf("tensor: read %d bytes of a %s tensor of shape %v: %w", len(dst), t.node.DType, t.node.Shape, ErrShape)
	}
	if err := download(ctx, t.node, dst); err != nil {
		return err
	}
	if !hostIsLittleEndian && !t.node.DType.IsQuantized() {
		copy(dst, hostOrder(t.node.DType.BlockBytes(), dst))
	}
	return nil
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
