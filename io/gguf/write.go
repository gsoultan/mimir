package gguf

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

// KV is a metadata key and its value.
type KV struct {
	Key   string
	Value Value
}

// Write encodes a version 3 GGUF file of metadata and tensors, in the
// order given. Each tensor's Data must hold its Shape of whole blocks of
// its Type. Tensor data is aligned to general.alignment, a uint32 among
// the metadata, or to DefaultAlignment.
func Write(w io.Writer, metadata []KV, tensors []Tensor) error {
	alignment := DefaultAlignment
	for _, kv := range metadata {
		if kv.Key != "general.alignment" {
			continue
		}
		a, ok := kv.Value.Interface().(uint32)
		if !ok || a == 0 || a&(a-1) != 0 {
			return fmt.Errorf("gguf: write general.alignment %v: %w", kv.Value, ErrFormat)
		}
		alignment = int(a)
	}
	bw := bufio.NewWriter(w)
	e := &encoder{w: bw}
	e.raw([]byte(Magic))
	e.u32(3)
	e.u64(uint64(len(tensors)))
	e.u64(uint64(len(metadata)))
	for _, kv := range metadata {
		t, _ := typeOf(kv.Value.v)
		if t == invalidValue {
			return fmt.Errorf("gguf: write %q: value of Go type %T: %w", kv.Key, kv.Value.v, ErrFormat)
		}
		e.str(kv.Key)
		e.u32(uint32(t))
		if err := e.value(kv.Value, 0); err != nil {
			return fmt.Errorf("gguf: write %q: %w", kv.Key, err)
		}
	}
	offset := 0
	for _, t := range tensors {
		ne := make([]uint64, len(t.Shape))
		for j, d := range t.Shape {
			if d < 0 {
				return fmt.Errorf("gguf: write tensor %q of shape %v: %w", t.Name, t.Shape, ErrFormat)
			}
			ne[len(t.Shape)-1-j] = uint64(d)
		}
		n, err := tensorBytes(t.Type, ne, math.MaxInt)
		if err != nil || len(ne) > maxDims || n != len(t.Data) {
			return fmt.Errorf("gguf: write tensor %q of %d bytes as %s%v: %w", t.Name, len(t.Data), t.Type, t.Shape, ErrFormat)
		}
		e.str(t.Name)
		e.u32(uint32(len(ne)))
		for _, d := range ne {
			e.u64(d)
		}
		e.u32(uint32(t.Type))
		e.u64(uint64(offset))
		offset += (n + alignment - 1) / alignment * alignment
	}
	e.pad(alignment)
	for i, t := range tensors {
		e.raw(t.Data)
		if i < len(tensors)-1 {
			e.pad(alignment)
		}
	}
	if e.err != nil {
		return e.err
	}
	return bw.Flush()
}

// encoder writes little-endian values, keeping the first error and the
// number of bytes written.
type encoder struct {
	w   io.Writer
	n   int
	err error
}

func (e *encoder) raw(b []byte) {
	if e.err != nil {
		return
	}
	n, err := e.w.Write(b)
	e.n += n
	e.err = err
}

func (e *encoder) u32(v uint32) { e.raw(binary.LittleEndian.AppendUint32(nil, v)) }
func (e *encoder) u64(v uint64) { e.raw(binary.LittleEndian.AppendUint64(nil, v)) }

func (e *encoder) str(s string) {
	e.u64(uint64(len(s)))
	e.raw([]byte(s))
}

// pad writes zeros up to a multiple of alignment bytes.
func (e *encoder) pad(alignment int) {
	e.raw(make([]byte, (alignment-e.n%alignment)%alignment))
}

// value writes v without its type, nested in depth arrays.
func (e *encoder) value(v Value, depth int) error {
	var b []byte
	switch x := v.v.(type) {
	case uint8:
		b = []byte{x}
	case int8:
		b = []byte{byte(x)}
	case uint16:
		b = binary.LittleEndian.AppendUint16(nil, x)
	case int16:
		b = binary.LittleEndian.AppendUint16(nil, uint16(x))
	case uint32:
		b = binary.LittleEndian.AppendUint32(nil, x)
	case int32:
		b = binary.LittleEndian.AppendUint32(nil, uint32(x))
	case float32:
		b = binary.LittleEndian.AppendUint32(nil, math.Float32bits(x))
	case bool:
		if x {
			b = []byte{1}
		} else {
			b = []byte{0}
		}
	case string:
		e.str(x)
	case uint64:
		b = binary.LittleEndian.AppendUint64(nil, x)
	case int64:
		b = binary.LittleEndian.AppendUint64(nil, uint64(x))
	case float64:
		b = binary.LittleEndian.AppendUint64(nil, math.Float64bits(x))
	default:
		return e.array(v, depth)
	}
	e.raw(b)
	return nil
}

// array writes the array v: its element type, length and elements.
func (e *encoder) array(v Value, depth int) error {
	if depth >= maxDepth {
		return fmt.Errorf("arrays nested more than %d deep: %w", maxDepth, ErrFormat)
	}
	t, elem := typeOf(v.v)
	if t != Array {
		return fmt.Errorf("value of Go type %T: %w", v.v, ErrFormat)
	}
	e.u32(uint32(elem))
	e.u64(uint64(v.Len()))
	each := func(n int, at func(i int) any) error {
		for i := range n {
			if err := e.value(Value{at(i)}, depth+1); err != nil {
				return err
			}
		}
		return nil
	}
	switch x := v.v.(type) {
	case []uint8:
		e.raw(x)
		return nil
	case []Value:
		for _, a := range x {
			if t, _ := typeOf(a.v); t != Array {
				return fmt.Errorf("array of arrays holding %T: %w", a.v, ErrFormat)
			}
			if err := e.array(a, depth+1); err != nil {
				return err
			}
		}
		return nil
	case []int8:
		return each(len(x), func(i int) any { return x[i] })
	case []uint16:
		return each(len(x), func(i int) any { return x[i] })
	case []int16:
		return each(len(x), func(i int) any { return x[i] })
	case []uint32:
		return each(len(x), func(i int) any { return x[i] })
	case []int32:
		return each(len(x), func(i int) any { return x[i] })
	case []float32:
		return each(len(x), func(i int) any { return x[i] })
	case []bool:
		return each(len(x), func(i int) any { return x[i] })
	case []string:
		return each(len(x), func(i int) any { return x[i] })
	case []uint64:
		return each(len(x), func(i int) any { return x[i] })
	case []int64:
		return each(len(x), func(i int) any { return x[i] })
	default: // []float64
		y := x.([]float64)
		return each(len(y), func(i int) any { return y[i] })
	}
}
