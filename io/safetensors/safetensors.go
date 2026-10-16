// Package safetensors reads and writes the safetensors format: an 8-byte little-endian
// header length, a JSON header describing each tensor, then the raw
// little-endian tensor data. See https://github.com/huggingface/safetensors.
//
// Parse validates everything before returning, so a corrupt or hostile file
// yields an error, never a panic or an out-of-range read.
package safetensors

import (
	"bytes"
	"cmp"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"math"
	"slices"
)

// ErrFormat reports a malformed file.
var ErrFormat = errors.New("safetensors: invalid file")

// maxHeader bounds the JSON header, as the reference implementation does.
const maxHeader = 100 << 20

var dtypeSizes = map[string]int{
	"BOOL": 1, "U8": 1, "I8": 1, "F8_E5M2": 1, "F8_E4M3": 1,
	"I16": 2, "U16": 2, "F16": 2, "BF16": 2,
	"I32": 4, "U32": 4, "F32": 4,
	"I64": 8, "U64": 8, "F64": 8,
}

// Tensor is one named tensor in a file. Data aliases the parsed bytes.
type Tensor struct {
	DType string
	Shape []int
	Data  []byte
}

// File is a parsed safetensors file.
type File struct {
	Metadata map[string]string
	tensors  map[string]Tensor
	mapping  *mapping // the mapping the tensors alias, when Open made one
}

type entry struct {
	DType       string `json:"dtype"`
	Shape       []int  `json:"shape"`
	DataOffsets [2]int `json:"data_offsets"`
}

// Parse decodes b. The returned tensors alias b.
func Parse(b []byte) (*File, error) {
	if len(b) < 8 {
		return nil, fmt.Errorf("%w: %d bytes", ErrFormat, len(b))
	}
	n := binary.LittleEndian.Uint64(b)
	if n > maxHeader || n > uint64(len(b)-8) {
		return nil, fmt.Errorf("%w: header of %d bytes in %d-byte file", ErrFormat, n, len(b))
	}
	header, data := b[8:8+n], b[8+n:]

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(header, &raw); err != nil {
		return nil, fmt.Errorf("%w: header: %v", ErrFormat, err)
	}
	f := &File{tensors: make(map[string]Tensor, len(raw))}
	type span struct{ lo, hi int }
	spans := make([]span, 0, len(raw))
	for name, msg := range raw {
		if name == "__metadata__" {
			if err := json.Unmarshal(msg, &f.Metadata); err != nil {
				return nil, fmt.Errorf("%w: metadata: %v", ErrFormat, err)
			}
			continue
		}
		var e entry
		if err := json.Unmarshal(msg, &e); err != nil {
			return nil, fmt.Errorf("%w: tensor %q: %v", ErrFormat, name, err)
		}
		size, ok := dtypeSizes[e.DType]
		if !ok {
			return nil, fmt.Errorf("%w: tensor %q: dtype %q", ErrFormat, name, e.DType)
		}
		count := 1
		for _, d := range e.Shape {
			if d < 0 || (d != 0 && count > math.MaxInt/size/d) {
				return nil, fmt.Errorf("%w: tensor %q: shape %v", ErrFormat, name, e.Shape)
			}
			count *= d
		}
		lo, hi := e.DataOffsets[0], e.DataOffsets[1]
		if lo < 0 || lo > hi || hi > len(data) || hi-lo != count*size {
			return nil, fmt.Errorf("%w: tensor %q: offsets %v for %d bytes of %v %s", ErrFormat, name, e.DataOffsets, len(data), e.Shape, e.DType)
		}
		f.tensors[name] = Tensor{DType: e.DType, Shape: e.Shape, Data: data[lo:hi:hi]}
		spans = append(spans, span{lo, hi})
	}
	// The data section must be covered exactly, without gaps or overlaps.
	// Empty tensors sort before a tensor starting at the same byte.
	slices.SortFunc(spans, func(a, b span) int { return cmp.Or(cmp.Compare(a.lo, b.lo), cmp.Compare(a.hi, b.hi)) })
	end := 0
	for _, s := range spans {
		if s.lo != end {
			return nil, fmt.Errorf("%w: data is not packed at byte %d", ErrFormat, end)
		}
		end = s.hi
	}
	if end != len(data) {
		return nil, fmt.Errorf("%w: %d unused data bytes", ErrFormat, len(data)-end)
	}
	return f, nil
}

// Names returns the tensor names in sorted order.
func (f *File) Names() []string {
	names := make([]string, 0, len(f.tensors))
	for name := range f.tensors {
		names = append(names, name)
	}
	slices.Sort(names)
	return names
}

// Tensor returns the named tensor.
func (f *File) Tensor(name string) (Tensor, bool) {
	t, ok := f.tensors[name]
	return t, ok
}

// Float32s decodes an F32 tensor, or an F16 or BF16 tensor, whose every
// element is a float32 value.
func (t Tensor) Float32s() ([]float32, error) {
	switch t.DType {
	case "F16", "BF16":
		return t.halfFloat32s(), nil
	case "F32":
	default:
		return nil, fmt.Errorf("safetensors: %s tensor read as F32", t.DType)
	}
	out := make([]float32, len(t.Data)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(t.Data[4*i:]))
	}
	return out, nil
}

// Int32s decodes an I32 tensor.
func (t Tensor) Int32s() ([]int32, error) {
	if t.DType != "I32" {
		return nil, fmt.Errorf("safetensors: %s tensor read as I32", t.DType)
	}
	out := make([]int32, len(t.Data)/4)
	for i := range out {
		out[i] = int32(binary.LittleEndian.Uint32(t.Data[4*i:]))
	}
	return out, nil
}

// Float32Tensor returns an F32 tensor of shape holding v.
func Float32Tensor(shape []int, v []float32) Tensor {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	return Tensor{DType: "F32", Shape: slices.Clone(shape), Data: b}
}

// Int32Tensor returns an I32 tensor of shape holding v.
func Int32Tensor(shape []int, v []int32) Tensor {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], uint32(x))
	}
	return Tensor{DType: "I32", Shape: slices.Clone(shape), Data: b}
}

// Write encodes tensors and metadata. Tensors are laid out in name order,
// so equal inputs produce identical files.
func Write(w io.Writer, tensors map[string]Tensor, metadata map[string]string) error {
	header := make(map[string]any, len(tensors)+1)
	if len(metadata) > 0 {
		header["__metadata__"] = metadata
	}
	names := slices.Sorted(maps.Keys(tensors))
	offset := 0
	for _, name := range names {
		t := tensors[name]
		size, ok := dtypeSizes[t.DType]
		if !ok || name == "__metadata__" {
			return fmt.Errorf("safetensors: write tensor %q of dtype %q: %w", name, t.DType, ErrFormat)
		}
		n := size
		for _, d := range t.Shape {
			if d < 0 || (d > 0 && n > math.MaxInt/d) {
				return fmt.Errorf("safetensors: write tensor %q of shape %v: %w", name, t.Shape, ErrFormat)
			}
			n *= d
		}
		if n != len(t.Data) {
			return fmt.Errorf("safetensors: tensor %q has %d bytes for %s%v: %w", name, len(t.Data), t.DType, t.Shape, ErrFormat)
		}
		shape := t.Shape
		if shape == nil {
			shape = []int{}
		}
		header[name] = entry{DType: t.DType, Shape: shape, DataOffsets: [2]int{offset, offset + n}}
		offset += n
	}
	h, err := json.Marshal(header)
	if err != nil {
		return fmt.Errorf("safetensors: header: %w", err)
	}
	h = append(h, bytes.Repeat([]byte(" "), (8-len(h)%8)%8)...) // align the data to 8 bytes
	var size [8]byte
	binary.LittleEndian.PutUint64(size[:], uint64(len(h)))
	if _, err := w.Write(size[:]); err != nil {
		return err
	}
	if _, err := w.Write(h); err != nil {
		return err
	}
	for _, name := range names {
		if _, err := w.Write(tensors[name].Data); err != nil {
			return err
		}
	}
	return nil
}
