// Package gguf reads and writes GGUF files, the format in which llama.cpp
// and ggml store models: typed metadata, such as a model's architecture,
// hyperparameters and tokenizer, then tensors, quantized or not. See
// https://github.com/ggml-org/ggml/blob/master/docs/gguf.md.
//
// Open maps a file and its tensors alias the mapping, so a model larger
// than memory opens at once and Load reads its weights in place, as
// quantized tensors where the file quantizes them:
//
//	f, err := gguf.Open("model.gguf")
//	...
//	n, _ := f.Metadata("llama.block_count")
//	vocab, _ := f.Metadata("tokenizer.ggml.tokens")
//	w, err := f.Load("blk.0.attn_q.weight") // [out, in] of Q4_K, say
//	y := x.MatMulT(w)
//
// Parse validates the whole header before returning, bounding every
// count, length and offset it reads by the size of the file, so a corrupt
// or hostile file yields an error, never a panic or a read out of range.
// Versions 2 and 3 are read, little-endian.
package gguf

import (
	"cmp"
	"encoding/binary"
	"errors"
	"fmt"
	"io/fs"
	"math"
	"runtime"
	"slices"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/internal/filemap"
	"github.com/gsoultan/mimir/tensor"
)

// ErrFormat reports a malformed or unsupported file.
var ErrFormat = errors.New("gguf: invalid file")

// Magic is the four bytes a GGUF file starts with.
const Magic = "GGUF"

// DefaultAlignment is the alignment of tensor data in a file without the
// metadata key general.alignment.
const DefaultAlignment = 32

// Limits of what Parse accepts, besides the size of the file.
const (
	maxDims  = 4 // GGML_MAX_DIMS
	maxDepth = 8 // nesting of arrays of arrays
)

// Tensor is one tensor of a file.
type Tensor struct {
	Name string
	Type Type
	// Shape is in row-major order, slowest dimension first: the reverse
	// of ggml's ne, which lists the contiguous dimension first. A weight
	// that ggml multiplies as [in, out] is [out, in], as in PyTorch.
	Shape []int
	// Data aliases the file's bytes of the tensor, whole blocks of Type
	// along the last dimension.
	Data []byte
}

// DType returns the mimir dtype of t's elements (see Type.DType), or a
// *TypeError, which matches tensor.ErrDType, for a type mimir does not
// read.
func (t Tensor) DType() (backend.DType, error) {
	dt, ok := t.Type.DType()
	if !ok {
		return 0, &TypeError{Tensor: t.Name, Type: t.Type}
	}
	return dt, nil
}

// File is a parsed GGUF file.
type File struct {
	// Version is the file's format version, 2 or 3.
	Version uint32

	keys      []string
	metadata  map[string]Value
	tensors   []Tensor
	byName    map[string]int
	alignment int
	mapping   *filemap.Mapping // the mapping the tensors alias, when Open made one
	cleanup   runtime.Cleanup  // lets go of the mapping if the File is garbage
}

// Keys returns the metadata keys in the order of the file.
func (f *File) Keys() []string { return slices.Clone(f.keys) }

// Metadata returns the value of a metadata key, such as
// "llama.block_count" or "tokenizer.ggml.tokens".
func (f *File) Metadata(key string) (Value, bool) {
	v, ok := f.metadata[key]
	return v, ok
}

// Architecture returns the model's architecture, general.architecture,
// such as "llama", which prefixes the keys of its hyperparameters, or ""
// if the file has none.
func (f *File) Architecture() string {
	s, _ := f.metadata["general.architecture"].Text()
	return s
}

// Alignment returns the alignment of the tensors' data within the file.
func (f *File) Alignment() int { return f.alignment }

// Tensors returns the tensors in the order of the file.
func (f *File) Tensors() []Tensor { return slices.Clone(f.tensors) }

// Tensor returns the named tensor.
func (f *File) Tensor(name string) (Tensor, bool) {
	i, ok := f.byName[name]
	if !ok {
		return Tensor{}, false
	}
	return f.tensors[i], true
}

// reader reads the header, bounded by the bytes of the file. Its first
// error sticks: later reads return zero values.
type reader struct {
	b   []byte
	off int
	err error
}

func (r *reader) fail(format string, args ...any) {
	if r.err == nil {
		r.err = fmt.Errorf("%w: at byte %d: %s", ErrFormat, r.off, fmt.Sprintf(format, args...))
	}
}

// left returns the number of bytes not read yet.
func (r *reader) left() int { return len(r.b) - r.off }

// next returns the next n bytes, or nil if the file ends first.
func (r *reader) next(n int) []byte {
	if r.err != nil {
		return nil
	}
	if n < 0 || n > r.left() {
		r.fail("%d bytes past the end of the file", n)
		return nil
	}
	b := r.b[r.off : r.off+n]
	r.off += n
	return b
}

func (r *reader) u32() uint32 {
	if b := r.next(4); b != nil {
		return binary.LittleEndian.Uint32(b)
	}
	return 0
}

func (r *reader) u64() uint64 {
	if b := r.next(8); b != nil {
		return binary.LittleEndian.Uint64(b)
	}
	return 0
}

// count reads a uint64 count of items of at least min bytes each, which
// the rest of the file must be able to hold.
func (r *reader) count(what string, min int) int {
	n := r.u64()
	if r.err == nil && n > uint64(r.left()/min) {
		r.fail("%d %s in %d bytes", n, what, r.left())
		return 0
	}
	return int(n)
}

func (r *reader) str() string {
	return string(r.next(r.count("string bytes", 1)))
}

// value reads a value of type t, nested in depth arrays.
func (r *reader) value(t ValueType, depth int) Value {
	switch t {
	case Uint8:
		return Value{uint8(r.scalar(1))}
	case Int8:
		return Value{int8(r.scalar(1))}
	case Uint16:
		return Value{uint16(r.scalar(2))}
	case Int16:
		return Value{int16(r.scalar(2))}
	case Uint32:
		return Value{uint32(r.scalar(4))}
	case Int32:
		return Value{int32(r.scalar(4))}
	case Float32:
		return Value{math.Float32frombits(uint32(r.scalar(4)))}
	case Bool:
		return Value{r.boolean()}
	case String:
		return Value{r.str()}
	case Uint64:
		return Value{r.scalar(8)}
	case Int64:
		return Value{int64(r.scalar(8))}
	case Float64:
		return Value{math.Float64frombits(r.scalar(8))}
	case Array:
		return r.array(depth)
	}
	r.fail("value type %d", uint32(t))
	return Value{}
}

// scalar reads a little-endian integer of size bytes.
func (r *reader) scalar(size int) uint64 {
	b := r.next(size)
	if b == nil {
		return 0
	}
	var v uint64
	for i := size - 1; i >= 0; i-- {
		v = v<<8 | uint64(b[i])
	}
	return v
}

func (r *reader) boolean() bool {
	switch v := r.scalar(1); v {
	case 0, 1:
		return v == 1
	default:
		r.fail("bool %d", v)
		return false
	}
}

// array reads an array nested in depth arrays.
func (r *reader) array(depth int) Value {
	if depth >= maxDepth {
		r.fail("arrays nested more than %d deep", maxDepth)
		return Value{}
	}
	t := ValueType(r.u32())
	min := t.size()
	switch t {
	case String:
		min = 8
	case Array:
		min = 12
	case invalidValue:
		min = 0
	}
	if r.err != nil {
		return Value{}
	}
	if min == 0 {
		r.fail("array of value type %d", uint32(t))
		return Value{}
	}
	n := r.count("array elements", min)
	if r.err != nil {
		return Value{}
	}
	switch t {
	case Uint8:
		return Value{slices.Clone(r.next(n))}
	case Int8:
		return Value{fixed(r, n, 1, func(b []byte) int8 { return int8(b[0]) })}
	case Uint16:
		return Value{fixed(r, n, 2, binary.LittleEndian.Uint16)}
	case Int16:
		return Value{fixed(r, n, 2, func(b []byte) int16 { return int16(binary.LittleEndian.Uint16(b)) })}
	case Uint32:
		return Value{fixed(r, n, 4, binary.LittleEndian.Uint32)}
	case Int32:
		return Value{fixed(r, n, 4, func(b []byte) int32 { return int32(binary.LittleEndian.Uint32(b)) })}
	case Float32:
		return Value{fixed(r, n, 4, func(b []byte) float32 { return math.Float32frombits(binary.LittleEndian.Uint32(b)) })}
	case Uint64:
		return Value{fixed(r, n, 8, binary.LittleEndian.Uint64)}
	case Int64:
		return Value{fixed(r, n, 8, func(b []byte) int64 { return int64(binary.LittleEndian.Uint64(b)) })}
	case Float64:
		return Value{fixed(r, n, 8, func(b []byte) float64 { return math.Float64frombits(binary.LittleEndian.Uint64(b)) })}
	case Bool:
		out := make([]bool, n)
		for i := range out {
			out[i] = r.boolean()
		}
		return Value{out}
	case String:
		out := make([]string, n)
		for i := range out {
			out[i] = r.str()
		}
		return Value{out}
	default: // Array
		out := make([]Value, n)
		for i := range out {
			out[i] = r.array(depth + 1)
		}
		return Value{out}
	}
}

// fixed reads n values of size bytes each with decode.
func fixed[T any](r *reader, n, size int, decode func([]byte) T) []T {
	b := r.next(n * size)
	if b == nil {
		return nil
	}
	out := make([]T, n)
	for i := range out {
		out[i] = decode(b[i*size:])
	}
	return out
}

// Parse decodes a GGUF file. The tensors' Data alias b.
func Parse(b []byte) (*File, error) {
	r := &reader{b: b}
	if string(r.next(4)) != Magic {
		return nil, fmt.Errorf("%w: no GGUF magic", ErrFormat)
	}
	f := &File{Version: r.u32(), metadata: map[string]Value{}, byName: map[string]int{}, alignment: DefaultAlignment}
	if r.err == nil && f.Version != 2 && f.Version != 3 {
		return nil, fmt.Errorf("%w: version %d, want 2 or 3 little-endian", ErrFormat, f.Version)
	}
	// A tensor's info takes at least 24 bytes, and a key and value 13.
	nt := r.count("tensors", 24)
	nkv := r.count("metadata keys", 13)
	if r.err != nil {
		return nil, r.err
	}
	for range nkv {
		key := r.str()
		v := r.value(ValueType(r.u32()), 0)
		if r.err != nil {
			return nil, r.err
		}
		if _, dup := f.metadata[key]; dup {
			return nil, fmt.Errorf("%w: key %q repeated", ErrFormat, key)
		}
		f.keys = append(f.keys, key)
		f.metadata[key] = v
	}
	if v, ok := f.metadata["general.alignment"]; ok {
		a, isU32 := v.Interface().(uint32)
		if !isU32 || a == 0 || a&(a-1) != 0 {
			return nil, fmt.Errorf("%w: general.alignment %v", ErrFormat, v)
		}
		f.alignment = int(a)
	}

	type span struct {
		lo, hi int
		name   string
	}
	spans := make([]span, 0, nt)
	offsets := make([]uint64, 0, nt)
	f.tensors = make([]Tensor, 0, nt)
	for range nt {
		t := Tensor{Name: r.str()}
		nd := r.u32()
		if r.err == nil && nd > maxDims {
			r.fail("tensor %q has %d dimensions", t.Name, nd)
		}
		if r.err != nil {
			return nil, r.err
		}
		ne := make([]uint64, 0, maxDims)
		for range nd {
			ne = append(ne, r.u64())
		}
		t.Type = Type(r.u32())
		off := r.u64()
		if r.err != nil {
			return nil, r.err
		}
		n, err := tensorBytes(t.Type, ne, len(b))
		if err != nil {
			return nil, fmt.Errorf("%w: tensor %q: %w", ErrFormat, t.Name, err)
		}
		if _, dup := f.byName[t.Name]; dup {
			return nil, fmt.Errorf("%w: tensor %q repeated", ErrFormat, t.Name)
		}
		if off%uint64(f.alignment) != 0 || off > uint64(len(b)) {
			return nil, fmt.Errorf("%w: tensor %q at offset %d, alignment %d", ErrFormat, t.Name, off, f.alignment)
		}
		t.Shape = make([]int, len(ne))
		for i, d := range ne {
			t.Shape[len(ne)-1-i] = int(d)
		}
		f.byName[t.Name] = len(f.tensors)
		f.tensors = append(f.tensors, t)
		spans = append(spans, span{int(off), int(off) + n, t.Name})
		offsets = append(offsets, off)
	}
	start := (r.off + f.alignment - 1) / f.alignment * f.alignment
	data := b[min(start, len(b)):]
	for i, s := range spans {
		if s.hi > len(data) {
			return nil, fmt.Errorf("%w: tensor %q of %d bytes at offset %d, in %d bytes of data", ErrFormat, s.name, s.hi-s.lo, offsets[i], len(data))
		}
		f.tensors[i].Data = data[s.lo:s.hi:s.hi]
	}
	// Tensors must not overlap: a write to one, as an optimizer makes to
	// a parameter it updates in place, would change another.
	slices.SortFunc(spans, func(a, b span) int { return cmp.Or(cmp.Compare(a.lo, b.lo), cmp.Compare(a.hi, b.hi)) })
	end := 0
	for _, s := range spans {
		if s.lo == s.hi {
			continue
		}
		if s.lo < end {
			return nil, fmt.Errorf("%w: tensor %q overlaps another", ErrFormat, s.name)
		}
		end = s.hi
	}
	return f, nil
}

// tensorBytes returns the size of a tensor of type t and dimensions ne,
// in ggml's order, checking that its rows hold whole blocks and that it
// is no larger than limit bytes.
func tensorBytes(t Type, ne []uint64, limit int) (int, error) {
	bs, bb := t.BlockSize(), t.BlockBytes()
	if bs == 0 {
		return 0, fmt.Errorf("type %d", uint32(t))
	}
	// A dimension ne does not list is 1, as in ggml, so a scalar holds
	// no whole block of a quantized type.
	if (len(ne) == 0 && bs > 1) || (len(ne) > 0 && ne[0]%uint64(bs) != 0) {
		return 0, fmt.Errorf("dimensions %v of %s, in blocks of %d", ne, t, bs)
	}
	// Every dimension, and the number of blocks, must fit in the file,
	// whose size bounds every product computed here.
	blocks := uint64(1)
	for i, d := range ne {
		if i == 0 {
			d /= uint64(bs)
		}
		if d > uint64(limit) {
			return 0, fmt.Errorf("dimensions %v larger than the file", ne)
		}
		if d != 0 && blocks > uint64(limit)/d {
			blocks = uint64(limit) + 1 // too many, and no product can overflow
		} else {
			blocks *= d
		}
	}
	if blocks > uint64(limit/bb) {
		return 0, fmt.Errorf("dimensions %v larger than the file", ne)
	}
	return int(blocks) * bb, nil
}

// Open parses the file at path. Like safetensors.Open it maps the file
// where the platform can, privately, and the tensors alias the mapping:
// it lasts until Close, or until the File is garbage, and past that for
// as long as a reader that Hold added keeps it. Elsewhere Open reads the
// file into memory.
func Open(path string) (*File, error) {
	data, m, err := filemap.Read(path)
	if errors.Is(err, filemap.ErrTooLarge) {
		return nil, fmt.Errorf("gguf: %s: %w: %w", path, err, ErrFormat)
	}
	if err != nil {
		return nil, fmt.Errorf("gguf: %w", err)
	}
	f, err := Parse(data)
	if err != nil {
		if m != nil {
			_ = m.Close()
		}
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if m != nil {
		f.mapping = m
		f.cleanup = runtime.AddCleanup(f, func(m *filemap.Mapping) { _ = m.Close() }, m)
	}
	return f, nil
}

// Load returns the named tensor as a mimir tensor of its type (see
// Type.DType) that reads the file's bytes in place, keeping the mapping
// (see Hold) for as long as it reads them. It fails with an error that
// matches fs.ErrNotExist if the file has no such tensor, and with a
// *TypeError, which matches tensor.ErrDType, if mimir does not read its
// type.
func (f *File) Load(name string) (*tensor.Tensor, error) {
	t, ok := f.Tensor(name)
	if !ok {
		return nil, fmt.Errorf("gguf: tensor %q: %w", name, fs.ErrNotExist)
	}
	dt, err := t.DType()
	if err != nil {
		return nil, err
	}
	x := tensor.FromBytesFunc(dt, t.Data, f.Hold(), t.Shape...)
	return x, x.Err()
}

// Mapped reports whether the tensors alias a mapping of the file.
func (f *File) Mapped() bool { return f.mapping != nil }

// Hold keeps the mapping of a file that Open mapped until the returned
// function is called, even past Close, for a reader of the tensors' bytes
// that outlives the File, such as a tensor that tensor.FromBytesFunc
// builds over them. The function lets go once, however often it is
// called. For a file that is not mapped, or is closed, it does nothing.
func (f *File) Hold() func() {
	if f.mapping == nil {
		return func() {}
	}
	return f.mapping.Hold()
}

// Close lets go of the File's hold on the mapping of a file that Open
// mapped, which unmaps it unless a reader that Hold added still keeps it.
// Afterwards the file has no tensors. Closing a file that is not mapped,
// or closing again, does nothing.
func (f *File) Close() error {
	if f.mapping == nil {
		return nil
	}
	f.cleanup.Stop()
	f.tensors, f.byName = nil, nil
	return f.mapping.Close()
}
