package gguf

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"
	"unsafe"

	"github.com/gsoultan/mimir/tensor"
)

const fixture = "testdata/small.gguf"

// TestFixture reads the file that llama.cpp's writer wrote (see
// tools/gguf/gen.py) and checks that mimir dequantizes each of its
// weights bit for bit as ggml does.
func TestFixture(t *testing.T) {
	f, err := Open(fixture)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if f.Version != 3 || f.Alignment() != 64 || f.Architecture() != "llama" {
		t.Errorf("version %d, alignment %d, architecture %q", f.Version, f.Alignment(), f.Architecture())
	}
	keys := f.Keys()
	if len(keys) != 27 || keys[0] != "general.architecture" || keys[len(keys)-1] != "test.nested" {
		t.Errorf("keys %q", keys)
	}
	scalars := map[string]any{
		"test.uint8": uint8(200), "test.int8": int8(-100), "test.uint16": uint16(60000),
		"test.int16": int16(-30000), "test.uint32": uint32(4000000000), "test.int32": int32(-2000000000),
		"test.float32": float32(0.25), "test.bool": true, "test.string": "héllo",
		"test.uint64": uint64(1<<63 + 5), "test.int64": int64(-1 << 62), "test.float64": -1.5e300,
		"llama.block_count": uint32(1), "llama.rope.freq_base": float32(10000),
		"tokenizer.ggml.model": "gpt2",
	}
	for key, want := range scalars {
		if v, ok := f.Metadata(key); !ok || v.Interface() != want {
			t.Errorf("%s = %#v, want %#v", key, v.Interface(), want)
		}
	}
	if v, _ := f.Metadata("llama.context_length"); v.Type() != Uint32 {
		t.Errorf("llama.context_length has type %s", v.Type())
	} else if n, ok := v.Int(); !ok || n != 128 {
		t.Errorf("llama.context_length = %d", n)
	}
	vocab, _ := f.Metadata("tokenizer.ggml.tokens")
	if tokens, ok := vocab.Strings(); !ok || !slices.Equal(tokens, []string{"<s>", "</s>", "a", "b", "ab", "été"}) {
		t.Errorf("tokens %q", tokens)
	}
	scores, _ := f.Metadata("tokenizer.ggml.scores")
	if s, ok := scores.Float32s(); !ok || !slices.Equal(s, []float32{0, 0, -1, -1.5, -2, -2.25}) {
		t.Errorf("scores %v", s)
	}
	types, _ := f.Metadata("tokenizer.ggml.token_type")
	if tt, ok := types.Int32s(); !ok || !slices.Equal(tt, []int32{3, 3, 1, 1, 1, 1}) {
		t.Errorf("token types %v", tt)
	}
	nested, _ := f.Metadata("test.nested")
	if want := []Value{{[]int32{1, 2}}, {[]int32{3}}}; nested.ElemType() != Array || !reflect.DeepEqual(nested.Interface(), want) {
		t.Errorf("test.nested = %v", nested.Interface())
	}

	ctx := context.Background()
	checked := 0
	for _, w := range f.Tensors() {
		if strings.HasSuffix(w.Name, ".want") {
			continue
		}
		dt, err := w.DType()
		if w.Type == Q5_K {
			var te *TypeError
			if !errors.As(err, &te) || te.Tensor != w.Name || !errors.Is(err, tensor.ErrDType) {
				t.Errorf("%s: DType error %v", w.Name, err)
			}
			if _, err := f.Load(w.Name); !errors.Is(err, tensor.ErrDType) {
				t.Errorf("%s: Load error %v", w.Name, err)
			}
			continue
		}
		if err != nil {
			t.Fatalf("%s: %v", w.Name, err)
		}
		x, err := f.Load(w.Name)
		if err != nil || x.DType() != dt || !slices.Equal(x.Shape(), w.Shape) {
			t.Fatalf("%s: loaded %v, %v", w.Name, x, err)
		}
		got, err := x.Cast(tensor.Float32).Data[float32](ctx)
		if err != nil {
			t.Fatalf("%s: %v", w.Name, err)
		}
		want, ok := f.Tensor(w.Name + ".want")
		if w.Type == F32 {
			want, ok = w, true
		}
		if !ok || want.Type != F32 || !slices.Equal(want.Shape, w.Shape) {
			t.Fatalf("%s: no float32 tensor of shape %v to compare with", w.Name, w.Shape)
		}
		for i, x := range got {
			if bits := binary.LittleEndian.Uint32(want.Data[4*i:]); math.Float32bits(x) != bits {
				t.Fatalf("%s %s: element %d = %v, want %v", w.Name, w.Type, i, x, math.Float32frombits(bits))
			}
		}
		checked++
	}
	if checked != 7 {
		t.Errorf("checked %d tensors, want 7", checked)
	}
	if _, err := f.Load("missing"); !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("missing tensor: %v", err)
	}
	if w, _ := f.Tensor("blk.0.ffn_up.weight"); !slices.Equal(w.Shape, []int{2, 1, 2, 32}) || len(w.Data) != 4*34 {
		t.Errorf("4-D tensor of shape %v and %d bytes", w.Shape, len(w.Data))
	}
}

// metadata has a value of every type.
var metadata = []KV{
	{"general.architecture", ValueOf("llama")},
	{"u8", ValueOf(uint8(7))}, {"i8", ValueOf(int8(-7))},
	{"u16", ValueOf(uint16(700))}, {"i16", ValueOf(int16(-700))},
	{"u32", ValueOf(uint32(70000))}, {"i32", ValueOf(int32(-70000))},
	{"f32", ValueOf(float32(1.5))}, {"bool", ValueOf(true)}, {"str", ValueOf("")},
	{"u64", ValueOf(uint64(math.MaxUint64))}, {"i64", ValueOf(int64(math.MinInt64))},
	{"f64", ValueOf(math.Pi)},
	{"[]u8", ValueOf([]uint8{1, 2, 3})}, {"[]i8", ValueOf([]int8{-1})},
	{"[]u16", ValueOf([]uint16{1, 2})}, {"[]i16", ValueOf([]int16{-2})},
	{"[]u32", ValueOf([]uint32{})}, {"[]i32", ValueOf([]int32{-3, 4})},
	{"[]f32", ValueOf([]float32{0.5})}, {"[]bool", ValueOf([]bool{true, false})},
	{"[]str", ValueOf([]string{"a b", "", "ü"})}, {"[]u64", ValueOf([]uint64{1 << 40})},
	{"[]i64", ValueOf([]int64{-1 << 40})}, {"[]f64", ValueOf([]float64{-0.25})},
	{"[][]", ValueOf([]Value{ValueOf([]string{"x"}), ValueOf([]Value{ValueOf([]int8{1})})})},
}

// weights has a tensor of each type mimir reads and one it does not.
func weights() []Tensor {
	ramp := func(n int) []byte {
		b := make([]byte, n)
		for i := range b {
			b[i] = byte(i * 7)
		}
		return b
	}
	return []Tensor{
		{"q8", Q8_0, []int{2, 64}, ramp(4 * 34)},
		{"q4", Q4_K, []int{256}, ramp(144)},
		{"q6", Q6_K, []int{1, 1, 1, 256}, ramp(210)},
		{"f32", F32, []int{3}, ramp(12)},
		{"empty", F16, []int{0, 4}, []byte{}},
		{"bf16", BF16, []int{2, 2}, ramp(8)},
		{"q5", Q5_K, []int{256}, ramp(176)},
		{"scalar", F32, nil, ramp(4)},
	}
}

func encode(t testing.TB, metadata []KV, tensors []Tensor) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := Write(&buf, metadata, tensors); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func TestWriteRoundTrips(t *testing.T) {
	ws := weights()
	b := encode(t, metadata, ws)
	f, err := Parse(b)
	if err != nil {
		t.Fatal(err)
	}
	if f.Version != 3 || f.Alignment() != DefaultAlignment || f.Mapped() {
		t.Errorf("version %d, alignment %d, mapped %v", f.Version, f.Alignment(), f.Mapped())
	}
	for i, key := range f.Keys() {
		v, _ := f.Metadata(key)
		if key != metadata[i].Key || !reflect.DeepEqual(v, metadata[i].Value) {
			t.Errorf("key %d: %s = %v, want %s = %v", i, key, v.Interface(), metadata[i].Key, metadata[i].Value.Interface())
		}
	}
	got := f.Tensors()
	if len(got) != len(ws) {
		t.Fatalf("%d tensors, want %d", len(got), len(ws))
	}
	base := uintptr(unsafe.Pointer(unsafe.SliceData(b)))
	for i, w := range ws {
		g := got[i]
		if g.Name != w.Name || g.Type != w.Type || !slices.Equal(g.Shape, w.Shape) || !bytes.Equal(g.Data, w.Data) {
			t.Errorf("tensor %d = %s %s%v, want %s %s%v", i, g.Name, g.Type, g.Shape, w.Name, w.Type, w.Shape)
		}
		if len(g.Data) > 0 && (uintptr(unsafe.Pointer(unsafe.SliceData(g.Data)))-base)%DefaultAlignment != 0 {
			t.Errorf("%s is not aligned", g.Name)
		}
		if cap(g.Data) != len(g.Data) {
			t.Errorf("%s can grow over the next tensor", g.Name)
		}
	}

	// A file of version 2 has the same layout.
	v2 := slices.Clone(b)
	v2[4] = 2
	if f, err := Parse(v2); err != nil || f.Version != 2 {
		t.Errorf("version 2: %v", err)
	}

	// The alignment comes from the metadata.
	aligned := encode(t, []KV{{"general.alignment", ValueOf(uint32(256))}}, ws[:2])
	if f, err := Parse(aligned); err != nil || f.Alignment() != 256 {
		t.Fatalf("alignment 256: %v", err)
	} else if q4, _ := f.Tensor("q4"); uintptr(unsafe.Pointer(unsafe.SliceData(q4.Data)))-uintptr(unsafe.Pointer(&aligned[0])) != 2*256 {
		t.Errorf("q4 is not at the 2nd block of 256 bytes after the header's")
	}
}

func TestWriteRejectsMalformedInput(t *testing.T) {
	for name, c := range map[string]struct {
		metadata []KV
		tensors  []Tensor
	}{
		"alignment of 3":           {metadata: []KV{{"general.alignment", ValueOf(uint32(3))}}},
		"alignment as int64":       {metadata: []KV{{"general.alignment", ValueOf(int64(32))}}},
		"value of a Go int":        {metadata: []KV{{"k", ValueOf(1)}}},
		"no value":                 {metadata: []KV{{"k", Value{}}}},
		"array of non-arrays":      {metadata: []KV{{"k", ValueOf([]Value{ValueOf(1.5)})}}},
		"data too short":           {tensors: []Tensor{{"w", F32, []int{2}, make([]byte, 4)}}},
		"partial block":            {tensors: []Tensor{{"w", Q8_0, []int{16}, make([]byte, 17)}}},
		"negative dimension":       {tensors: []Tensor{{"w", F32, []int{-1}, nil}}},
		"five dimensions":          {tensors: []Tensor{{"w", F32, []int{1, 1, 1, 1, 1}, make([]byte, 4)}}},
		"unknown type":             {tensors: []Tensor{{"w", Type(99), []int{1}, make([]byte, 4)}}},
		"arrays nested too deeply": {metadata: []KV{{"k", deep(maxDepth + 1)}}},
	} {
		if err := Write(new(bytes.Buffer), c.metadata, c.tensors); !errors.Is(err, ErrFormat) {
			t.Errorf("%s: error %v", name, err)
		}
	}
	// Arrays nested as deeply as Parse reads them are written.
	if _, err := Parse(encode(t, []KV{{"k", deep(maxDepth)}}, nil)); err != nil {
		t.Error(err)
	}
}

// deep returns arrays nested depth deep.
func deep(depth int) Value {
	v := ValueOf([]uint8{1})
	for range depth - 1 {
		v = ValueOf([]Value{v})
	}
	return v
}

// header builds a GGUF header field by field.
type header struct{ bytes.Buffer }

func newHeader(tensors, kvs uint64) *header {
	h := new(header)
	h.WriteString(Magic)
	return h.u32(3).u64(tensors).u64(kvs)
}

func (h *header) u32(v uint32) *header {
	h.Write(binary.LittleEndian.AppendUint32(nil, v))
	return h
}

func (h *header) u64(v uint64) *header {
	h.Write(binary.LittleEndian.AppendUint64(nil, v))
	return h
}

func (h *header) str(s string) *header {
	h.u64(uint64(len(s)))
	h.WriteString(s)
	return h
}

// tensor appends a tensor's info, with ggml's dimensions ne.
func (h *header) tensor(name string, typ Type, off uint64, ne ...uint64) *header {
	h.str(name).u32(uint32(len(ne)))
	for _, d := range ne {
		h.u64(d)
	}
	return h.u32(uint32(typ)).u64(off)
}

// data pads the header to 32 bytes and appends n bytes of data.
func (h *header) data(n int) []byte {
	h.Write(make([]byte, (32-h.Len()%32)%32+n))
	return h.Bytes()
}

func TestParseRejectsMalformedFiles(t *testing.T) {
	valid := encode(t, metadata, weights())
	for name, b := range map[string][]byte{
		"empty":             nil,
		"magic":             []byte("GGUG\x03\x00\x00\x00"),
		"version 1":         append([]byte(Magic), 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
		"big-endian":        append([]byte(Magic), 0, 0, 0, 3),
		"version 4":         append([]byte(Magic), 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
		"tensor count":      newHeader(1<<62, 0).Bytes(),
		"key count":         newHeader(0, math.MaxUint64).Bytes(),
		"key length":        newHeader(0, 1).u64(math.MaxUint64).u32(0).data(64),
		"value type":        newHeader(0, 1).str("k").u32(13).Bytes(),
		"bool":              append(newHeader(0, 1).str("k").u32(uint32(Bool)).Bytes(), 2),
		"array type":        newHeader(0, 1).str("k").u32(uint32(Array)).u32(13).u64(1).data(8),
		"array length":      newHeader(0, 1).str("k").u32(uint32(Array)).u32(uint32(Uint64)).u64(1 << 60).data(64),
		"string array":      newHeader(0, 1).str("k").u32(uint32(Array)).u32(uint32(String)).u64(1 << 20).data(64),
		"nested arrays":     nestedArrays(maxDepth + 1),
		"repeated key":      newHeader(0, 2).str("k").u32(uint32(Uint32)).u32(1).str("k").u32(uint32(Uint32)).u32(2).data(0),
		"alignment of 48":   newHeader(0, 1).str("general.alignment").u32(uint32(Uint32)).u32(48).data(0),
		"alignment as u64":  newHeader(0, 1).str("general.alignment").u32(uint32(Uint64)).u64(32).data(0),
		"five dimensions":   newHeader(1, 0).tensor("w", F32, 0, 1, 1, 1, 1, 1).data(4),
		"dimension count":   newHeader(1, 0).str("w").u32(math.MaxUint32).data(64),
		"dimensions":        newHeader(1, 0).tensor("w", F32, 0, 1<<40, 1<<40).data(64),
		"block product":     newHeader(1, 0).tensor("w", Q8_0, 0, 32, 1<<31, 1<<31, 1<<31).data(64),
		"partial block":     newHeader(1, 0).tensor("w", Q8_0, 0, 48).data(64),
		"quantized scalar":  newHeader(1, 0).tensor("w", Q8_0, 0).data(64),
		"unknown type":      newHeader(1, 0).tensor("w", Type(5), 0, 4).data(64),
		"misaligned":        newHeader(1, 0).tensor("w", F32, 4, 1).data(64),
		"offset":            newHeader(1, 0).tensor("w", F32, 1<<62, 1).data(64),
		"past the end":      newHeader(1, 0).tensor("w", F32, 32, 9).data(64),
		"overlap":           newHeader(2, 0).tensor("a", F32, 0, 16).tensor("b", F32, 32, 1).data(64),
		"repeated tensor":   newHeader(2, 0).tensor("a", F32, 0, 1).tensor("a", F32, 32, 1).data(64),
		"no data":           newHeader(1, 0).tensor("w", F32, 0, 1).Bytes(),
		"tensor name":       newHeader(1, 0).u64(1 << 40).data(64),
		"truncated tensors": valid[:len(valid)-1],
	} {
		f, err := Parse(b)
		if !errors.Is(err, ErrFormat) || f != nil {
			t.Errorf("%s: error %v", name, err)
		}
	}
	// Every prefix of a file, down to its header, is malformed: the
	// last tensor ends the file.
	for n := range len(valid) {
		if _, err := Parse(valid[:n]); err == nil {
			t.Fatalf("a file cut to %d of %d bytes parsed", n, len(valid))
		}
	}
	// Tensors of no elements may share an offset, and a tensor's data
	// may end the file without padding.
	b := newHeader(3, 0).tensor("a", F32, 0, 0).tensor("b", F32, 0, 1).tensor("c", F16, 0, 0).data(4)
	if f, err := Parse(b); err != nil {
		t.Error(err)
	} else if a, _ := f.Tensor("a"); len(a.Data) != 0 || !slices.Equal(a.Shape, []int{0}) {
		t.Errorf("empty tensor of shape %v, %d bytes", a.Shape, len(a.Data))
	}
}

// nestedArrays returns a file whose one key holds arrays nested depth
// deep.
func nestedArrays(depth int) []byte {
	h := newHeader(0, 1).str("k").u32(uint32(Array))
	for range depth - 1 {
		h.u32(uint32(Array)).u64(1)
	}
	return h.u32(uint32(Uint8)).u64(1).data(1)
}

// canMap reports whether Open maps files on this platform.
func canMap() bool {
	switch runtime.GOOS {
	case "plan9", "js", "wasip1":
		return false
	}
	return true
}

// mapsPath reports whether the process maps the file at path, where that
// can be seen.
func mapsPath(t *testing.T, path string) (mapped, known bool) {
	t.Helper()
	if runtime.GOOS != "linux" {
		return false, false
	}
	maps, err := os.ReadFile("/proc/self/maps")
	if err != nil {
		t.Fatal(err)
	}
	return strings.Contains(string(maps), path), true
}

func writeTestFile(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "t.gguf")
	if err := os.WriteFile(path, encode(t, metadata, weights()), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestOpenReportsMissingAndCorruptFiles(t *testing.T) {
	if _, err := Open(filepath.Join(t.TempDir(), "missing.gguf")); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("missing file: %v", err)
	}
	path := filepath.Join(t.TempDir(), "bad.gguf")
	if err := os.WriteFile(path, []byte("GGUF\x09"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Open(path); !errors.Is(err, ErrFormat) || !strings.Contains(err.Error(), path) {
		t.Errorf("corrupt file: %v", err)
	}
	if mapped, known := mapsPath(t, path); known && mapped {
		t.Error("a corrupt file stays mapped")
	}
}

// TestHoldKeepsTheMappingPastClose checks that the mapping lasts while a
// reader that Hold added keeps it, and goes with the last of them.
func TestHoldKeepsTheMappingPastClose(t *testing.T) {
	path := writeTestFile(t)
	f, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	if f.Mapped() != canMap() {
		t.Errorf("Mapped() = %v on %s", f.Mapped(), runtime.GOOS)
	}
	if !f.Mapped() {
		f.Hold()()
		if err := f.Close(); err != nil {
			t.Fatal(err)
		}
		t.Skip("files are not mapped on " + runtime.GOOS)
	}
	w, _ := f.Tensor("q8")
	x := tensor.FromBytesFunc(tensor.Q8_0, w.Data, f.Hold(), w.Shape...)
	held := f.Hold()
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Errorf("second Close: %v", err)
	}
	if _, ok := f.Tensor("q8"); ok || len(f.Tensors()) != 0 {
		t.Error("a closed file still has tensors")
	}
	f.Hold()() // a closed file adds no hold
	held()
	if _, err := x.Cast(tensor.Float32).Data[float32](context.Background()); err != nil {
		t.Fatal(err)
	}
	if mapped, known := mapsPath(t, path); known && !mapped {
		t.Error("the file was unmapped while a tensor read it")
	}
	runtime.KeepAlive(x)
	deadline := time.Now().Add(10 * time.Second)
	for {
		runtime.GC()
		if mapped, known := mapsPath(t, path); !known || !mapped {
			return
		}
		if time.Now().After(deadline) {
			t.Fatal("the file is still mapped after its last reader went")
		}
		time.Sleep(time.Millisecond)
	}
}

func TestTypes(t *testing.T) {
	for typ, want := range map[Type][2]int{Q8_0: {32, 34}, Q4_K: {256, 144}, Q6_K: {256, 210}, BF16: {1, 2}, Type(99): {0, 0}} {
		if typ.BlockSize() != want[0] || typ.BlockBytes() != want[1] {
			t.Errorf("%s: blocks of %d in %d bytes", typ, typ.BlockSize(), typ.BlockBytes())
		}
	}
	for _, typ := range []Type{Q8_0, Q4_K, Q6_K, F32, F16, BF16} {
		dt, ok := typ.DType()
		if !ok || dt.BlockSize() != typ.BlockSize() || dt.BlockBytes() != typ.BlockBytes() {
			t.Errorf("%s reads as %s", typ, dt)
		}
	}
	if Q4_K.String() != "q4_K" || Type(99).String() != "Type(99)" || Float64.String() != "float64" {
		t.Error(Q4_K, Type(99), Float64)
	}
}

func TestValueAccessors(t *testing.T) {
	if n, ok := ValueOf(uint64(math.MaxUint64)).Int(); ok {
		t.Errorf("max uint64 as an int: %d", n)
	}
	if n, ok := ValueOf(uint64(math.MaxUint64)).Uint(); !ok || n != math.MaxUint64 {
		t.Errorf("max uint64: %d", n)
	}
	if _, ok := ValueOf(int8(-1)).Uint(); ok {
		t.Error("-1 as a uint")
	}
	if x, ok := ValueOf(int32(-3)).Float(); !ok || x != -3 {
		t.Errorf("int32 as a float: %v", x)
	}
	if _, ok := ValueOf("1").Int(); ok {
		t.Error("a string as an int")
	}
	if s, ok := ValueOf([]uint8{255}).Int32s(); !ok || s[0] != 255 {
		t.Errorf("uint8s as int32s: %v", s)
	}
	if v := ValueOf([]string{"a", "b"}); v.String() != "[2]string" || v.Len() != 2 || v.Type() != Array || v.ElemType() != String {
		t.Errorf("array of strings: %v", v)
	}
}

func FuzzOpen(f *testing.F) {
	small, err := os.ReadFile(fixture)
	if err != nil {
		f.Fatal(err)
	}
	f.Add(small)
	f.Add(encode(f, metadata, weights()))
	f.Add(nestedArrays(3))
	f.Fuzz(func(t *testing.T, b []byte) {
		gf, err := Parse(b)
		if err != nil {
			if gf != nil || !errors.Is(err, ErrFormat) {
				t.Fatalf("file %v with error %v", gf, err)
			}
			return
		}
		for _, key := range gf.Keys() {
			v, _ := gf.Metadata(key)
			_ = v.String()
		}
		// Every tensor lies within the file, and those that mimir reads,
		// whatever their bytes, dequantize.
		for _, w := range gf.Tensors() {
			n := 1
			for _, d := range w.Shape {
				n *= d
			}
			dt, err := w.DType()
			if err != nil || n > 1<<14 || n == 0 {
				continue
			}
			x := tensor.FromBytes(dt, w.Data, w.Shape...).Cast(tensor.Float32)
			if got, err := x.Data[float32](context.Background()); err != nil || len(got) != n {
				t.Fatalf("%s %s%v: %d values, %v", w.Name, w.Type, w.Shape, len(got), err)
			}
		}
	})
}
