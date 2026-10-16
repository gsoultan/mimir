package parquet

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/klauspost/compress/snappy"
)

// The fixtures in testdata, written by tools/parquet/gen.py, which also
// gives the formula for row i of every column.
var fixtures = map[string]struct {
	rows   int64
	groups []int64
}{
	"plain_v1":            {1000, []int64{1000}},
	"dict_v1_snappy":      {1000, []int64{1000}},
	"dict_v2_gzip":        {1000, []int64{1000}},
	"dict_v2_zstd":        {1000, []int64{1000}},
	"delta_v2":            {1000, []int64{1000}},
	"delta_byte_array_v2": {1000, []int64{1000}},
	"pages_v2_snappy":     {10000, []int64{4000, 4000, 2000}},
	"empty":               {0, []int64{0}},
}

var fixtureColumns = []Column{
	{Name: "i32", Type: Int32, Nullable: true},
	{Name: "i64", Type: Int64, Nullable: true},
	{Name: "f32", Type: Float, Nullable: true},
	{Name: "f64", Type: Double, Nullable: true},
	{Name: "flag", Type: Boolean, Nullable: true},
	{Name: "name", Type: ByteArray, Nullable: true},
	{Name: "opt", Type: Float, Nullable: true},
}

func i32Of(i int64) int32   { return int32(7*i - 100) }
func i64Of(i int64) int64   { return i*10_000_000_000 - 5 }
func f32Of(i int64) float32 { return float32(float64(i) / 3) } // Python divides in float64, pyarrow narrows
func f64Of(i int64) float64 { return math.Sqrt(float64(i)) }
func flagOf(i int64) bool   { return i%3 == 0 }
func nameOf(i int64) string { return "s" + strconv.FormatInt(i%17, 10) }
func optNull(i int64) bool  { return i%5 == 0 }
func optOf(i int64) float32 { return float32(float64(i) / 2) }

func b2i[T int32 | int64 | float32 | float64](b bool) T {
	if b {
		return 1
	}
	return 0
}

var chunkSizes = []int{1, 7, 100, 100000}

func openFixture(t *testing.T, name string) *File {
	t.Helper()
	f, err := OpenFile(filepath.Join("testdata", name+".parquet"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { f.Close() })
	return f
}

func fixtureBytes(t testing.TB, name string) []byte {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", name+".parquet"))
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func reader(t *testing.T, f *File, group int, name string) *Reader {
	t.Helper()
	i, ok := f.Column(name)
	if !ok {
		t.Fatalf("no column %q", name)
	}
	r, err := f.Reader(group, i)
	if err != nil {
		t.Fatalf("Reader(%d, %q): %v", group, name, err)
	}
	return r
}

// readAll reads a chunk to its end in pieces of size values, checking the
// contract of the Read methods along the way.
func readAll[T any](t *testing.T, read func([]T, []bool) (int, error), rows int64, size int, withNulls bool) ([]T, []bool) {
	t.Helper()
	var vals []T
	var nulls []bool
	buf := make([]T, size)
	var nb []bool
	if withNulls {
		nb = make([]bool, size)
	}
	for {
		n, err := read(buf, nb)
		if err == io.EOF {
			if n != 0 {
				t.Fatalf("read %d values with io.EOF", n)
			}
			break
		}
		if err != nil {
			t.Fatalf("after %d values: %v", len(vals), err)
		}
		if n <= 0 || n > size {
			t.Fatalf("read %d values into %d", n, size)
		}
		vals = append(vals, buf[:n]...)
		if withNulls {
			nulls = append(nulls, nb[:n]...)
		}
	}
	if int64(len(vals)) != rows {
		t.Fatalf("read %d values, want %d", len(vals), rows)
	}
	if n, err := read(buf, nb); n != 0 || err != io.EOF {
		t.Fatalf("read after the end: %d, %v", n, err)
	}
	return vals, nulls
}

// expect checks every value of got against the formula for its row.
func expect[T comparable](t *testing.T, what string, got []T, start int64, want func(int64) T) {
	t.Helper()
	for i, v := range got {
		if w := want(start + int64(i)); v != w {
			t.Fatalf("%s: row %d = %v, want %v", what, start+int64(i), v, w)
		}
	}
}

func allFalse(t *testing.T, what string, nulls []bool) {
	t.Helper()
	if i := slices.Index(nulls, true); i >= 0 {
		t.Fatalf("%s: row %d is null", what, i)
	}
}

func wantErr(t *testing.T, what string, err, sentinel error, text string) {
	t.Helper()
	if !errors.Is(err, sentinel) {
		t.Fatalf("%s: got %v, want %v", what, err, sentinel)
	}
	if !strings.Contains(err.Error(), text) {
		t.Fatalf("%s: %q does not mention %q", what, err, text)
	}
}

func TestFixtures(t *testing.T) {
	for name, want := range fixtures {
		t.Run(name, func(t *testing.T) {
			f := openFixture(t, name)
			if f.Rows() != want.rows {
				t.Errorf("Rows() = %d, want %d", f.Rows(), want.rows)
			}
			if f.RowGroups() != len(want.groups) {
				t.Fatalf("RowGroups() = %d, want %d", f.RowGroups(), len(want.groups))
			}
			if got := f.Columns(); !slices.Equal(got, fixtureColumns) {
				t.Fatalf("Columns() = %v", got)
			}
			for i, c := range fixtureColumns {
				if idx, ok := f.Column(c.Name); !ok || idx != i {
					t.Errorf("Column(%q) = %d, %v", c.Name, idx, ok)
				}
			}
			if _, ok := f.Column("nope"); ok {
				t.Error("Column(nope) found")
			}
			if !strings.HasPrefix(f.CreatedBy(), "parquet-cpp-arrow") {
				t.Errorf("CreatedBy() = %q", f.CreatedBy())
			}
			if _, ok := f.Metadata()["ARROW:schema"]; !ok {
				t.Errorf("Metadata() = %v", f.Metadata())
			}
			for _, size := range chunkSizes {
				t.Run(fmt.Sprint("chunk", size), func(t *testing.T) {
					start := int64(0)
					for g, rows := range want.groups {
						if got := f.RowGroupRows(g); got != rows {
							t.Fatalf("RowGroupRows(%d) = %d, want %d", g, got, rows)
						}
						checkGroup(t, f, g, start, rows, size)
						start += rows
					}
				})
			}
		})
	}
}

// checkGroup reads every column of a row group with every applicable
// method.
func checkGroup(t *testing.T, f *File, g int, start, rows int64, size int) {
	// i32: exact as int32, widened and converted to floats.
	v32, nulls := readAll(t, reader(t, f, g, "i32").ReadInt32, rows, size, true)
	expect(t, "i32", v32, start, i32Of)
	allFalse(t, "i32", nulls)
	v64, _ := readAll(t, reader(t, f, g, "i32").ReadInt64, rows, size, false)
	expect(t, "i32 as int64", v64, start, func(i int64) int64 { return int64(i32Of(i)) })
	vf32, _ := readAll(t, reader(t, f, g, "i32").ReadFloat32, rows, size, false)
	expect(t, "i32 as float32", vf32, start, func(i int64) float32 { return float32(i32Of(i)) })
	vf64, _ := readAll(t, reader(t, f, g, "i32").ReadFloat64, rows, size, false)
	expect(t, "i32 as float64", vf64, start, func(i int64) float64 { return float64(i32Of(i)) })
	if _, err := reader(t, f, g, "i32").ReadStrings(make([]string, 1), nil); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("ReadStrings of i32: %v", err)
	}

	// i64: exact, as floats, and rejected as int32 once a value does not fit.
	v64, nulls = readAll(t, reader(t, f, g, "i64").ReadInt64, rows, size, true)
	expect(t, "i64", v64, start, i64Of)
	allFalse(t, "i64", nulls)
	vf32, _ = readAll(t, reader(t, f, g, "i64").ReadFloat32, rows, size, false)
	expect(t, "i64 as float32", vf32, start, func(i int64) float32 { return float32(i64Of(i)) })
	vf64, _ = readAll(t, reader(t, f, g, "i64").ReadFloat64, rows, size, false)
	expect(t, "i64 as float64", vf64, start, func(i int64) float64 { return float64(i64Of(i)) })
	if rows > 1 {
		r := reader(t, f, g, "i64")
		var err error
		for err == nil {
			_, err = r.ReadInt32(make([]int32, size), nil)
		}
		wantErr(t, "i64 as int32", err, ErrUnsupported, "does not fit")
		if _, err2 := r.ReadInt32(make([]int32, size), nil); err2 != err {
			t.Fatalf("error is not sticky: %v", err2)
		}
	}

	// f32 and f64: exact, and converted to each other.
	vf32, nulls = readAll(t, reader(t, f, g, "f32").ReadFloat32, rows, size, true)
	expect(t, "f32", vf32, start, f32Of)
	allFalse(t, "f32", nulls)
	vf64, _ = readAll(t, reader(t, f, g, "f32").ReadFloat64, rows, size, false)
	expect(t, "f32 as float64", vf64, start, func(i int64) float64 { return float64(f32Of(i)) })
	if _, err := reader(t, f, g, "f32").ReadInt32(make([]int32, 1), nil); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("ReadInt32 of f32: %v", err)
	}
	vf64, nulls = readAll(t, reader(t, f, g, "f64").ReadFloat64, rows, size, true)
	expect(t, "f64", vf64, start, f64Of)
	allFalse(t, "f64", nulls)
	vf32, _ = readAll(t, reader(t, f, g, "f64").ReadFloat32, rows, size, false)
	expect(t, "f64 as float32", vf32, start, func(i int64) float32 { return float32(f64Of(i)) })
	if _, err := reader(t, f, g, "f64").ReadInt64(make([]int64, 1), nil); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("ReadInt64 of f64: %v", err)
	}

	// flag: 0 or 1 in every numeric type.
	v32, nulls = readAll(t, reader(t, f, g, "flag").ReadInt32, rows, size, true)
	expect(t, "flag", v32, start, func(i int64) int32 { return b2i[int32](flagOf(i)) })
	allFalse(t, "flag", nulls)
	v64, _ = readAll(t, reader(t, f, g, "flag").ReadInt64, rows, size, false)
	expect(t, "flag as int64", v64, start, func(i int64) int64 { return b2i[int64](flagOf(i)) })
	vf32, _ = readAll(t, reader(t, f, g, "flag").ReadFloat32, rows, size, false)
	expect(t, "flag as float32", vf32, start, func(i int64) float32 { return b2i[float32](flagOf(i)) })
	vf64, _ = readAll(t, reader(t, f, g, "flag").ReadFloat64, rows, size, false)
	expect(t, "flag as float64", vf64, start, func(i int64) float64 { return b2i[float64](flagOf(i)) })
	if _, err := reader(t, f, g, "flag").ReadBytes(make([][]byte, 1), nil); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("ReadBytes of flag: %v", err)
	}

	// name: as strings and as bytes.
	vs, nulls := readAll(t, reader(t, f, g, "name").ReadStrings, rows, size, true)
	expect(t, "name", vs, start, nameOf)
	allFalse(t, "name", nulls)
	vb, _ := readAll(t, reader(t, f, g, "name").ReadBytes, rows, size, false)
	for i, b := range vb {
		if string(b) != nameOf(start+int64(i)) || cap(b) != len(b) {
			t.Fatalf("name: row %d = %q with capacity %d", start+int64(i), b, cap(b))
		}
	}
	if _, err := reader(t, f, g, "name").ReadFloat32(make([]float32, 1), nil); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("ReadFloat32 of name: %v", err)
	}

	// opt: nulls reported through the slice, or as NaN without one.
	vf32, nulls = readAll(t, reader(t, f, g, "opt").ReadFloat32, rows, size, true)
	for i := range vf32 {
		row := start + int64(i)
		if nulls[i] != optNull(row) {
			t.Fatalf("opt: null at row %d = %v", row, nulls[i])
		}
		if want := optOf(row); nulls[i] && vf32[i] != 0 || !nulls[i] && vf32[i] != want {
			t.Fatalf("opt: row %d = %v, want %v", row, vf32[i], want)
		}
	}
	vf32, _ = readAll(t, reader(t, f, g, "opt").ReadFloat32, rows, size, false)
	for i, v := range vf32 {
		row := start + int64(i)
		if optNull(row) != (v != v) || !optNull(row) && v != optOf(row) {
			t.Fatalf("opt: row %d = %v", row, v)
		}
	}
	vf64, nulls = readAll(t, reader(t, f, g, "opt").ReadFloat64, rows, size, true)
	for i, v := range vf64 {
		row := start + int64(i)
		if nulls[i] != optNull(row) || !nulls[i] && v != float64(optOf(row)) {
			t.Fatalf("opt as float64: row %d = %v, null %v", row, v, nulls[i])
		}
	}
	vf64, _ = readAll(t, reader(t, f, g, "opt").ReadFloat64, rows, size, false)
	for i, v := range vf64 {
		if row := start + int64(i); optNull(row) != math.IsNaN(v) {
			t.Fatalf("opt as float64: row %d = %v", row, v)
		}
	}
	if _, err := reader(t, f, g, "opt").ReadInt32(make([]int32, 1), nil); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("ReadInt32 of opt: %v", err)
	}
}

func TestReaderRejectsBadArguments(t *testing.T) {
	f := openFixture(t, "plain_v1")
	for _, c := range [][2]int{{-1, 0}, {1, 0}, {0, -1}, {0, 7}} {
		if _, err := f.Reader(c[0], c[1]); err == nil {
			t.Errorf("Reader(%d, %d) succeeded", c[0], c[1])
		}
	}
	r := reader(t, f, 0, "i32")
	if _, err := r.ReadInt32(make([]int32, 10), make([]bool, 9)); err == nil {
		t.Error("short nulls slice accepted")
	}
	if n, err := r.ReadInt32(nil, nil); n != 0 || err != nil {
		t.Errorf("empty read: %d, %v", n, err)
	}
}

func TestOpenRejectsMalformedFiles(t *testing.T) {
	good := fixtureBytes(t, "plain_v1")
	patch := func(off int, b ...byte) []byte {
		c := slices.Clone(good)
		copy(c[off:], b)
		return c
	}
	tests := map[string]struct {
		b    []byte
		want error
	}{
		"empty":            {nil, ErrFormat},
		"short":            {[]byte("PAR1PAR1"), ErrFormat},
		"truncated":        {good[:len(good)-100], ErrFormat},
		"half":             {good[:len(good)/2], ErrFormat},
		"bad magic":        {patch(0, 'X'), ErrFormat},
		"bad footer magic": {patch(len(good)-4, 'P', 'A', 'R', '2'), ErrFormat},
		"encrypted":        {patch(len(good)-4, 'P', 'A', 'R', 'E'), ErrUnsupported},
		"footer length":    {patch(len(good)-8, 0xff, 0xff, 0xff, 0x7f), ErrFormat},
		"footer garbage":   {patch(len(good)-8-int(binary.LittleEndian.Uint32(good[len(good)-8:])), 0xff, 0xff, 0xff), ErrFormat},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			f, err := Open(bytes.NewReader(tc.b), int64(len(tc.b)))
			if !errors.Is(err, tc.want) {
				t.Fatalf("Open: %v, want %v", err, tc.want)
			}
			if f != nil {
				t.Error("Open returned a file with an error")
			}
		})
	}
	if _, err := OpenFile(filepath.Join("testdata", "missing.parquet")); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("OpenFile(missing): %v", err)
	}
}

// What follows writes files by hand to reach what the fixtures do not:
// nulls in integer and string columns, required, nested, INT96 and
// fixed-length columns, unknown footer fields, and malformed pages.

// tw writes the thrift compact protocol.
type tw struct {
	b    []byte
	last int16
}

func zz(v int64) uint64 { return uint64((v << 1) ^ (v >> 63)) }

func (w *tw) uvarint(v uint64) { w.b = binary.AppendUvarint(w.b, v) }

func (w *tw) field(id int16, typ byte) {
	if d := id - w.last; d > 0 && d <= 15 {
		w.b = append(w.b, byte(d)<<4|typ)
	} else {
		w.b = append(w.b, typ)
		w.uvarint(zz(int64(id)))
	}
	w.last = id
}

func (w *tw) i32(id int16, v int32)        { w.field(id, ttI32); w.uvarint(zz(int64(v))) }
func (w *tw) i64(id int16, v int64)        { w.field(id, ttI64); w.uvarint(zz(v)) }
func (w *tw) bin(id int16, s string)       { w.field(id, ttBinary); w.str(s) }
func (w *tw) str(s string)                 { w.uvarint(uint64(len(s))); w.b = append(w.b, s...) }
func (w *tw) list(id int16, e byte, n int) { w.field(id, ttList); w.listHeader(e, n) }

func (w *tw) listHeader(elem byte, n int) {
	if n < 15 {
		w.b = append(w.b, byte(n)<<4|elem)
	} else {
		w.b = append(w.b, 0xf0|elem)
		w.uvarint(uint64(n))
	}
}

func (w *tw) strct(id int16, body func()) { w.field(id, ttStruct); w.elem(body) }

// elem writes a struct as a field value or list element.
func (w *tw) elem(body func()) {
	saved := w.last
	w.last = 0
	body()
	w.b = append(w.b, ttStop)
	w.last = saved
}

// tchunk is a column chunk of one data page.
type tchunk struct {
	typ       Type
	path      string
	numValues int32
	nulls     int32
	enc       encoding
	codec     codec
	levels    []byte // definition levels, without a length prefix
	values    []byte
	v2        bool  // a data page V2 rather than V1
	compress  bool  // with Snappy, which sets the codec
	sizeFix   int32 // added to the page's declared sizes
}

type tfile struct {
	schema  []schemaElement
	chunks  []tchunk
	numRows int64
	extra   func(w *tw) // extra FileMetaData fields
}

func (tf tfile) bytes() []byte {
	b := []byte(magic)
	type loc struct{ off, size int64 }
	locs := make([]loc, len(tf.chunks))
	for i := range tf.chunks {
		c := &tf.chunks[i]
		if c.compress {
			c.codec = codecSnappy
		}
		off := len(b)
		h := &tw{}
		var page []byte
		if c.v2 {
			stored := c.values
			if c.compress {
				stored = snappy.Encode(nil, c.values)
			}
			h.i32(1, int32(dataPageV2))
			h.i32(2, int32(len(c.levels)+len(c.values))+c.sizeFix)
			h.i32(3, int32(len(c.levels)+len(stored))+c.sizeFix)
			h.strct(8, func() {
				h.i32(1, c.numValues)
				h.i32(2, c.nulls)
				h.i32(3, c.numValues)
				h.i32(4, int32(c.enc))
				h.i32(5, int32(len(c.levels)))
				h.i32(6, 0)
				if !c.compress {
					h.field(7, ttFalse)
				}
			})
			page = concat(c.levels, stored)
		} else {
			raw := c.values
			if len(c.levels) > 0 {
				raw = concat(prefixed(c.levels), c.values)
			}
			page = raw
			if c.compress {
				page = snappy.Encode(nil, raw)
			}
			h.i32(1, int32(dataPage))
			h.i32(2, int32(len(raw))+c.sizeFix)
			h.i32(3, int32(len(page))+c.sizeFix)
			h.strct(5, func() {
				h.i32(1, c.numValues)
				h.i32(2, int32(c.enc))
				h.i32(3, int32(encRLE))
				h.i32(4, int32(encRLE))
			})
		}
		h.b = append(h.b, ttStop)
		b = append(append(b, h.b...), page...)
		locs[i] = loc{int64(off), int64(len(b) - off)}
	}
	w := &tw{}
	w.i32(1, 1)
	w.list(2, ttStruct, len(tf.schema))
	for _, e := range tf.schema {
		w.elem(func() {
			if e.hasType {
				w.i32(1, int32(e.typ))
			}
			if e.typeLength != 0 {
				w.i32(2, e.typeLength)
			}
			w.i32(3, int32(e.repetition))
			w.bin(4, e.name)
			if e.numChildren != 0 {
				w.i32(5, e.numChildren)
			}
		})
	}
	w.i64(3, tf.numRows)
	w.list(4, ttStruct, 1)
	w.elem(func() {
		w.list(1, ttStruct, len(tf.chunks))
		for i, c := range tf.chunks {
			w.elem(func() {
				w.i64(2, locs[i].off)
				w.strct(3, func() {
					w.i32(1, int32(c.typ))
					w.list(2, ttI32, 2)
					w.uvarint(zz(int64(c.enc)))
					w.uvarint(zz(int64(encRLE)))
					path := strings.Split(c.path, ".")
					w.list(3, ttBinary, len(path))
					for _, p := range path {
						w.str(p)
					}
					w.i32(4, int32(c.codec))
					w.i64(5, int64(c.numValues))
					w.i64(6, locs[i].size)
					w.i64(7, locs[i].size)
					w.i64(9, locs[i].off)
				})
			})
		}
		w.i64(2, int64(len(b)-len(magic)))
		w.i64(3, tf.numRows)
	})
	w.bin(6, "mimir test")
	if tf.extra != nil {
		tf.extra(w)
	}
	w.b = append(w.b, ttStop)
	b = append(b, w.b...)
	b = binary.LittleEndian.AppendUint32(b, uint32(len(w.b)))
	return append(b, magic...)
}

func openBytes(t *testing.T, b []byte) *File {
	t.Helper()
	f, err := Open(bytes.NewReader(b), int64(len(b)))
	if err != nil {
		t.Fatal(err)
	}
	return f
}

// rleBits encodes bits as one bit-packed run of the RLE hybrid.
func rleBits(bits ...uint8) []byte {
	groups := (len(bits) + 7) / 8
	data := binary.AppendUvarint(nil, uint64(groups)<<1|1)
	packed := make([]byte, groups)
	for i, b := range bits {
		packed[i>>3] |= b << (i & 7)
	}
	return append(data, packed...)
}

// prefixed adds the 4-byte length prefix of data page V1 levels and of RLE
// booleans.
func prefixed(b []byte) []byte {
	return append(binary.LittleEndian.AppendUint32(nil, uint32(len(b))), b...)
}

func plainInt32(vals ...int32) []byte {
	var b []byte
	for _, v := range vals {
		b = binary.LittleEndian.AppendUint32(b, uint32(v))
	}
	return b
}

func plainInt64(vals ...int64) []byte {
	var b []byte
	for _, v := range vals {
		b = binary.LittleEndian.AppendUint64(b, uint64(v))
	}
	return b
}

func plainStrings(vals ...string) []byte {
	var b []byte
	for _, v := range vals {
		b = append(binary.LittleEndian.AppendUint32(b, uint32(len(v))), v...)
	}
	return b
}

func concat(parts ...[]byte) []byte { return bytes.Join(parts, nil) }

func optional(name string, typ Type) schemaElement {
	return schemaElement{typ: typ, hasType: true, repetition: repOptional, name: name}
}

func required(name string, typ Type) schemaElement {
	return schemaElement{typ: typ, hasType: true, repetition: repRequired, name: name}
}

func root(children int32) schemaElement {
	return schemaElement{name: "schema", numChildren: children}
}

func TestNullsWithoutSlice(t *testing.T) {
	f := openBytes(t, tfile{
		schema: []schemaElement{root(2), optional("n", Int32), optional("s", ByteArray)},
		chunks: []tchunk{
			{typ: Int32, path: "n", numValues: 3, levels: rleBits(1, 0, 1), values: plainInt32(1, 3)},
			{typ: ByteArray, path: "s", numValues: 3, levels: rleBits(1, 0, 1), values: plainStrings("a", "bc")},
		},
		numRows: 3,
	}.bytes())
	if got := f.Columns(); !slices.Equal(got, []Column{{Name: "n", Type: Int32, Nullable: true}, {Name: "s", Type: ByteArray, Nullable: true}}) {
		t.Fatalf("Columns() = %v", got)
	}
	v, nulls := readAll(t, reader(t, f, 0, "n").ReadInt32, 3, 2, true)
	if !slices.Equal(v, []int32{1, 0, 3}) || !slices.Equal(nulls, []bool{false, true, false}) {
		t.Errorf("n = %v, nulls %v", v, nulls)
	}
	_, err := reader(t, f, 0, "n").ReadInt32(make([]int32, 3), nil)
	wantErr(t, "int null without a nulls slice", err, ErrFormat, "row 1")
	_, err = reader(t, f, 0, "n").ReadInt64(make([]int64, 3), nil)
	wantErr(t, "int null without a nulls slice", err, ErrFormat, "row 1")
	fl, _ := readAll(t, reader(t, f, 0, "n").ReadFloat32, 3, 3, false)
	if fl[0] != 1 || !math.IsNaN(float64(fl[1])) || fl[2] != 3 {
		t.Errorf("n as float32 = %v", fl)
	}
	s, nulls := readAll(t, reader(t, f, 0, "s").ReadStrings, 3, 1, true)
	if !slices.Equal(s, []string{"a", "", "bc"}) || !slices.Equal(nulls, []bool{false, true, false}) {
		t.Errorf("s = %q, nulls %v", s, nulls)
	}
	b, nulls := readAll(t, reader(t, f, 0, "s").ReadBytes, 3, 3, true)
	if string(b[0]) != "a" || b[1] != nil || string(b[2]) != "bc" || !slices.Equal(nulls, []bool{false, true, false}) {
		t.Errorf("s as bytes = %q, nulls %v", b, nulls)
	}
	_, err = reader(t, f, 0, "s").ReadStrings(make([]string, 3), nil)
	wantErr(t, "string null without a nulls slice", err, ErrFormat, "row 1")
	_, err = reader(t, f, 0, "s").ReadBytes(make([][]byte, 3), nil)
	wantErr(t, "bytes null without a nulls slice", err, ErrFormat, "row 1")
}

func TestSchemas(t *testing.T) {
	f := openBytes(t, tfile{
		schema: []schemaElement{
			root(6),
			{name: "g", repetition: repOptional, numChildren: 1},
			optional("x", Int32),
			required("y", Int32),
			required("t", Int96),
			{typ: FixedLenByteArray, hasType: true, typeLength: 3, repetition: repRequired, name: "fix"},
			{typ: Boolean, hasType: true, repetition: repRequired, name: "b"},
			{typ: Int32, hasType: true, repetition: repRepeated, name: "many"},
		},
		chunks: []tchunk{
			{typ: Int32, path: "g.x", numValues: 2, levels: rleBits(1, 1), values: plainInt32(1, 2)},
			{typ: Int32, path: "y", numValues: 2, values: plainInt32(5, 6)},
			{typ: Int96, path: "t", numValues: 2, values: make([]byte, 24)},
			{typ: FixedLenByteArray, path: "fix", numValues: 2, values: []byte("abcdef")},
			{typ: Boolean, path: "b", numValues: 10, enc: encRLE, values: prefixed(rleBits(1, 0, 1, 1, 0, 0, 0, 1, 1, 0))},
			{typ: Int32, path: "many", numValues: 2, values: plainInt32(1, 2)},
		},
		numRows: 2,
	}.bytes())
	want := []Column{
		{Name: "g.x", Type: Int32, Nullable: true},
		{Name: "y", Type: Int32},
		{Name: "t", Type: Int96},
		{Name: "fix", Type: FixedLenByteArray, Length: 3},
		{Name: "b", Type: Boolean},
		{Name: "many", Type: Int32, Nullable: true},
	}
	if got := f.Columns(); !slices.Equal(got, want) {
		t.Fatalf("Columns() = %v, want %v", got, want)
	}
	_, err := f.Reader(0, 0)
	wantErr(t, "nested column", err, ErrUnsupported, "g.x")
	_, err = f.Reader(0, 2)
	wantErr(t, "INT96 column", err, ErrUnsupported, "INT96")
	_, err = f.Reader(0, 5)
	wantErr(t, "repeated column", err, ErrUnsupported, "many")
	v, nulls := readAll(t, reader(t, f, 0, "y").ReadInt32, 2, 5, true)
	if !slices.Equal(v, []int32{5, 6}) || !slices.Equal(nulls, []bool{false, false}) {
		t.Errorf("y = %v, nulls %v", v, nulls)
	}
	s, _ := readAll(t, reader(t, f, 0, "fix").ReadStrings, 2, 5, false)
	if !slices.Equal(s, []string{"abc", "def"}) {
		t.Errorf("fix = %q", s)
	}
	b, _ := readAll(t, reader(t, f, 0, "b").ReadInt32, 10, 3, false)
	if !slices.Equal(b, []int32{1, 0, 1, 1, 0, 0, 0, 1, 1, 0}) {
		t.Errorf("b = %v", b)
	}
	for _, name := range []string{"INT96", "TYPE(200)", "BOOLEAN"} {
		var typ Type
		switch name {
		case "INT96":
			typ = Int96
		case "TYPE(200)":
			typ = 200
		}
		if typ.String() != name {
			t.Errorf("%v.String() = %q", uint8(typ), typ.String())
		}
	}
}

func TestInt64Narrowing(t *testing.T) {
	f := openBytes(t, tfile{
		schema:  []schemaElement{root(1), required("n", Int64)},
		chunks:  []tchunk{{typ: Int64, path: "n", numValues: 3, values: plainInt64(-1, math.MaxInt32, 1<<40)}},
		numRows: 3,
	}.bytes())
	r := reader(t, f, 0, "n")
	v := make([]int32, 2)
	if n, err := r.ReadInt32(v, nil); n != 2 || err != nil || v[0] != -1 || v[1] != math.MaxInt32 {
		t.Fatalf("ReadInt32: %d, %v, %v", n, err, v)
	}
	_, err := r.ReadInt32(v, nil)
	wantErr(t, "ReadInt32 of 1<<40", err, ErrUnsupported, "1099511627776")
}

func TestUnknownFieldsAreSkipped(t *testing.T) {
	tf := tfile{
		schema:  []schemaElement{root(1), required("n", Int32)},
		chunks:  []tchunk{{typ: Int32, path: "n", numValues: 2, values: plainInt32(7, 8)}},
		numRows: 2,
		extra: func(w *tw) {
			// Fields a newer writer might add, of every wire type.
			w.field(20, ttTrue)
			w.field(21, ttFalse)
			w.field(22, ttByte)
			w.b = append(w.b, 1)
			w.field(23, ttI16)
			w.uvarint(zz(-7))
			w.field(24, ttDouble)
			w.b = append(w.b, 1, 2, 3, 4, 5, 6, 7, 8)
			w.bin(25, "ignored")
			w.field(26, ttUUID)
			w.b = append(w.b, make([]byte, 16)...)
			w.list(27, ttTrue, 3)
			w.b = append(w.b, 1, 2, 1)
			w.list(28, ttList, 1)
			w.listHeader(ttI64, 2)
			w.uvarint(zz(1))
			w.uvarint(zz(-1))
			w.field(29, ttMap)
			w.uvarint(2)
			w.b = append(w.b, ttBinary<<4|ttStruct)
			for range 2 {
				w.str("k")
				w.elem(func() { w.i32(1, 1); w.list(3, ttStruct, 1); w.elem(func() { w.bin(1, "deep") }) })
			}
			w.field(30, ttMap)
			w.uvarint(0)
			w.strct(31, func() { w.i32(1, 1) })
			w.i32(1000, 5)
		},
	}
	f := openBytes(t, tf.bytes())
	v, _ := readAll(t, reader(t, f, 0, "n").ReadInt32, 2, 2, false)
	if !slices.Equal(v, []int32{7, 8}) {
		t.Errorf("n = %v", v)
	}
	if f.CreatedBy() != "mimir test" || f.Metadata() != nil {
		t.Errorf("CreatedBy() = %q, Metadata() = %v", f.CreatedBy(), f.Metadata())
	}
}

func TestUnsupportedCodecsAndEncodings(t *testing.T) {
	file := func(c tchunk) []byte {
		c.typ, c.path, c.numValues = Int32, "n", 2
		return tfile{schema: []schemaElement{root(1), required("n", Int32)}, chunks: []tchunk{c}, numRows: 2}.bytes()
	}
	for name, c := range map[string]codec{"LZ4": codecLZ4, "LZ4_RAW": codecLZ4Raw, "BROTLI": codecBrotli, "LZO": codecLZO} {
		f := openBytes(t, file(tchunk{codec: c, values: plainInt32(1, 2)}))
		_, err := f.Reader(0, 0)
		wantErr(t, name, err, ErrUnsupported, name)
	}
	for name, e := range map[string]encoding{"BIT_PACKED": encBitPacked, "encoding 77": 77} {
		f := openBytes(t, file(tchunk{enc: e, values: plainInt32(1, 2)}))
		_, err := reader(t, f, 0, "n").ReadInt32(make([]int32, 2), nil)
		wantErr(t, name, err, ErrUnsupported, name)
	}
	f := openBytes(t, file(tchunk{enc: encRLEDictionary, values: []byte{1, 3, 0}}))
	_, err := reader(t, f, 0, "n").ReadInt32(make([]int32, 2), nil)
	wantErr(t, "indices without a dictionary", err, ErrFormat, "dictionary")
}

func TestMalformedPages(t *testing.T) {
	file := func(c tchunk) []byte {
		c.typ, c.path = Int32, "n"
		return tfile{schema: []schemaElement{root(1), optional("n", Int32)}, chunks: []tchunk{c}, numRows: 2}.bytes()
	}
	tests := map[string][]byte{
		"page past the chunk":     file(tchunk{numValues: 2, levels: rleBits(1, 1), values: plainInt32(1, 2), sizeFix: 100}),
		"values missing":          file(tchunk{numValues: 4, levels: rleBits(1, 1, 1, 1), values: plainInt32(1, 2)}),
		"levels missing":          file(tchunk{numValues: 2, values: plainInt32(1)}),
		"level run too long":      file(tchunk{numValues: 2, levels: []byte{9, 0, 0}, values: plainInt32(1, 2)}),
		"level out of range":      file(tchunk{numValues: 2, levels: []byte{4, 2}, values: plainInt32(1, 2)}),
		"chunk too short":         file(tchunk{numValues: 3, levels: rleBits(1, 1, 1), values: plainInt32(1, 2, 3)})[:60],
		"no page":                 file(tchunk{numValues: 2}),
		"snappy too short":        file(tchunk{numValues: 4, levels: rleBits(1, 1, 1, 1), values: plainInt32(1, 2), compress: true}),
		"snappy size":             file(tchunk{numValues: 2, levels: rleBits(1, 1), values: plainInt32(1, 2), compress: true, sizeFix: 1}),
		"v2 levels past the page": file(tchunk{numValues: 2, levels: rleBits(1, 1), values: plainInt32(1, 2), v2: true, sizeFix: -3}),
		"v2 values missing":       file(tchunk{numValues: 4, levels: rleBits(1, 1, 1, 1), values: plainInt32(1, 2), v2: true, compress: true}),
		"v2 size":                 file(tchunk{numValues: 2, levels: rleBits(1, 1), values: plainInt32(1, 2), v2: true, sizeFix: 2}),
	}
	for name, b := range tests {
		t.Run(name, func(t *testing.T) {
			f, err := Open(bytes.NewReader(b), int64(len(b)))
			if err != nil {
				if !errors.Is(err, ErrFormat) {
					t.Fatalf("Open: %v", err)
				}
				return
			}
			r, err := f.Reader(0, 0)
			if err == nil {
				_, err = r.ReadInt32(make([]int32, 10), make([]bool, 10))
			}
			if !errors.Is(err, ErrFormat) {
				t.Fatalf("got %v, want ErrFormat", err)
			}
		})
	}
	b := file(tchunk{numValues: 2, levels: rleBits(1, 1), values: plainInt32(1, 2)})
	for _, cut := range []int{len(b) - 9, len(b) - 20, 30} {
		if _, err := Open(bytes.NewReader(b[:cut]), int64(cut)); !errors.Is(err, ErrFormat) {
			t.Errorf("Open of %d of %d bytes: %v", cut, len(b), err)
		}
	}
	// A chunk whose metadata disagrees with the schema.
	bad := tfile{schema: []schemaElement{root(1), optional("n", Int32)}, chunks: []tchunk{{typ: Int64, path: "n", numValues: 1, values: plainInt64(1)}}, numRows: 1}.bytes()
	if _, err := Open(bytes.NewReader(bad), int64(len(bad))); !errors.Is(err, ErrFormat) {
		t.Errorf("type mismatch: %v", err)
	}
	bad = tfile{schema: []schemaElement{root(1), optional("n", Int32)}, chunks: []tchunk{{typ: Int32, path: "m", numValues: 1, values: plainInt32(1)}}, numRows: 1}.bytes()
	if _, err := Open(bytes.NewReader(bad), int64(len(bad))); !errors.Is(err, ErrFormat) {
		t.Errorf("path mismatch: %v", err)
	}
	bad = tfile{schema: []schemaElement{root(2), optional("n", Int32)}, chunks: []tchunk{{typ: Int32, path: "n", numValues: 1, values: plainInt32(1)}}, numRows: 1}.bytes()
	if _, err := Open(bytes.NewReader(bad), int64(len(bad))); !errors.Is(err, ErrFormat) {
		t.Errorf("schema element missing: %v", err)
	}
}

// TestDataPageV2 covers the ways a data page V2 may be laid out, beyond
// the v2 fixtures: the levels stay uncompressed with their lengths in the
// header, and the values are compressed on their own, or not at all when
// is_compressed is false.
func TestDataPageV2(t *testing.T) {
	cases := map[string]tchunk{
		"uncompressed":   {v2: true},
		"snappy":         {v2: true, compress: true},
		"not compressed": {v2: true, codec: codecSnappy},
	}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			n := c
			n.typ, n.path, n.numValues, n.nulls = Int32, "n", 5, 2
			n.levels, n.values = rleBits(1, 0, 1, 1, 0), plainInt32(1, 3, 4)
			s := c
			s.typ, s.path, s.numValues, s.nulls = ByteArray, "s", 5, 2
			s.levels, s.values = rleBits(0, 1, 1, 0, 1), plainStrings("a", "", "bcd")
			y := c
			y.typ, y.path, y.numValues = Int64, "y", 5
			y.values = plainInt64(-1, 0, 1, 2, 3)
			f := openBytes(t, tfile{
				schema:  []schemaElement{root(3), optional("n", Int32), optional("s", ByteArray), required("y", Int64)},
				chunks:  []tchunk{n, s, y},
				numRows: 5,
			}.bytes())
			for _, size := range []int{1, 2, 10} {
				v, nulls := readAll(t, reader(t, f, 0, "n").ReadInt32, 5, size, true)
				if !slices.Equal(v, []int32{1, 0, 3, 4, 0}) || !slices.Equal(nulls, []bool{false, true, false, false, true}) {
					t.Errorf("n = %v, nulls %v", v, nulls)
				}
				str, nulls := readAll(t, reader(t, f, 0, "s").ReadStrings, 5, size, true)
				if !slices.Equal(str, []string{"", "a", "", "", "bcd"}) || !slices.Equal(nulls, []bool{true, false, false, true, false}) {
					t.Errorf("s = %q, nulls %v", str, nulls)
				}
				w, _ := readAll(t, reader(t, f, 0, "y").ReadInt64, 5, size, false)
				if !slices.Equal(w, []int64{-1, 0, 1, 2, 3}) {
					t.Errorf("y = %v", w)
				}
			}
		})
	}
}

func TestConcurrentReaders(t *testing.T) {
	f := openFixture(t, "pages_v2_snappy")
	var wg sync.WaitGroup
	for g := range f.RowGroups() {
		for c := range f.Columns() {
			wg.Add(1)
			go func() {
				defer wg.Done()
				r, err := f.Reader(g, c)
				if err != nil {
					t.Error(err)
					return
				}
				buf := make([]float64, 333)
				nulls := make([]bool, 333)
				total := 0
				for {
					var n int
					if f.Columns()[c].Type == ByteArray {
						n, err = r.ReadStrings(make([]string, 333), nulls)
					} else {
						n, err = r.ReadFloat64(buf, nulls)
					}
					if err == io.EOF {
						break
					}
					if err != nil {
						t.Error(err)
						return
					}
					total += n
				}
				if int64(total) != f.RowGroupRows(g) {
					t.Errorf("group %d column %d: %d values", g, c, total)
				}
			}()
		}
	}
	wg.Wait()
}

func TestReset(t *testing.T) {
	f := openFixture(t, "pages_v2_snappy")
	r := reader(t, f, 2, "name")
	s, _ := readAll(t, r.ReadStrings, 2000, 300, false)
	expect(t, "name", s, 8000, nameOf)
	i, _ := f.Column("f32")
	if err := r.Reset(1, i); err != nil {
		t.Fatal(err)
	}
	v, _ := readAll(t, r.ReadFloat32, 4000, 300, true)
	expect(t, "f32", v, 4000, f32Of)
	if err := r.Reset(0, i); err != nil {
		t.Fatal(err)
	}
	v, _ = readAll(t, r.ReadFloat32, 4000, 4096, false)
	expect(t, "f32", v, 0, f32Of)
	if err := r.Reset(3, i); err == nil {
		t.Error("Reset to a missing row group succeeded")
	}
}

func TestNoAllocationPerValue(t *testing.T) {
	f := openFixture(t, "pages_v2_snappy")
	for _, name := range []string{"i32", "f32", "opt", "flag"} {
		r := reader(t, f, 0, name)
		col, _ := f.Column(name)
		for _, size := range []int{16, 4096} {
			buf := make([]float32, size)
			nulls := make([]bool, size)
			// The first chunk allocates the reader's buffers, which the
			// following ones reuse.
			allocs := testing.AllocsPerRun(3, func() {
				for g := range f.RowGroups() {
					if err := r.Reset(g, col); err != nil {
						t.Fatal(err)
					}
					for {
						if _, err := r.ReadFloat32(buf, nulls); err != nil {
							break
						}
					}
				}
			})
			if allocs > 0 {
				t.Errorf("%s in pieces of %d: %v allocations for 10000 values", name, size, allocs)
			}
		}
	}
}

func BenchmarkReadFloat32(b *testing.B) {
	benchRead(b, "f32", func(r *Reader, buf []float32) (int, error) { return r.ReadFloat32(buf, nil) })
}

func BenchmarkReadInt32(b *testing.B) {
	benchRead(b, "i32", func(r *Reader, buf []int32) (int, error) { return r.ReadInt32(buf, nil) })
}

func BenchmarkReadStrings(b *testing.B) {
	benchRead(b, "name", func(r *Reader, buf []string) (int, error) { return r.ReadStrings(buf, nil) })
}

// benchRead reads a column of each fixture repeatedly with one reader,
// counting the column's uncompressed bytes.
func benchRead[T any](b *testing.B, column string, read func(*Reader, []T) (int, error)) {
	for _, name := range []string{"pages_v2_snappy", "plain_v1"} {
		b.Run(name, func(b *testing.B) {
			f, err := OpenFile(filepath.Join("testdata", name+".parquet"))
			if err != nil {
				b.Fatal(err)
			}
			defer f.Close()
			benchFile(b, f, column, read)
		})
	}
}

// BenchmarkReadFile reads the f32, i32 and name columns of the file named
// by MIMIR_PARQUET_BENCH, for timing on files of realistic size, such as
// those tools/parquet/gen.py would write with a larger n.
func BenchmarkReadFile(b *testing.B) {
	path := os.Getenv("MIMIR_PARQUET_BENCH")
	if path == "" {
		b.Skip("MIMIR_PARQUET_BENCH names no file")
	}
	f, err := OpenFile(path)
	if err != nil {
		b.Fatal(err)
	}
	defer f.Close()
	b.Run("f32", func(b *testing.B) {
		benchFile(b, f, "f32", func(r *Reader, buf []float32) (int, error) { return r.ReadFloat32(buf, nil) })
	})
	b.Run("i32", func(b *testing.B) {
		benchFile(b, f, "i32", func(r *Reader, buf []int32) (int, error) { return r.ReadInt32(buf, nil) })
	})
	b.Run("name", func(b *testing.B) {
		benchFile(b, f, "name", func(r *Reader, buf []string) (int, error) { return r.ReadStrings(buf, nil) })
	})
}

func benchFile[T any](b *testing.B, f *File, column string, read func(*Reader, []T) (int, error)) {
	col, ok := f.Column(column)
	if !ok {
		b.Skipf("no column %q", column)
	}
	var size int64
	for g := range f.RowGroups() {
		size += f.meta.rowGroups[g].columns[col].meta.totalUncompressedSize
	}
	b.SetBytes(size)
	b.ReportAllocs()
	buf := make([]T, 4096)
	r, err := f.Reader(0, col)
	if err != nil {
		b.Fatal(err)
	}
	for b.Loop() {
		for g := range f.RowGroups() {
			if err := r.Reset(g, col); err != nil {
				b.Fatal(err)
			}
			for {
				if _, err := read(r, buf); err != nil {
					if err != io.EOF {
						b.Fatal(err)
					}
					break
				}
			}
		}
	}
}

func FuzzOpen(f *testing.F) {
	for name := range fixtures {
		f.Add(fixtureBytes(f, name))
	}
	f.Add(tfile{
		schema: []schemaElement{root(2), optional("n", Int32), optional("s", ByteArray)},
		chunks: []tchunk{
			{typ: Int32, path: "n", numValues: 3, levels: rleBits(1, 0, 1), values: plainInt32(1, 3)},
			{typ: ByteArray, path: "s", numValues: 3, levels: rleBits(1, 0, 1), values: plainStrings("a", "bc")},
		},
		numRows: 3,
	}.bytes())
	f.Fuzz(func(t *testing.T, b []byte) {
		pf, err := Open(bytes.NewReader(b), int64(len(b)))
		if err != nil {
			if pf != nil {
				t.Fatal("file with an error")
			}
			return
		}
		// Read what a mutated footer still points at, bounded so that a
		// hostile value count cannot ask for a huge page or a long read.
		columns := pf.Columns()
		for g := range min(pf.RowGroups(), 4) {
			for c := range columns {
				ch := &pf.meta.rowGroups[g].columns[c]
				if ch.meta.numValues > 1<<16 || ch.meta.totalUncompressedSize > 1<<20 {
					continue
				}
				r, err := pf.Reader(g, c)
				if err != nil {
					continue
				}
				nulls := make([]bool, 100)
				strs := make([]string, 100)
				vals := make([]float64, 100)
				for budget := 1 << 22; budget > 0 && err == nil; {
					var n int
					switch columns[c].Type {
					case ByteArray, FixedLenByteArray:
						n, err = r.ReadStrings(strs, nulls)
						for _, s := range strs[:n] {
							budget -= len(s) + 1
						}
					default:
						n, err = r.ReadFloat64(vals, nulls)
						budget -= 8 * n
					}
				}
			}
		}
	})
}
