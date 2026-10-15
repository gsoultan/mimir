package safetensors

import (
	"encoding/binary"
	"errors"
	"math"
	"slices"
	"testing"
)

func file(header string, data ...byte) []byte {
	b := binary.LittleEndian.AppendUint64(nil, uint64(len(header)))
	return append(append(b, header...), data...)
}

func f32le(v ...float32) []byte {
	var b []byte
	for _, x := range v {
		b = binary.LittleEndian.AppendUint32(b, math.Float32bits(x))
	}
	return b
}

func TestParse(t *testing.T) {
	data := append(f32le(1, -2.5), binary.LittleEndian.AppendUint32(nil, uint32(0xfffffffd))...)
	b := file(`{"__metadata__":{"oracle":"numpy"},"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"i":{"dtype":"I32","shape":[1,1],"data_offsets":[8,12]}}`, data...)
	f, err := Parse(b)
	if err != nil {
		t.Fatal(err)
	}
	if got := f.Names(); !slices.Equal(got, []string{"a", "i"}) {
		t.Fatalf("Names() = %v", got)
	}
	if f.Metadata["oracle"] != "numpy" {
		t.Errorf("metadata = %v", f.Metadata)
	}
	a, _ := f.Tensor("a")
	if v, err := a.Float32s(); err != nil || !slices.Equal(v, []float32{1, -2.5}) {
		t.Errorf("a = %v, %v", v, err)
	}
	i, _ := f.Tensor("i")
	if v, err := i.Int32s(); err != nil || !slices.Equal(v, []int32{-3}) || !slices.Equal(i.Shape, []int{1, 1}) {
		t.Errorf("i = %v %v, %v", v, i.Shape, err)
	}
	if _, err := i.Float32s(); err == nil {
		t.Error("reading I32 as F32 succeeded")
	}
}

func TestParseRejectsMalformedFiles(t *testing.T) {
	tests := map[string][]byte{
		"short":         {1, 2, 3},
		"header length": binary.LittleEndian.AppendUint64(nil, 1<<40),
		"not json":      file(`{"a":`),
		"dtype":         file(`{"a":{"dtype":"Q7","shape":[1],"data_offsets":[0,4]}}`, 0, 0, 0, 0),
		"size":          file(`{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}`, 0, 0, 0, 0),
		"past end":      file(`{"a":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}`, 0, 0, 0, 0),
		"negative dim":  file(`{"a":{"dtype":"F32","shape":[-1],"data_offsets":[0,0]}}`),
		"overflow":      file(`{"a":{"dtype":"F64","shape":[4611686018427387904,4],"data_offsets":[0,0]}}`),
		"gap":           file(`{"a":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}`, 0, 0),
		"overlap": file(`{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"b":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}`,
			0, 0),
		"trailing": file(`{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}`, 0, 0),
	}
	for name, b := range tests {
		if _, err := Parse(b); !errors.Is(err, ErrFormat) {
			t.Errorf("%s: got %v, want ErrFormat", name, err)
		}
	}
}

func FuzzParse(f *testing.F) {
	f.Add(file(`{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}`, f32le(1, 2)...))
	f.Add(file(`{"__metadata__":{"k":"v"}}`))
	f.Add([]byte{})
	f.Fuzz(func(t *testing.T, b []byte) {
		file, err := Parse(b)
		if err != nil {
			return
		}
		for _, name := range file.Names() {
			tt, _ := file.Tensor(name)
			_, _ = tt.Float32s()
			_, _ = tt.Int32s()
		}
	})
}
