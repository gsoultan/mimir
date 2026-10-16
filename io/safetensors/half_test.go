package safetensors

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

func TestHalfTensorsRoundTrip(t *testing.T) {
	for _, dtype := range []string{"F16", "BF16"} {
		data := make([]byte, 2*65536)
		for i := range 65536 {
			binary.LittleEndian.PutUint16(data[2*i:], uint16(i))
		}
		v := Tensor{DType: dtype, Shape: []int{65536}, Data: data}.halfFloat32s()
		enc := Float16Tensor([]int{65536}, v)
		if dtype == "BF16" {
			enc = BFloat16Tensor([]int{65536}, v)
		}
		for i := range 65536 {
			got := binary.LittleEndian.Uint16(enc.Data[2*i:])
			// Every element but a signalling NaN comes back as it was.
			if got != uint16(i) && !(math.IsNaN(float64(v[i])) && math.IsNaN(float64(Tensor{DType: dtype, Data: enc.Data[2*i : 2*i+2]}.halfFloat32s()[0]))) {
				t.Fatalf("%s %#04x read as %v and written as %#04x", dtype, i, v[i], got)
			}
		}
		if enc.DType != dtype || len(enc.Shape) != 1 || enc.Shape[0] != 65536 {
			t.Errorf("%s tensor encoded as %s%v", dtype, enc.DType, enc.Shape)
		}
	}
}

func TestHalfTensorsRound(t *testing.T) {
	v := []float32{1, 1 + 1.0/2048, 1 + 3.0/2048, 65520, 1e-8, float32(math.Copysign(0, -1)), float32(math.Inf(-1)), 1.0 / 3}
	cases := []struct {
		dtype string
		enc   Tensor
		want  []uint16
	}{
		{"F16", Float16Tensor([]int{2, 4}, v), []uint16{0x3c00, 0x3c00, 0x3c02, 0x7c00, 0x0000, 0x8000, 0xfc00, 0x3555}},
		{"BF16", BFloat16Tensor([]int{2, 4}, v), []uint16{0x3f80, 0x3f80, 0x3f80, 0x4780, 0x322c, 0x8000, 0xff80, 0x3eab}},
	}
	for _, c := range cases {
		want := make([]byte, 2*len(c.want))
		for i, h := range c.want {
			binary.LittleEndian.PutUint16(want[2*i:], h)
		}
		if !bytes.Equal(c.enc.Data, want) {
			t.Errorf("%s: encoded %x, want %x", c.dtype, c.enc.Data, want)
		}
		f, err := Parse(write(t, map[string]Tensor{"x": c.enc}))
		if err != nil {
			t.Fatal(err)
		}
		got, err := f.tensors["x"].Float32s()
		if err != nil {
			t.Fatal(err)
		}
		for i, g := range got {
			w := Tensor{DType: c.dtype, Data: want[2*i : 2*i+2]}.halfFloat32s()[0]
			if g != w {
				t.Errorf("%s element %d read as %v, want %v", c.dtype, i, g, w)
			}
		}
	}
}

func write(t *testing.T, ts map[string]Tensor) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := Write(&buf, ts, nil); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}
