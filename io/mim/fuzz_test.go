package mim_test

import (
	"context"
	"slices"
	"testing"

	mimfile "github.com/gsoultan/mimir-model"

	"github.com/gsoultan/mimir/io/mim"
	"github.com/gsoultan/mimir/tensor"
)

// FuzzOp builds a graph of one primitive or call from arbitrary bytes,
// with inputs and an output of arbitrary small shapes and dtypes and
// arbitrary attributes, and runs it: nothing may panic, whatever the file
// declares.
func FuzzOp(f *testing.F) {
	codes := []mimfile.Opcode{}
	for c := mimfile.OpNeg; c <= mimfile.OpCall; c++ {
		codes = append(codes, c)
	}
	names := []string{"matmul", "linear", "conv2d", "softmax", "layer_norm", "rms_norm", "cumsum", "rope",
		"attention", "embedding", "top_k", "argsort", "mean", "variance", "dequant_matmul", "geglu"}
	dtypes := []mimfile.DType{mimfile.F32, mimfile.F16, mimfile.I32, mimfile.I64, mimfile.Bool, mimfile.U8}
	keys := []string{"axis", "axes", "perm", "pads", "begin", "end", "shape", "mode", "to", "value", "k",
		"strides", "dilations", "groups", "causal", "exclusive", "reverse", "rotary_dims", "interleaved", "num_kv_heads"}
	f.Add([]byte{0, 1, 2, 3, 4, 0, 2, 3, 4})
	f.Add([]byte{33, 3, 0, 2, 2, 3, 0, 2, 3, 3, 0, 1, 2, 2, 1, 7})
	f.Add([]byte{29, 1, 0, 2, 4, 4, 0, 2, 2, 2, 2, 4, 1, 1, 2, 5, 2, 3, 3})
	tensor.SetMemoryBudget(tensor.CPU(), 256<<20)
	f.Cleanup(func() { tensor.SetMemoryBudget(tensor.CPU(), 0) })
	f.Fuzz(func(t *testing.T, b []byte) {
		next := func() int {
			if len(b) == 0 {
				return 0
			}
			v := int(b[0])
			b = b[1:]
			return v
		}
		shape := func() []int {
			s := make([]int, next()%5)
			for i := range s {
				s[i] = next() % 5
			}
			return s
		}
		code := codes[next()%len(codes)]
		bl := newBuilder()
		in := make([]uint32, next()%4)
		feeds := map[string]*tensor.Tensor{}
		for i := range in {
			name := string(rune('a' + i))
			dt := dtypes[next()%len(dtypes)]
			s := shape()
			in[i] = bl.input(name, dt, s...)
			st := mim.ValueInfo{Type: dt}.DType()
			feeds[name] = tensor.Zeros(st, s...)
			if len(s) == 0 {
				feeds[name] = feeds[name].Reshape()
			}
		}
		var attrs []mimfile.Attr
		if code == mimfile.OpCall {
			attrs = calls(names[next()%len(names)], 1)
		}
		for range next() % 4 {
			key := keys[next()%len(keys)]
			if slices.ContainsFunc(attrs, func(a mimfile.Attr) bool { return a.Key == key }) {
				continue
			}
			switch next() % 4 {
			case 0:
				attrs = append(attrs, aInt(key, int64(next()%7)-2))
			case 1:
				v := make([]int64, next()%7)
				for i := range v {
					v[i] = int64(next()%7) - 1
				}
				attrs = append(attrs, aInts(key, v...))
			case 2:
				attrs = append(attrs, aStr(key, []string{"add", "max", "mul", "none"}[next()%4]))
			default:
				attrs = append(attrs, aDType(key, dtypes[next()%len(dtypes)]))
			}
		}
		bl.output("y", bl.op(0, code, dtypes[next()%len(dtypes)], shape(), in, attrs...))
		if bl.g.Validate() != nil {
			return
		}
		enc, err := bl.g.Encode()
		if err != nil {
			return
		}
		data := writeGraph(t, enc)
		m, err := mim.Parse(data)
		if err != nil {
			return
		}
		defer m.Close()
		out, err := m.Run(context.Background(), feeds)
		if err == nil {
			_ = out["y"].Realize(context.Background())
		}
	})
}

func writeGraph(t *testing.T, enc []byte) []byte {
	t.Helper()
	w := mimfile.NewWriter()
	if err := w.AddSection(mimfile.SectionGraph, true, enc); err != nil {
		t.Fatal(err)
	}
	var buf bytesBuffer
	if _, err := w.WriteTo(&buf); err != nil {
		t.Fatal(err)
	}
	return buf
}

// bytesBuffer collects what is written to it.
type bytesBuffer []byte

func (b *bytesBuffer) Write(p []byte) (int, error) {
	*b = append(*b, p...)
	return len(p), nil
}
