package onnx

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/gsoultan/mimir/tensor"
)

// FuzzRun builds a node of every operator from arbitrary bytes, with
// inputs of arbitrary small shapes and integer constants and arbitrary
// attributes, and runs it: no node may panic, whatever its inputs.
func FuzzRun(f *testing.F) {
	names := make([]string, 0, len(ops))
	for op := range ops {
		names = append(names, op)
	}
	slices.Sort(names)
	attrNames := []string{"axis", "axes", "perm", "keepdims", "kernel_shape", "pads", "strides", "dilations", "group",
		"transB", "upper", "exclusive", "reverse", "to", "mode", "split", "num_outputs", "allowzero", "output_padding",
		"output_shape", "ceil_mode", "count_include_pad", "auto_pad", "epsilon", "alpha", "value", "start", "end", "fmod"}
	f.Add([]byte{0, 2, 1, 2, 3, 4, 0, 0})
	f.Add([]byte("conv\x02\x04\x01\x03\x05\x05\x02\x04\x02\x02\x03"))
	f.Add([]byte("\x30\x03\x02\x02\x03\x01\x02\x00\x01\x05"))
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
		op := names[next()%len(names)]
		g := newGraph()
		inputs := map[string]*tensor.Tensor{}
		var io []string
		for i := range next() % 5 {
			name := fmt.Sprintf("in%d", i)
			if next()%4 == 0 {
				io = append(io, "")
				continue
			}
			shape := make([]int, next()%5)
			for d := range shape {
				shape[d] = next() % 6
			}
			if next()%2 == 0 {
				n := 1
				for _, d := range shape {
					n *= d
				}
				v := make([]int64, n)
				for j := range v {
					v[j] = int64(int8(next()))
				}
				g.initInts(name, shape, v)
			} else {
				dims := make([]any, len(shape))
				for d, s := range shape {
					dims[d] = s
				}
				g.input(name, Float, dims...)
				inputs[name] = tensor.Randn(uint64(i), shape...)
			}
			io = append(io, name)
		}
		io = append(io, "->")
		outs := 1 + next()%3
		for i := range outs {
			io = append(io, fmt.Sprintf("out%d", i))
		}
		var attrs []*pb
		for range next() % 4 {
			name := attrNames[next()%len(attrNames)]
			switch next() % 4 {
			case 0:
				attrs = append(attrs, aInt(name, int64(int8(next()))))
			case 1:
				v := make([]int64, next()%5)
				for j := range v {
					v[j] = int64(int8(next()))
				}
				attrs = append(attrs, aInts(name, v...))
			case 2:
				attrs = append(attrs, aFloat(name, float32(int8(next()))/16))
			default:
				modes := []string{"constant", "reflect", "edge", "linear", "nearest", "SAME_UPPER", "VALID", "tanh"}
				attrs = append(attrs, aString(name, modes[next()%len(modes)]))
			}
		}
		g.node(op, io, attrs...)
		for i := range outs {
			g.output(fmt.Sprintf("out%d", i))
		}
		m, err := Parse(g.model())
		if err != nil {
			return
		}
		defer m.Close()
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_, _ = m.Run(ctx, inputs)
	})
}

// FuzzOpen parses arbitrary bytes as a model, and runs those that parse
// on zeros when their inputs are small: neither may panic.
func FuzzOpen(f *testing.F) {
	f.Add(newGraph().input("x", Float, 2, 3).init("w", []int{3, 2}, []float32{1, 2, 3, 4, 5, 6}).
		node("MatMul", []string{"x", "w", "->", "h"}).node("Relu", []string{"h", "->", "y"}).output("y").model())
	f.Add(newGraph().input("x", Float, 1, 2, 4, 4).init("w", []int{2, 1, 3, 3}, make([]float32, 18)).
		node("Conv", []string{"x", "w", "->", "c"}, aInts("pads", 1, 1, 1, 1), aInt("group", 2)).
		node("MaxPool", []string{"c", "->", "p"}, aInts("kernel_shape", 2, 2), aInts("strides", 2, 2)).
		node("Shape", []string{"p", "->", "s"}).node("ConstantOfShape", []string{"s", "->", "z"}).
		node("Add", []string{"p", "z", "->", "y"}).output("y").model())
	f.Add(newGraph().input("x", Float, 2, 4).initInts("i", []int{2}, []int64{1, -1}).
		node("Gather", []string{"x", "i", "->", "g"}, aInt("axis", 1)).
		node("Softmax", []string{"g", "->", "y"}).output("y").model())
	if files, _ := filepath.Glob("testdata/*.onnx"); len(files) > 0 {
		for _, path := range files {
			if b, err := os.ReadFile(path); err == nil && len(b) < 1<<16 {
				f.Add(b)
			}
		}
	}
	// A model may ask for any size of tensor; refuse what passes 256 MiB.
	tensor.SetMemoryBudget(tensor.CPU(), 256<<20)
	f.Cleanup(func() { tensor.SetMemoryBudget(tensor.CPU(), 0) })
	f.Fuzz(func(t *testing.T, b []byte) {
		m, err := Parse(b)
		if err != nil {
			if m != nil {
				t.Fatalf("model %v with error %v", m, err)
			}
			return
		}
		defer m.Close()
		inputs := map[string]*tensor.Tensor{}
		for _, in := range m.Inputs() {
			shape := make([]int, len(in.Shape))
			n := 1
			for i, d := range in.Shape {
				shape[i] = max(d.Size, 1)
				n *= shape[i]
			}
			if n > 1<<12 {
				return
			}
			inputs[in.Name] = tensor.Zeros(in.DType(), shape...)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_, _ = m.Run(ctx, inputs)
	})
}
