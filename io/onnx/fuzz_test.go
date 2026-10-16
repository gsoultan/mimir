package onnx

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/gsoultan/mimir/tensor"
)

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
