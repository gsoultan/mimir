package onnx

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gsoultan/mimir/io/safetensors"
	"github.com/gsoultan/mimir/tensor"
)

// goldenDir returns the directory of the golden models that
// tools/onnx/gen.py writes: $MIMIR_ONNX_DIR if set, so CI can test the
// models torch.onnx.export writes, else testdata.
func goldenDir() string {
	if dir := os.Getenv("MIMIR_ONNX_DIR"); dir != "" {
		return dir
	}
	return "testdata"
}

// golden returns the model name and its inputs and expected outputs.
func golden(t testing.TB, name string) (m *Model, inputs map[string]*tensor.Tensor, want map[string]safetensors.Tensor) {
	t.Helper()
	dir := goldenDir()
	m, err := Open(filepath.Join(dir, name+".onnx"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = m.Close() })
	f, err := safetensors.Open(filepath.Join(dir, name+".safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = f.Close() })
	inputs, want = map[string]*tensor.Tensor{}, map[string]safetensors.Tensor{}
	for _, k := range f.Names() {
		st, _ := f.Tensor(k)
		switch {
		case strings.HasPrefix(k, "in/"):
			var x *tensor.Tensor
			if st.DType == "I32" {
				v, err := st.Int32s()
				if err != nil {
					t.Fatal(err)
				}
				x = tensor.FromSlice(v, st.Shape...)
			} else {
				v, err := st.Float32s()
				if err != nil {
					t.Fatal(err)
				}
				x = tensor.FromSlice(v, st.Shape...)
			}
			inputs[strings.TrimPrefix(k, "in/")] = x
		case strings.HasPrefix(k, "out/"):
			want[strings.TrimPrefix(k, "out/")] = st
		}
	}
	return m, inputs, want
}

// TestGolden runs the models of tools/onnx/gen.py and compares their
// outputs with NumPy's, or PyTorch's.
func TestGolden(t *testing.T) {
	for _, name := range []string{"mlp", "cnn", "decoder", "transformer"} {
		t.Run(name, func(t *testing.T) {
			m, inputs, want := golden(t, name)
			if len(m.Outputs()) != len(want) {
				t.Fatalf("%d outputs, want %d", len(m.Outputs()), len(want))
			}
			// The second run replays the first's schedule.
			for range 2 {
				res, err := m.Run(context.Background(), inputs)
				if err != nil {
					t.Fatal(err)
				}
				for k, w := range want {
					wf, err := w.Float32s()
					if err != nil {
						t.Fatal(err)
					}
					got, err := res[k].Data[float32](context.Background())
					if err != nil {
						t.Fatalf("output %s: %v", k, err)
					}
					if len(got) != len(wf) {
						t.Fatalf("output %s of %v, want %v", k, res[k].Shape(), w.Shape)
					}
					for i := range wf {
						if d := math.Abs(float64(got[i] - wf[i])); !(d <= 1e-4*max(1, math.Abs(float64(wf[i])))) {
							t.Fatalf("output %s element %d is %v, want %v", k, i, got[i], wf[i])
						}
					}
				}
			}
		})
	}
}
