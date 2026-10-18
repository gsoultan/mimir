package mim_test

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/gsoultan/mimir/io/mim"
	"github.com/gsoultan/mimir/tensor"
)

// The files in testdata come from the repository of the format,
// github.com/gsoultan/mimir-model: models exported from PyTorch and converted from
// ONNX, with the outputs PyTorch and onnxruntime computed, and a tiny
// Q4_K_M Llama converted from GGUF, with the logits of a float64 forward
// pass.

type array struct {
	DType string `json:"dtype"`
	Shape []int  `json:"shape"`
	Data  string `json:"data"`
}

// tensor returns the array as a mimir tensor: i64 arrays as Int32.
func (a array) tensor(t *testing.T) *tensor.Tensor {
	t.Helper()
	b, err := base64.StdEncoding.DecodeString(a.Data)
	if err != nil {
		t.Fatal(err)
	}
	switch a.DType {
	case "", "f32":
		return tensor.FromSlice(floats(b), a.Shape...)
	case "i64":
		v := make([]int32, len(b)/8)
		for i := range v {
			v[i] = int32(binary.LittleEndian.Uint64(b[8*i:]))
		}
		return tensor.FromSlice(v, a.Shape...)
	}
	t.Fatalf("array of dtype %s", a.DType)
	return nil
}

func floats(b []byte) []float32 {
	v := make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return v
}

type testRun struct {
	Inputs  map[string]array `json:"inputs"`
	Outputs map[string]array `json:"outputs"`
}

// near fails unless got matches want within tol times want's largest
// magnitude.
func near(t *testing.T, what string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", what, len(got), len(want))
	}
	scale, diff := 1e-30, 0.0
	for i := range want {
		scale = max(scale, math.Abs(float64(want[i])))
		diff = max(diff, math.Abs(float64(got[i]-want[i])))
	}
	if diff > tol*scale || math.IsNaN(diff) {
		t.Errorf("%s: differs by %g, allowed %g (largest value %g)", what, diff, tol*scale, scale)
	}
}

// check runs the runs in one session of m and compares the outputs.
func check(t *testing.T, m *mim.Model, runs []testRun, tol float64) {
	t.Helper()
	ctx := context.Background()
	for ri, r := range runs {
		in := map[string]*tensor.Tensor{}
		for k, a := range r.Inputs {
			in[k] = a.tensor(t)
		}
		out, err := m.Run(ctx, in)
		if err != nil {
			t.Fatalf("run %d: %v", ri, err)
		}
		if len(out) != len(r.Outputs) {
			t.Fatalf("run %d: %d outputs, want %d", ri, len(out), len(r.Outputs))
		}
		for k, a := range r.Outputs {
			got, err := out[k].Data[float32](ctx)
			if err != nil {
				t.Fatalf("run %d: output %s: %v", ri, k, err)
			}
			b, _ := base64.StdEncoding.DecodeString(a.Data)
			near(t, k, got, floats(b), tol)
		}
	}
}

func readJSON(t *testing.T, path string, v any) {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, v); err != nil {
		t.Fatal(err)
	}
}

func TestPyTorchExports(t *testing.T) {
	var e struct {
		Tolerance float64 `json:"tolerance"`
		Models    []struct {
			Name string    `json:"name"`
			MIM  string    `json:"mim"`
			Runs []testRun `json:"runs"`
		} `json:"models"`
	}
	readJSON(t, filepath.Join("testdata", "pytorch", "expected.json"), &e)
	if len(e.Models) == 0 {
		t.Fatal("no models")
	}
	for _, model := range e.Models {
		t.Run(model.Name, func(t *testing.T) {
			m, err := mim.Open(filepath.Join("testdata", "pytorch", model.MIM))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			check(t, m, model.Runs, e.Tolerance)
		})
	}
}

func TestONNXConversion(t *testing.T) {
	var e struct {
		MIM       string    `json:"mim"`
		Tolerance float64   `json:"tolerance"`
		Runs      []testRun `json:"runs"`
	}
	readJSON(t, filepath.Join("testdata", "onnx", "expected.json"), &e)
	m, err := mim.Open(filepath.Join("testdata", "onnx", e.MIM))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	check(t, m, e.Runs, e.Tolerance)
}

// llama opens the tiny Llama in testdata, which mim-import -graph
// converted from GGUF with a context of 64 and the options name names:
// "exact" with a float32 cache, "f16-cache" with a float16 one, and "a8"
// with a float16 cache and activations rounded to 8 bits.
func llama(t *testing.T, name string) *mim.Model {
	t.Helper()
	m, err := mim.Open(filepath.Join("testdata", "llama", "tiny-q4km-"+name+".mim"))
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func TestQuantizedLlama(t *testing.T) {
	var e struct {
		Vocab        int      `json:"vocab"`
		PromptIDs    []int32  `json:"prompt_ids"`
		PromptLogits string   `json:"prompt_logits"`
		Greedy       []int32  `json:"greedy"`
		GreedyLogits []string `json:"greedy_logits"`
		LlamaCpp     float64  `json:"llama_cpp_max_diff"`
	}
	readJSON(t, filepath.Join("testdata", "llama", "expected.json"), &e)
	decode := func(s string) []float32 {
		b, err := base64.StdEncoding.DecodeString(s)
		if err != nil {
			t.Fatal(err)
		}
		return floats(b)
	}
	for _, tc := range []struct {
		name string
		// tol bounds the logits' difference from the oracle's, relative
		// to the largest; abs, if set, bounds it instead.
		tol, abs float64
	}{
		// The weights decoded, products in float32 and a float32 cache:
		// as close to the float64 oracle as float32 allows, and so
		// choosing the same tokens.
		{name: "exact", tol: 2e-5},
		{name: "f16-cache", tol: 1e-3},
		// Activations rounded to 8 bits, as llama.cpp does: about as far
		// from the oracle as llama.cpp, which can choose another token
		// where two are close, so the steps take the oracle's tokens.
		{name: "a8", abs: 1.5 * e.LlamaCpp},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := llama(t, tc.name)
			defer m.Close()
			ctx := context.Background()
			step := func(ids []int32, pos int) []float32 {
				t.Helper()
				p := make([]int32, len(ids))
				for i := range p {
					p[i] = int32(pos + i)
				}
				out, err := m.Run(ctx, map[string]*tensor.Tensor{
					"tokens":    tensor.FromSlice(ids, 1, len(ids)),
					"positions": tensor.FromSlice(p, 1, len(ids)),
				})
				if err != nil {
					t.Fatal(err)
				}
				l, err := out["logits"].Data[float32](ctx)
				if err != nil {
					t.Fatal(err)
				}
				return l
			}
			compare := func(what string, got, want []float32) {
				t.Helper()
				if tc.abs > 0 {
					near(t, what, got, want, tc.abs/maxAbs(want))
				} else {
					near(t, what, got, want, tc.tol)
				}
			}
			all := step(e.PromptIDs, 0)
			compare("prompt", all, decode(e.PromptLogits))
			logits := all[len(all)-e.Vocab:]
			pos := len(e.PromptIDs)
			for i, want := range e.Greedy {
				compare("step", logits, decode(e.GreedyLogits[i]))
				if got := argmax(logits); got != want && tc.abs == 0 {
					t.Fatalf("step %d: token %d, want %d", i, got, want)
				}
				logits = step([]int32{want}, pos)
				pos++
			}
		})
	}
}

func maxAbs(v []float32) float64 {
	m := 0.0
	for _, x := range v {
		m = max(m, math.Abs(float64(x)))
	}
	return m
}

func argmax(v []float32) int32 {
	best := 0
	for i, x := range v {
		if x > v[best] {
			best = i
		}
	}
	return int32(best)
}
