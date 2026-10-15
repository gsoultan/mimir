// Package goldentest loads golden files written by tools/golden/gen.py and
// compares results against them.
package goldentest

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/gsoultan/mimir/io/safetensors"
	"github.com/gsoultan/mimir/tensor"
)

// Case is one golden case.
type Case struct {
	Inputs []safetensors.Tensor // in argument order
	Want   safetensors.Tensor   // the expected output
	// Gout is the gradient of a loss with respect to the output, and
	// Grads[i] the expected gradient with respect to input i. Both are
	// empty for cases without gradients.
	Gout  safetensors.Tensor
	Grads map[int]safetensors.Tensor
	// Extra holds any other expected results by name.
	Extra map[string]safetensors.Tensor
}

// Dir returns the golden directory: $MIMIR_GOLDEN_DIR if set, so CI can test
// against freshly generated PyTorch goldens, else testdata/golden at the
// repository root.
func Dir() string {
	if dir := os.Getenv("MIMIR_GOLDEN_DIR"); dir != "" {
		return dir
	}
	_, file, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(file), "..", "..", "testdata", "golden")
}

// Load reads name.safetensors from Dir and groups its tensors, named
// <case>/<part>, into cases. It also returns the file metadata, which names
// the oracle.
func Load(tb testing.TB, name string) (map[string]Case, map[string]string) {
	tb.Helper()
	b, err := os.ReadFile(filepath.Join(Dir(), name+".safetensors"))
	if err != nil {
		tb.Fatal(err)
	}
	f, err := safetensors.Parse(b)
	if err != nil {
		tb.Fatal(err)
	}
	inputs := map[string]map[int]safetensors.Tensor{}
	cases := map[string]Case{}
	hasOut := map[string]bool{}
	for _, full := range f.Names() {
		c, part, ok := strings.Cut(full, "/")
		if !ok {
			tb.Fatalf("golden tensor %q is not named <case>/<part>", full)
		}
		t, _ := f.Tensor(full)
		cs := cases[c]
		switch {
		case part == "out":
			cs.Want, hasOut[c] = t, true
		case part == "gout":
			cs.Gout = t
		case index(part, "in") >= 0:
			if inputs[c] == nil {
				inputs[c] = map[int]safetensors.Tensor{}
			}
			inputs[c][index(part, "in")] = t
		case index(part, "grad") >= 0:
			if cs.Grads == nil {
				cs.Grads = map[int]safetensors.Tensor{}
			}
			cs.Grads[index(part, "grad")] = t
		default:
			if cs.Extra == nil {
				cs.Extra = map[string]safetensors.Tensor{}
			}
			cs.Extra[part] = t
		}
		cases[c] = cs
	}
	for c, cs := range cases {
		if !hasOut[c] {
			tb.Fatalf("golden case %q has no out", c)
		}
		for i := range len(inputs[c]) {
			t, ok := inputs[c][i]
			if !ok {
				tb.Fatalf("golden case %q is missing in%d", c, i)
			}
			cs.Inputs = append(cs.Inputs, t)
		}
		cases[c] = cs
	}
	return cases, f.Metadata
}

// index returns N for a part named prefix<N>, or -1.
func index(part, prefix string) int {
	n, err := strconv.Atoi(strings.TrimPrefix(part, prefix))
	if err != nil || !strings.HasPrefix(part, prefix) || n < 0 {
		return -1
	}
	return n
}

// Tol bounds the allowed error: |got - want| <= Atol + Rtol*|want|.
type Tol struct {
	Atol, Rtol float64
}

// Compare fails tb unless got, of shape gotShape, matches want within tol.
// NaN must match NaN and infinities must match exactly. Int32 results must
// match exactly whatever tol says.
func Compare[T float32 | int32](tb testing.TB, want safetensors.Tensor, got []T, gotShape []int, tol Tol) {
	tb.Helper()
	if !slices.Equal(want.Shape, gotShape) {
		tb.Fatalf("shape %v, want %v", gotShape, want.Shape)
	}
	var w []float64
	switch any(got).(type) {
	case []float32:
		v, err := want.Float32s()
		if err != nil {
			tb.Fatal(err)
		}
		w = widen(v)
	default:
		v, err := want.Int32s()
		if err != nil {
			tb.Fatal(err)
		}
		w = widen(v)
		tol = Tol{}
	}
	g := widen(got)
	var bad []string
	worst := 0.0
	for i := range w {
		if within(g[i], w[i], tol) {
			continue
		}
		worst = max(worst, math.Abs(g[i]-w[i]))
		if len(bad) < 5 {
			bad = append(bad, fmt.Sprintf("[%d] got %v, want %v", i, g[i], w[i]))
		}
	}
	if bad != nil {
		tb.Errorf("mismatch (atol %g, rtol %g; worst abs error %g):\n  %s", tol.Atol, tol.Rtol, worst, strings.Join(bad, "\n  "))
	}
}

func within(got, want float64, tol Tol) bool {
	switch {
	case math.IsNaN(want) || math.IsNaN(got):
		return math.IsNaN(want) && math.IsNaN(got)
	case math.IsInf(want, 0) || math.IsInf(got, 0):
		return got == want
	}
	return math.Abs(got-want) <= tol.Atol+tol.Rtol*math.Abs(want)
}

func widen[T float32 | int32](v []T) []float64 {
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = float64(x)
	}
	return out
}

// Tensor returns a golden F32 or I32 tensor as a tensor.
func Tensor(tb testing.TB, g safetensors.Tensor) *tensor.Tensor {
	tb.Helper()
	switch g.DType {
	case "F32":
		v, err := g.Float32s()
		if err != nil {
			tb.Fatal(err)
		}
		return tensor.FromSlice(v, g.Shape...)
	case "I32":
		v, err := g.Int32s()
		if err != nil {
			tb.Fatal(err)
		}
		return tensor.FromSlice(v, g.Shape...)
	}
	tb.Fatalf("unsupported golden dtype %s", g.DType)
	return nil
}

// Inputs returns c's inputs as tensors. Inputs with a golden gradient
// require gradients.
func Inputs(tb testing.TB, c Case) []*tensor.Tensor {
	tb.Helper()
	in := make([]*tensor.Tensor, len(c.Inputs))
	for i, g := range c.Inputs {
		in[i] = Tensor(tb, g)
		if _, ok := c.Grads[i]; ok {
			in[i] = in[i].RequiresGrad()
		}
	}
	return in
}

// Grader has an accumulated gradient: a tensor from RequiresGrad, or a
// Param.
type Grader interface {
	Grad() *tensor.Tensor
}

// CheckGradients compares out with c.Want, runs Backward on
// sum(out * c.Gout), and compares the gradient for each input i that has
// a golden gradient with grads[i].Grad().
func CheckGradients(t *testing.T, c Case, out *tensor.Tensor, grads map[int]Grader, tol Tol) {
	t.Helper()
	ctx := t.Context()
	if err := out.Err(); err != nil {
		t.Fatal(err)
	}
	if err := out.Mul(Tensor(t, c.Gout)).Sum().Backward(ctx); err != nil {
		t.Fatal(err)
	}
	CompareTensor(t, c.Want, out, tol)
	for i, want := range c.Grads {
		src, ok := grads[i]
		if !ok {
			t.Fatalf("no gradient source for input %d", i)
		}
		g := src.Grad()
		if g == nil {
			t.Fatalf("input %d has no gradient", i)
		}
		t.Run(fmt.Sprintf("grad%d", i), func(t *testing.T) { CompareTensor(t, want, g, tol) })
	}
}

// CompareTensor computes got and compares it with want, like Compare.
func CompareTensor(tb testing.TB, want safetensors.Tensor, got *tensor.Tensor, tol Tol) {
	tb.Helper()
	ctx := context.Background()
	if want.DType == "I32" {
		v, err := got.Data[int32](ctx)
		if err != nil {
			tb.Fatal(err)
		}
		Compare(tb, want, v, got.Shape(), tol)
		return
	}
	v, err := got.Data[float32](ctx)
	if err != nil {
		tb.Fatal(err)
	}
	Compare(tb, want, v, got.Shape(), tol)
}
