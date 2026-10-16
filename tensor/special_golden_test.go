package tensor_test

import (
	"strings"
	"testing"

	"github.com/gsoultan/mimir/internal/goldentest"
	"github.com/gsoultan/mimir/tensor"
)

// specialGoldenCases mirrors the special_ cases of distribution_cases in
// tools/golden/gen.py.
var specialGoldenCases = map[string]func(x *tensor.Tensor) *tensor.Tensor{
	"special_log1p":           (*tensor.Tensor).Log1p,
	"special_expm1":           (*tensor.Tensor).Expm1,
	"special_erf":             (*tensor.Tensor).Erf,
	"special_erfinv":          (*tensor.Tensor).Erfinv,
	"special_lgamma":          (*tensor.Tensor).Lgamma,
	"special_lgamma_negative": (*tensor.Tensor).Lgamma,
	"special_digamma":         (*tensor.Tensor).Digamma,
}

// TestGoldenSpecialFunctions checks the special functions and their
// gradients against PyTorch or NumPy.
func TestGoldenSpecialFunctions(t *testing.T) {
	cases, meta := goldentest.Load(t, "distributions")
	t.Logf("golden oracle: %s", meta["oracle"])
	for name := range specialGoldenCases {
		if _, ok := cases[name]; !ok {
			t.Errorf("case %q has no golden data; regenerate with tools/golden/gen.py", name)
		}
	}
	for name, c := range cases {
		if !strings.HasPrefix(name, "special_") {
			continue
		}
		t.Run(name, func(t *testing.T) {
			f, ok := specialGoldenCases[name]
			if !ok {
				t.Fatalf("golden case %q has no Go counterpart", name)
			}
			x := goldentest.Inputs(t, c)[0]
			goldentest.CheckGradients(t, c, f(x), map[int]goldentest.Grader{0: x}, goldentest.Tol{Atol: 1e-6, Rtol: 1e-5})
		})
	}
}
