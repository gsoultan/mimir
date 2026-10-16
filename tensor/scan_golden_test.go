package tensor_test

import (
	"testing"

	"github.com/gsoultan/mimir/internal/goldentest"
	"github.com/gsoultan/mimir/tensor"
)

// scanGoldenCases mirrors scan_cases in tools/golden/gen.py.
var scanGoldenCases = map[string]goldenCase{
	"cumsum_axis1": un(func(x *tensor.Tensor) *tensor.Tensor { return x.CumSum(1) }, tight),
	"cumsum_last":  un(func(x *tensor.Tensor) *tensor.Tensor { return x.CumSum(-1) }, tight),
	"cumsum_int":   un(func(x *tensor.Tensor) *tensor.Tensor { return x.CumSum(1) }, exact),
	"cummax_axis1": un(func(x *tensor.Tensor) *tensor.Tensor { v, _ := x.CumMax(1); return v }, exact),
	"cummax_axis0": un(func(x *tensor.Tensor) *tensor.Tensor { v, _ := x.CumMax(0); return v }, exact),
	"cummax_indices": un(func(x *tensor.Tensor) *tensor.Tensor {
		_, i := x.CumMax(1)
		return i
	}, exact),
	"logcumsumexp_last":  un(func(x *tensor.Tensor) *tensor.Tensor { return x.LogCumSumExp(-1) }, tight),
	"logcumsumexp_large": un(func(x *tensor.Tensor) *tensor.Tensor { return x.LogCumSumExp(0) }, tight),
}

// TestGoldenScans checks the scans, and the gradients of those that have
// them, against PyTorch or NumPy (see tools/golden/gen.py).
func TestGoldenScans(t *testing.T) {
	cases, meta := goldentest.Load(t, "scan")
	t.Logf("golden oracle: %s", meta["oracle"])
	for name := range scanGoldenCases {
		if _, ok := cases[name]; !ok {
			t.Errorf("case %q has no golden data; regenerate with tools/golden/gen.py", name)
		}
	}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			gc, ok := scanGoldenCases[name]
			if !ok {
				t.Fatalf("golden case %q has no Go counterpart", name)
			}
			in := goldentest.Inputs(t, c)
			if len(c.Grads) == 0 {
				goldentest.CompareTensor(t, c.Want, gc.fn(in...), gc.tol)
				return
			}
			grads := map[int]goldentest.Grader{}
			for i := range c.Grads {
				grads[i] = in[i]
			}
			goldentest.CheckGradients(t, c, gc.fn(in...), grads, goldentest.Tol{Atol: 1e-5, Rtol: 1e-4})
		})
	}
}
