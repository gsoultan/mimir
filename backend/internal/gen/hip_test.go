package gen_test

import (
	"bytes"
	"fmt"
	"strings"
	"testing"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/backend/internal/gen"
	"github.com/gsoultan/mimir/backend/internal/kerneltest"
)

// TestHIPShufflesSumAsCUDASums checks that the HIP dialects shuffle within
// wavefronts where the CUDA dialect synchronizes the group, in the long
// reductions and the quantized products, that their kernels declare their
// group size, and, with MIMIR_EMULATE=1, that they compute what CUDA's
// kernels do bit for bit, for wavefronts of 64 and of 32.
func TestHIPShufflesSumAsCUDASums(t *testing.T) {
	if gen.HIP(64) != gen.HIP64 || gen.HIP(32) != gen.HIP32 || gen.HIP(16) != nil {
		t.Fatal("HIP returns the wrong dialects")
	}
	cuda := kerneltest.NewEmulator(t, gen.CUDA)
	for _, d := range []*gen.Dialect{gen.HIP64, gen.HIP32} {
		e := kerneltest.NewEmulator(t, d)
		shuffled := map[backend.Op]int{}
		for _, c := range append(kerneltest.Reduce(t), kerneltest.Quant(t)...) {
			n, _ := backend.NumElements(c.K.Shape)
			outs := []backend.Buffer{kerneltest.HostBuffer(t, nil, c.K.DType, n, 0, 0)}
			l, err := gen.Plan(d, &c.K, outs, c.Ins)
			if err != nil || l == nil {
				continue
			}
			if !strings.Contains(l.Src, fmt.Sprintf("__launch_bounds__(%d) k(", gen.GroupSize)) {
				t.Fatalf("%s %s: the kernel does not declare its group size:\n%s", d.Name, c.Name, l.Src)
			}
			if !strings.Contains(l.Src, "__shfl_down(") {
				continue
			}
			shuffled[c.K.Op]++
			if c.K.Op == backend.OpQMatMul && strings.Contains(l.Src, "__shared__") {
				t.Errorf("%s %s: the product sums through shared memory", d.Name, c.Name)
			}
			if e == nil || cuda == nil {
				continue
			}
			want, _ := cuda.Run(t, &c.K, outs, c.Ins)
			got, _ := e.Run(t, &c.K, []backend.Buffer{kerneltest.HostBuffer(t, nil, c.K.DType, n, 0, 0)}, c.Ins)
			if !bytes.Equal(got[0], want[0]) {
				t.Errorf("%s %s: the shuffles sum otherwise than CUDA's kernel", d.Name, c.Name)
			}
		}
		for _, op := range []backend.Op{backend.OpReduceSum, backend.OpReduceMax, backend.OpQMatMul} {
			if shuffled[op] == 0 {
				t.Errorf("%s: no %s kernel shuffles", d.Name, op)
			}
		}
	}
	for _, d := range []*gen.Dialect{gen.Metal, gen.CUDA} {
		if d.Wave() != 0 {
			t.Errorf("%s: wavefronts of %d, want none", d.Name, d.Wave())
		}
	}
}
