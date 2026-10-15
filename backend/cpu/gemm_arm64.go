//go:build !purego

package cpu

//go:generate sh -c "cd ../../tools/asmgen && go run ./arm64 -out ../../backend/cpu/gemm_arm64.s"

// Every arm64 CPU has NEON. The kernel's B panel of kc×nr stays well
// inside the 64 KB L1 cache of current cores, and nc is a multiple of nr.
var neonKernel = gemmKernel{name: "neon", mr: 8, nr: 12, kc: 256, mc: 128, nc: 1536, tile: tileNEON}

func init() {
	gemmKernels = append([]*gemmKernel{&neonKernel}, gemmKernels...)
}

// tileNEON computes one 8×12 tile; see gemmKernel.tile.
//
//go:noescape
func tileNEON(kc int, a, b, c *float32, ldc int, accumulate bool)
