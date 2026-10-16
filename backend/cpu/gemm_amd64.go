//go:build !purego

package cpu

//go:generate sh -c "cd ../../tools/asmgen && go run ./amd64 -out ../../backend/cpu/gemm_amd64.s -stubs ../../backend/cpu/gemm_stub_amd64.go -pkg cpu"

// The kernels are written in assembly rather than with archsimd: built
// from archsimd with Go 1.27, the compiler spills their accumulators and
// they run at a quarter of the speed.
var (
	avx512Kernel = gemmKernel{name: "avx512", mr: 8, nr: 32, kc: 256, mc: 144, nc: 2048, tile: tileAVX512}
	avx2Kernel   = gemmKernel{name: "avx2", mr: 6, nr: 16, kc: 256, mc: 144, nc: 2048, tile: tileAVX2}
)

func init() {
	f := x86Features()
	if f.avx {
		transpose8x8 = transpose8
	}
	if f.avx2 && f.fma {
		gemmKernels = append([]*gemmKernel{&avx2Kernel}, gemmKernels...)
	}
	if f.avx512 {
		gemmKernels = append([]*gemmKernel{&avx512Kernel}, gemmKernels...)
	}
}

type x86 struct {
	avx, avx2, fma, avx512 bool
	f16c                   bool // VCVTPH2PS, converting float16
}

// x86Features reports the vector extensions that both the CPU and the
// operating system support.
func x86Features() x86 {
	maxID, _, _, _ := cpuid(0, 0)
	if maxID < 7 {
		return x86{}
	}
	_, _, ecx1, _ := cpuid(1, 0)
	const osxsave, avx, fma, f16c = 1 << 27, 1 << 28, 1 << 12, 1 << 29
	if ecx1&osxsave == 0 || ecx1&avx == 0 {
		return x86{}
	}
	// The OS must save the vector state on context switches: XMM and YMM
	// for AVX, plus the opmask and ZMM registers for AVX-512.
	xcr0, _ := xgetbv()
	const ymmState, zmmState = 0x6, 0xe6
	if xcr0&ymmState != ymmState {
		return x86{}
	}
	_, ebx7, _, _ := cpuid(7, 0)
	const avx2, avx512f = 1 << 5, 1 << 16
	return x86{
		avx:    true,
		avx2:   ebx7&avx2 != 0,
		fma:    ecx1&fma != 0,
		f16c:   ecx1&f16c != 0,
		avx512: ebx7&avx512f != 0 && xcr0&zmmState == zmmState,
	}
}
