//go:build !purego

package cpu

//go:generate sh -c "cd ../../tools/asmgen && go run ./pool -out ../../backend/cpu/pool_amd64.s -stubs ../../backend/cpu/pool_stub_amd64.go -pkg cpu"

func init() {
	if x86Features().avx512 {
		poolMax2x2, poolGrad2x2 = poolMax2x2AVX512, poolGrad2x2AVX512
	}
}
