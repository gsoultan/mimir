//go:build !purego

package cpu

//go:generate sh -c "cd ../../tools/asmgen && go run ./linalg -out ../../backend/cpu/linalg_amd64.s -stubs ../../backend/cpu/linalg_stub_amd64.go -pkg cpu"

func init() {
	switch f := x86Features(); {
	case f.avx512:
		axpyVector, axpyLanes = axpyAVX512, 16
	case f.avx2 && f.fma:
		axpyVector, axpyLanes = axpyAVX2, 8
	}
}
