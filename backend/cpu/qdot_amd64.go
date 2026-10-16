//go:build !purego

package cpu

//go:generate sh -c "cd ../../tools/asmgen && go run ./qamd64 -out ../../backend/cpu/qdot_amd64.s -stubs ../../backend/cpu/qdot_stub_amd64.go -pkg cpu"

import "github.com/gsoultan/mimir/backend"

func init() {
	if f := x86Features(); !f.avx2 || !f.f16c {
		return
	}
	avx2 := map[backend.DType]*[maxRows]qdot{
		backend.Q8_0: {qdotQ8_0AVX2x1, qdotQ8_0AVX2x2, qdotQ8_0AVX2x3, qdotQ8_0AVX2x4},
		backend.Q4_K: {qdotQ4KAVX2x1, qdotQ4KAVX2x2, qdotQ4KAVX2x3, qdotQ4KAVX2x4},
		backend.Q6_K: {qdotQ6KAVX2x1, qdotQ6KAVX2x2, qdotQ6KAVX2x3, qdotQ6KAVX2x4},
	}
	qdotSets["avx2"] = avx2
	qdotISA, qdots = "avx2", avx2
}
