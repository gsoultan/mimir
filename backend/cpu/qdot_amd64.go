//go:build !purego

package cpu

//go:generate sh -c "cd ../../tools/asmgen && go run ./qamd64 -out ../../backend/cpu/qdot_amd64.s -stubs ../../backend/cpu/qdot_stub_amd64.go -pkg cpu"

import "github.com/gsoultan/mimir/backend"

func init() {
	if f := x86Features(); !f.avx2 || !f.f16c {
		return
	}
	qdotISA = "avx2"
	qdots[backend.Q8_0] = &[maxRows]qdot{qdotQ8_0AVX2x1, qdotQ8_0AVX2x2, qdotQ8_0AVX2x3, qdotQ8_0AVX2x4}
	qdots[backend.Q4_K] = &[maxRows]qdot{qdotQ4KAVX2x1, qdotQ4KAVX2x2, qdotQ4KAVX2x3, qdotQ4KAVX2x4}
	qdots[backend.Q6_K] = &[maxRows]qdot{qdotQ6KAVX2x1, qdotQ6KAVX2x2, qdotQ6KAVX2x3, qdotQ6KAVX2x4}
}
