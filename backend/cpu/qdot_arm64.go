//go:build !purego

package cpu

//go:generate sh -c "cd ../../tools/asmgen && go run ./qarm64 -out ../../backend/cpu/qdot_arm64.s -stubs ../../backend/cpu/qdot_stub_arm64.go"

import (
	"encoding/binary"
	"os"
	"runtime"

	"github.com/gsoultan/mimir/backend"
)

// The NEON kernels of every arm64 CPU, and those that sum with SDOT.
var (
	qdotsNEON = map[backend.DType]*[maxRows]qdot{
		backend.Q8_0: {qdotQ8_0NEONx1, qdotQ8_0NEONx2, qdotQ8_0NEONx3, qdotQ8_0NEONx4},
		backend.Q4_K: {qdotQ4_KNEONx1, qdotQ4_KNEONx2, qdotQ4_KNEONx3, qdotQ4_KNEONx4},
		backend.Q6_K: {qdotQ6_KNEONx1, qdotQ6_KNEONx2, qdotQ6_KNEONx3, qdotQ6_KNEONx4},
	}
	qdotsSDOT = map[backend.DType]*[maxRows]qdot{
		backend.Q8_0: {qdotQ8_0SDOTx1, qdotQ8_0SDOTx2, qdotQ8_0SDOTx3, qdotQ8_0SDOTx4},
		backend.Q4_K: {qdotQ4_KSDOTx1, qdotQ4_KSDOTx2, qdotQ4_KSDOTx3, qdotQ4_KSDOTx4},
		backend.Q6_K: {qdotQ6_KSDOTx1, qdotQ6_KSDOTx2, qdotQ6_KSDOTx3, qdotQ6_KSDOTx4},
	}
)

func init() {
	qdotSets["neon"] = qdotsNEON
	qdotISA, qdots = "neon", qdotsNEON
	if hasDotProd() {
		qdotSets["sdot"] = qdotsSDOT
		qdotISA, qdots = "sdot", qdotsSDOT
		tiles := map[backend.DType]qtile{backend.Q4_K: qtileQ4_KSDOT, backend.Q6_K: qtileQ6_KSDOT}
		qtileSets["sdot"] = tiles
		qtiles = tiles
	}
}

// hasDotProd reports whether the CPU has the dot product instructions,
// SDOT among them: every Apple CPU does, and Linux reports them in the
// hardware capabilities of the auxiliary vector.
func hasDotProd() bool {
	switch runtime.GOOS {
	case "darwin", "ios":
		return true
	case "linux", "android":
	default:
		return false
	}
	auxv, err := os.ReadFile("/proc/self/auxv")
	if err != nil {
		return false
	}
	const atHWCap, hwcapASIMDDP = 16, 1 << 20
	for ; len(auxv) >= 16; auxv = auxv[16:] {
		if binary.LittleEndian.Uint64(auxv) == atHWCap {
			return binary.LittleEndian.Uint64(auxv[8:])&hwcapASIMDDP != 0
		}
	}
	return false
}
