package metal

import "github.com/gsoultan/mimir/backend/internal/gen"

// packed reports whether each operand's matrices lie one after another,
// rows × ld elements apart, or there is one matrix. Metal Performance
// Shaders on an M5 under macOS 26 reads batched matrices that far apart
// whatever the descriptor's matrix stride says, so a product of batches
// spaced further apart runs elsewhere.
func packed(g gen.GEMM) bool {
	return g.Batch == 1 || g.A.Stride == g.A.Rows*g.A.LD && g.B.Stride == g.B.Rows*g.B.LD
}
