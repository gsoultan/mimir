//go:build linux && (amd64 || arm64) && !nocuda

package cuda

import (
	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/backend/internal/gen"
)

// The backend runs convolutions as the generator's implicit-GEMM kernels,
// which read the windows of their image where they lie instead of
// copying them out first.
var _ backend.Convolver = (*Backend)(nil)

// ConvParts returns the number of parts into which the weight gradient k
// splits so that the device sums them in parallel.
func (*Backend) ConvParts(k *backend.Kernel) int { return gen.ConvParts(k) }
