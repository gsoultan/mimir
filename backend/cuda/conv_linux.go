//go:build linux && (amd64 || arm64) && !nocuda

package cuda

import (
	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/backend/internal/gen"
)

// The backend runs convolutions as the generator's implicit-GEMM kernels,
// which read the windows of their image where they lie instead of
// copying them out first, except those that cuBLAS runs faster as the
// product of their windows.
var _ backend.WindowChooser = (*Backend)(nil)

// ConvParts returns the number of parts into which the weight gradient k
// splits so that the device sums them in parallel.
func (*Backend) ConvParts(k *backend.Kernel) int { return gen.ConvParts(k) }

// A convolution kernel computes a tile of its product per group and
// walks the product's whole inner dimension, so a product of few tiles
// and a long inner dimension, as the deep layers of a network with many
// channels over a small image have, leaves most of the device idle while
// its few groups walk. cuBLAS splits such a product along its inner
// dimension, and runs at more than twice the kernels' speed besides; the
// windows it multiplies cost a copy out and back, which a product of many
// rows repays.
const (
	// windowedTiles is the fewest groups per multiprocessor that a
	// convolution kernel needs to keep the device busy.
	windowedTiles = 4
	// windowedRows is the fewest rows of the product that repay copying
	// out the windows, which each row reads in full.
	windowedRows = 64
	// windowedInner is the shortest inner dimension worth splitting.
	windowedInner = 1024
)

// Windowed reports whether the convolution k, of a layer whose items
// share its weights, runs faster as cuBLAS's product of its windows than
// as a convolution kernel: a forward pass or an input gradient of a
// product that has too few tiles to fill the device, a long inner
// dimension and enough rows to repay copying out the windows. A weight
// gradient, which sums over the items, has a kernel that splits it and
// no single product for cuBLAS.
func (b *Backend) Windowed(k *backend.Kernel) bool {
	b.mu.Lock()
	blas := b.blasMin >= 0
	b.mu.Unlock()
	c, err := backend.ConvOf(k)
	if !blas || err != nil || c.G != 1 {
		return false
	}
	var rows, inner int
	switch k.Op {
	case backend.OpConv:
		rows, inner = c.M, c.Rows()
	case backend.OpConvInput:
		rows, inner = c.C, c.M*c.Window
	default:
		return false
	}
	return rows >= windowedRows && inner >= windowedInner && gen.ConvGroups(k) < windowedTiles*b.sms
}
