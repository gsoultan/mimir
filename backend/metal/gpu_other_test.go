//go:build !(darwin && arm64)

package metal

import (
	"testing"

	"github.com/gsoultan/mimir/backend/internal/kerneltest"
)

// newGPU returns nil: there is no Metal device on this platform.
func newGPU(*testing.T, int, int) kerneltest.Target { return nil }
