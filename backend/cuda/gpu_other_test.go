//go:build !(linux && (amd64 || arm64)) || nocuda

package cuda

import (
	"testing"

	"github.com/gsoultan/mimir/backend/internal/kerneltest"
)

// newGPU returns nil: the backend is not built in.
func newGPU(*testing.T, int) kerneltest.Target { return nil }
