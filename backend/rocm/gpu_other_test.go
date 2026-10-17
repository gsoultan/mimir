//go:build !(linux && (amd64 || arm64)) || norocm

package rocm

import (
	"testing"

	"github.com/gsoultan/mimir/backend/internal/kerneltest"
)

// newGPU returns nil: the backend is not built in.
func newGPU(*testing.T, int) kerneltest.Target { return nil }
