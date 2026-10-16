//go:build !(darwin && arm64)

package metal

import "testing"

// newGPU returns nil: there is no Metal device on this platform.
func newGPU(*testing.T, int) target { return nil }
