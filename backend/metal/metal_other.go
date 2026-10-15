//go:build !(darwin && arm64)

package metal

import (
	"context"
	"fmt"

	"github.com/gsoultan/mimir/backend"
)

// Backend runs kernels on a Metal device. On this platform there is none,
// and New always fails.
type Backend struct{}

var _ backend.Backend = (*Backend)(nil)

// New reports ErrUnavailable: Metal runs on Apple silicon Macs only.
func New() (*Backend, error) {
	return nil, fmt.Errorf("metal: needs an Apple silicon Mac: %w", ErrUnavailable)
}

// Name returns "metal".
func (*Backend) Name() string { return "metal" }

// DeviceName returns the empty string.
func (*Backend) DeviceName() string { return "" }

func (*Backend) Alloc(backend.DType, int) (backend.Buffer, error) { return nil, ErrUnavailable }
func (*Backend) Free(backend.Buffer)                              {}
func (*Backend) Upload(context.Context, backend.Buffer, []byte) error {
	return ErrUnavailable
}
func (*Backend) Download(context.Context, []byte, backend.Buffer) error {
	return ErrUnavailable
}
func (*Backend) Run(context.Context, *backend.Kernel, []backend.Buffer, []backend.Buffer) error {
	return ErrUnavailable
}
func (*Backend) Sync(context.Context) error { return ErrUnavailable }
