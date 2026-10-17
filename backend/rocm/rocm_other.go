//go:build !(linux && (amd64 || arm64)) || norocm

package rocm

import (
	"context"
	"fmt"

	"github.com/gsoultan/mimir/backend"
)

// Backend runs kernels on a ROCm device. This build has none: the
// platform is not Linux on amd64 or arm64, or the norocm build tag is
// set, and New always fails.
type Backend struct{}

var _ backend.Backend = (*Backend)(nil)

// New reports ErrUnavailable: the backend is not built in.
func New() (*Backend, error) {
	return nil, fmt.Errorf("rocm: not built in (needs Linux on amd64 or arm64, without the norocm tag): %w", ErrUnavailable)
}

// Name returns "rocm".
func (*Backend) Name() string { return "rocm" }

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
