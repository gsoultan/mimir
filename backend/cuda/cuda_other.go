//go:build !(linux && (amd64 || arm64)) || nocuda

package cuda

import (
	"context"
	"fmt"

	"github.com/gsoultan/mimir/backend"
)

// Backend runs kernels on a CUDA device. This build has none: the
// platform is not Linux on amd64 or arm64, or the nocuda build tag is
// set, and New always fails.
type Backend struct{}

var _ backend.Backend = (*Backend)(nil)

// New reports ErrUnavailable: the backend is not built in.
func New() (*Backend, error) {
	return nil, fmt.Errorf("cuda: not built in (needs Linux on amd64 or arm64, without the nocuda tag): %w", ErrUnavailable)
}

// Name returns "cuda".
func (*Backend) Name() string { return "cuda" }

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
