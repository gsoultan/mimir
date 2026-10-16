//go:build !unix && !windows

package safetensors

import (
	"errors"
	"os"
)

// mapFile reports that the platform cannot map files, so Open reads them
// into memory instead.
func mapFile(*os.File, int) ([]byte, func() error, error) {
	return nil, nil, errors.ErrUnsupported
}
