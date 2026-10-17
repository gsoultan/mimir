//go:build !unix && !windows

package filemap

import (
	"errors"
	"os"
)

// mapFile reports that the platform cannot map files, so Read reads them
// into memory instead.
func mapFile(*os.File, int) ([]byte, func() error, error) {
	return nil, nil, errors.ErrUnsupported
}
