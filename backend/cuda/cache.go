//go:build linux && (amd64 || arm64) && !nocuda

package cuda

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
)

// diskCache keeps compiled kernels between runs, so that a program's
// second run compiles nothing. It lives under MIMIR_CACHE_DIR or the
// user's cache directory, and a run that cannot write there compiles
// everything again.
type diskCache struct {
	dir string
}

// openDiskCache returns the cache, or nil if there is no directory for
// it.
func openDiskCache() *diskCache {
	dir := os.Getenv("MIMIR_CACHE_DIR")
	if dir == "" {
		base, err := os.UserCacheDir()
		if err != nil {
			return nil
		}
		dir = filepath.Join(base, "mimir")
	}
	dir = filepath.Join(dir, "cuda")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil
	}
	return &diskCache{dir: dir}
}

// key returns the name of the entry compiled from src with the given
// compiler and options.
func (*diskCache) key(compiler string, options []string, src string) string {
	h := sha256.New()
	h.Write([]byte(compiler))
	for _, o := range options {
		h.Write([]byte{0})
		h.Write([]byte(o))
	}
	h.Write([]byte{0})
	h.Write([]byte(src))
	return hex.EncodeToString(h.Sum(nil)[:16])
}

// get returns the entry key, if there is one.
func (c *diskCache) get(key string) ([]byte, bool) {
	if c == nil {
		return nil, false
	}
	data, err := os.ReadFile(filepath.Join(c.dir, key))
	return data, err == nil && len(data) > 0
}

// put stores data as the entry key. Another process storing the same
// entry at the same time is harmless: whichever rename lands last wins,
// and both wrote the same bytes.
func (c *diskCache) put(key string, data []byte) {
	if c == nil {
		return
	}
	tmp, err := os.CreateTemp(c.dir, key+".*")
	if err != nil {
		return
	}
	_, err = tmp.Write(data)
	if cerr := tmp.Close(); err == nil {
		err = cerr
	}
	if err == nil {
		err = os.Rename(tmp.Name(), filepath.Join(c.dir, key))
	}
	if err != nil {
		os.Remove(tmp.Name())
	}
}
