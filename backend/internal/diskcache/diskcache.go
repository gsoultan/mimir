// Package diskcache keeps the kernels a GPU backend compiles between
// runs, so that a program's second run compiles nothing. Entries live
// under MIMIR_CACHE_DIR or the user's cache directory, in a directory per
// backend, and a run that cannot write there compiles everything again.
package diskcache

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
)

// Cache is one backend's directory of compiled kernels. A nil Cache holds
// nothing and stores nothing.
type Cache struct {
	dir string
}

// Open returns the cache of the backend name, such as "cuda", or nil if
// there is no directory for it.
func Open(name string) *Cache {
	dir := os.Getenv("MIMIR_CACHE_DIR")
	if dir == "" {
		base, err := os.UserCacheDir()
		if err != nil {
			return nil
		}
		dir = filepath.Join(base, "mimir")
	}
	dir = filepath.Join(dir, name)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil
	}
	return &Cache{dir: dir}
}

// Key returns the name of the entry compiled from src with the given
// compiler and options.
func (*Cache) Key(compiler string, options []string, src string) string {
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

// Get returns the entry key, if there is one.
func (c *Cache) Get(key string) ([]byte, bool) {
	if c == nil {
		return nil, false
	}
	data, err := os.ReadFile(filepath.Join(c.dir, key))
	return data, err == nil && len(data) > 0
}

// Put stores data as the entry key. Another process storing the same
// entry at the same time is harmless: whichever rename lands last wins,
// and both wrote the same bytes.
func (c *Cache) Put(key string, data []byte) {
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
