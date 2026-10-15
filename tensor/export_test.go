package tensor

import "github.com/gsoultan/mimir/backend"

// SetDevice makes every tensor run on be until the returned function
// restores the previous device.
func SetDevice(be backend.Backend) (restore func()) {
	prev := device
	device = be
	return func() { device = prev }
}

// Eager reports whether operations are computed as soon as they are
// created.
func Eager() bool { return eager }
