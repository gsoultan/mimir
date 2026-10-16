// Package metal implements the mimir backend for Apple GPUs through
// Metal, without cgo.
//
// It calls the Objective-C runtime and Metal through purego and runs every
// kernel as Metal Shading Language source that it generates and compiles
// at runtime, once per kernel structure. Buffers live in memory the CPU
// and GPU share, so uploads and downloads are copies within RAM.
//
// The backend is available on Apple silicon Macs. Elsewhere New reports
// ErrUnavailable.
package metal

import "errors"

// ErrUnavailable reports that no Metal device can be used.
var ErrUnavailable = errors.New("metal unavailable")
