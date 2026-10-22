//go:build !purego && !linux

package cpu

// amxPermitted is false outside Linux, where mimir doesn't yet ask the
// operating system for the AMX tile state.
func amxPermitted() bool { return false }
