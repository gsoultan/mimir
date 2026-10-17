//go:build !amd64 || purego

package cpu

// rowRoutineSets returns every set of row routines this CPU supports, by
// name, for tests: none without amd64 assembly.
func rowRoutineSets() map[string]*rowRoutines { return nil }
