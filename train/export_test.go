package train

import "time"

// SetPhases makes Parallel.Backward report the host's time in each of its
// phases to f until restore is called.
func SetPhases(f func(phase string, rank int, d time.Duration)) (restore func()) {
	was := phases
	phases = f
	return func() { phases = was }
}

// SetWaitPhases makes Parallel.Backward wait for the devices after each
// phase and report the waits to the function SetPhases set, until restore
// is called.
func SetWaitPhases(on bool) (restore func()) {
	was := waitPhases
	waitPhases = on
	return func() { waitPhases = was }
}
