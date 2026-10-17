package train

import "time"

// SetPhases makes Parallel.Backward report the host's time in each of its
// phases to f until restore is called.
func SetPhases(f func(phase string, rank int, d time.Duration)) (restore func()) {
	was := phases
	phases = f
	return func() { phases = was }
}
