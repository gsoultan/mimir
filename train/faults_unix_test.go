//go:build unix

package train_test

import "syscall"

// pageFaults returns the page faults the process has taken, and whether
// the platform counts them.
func pageFaults() (int64, bool) {
	var ru syscall.Rusage
	if syscall.Getrusage(syscall.RUSAGE_SELF, &ru) != nil {
		return 0, false
	}
	return int64(ru.Minflt + ru.Majflt), true
}
