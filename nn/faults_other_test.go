//go:build !unix

package nn_test

// pageFaults reports that the platform does not count page faults.
func pageFaults() (int64, bool) { return 0, false }
