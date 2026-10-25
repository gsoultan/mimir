package cpu

// NewWorkers returns a CPU backend that uses up to workers goroutines per
// kernel.
func NewWorkers(workers int) *Backend {
	b := New()
	b.workers = workers
	return b
}

// SetDirectPositions sets the output positions below which a forward
// convolution of many channels runs as a direct one, and returns a
// function that restores the setting.
func SetDirectPositions(n int) (restore func()) {
	old := directPositions
	directPositions = n
	return func() { directPositions = old }
}
