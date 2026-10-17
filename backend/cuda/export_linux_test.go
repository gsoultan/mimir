//go:build linux && (amd64 || arm64) && !nocuda

package cuda

// SetGraphs turns CUDA graphs on or off for the tests of other packages
// until restore is called.
func SetGraphs(on bool) (restore func()) {
	was := graphsOn
	graphsOn = on
	return func() { graphsOn = was }
}

// GraphLaunches returns the number of batches launched as CUDA graphs so
// far, the number launched one call at a time, and the graphs captured.
func GraphLaunches() (graphs, direct, captures int64) {
	return graphCounts.launches.Load(), graphCounts.direct.Load(), graphCounts.captures.Load()
}

// GraphUpdates returns the number of kernel nodes updated in executable
// graphs so far.
func GraphUpdates() int64 { return graphCounts.updates.Load() }
