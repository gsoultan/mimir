//go:build linux && (amd64 || arm64) && !norocm

package rocm

// SetGraphs turns HIP graphs on or off for the tests of other packages
// until restore is called.
func SetGraphs(on bool) (restore func()) {
	was := graphsOn
	graphsOn = on
	return func() { graphsOn = was }
}

// GraphLaunches returns the number of batches launched as HIP graphs so
// far, the number launched one call at a time, and the graphs captured.
func GraphLaunches() (graphs, direct, captures int64) {
	return graphCounts.launches.Load(), graphCounts.direct.Load(), graphCounts.captures.Load()
}
