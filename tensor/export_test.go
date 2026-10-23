package tensor

import "github.com/gsoultan/mimir/backend"

// SetDevice makes every tensor run on be until the returned function
// restores the previous device.
func SetDevice(be backend.Backend) (restore func()) {
	prev := SetDefaultDevice(Device{be})
	return func() { SetDefaultDevice(prev) }
}

// Eager reports whether operations are computed as soon as they are
// created.
func Eager() bool { return eager }

// DeviceOn returns a device that runs on be.
func DeviceOn(be backend.Backend) Device { return Device{be} }

// BackendOf returns the backend d runs on.
func BackendOf(d Device) backend.Backend { return d.backend() }

// Every analysis the plan cache finds is checked against one made anew.
func init() { checkPlans = true }

// PlanHits returns the number of analyses m found in its cache.
func PlanHits(m *Mesh) int {
	meshMu.Lock()
	defer meshMu.Unlock()
	return m.plans.hits
}

// SetCheckPlans turns the check of cached analyses on or off.
func SetCheckPlans(on bool) { checkPlans = on }

// PlanMisses returns the number of analyses m did not find in its cache.
func PlanMisses(m *Mesh) int {
	meshMu.Lock()
	defer meshMu.Unlock()
	return m.plans.misses
}

// PlanReplays returns how many runs on m replayed a recorded run.
func PlanReplays(m *Mesh) int {
	meshMu.Lock()
	defer meshMu.Unlock()
	return m.plans.replays
}

// ReplayMisses returns how many traced device schedules of replays on m
// computed their graphs rather than replay their programs.
func ReplayMisses(m *Mesh) int {
	meshMu.Lock()
	defer meshMu.Unlock()
	return m.plans.missed
}

// SetReplays turns the replay of recorded runs of sharded graphs on and
// off, and, if programs is false, makes replays compute the graphs their
// scripts build rather than replay the schedules' programs. It returns a
// function that restores the old settings.
func SetReplays(on, programs bool) func() {
	meshMu.Lock()
	defer meshMu.Unlock()
	was, wasPrograms := replays, replayPrograms
	replays, replayPrograms = on, programs
	return func() {
		meshMu.Lock()
		defer meshMu.Unlock()
		replays, replayPrograms = was, wasPrograms
	}
}

// BufferOf returns the buffer that holds the computed t.
func BufferOf(t *Tensor) backend.Buffer { return t.node.Buf() }
