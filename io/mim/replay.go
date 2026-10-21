package mim

import (
	"strconv"
	"strings"
	"sync/atomic"

	mimfile "github.com/gsoultan/mimir-model"

	"github.com/gsoultan/mimir/tensor"
)

// Runs of a model with inputs of the same shapes build the same graph
// over the same weights, and the scheduler compiles it to the same
// program; only the inputs and the variables differ. So the first such
// run on a device is recorded (see tensor.Record), and later runs replay
// its program over their inputs and variables, building and walking no
// graph. A run whose inputs or variables do not fit the recording, as
// ones laid out another way, builds its graph as the first did.
//
// A binding of the symbolic dimensions names the recordings, since the
// shapes of every value follow from it. A binding may have a few: a
// session's first run reads variables at their initial values, laid out
// as the zeros they start from, and the runs after it reads the values
// the last run left, so a run that fits no recording of its binding is
// recorded too, up to maxRecordings of them. A model whose graph depends
// on the values it reads, as one with a while or an if does, is not
// recorded: a replay computes the graph of the recorded run, whatever
// the values. The tables a composite builds from its attributes, such as
// rope's frequencies, a replay reads again, as tensor.Record documents.

// maxRecordings is the most recordings a binding keeps, and the most
// runs of it that are recorded or found unrecordable.
const maxRecordings = 3

// replaying turns the replay of recorded runs off, for tests that
// compare a replay with the run it stands for; replayed counts the runs
// that replayed a recording, for tests.
var (
	replaying = true
	replayed  atomic.Int64
)

// dataDependent reports whether the graph a run of m builds depends on
// the values it reads: whether a subgraph runs a while or an if.
func dataDependent(g *mimfile.Graph) bool {
	for _, sg := range g.Subgraphs {
		for _, op := range sg.Ops {
			if op.Opcode == mimfile.OpWhile || op.Opcode == mimfile.OpIf {
				return true
			}
		}
	}
	return false
}

// recordingKey names the recording of a device and binding.
type recordingKey struct {
	dev  tensor.Device
	syms string
}

// recordings returns the recordings of key, and whether a run of key
// that fits none of them is to be recorded.
func (m *Model) recordings(key recordingKey) ([]*tensor.Recording, bool) {
	if !replaying || m.dataDependent {
		return nil, false
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	recs := m.recorded[key]
	return recs, len(recs) < maxRecordings
}

// addRecording keeps rec, nil for a run that could not be recorded,
// among the recordings of key.
func (m *Model) addRecording(key recordingKey, rec *tensor.Recording) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed || len(m.recorded[key]) >= maxRecordings {
		return
	}
	if m.recorded == nil {
		m.recorded = map[recordingKey][]*tensor.Recording{}
	}
	m.recorded[key] = append(m.recorded[key], rec)
}

// key returns the recording key of r's device and binding.
func (r *run) key() recordingKey {
	var b strings.Builder
	for i, s := range r.syms {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(strconv.Itoa(s))
	}
	return recordingKey{dev: r.dev, syms: b.String()}
}

// leaves returns the tensors a run of the main graph reads: its inputs,
// then the weights, constants and variables, in the graph's order.
func (r *run) leaves() []*tensor.Tensor {
	g := r.m.g
	main := g.Main()
	leaves := make([]*tensor.Tensor, 0, len(main.Inputs)+len(r.m.consts))
	for _, vi := range main.Inputs {
		leaves = append(leaves, r.env[vi])
	}
	for vi, v := range g.Values {
		switch v.Kind {
		case mimfile.KindWeight, mimfile.KindConst, mimfile.KindVariable:
			leaves = append(leaves, r.env[vi])
		}
	}
	return leaves
}
