package mim

import (
	"context"
	"fmt"
	"slices"
	"sync"

	mimfile "github.com/gsoultan/mimir-model"

	"github.com/gsoultan/mimir/tensor"
)

// Session runs a model and holds its variables, such as the key-value
// cache of a language model, between runs. Its methods may be called from
// several goroutines; runs of one session take turns.
//
// Sessions of one Model are independent: each holds its own variables,
// and they share only the model's weights, which no run writes, so runs
// of different sessions may go on at once on different goroutines,
// sharing the devices as any concurrent computations do. A
// server gives each worker slot, or each conversation whose cache it
// keeps, a session of its own from NewSession, rather than calling
// Model.Run, whose one session every caller shares and waits for.
type Session struct {
	m    *Model
	mu   sync.Mutex
	vars map[uint32]*tensor.Tensor // current values, once a run set them
}

// NewSession returns a session whose variables start from their initial
// values, independent of the model's other sessions. It is cheap: the
// variables take memory once a run sets them.
func (m *Model) NewSession() *Session { return &Session{m: m} }

// Reset puts the session's variables back to their initial values.
func (s *Session) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.vars = nil
}

// Run computes the model's outputs from its inputs, keyed by name. Each
// input must be given, with the dtype that holds its type (see
// ValueInfo.DType) and a shape that matches the declared one, whose
// symbolic dimensions take any size up to their bound. The inputs must
// share a device, on which the model runs and the outputs lie; with no
// inputs it runs on the default device.
//
// Variables read the values the previous run of the session left. Once
// the outputs and the variables' new values are computed, the variables
// take them; a run that fails changes none.
//
// A run is inference: it records no tape, even from inputs that require
// gradients, so its outputs are not tracked (see tensor.Tensor.Detach)
// and the intermediate values are freed once computed.
func (s *Session) Run(ctx context.Context, inputs map[string]*tensor.Tensor) (map[string]*tensor.Tensor, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	m := s.m
	if m.g == nil {
		return nil, ErrNoGraph
	}
	r := &run{ctx: ctx, m: m, dev: tensor.DefaultDevice(), syms: slices.Repeat([]int{-1}, len(m.g.SymDims))}
	r.env = make([]*tensor.Tensor, len(m.g.Values))
	main := m.g.Main()
	for i, vi := range main.Inputs {
		info := m.inputs[i]
		t := inputs[info.Name]
		if t == nil {
			return nil, fmt.Errorf("mim: input %q is missing: %w", info.Name, tensor.ErrShape)
		}
		if err := t.Err(); err != nil {
			return nil, fmt.Errorf("mim: input %q: %w", info.Name, err)
		}
		if t.DType() != info.DType() {
			return nil, fmt.Errorf("mim: input %q of type %s is a tensor of %s, want %s: %w", info.Name, info.Type, t.DType(), info.DType(), tensor.ErrDType)
		}
		if err := r.bind(vi, t.Shape()); err != nil {
			return nil, fmt.Errorf("mim: input %q: %w", info.Name, err)
		}
		if i == 0 {
			r.dev = t.Device()
		} else if t.Device() != r.dev {
			return nil, fmt.Errorf("mim: input %q on %v, others on %v: %w", info.Name, t.Device(), r.dev, tensor.ErrDevice)
		}
		// The weights and constants are not tracked, so untracked inputs
		// leave the whole run untracked.
		r.env[vi] = t.Detach()
	}
	for name := range inputs {
		if !slices.ContainsFunc(m.inputs, func(v ValueInfo) bool { return v.Name == name }) {
			return nil, fmt.Errorf("mim: the model has no input %q: %w", name, tensor.ErrShape)
		}
	}
	consts, err := m.constsOn(r.dev)
	if err != nil {
		return nil, err
	}
	for i, v := range m.g.Values {
		vi := uint32(i)
		switch v.Kind {
		case mimfile.KindWeight, mimfile.KindConst:
			r.env[vi] = consts[vi]
		case mimfile.KindVariable:
			t := s.vars[vi]
			if t == nil || t.Device() != r.dev {
				if v.Zero {
					if err := m.checkSize(v, vi, r.declared(vi)); err != nil {
						return nil, err
					}
				}
				t = initial(v, consts[vi], r.dev)
			}
			r.env[vi] = t
		default:
			continue
		}
		if err := r.bind(vi, r.env[vi].Shape()); err != nil {
			return nil, fmt.Errorf("mim: %s %q: %w", kindName(v.Kind), valueName(v, vi), err)
		}
	}

	// A run of this binding on this device replays the program of an
	// earlier one that this run's inputs and variables fit, if there is
	// one (see replay.go).
	key, leaves := r.key(), r.leaves()
	recs, record := m.recordings(key)
	for _, rec := range recs {
		if rec == nil {
			continue
		}
		outs, ok, err := rec.Replay(ctx, leaves)
		if err != nil {
			return nil, err
		}
		if ok {
			replayed.Add(1)
			return s.finish(outs)
		}
	}
	outs, err := r.body(main)
	if err != nil {
		return nil, err
	}
	ts := make([]*tensor.Tensor, 0, len(outs)+len(m.g.Updates))
	ts = append(ts, outs...)
	for _, u := range m.g.Updates {
		t := r.env[u.Value]
		if t == nil {
			return nil, fmt.Errorf("%w: the update of variable %q is not computed", ErrFormat, valueName(m.g.Values[u.Variable], u.Variable))
		}
		ts = append(ts, t)
	}
	if !record {
		if err := tensor.Realize(ctx, ts...); err != nil {
			return nil, err
		}
	} else {
		rec, err := tensor.Record(ctx, leaves, ts)
		if err != nil {
			return nil, err
		}
		m.addRecording(key, rec)
	}
	return s.finish(ts)
}

// finish returns the outputs of a run, the first of ts, computed, and
// gives the variables their new values, the rest of ts, one per update
// of the graph.
func (s *Session) finish(ts []*tensor.Tensor) (map[string]*tensor.Tensor, error) {
	m := s.m
	res := make(map[string]*tensor.Tensor, len(m.outputs))
	for i, info := range m.outputs {
		res[info.Name] = ts[i]
	}
	if updates := ts[len(m.outputs):]; len(updates) > 0 {
		if s.vars == nil {
			s.vars = map[uint32]*tensor.Tensor{}
		}
		for i, u := range m.g.Updates {
			s.vars[u.Variable] = updates[i]
		}
	}
	return res, nil
}

// initial returns the value a variable starts from: its stored tensor,
// or zeros.
func initial(v mimfile.Value, stored *tensor.Tensor, d tensor.Device) *tensor.Tensor {
	if !v.Zero {
		return stored
	}
	dt, _ := storage(v.DType)
	shape := make([]int, len(v.Shape))
	for i, d := range v.Shape {
		shape[i] = int(d) // a zero-initialized variable has a literal shape
	}
	return tensor.Zeros(dt, shape...).To(d)
}

func kindName(k mimfile.ValueKind) string {
	switch k {
	case mimfile.KindWeight:
		return "weight"
	case mimfile.KindConst:
		return "constant"
	case mimfile.KindVariable:
		return "variable"
	case mimfile.KindInput:
		return "input"
	case mimfile.KindOutput:
		return "output"
	}
	return "value"
}

// run is the state of one run of a graph.
type run struct {
	ctx  context.Context
	m    *Model
	dev  tensor.Device
	syms []int            // the size of each symbolic dimension, -1 until bound
	env  []*tensor.Tensor // the value of each graph value, once computed
}

// bind checks shape against the declared shape of value vi, binding the
// symbolic dimensions not yet bound.
func (r *run) bind(vi uint32, shape []int) error {
	decl := r.m.g.Values[vi].Shape
	if len(shape) != len(decl) {
		return fmt.Errorf("shape %v, want rank %d: %w", shape, len(decl), tensor.ErrShape)
	}
	for i, d := range decl {
		if !d.Symbolic() {
			if int64(shape[i]) != int64(d) {
				return fmt.Errorf("shape %v, want %v: %w", shape, r.declared(vi), tensor.ErrShape)
			}
			continue
		}
		s := d.SymIndex()
		switch {
		case r.syms[s] >= 0 && r.syms[s] != shape[i]:
			return fmt.Errorf("shape %v, want %v with %s = %d: %w", shape, r.declared(vi), r.m.g.SymDims[s].Name, r.syms[s], tensor.ErrShape)
		case r.syms[s] < 0:
			if b := r.m.g.SymDims[s].Bound; b >= 0 && int64(shape[i]) > b {
				return fmt.Errorf("shape %v: %s = %d is above its bound %d: %w", shape, r.m.g.SymDims[s].Name, shape[i], b, tensor.ErrShape)
			}
			r.syms[s] = shape[i]
		}
	}
	return nil
}

// declared returns the declared shape of value vi, with -1 for the
// symbolic dimensions not yet bound.
func (r *run) declared(vi uint32) []int {
	decl := r.m.g.Values[vi].Shape
	shape := make([]int, len(decl))
	for i, d := range decl {
		if d.Symbolic() {
			shape[i] = r.syms[d.SymIndex()]
		} else {
			shape[i] = int(d)
		}
	}
	return shape
}

// subgraph runs the ops of sg on its inputs, given in order, and returns
// its outputs.
func (r *run) subgraph(sg *mimfile.Subgraph, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) != len(sg.Inputs) {
		return nil, fmt.Errorf("%w: a subgraph of %d inputs given %d", ErrFormat, len(sg.Inputs), len(in))
	}
	for i, vi := range sg.Inputs {
		if err := r.bind(vi, in[i].Shape()); err != nil {
			return nil, fmt.Errorf("mim: subgraph input %q: %w", valueName(r.m.g.Values[vi], vi), err)
		}
		r.env[vi] = in[i]
	}
	return r.body(sg)
}

// body runs the ops of sg, whose inputs are set, and returns its outputs.
func (r *run) body(sg *mimfile.Subgraph) ([]*tensor.Tensor, error) {
	for oi := range sg.Ops {
		if err := r.ctx.Err(); err != nil {
			return nil, err
		}
		if err := r.op(&sg.Ops[oi]); err != nil {
			return nil, err
		}
	}
	out := make([]*tensor.Tensor, len(sg.Outputs))
	for i, vi := range sg.Outputs {
		if out[i] = r.env[vi]; out[i] == nil {
			return nil, fmt.Errorf("%w: output %q is not computed", ErrFormat, valueName(r.m.g.Values[vi], vi))
		}
	}
	return out, nil
}
