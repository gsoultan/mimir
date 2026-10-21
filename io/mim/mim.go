// Package mim loads models in the Mimir Model format (.mim) and runs them
// with mimir tensors, on any device. The format is specified, and read,
// by github.com/gsoultan/mimir-model: a file holds a model's weights, its graph
// written in 33 primitive operations and calls of a standard library of
// composites, and typed metadata, such as a tokenizer.
//
// Open reads a model, Run computes it:
//
//	m, err := mim.Open("model.mim")
//	...
//	defer m.Close()
//	for _, in := range m.Inputs() {
//		fmt.Println(in) // x: f32[batch 3 224 224]
//	}
//	out, err := m.Run(ctx, map[string]*tensor.Tensor{"x": x})
//	probs := out["probs"]
//
// Run turns the graph into lazy mimir operations and realizes the outputs
// together, so a model is one schedule. Calls of the standard library run
// on mimir's own kernels where it has one, such as attention, the
// products with block-quantized weights and the convolutions, and as
// their canonical bodies otherwise. Weights stay on the CPU, read in place
// from a mapping of the file where mimir holds their dtype, and a run on
// another device copies them there the first time.
//
// A graph may keep state between runs in variables, such as the key-value
// cache of a language model. A Session holds them: each run reads the
// values the previous run of the session left, and updates them once
// all its outputs are computed. Model.Run runs in a session of its own,
// which its callers share; sessions from NewSession are independent and
// run at once, which is how a server serves one model to many requests:
//
//	s := m.NewSession() // one per worker or conversation
//	out, err := s.Run(ctx, inputs)
//
// Runs are inference: they record no tape for gradients.
//
// Values of the integer types and bool are held as Int32 tensors, so they
// must fit in an int32: inputs of type i64 or bool are Int32 tensors, and
// so are such outputs. Values of the 8-bit float types, and weights of the
// block types mimir does not compute with (q4_0, mxfp4, affine4_g64), are
// decoded to Float32 when the model is opened. A model that uses f64, a
// bitwise operation on integers or a composite mimir does not know is
// reported by Open as an *UnsupportedError naming it.
//
// Export writes a mimir model as a .mim file, and ExportFunc a function of
// tensors: they trace it once (see tensor.Trace) and write the operations
// it records, with the tensors it reads as weights and constants.
//
// Save writes the parameters of a model alone, as a checkpoint does, and
// Model.Load sets a model's parameters from such a file, or from the
// weights of any other:
//
//	err := mim.Save(ctx, w, model, nil)
//	...
//	f, err := mim.Open("model.mim")
//	...
//	defer f.Close()
//	err = f.Load(model)
package mim

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	mimfile "github.com/gsoultan/mimir-model"

	"github.com/gsoultan/mimir/internal/filemap"
	"github.com/gsoultan/mimir/tensor"
)

var (
	// ErrFormat reports a malformed file.
	ErrFormat = errors.New("mim: invalid file")
	// ErrUnsupported reports a model that is valid but uses an operation,
	// an attribute value or a data type that mimir does not run. An
	// *UnsupportedError carries the details.
	ErrUnsupported = errors.New("mim: unsupported")
	// ErrNoGraph reports a run of a file that holds weights alone, as Save
	// writes.
	ErrNoGraph = errors.New("mim: the file has no graph to run")
)

// UnsupportedError reports what of a model mimir does not run. It matches
// ErrUnsupported.
type UnsupportedError struct {
	Op     string // the operation, such as "shl" or "call top_p.v1"
	Detail string // what is unsupported, when it is not the operation itself
}

func (e *UnsupportedError) Error() string {
	s := "mim: unsupported " + e.Op
	if e.Detail != "" {
		s += ": " + e.Detail
	}
	return s
}

// Is reports whether target is ErrUnsupported.
func (e *UnsupportedError) Is(target error) bool { return target == ErrUnsupported }

// Dim is one dimension of a declared shape: a size, or, when Size is -1,
// the symbolic dimension Name, which takes its size from the inputs of a
// run and is at most Bound, or unbounded when Bound is -1.
type Dim struct {
	Size  int
	Name  string
	Bound int
}

func (d Dim) String() string {
	if d.Size >= 0 {
		return strconv.Itoa(d.Size)
	}
	return d.Name
}

// ValueInfo describes an input or an output of a model.
type ValueInfo struct {
	Name  string
	Type  mimfile.DType // as the file declares it
	Shape []Dim
}

// DType returns the dtype of the tensors that hold the value: Int32 for
// the integer types and bool.
func (v ValueInfo) DType() tensor.DType {
	dt, _ := storage(v.Type)
	return dt
}

func (v ValueInfo) String() string {
	dims := make([]string, len(v.Shape))
	for i, d := range v.Shape {
		dims[i] = d.String()
	}
	return fmt.Sprintf("%s: %s[%s]", v.Name, v.Type, strings.Join(dims, " "))
}

// Model is a loaded mim model. Its methods may be called from several
// goroutines; runs of one session take turns, and runs of different
// sessions (see NewSession) go on at once.
type Model struct {
	f        *mimfile.File
	g        *mimfile.Graph
	inputs   []ValueInfo
	outputs  []ValueInfo
	consts   map[uint32]*tensor.Tensor // weights, constants and initial variables, on the CPU
	producer []*mimfile.Op             // the op that computes each value, if any
	locals   map[string]uint32         // model-local composites by name.vN, their subgraphs

	maxBytes atomic.Int64 // the bound on each value a run computes; see SetMaxValueBytes

	mu       sync.Mutex
	onDevice map[tensor.Device]map[uint32]*tensor.Tensor
	recorded map[recordingKey][]*tensor.Recording // see replay.go; a nil entry is a run that could not be recorded
	// dataDependent is set for a graph that reads values as it is built,
	// whose runs are not recorded.
	dataDependent bool
	session       *Session
	mapping       *filemap.Mapping
	cleanup       runtime.Cleanup
	closed        bool
}

// Open reads the model at path. It maps the file, and the model's weights
// read the mapping in place where they can, until Close, and past it for
// as long as a tensor reads them.
func Open(path string) (*Model, error) {
	data, mp, err := filemap.Read(path)
	if errors.Is(err, filemap.ErrTooLarge) {
		return nil, fmt.Errorf("mim: %s: %w: %w", path, err, ErrFormat)
	}
	if err != nil {
		return nil, fmt.Errorf("mim: %w", err)
	}
	m, err := parse(data, mp)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return m, nil
}

// Parse reads a model from the bytes of a file. Its weights may alias b,
// which must stay unchanged while the model is in use.
func Parse(b []byte) (*Model, error) { return parse(b, nil) }

func parse(b []byte, mp *filemap.Mapping) (*Model, error) {
	m := &Model{mapping: mp, onDevice: map[tensor.Device]map[uint32]*tensor.Tensor{}}
	m.maxBytes.Store(DefaultMaxValueBytes)
	if err := m.load(b); err != nil {
		if mp != nil {
			_ = mp.Close()
		}
		return nil, err
	}
	if mp != nil {
		m.cleanup = runtime.AddCleanup(m, func(mp *filemap.Mapping) { _ = mp.Close() }, mp)
	}
	m.session = m.NewSession()
	return m, nil
}

func (m *Model) load(b []byte) error {
	f, err := mimfile.Parse(b)
	if err != nil {
		return fmt.Errorf("%w: %w", ErrFormat, err)
	}
	m.f = f
	if _, ok := f.Section(mimfile.SectionGraph); !ok {
		return nil
	}
	g, err := f.Graph()
	if err != nil {
		return fmt.Errorf("%w: %w", ErrFormat, err)
	}
	m.g = g
	m.dataDependent = dataDependent(g)
	if g.PrimitiveVersion > mimfile.CurrentPrimitive || g.StdlibVersion > mimfile.CurrentStdlib {
		return &UnsupportedError{Op: "graph", Detail: fmt.Sprintf("primitives v%d, stdlib v%d", g.PrimitiveVersion, g.StdlibVersion)}
	}
	m.locals = make(map[string]uint32, len(g.Composites))
	for _, c := range g.Composites {
		m.locals[c.Name+".v"+strconv.Itoa(int(c.Version))] = c.Subgraph
	}
	m.producer = make([]*mimfile.Op, len(g.Values))
	for si := range g.Subgraphs {
		for oi := range g.Subgraphs[si].Ops {
			op := &g.Subgraphs[si].Ops[oi]
			if err := m.check(op); err != nil {
				return err
			}
			for _, o := range op.Outputs {
				m.producer[o] = op
			}
		}
	}
	m.consts = map[uint32]*tensor.Tensor{}
	for i, v := range g.Values {
		if _, ok := storage(v.DType); !ok && !v.DType.IsBlock() {
			return &UnsupportedError{Op: "dtype", Detail: fmt.Sprintf("value %q of dtype %s", v.Name, v.DType)}
		}
		switch v.Kind {
		case mimfile.KindWeight, mimfile.KindConst, mimfile.KindVariable:
			if v.Zero {
				continue
			}
			t, err := m.weight(v)
			if err != nil {
				return fmt.Errorf("value %q: %w", v.Name, err)
			}
			m.consts[uint32(i)] = t
		}
	}
	main := g.Main()
	m.inputs = m.infos(main.Inputs)
	m.outputs = m.infos(main.Outputs)
	return nil
}

// infos describes the values vs.
func (m *Model) infos(vs []uint32) []ValueInfo {
	out := make([]ValueInfo, len(vs))
	for i, vi := range vs {
		v := m.g.Values[vi]
		info := ValueInfo{Name: valueName(v, vi), Type: v.DType, Shape: make([]Dim, len(v.Shape))}
		for j, d := range v.Shape {
			if !d.Symbolic() {
				info.Shape[j] = Dim{Size: int(d), Bound: int(d)}
				continue
			}
			s := m.g.SymDims[d.SymIndex()]
			info.Shape[j] = Dim{Size: -1, Name: s.Name, Bound: int(s.Bound)}
		}
		out[i] = info
	}
	return out
}

// valueName returns the name of value i, or its index for an anonymous
// value.
func valueName(v mimfile.Value, i uint32) string {
	if v.Name != "" {
		return v.Name
	}
	return "%" + strconv.Itoa(int(i))
}

// DefaultMaxValueBytes is the bound a model starts with on the bytes of
// each value a run computes (see Model.SetMaxValueBytes).
const DefaultMaxValueBytes = 4 << 30

// SetMaxValueBytes bounds the bytes of each value a run computes, and of
// each variable that starts as zeros, counted in the dtype the file
// declares: a run that would compute a larger one fails with
// tensor.ErrMemoryBudget before it allocates it. The bound keeps a broken
// or hostile file, which may declare an intermediate value of any size,
// from exhausting memory; it starts at DefaultMaxValueBytes. A negative n
// removes it. Inputs, weights and constants are not counted.
func (m *Model) SetMaxValueBytes(n int64) { m.maxBytes.Store(n) }

// checkSize fails with tensor.ErrMemoryBudget when a value of dtype dt and
// shape is over the model's bound on the bytes of a value.
func (m *Model) checkSize(v mimfile.Value, vi uint32, shape []int) error {
	limit := m.maxBytes.Load()
	if limit < 0 {
		return nil
	}
	dims := make([]uint64, len(shape))
	for i, d := range shape {
		if d < 0 {
			return nil // not bound yet; checked once computed
		}
		dims[i] = uint64(d)
	}
	n, err := mimfile.ByteSize(v.DType, dims)
	if err == nil && n <= uint64(limit) {
		return nil
	}
	return fmt.Errorf("mim: %s %q of shape %v takes more than the %d bytes a value may: %w", kindName(v.Kind), valueName(v, vi), shape, limit, tensor.ErrMemoryBudget)
}

// Inputs returns the inputs a run takes, in the order of the graph.
func (m *Model) Inputs() []ValueInfo { return slices.Clone(m.inputs) }

// Outputs returns the outputs a run computes, in the order of the graph.
func (m *Model) Outputs() []ValueInfo { return slices.Clone(m.outputs) }

// Metadata returns the model's metadata, such as its architecture and
// tokenizer, or nil if the file has none.
func (m *Model) Metadata() (mimfile.Metadata, error) {
	md, err := m.f.Metadata()
	if mimfile.CodeOf(err) == mimfile.CodeNotFound {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrFormat, err)
	}
	return md, nil
}

// Run computes the model's outputs from its inputs, in the session the
// model holds: variables keep the values that the previous call of Run
// left. See Session.Run. Every caller of Run shares that session, so
// calls take turns; to run at once, as a server's workers do, each uses a
// session of its own from NewSession.
func (m *Model) Run(ctx context.Context, inputs map[string]*tensor.Tensor) (map[string]*tensor.Tensor, error) {
	return m.session.Run(ctx, inputs)
}

// Close lets go of the mapping of the model's file, which is unmapped
// unless a tensor still reads it. A closed model does not run.
func (m *Model) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil
	}
	m.closed = true
	m.cleanup.Stop()
	m.consts, m.onDevice, m.recorded = nil, nil, nil
	if m.mapping == nil {
		return nil
	}
	return m.mapping.Close()
}

// constsOn returns the weights, constants and initial variable values of
// the model on d, copying them there the first time.
func (m *Model) constsOn(d tensor.Device) (map[uint32]*tensor.Tensor, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil, errors.New("mim: run of a closed model")
	}
	if d == tensor.CPU() {
		return m.consts, nil
	}
	if cs, ok := m.onDevice[d]; ok {
		return cs, nil
	}
	cs := make(map[uint32]*tensor.Tensor, len(m.consts))
	ts := make([]*tensor.Tensor, 0, len(m.consts))
	for i, t := range m.consts {
		cs[i] = t.To(d)
		ts = append(ts, cs[i])
	}
	if err := tensor.Realize(context.Background(), ts...); err != nil {
		return nil, err
	}
	m.onDevice[d] = cs
	return cs, nil
}

// hold returns a function that keeps the model's mapping until it is
// called, for a tensor that reads it in place.
func (m *Model) hold() func() {
	if m.mapping == nil {
		return func() {}
	}
	return m.mapping.Hold()
}
