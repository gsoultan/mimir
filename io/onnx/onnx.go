// Package onnx loads models in the ONNX format and runs them with mimir
// tensors, on any device. See https://onnx.ai/onnx/repo-docs/IR.html.
//
// Open reads a model, Run computes it:
//
//	m, err := onnx.Open("model.onnx")
//	...
//	defer m.Close()
//	for _, in := range m.Inputs() {
//		fmt.Println(in) // input: float[batch 3 224 224]
//	}
//	out, err := m.Run(ctx, map[string]*tensor.Tensor{"input": x})
//	logits := out["output"]
//
// Run turns the whole graph into lazy mimir operations and realizes the
// outputs together, so a model is one schedule: its elementwise chains
// fuse, its buffers are planned at once, and from the second run with
// inputs of the same shapes the schedule cache replays it without
// planning anything. The shapes, indices and scalars a graph computes,
// such as the chain Shape, Gather, Concat that feeds a Reshape, are
// computed on the host with the input shapes in hand, never on a device.
// Nodes that depend only on initializers are computed once, when the
// model is opened. Weights stay on the CPU, read in place from a mapping
// of the file where they are aligned, and a run on another device copies
// them there the first time.
//
// Opsets 11 to 22 of the default domain are read. Values of type int64
// and bool are held as Int32 tensors (see DataType.DType), so their
// values must fit in an int32 once they reach a device; on the host they
// keep 64 bits. An operator, attribute value or data type that mimir does
// not run is reported by Open, or at the latest by Run, as an
// *UnsupportedError naming it.
//
// The decoder bounds every length, count and dimension it reads by the
// size of the file, and external data files by theirs, so a corrupt or
// hostile file yields an error, never a panic or a read out of range.
package onnx

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"sync"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/internal/filemap"
	"github.com/gsoultan/mimir/tensor"
)

// Opsets of the default domain that Open accepts.
const (
	MinOpset = 11
	MaxOpset = 22
)

// Model is a loaded ONNX model. Its methods may be called from several
// goroutines; runs take turns.
type Model struct {
	mu       sync.Mutex
	opset    int64
	producer string
	inputs   []ValueInfo
	outputs  []ValueInfo
	consts   map[string]*value // initializers and folded nodes
	nodes    []*node           // in topological order
	cache    map[cacheKey]*cached
	external map[string][]byte // external data files, by location
	dir      string            // the model's directory, for external data
	mappings []*filemap.Mapping
	cleanup  runtime.Cleanup
	closed   bool
}

// node is a node of the graph with the function that computes it.
type node struct {
	*nodeProto
	fn opFunc
}

// Inputs returns the inputs a run takes, in the order of the graph.
// Initializers that a graph also lists as inputs are not among them.
func (m *Model) Inputs() []ValueInfo { return slices.Clone(m.inputs) }

// Outputs returns the outputs a run computes, in the order of the graph.
func (m *Model) Outputs() []ValueInfo { return slices.Clone(m.outputs) }

// Opset returns the version of the default operator set the model uses.
func (m *Model) Opset() int { return int(m.opset) }

// Producer returns the name of the tool that wrote the model, such as
// "pytorch".
func (m *Model) Producer() string { return m.producer }

// Open reads the model at path. It maps the file, and any external data
// file the model names, which must lie in the model's directory or below
// it. Weights read the mappings in place where they are aligned, until
// Close, and past it for as long as a tensor reads them.
func Open(path string) (*Model, error) {
	data, mp, err := filemap.Read(path)
	if errors.Is(err, filemap.ErrTooLarge) {
		return nil, fmt.Errorf("onnx: %s: %w: %w", path, err, ErrFormat)
	}
	if err != nil {
		return nil, fmt.Errorf("onnx: %w", err)
	}
	m, err := parse(data, mp, filepath.Dir(path))
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return m, nil
}

// Parse reads a model from the bytes of a file. Its weights may alias b,
// which must stay unchanged while the model is in use. A model with
// external data needs Open, which knows where its files are.
func Parse(b []byte) (*Model, error) { return parse(b, nil, "") }

func parse(b []byte, mp *filemap.Mapping, dir string) (*Model, error) {
	m := &Model{consts: map[string]*value{}, cache: map[cacheKey]*cached{}, external: map[string][]byte{}}
	if mp != nil {
		m.mappings = append(m.mappings, mp)
	}
	m.dir = dir
	if err := m.load(b, dir); err != nil {
		m.release()
		return nil, err
	}
	m.cleanup = runtime.AddCleanup(m, func(ms []*filemap.Mapping) {
		for _, mp := range ms {
			_ = mp.Close()
		}
	}, m.mappings)
	return m, nil
}

func (m *Model) load(b []byte, dir string) error {
	p, err := parseModel(b)
	if err != nil {
		return err
	}
	m.producer = p.producer
	for _, o := range p.opsets {
		switch o.domain {
		case "", "ai.onnx":
			m.opset = o.version
		}
	}
	if m.opset < MinOpset || m.opset > MaxOpset {
		return &UnsupportedError{Op: "opset", Detail: fmt.Sprintf("default opset %d, want %d to %d", m.opset, MinOpset, MaxOpset)}
	}
	g := p.graph
	if g.sparseInits > 0 {
		return &UnsupportedError{Op: "sparse_initializer"}
	}
	for _, t := range g.inits {
		if t.name == "" {
			return fmt.Errorf("%w: initializer without a name", ErrFormat)
		}
		if _, dup := m.consts[t.name]; dup {
			return fmt.Errorf("%w: initializer %q repeated", ErrFormat, t.name)
		}
		v, err := m.constant(t, dir, true)
		if err != nil {
			return fmt.Errorf("initializer %q: %w", t.name, err)
		}
		v.key = t.name
		m.consts[t.name] = v
	}
	defined := map[string]bool{}
	for _, in := range g.inputs {
		if _, isInit := m.consts[in.Name]; isInit {
			continue
		}
		if in.Name == "" || defined[in.Name] {
			return fmt.Errorf("%w: input %q", ErrFormat, in.Name)
		}
		if _, ok := in.Type.DType(); !ok {
			return &UnsupportedError{Op: "input", Detail: fmt.Sprintf("input %q of type %s", in.Name, in.Type)}
		}
		defined[in.Name] = true
		m.inputs = append(m.inputs, in)
	}
	if len(g.outputs) == 0 {
		return fmt.Errorf("%w: the graph has no outputs", ErrFormat)
	}
	m.outputs = g.outputs
	nodes, err := m.sort(g.nodes, defined)
	if err != nil {
		return err
	}
	for _, out := range m.outputs {
		if _, isConst := m.consts[out.Name]; !defined[out.Name] && !isConst {
			return fmt.Errorf("%w: output %q is not computed", ErrFormat, out.Name)
		}
	}
	m.nodes = nodes
	limit := len(b)
	for _, data := range m.external {
		limit += len(data)
	}
	if err := m.foldConstants(limit); err != nil {
		return err
	}
	m.fuseAttention()
	return nil
}

// sort returns the nodes in an order in which each follows the nodes
// that compute its inputs, with their functions, and adds their outputs
// to defined.
func (m *Model) sort(nodes []*nodeProto, defined map[string]bool) ([]*node, error) {
	producer := map[string]int{}
	for i, n := range nodes {
		if len(n.outputs) == 0 {
			return nil, fmt.Errorf("%w: node %q has no outputs", ErrFormat, n.name)
		}
		for _, out := range n.outputs {
			if out == "" {
				continue
			}
			_, isConst := m.consts[out]
			if _, dup := producer[out]; dup || defined[out] || isConst {
				return nil, fmt.Errorf("%w: value %q is defined twice", ErrFormat, out)
			}
			producer[out] = i
		}
	}
	// Kahn's algorithm, which keeps the file's order among ready nodes.
	waiting := make([]int, len(nodes))
	readers := make([][]int, len(nodes))
	for i, n := range nodes {
		for _, in := range n.inputs {
			if in == "" {
				continue
			}
			if p, ok := producer[in]; ok {
				waiting[i]++
				readers[p] = append(readers[p], i)
				continue
			}
			if _, isConst := m.consts[in]; !isConst && !defined[in] {
				return nil, fmt.Errorf("%w: node %q reads %q, which nothing defines", ErrFormat, n.name, in)
			}
		}
	}
	var ready []int
	for i := range nodes {
		if waiting[i] == 0 {
			ready = append(ready, i)
		}
	}
	out := make([]*node, 0, len(nodes))
	for len(ready) > 0 {
		i := ready[0]
		ready = ready[1:]
		n := nodes[i]
		fn, err := lookup(n, m.opset)
		if err != nil {
			return nil, err
		}
		out = append(out, &node{nodeProto: n, fn: fn})
		for _, o := range n.outputs {
			if o != "" {
				defined[o] = true
			}
		}
		for _, j := range readers[i] {
			if waiting[j]--; waiting[j] == 0 {
				ready = append(ready, j)
			}
		}
	}
	if len(out) != len(nodes) {
		return nil, fmt.Errorf("%w: the graph has a cycle", ErrFormat)
	}
	return out, nil
}

// lookup returns the function of n's operator.
func lookup(n *nodeProto, opset int64) (opFunc, error) {
	if n.domain != "" && n.domain != "ai.onnx" {
		return nil, &UnsupportedError{Node: n.name, Domain: n.domain, Op: n.op}
	}
	fn, ok := ops[n.op]
	if !ok {
		return nil, &UnsupportedError{Node: n.name, Op: n.op}
	}
	for _, a := range n.attrs {
		if a.ref != "" {
			return nil, &UnsupportedError{Node: n.name, Op: n.op, Detail: "attribute references"}
		}
	}
	return fn, nil
}

// foldConstants computes, once, the nodes whose inputs are all constants,
// and keeps their outputs as constants: tensors realized on the CPU, or
// host values. A tensor of more elements than the model's files have
// bytes, which no transposition or cast of weights makes, stays a node
// to compute in each run, within the memory budget of its device.
func (m *Model) foldConstants(limit int) error {
	r := &run{ctx: context.Background(), m: m, dev: tensor.CPU(), env: map[string]*value{}, fold: true}
	kept := m.nodes[:0]
	for _, n := range m.nodes {
		if !m.foldable(n) {
			kept = append(kept, n)
			continue
		}
		outs, err := r.eval(n)
		if err != nil {
			return err
		}
		if slices.ContainsFunc(outs, func(v *value) bool { return !v.host && v.size() > limit }) {
			kept = append(kept, n)
			continue
		}
		var realize []*tensor.Tensor
		for i, o := range n.outputs {
			if o == "" || i >= len(outs) {
				continue
			}
			v := outs[i]
			v.key, v.constant = o, true
			m.consts[o] = v
			if v.t != nil && !v.host {
				realize = append(realize, v.t)
			}
		}
		if err := tensor.Realize(r.ctx, realize...); err != nil {
			return fmt.Errorf("onnx: node %q (%s): %w", n.name, n.op, err)
		}
	}
	m.nodes = kept
	return nil
}

// foldable reports whether every input of n is a constant, and n is
// deterministic.
func (m *Model) foldable(n *node) bool {
	switch n.op {
	case "RandomNormal", "RandomUniform", "RandomNormalLike", "RandomUniformLike", "Multinomial", "Bernoulli":
		return false
	}
	for _, in := range n.inputs {
		if _, ok := m.consts[in]; in != "" && !ok {
			return false
		}
	}
	return true
}

// eval computes node n.
func (r *run) eval(n *node) ([]*value, error) {
	c := &call{run: r, n: n, in: make([]*value, len(n.inputs))}
	for i, name := range n.inputs {
		v, err := r.get(name)
		if err != nil {
			return nil, err
		}
		c.in[i] = v
	}
	outs, err := n.fn(c)
	if err == nil {
		err = c.err
	}
	if err == nil {
		for _, v := range outs {
			if v == nil {
				err = fmt.Errorf("onnx: an output is missing: %w", tensor.ErrInternal)
				break
			}
			if v.t != nil && !v.host {
				if err = v.t.Err(); err != nil {
					break
				}
			}
		}
	}
	if err != nil {
		if u := (*UnsupportedError)(nil); errors.As(err, &u) {
			return nil, err
		}
		return nil, fmt.Errorf("onnx: node %q (%s): %w", n.name, n.op, err)
	}
	for i, o := range n.outputs {
		if o != "" && i < len(outs) && outs[i].key == "" && outs[i].host {
			outs[i].key = o
		}
	}
	return outs, nil
}

// Run computes the model's outputs from its inputs, keyed by name. Every
// input must be given, with the dtype that its type maps to (see
// DataType.DType) and a shape that matches the declared one, whose
// symbolic dimensions take any size. The inputs must share a device, on
// which the model runs and the outputs lie; with no inputs it runs on the
// default device. Outputs of type int64 or bool are Int32 tensors.
func (m *Model) Run(ctx context.Context, inputs map[string]*tensor.Tensor) (map[string]*tensor.Tensor, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil, errors.New("onnx: run of a closed model")
	}
	r := &run{ctx: ctx, m: m, dev: tensor.DefaultDevice(), env: make(map[string]*value, len(m.inputs)+len(m.nodes))}
	for i, in := range m.inputs {
		t, ok := inputs[in.Name]
		if !ok || t == nil {
			return nil, fmt.Errorf("onnx: input %q is missing: %w", in.Name, tensor.ErrShape)
		}
		if err := t.Err(); err != nil {
			return nil, fmt.Errorf("onnx: input %q: %w", in.Name, err)
		}
		if err := check(in, t); err != nil {
			return nil, err
		}
		if i == 0 {
			r.dev = t.Device()
		} else if t.Device() != r.dev {
			return nil, fmt.Errorf("onnx: input %q on %v, others on %v: %w", in.Name, t.Device(), r.dev, tensor.ErrDevice)
		}
		r.env[in.Name] = devValue(in.Type, t)
	}
	if len(inputs) > len(m.inputs) {
		for name := range inputs {
			if !slices.ContainsFunc(m.inputs, func(v ValueInfo) bool { return v.Name == name }) {
				return nil, fmt.Errorf("onnx: the model has no input %q: %w", name, tensor.ErrShape)
			}
		}
	}
	for _, n := range m.nodes {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		outs, err := r.eval(n)
		if err != nil {
			return nil, err
		}
		for i, o := range n.outputs {
			if o != "" && i < len(outs) {
				r.env[o] = outs[i]
			}
		}
	}
	out := make(map[string]*tensor.Tensor, len(m.outputs))
	ts := make([]*tensor.Tensor, 0, len(m.outputs))
	for _, o := range m.outputs {
		v, err := r.get(o.Name)
		if err != nil {
			return nil, err
		}
		t, err := r.tensor(v)
		if err != nil {
			return nil, fmt.Errorf("onnx: output %q: %w", o.Name, err)
		}
		if t.Device() != r.dev {
			t = t.To(r.dev)
		}
		out[o.Name] = t
		ts = append(ts, t)
	}
	if err := tensor.Realize(ctx, ts...); err != nil {
		return nil, err
	}
	return out, nil
}

// check reports whether t fits input in.
func check(in ValueInfo, t *tensor.Tensor) error {
	if dt := in.DType(); t.DType() != dt {
		return fmt.Errorf("onnx: input %q of %s is a tensor of %s, want %s: %w", in.Name, in.Type, t.DType(), dt, tensor.ErrDType)
	}
	if !in.Ranked {
		return nil
	}
	shape := t.Shape()
	ok := len(shape) == len(in.Shape)
	for i := 0; ok && i < len(shape); i++ {
		ok = in.Shape[i].Size < 0 || in.Shape[i].Size == shape[i]
	}
	if !ok {
		return fmt.Errorf("onnx: input %q of shape %v, want %v: %w", in.Name, shape, in.Shape, tensor.ErrShape)
	}
	return nil
}

// Close lets go of the mappings of the model's files, which are unmapped
// unless a tensor still reads them. A closed model does not run.
func (m *Model) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil
	}
	m.closed = true
	m.cleanup.Stop()
	m.consts, m.nodes, m.cache = nil, nil, nil
	return m.release()
}

func (m *Model) release() error {
	var errs []error
	for _, mp := range m.mappings {
		errs = append(errs, mp.Close())
	}
	m.mappings = nil
	return errors.Join(errs...)
}

// hold returns a function that keeps the model's mappings until it is
// called, for a tensor that reads them in place.
func (m *Model) hold() func() {
	releases := make([]func(), len(m.mappings))
	for i, mp := range m.mappings {
		releases[i] = mp.Hold()
	}
	return func() {
		for _, r := range releases {
			r()
		}
	}
}

// constant returns the value of an initializer or a constant attribute.
// With weights true, a float tensor of more than hostLimit elements
// becomes a tensor on the CPU that reads the file in place where it can.
func (m *Model) constant(t *tensorProto, dir string, weights bool) (*value, error) {
	if t.segment {
		return nil, &UnsupportedError{Op: "TensorProto", Detail: "segments"}
	}
	shape := make([]int, len(t.dims))
	for i, d := range t.dims {
		if d < 0 || d > math.MaxInt32 {
			return nil, fmt.Errorf("%w: dimensions %v", ErrFormat, t.dims)
		}
		shape[i] = int(d)
	}
	typ := t.dataType
	size := typ.size()
	if _, ok := typ.DType(); !ok || size == 0 {
		return nil, &UnsupportedError{Op: "TensorProto", Detail: fmt.Sprintf("data type %s", typ)}
	}
	raw, hasRaw := t.raw, t.hasRaw
	if t.location == 1 {
		var err error
		if raw, err = m.externalData(t, dir); err != nil {
			return nil, err
		}
		hasRaw = true
	}
	// Every element takes at least a byte of the file, but an empty raw
	// field of a tensor with no elements is all there is.
	limit := max(len(raw), len(t.floats), len(t.int32s), len(t.int64s), len(t.doubles), len(t.uint64s))
	n, ok := numElements(shape, limit)
	if !ok {
		return nil, fmt.Errorf("%w: dimensions %v for %d bytes of data", ErrFormat, t.dims, limit)
	}
	if hasRaw && len(raw) != n*size {
		return nil, fmt.Errorf("%w: %d bytes of %s for dimensions %v", ErrFormat, len(raw), typ, t.dims)
	}
	if hasRaw && weights && n > hostLimit && typ.IsFloat() && typ != Double {
		dt, _ := typ.DType()
		x := tensor.FromBytesFunc(dt, raw, m.hold(), shape...)
		return &value{typ: typ, shape: shape, t: x, constant: true}, x.Err()
	}
	if !hasRaw && weights && n > hostLimit && typ == Float && len(t.floats) == n {
		b := make([]byte, 4*n)
		for i, f := range t.floats {
			binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(f))
		}
		x := tensor.FromBytes(tensor.Float32, b, shape...)
		return &value{typ: typ, shape: shape, t: x, constant: true}, x.Err()
	}
	var count int
	switch {
	case hasRaw:
		count = n
	case typ == Float:
		count = len(t.floats)
	case typ == Double:
		count = len(t.doubles)
	case typ == Int64:
		count = len(t.int64s)
	case typ == Uint32 || typ == Uint64:
		count = len(t.uint64s)
	default:
		count = len(t.int32s)
	}
	if count != n {
		return nil, fmt.Errorf("%w: %d elements of %s for dimensions %v", ErrFormat, count, typ, t.dims)
	}
	v := &value{typ: typ, shape: shape, host: true, constant: true}
	if typ.IsFloat() {
		v.floats = make([]float64, n)
		for i := range v.floats {
			v.floats[i] = t.floatAt(i, raw, hasRaw)
		}
	} else {
		v.ints = make([]int64, n)
		for i := range v.ints {
			v.ints[i] = t.intAt(i, raw, hasRaw)
		}
	}
	if weights && n > hostLimit {
		// A large integer tensor is a tensor too; its values must fit
		// in an int32.
		dt, _ := typ.DType()
		x, err := upload(tensor.CPU(), dt, v)
		if err != nil {
			return nil, err
		}
		return &value{typ: typ, shape: shape, t: x, constant: true}, nil
	}
	return v, nil
}

// floatAt returns element i of a float tensor.
func (t *tensorProto) floatAt(i int, raw []byte, hasRaw bool) float64 {
	switch t.dataType {
	case Float:
		if hasRaw {
			return float64(math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:])))
		}
		return float64(t.floats[i])
	case Double:
		if hasRaw {
			return math.Float64frombits(binary.LittleEndian.Uint64(raw[8*i:]))
		}
		return t.doubles[i]
	}
	var bits uint16
	if hasRaw {
		bits = binary.LittleEndian.Uint16(raw[2*i:])
	} else {
		bits = uint16(t.int32s[i])
	}
	if t.dataType == BFloat16 {
		return float64(math.Float32frombits(uint32(bits) << 16))
	}
	return halfToFloat(bits)
}

// intAt returns element i of an integer or bool tensor.
func (t *tensorProto) intAt(i int, raw []byte, hasRaw bool) int64 {
	if !hasRaw {
		switch t.dataType {
		case Int64:
			return t.int64s[i]
		case Uint32:
			return int64(uint32(t.uint64s[i]))
		case Uint64:
			return t.uint64s[i]
		case Uint8, Bool:
			return int64(uint8(t.int32s[i]))
		case Int8:
			return int64(int8(t.int32s[i]))
		case Uint16:
			return int64(uint16(t.int32s[i]))
		case Int16:
			return int64(int16(t.int32s[i]))
		}
		return int64(int32(t.int32s[i]))
	}
	switch t.dataType {
	case Uint8, Bool:
		return int64(raw[i])
	case Int8:
		return int64(int8(raw[i]))
	case Uint16:
		return int64(binary.LittleEndian.Uint16(raw[2*i:]))
	case Int16:
		return int64(int16(binary.LittleEndian.Uint16(raw[2*i:])))
	case Int32:
		return int64(int32(binary.LittleEndian.Uint32(raw[4*i:])))
	case Uint32:
		return int64(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	return int64(binary.LittleEndian.Uint64(raw[8*i:]))
}

// halfToFloat converts IEEE half-precision bits to a float64.
func halfToFloat(h uint16) float64 {
	sign := 1.0
	if h&0x8000 != 0 {
		sign = -1
	}
	exp, frac := int(h>>10&0x1f), float64(h&0x3ff)
	switch exp {
	case 0:
		return sign * math.Ldexp(frac, -24)
	case 0x1f:
		if frac != 0 {
			return math.NaN()
		}
		return math.Inf(int(sign))
	}
	return sign * math.Ldexp(1+frac/1024, exp-15)
}

// externalData returns the bytes of a tensor stored in a file next to the
// model.
func (m *Model) externalData(t *tensorProto, dir string) ([]byte, error) {
	loc := t.external["location"]
	if dir == "" {
		return nil, fmt.Errorf("onnx: tensor %q has external data, which needs Open: %w", t.name, ErrFormat)
	}
	if loc == "" || !filepath.IsLocal(loc) {
		return nil, fmt.Errorf("%w: tensor %q in external file %q outside the model's directory", ErrFormat, t.name, loc)
	}
	data, ok := m.external[loc]
	if !ok {
		var mp *filemap.Mapping
		var err error
		data, mp, err = filemap.Read(filepath.Join(dir, loc))
		if err != nil {
			return nil, fmt.Errorf("onnx: tensor %q: %w", t.name, err)
		}
		if mp != nil {
			m.mappings = append(m.mappings, mp)
		}
		m.external[loc] = data
	}
	field := func(key string, def int) (int, error) {
		s, ok := t.external[key]
		if !ok {
			return def, nil
		}
		n, err := strconv.ParseInt(s, 10, 64)
		if err != nil || n < 0 || n > int64(len(data)) {
			return 0, fmt.Errorf("%w: tensor %q: external %s %q in a file of %d bytes", ErrFormat, t.name, key, s, len(data))
		}
		return int(n), nil
	}
	off, err := field("offset", 0)
	if err != nil {
		return nil, err
	}
	n, err := field("length", len(data)-off)
	if err != nil {
		return nil, err
	}
	if n > len(data)-off {
		return nil, fmt.Errorf("%w: tensor %q: %d bytes at offset %d of a file of %d bytes", ErrFormat, t.name, n, off, len(data))
	}
	return data[off : off+n : off+n], nil
}

// dtypeOf returns the mimir dtype of typ, or an *UnsupportedError.
func dtypeOf(typ DataType) (backend.DType, error) {
	dt, ok := typ.DType()
	if !ok {
		return 0, &UnsupportedError{Op: "TensorProto", Detail: fmt.Sprintf("data type %s", typ)}
	}
	return dt, nil
}
