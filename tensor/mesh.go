package tensor

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/internal/graph"
	"github.com/gsoultan/mimir/internal/sched"
)

// Mesh is an n-dimensional arrangement of devices with named axes, like
// jax.sharding.Mesh: NewMesh([]int{2, 4}, []string{"data", "model"},
// devs...) arranges eight devices in two rows of four, and a tensor can
// be split along its first axis over the rows, "data", and along its last
// over the columns, "model", or kept whole on every device.
//
// A sharded tensor is one logical tensor whose shards live on the devices
// of a mesh, as a PartitionSpec says: Shard places a tensor there, To
// brings it back to one device, and Sharding reports how it is split.
// Operations on sharded tensors build one logical graph, with gradients
// as on one device; when it is computed, every device computes its shard
// of each operation from its shards of the inputs, and the collectives
// that this needs are inserted between them (see Shard for the rules).
// Combining a sharded tensor with a tensor on a device fails with
// ErrDevice, as tensors on two devices do: shard the other one too, kept
// whole with an empty PartitionSpec.
//
// The devices are those of CPUs or GPUs, of one kind that has
// collectives (see NewGroup). A Mesh is safe for concurrent use; Close
// releases its collectives.
type Mesh struct {
	shape  []int
	axes   []string
	stride []int // the distance between ranks along each axis
	devs   []Device
	bes    []backend.Backend
	dev    *meshDevice

	mu      sync.Mutex
	groups  map[axisSet][]meshGroup
	closed  bool
	traffic Traffic

	plans planCache // the analyses of recent graphs, guarded by meshMu
}

// Traffic counts the collectives that computing sharded tensors on a mesh
// has run, by kind, and the bytes of their inputs, summed over the
// devices.
type Traffic struct {
	AllReduces, ReduceScatters, AllGathers int
	Bytes                                  int64
}

// Traffic returns the collectives run on m so far.
func (m *Mesh) Traffic() Traffic {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.traffic
}

// count adds a collective of kind over inputs of n elements of dt on
// each of ranks devices to m's traffic.
func (m *Mesh) count(kind backend.CollectiveKind, dt DType, n, ranks int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	switch kind {
	case backend.AllReduce:
		m.traffic.AllReduces++
	case backend.ReduceScatter:
		m.traffic.ReduceScatters++
	case backend.AllGather:
		m.traffic.AllGathers++
	}
	b, _ := dt.Bytes(n)
	m.traffic.Bytes += int64(b) * int64(ranks)
}

// maxMeshAxes is the most axes a mesh has, as many as an axisSet holds.
const maxMeshAxes = 8

// meshGroup is a group of the ranks of a mesh that a collective over some
// of its axes connects: those that differ along the axes alone, in
// row-major order over them.
type meshGroup struct {
	ranks []int
	comm  backend.Communicator
}

// NewMesh arranges devs in a mesh of the given shape, in row-major order,
// with axes naming its dimensions: device i sits at the coordinates of
// position i in a row-major array of that shape. There are as many
// devices as the shape has positions, all distinct, and the axis names
// are distinct and not empty. A mesh of more than one device needs
// devices with collectives, as Group does.
func NewMesh(shape []int, axes []string, devs ...Device) (*Mesh, error) {
	n, ok := backend.NumElements(shape)
	switch {
	case len(shape) != len(axes) || len(shape) == 0 || len(shape) > maxMeshAxes:
		return nil, fmt.Errorf("tensor: mesh of shape %v with axes %q: %w", shape, axes, ErrShape)
	case !ok || n != len(devs) || slices.Contains(shape, 0):
		return nil, fmt.Errorf("tensor: mesh of shape %v over %d devices: %w", shape, len(devs), ErrShape)
	}
	for i, a := range axes {
		if a == "" || slices.Contains(axes[:i], a) {
			return nil, fmt.Errorf("tensor: mesh axes %q: names must be distinct and not empty: %w", axes, ErrShape)
		}
	}
	m := &Mesh{shape: slices.Clone(shape), axes: slices.Clone(axes), stride: make([]int, len(shape)), devs: slices.Clone(devs), bes: make([]backend.Backend, len(devs))}
	s := 1
	for a := len(shape) - 1; a >= 0; a-- {
		m.stride[a], s = s, s*shape[a]
	}
	for i, d := range devs {
		be := d.backend()
		if _, isMesh := be.(*meshDevice); isMesh {
			return nil, fmt.Errorf("tensor: mesh of meshes: %w", ErrDevice)
		}
		if be.Name() != devs[0].backend().Name() {
			return nil, fmt.Errorf("tensor: mesh of %s and %s devices: %w", devs[0].backend().Name(), be.Name(), ErrDevice)
		}
		if slices.Contains(m.bes[:i], be) {
			return nil, fmt.Errorf("tensor: device %d of a mesh appears twice: %w", i, ErrDevice)
		}
		m.bes[i] = be
	}
	if _, ok := m.bes[0].(backend.Grouper); !ok && len(devs) > 1 {
		return nil, fmt.Errorf("tensor: %v devices have no collectives: %w", devs[0], ErrDevice)
	}
	m.dev = &meshDevice{mesh: m}
	return m, nil
}

// Shape returns the number of devices along each axis.
func (m *Mesh) Shape() []int { return slices.Clone(m.shape) }

// Axes returns the names of the axes.
func (m *Mesh) Axes() []string { return slices.Clone(m.axes) }

// Devices returns the devices in row-major order.
func (m *Mesh) Devices() []Device { return slices.Clone(m.devs) }

// Size returns the number of devices along the named axis, or 0 if the
// mesh has no such axis.
func (m *Mesh) Size(axis string) int {
	if a := slices.Index(m.axes, axis); a >= 0 {
		return m.shape[a]
	}
	return 0
}

// Device returns the mesh as a device: the device of its sharded tensors,
// whose name is "mesh". To on it shards a tensor whole on every device.
func (m *Mesh) Device() Device { return Device{m.dev} }

func (m *Mesh) String() string {
	parts := make([]string, len(m.axes))
	for i, a := range m.axes {
		parts[i] = fmt.Sprintf("%s=%d", a, m.shape[i])
	}
	return fmt.Sprintf("Mesh(%s; %v)", strings.Join(parts, ", "), m.devs[0])
}

// Close releases the collectives the mesh has opened between its devices
// once those queued so far have run. Computing a sharded tensor that
// needs collectives afterwards fails with ErrDevice.
func (m *Mesh) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.closed = true
	var errs []error
	for _, gs := range m.groups {
		for _, g := range gs {
			if g.comm != nil {
				errs = append(errs, g.comm.Close())
			}
		}
	}
	m.groups = nil
	return errors.Join(errs...)
}

// coord returns the coordinate of rank r along axis a.
func (m *Mesh) coord(r, a int) int { return r / m.stride[a] % m.shape[a] }

// span returns the number of ranks in a group over the axes of s.
func (m *Mesh) span(s axisSet) int {
	n := 1
	for a := range m.shape {
		if s.has(a) {
			n *= m.shape[a]
		}
	}
	return n
}

// meshGroups returns the groups of ranks that collectives over the axes of
// s connect, opening their communicators the first time.
func (m *Mesh) meshGroups(s axisSet) ([]meshGroup, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil, fmt.Errorf("tensor: collective on a closed mesh: %w", ErrDevice)
	}
	if gs, ok := m.groups[s]; ok {
		return gs, nil
	}
	byRest := map[int]int{} // the index in gs of the group of each rank's coordinates outside s
	var gs []meshGroup
	for r := range m.devs {
		rest := 0
		for a := range m.shape {
			if !s.has(a) {
				rest = rest*m.shape[a] + m.coord(r, a)
			}
		}
		i, ok := byRest[rest]
		if !ok {
			i = len(gs)
			byRest[rest] = i
			gs = append(gs, meshGroup{})
		}
		gs[i].ranks = append(gs[i].ranks, r)
	}
	for i := range gs {
		if len(gs[i].ranks) == 1 {
			continue
		}
		bes := make([]backend.Backend, len(gs[i].ranks))
		for j, r := range gs[i].ranks {
			bes[j] = m.bes[r]
		}
		comm, err := m.bes[0].(backend.Grouper).Group(bes)
		if err != nil {
			for _, g := range gs[:i] {
				if g.comm != nil {
					err = errors.Join(err, g.comm.Close())
				}
			}
			return nil, fmt.Errorf("tensor: mesh group: %w", err)
		}
		gs[i].comm = comm
	}
	if m.groups == nil {
		m.groups = map[axisSet][]meshGroup{}
	}
	m.groups[s] = gs
	return gs, nil
}

// Split says how one axis of a tensor is cut between the devices of a
// mesh: into as many equal blocks as the mesh axis named Axis has
// devices, block i on the devices at coordinate i along it, or not at
// all when Axis is "". Sections, if not empty, first cut the tensor's
// axis into consecutive sections of these lengths and then each section
// into blocks, so that a device holds its block of every section: the
// layout of a fused projection, such as the queries, keys and values of
// nn.SelfAttention, whose parts are split apart afterwards.
type Split struct {
	Axis     string
	Sections []int
}

// PartitionSpec says how each axis of a tensor is split over a mesh, like
// jax.sharding.PartitionSpec: element i for axis i of the tensor, and
// axes past its end are not split. A mesh axis splits at most one axis of
// a tensor, and the length of an axis it splits, or of each of its
// sections, must divide evenly by the mesh axis's size. An empty spec
// keeps the tensor whole on every device: replicated.
type PartitionSpec []Split

// P returns the spec that splits axis i of a tensor over the mesh axis
// axes[i], or not at all for "", like jax.sharding.PartitionSpec:
// P("data") splits the first axis over "data", and P("", "model") the
// second over "model".
func P(axes ...string) PartitionSpec {
	s := make(PartitionSpec, len(axes))
	for i, a := range axes {
		s[i].Axis = a
	}
	return s
}

// WithSections returns a copy of s whose axis dim is cut into sections of
// the given lengths before it is split (see Split).
func (s PartitionSpec) WithSections(dim int, sizes ...int) PartitionSpec {
	out := slices.Clone(s)
	if dim >= 0 {
		for len(out) <= dim {
			out = append(out, Split{})
		}
		out[dim] = Split{Axis: out[dim].Axis, Sections: slices.Clone(sizes)}
	}
	return out
}

func (s PartitionSpec) String() string {
	parts := make([]string, len(s))
	for i, d := range s {
		switch {
		case d.Axis == "":
			parts[i] = "None"
		case len(d.Sections) > 0:
			parts[i] = fmt.Sprintf("%q%v", d.Axis, d.Sections)
		default:
			parts[i] = fmt.Sprintf("%q", d.Axis)
		}
	}
	return "P(" + strings.Join(parts, ", ") + ")"
}

// Sharding is how a tensor is laid out over a mesh, like
// jax.sharding.NamedSharding.
type Sharding struct {
	Mesh *Mesh
	Spec PartitionSpec
}

func (s Sharding) String() string { return fmt.Sprintf("%v on %v", s.Spec, s.Mesh) }

// layoutOf returns the layout of spec for a tensor of shape and dtype dt.
func (m *Mesh) layoutOf(spec PartitionSpec, shape []int, dt DType) (layout, error) {
	bad := func(format string, args ...any) (layout, error) {
		return layout{}, fmt.Errorf("tensor: %v for %s%v over %v: %s: %w", spec, dt, shape, m, fmt.Sprintf(format, args...), ErrShape)
	}
	if len(spec) > len(shape) {
		return bad("more axes than the tensor")
	}
	l := replicated(len(shape))
	var used axisSet
	for d, s := range spec {
		if s.Axis == "" {
			if len(s.Sections) > 0 {
				return bad("sections of an axis that is not split")
			}
			continue
		}
		a := slices.Index(m.axes, s.Axis)
		if a < 0 {
			return bad("no mesh axis %q", s.Axis)
		}
		if used.has(a) {
			return bad("mesh axis %q splits two axes", s.Axis)
		}
		used |= 1 << a
		k := m.shape[a]
		if len(s.Sections) > 0 {
			sum := 0
			for _, n := range s.Sections {
				if n <= 0 || n%k != 0 {
					return bad("section of %d over %d devices", n, k)
				}
				sum += n
			}
			if sum != shape[d] {
				return bad("sections %v of an axis of %d", s.Sections, shape[d])
			}
		}
		if shape[d]%k != 0 {
			return bad("axis of %d over %d devices", shape[d], k)
		}
		if k == 1 {
			continue // a mesh axis of one device splits nothing
		}
		l.dims[d] = dimLayout{axis: int8(a), sec: slices.Clone(s.Sections)}
		if len(s.Sections) == 1 {
			l.dims[d].sec = nil
		}
	}
	if dt.IsQuantized() && used != 0 {
		return bad("block-quantized tensors are only replicated")
	}
	return l, nil
}

// spec returns the PartitionSpec of the plain layout l.
func (m *Mesh) spec(l layout) PartitionSpec {
	s := make(PartitionSpec, len(l.dims))
	last := 0
	for d, dl := range l.dims {
		if dl.split() {
			s[d] = Split{Axis: m.axes[dl.axis], Sections: slices.Clone(dl.sec)}
			last = d + 1
		}
	}
	return s[:last]
}

// meshDevice is the device of a mesh's sharded tensors: their graphs'
// nodes name it as their device, so that combining one with a tensor of
// another device fails, and computing one comes to the mesh. It runs
// nothing itself.
type meshDevice struct{ mesh *Mesh }

var _ backend.Backend = (*meshDevice)(nil)

func (d *meshDevice) Name() string { return "mesh" }

func (d *meshDevice) unsupported(what string) error {
	return fmt.Errorf("tensor: %s on %v directly: %w", what, d.mesh, ErrDevice)
}

func (d *meshDevice) Alloc(DType, int) (backend.Buffer, error) {
	return nil, d.unsupported("allocate")
}
func (d *meshDevice) Free(backend.Buffer) {}
func (d *meshDevice) Upload(context.Context, backend.Buffer, []byte) error {
	return d.unsupported("upload")
}
func (d *meshDevice) Download(context.Context, []byte, backend.Buffer) error {
	return d.unsupported("download")
}
func (d *meshDevice) Run(context.Context, *backend.Kernel, []backend.Buffer, []backend.Buffer) error {
	return d.unsupported("run a kernel")
}
func (d *meshDevice) Sync(context.Context) error { return nil }

// meshBuffer is the memory of a computed sharded tensor: its shards, one
// realized node per device of the mesh, laid out as lay says, which is
// never partial nor packed.
type meshBuffer struct {
	dt     DType
	n      int
	shards []*graph.Node
	lay    layout
}

func (b *meshBuffer) DType() DType { return b.dt }
func (b *meshBuffer) Len() int     { return b.n }

// meshOf returns the mesh whose sharded tensor n is, or nil.
func meshOf(n *graph.Node) *Mesh {
	if d, ok := sched.Device(n).(*meshDevice); ok {
		return d.mesh
	}
	return nil
}

// meshMu serialises the computation of sharded tensors: their graphs'
// nodes, unlike those of a device, are realized outside the scheduler.
var meshMu sync.Mutex

// Shard places t on the devices of m as spec says, like jax.device_put
// with a NamedSharding: each device holds its block of t. It computes t
// now. Gradients flow back to t, and a t that is sharded already, on m
// or another mesh, is resharded.
//
// Operations on sharded tensors follow a small set of rules, which keep
// every device computing its own block wherever the layouts of the
// operands allow it:
//   - Elementwise operations on equally split operands compute locally,
//     and an operand kept whole is cut to the others' blocks for free.
//   - A product whose contracted axis is split computes partial sums,
//     which an all-reduce adds up, or a reduce-scatter when the result is
//     to stay split; split rows of one operand or columns of the other
//     split the result. A reduction over a split axis reduces locally,
//     then across the devices.
//   - Movement operations carry the split along: a reshape keeps an axis
//     split when it leads its group of merged or divided axes, a
//     transpose moves it, and slicing a section of an axis split by
//     sections (see Split) keeps it split.
//   - Attention splits over its batch and heads.
//   - Where operands disagree, one is resharded: an all-gather makes a
//     split axis whole and slicing splits a whole one. In a product the
//     operand split along its rows (or batch) keeps its split, so the
//     activations of data parallelism stay split and a weight split over
//     the same mesh axis is gathered, as fully sharded data parallelism
//     does; elsewhere the operand split most keeps its split. A tensor read
//     transposed or broadcast is resharded before it is, so a gathered
//     weight moves once per computation, at its own size. Everything
//     else, such as sorting along a split axis, a factorization or a
//     random sampler, makes its operands whole on every device first, so
//     every operation gives the same result as on one device.
//
// WithSharding steers the choices. Random tensors, such as dropout's
// masks, are drawn whole on every device and cut, so they match one
// device's.
func Shard(t *Tensor, m *Mesh, spec PartitionSpec) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	if m == nil {
		return fail(fmt.Errorf("tensor: shard on a nil mesh: %w", ErrDevice))
	}
	if err := computable(t); err != nil {
		return fail(err)
	}
	lay, err := m.layoutOf(spec, t.node.Shape, t.node.DType)
	if err != nil {
		return fail(err)
	}
	src := backendOf(t.node)
	if src == m.dev {
		out := constrain(t, m, lay)
		if err := out.Realize(context.Background()); err != nil {
			return fail(err)
		}
		return out
	}
	srcSpec, fromMesh := t.Sharding()
	data, err := hostBytes(context.Background(), t.node)
	if err != nil {
		return fail(err)
	}
	out := m.place(context.Background(), t.node.DType, data, t.node.Shape, lay)
	return record(out, rule{
		vjp: func(g *Tensor, _ []*Tensor, _ *Tensor) []*Tensor {
			if fromMesh {
				return grads(Shard(g, srcSpec.Mesh, srcSpec.Spec))
			}
			return grads(g.To(Device{src}))
		},
		jvp: func(tn, _ []*Tensor, _ *Tensor) *Tensor { return Shard(tn[0], m, spec) },
	}, t)
}

// WithSharding returns t, which is on m or computed from constants alone,
// constrained to be laid out over m as spec says, like
// jax.lax.with_sharding_constraint: the operations that read it see it
// laid out so, and the collectives that this takes run where it is
// computed. Its gradient is constrained the same way. Nothing is computed
// now.
func WithSharding(t *Tensor, m *Mesh, spec PartitionSpec) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	if m == nil {
		return fail(fmt.Errorf("tensor: sharding constraint on a nil mesh: %w", ErrDevice))
	}
	if be := sched.Device(t.node); be != nil && be != m.dev {
		return fail(fmt.Errorf("tensor: sharding constraint over %v on a tensor on %v: %w", m, Device{be}, ErrDevice))
	}
	lay, err := m.layoutOf(spec, t.node.Shape, t.node.DType)
	if err != nil {
		return fail(err)
	}
	return constrain(t, m, lay)
}

// constrain returns t, which is on m or computed from constants alone,
// constrained to the layout l over m, with a gradient constrained so too.
func constrain(t *Tensor, m *Mesh, l layout) *Tensor {
	if l2 := batchLevel(t); l2 != 0 {
		return fail(fmt.Errorf("tensor: sharding constraint inside VMap: %w", ErrNotBatchable))
	}
	n, err := graph.Reshape(t.node, t.node.Shape)
	if err != nil {
		return fail(err)
	}
	n.Dev = m.dev
	n.Constraint = &l
	out := wrap(n, nil)
	return record(out, rule{
		vjp: func(g *Tensor, _ []*Tensor, _ *Tensor) []*Tensor { return grads(constrain(g, m, l)) },
		jvp: func(tn, _ []*Tensor, _ *Tensor) *Tensor { return constrain(tn[0], m, l) },
	}, t)
}

// constrainLike returns g, a gradient of a tensor whose node is ref,
// constrained to ref's layout if ref is a computed sharded tensor, so
// that Backward computes gradients laid out like what they are added to.
func constrainLike(g *Tensor, ref *graph.Node) *Tensor {
	m := meshOf(ref)
	if m == nil || g.err != nil || !sched.Realized(ref) {
		return g
	}
	if be := sched.Device(g.node); be != nil && be != m.dev {
		return g
	}
	buf, ok := ref.Mem.Buf.(*meshBuffer)
	if !ok {
		return g
	}
	return constrain(g, m, buf.lay)
}

// Sharding reports how t is laid out over a mesh, and false if t is not
// sharded. For a sharded tensor not computed yet, it is the layout that
// computing it would give it, with partial sums added up.
func (t *Tensor) Sharding() (Sharding, bool) {
	if firstErr(t) != nil {
		return Sharding{}, false
	}
	m := meshOf(t.node)
	if m == nil {
		return Sharding{}, false
	}
	meshMu.Lock()
	defer meshMu.Unlock()
	l, err := m.predict(t.node)
	if err != nil {
		return Sharding{Mesh: m}, true
	}
	return Sharding{Mesh: m, Spec: m.spec(l)}, true
}

// Shard moves the parameter's value, and the gradient accumulated since
// the last ZeroGrad, onto m, laid out as spec says (see Shard), as To
// moves them to a device. Gradients that Backward computes for it then
// take the same layout, and so do the states an optimizer keeps for it.
func (p *Param) Shard(m *Mesh, spec PartitionSpec) error {
	return p.move(fmt.Sprintf("%v", Sharding{m, spec}), func(t *Tensor) *Tensor { return Shard(t, m, spec) })
}

// hostBytes computes n and returns its elements as row-major bytes.
func hostBytes(ctx context.Context, n *graph.Node) ([]byte, error) {
	size, ok := n.DType.Bytes(n.Size())
	if !ok {
		return nil, fmt.Errorf("tensor: %s tensor of shape %v: %w", n.DType, n.Shape, ErrShape)
	}
	data := make([]byte, size)
	if err := download(ctx, n, data); err != nil {
		return nil, err
	}
	return data, nil
}

// download computes n and copies its elements into dst as row-major
// bytes, gathering a sharded tensor's shards.
func download(ctx context.Context, n *graph.Node, dst []byte) error {
	m := meshOf(n)
	if m == nil {
		return sched.Download(ctx, backendOf(n), n, dst, lazy.values)
	}
	meshMu.Lock()
	defer meshMu.Unlock()
	if err := m.realize(ctx, []*graph.Node{n}, nil); err != nil {
		return err
	}
	buf, ok := n.Mem.Buf.(*meshBuffer)
	if !ok || n.Stale() {
		return fmt.Errorf("tensor: read a sharded tensor: %w", ErrOverwritten)
	}
	return m.gather(ctx, buf, n.Shape, dst)
}

// place returns a sharded tensor of shape laid out as l over m, holding
// data, the row-major bytes of its elements of dt.
func (m *Mesh) place(ctx context.Context, dt DType, data []byte, shape []int, l layout) *Tensor {
	if defaultErr != nil {
		return fail(defaultErr)
	}
	shards := make([]*graph.Node, len(m.devs))
	local := m.localShape(shape, l)
	elem := dt.Size()
	for r := range m.devs {
		var part []byte
		if l.used() == 0 || elem == 0 {
			part = data
		} else {
			size, _ := dt.Bytes(prod(local))
			part = make([]byte, size)
			copyBlock(part, data, elem, shape, local, m.indexMaps(r, l, shape), false)
		}
		n, err := sched.Upload(ctx, m.bes[r], dt, part, local)
		if err != nil {
			return fail(err)
		}
		shards[r] = n
	}
	n, err := graph.Load(m.dev, &meshBuffer{dt: dt, n: prod(shape), shards: shards, lay: l}, shape)
	if err != nil {
		return fail(err)
	}
	return &Tensor{node: n}
}

// gather copies the elements of the shards of buf, a sharded tensor of
// shape, into dst as row-major bytes, reading each block from one device.
func (m *Mesh) gather(ctx context.Context, buf *meshBuffer, shape []int, dst []byte) error {
	used := buf.lay.used()
	elem := buf.dt.Size()
	for r, s := range buf.shards {
		if s.Stale() {
			return fmt.Errorf("tensor: read a sharded tensor: %w", ErrOverwritten)
		}
		if !m.leads(r, used) {
			continue
		}
		if used == 0 || elem == 0 {
			return sched.Download(ctx, m.bes[r], s, dst, lazy.values)
		}
		local := s.Shape
		size, _ := buf.dt.Bytes(prod(local))
		part := make([]byte, size)
		if err := sched.Download(ctx, m.bes[r], s, part, lazy.values); err != nil {
			return err
		}
		copyBlock(dst, part, elem, shape, local, m.indexMaps(r, buf.lay, shape), true)
	}
	return nil
}

// leads reports whether rank r is at coordinate 0 along every axis
// outside used: one of the ranks that together hold every block once.
func (m *Mesh) leads(r int, used axisSet) bool {
	for a := range m.shape {
		if !used.has(a) && m.coord(r, a) != 0 {
			return false
		}
	}
	return true
}

// indexMaps returns, for each axis of a tensor of shape laid out as l,
// the position in the whole tensor of each position in rank r's block.
func (m *Mesh) indexMaps(r int, l layout, shape []int) [][]int {
	maps := make([][]int, len(shape))
	for d, n := range shape {
		dl := l.dims[d]
		if !dl.split() {
			maps[d] = nil
			continue
		}
		k := m.shape[dl.axis]
		c := m.coord(r, int(dl.axis))
		idx := make([]int, 0, n/k)
		off := 0
		for _, s := range dl.sections(n) {
			b := s / k
			for i := range b {
				idx = append(idx, off+c*b+i)
			}
			off += s
		}
		maps[d] = idx
	}
	return maps
}

// copyBlock copies a block of a row-major array of shape full, whose
// position along each axis d in the whole array is maps[d][i] for its
// position i, or i if maps[d] is nil, between the block, row-major of
// shape local in block, and the whole array in whole: into whole if
// toWhole is set, and out of it otherwise. Elements are elem bytes.
func copyBlock(a, b []byte, elem int, full, local []int, maps [][]int, toWhole bool) {
	whole, block := b, a
	if toWhole {
		whole, block = a, b
	}
	r := len(full)
	if r == 0 {
		copy(a, b[:len(a)])
		return
	}
	strides := make([]int, r)
	s := elem
	for d := r - 1; d >= 0; d-- {
		strides[d], s = s, s*full[d]
	}
	pos := func(d, i int) int {
		if maps[d] == nil {
			return i
		}
		return maps[d][i]
	}
	last := local[r-1]
	// A run along the last axis is contiguous in the whole array unless
	// it crosses sections.
	contiguous := maps[r-1] == nil || last == 0 || maps[r-1][last-1]-maps[r-1][0] == last-1
	idx := make([]int, r-1)
	for at := 0; at < len(block); at += last * elem {
		off := 0
		for d, i := range idx {
			off += pos(d, i) * strides[d]
		}
		row := block[at : at+last*elem]
		if contiguous {
			w := whole[off+pos(r-1, 0)*elem:][:last*elem]
			if toWhole {
				copy(w, row)
			} else {
				copy(row, w)
			}
		} else {
			for i := range last {
				w := whole[off+pos(r-1, i)*elem:][:elem]
				if toWhole {
					copy(w, row[i*elem:][:elem])
				} else {
					copy(row[i*elem:][:elem], w)
				}
			}
		}
		for d := r - 2; d >= 0; d-- {
			idx[d]++
			if idx[d] < local[d] {
				break
			}
			idx[d] = 0
		}
	}
}

// prod returns the number of elements of shape.
func prod(shape []int) int {
	n, _ := backend.NumElements(shape)
	return n
}

// realizeOn computes targets on be, writing targets[i] over dsts[i] where
// dsts is not nil and dsts[i] is not nil (see RealizeInto): on a device,
// by its scheduler, and on a mesh, by computing shards on its devices.
func realizeOn(ctx context.Context, be backend.Backend, targets, dsts []*graph.Node) error {
	d, ok := be.(*meshDevice)
	if !ok {
		return sched.RealizeInto(ctx, be, targets, dsts, lazy.values)
	}
	meshMu.Lock()
	defer meshMu.Unlock()
	return d.mesh.realize(ctx, targets, dsts)
}

// donate marks the memory of the realized node n as donated (see
// Tensor.Donate): a sharded tensor's shards are each donated.
func donate(n *graph.Node) {
	if meshOf(n) == nil {
		sched.Donate(n)
		return
	}
	meshMu.Lock()
	defer meshMu.Unlock()
	if buf, ok := n.Mem.Buf.(*meshBuffer); ok && sched.Realized(n) {
		for _, s := range buf.shards {
			sched.Donate(s)
		}
	}
}

// release gives the memory of the realized node n back for later
// computations, as sched.Release does: for a sharded tensor, its shards'.
func release(n *graph.Node) {
	if meshOf(n) == nil {
		sched.Release(n, lazy.values)
		return
	}
	meshMu.Lock()
	defer meshMu.Unlock()
	if !sched.Realized(n) || n.Stale() || n.Mem.Refs != 1 {
		return
	}
	if buf, ok := n.Mem.Buf.(*meshBuffer); ok {
		for _, s := range buf.shards {
			sched.Release(s, lazy.values)
		}
	}
	sched.Invalidate(n)
}
