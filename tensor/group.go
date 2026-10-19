package tensor

import (
	"context"
	"fmt"
	"runtime"
	"slices"
	"sync"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/backend/cpu"
	"github.com/gsoultan/mimir/backend/cuda"
	"github.com/gsoultan/mimir/internal/graph"
	"github.com/gsoultan/mimir/internal/sched"
)

// ReduceOp is how a collective combines the values of a group's devices.
type ReduceOp = backend.Reduction

// Reductions of collectives. ReduceMean divides the sum by the number of
// devices and takes float tensors only.
const (
	ReduceSum  = backend.ReduceSum
	ReduceMean = backend.ReduceMean
	ReduceMax  = backend.ReduceMax
	ReduceMin  = backend.ReduceMin
)

var (
	cpusMu sync.Mutex
	cpus   = map[int][]Device{}
)

// CPUs returns n CPU devices that share the host's processors, each
// running kernels with GOMAXPROCS/n goroutines, at least one, as a GPU
// runs them: in order, on a goroutine of its own, after the call that
// queued them returns. The kernels of the n devices run at the same time,
// which suits a Group, such as the replicas of data-parallel training,
// on a CPU of several sockets or with more cores than one batch keeps
// busy. Like a GPU's, their kernels report errors at the next read of a
// value from the device rather than when a computation runs. Every call
// with the same n returns the same devices.
func CPUs(n int) []Device {
	if n < 1 {
		return nil
	}
	cpusMu.Lock()
	defer cpusMu.Unlock()
	if ds, ok := cpus[n]; ok {
		return slices.Clone(ds)
	}
	ds := make([]Device, n)
	for i := range ds {
		ds[i] = Device{cpu.NewStream(runtime.GOMAXPROCS(0) / n)}
	}
	cpus[n] = ds
	return slices.Clone(ds)
}

var gpus = sync.OnceValue(func() []Device {
	d, err := GPU()
	if err != nil {
		return nil
	}
	ds := []Device{d}
	if _, ok := d.be.(*cuda.Backend); !ok {
		return ds
	}
	for i := 1; i < cuda.DeviceCount(); i++ {
		c, err := cuda.NewDevice(i)
		if err != nil {
			break
		}
		ds = append(ds, Device{c})
	}
	return ds
})

// GPUs returns every GPU mimir can use, the one GPU returns first, or
// none. On Linux they are the CUDA devices, which a Group connects through
// NCCL; a Mac has one GPU.
func GPUs() []Device { return slices.Clone(gpus()) }

// Group connects devices of one kind, such as GPUs or the devices of
// CPUs, to move and combine tensors between them with collectives, like
// a group of NCCL ranks in one process: device i of the group is rank i.
// CUDA devices communicate through NCCL, and CPU devices through host
// memory. A group of GPUs makes them asynchronous while it is open, as
// the devices of CPUs are: their kernels report errors at the next read
// of a value from them.
//
// Each collective takes one tensor per rank, on that rank's device, all
// of one dtype and shape, computes them, and returns realized tensors
// without gradients. Each rank's part runs after the work queued on its
// device so far, and the host does not wait for it. A Group is safe for
// concurrent use.
type Group struct {
	devs []Device
	bes  []backend.Backend
	comm backend.Communicator
}

// NewGroup returns a group of devs, in rank order. The devices must be
// distinct and of one kind that has collectives: the CPU, the devices of
// CPUs, or CUDA devices. Close releases it.
func NewGroup(devs ...Device) (*Group, error) {
	if len(devs) == 0 {
		return nil, fmt.Errorf("tensor: group of no devices: %w", ErrShape)
	}
	g := &Group{devs: slices.Clone(devs), bes: make([]backend.Backend, len(devs))}
	for i, d := range devs {
		g.bes[i] = d.backend()
		if g.bes[i].Name() != g.bes[0].Name() {
			return nil, fmt.Errorf("tensor: group of %s and %s devices: %w", g.bes[0].Name(), g.bes[i].Name(), ErrDevice)
		}
	}
	grouper, ok := g.bes[0].(backend.Grouper)
	if !ok {
		return nil, fmt.Errorf("tensor: %v devices have no collectives: %w", devs[0], ErrDevice)
	}
	comm, err := grouper.Group(g.bes)
	if err != nil {
		return nil, fmt.Errorf("tensor: group: %w", err)
	}
	g.comm = comm
	return g, nil
}

// Devices returns the devices of g in rank order.
func (g *Group) Devices() []Device { return slices.Clone(g.devs) }

// Close releases the group once the collectives queued so far have run.
func (g *Group) Close() error { return g.comm.Close() }

// AllReduce returns, on every rank, the reduction over the ranks of ts,
// elementwise: ts[i] is rank i's tensor.
func (g *Group) AllReduce(ctx context.Context, op ReduceOp, ts ...*Tensor) ([]*Tensor, error) {
	shape, err := g.check(ts, -1)
	if err != nil {
		return nil, err
	}
	return g.collective(ctx, backend.Collective{Kind: backend.AllReduce, Reduce: op}, ts, shape, -1)
}

// Broadcast returns root's tensor ts[root] on every rank: ts[root]
// itself on root, and a copy elsewhere. The other ranks' tensors may be
// nil; any that are not are ignored.
func (g *Group) Broadcast(ctx context.Context, root int, ts ...*Tensor) ([]*Tensor, error) {
	if root < 0 || root >= len(g.bes) || len(ts) != len(g.bes) {
		return nil, fmt.Errorf("tensor: broadcast from rank %d of %d tensors over %d ranks: %w", root, len(ts), len(g.bes), ErrShape)
	}
	ins := make([]*Tensor, len(ts))
	ins[root] = ts[root]
	shape, err := g.check(ins, root)
	if err != nil {
		return nil, err
	}
	out, err := g.collective(ctx, backend.Collective{Kind: backend.Broadcast, Root: root}, ins, shape, root)
	if err != nil {
		return nil, err
	}
	out[root] = ts[root]
	return out, nil
}

// AllGather returns, on every rank, the tensors of all ranks joined along
// their first axis in rank order, as Cat does.
func (g *Group) AllGather(ctx context.Context, ts ...*Tensor) ([]*Tensor, error) {
	shape, err := g.check(ts, -1)
	if err != nil {
		return nil, err
	}
	if len(shape) == 0 {
		return nil, fmt.Errorf("tensor: all-gather of scalars: %w", ErrShape)
	}
	shape[0] *= len(g.bes)
	return g.collective(ctx, backend.Collective{Kind: backend.AllGather}, ts, shape, -1)
}

// ReduceScatter cuts the reduction over the ranks of ts into as many
// equal parts along the first axis as there are ranks, and returns part
// i on rank i. The first axis must divide evenly.
func (g *Group) ReduceScatter(ctx context.Context, op ReduceOp, ts ...*Tensor) ([]*Tensor, error) {
	shape, err := g.check(ts, -1)
	if err != nil {
		return nil, err
	}
	if len(shape) == 0 || shape[0]%len(g.bes) != 0 {
		return nil, fmt.Errorf("tensor: reduce-scatter of shape %v over %d ranks: %w", shape, len(g.bes), ErrShape)
	}
	shape[0] /= len(g.bes)
	return g.collective(ctx, backend.Collective{Kind: backend.ReduceScatter, Reduce: op}, ts, shape, -1)
}

// check verifies that ts holds a tensor per rank, or, if only is not -1,
// only for rank only, each of one dtype and shape and not living on a
// device other than its rank's, and returns a copy of the shape.
func (g *Group) check(ts []*Tensor, only int) ([]int, error) {
	if len(ts) != len(g.bes) {
		return nil, fmt.Errorf("tensor: collective of %d tensors over %d ranks: %w", len(ts), len(g.bes), ErrShape)
	}
	var first *Tensor
	for r, t := range ts {
		if only >= 0 && r != only {
			continue
		}
		if err := firstErr(t); err != nil {
			return nil, err
		}
		if err := computable(t); err != nil {
			return nil, err
		}
		if be := sched.Device(t.node); be != nil && be != g.bes[r] {
			return nil, fmt.Errorf("tensor: collective tensor of rank %d on %v, want %v: %w", r, Device{be}, g.devs[r], ErrDevice)
		}
		if first == nil {
			first = t
			continue
		}
		if t.node.DType != first.node.DType || !slices.Equal(t.node.Shape, first.node.Shape) {
			return nil, fmt.Errorf("tensor: collective of %s%v and %s%v: %w", first.node.DType, first.node.Shape, t.node.DType, t.node.Shape, ErrShape)
		}
	}
	return slices.Clone(first.node.Shape), nil
}

// collective runs c over the tensors ts, which check verified, into new
// tensors of shape. c's buffers are filled in from those of ts, nil
// where ts[r] is, and the outputs; skip names a rank without an output,
// or is -1.
func (g *Group) collective(ctx context.Context, c backend.Collective, ts []*Tensor, shape []int, skip int) ([]*Tensor, error) {
	n := len(g.bes)
	var dt DType
	ins := make([][]*graph.Node, n)
	outs := make([][]sched.Out, n)
	for r, t := range ts {
		if t != nil {
			ins[r], dt = []*graph.Node{t.node}, t.node.DType
		} else {
			ins[r] = []*graph.Node{nil}
		}
	}
	for r := range outs {
		outs[r] = []sched.Out{{DType: dt, Shape: shape, Over: -1}}
		if r == skip {
			outs[r][0].DType = backend.Invalid
		}
	}
	nodes, err := sched.Collective(ctx, g.bes, ins, outs, lazy.values, func(in, out [][]backend.Buffer) error {
		c.In, c.Out = make([]backend.Buffer, n), make([]backend.Buffer, n)
		for r := range n {
			c.In[r], c.Out[r] = in[r][0], out[r][0]
		}
		return g.comm.Run(ctx, []backend.Collective{c})
	})
	if err != nil {
		return nil, err
	}
	res := make([]*Tensor, n)
	for r := range n {
		if nodes[r][0] != nil {
			res[r] = wrap(nodes[r][0], nil)
		}
	}
	return res, nil
}

// bucketBytes is the most gradient or parameter bytes ReduceGrads and
// BroadcastParams move in one group of collectives, as PyTorch's
// DistributedDataParallel buckets them.
const bucketBytes = 25 << 20

// ReduceGrads reduces the gradients of the parameters across the group
// into the gradients of rank root, in place: params[r] are rank r's
// parameters, such as a replica's nn.Parameters, the same number on every
// rank with the same shapes, and params[root][i]'s gradient becomes the
// reduction of the gradients of params[r][i] over every r. A parameter
// without a gradient on some ranks counts as zeros there, and one without
// a gradient on any rank is skipped. The other ranks' gradients are left
// as they are. The reduction goes over the memory of root's gradients
// where they hold it alone, and a gradient read from them before is then
// stale.
//
// The gradients go in buckets of about 25 MiB, last parameter first, as
// a backward pass computes them, each bucket's collectives queued as one
// group.
func (g *Group) ReduceGrads(ctx context.Context, op ReduceOp, root int, params ...[]*Param) error {
	return g.reduceGrads(ctx, op, root, params)
}

// AllReduceGrads reduces the gradients of the parameters across the group
// as ReduceGrads does, into the gradients of every rank: afterwards every
// rank's gradients hold the same values, bit for bit, so that an
// optimizer step on each rank computes the same parameters as on any
// other (see train.Parallel.Step).
func (g *Group) AllReduceGrads(ctx context.Context, op ReduceOp, params ...[]*Param) error {
	return g.reduceGrads(ctx, op, -1, params)
}

// reduceGrads reduces the gradients of params into rank root's, or into
// every rank's if root is -1.
func (g *Group) reduceGrads(ctx context.Context, op ReduceOp, root int, params [][]*Param) error {
	kind, all := backend.Reduce, root < 0
	if all {
		kind, root = backend.AllReduce, 0
	}
	values, err := g.checkParams(root, params)
	if err != nil {
		return err
	}
	gradMu.Lock()
	defer gradMu.Unlock()
	// The gradients are read and replaced once no Backward holds any of
	// them.
	for slices.ContainsFunc(values, func(vs []*Tensor) bool {
		return slices.ContainsFunc(vs, func(v *Tensor) bool { return v.tape != nil && v.tape.busy })
	}) {
		gradIdle.Wait()
	}
	n := len(g.bes)
	var idx []int
	for i := len(params[root]) - 1; i >= 0; i-- {
		for r := range n {
			if t := values[r][i].tape; t != nil && t.leaf && t.grad != nil {
				idx = append(idx, i)
				break
			}
		}
	}
	return g.buckets(idx, values[root], func(bucket []int) error {
		ins := make([][]*graph.Node, n)
		outs := make([][]sched.Out, n)
		for r := range n {
			ins[r] = make([]*graph.Node, len(bucket))
			outs[r] = make([]sched.Out, len(bucket))
		}
		for j, i := range bucket {
			var like *Tensor
			for r := range n {
				if gr := values[r][i].tape.grad; gr != nil {
					like = gr
					break
				}
			}
			if err := firstErr(like); err != nil {
				return err
			}
			for r := range n {
				gr := values[r][i].tape.grad
				if gr == nil {
					gr = Zeros(like.node.DType, like.node.Shape...)
				}
				if be := sched.Device(gr.node); be != nil && be != g.bes[r] {
					// A gradient computed from constants alone, such as
					// that of a bias summed into the loss, was computed
					// on the default device, as constants are with
					// MIMIR_EAGER=1.
					gr = gr.To(g.devs[r])
				}
				if err := firstErr(gr); err != nil {
					return err
				}
				ins[r][j] = gr.node
				if all || r == root {
					outs[r][j] = sched.Out{DType: like.node.DType, Shape: like.node.Shape, Over: j}
				}
			}
		}
		nodes, err := sched.Collective(ctx, g.bes, ins, outs, lazy.values, func(in, out [][]backend.Buffer) error {
			cs := make([]backend.Collective, len(bucket))
			for j := range bucket {
				c := backend.Collective{Kind: kind, Reduce: op, Root: root, In: make([]backend.Buffer, n), Out: make([]backend.Buffer, n)}
				for r := range n {
					c.In[r], c.Out[r] = in[r][j], out[r][j]
				}
				cs[j] = c
			}
			return g.comm.Run(ctx, cs)
		})
		if err != nil {
			return err
		}
		for r := range n {
			if !all && r != root {
				continue
			}
			for j, i := range bucket {
				t := values[r][i].tape
				t.grad, t.own = wrap(nodes[r][j], nil), true
			}
		}
		return nil
	})
}

// BroadcastParams copies the values of rank root's parameters over those
// of the other ranks, in place, as ReduceGrads takes its parameters: the
// trainable ones and buffers alike, such as the running statistics of
// batch normalization. Afterwards every rank's parameters hold the same
// values, bit for bit. Each goes over the memory of the value it
// replaces where that value holds it alone, as an optimizer step's does,
// and a value read from the parameter before is then stale.
// Block-quantized buffers, which never change, are left alone.
func (g *Group) BroadcastParams(ctx context.Context, root int, params ...[]*Param) error {
	values, err := g.checkParams(root, params)
	if err != nil {
		return err
	}
	n := len(g.bes)
	var idx []int
	for i, v := range values[root] {
		if !v.node.DType.IsQuantized() {
			idx = append(idx, i)
		}
	}
	return g.buckets(idx, values[root], func(bucket []int) error {
		ins := make([][]*graph.Node, n)
		outs := make([][]sched.Out, n)
		for r := range n {
			ins[r] = make([]*graph.Node, len(bucket))
			outs[r] = make([]sched.Out, len(bucket))
			for j, i := range bucket {
				v := values[r][i]
				if r == root {
					ins[r][j] = v.node
					continue
				}
				// The value about to be replaced is computed only if it
				// is already: its memory is then written over.
				if sched.Realized(v.node) {
					ins[r][j] = v.node
				}
				src := values[root][i].node
				outs[r][j] = sched.Out{DType: src.DType, Shape: src.Shape, Over: j}
			}
		}
		nodes, err := sched.Collective(ctx, g.bes, ins, outs, lazy.values, func(in, out [][]backend.Buffer) error {
			cs := make([]backend.Collective, len(bucket))
			for j := range bucket {
				c := backend.Collective{Kind: backend.Broadcast, Root: root, In: make([]backend.Buffer, n), Out: make([]backend.Buffer, n)}
				c.In[root] = in[root][j]
				for r := range n {
					c.Out[r] = out[r][j]
				}
				cs[j] = c
			}
			return g.comm.Run(ctx, cs)
		})
		if err != nil {
			return err
		}
		for r := range n {
			if r == root {
				continue
			}
			for j, i := range bucket {
				if err := params[r][i].Set(wrap(nodes[r][j], nil)); err != nil {
					return err
				}
			}
		}
		return nil
	})
}

// checkParams verifies that params holds as many parameters for every
// rank, of the same dtypes and shapes, with values on their ranks'
// devices, and returns their values.
func (g *Group) checkParams(root int, params [][]*Param) ([][]*Tensor, error) {
	n := len(g.bes)
	if root < 0 || root >= n || len(params) != n {
		return nil, fmt.Errorf("tensor: parameters of %d ranks to rank %d of a group of %d: %w", len(params), root, n, ErrShape)
	}
	values := make([][]*Tensor, n)
	for r := range n {
		if len(params[r]) != len(params[root]) {
			return nil, fmt.Errorf("tensor: %d parameters on rank %d and %d on rank %d: %w", len(params[r]), r, len(params[root]), root, ErrShape)
		}
		values[r] = make([]*Tensor, len(params[r]))
		for i, p := range params[r] {
			v := p.Value()
			if err := firstErr(v); err != nil {
				return nil, err
			}
			if be := sched.Device(v.node); be != nil && be != g.bes[r] {
				return nil, fmt.Errorf("tensor: parameter %d of rank %d on %v, want %v: %w", i, r, Device{be}, g.devs[r], ErrDevice)
			}
			if w := values[0]; r > 0 && (v.node.DType != w[i].node.DType || !slices.Equal(v.node.Shape, w[i].node.Shape)) {
				return nil, fmt.Errorf("tensor: parameter %d of %s%v on rank %d and %s%v on rank 0: %w", i, v.node.DType, v.node.Shape, r, w[i].node.DType, w[i].node.Shape, ErrShape)
			}
			values[r][i] = v
		}
	}
	return values, nil
}

// buckets calls run with the indexes of idx in order, in runs whose
// tensors among like hold up to bucketBytes, or one tensor if it is
// larger.
func (g *Group) buckets(idx []int, like []*Tensor, run func([]int) error) error {
	for len(idx) > 0 {
		end, bytes := 0, 0
		for end < len(idx) {
			n := like[idx[end]].node
			size, _ := n.DType.Bytes(n.Size())
			if end > 0 && bytes+size > bucketBytes {
				break
			}
			bytes += size
			end++
		}
		if err := run(idx[:end]); err != nil {
			return err
		}
		idx = idx[end:]
	}
	return nil
}
