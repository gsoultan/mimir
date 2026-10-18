package train

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"
	"time"

	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/nn/loss"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
)

// DataParallel trains on every device of devs at once: Fit moves the
// model to devs[0], replicates it onto the others, and splits each batch
// between them along its first axis (see Parallel). The optimizer steps
// the model on devs[0], and the replicas take its new parameters before
// the next batch, so every replica holds the same values, bit for bit.
// Evaluation and checkpoints use the model on devs[0].
//
// The devices are those of tensor.GPUs or tensor.CPUs, of one kind.
// DataParallel does not combine with MicroBatch or MemoryBudget.
func DataParallel(devs ...tensor.Device) Option {
	return func(c *config) {
		c.parallel = devs
		if len(devs) == 0 {
			c.fail("data parallelism over no devices")
		}
	}
}

// Parallel trains a model on several devices at once, as PyTorch's
// DistributedDataParallel does within one process: it holds a replica of
// the model on each device of a group, the model itself on the first,
// and computes the gradient of a batch by splitting the batch between
// them. Each replica computes the loss and gradients of its share, and
// the gradients are summed into the model's, weighted by each share's
// part of the batch, so the model's gradients are those of the whole
// batch's mean loss. Step the optimizer over the model's parameters as
// with one device; the replicas take the new values at the next Backward
// (or see Step):
//
//	dp, err := train.NewParallel(model, tensor.GPUs()...)
//	if err != nil { ... }
//	defer dp.Close()
//	for batch, err := range trainSet.All(ctx) {
//		...
//		l, err := dp.Backward(ctx, loss.CrossEntropy, batch.X, batch.Y)
//		if err != nil { ... }
//		if err := opt.Step(ctx); err != nil { ... }
//		opt.ZeroGrad()
//	}
//
// The devices run their shares at the same time: their kernels are
// queued, device after device, and the host waits only to read the
// losses. The replicas' buffers, such as the running statistics of batch
// normalization, take the model's values at each Backward too, so batch
// normalization normalizes each device's share by its own statistics, as
// in PyTorch without SyncBatchNorm, and each replica's dropout draws the
// masks that the model would on its share.
type Parallel struct {
	model    nn.Module
	group    *tensor.Group
	devs     []tensor.Device
	replicas []nn.Module
	params   [][]*tensor.Param // every replica's parameters and buffers, in NamedParams order
	trained  [][]*tensor.Param // every replica's trainable parameters
	// values are the model's parameter values that the replicas hold,
	// to tell when a step or anything else has set new ones.
	values []*tensor.Tensor
	// at[i] is the index in params of trained[.][i].
	at []int

	// The optimizers of Step: opt over the model's parameters, and
	// local[r] over replica r's, which hold its state too once synced is
	// set. state and steps are opt's state and step count after the last
	// Step, to tell when anything else has stepped opt or set its state.
	opt    *optim.Optimizer
	local  []*optim.Optimizer
	state  map[string]*tensor.Tensor
	steps  int
	synced bool
	// allReduced reports whether the last Backward reduced the gradients
	// into every replica's, for the replicas' optimizers to step on.
	allReduced bool
}

// NewParallel returns a Parallel that trains model, which must be on
// devs[0], on devs. Close releases it.
func NewParallel(model nn.Module, devs ...tensor.Device) (*Parallel, error) {
	if len(devs) == 0 {
		return nil, fmt.Errorf("%w: data parallelism over no devices", ErrConfig)
	}
	if d := deviceOf(model); !d.Equal(devs[0]) {
		return nil, fmt.Errorf("train: model on %v, but the first device of the group is %v; move it there with nn.To: %w", d, devs[0], tensor.ErrDevice)
	}
	g, err := tensor.NewGroup(devs...)
	if err != nil {
		return nil, err
	}
	p := &Parallel{model: model, group: g, devs: g.Devices(), replicas: []nn.Module{model}}
	for _, d := range p.devs[1:] {
		r, err := nn.Replicate(model, d)
		if err != nil {
			return nil, errors.Join(err, g.Close())
		}
		p.replicas = append(p.replicas, r)
	}
	for _, r := range p.replicas {
		var all []*tensor.Param
		for _, prm := range nn.NamedParams(r) {
			all = append(all, prm)
		}
		p.params = append(p.params, all)
		p.trained = append(p.trained, nn.Parameters(r))
	}
	p.values = p.current()
	for _, prm := range p.trained[0] {
		p.at = append(p.at, slices.Index(p.params[0], prm))
	}
	return p, nil
}

// Devices returns the devices of the replicas, the model's first.
func (p *Parallel) Devices() []tensor.Device { return p.group.Devices() }

// Replicas returns the model and its replicas, one per device.
func (p *Parallel) Replicas() []nn.Module { return append([]nn.Module(nil), p.replicas...) }

// Close releases the group of devices.
func (p *Parallel) Close() error { return p.group.Close() }

// current returns the values the model's parameters hold now.
func (p *Parallel) current() []*tensor.Tensor {
	vs := make([]*tensor.Tensor, len(p.params[0]))
	for i, prm := range p.params[0] {
		vs[i] = prm.Value()
	}
	return vs
}

// Backward computes the loss of the batch x, y with lossFn and its
// gradients, split between the devices along the first axis, and adds
// the gradients to the model's, as Backward on the model's loss would.
// It returns the batch's loss, which lossFn must compute as a mean over
// the examples, as every function of package loss does. The replicas'
// own gradients are discarded, unless the last Step stepped the replicas
// alongside the model: then the gradients are reduced into every
// replica's, for the next Step.
//
// First, if the model's parameters or buffers hold values the replicas
// do not, such as those of an optimizer step, the replicas take them.
// A batch smaller than the number of devices leaves some devices idle.
// x and y may live on any device; each device's share is copied to it.
func (p *Parallel) Backward(ctx context.Context, lossFn loss.Func, x, y *tensor.Tensor) (float64, error) {
	mark := phaseTimer()
	local := p.stepsLocally()
	p.allReduced = false
	if err := p.broadcast(ctx); err != nil {
		return 0, err
	}
	// The replicas may hold the gradients the last Backward reduced into
	// them, which the model holds too: each adds only its share's.
	p.discard()
	mark("broadcast", -1)
	if waitPhases && len(p.params[0]) > 0 {
		// The broadcast has reached every replica once the last can read
		// a value it wrote.
		last := p.params[len(p.params)-1][0].Value()
		if _, err := last.Reshape(-1).Sum().Item[float32](ctx); err != nil {
			return 0, err
		}
		mark("gpu-broadcast", -1)
	}
	if err := x.Err(); err != nil {
		return 0, err
	}
	if err := y.Err(); err != nil {
		return 0, err
	}
	if x.Rank() == 0 || y.Rank() == 0 || y.Shape()[0] != x.Shape()[0] {
		return 0, fmt.Errorf("train: split a batch of shape %v with targets of shape %v: %w", x.Shape(), y.Shape(), tensor.ErrShape)
	}
	rows, n := x.Shape()[0], len(p.devs)
	// A lazy batch, such as a shuffled gather, is computed once, not once
	// per device.
	if err := tensor.Realize(ctx, x, y); err != nil {
		return 0, err
	}
	// Every share reaches its device before any device computes: a copy
	// from the host, or between devices that do not copy between
	// themselves, waits for the devices to finish what they were given,
	// and so would wait for the replicas on them to run their steps.
	xs, ys := make([]*tensor.Tensor, n), make([]*tensor.Tensor, n)
	moved := make([]*tensor.Tensor, 0, 2*n)
	for r, d := range p.devs {
		lo, hi := rows*r/n, rows*(r+1)/n
		if lo == hi {
			continue
		}
		xs[r], ys[r] = share(rowsOf(x, lo, hi), d), share(rowsOf(y, lo, hi), d)
		moved = append(moved, xs[r], ys[r])
	}
	if err := tensor.Realize(ctx, moved...); err != nil {
		return 0, err
	}
	mark("share", -1)
	losses := make([]*tensor.Tensor, 0, n)
	for r := range p.devs {
		lo, hi := rows*r/n, rows*(r+1)/n
		if lo == hi {
			continue
		}
		l := lossFn(p.replicas[r].Forward(xs[r]), ys[r])
		if hi-lo != rows {
			l = l.Mul(tensor.Scalar(float32(hi-lo) / float32(rows)))
		}
		mark("forward", r)
		// On the devices of a group the backward pass is queued, and the
		// next device's starts while this one's runs.
		if err := l.Backward(ctx); err != nil {
			p.discard()
			return 0, err
		}
		mark("backward", r)
		losses = append(losses, l)
	}
	if waitPhases {
		// Each device's wait is the time it took past the one before it.
		for i, l := range losses {
			if _, err := l.Item[float32](ctx); err != nil {
				return 0, err
			}
			mark("gpu-backward", i)
		}
	}
	var err error
	if local {
		err = p.group.AllReduceGrads(ctx, tensor.ReduceSum, p.trained...)
		p.allReduced = err == nil
	} else {
		err = p.group.ReduceGrads(ctx, tensor.ReduceSum, 0, p.trained...)
	}
	if !p.allReduced {
		p.discard()
	}
	if err != nil {
		return 0, err
	}
	mark("reduce", -1)
	if waitPhases {
		if g := p.trained[0][0].Grad(); g != nil {
			if _, err := g.Reshape(-1).Sum().Item[float32](ctx); err != nil {
				return 0, err
			}
		}
		mark("gpu-reduce", -1)
	}
	var sum float64
	for _, l := range losses {
		v, err := l.Item[float32](ctx) // computed by Backward
		if err != nil {
			return 0, err
		}
		sum += float64(v)
	}
	mark("losses", -1)
	return sum, nil
}

// phases, when a test sets it, receives the host's time in each phase of
// Parallel.Backward: the phase's name, the rank it ran for or -1, and its
// duration.
var phases func(phase string, rank int, d time.Duration)

// waitPhases, when a test sets it with phases, makes Parallel.Backward
// wait for the devices after the broadcast, after each device's backward
// pass and after the reduction, and report the waits as phases named
// gpu-..., to tell how long the devices take in each. The waits keep the
// devices from overlapping one phase with the next, so a step takes
// longer.
var waitPhases bool

// phaseTimer returns a function that reports to phases the time since
// its last call, or since phaseTimer's, and does nothing when phases is
// nil.
func phaseTimer() func(phase string, rank int) {
	report := phases
	if report == nil {
		return func(string, int) {}
	}
	last := time.Now()
	return func(phase string, rank int) {
		now := time.Now()
		report(phase, rank, now.Sub(last))
		last = now
	}
}

// broadcast gives the replicas the model's parameter values they do not
// hold already.
func (p *Parallel) broadcast(ctx context.Context) error {
	cur := p.current()
	var changed []int
	for i, v := range cur {
		if v != p.values[i] {
			changed = append(changed, i)
		}
	}
	if len(changed) == 0 {
		return nil
	}
	params := p.params
	if len(changed) < len(cur) {
		params = make([][]*tensor.Param, len(p.params))
		for r, ps := range p.params {
			for _, i := range changed {
				params[r] = append(params[r], ps[i])
			}
		}
	}
	if err := p.group.BroadcastParams(ctx, 0, params...); err != nil {
		return err
	}
	p.values = cur
	return nil
}

// Step updates the model's parameters with opt, an optimizer over them,
// as opt.Step does, and the replicas' with optimizers like it over theirs,
// so that the replicas need not take the model's new values at the next
// Backward. Every replica computes the same step from the same values
// and gradients, so the replicas still hold the model's values, bit for
// bit. Stepping opt itself works too, and then the next Backward copies
// every new value from the model's device to the others.
//
// Which is faster depends on the devices. Step trades the copy of the
// new values for an all-reduce in place of a reduction, which moves as
// many bytes, and for the host's work of one step per device, which it
// queues one after another. On 4 T4 GPUs, which NCCL connects through
// host memory, the all-reduce takes less GPU time than the reduction and
// the copy together, but a GPT-2 block's step takes 8% longer than with
// opt.Step: the host's extra work outweighs it. Fit steps opt.
//
// The first Step, and a Step after anything else stepped opt or set its
// state or the model's parameters, steps opt alone and gives the
// replicas' optimizers its state: Backward reduces the gradients into the
// model's alone until then, and into every replica's from then on.
func (p *Parallel) Step(ctx context.Context, opt *optim.Optimizer) error {
	if !p.allReduced || !p.stepsLocally() || opt != p.opt {
		return p.sync(ctx, opt)
	}
	p.allReduced = false
	for _, o := range p.local[1:] {
		o.SetLR(opt.LR())
		o.SetGradScale(opt.GradScale())
	}
	// A failure leaves the replicas out of step with the model: the next
	// Backward gives them its values, and the next Step its state.
	p.synced = false
	if err := opt.Step(ctx); err != nil {
		return err
	}
	for _, o := range p.local[1:] {
		if err := o.Step(ctx); err != nil {
			return err
		}
		o.ZeroGrad()
	}
	cur := p.current()
	for _, i := range p.at {
		p.values[i] = cur[i]
	}
	p.mark(opt)
	return nil
}

// sync steps opt alone and gives the replicas' optimizers its state.
func (p *Parallel) sync(ctx context.Context, opt *optim.Optimizer) error {
	p.allReduced, p.synced = false, false
	if err := opt.Step(ctx); err != nil {
		return err
	}
	p.discard()
	if opt != p.opt {
		p.opt, p.local = opt, p.replicaOptimizers(opt)
	}
	if p.local == nil {
		return nil // opt steps more than the model's parameters
	}
	state, steps := opt.Snapshot()
	for r, o := range p.local[1:] {
		copies := make(map[string]*tensor.Tensor, len(state))
		for key, t := range state {
			copies[key] = t.To(p.devs[r+1])
		}
		if err := o.Restore(copies, steps); err != nil {
			return err
		}
	}
	p.mark(opt)
	return nil
}

// replicaOptimizers returns optimizers like opt over each replica's
// parameters, nil for the model's, or nil if opt updates parameters that
// are not the model's trainable ones.
func (p *Parallel) replicaOptimizers(opt *optim.Optimizer) []*optim.Optimizer {
	params := opt.Params()
	idx := make([]int, len(params))
	for j, prm := range params {
		if idx[j] = slices.Index(p.trained[0], prm); idx[j] < 0 {
			return nil
		}
	}
	local := make([]*optim.Optimizer, len(p.replicas))
	for r := 1; r < len(local); r++ {
		ps := make([]*tensor.Param, len(idx))
		for j, i := range idx {
			ps[j] = p.trained[r][i]
		}
		local[r] = opt.Over(ps)
	}
	return local
}

// mark records that the replicas' optimizers hold opt's state as it is.
func (p *Parallel) mark(opt *optim.Optimizer) {
	p.state, p.steps = opt.Snapshot()
	p.synced = true
}

// stepsLocally reports whether the replicas' optimizers hold the state of
// the optimizer of the last Step as it is, so that the replicas can step
// themselves.
func (p *Parallel) stepsLocally() bool {
	if !p.synced || len(p.devs) < 2 {
		return false
	}
	state, steps := p.opt.Snapshot()
	return steps == p.steps && maps.Equal(state, p.state)
}

// discard drops the replicas' gradients, which ReduceGrads has added to
// the model's.
func (p *Parallel) discard() {
	for _, ps := range p.trained[1:] {
		for _, prm := range ps {
			prm.ZeroGrad()
		}
	}
}

// share returns t on d. A copy that To makes is donated to the step that
// reads it, as Fit donates its batches.
func share(t *tensor.Tensor, d tensor.Device) *tensor.Tensor {
	moved := t.To(d)
	if moved == t {
		return moved
	}
	return moved.Donate()
}
