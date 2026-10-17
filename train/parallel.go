package train

import (
	"context"
	"errors"
	"fmt"

	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/nn/loss"
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
// with one device; the replicas take the new values at the next Backward:
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
// own gradients are discarded.
//
// First, if the model's parameters or buffers hold values the replicas
// do not, such as those of an optimizer step, the replicas take them.
// A batch smaller than the number of devices leaves some devices idle.
// x and y may live on any device; each device's share is copied to it.
func (p *Parallel) Backward(ctx context.Context, lossFn loss.Func, x, y *tensor.Tensor) (float64, error) {
	if err := p.broadcast(ctx); err != nil {
		return 0, err
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
	// between devices goes through the host, which waits for the source
	// device to finish what it was given, and so would wait for the
	// replica on that device to run its whole step.
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
		// On the devices of a group the backward pass is queued, and the
		// next device's starts while this one's runs.
		if err := l.Backward(ctx); err != nil {
			p.discard()
			return 0, err
		}
		losses = append(losses, l)
	}
	err := p.group.ReduceGrads(ctx, tensor.ReduceSum, 0, p.trained...)
	p.discard()
	if err != nil {
		return 0, err
	}
	var sum float64
	for _, l := range losses {
		v, err := l.Item[float32](ctx) // computed by Backward
		if err != nil {
			return 0, err
		}
		sum += float64(v)
	}
	return sum, nil
}

// broadcast gives the replicas the model's parameter values, unless they
// hold them already.
func (p *Parallel) broadcast(ctx context.Context) error {
	cur := p.current()
	same := true
	for i, v := range cur {
		if v != p.values[i] {
			same = false
			break
		}
	}
	if same {
		return nil
	}
	if err := p.group.BroadcastParams(ctx, 0, p.params...); err != nil {
		return err
	}
	p.values = cur
	return nil
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
