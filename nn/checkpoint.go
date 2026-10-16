package nn

import (
	"reflect"
	"sync/atomic"
	"unsafe"

	"github.com/gsoultan/mimir/tensor"
)

// Checkpoint returns m with activation checkpointing, like
// torch.utils.checkpoint: in training mode Backward recomputes m's
// forward pass to get its gradients instead of keeping the activations
// inside m alive from the forward pass to the backward pass, which trades
// compute for memory (see tensor.Checkpoint). In inference mode the
// result applies m directly. It composes like Sequential: the parameters
// and buffers of m are found under "Inner", as in "Inner.Weight", and
// Train and Eval reach m.
//
//	model := nn.NewSequential(
//		nn.Checkpoint(nn.NewSequential(nn.NewLinear(512, 2048), nn.GELU(), nn.NewLinear(2048, 512))),
//		nn.NewLayerNorm(512),
//	)
//
// The recompute runs m's forward pass a second time, so a layer whose
// forward pass changes state it reads first gets that state back: the
// buffers of m, such as running statistics, take the values they had
// before the forward pass, and Dropout layers draw the mask they drew.
// The running statistics are then updated once, as without Checkpoint.
func Checkpoint(m Module) Module {
	c := &checkpointed{Inner: m}
	c.training.Store(true)
	return c
}

// checkpointed is a module computed with tensor.Checkpoint in training
// mode.
type checkpointed struct {
	Inner    Module
	training atomic.Bool
}

// SetTraining switches between recomputing and applying Inner directly.
func (c *checkpointed) SetTraining(on bool) { c.training.Store(on) }

// Forward applies Inner to x.
func (c *checkpointed) Forward(x *tensor.Tensor) *tensor.Tensor {
	if !c.training.Load() {
		return c.Inner.Forward(x)
	}
	restore := snapshot(c.Inner)
	return tensor.Checkpoint(func(in ...*tensor.Tensor) *tensor.Tensor {
		if err := restore(); err != nil {
			return tensor.Failed(err)
		}
		return c.Inner.Forward(in[0])
	}, x)
}

// snapshot records the state that a forward pass of m changes and reads:
// the values of its buffers and the masks its dropout layers are up to.
// The returned function puts that state back, so that a forward pass run
// after it does what one run now would.
func snapshot(m Module) (restore func() error) {
	var buffers []*tensor.Param
	var values []*tensor.Tensor
	var dropouts []*Dropout
	var calls []uint64
	w := walker{seen: map[unsafe.Pointer]bool{}}
	w.visit(reflect.ValueOf(m), "", func(_ string, v reflect.Value) bool {
		switch x := v.Interface().(type) {
		case *tensor.Param:
			if !x.Trainable() {
				buffers, values = append(buffers, x), append(values, x.Value())
			}
		case *Dropout:
			dropouts, calls = append(dropouts, x), append(calls, x.calls.Load())
		}
		return true
	})
	return func() error {
		for i, p := range buffers {
			if err := p.Set(values[i]); err != nil {
				return err
			}
		}
		for i, d := range dropouts {
			d.calls.Store(calls[i])
		}
		return nil
	}
}
