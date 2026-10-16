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
// The recompute runs m's forward pass a second time, and a layer's
// forward pass changes state it reads first: the buffers of m, such as
// running statistics, and the masks its Dropout layers are up to. Each
// forward pass changes that state once, in the order the passes ran, as
// without Checkpoint, however many passes run before Backward: the
// recompute of a pass runs from the state the pass started from, so it
// draws the masks the pass drew, and then puts back the state it found.
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
//
// The function given to tensor.Checkpoint is called first for the forward
// pass, whose changes to the state of Inner stand and whose effects the
// result carries. Every later call is a recompute, which must leave the
// state of Inner exactly as it found it: it runs from the state the
// forward pass started from, then restores the state it found, so its own
// changes, like its effects, are dropped.
func (c *checkpointed) Forward(x *tensor.Tensor) *tensor.Tensor {
	if !c.training.Load() {
		return c.Inner.Forward(x)
	}
	var before *moduleState // the state the forward pass started from
	return tensor.Checkpoint(func(in ...*tensor.Tensor) *tensor.Tensor {
		if before == nil {
			before = stateOf(c.Inner)
			return c.Inner.Forward(in[0])
		}
		return replay(c.Inner, before, in[0])
	}, x)
}

// replay runs m's forward pass on x from the state from and returns its
// result, leaving m in the state it found it in.
func replay(m Module, from *moduleState, x *tensor.Tensor) *tensor.Tensor {
	found := stateOf(m)
	err := from.restore()
	var y *tensor.Tensor
	if err == nil {
		y = m.Forward(x)
	}
	if rerr := found.restore(); err == nil {
		err = rerr
	}
	if err != nil {
		return tensor.Failed(err)
	}
	return y
}

// moduleState is the state that a forward pass of a module reads and
// changes: the values of its buffers and the masks its dropout layers are
// up to.
type moduleState struct {
	buffers  []*tensor.Param
	values   []*tensor.Tensor
	dropouts []*Dropout
	calls    []uint64
}

// stateOf records the state of m now.
func stateOf(m Module) *moduleState {
	s := &moduleState{}
	w := walker{seen: map[unsafe.Pointer]bool{}}
	w.visit(reflect.ValueOf(m), "", func(_ string, v reflect.Value) bool {
		switch x := v.Interface().(type) {
		case *tensor.Param:
			if !x.Trainable() {
				s.buffers, s.values = append(s.buffers, x), append(s.values, x.Value())
			}
		case *Dropout:
			s.dropouts, s.calls = append(s.dropouts, x), append(s.calls, x.calls.Load())
		}
		return true
	})
	return s
}

// restore puts the state s recorded back, so that a forward pass run after
// it does what one run when s was recorded would.
func (s *moduleState) restore() error {
	for i, p := range s.buffers {
		if err := p.Set(s.values[i]); err != nil {
			return err
		}
	}
	for i, d := range s.dropouts {
		d.calls.Store(s.calls[i])
	}
	return nil
}
