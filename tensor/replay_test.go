package tensor_test

import (
	"context"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/backend/cpu"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/nn/loss"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
)

// batchRecorder runs batches on the CPU kernel by kernel and records the
// buffers of each kernel in the last run of each batch.
type batchRecorder struct {
	*cpu.Backend
	runs map[*backend.Batch][][]backend.Buffer
	// moved is, for each run of a batch run before, the buffers that its
	// kernels did not run over in the run before.
	moved [][]backend.Buffer
	ops   [][]backend.Op // the kernels of moved
}

func (r *batchRecorder) RunBatch(ctx context.Context, bt *backend.Batch) error {
	last, seen := r.runs[bt]
	bufs := make([][]backend.Buffer, len(bt.Kernels))
	var moved []backend.Buffer
	var ops []backend.Op
	for i, k := range bt.Kernels {
		bufs[i] = slices.Concat(bt.Outs[i], bt.Ins[i])
		if seen && !slices.Equal(bufs[i], last[i]) {
			ops = append(ops, k.Op)
			for j, b := range bufs[i] {
				if b != last[i][j] && !slices.Contains(moved, b) {
					moved = append(moved, b)
				}
			}
		}
		if err := r.Run(ctx, k, bt.Outs[i], bt.Ins[i]); err != nil {
			return err
		}
	}
	r.runs[bt] = bufs
	if seen {
		r.moved, r.ops = append(r.moved, moved), append(r.ops, ops)
	}
	return nil
}

// TestReplayedStepsRunOverTheSameBuffers checks that the kernels of a
// replayed training step run over the buffers they ran over in the step
// before, but for the loss, which each step returns in a buffer of its
// own: a backend can then keep what it derived from the buffers, such as
// the CUDA graph of a step and the products in it.
func TestReplayedStepsRunOverTheSameBuffers(t *testing.T) {
	for _, c := range []struct {
		name  string
		model func() nn.Module
	}{
		{"mlp", func() nn.Module {
			return nn.NewSequential(nn.Flatten(), nn.NewLinear(28*28, 32), nn.ReLU(), nn.NewLinear(32, 10))
		}},
		{"cnn", func() nn.Module {
			return nn.NewSequential(
				nn.NewConv2d(1, 4, 3, nn.Padding(1)), nn.ReLU(), nn.MaxPool2d(2),
				nn.Flatten(), nn.NewLinear(4*14*14, 10),
			)
		}},
	} {
		t.Run(c.name, func(t *testing.T) {
			be := &batchRecorder{Backend: cpu.New(), runs: map[*backend.Batch][][]backend.Buffer{}}
			defer tensor.SetDevice(be)()
			nn.ManualSeed(1)
			m := c.model()
			opt := optim.AdamW(nn.Parameters(m), optim.LR(1e-3))
			x := tensor.Rand(1, 16, 1, 28, 28)
			y := tensor.FromSlice(make([]int32, 16), 16)
			if err := tensor.Realize(t.Context(), x, y); err != nil {
				t.Fatal(err)
			}
			for step := range 8 {
				// The first steps fill the pool and the optimizer's state.
				if step == 3 {
					be.moved, be.ops = nil, nil
				}
				if err := loss.CrossEntropy(m.Forward(x), y).Backward(t.Context()); err != nil {
					t.Fatal(err)
				}
				if err := opt.Step(t.Context()); err != nil {
					t.Fatal(err)
				}
				opt.ZeroGrad()
			}
			if len(be.moved) == 0 {
				t.Fatal("no batch was replayed")
			}
			for i, moved := range be.moved {
				// The new loss, and the kernels that write and read it.
				if len(moved) > 1 {
					t.Errorf("replayed batch %d: kernels %v ran over %d other buffers than the run before, want the loss's alone", i, be.ops[i], len(moved))
				}
			}
		})
	}
}
