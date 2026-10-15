package train_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/gsoultan/mimir/data"
	"github.com/gsoultan/mimir/metrics"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/nn/loss"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
	"github.com/gsoultan/mimir/train"
)

// blobs returns n points in 2-D around three class centers, labelled by
// center, as a shuffled dataset.
func blobs(t *testing.T, n int, seed uint64, opts ...data.Option) *data.Tensors {
	t.Helper()
	centers := [][2]float32{{-2, -2}, {2, -2}, {0, 2}}
	noise, err := tensor.Randn(seed, n, 2).Mul(tensor.Scalar[float32](0.5)).Data[float32](t.Context())
	if err != nil {
		t.Fatal(err)
	}
	x := make([]float32, 2*n)
	y := make([]int32, n)
	for i := range n {
		c := i % 3
		x[2*i], x[2*i+1], y[i] = centers[c][0]+noise[2*i], centers[c][1]+noise[2*i+1], int32(c)
	}
	d, err := data.FromTensors(tensor.FromSlice(x, n, 2), tensor.FromSlice(y), opts...)
	if err != nil {
		t.Fatal(err)
	}
	return d
}

func model() nn.Module {
	nn.ManualSeed(7)
	return nn.NewSequential(nn.NewLinear(2, 16), nn.ReLU(), nn.NewLinear(16, 3))
}

func TestFitLearnsAndReportsEpochs(t *testing.T) {
	m := model()
	opt := optim.Adam(nn.Parameters(m), optim.LR(0.05))
	var epochs []train.Epoch
	var batches int
	var log strings.Builder
	err := train.Fit(t.Context(), m, opt, blobs(t, 300, 1, data.BatchSize(32), data.Shuffle(3)),
		train.Loss(loss.CrossEntropy),
		train.Epochs(4),
		train.Eval(blobs(t, 150, 2, data.BatchSize(50)), metrics.Accuracy),
		train.Schedule(optim.Cosine(0.05, 0.005, 4)),
		train.OnEpoch(func(e train.Epoch) error { epochs = append(epochs, e); return nil }),
		train.OnBatch(func(train.Batch) error { batches++; return nil }),
		train.Progress(&log),
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(epochs) != 4 || batches != 4*10 || epochs[3].Steps != 40 {
		t.Fatalf("%d epochs, %d batches, %d steps", len(epochs), batches, epochs[3].Steps)
	}
	if epochs[3].Loss >= epochs[0].Loss {
		t.Errorf("loss did not fall: %v then %v", epochs[0].Loss, epochs[3].Loss)
	}
	if acc := epochs[3].Metrics["accuracy"]; acc < 0.95 {
		t.Errorf("held-out accuracy %v, want above 0.95", acc)
	}
	if _, ok := epochs[3].Metrics["val_loss"]; !ok {
		t.Error("no val_loss")
	}
	if epochs[0].LR != 0.05 || epochs[3].LR >= epochs[2].LR {
		t.Errorf("schedule not applied: lr %v ... %v", epochs[0].LR, epochs[3].LR)
	}
	if lines := strings.Count(log.String(), "\n"); lines != 4 || !strings.Contains(log.String(), "accuracy") {
		t.Errorf("progress log:\n%s", log.String())
	}
}

func TestCheckpointResumesExactly(t *testing.T) {
	dir := t.TempDir()
	run := func(epochs int, ckpt string) []float32 {
		m := model()
		opt := optim.AdamW(nn.Parameters(m), optim.LR(0.02))
		opts := []train.Option{train.Loss(loss.CrossEntropy), train.Epochs(epochs)}
		if ckpt != "" {
			opts = append(opts, train.Checkpoint(ckpt))
		}
		if err := train.Fit(t.Context(), m, opt, blobs(t, 90, 1, data.BatchSize(16), data.Shuffle(5)), opts...); err != nil {
			t.Fatal(err)
		}
		w, err := nn.Parameters(m)[0].Value().Data[float32](t.Context())
		if err != nil {
			t.Fatal(err)
		}
		return w
	}
	straight := run(3, "")
	run(2, dir)            // stops after epoch 2, leaving a checkpoint
	resumed := run(3, dir) // a fresh model and optimizer resume for epoch 3
	if !slices.Equal(straight, resumed) {
		t.Error("training resumed from a checkpoint diverged from an uninterrupted run")
	}
	if entries, _ := os.ReadDir(dir); len(entries) != 1 || entries[0].Name() != "checkpoint.safetensors" {
		t.Errorf("checkpoint dir holds %v", entries)
	}
}

func TestEarlyStoppingAndStop(t *testing.T) {
	m := model()
	opt := optim.SGD(nn.Parameters(m), optim.LR(0)) // never improves
	var seen int
	err := train.Fit(t.Context(), m, opt, blobs(t, 30, 1, data.BatchSize(10)),
		train.Loss(loss.CrossEntropy),
		train.Epochs(20),
		train.Eval(blobs(t, 30, 2, data.BatchSize(10)), metrics.Accuracy),
		train.EarlyStopping("accuracy", 2),
		train.OnEpoch(func(train.Epoch) error { seen++; return nil }),
	)
	if err != nil || seen != 3 {
		t.Errorf("early stopping after %d epochs (%v), want 3", seen, err)
	}

	seen = 0
	err = train.Fit(t.Context(), m, opt, blobs(t, 30, 1, data.BatchSize(10)),
		train.Loss(loss.CrossEntropy),
		train.Epochs(5),
		train.OnBatch(func(b train.Batch) error {
			seen++
			if b.Index == 1 {
				return train.ErrStop
			}
			return nil
		}),
	)
	if err != nil || seen != 2 {
		t.Errorf("Stop from OnBatch after %d batches: %v", seen, err)
	}
}

func TestFitErrors(t *testing.T) {
	m := model()
	opt := optim.SGD(nn.Parameters(m))
	ds := blobs(t, 30, 1, data.BatchSize(10))
	if err := train.Fit(t.Context(), m, opt, ds); !errors.Is(err, train.ErrConfig) {
		t.Errorf("no loss: %v", err)
	}
	if err := train.Fit(t.Context(), m, opt, ds, train.Loss(loss.CrossEntropy), train.EarlyStopping("loss", 0)); !errors.Is(err, train.ErrConfig) {
		t.Errorf("zero patience: %v", err)
	}
	if err := train.Fit(t.Context(), m, opt, ds, train.Loss(loss.MSE)); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("loss of mismatched shapes: %v, want ErrShape", err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if err := train.Fit(ctx, m, opt, ds, train.Loss(loss.CrossEntropy)); !errors.Is(err, context.Canceled) {
		t.Errorf("cancelled: %v", err)
	}
	bad := t.TempDir()
	if err := os.WriteFile(filepath.Join(bad, "checkpoint.safetensors"), []byte("not a checkpoint"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := train.Fit(t.Context(), m, opt, ds, train.Loss(loss.CrossEntropy), train.Checkpoint(bad)); err == nil {
		t.Error("resumed from a corrupt checkpoint")
	}
}
