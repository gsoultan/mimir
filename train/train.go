// Package train runs training loops: Fit trains a model for some epochs,
// evaluating, scheduling the learning rate, checkpointing and stopping
// early as asked.
//
//	err := train.Fit(ctx, model, opt, trainSet,
//		train.Loss(loss.CrossEntropy),
//		train.Epochs(5),
//		train.Eval(testSet, metrics.Accuracy),
//		train.Checkpoint("ckpt/"))
package train

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"strings"
	"time"

	"github.com/gsoultan/mimir/data"
	"github.com/gsoultan/mimir/metrics"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/nn/loss"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
)

// ErrStop, returned by an OnEpoch or OnBatch callback, ends training without
// an error.
var ErrStop = errors.New("train: stop")

// ErrConfig reports an invalid option.
var ErrConfig = errors.New("train: invalid option")

// Epoch summarizes one epoch of training.
type Epoch struct {
	Epoch    int     // counted from 1
	Loss     float64 // mean training loss over the epoch's batches
	LR       float64 // learning rate during the epoch
	Steps    int     // optimizer steps taken since training began
	Duration time.Duration
	// Metrics holds the evaluation results by name: "val_loss" and each
	// metric's Name. It is empty without Eval.
	Metrics map[string]float64
}

// Batch describes one training step.
type Batch struct {
	Epoch int // counted from 1
	Index int // within the epoch, from 0
	Loss  float64
}

// Option configures Fit.
type Option func(*config)

type config struct {
	loss      loss.Func
	epochs    int
	eval      data.Dataset
	metrics   []func() metrics.Metric
	ckptDir   string
	schedule  optim.Schedule
	monitor   string
	patience  int
	onEpoch   []func(Epoch) error
	onBatch   []func(Batch) error
	configErr error
}

// Loss sets the loss to minimize, such as loss.CrossEntropy. It is
// required.
func Loss(f loss.Func) Option { return func(c *config) { c.loss = f } }

// Epochs sets the number of passes over the training data. The default
// is 1.
func Epochs(n int) Option { return func(c *config) { c.epochs = n } }

// Eval evaluates the model on ds after every epoch, reporting the mean
// loss as "val_loss" and each metric under its name.
func Eval(ds data.Dataset, ms ...func() metrics.Metric) Option {
	return func(c *config) { c.eval, c.metrics = ds, ms }
}

// Checkpoint saves the model, the optimizer and the epoch to
// dir/checkpoint.safetensors after every epoch, and resumes from that file
// when Fit starts and finds it.
func Checkpoint(dir string) Option { return func(c *config) { c.ckptDir = dir } }

// Schedule sets the learning rate at the start of every epoch.
func Schedule(s optim.Schedule) Option { return func(c *config) { c.schedule = s } }

// EarlyStopping stops training when the monitored value has not improved
// for patience epochs. monitor is "loss", "val_loss" or a metric name;
// values whose name contains "loss" improve by falling, others by rising.
func EarlyStopping(monitor string, patience int) Option {
	return func(c *config) {
		c.monitor, c.patience = monitor, patience
		if patience < 1 {
			c.configErr = fmt.Errorf("%w: early stopping patience %d", ErrConfig, patience)
		}
	}
}

// OnEpoch calls f after every epoch. Returning ErrStop ends training; any
// other error ends it with that error.
func OnEpoch(f func(Epoch) error) Option {
	return func(c *config) { c.onEpoch = append(c.onEpoch, f) }
}

// OnBatch calls f after every training step, as OnEpoch does.
func OnBatch(f func(Batch) error) Option {
	return func(c *config) { c.onBatch = append(c.onBatch, f) }
}

// Progress prints one line per epoch to w.
func Progress(w io.Writer) Option {
	return OnEpoch(func(e Epoch) error {
		var b strings.Builder
		fmt.Fprintf(&b, "epoch %d: loss %.4f", e.Epoch, e.Loss)
		for _, k := range sortedKeys(e.Metrics) {
			fmt.Fprintf(&b, ", %s %.4f", k, e.Metrics[k])
		}
		fmt.Fprintf(&b, " (lr %.3g, %s)\n", e.LR, e.Duration.Round(time.Millisecond))
		_, err := io.WriteString(w, b.String())
		return err
	})
}

// Fit trains model by minimizing the loss over ds with opt.
//
// Each step computes the loss and gradients in one schedule with Backward,
// then updates the parameters with opt.Step. If ds implements
// SetEpoch(int), as data.Tensors does, Fit calls it before each epoch so
// shuffling repeats exactly when training resumes from a checkpoint.
func Fit(ctx context.Context, model nn.Module, opt *optim.Optimizer, ds data.Dataset, opts ...Option) error {
	c := config{epochs: 1}
	for _, o := range opts {
		o(&c)
	}
	switch {
	case c.configErr != nil:
		return c.configErr
	case c.loss == nil:
		return fmt.Errorf("%w: no Loss", ErrConfig)
	case c.epochs < 0:
		return fmt.Errorf("%w: %d epochs", ErrConfig, c.epochs)
	}

	start := 0
	if c.ckptDir != "" {
		var err error
		if start, err = resume(c.ckptDir, model, opt); err != nil {
			return err
		}
	}
	stopper := newStopper(c.monitor, c.patience)
	for epoch := start; epoch < c.epochs; epoch++ {
		stats, err := runEpoch(ctx, model, opt, ds, &c, epoch)
		if err != nil {
			return stopped(err)
		}
		if c.eval != nil {
			if stats.Metrics, err = Evaluate(ctx, model, c.eval, c.loss, c.metrics...); err != nil {
				return err
			}
		}
		if c.ckptDir != "" {
			if err := save(ctx, c.ckptDir, model, opt, epoch+1); err != nil {
				return err
			}
		}
		for _, f := range c.onEpoch {
			if err := f(stats); err != nil {
				return stopped(err)
			}
		}
		if stop, err := stopper.observe(stats); err != nil || stop {
			return err
		}
	}
	return nil
}

// deviceOf returns the device that holds model's parameters, where its
// batches go, or the default device if it has none.
func deviceOf(model nn.Module) tensor.Device {
	for _, p := range nn.NamedParams(model) {
		return p.Value().Device()
	}
	return tensor.DefaultDevice()
}

func stopped(err error) error {
	if errors.Is(err, ErrStop) {
		return nil
	}
	return err
}

func runEpoch(ctx context.Context, model nn.Module, opt *optim.Optimizer, ds data.Dataset, c *config, epoch int) (Epoch, error) {
	began := time.Now()
	nn.Train(model)
	if c.schedule != nil {
		opt.SetLR(c.schedule(epoch))
	}
	if s, ok := ds.(interface{ SetEpoch(int) }); ok {
		s.SetEpoch(epoch)
	}
	var sum float64
	n := 0
	dev := deviceOf(model)
	for b, err := range ds.All(ctx) {
		if err != nil {
			return Epoch{}, err
		}
		l := c.loss(model.Forward(b.X.To(dev)), b.Y.To(dev))
		if err := l.Backward(ctx); err != nil {
			return Epoch{}, err
		}
		if err := opt.Step(ctx); err != nil {
			return Epoch{}, err
		}
		opt.ZeroGrad()
		v, err := l.Item[float32](ctx) // computed by Backward
		if err != nil {
			return Epoch{}, err
		}
		sum += float64(v)
		for _, f := range c.onBatch {
			if err := f(Batch{Epoch: epoch + 1, Index: n, Loss: float64(v)}); err != nil {
				return Epoch{}, err
			}
		}
		n++
	}
	return Epoch{
		Epoch:    epoch + 1,
		Loss:     sum / float64(max(n, 1)),
		LR:       opt.LR(),
		Steps:    opt.Steps(),
		Duration: time.Since(began),
	}, nil
}

// Evaluate runs model in inference mode over ds and returns the mean loss
// as "val_loss", if lossFn is not nil, and each metric under its name. The
// model is left in training mode.
func Evaluate(ctx context.Context, model nn.Module, ds data.Dataset, lossFn loss.Func, ms ...func() metrics.Metric) (map[string]float64, error) {
	nn.Eval(model)
	defer nn.Train(model)
	live := make([]metrics.Metric, len(ms))
	for i, m := range ms {
		live[i] = m()
	}
	var sum float64
	n := 0
	dev := deviceOf(model)
	for b, err := range ds.All(ctx) {
		if err != nil {
			return nil, err
		}
		b.X, b.Y = b.X.To(dev), b.Y.To(dev)
		pred := model.Forward(b.X).Detach()
		if lossFn != nil {
			v, err := lossFn(pred, b.Y).Item[float32](ctx)
			if err != nil {
				return nil, err
			}
			sum += float64(v)
		}
		for _, m := range live {
			if err := m.Update(ctx, pred, b.Y); err != nil {
				return nil, fmt.Errorf("train: %s: %w", m.Name(), err)
			}
		}
		n++
	}
	out := make(map[string]float64, len(live)+1)
	if lossFn != nil {
		out["val_loss"] = sum / float64(max(n, 1))
	}
	for _, m := range live {
		out[m.Name()] = m.Value()
	}
	return out, nil
}

// stopper implements EarlyStopping.
type stopper struct {
	monitor  string
	patience int
	best     float64
	since    int
}

func newStopper(monitor string, patience int) *stopper {
	return &stopper{monitor: monitor, patience: patience, best: math.NaN()}
}

func (s *stopper) observe(e Epoch) (bool, error) {
	if s.monitor == "" {
		return false, nil
	}
	v, ok := e.Metrics[s.monitor]
	if s.monitor == "loss" {
		v, ok = e.Loss, true
	}
	if !ok {
		return false, fmt.Errorf("%w: early stopping monitors %q, which is not reported", ErrConfig, s.monitor)
	}
	better := v > s.best
	if strings.Contains(s.monitor, "loss") {
		better = v < s.best
	}
	if math.IsNaN(s.best) || better {
		s.best, s.since = v, 0
		return false, nil
	}
	s.since++
	return s.since >= s.patience, nil
}
