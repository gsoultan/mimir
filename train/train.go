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
	Loss     float64 // mean training loss over the epoch's batches, excluding steps DynamicLossScale skipped
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
	// Skipped reports a step that DynamicLossScale skipped because a
	// gradient was not finite: the parameters are unchanged.
	Skipped bool
	// LossScale is the loss scale after the step: the one LossScale set,
	// or DynamicLossScale's, which the step may have halved or doubled.
	// It is 0 without loss scaling.
	LossScale float64
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
	micro     int     // examples per chunk, or 0 for whole batches
	budget    int64   // the device's memory budget, or 0 for none
	scale     float64 // the loss scale, or 0 for none
	dynamic   bool    // the loss scale adjusts itself
	onEpoch   []func(Epoch) error
	onBatch   []func(Batch) error
	configErr error
}

func (c *config) fail(format string, args ...any) {
	if c.configErr == nil {
		c.configErr = fmt.Errorf("%w: %s", ErrConfig, fmt.Sprintf(format, args...))
	}
}

// Loss sets the loss to minimize, such as loss.CrossEntropy. It is
// required. The loss must be a mean over the examples of the batch, as
// every function of package loss is, for the chunks of MicroBatch and
// MemoryBudget to add up to the batch.
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
			c.fail("early stopping patience %d", patience)
		}
	}
}

// MicroBatch splits every batch into chunks of at most n examples along
// its first dimension and accumulates the gradients of the chunks before
// each optimizer step, so that a batch too large for memory in one pass
// trains as if it ran in one. Each chunk's loss is weighted by its share
// of the batch, so the step and the reported loss are those of the whole
// batch's mean loss (a loss that sums over the batch instead would be
// divided by the number of chunks); only layers whose result depends on
// the batch, such as batch normalization, see each chunk instead. A lazy
// batch, such as the shuffled gather of data.Tensors, is computed once
// before it is split.
func MicroBatch(n int) Option {
	return func(c *config) {
		c.micro = n
		if n < 1 {
			c.fail("micro-batch of %d examples", n)
		}
	}
}

// MemoryBudget limits the memory that the model's device may hold to
// bytes while Fit runs (see tensor.SetMemoryBudget), restoring the
// previous limit when Fit returns, and makes Fit split the batches that
// do not fit: a step refused with tensor.ErrMemoryBudget starts its
// batch over in chunks of half the size, rounding up, and the size
// stays for the rest of training. The chunks accumulate gradients like
// MicroBatch, which sets the size they start from. A chunk of one
// example that does not fit fails Fit with the error, and so does
// anything else Fit computes past the budget: the optimizer step, a
// checkpoint or an evaluation batch, which is not split.
func MemoryBudget(bytes int64) Option {
	return func(c *config) {
		c.budget = bytes
		if bytes < 1 {
			c.fail("memory budget of %d bytes", bytes)
		}
	}
}

// LossScale multiplies the loss by scale before Backward and has the
// optimizer divide the gradients by scale again (see
// optim.Optimizer.SetGradScale), which keeps small gradients from
// underflowing in half precision. Reported losses are unscaled.
func LossScale(scale float64) Option {
	return func(c *config) {
		c.scale, c.dynamic = scale, false
		if !(scale > 0) || math.IsInf(scale, 0) {
			c.fail("loss scale %v", scale)
		}
	}
}

// initialLossScale is where DynamicLossScale starts, and growAfter the
// steps with finite gradients in a row after which it doubles, like
// torch.amp.GradScaler's defaults.
const (
	initialLossScale = 65536
	growAfter        = 2000
)

// DynamicLossScale scales the loss like LossScale, with a scale that
// adjusts itself as training goes: it starts at 65536; a step whose
// gradients are not all finite is skipped, leaving the parameters alone,
// and halves the scale, to no less than 1, below which it would protect
// nothing; and after 2000 steps in a row with finite gradients the scale
// doubles. Batch reports skipped steps and the scale in effect.
func DynamicLossScale() Option {
	return func(c *config) { c.scale, c.dynamic = initialLossScale, true }
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
// or one per chunk of the batch (see MicroBatch and MemoryBudget), then
// updates the parameters with opt.Step. If ds implements SetEpoch(int), as
// data.Tensors does, Fit calls it before each epoch so shuffling repeats
// exactly when training resumes from a checkpoint.
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
	l := &loop{c: &c, dev: deviceOf(model), chunk: c.micro, scale: c.scale}
	if c.budget > 0 {
		for _, p := range nn.NamedParams(model) {
			l.params = append(l.params, p)
		}
		prev := tensor.MemoryBudget(l.dev)
		tensor.SetMemoryBudget(l.dev, c.budget)
		defer tensor.SetMemoryBudget(l.dev, prev)
	}
	if c.scale != 0 {
		defer opt.SetGradScale(opt.GradScale())
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
		stats, err := l.epoch(ctx, model, opt, ds, epoch)
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

// loop is the state of one Fit that carries from step to step.
type loop struct {
	c      *config
	dev    tensor.Device   // where the batches go
	params []*tensor.Param // the model's, under a memory budget
	chunk  int             // examples per chunk, or 0 for whole batches
	scale  float64         // the loss scale, or 0 for none
	finite int             // steps in a row whose gradients were finite
}

func (l *loop) epoch(ctx context.Context, model nn.Module, opt *optim.Optimizer, ds data.Dataset, epoch int) (Epoch, error) {
	began := time.Now()
	c := l.c
	nn.Train(model)
	if c.schedule != nil {
		opt.SetLR(c.schedule(epoch))
	}
	if s, ok := ds.(interface{ SetEpoch(int) }); ok {
		s.SetEpoch(epoch)
	}
	var sum float64
	n, steps := 0, 0
	for b, err := range ds.All(ctx) {
		if err != nil {
			return Epoch{}, err
		}
		v, skipped, err := l.step(ctx, model, opt, b.X.To(l.dev), b.Y.To(l.dev))
		if err != nil {
			return Epoch{}, err
		}
		if !skipped {
			sum += v
			steps++
		}
		for _, f := range c.onBatch {
			if err := f(Batch{Epoch: epoch + 1, Index: n, Loss: v, Skipped: skipped, LossScale: l.scale}); err != nil {
				return Epoch{}, err
			}
		}
		n++
	}
	return Epoch{
		Epoch:    epoch + 1,
		Loss:     sum / float64(max(steps, 1)),
		LR:       opt.LR(),
		Steps:    opt.Steps(),
		Duration: time.Since(began),
	}, nil
}

// step trains on one batch: the loss and gradients of each chunk, then
// one optimizer step. It returns the batch's loss and whether the step
// was skipped (see DynamicLossScale). A chunk refused by the memory
// budget starts the batch over with smaller chunks. A step that fails
// leaves no gradient behind.
func (l *loop) step(ctx context.Context, model nn.Module, opt *optim.Optimizer, x, y *tensor.Tensor) (v float64, skipped bool, err error) {
	defer func() {
		if err != nil {
			opt.ZeroGrad()
		}
	}()
	split := false // the batch did not fit and was split
	for {
		values := l.values()
		if v, err = l.backward(ctx, model, x, y); err == nil {
			break
		}
		if split && !errors.Is(err, tensor.ErrMemoryBudget) {
			_, chunk := l.chunks(x)
			return 0, false, fmt.Errorf("train: chunk of %d examples, after the memory budget refused the batch: %w (%w)", chunk, err, tensor.ErrMemoryBudget)
		}
		if !errors.Is(err, tensor.ErrMemoryBudget) || l.c.budget == 0 {
			return 0, false, err
		}
		// The chunks so far are accumulated again from the start, and
		// the buffers that the refused attempt set to values it would
		// have computed, as batch normalization sets its running
		// statistics, go back to their values before it: a retry that
		// kept them would compute the refused forward pass too.
		opt.ZeroGrad()
		if rerr := l.restore(values); rerr != nil {
			return 0, false, rerr
		}
		if !l.halve(x) {
			return 0, false, err
		}
		split = true
	}
	if l.c.dynamic {
		ok, err := opt.GradsFinite(ctx)
		if err != nil {
			return 0, false, err
		}
		if !ok {
			opt.ZeroGrad()
			l.scale = max(l.scale/2, 1)
			l.finite = 0
			return v, true, nil
		}
	}
	if l.scale != 0 {
		opt.SetGradScale(l.scale)
	}
	if err := opt.Step(ctx); err != nil {
		return 0, false, err
	}
	opt.ZeroGrad()
	if l.c.dynamic {
		if l.finite++; l.finite == growAfter {
			l.scale *= 2
			l.finite = 0
		}
	}
	return v, false, nil
}

// values returns the current value of every parameter, under a memory
// budget, for restore.
func (l *loop) values() []*tensor.Tensor {
	if l.params == nil {
		return nil
	}
	values := make([]*tensor.Tensor, len(l.params))
	for i, p := range l.params {
		values[i] = p.Value()
	}
	return values
}

// restore puts back the values that the parameters had when values was
// taken, where a parameter was set since.
func (l *loop) restore(values []*tensor.Tensor) error {
	for i, p := range l.params {
		if p.Value() != values[i] {
			if err := p.Set(values[i]); err != nil {
				return err
			}
		}
	}
	return nil
}

// backward computes the loss of each chunk of the batch x, y, weighted by
// the chunk's share of the batch and scaled by the loss scale, and its
// gradients, accumulating them. It returns the batch's loss, unscaled.
func (l *loop) backward(ctx context.Context, model nn.Module, x, y *tensor.Tensor) (float64, error) {
	rows, chunk := l.chunks(x)
	if chunk < rows {
		if y.Rank() == 0 || y.Shape()[0] != rows {
			return 0, fmt.Errorf("train: split a batch of %d examples with targets of shape %v: %w", rows, y.Shape(), tensor.ErrShape)
		}
		// A lazy batch, such as a shuffled gather, is computed once, not
		// once per chunk.
		if err := tensor.Realize(ctx, x, y); err != nil {
			return 0, err
		}
	}
	var sum float64
	for lo := 0; ; {
		hi := min(lo+chunk, rows)
		cx, cy := x, y
		if chunk < rows {
			cx, cy = rowsOf(x, lo, hi), rowsOf(y, lo, hi)
		}
		loss := l.c.loss(model.Forward(cx), cy)
		if chunk < rows {
			loss = loss.Mul(tensor.Scalar(float32(hi-lo) / float32(rows)))
		}
		if l.scale != 0 {
			loss = loss.Mul(tensor.Scalar(float32(l.scale)))
		}
		if err := loss.Backward(ctx); err != nil {
			return 0, err
		}
		v, err := loss.Item[float32](ctx) // computed by Backward
		if err != nil {
			return 0, err
		}
		sum += float64(v)
		if lo = hi; lo >= rows {
			break
		}
	}
	if l.scale != 0 {
		sum /= l.scale
	}
	return sum, nil
}

// chunks returns the examples in the batch x and the examples per chunk
// it is trained in: the whole batch unless a smaller chunk size is set.
func (l *loop) chunks(x *tensor.Tensor) (rows, chunk int) {
	if x.Rank() > 0 {
		rows = x.Shape()[0]
	}
	if l.chunk > 0 && l.chunk < rows {
		return rows, l.chunk
	}
	return rows, rows
}

// halve halves the chunk size after a chunk of the batch x did not fit,
// and reports whether that leaves a smaller chunk to try.
func (l *loop) halve(x *tensor.Tensor) bool {
	_, chunk := l.chunks(x)
	if chunk <= 1 {
		return false
	}
	l.chunk = (chunk + 1) / 2
	return true
}

// rowsOf returns examples [lo, hi) of t as a view.
func rowsOf(t *tensor.Tensor, lo, hi int) *tensor.Tensor {
	ranges := make([][2]int, t.Rank())
	for i, d := range t.Shape() {
		ranges[i] = [2]int{0, d}
	}
	ranges[0] = [2]int{lo, hi}
	return t.Shrink(ranges)
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
