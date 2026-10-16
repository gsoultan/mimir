package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/gsoultan/mimir/io/safetensors"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/nn/loss"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
)

// TrainConfig sets how a model trains.
type TrainConfig struct {
	Batch    int     // sequences per step
	Steps    int     // optimizer steps in all, counting those a resumed checkpoint took
	LR       float64 // peak learning rate
	MinLR    float64 // learning rate at the end of the cosine decay
	Warmup   int     // steps of linear warmup from LR/Warmup to LR
	Clip     float64 // largest gradient norm; 0 does not clip
	Decay    float64 // AdamW's weight decay
	Half     tensor.DType
	Seed     uint64
	Log      int // steps between progress lines
	EvalIter int // validation batches per progress line; 0 skips validation
}

// LRAt returns the learning rate of step, counted from 0: a linear warmup
// over Warmup steps, then half a cosine from LR down to MinLR at Steps.
func (c TrainConfig) LRAt(step int) float64 {
	if step < c.Warmup {
		return optim.Warmup(c.LR, 0, c.Warmup)(step + 1)
	}
	return optim.Cosine(c.LR, c.MinLR, max(c.Steps-c.Warmup, 1))(step - c.Warmup)
}

// Initial loss scale for float16 and the steps with finite gradients in a
// row after which it doubles, as torch.amp.GradScaler does.
const (
	initialLossScale = 1 << 16
	growAfter        = 2000
)

var (
	errConfig     = errors.New("gpt2: invalid training configuration")
	errDiverged   = errors.New("gpt2: training diverged")
	errCheckpoint = errors.New("gpt2: checkpoint does not fit")
)

// Trainer trains a GPT on a stream of tokens.
type Trainer struct {
	Model *GPT
	Opt   *optim.Optimizer
	Cfg   TrainConfig

	train, val []int32
	rng        *rand.Rand
	scale      float64 // loss scale, for float16; 0 for none
	finite     int     // steps with finite gradients since the scale changed
}

// NewTrainer returns a trainer of model over tokens, the last tenth of
// which it holds out for validation.
func NewTrainer(model *GPT, tokens []int32, cfg TrainConfig) (*Trainer, error) {
	n := len(tokens) * 9 / 10
	if cfg.EvalIter == 0 {
		n = len(tokens)
	}
	if n <= model.Config.Context || cfg.EvalIter > 0 && len(tokens)-n <= model.Config.Context {
		return nil, fmt.Errorf("gpt2: %d tokens are too few for a context of %d: %w", len(tokens), model.Config.Context, tensor.ErrShape)
	}
	if cfg.Batch <= 0 || cfg.Steps <= 0 || cfg.Warmup < 0 || !(cfg.LR > 0) || cfg.Clip < 0 {
		return nil, fmt.Errorf("gpt2: training configuration %+v: %w", cfg, errConfig)
	}
	opts := []optim.Option{optim.LR(cfg.LR), optim.WeightDecay(cfg.Decay), optim.Betas(0.9, 0.95)}
	if cfg.Half != tensor.Float32 {
		if err := nn.Cast(model, cfg.Half); err != nil {
			return nil, err
		}
		opts = append(opts, optim.MasterWeights())
	}
	t := &Trainer{
		Model: model,
		Opt:   optim.AdamW(nn.Parameters(model), opts...),
		Cfg:   cfg,
		train: tokens[:n], val: tokens[n:],
		rng: rand.New(rand.NewPCG(cfg.Seed, 0x9e3779b97f4a7c15)),
	}
	if cfg.Half == tensor.Float16 {
		t.scale = initialLossScale
	}
	return t, nil
}

// batch returns Batch random windows of the context's length from
// tokens, and the tokens that follow each position, both [Batch, Context].
func (t *Trainer) batch(tokens []int32, rng *rand.Rand) (x, y *tensor.Tensor) {
	b, s := t.Cfg.Batch, t.Model.Config.Context
	xs, ys := make([]int32, b*s), make([]int32, b*s)
	for i := range b {
		at := rng.IntN(len(tokens) - s)
		copy(xs[i*s:], tokens[at:at+s])
		copy(ys[i*s:], tokens[at+1:at+s+1])
	}
	dev := t.Model.Tok.Weight.Value().Device()
	return tensor.FromSlice(xs, b, s).To(dev), tensor.FromSlice(ys, b, s).To(dev)
}

// lossOf returns the mean cross-entropy of the model's predictions of y
// from x.
func (t *Trainer) lossOf(x, y *tensor.Tensor) *tensor.Tensor {
	logits := t.Model.Forward(x)
	v := t.Model.Config.Vocab
	return loss.CrossEntropy(logits.Reshape(-1, v).Cast(tensor.Float32), y.Reshape(-1))
}

// gradNorm returns the norm of all the gradients together, which is
// infinite or NaN if a gradient is.
func (t *Trainer) gradNorm(ctx context.Context) (float64, error) {
	var sum *tensor.Tensor
	for _, p := range nn.Parameters(t.Model) {
		g := p.Grad()
		if g == nil {
			continue
		}
		g = g.Cast(tensor.Float32)
		if s := g.Mul(g).Sum(); sum == nil {
			sum = s
		} else {
			sum = sum.Add(s)
		}
	}
	if sum == nil {
		return 0, nil
	}
	v, err := sum.Item[float32](ctx)
	return math.Sqrt(float64(v)), err
}

// Step takes one optimizer step on a random batch and returns its loss,
// the gradients' norm before clipping and whether the step was skipped
// because a float16 gradient overflowed.
func (t *Trainer) Step(ctx context.Context) (lossv, norm float64, skipped bool, err error) {
	defer func() {
		if err != nil || skipped {
			t.Opt.ZeroGrad()
		}
	}()
	t.Opt.SetLR(t.Cfg.LRAt(t.Opt.Steps()))
	x, y := t.batch(t.train, t.rng)
	l := t.lossOf(x, y)
	if t.scale != 0 {
		l = l.Mul(tensor.Scalar(float32(t.scale)))
	}
	if err := l.Backward(ctx); err != nil {
		return 0, 0, false, err
	}
	v, err := l.Item[float32](ctx) // computed by Backward
	if err != nil {
		return 0, 0, false, err
	}
	lossv = float64(v)
	if norm, err = t.gradNorm(ctx); err != nil {
		return 0, 0, false, err
	}
	scale := max(t.scale, 1)
	lossv, norm = lossv/scale, norm/scale
	if math.IsNaN(norm) || math.IsInf(norm, 0) {
		if t.scale == 0 {
			return lossv, norm, false, fmt.Errorf("gpt2: step %d: gradient norm %v: %w", t.Opt.Steps(), norm, errDiverged)
		}
		t.scale, t.finite = max(t.scale/2, 1), 0
		return lossv, norm, true, nil
	}
	// The optimizer divides the gradients by the factor they carry: the
	// loss scale, times the factor that brings their norm down to Clip.
	clip := 1.0
	if t.Cfg.Clip > 0 && norm > t.Cfg.Clip {
		clip = norm / t.Cfg.Clip
	}
	t.Opt.SetGradScale(scale * clip)
	if err := t.Opt.Step(ctx); err != nil {
		return 0, 0, false, err
	}
	t.Opt.ZeroGrad()
	if t.scale != 0 {
		if t.finite++; t.finite == growAfter {
			t.scale, t.finite = t.scale*2, 0
		}
	}
	return lossv, norm, false, nil
}

// Validate returns the mean loss over EvalIter batches of the held-out
// tokens, the same batches every time.
func (t *Trainer) Validate(ctx context.Context) (float64, error) {
	rng := rand.New(rand.NewPCG(t.Cfg.Seed+1, 1))
	var sum float64
	for range t.Cfg.EvalIter {
		x, y := t.batch(t.val, rng)
		v, err := t.lossOf(x, y).Detach().Item[float32](ctx)
		if err != nil {
			return 0, err
		}
		sum += float64(v)
	}
	return sum / float64(t.Cfg.EvalIter), nil
}

// Run trains until Steps, printing progress to w every Log steps and
// saving a checkpoint to path, if it is not empty, at every progress line
// and at the end.
func (t *Trainer) Run(ctx context.Context, w io.Writer, path string) error {
	tokensPerStep := t.Cfg.Batch * t.Model.Config.Context
	start, done := time.Now(), 0
	var sum float64
	n := 0
	for t.Opt.Steps() < t.Cfg.Steps {
		if err := ctx.Err(); err != nil {
			return err
		}
		lr := t.Cfg.LRAt(t.Opt.Steps())
		v, norm, skipped, err := t.Step(ctx)
		if err != nil {
			return err
		}
		done++
		if skipped {
			fmt.Fprintf(w, "step %d: gradients overflowed in float16; loss scale now %g\n", t.Opt.Steps(), t.scale)
			continue
		}
		sum += v
		n++
		step := t.Opt.Steps()
		if step%t.Cfg.Log != 0 && step != t.Cfg.Steps {
			continue
		}
		elapsed := time.Since(start)
		line := fmt.Sprintf("step %5d/%d  loss %.4f  lr %.2e  norm %.2f  %.0f tokens/s",
			step, t.Cfg.Steps, sum/float64(n), lr, norm, float64(done*tokensPerStep)/elapsed.Seconds())
		if t.Cfg.EvalIter > 0 {
			val, err := t.Validate(ctx)
			if err != nil {
				return err
			}
			line += fmt.Sprintf("  val %.4f", val)
		}
		fmt.Fprintln(w, line)
		if path != "" {
			if err := t.Save(ctx, path); err != nil {
				return err
			}
		}
		sum, n, done, start = 0, 0, 0, time.Now()
	}
	return nil
}

// Save writes the model, the optimizer's state and the progress to a
// safetensors file at path, replacing it atomically.
func (t *Trainer) Save(ctx context.Context, path string) error {
	ts := map[string]safetensors.Tensor{}
	for name, p := range nn.NamedParams(t.Model) {
		st, err := nn.Encode(ctx, p.Value())
		if err != nil {
			return fmt.Errorf("gpt2: checkpoint %s: %w", name, err)
		}
		ts["model."+name] = st
	}
	state, steps := t.Opt.Snapshot()
	for key, v := range state {
		st, err := nn.Encode(ctx, v)
		if err != nil {
			return fmt.Errorf("gpt2: checkpoint optimizer %s: %w", key, err)
		}
		ts["optim."+key] = st
	}
	meta := configMeta(t.Model.Config)
	meta["steps"] = strconv.Itoa(steps)
	meta["loss_scale"] = strconv.FormatFloat(t.scale, 'g', -1, 64)
	return writeAtomic(path, ts, meta)
}

// Resume loads the model, the optimizer's state and the progress from a
// checkpoint at path that Save wrote for a model of the same shape. It
// returns false, and changes nothing, if there is no file at path.
func (t *Trainer) Resume(path string) (bool, error) {
	f, err := safetensors.Open(path)
	if errors.Is(err, fs.ErrNotExist) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	defer f.Close()
	c, err := metaConfig(f.Metadata)
	if err != nil {
		return false, err
	}
	if c != t.Model.Config {
		return false, fmt.Errorf("gpt2: checkpoint %s holds a model of %+v, not %+v: %w", path, c, t.Model.Config, errCheckpoint)
	}
	steps, errS := strconv.Atoi(f.Metadata["steps"])
	scale, errL := strconv.ParseFloat(f.Metadata["loss_scale"], 64)
	if err := errors.Join(errS, errL); err != nil {
		return false, fmt.Errorf("gpt2: checkpoint %s: %w", path, err)
	}
	if err := nn.LoadPrefixed(f, "model.", t.Model); err != nil {
		return false, err
	}
	state := map[string]*tensor.Tensor{}
	for _, name := range f.Names() {
		key, ok := strings.CutPrefix(name, "optim.")
		if !ok {
			continue
		}
		st, _ := f.Tensor(name)
		if state[key], err = nn.Decode(st); err != nil {
			return false, err
		}
	}
	if err := t.Opt.Restore(state, steps); err != nil {
		return false, err
	}
	if t.scale != 0 {
		t.scale = max(scale, 1)
	}
	return true, nil
}

// LoadGPT returns the model a checkpoint at path holds, in float32, on
// the default device.
func LoadGPT(path string) (*GPT, error) {
	f, err := safetensors.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	c, err := metaConfig(f.Metadata)
	if err != nil {
		return nil, err
	}
	g, err := NewGPT(c, 0)
	if err != nil {
		return nil, err
	}
	// Load casts the weights of a model trained in half precision to
	// float32.
	if err := nn.LoadPrefixed(f, "model.", g); err != nil {
		return nil, err
	}
	return g, nil
}

func configMeta(c Config) map[string]string {
	return map[string]string{
		"vocab": strconv.Itoa(c.Vocab), "context": strconv.Itoa(c.Context),
		"embed": strconv.Itoa(c.Embed), "heads": strconv.Itoa(c.Heads), "layers": strconv.Itoa(c.Layers),
		"tokenizer": "bytes",
	}
}

func metaConfig(m map[string]string) (Config, error) {
	var c Config
	var errs []error
	for key, dst := range map[string]*int{"vocab": &c.Vocab, "context": &c.Context, "embed": &c.Embed, "heads": &c.Heads, "layers": &c.Layers} {
		v, err := strconv.Atoi(m[key])
		errs = append(errs, err)
		*dst = v
	}
	if m["tokenizer"] != "bytes" {
		errs = append(errs, fmt.Errorf("tokenizer %q", m["tokenizer"]))
	}
	if err := errors.Join(errs...); err != nil {
		return c, fmt.Errorf("gpt2: checkpoint metadata %v: %w: %w", m, errCheckpoint, err)
	}
	return c, c.validate()
}

// writeAtomic writes a safetensors file at path through a temporary file
// in the same directory, renamed over path once complete.
func writeAtomic(path string, ts map[string]safetensors.Tensor, meta map[string]string) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".gpt2-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name()) // no-op after the rename
	err = safetensors.Write(tmp, ts, meta)
	if serr := tmp.Sync(); err == nil {
		err = serr
	}
	if cerr := tmp.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return fmt.Errorf("gpt2: checkpoint: %w", err)
	}
	return os.Rename(tmp.Name(), path)
}
