// Package automl picks the best model for a table and tunes it: it tries
// boosted trees, random forests and linear models with a range of
// settings, scores each on rows it didn't learn from, and fits the best
// to the whole table, as AutoGluon and FLAML do for tables.
//
//	r, err := automl.Search(ctx, d, tabular.Classification, automl.Classes(3))
//	if err != nil { ... }
//	p := r.Best.Predict(rows) // the winner, fit to every row of d
//	for _, t := range r.Trials { fmt.Println(t.Name, t.Score) }
//
// Classification is scored by log loss and regression by RMSE, lower
// better, with k-fold cross-validation on small tables and one held-out
// fifth on large ones.
package automl

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"time"

	"github.com/gsoultan/mimir/linear"
	"github.com/gsoultan/mimir/metrics"
	"github.com/gsoultan/mimir/tabular"
	"github.com/gsoultan/mimir/tensor"
	"github.com/gsoultan/mimir/tree"
)

// Option configures Search.
type Option func(*config)

type config struct {
	trials   int
	folds    int
	classes  int
	seed     uint64
	budget   time.Duration
	families []string
	progress func(done, total int)
	workers  int
}

// Trials sets how many candidates Search scores, the defaults of each
// family first, then boosted trees with random settings. The default is
// 12.
func Trials(n int) Option { return func(c *config) { c.trials = n } }

// Folds sets the folds of cross-validation on tables of up to 20,000
// rows; larger ones hold out a fifth. The default is 3.
func Folds(k int) Option { return func(c *config) { c.folds = k } }

// Classes sets the number of classes of a classification, as
// tree.Classes does. The default counts them in d.Y.
func Classes(k int) Option { return func(c *config) { c.classes = k } }

// Seed seeds the folds and the random settings. The default is 1.
func Seed(s uint64) Option { return func(c *config) { c.seed = s } }

// Budget stops trying candidates once this much time has passed; the
// best so far wins. The default is no limit.
func Budget(d time.Duration) Option { return func(c *config) { c.budget = d } }

// Families limits the candidates to these families: "boost", "forest"
// and "linear". The default is all three.
func Families(names ...string) Option { return func(c *config) { c.families = names } }

// Progress is told how many candidates are scored, of how many, after
// each one.
func Progress(f func(done, total int)) Option { return func(c *config) { c.progress = f } }

// Workers sets the goroutines each fit runs on. The default is
// GOMAXPROCS.
func Workers(n int) Option { return func(c *config) { c.workers = n } }

// Candidate is a family and its settings.
type Candidate struct {
	Family string // "boost", "forest" or "linear"
	// Settings, by the name of the option that takes them: LearningRate,
	// MaxDepth, MinLeaf, L2, Subsample and ColSample for boosted trees,
	// MinLeaf for forests, L2 for linear models.
	Settings map[string]float64
}

// String names the candidate and its settings.
func (c Candidate) String() string {
	keys := make([]string, 0, len(c.Settings))
	for k := range c.Settings {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	s := c.Family
	for _, k := range keys {
		s += fmt.Sprintf(" %s=%g", k, c.Settings[k])
	}
	return s
}

// Trial is a candidate and how it scored.
type Trial struct {
	Candidate
	Score  float64 // log loss or RMSE on held-out rows; lower is better
	Millis int64
	Err    error // why it couldn't be fit, if it couldn't
}

// Result is what Search found.
type Result struct {
	// Best is the best candidate fit to every row: a *tree.Model or a
	// *linear.Model.
	Best tabular.Model
	// Winner is Best's candidate and score.
	Winner Trial
	// Trials are every candidate scored, best first.
	Trials []Trial
	// Metric names the score: "log loss" or "rmse".
	Metric string
}

// Search finds the best model of the candidates for d answering task.
func Search(ctx context.Context, d tabular.Data, task tabular.Task, opts ...Option) (*Result, error) {
	c := &config{trials: 12, folds: 3, seed: 1, families: []string{"boost", "forest", "linear"}}
	for _, o := range opts {
		o(c)
	}
	if err := d.Check(task); err != nil {
		return nil, err
	}
	if c.trials < 1 || c.folds < 2 {
		return nil, fmt.Errorf("automl: %d trials of %d folds: %w", c.trials, c.folds, tensor.ErrShape)
	}
	if task == tabular.Classification && c.classes == 0 {
		c.classes = d.Classes()
	}
	cands := candidates(c)
	if len(cands) == 0 {
		return nil, fmt.Errorf("automl: no candidates of families %v: %w", c.families, tensor.ErrShape)
	}
	score, metric := tabular.Score(metrics.LogLoss), "log loss"
	if task == tabular.Regression {
		score, metric = metrics.RMSE, "rmse"
	}
	start := time.Now()
	var trials []Trial
	for i, cand := range cands {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if c.budget > 0 && i > 0 && time.Since(start) > c.budget {
			break
		}
		t0 := time.Now()
		tr := Trial{Candidate: cand}
		s, err := evaluate(ctx, d, c, fitter(cand, task, c), score)
		if err != nil {
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			tr.Err, tr.Score = err, math.Inf(1)
		} else {
			tr.Score = s
		}
		tr.Millis = time.Since(t0).Milliseconds()
		trials = append(trials, tr)
		if c.progress != nil {
			c.progress(i+1, len(cands))
		}
	}
	slices.SortStableFunc(trials, func(a, b Trial) int {
		switch {
		case a.Score < b.Score:
			return -1
		case a.Score > b.Score:
			return 1
		}
		return 0
	})
	if trials[0].Err != nil {
		return nil, fmt.Errorf("automl: no candidate could be fit: %w", trials[0].Err)
	}
	best, err := fitter(trials[0].Candidate, task, c)(ctx, d)
	if err != nil {
		return nil, err
	}
	return &Result{Best: best, Winner: trials[0], Trials: trials, Metric: metric}, nil
}

// evaluate scores fit on d: the mean over folds, or on a held-out fifth
// of a large table.
func evaluate(ctx context.Context, d tabular.Data, c *config, fit tabular.Fit, score tabular.Score) (float64, error) {
	if d.Rows() > 20_000 {
		train, held := d.Split(0.2, c.seed)
		m, err := fit(ctx, train)
		if err != nil {
			return 0, err
		}
		return score(m.Predict(held.X), held.Y), nil
	}
	scores, err := tabular.CrossValidate(ctx, d, min(c.folds, d.Rows()), c.seed, fit, score)
	if err != nil {
		return 0, err
	}
	var s float64
	for _, v := range scores {
		if math.IsNaN(v) {
			return 0, errors.New("automl: a fold scored NaN")
		}
		s += v
	}
	return s / float64(len(scores)), nil
}

// candidates returns the defaults of each family, then boosted trees
// with random settings, up to c.trials of them.
func candidates(c *config) []Candidate {
	has := func(f string) bool { return slices.Contains(c.families, f) }
	var out []Candidate
	if has("boost") {
		out = append(out, Candidate{Family: "boost", Settings: map[string]float64{"LearningRate": 0.05, "MaxDepth": 6, "MinLeaf": 5, "L2": 1}})
	}
	if has("forest") {
		out = append(out, Candidate{Family: "forest", Settings: map[string]float64{"MinLeaf": 2}})
	}
	if has("linear") {
		out = append(out, Candidate{Family: "linear", Settings: map[string]float64{}})
	}
	r := rand.New(rand.NewPCG(c.seed, 0x6175746f))
	pick := func(v ...float64) float64 { return v[r.IntN(len(v))] }
	for tries := 0; len(out) < c.trials && tries < 10*c.trials; tries++ {
		var cand Candidate
		switch {
		case has("boost"):
			cand = Candidate{Family: "boost", Settings: map[string]float64{
				"LearningRate": pick(0.02, 0.05, 0.1, 0.2),
				"MaxDepth":     pick(3, 4, 6, 8, 10),
				"MinLeaf":      pick(1, 5, 20, 50),
				"L2":           pick(0, 1, 5, 20),
				"Subsample":    pick(0.7, 0.85, 1),
				"ColSample":    pick(0.5, 0.8, 1),
			}}
		case has("forest"):
			cand = Candidate{Family: "forest", Settings: map[string]float64{"MinLeaf": pick(1, 2, 5, 10)}}
		case has("linear"):
			cand = Candidate{Family: "linear", Settings: map[string]float64{"L2": pick(1e-4, 1e-2, 1, 10)}}
		}
		if !slices.ContainsFunc(out, func(o Candidate) bool { return o.String() == cand.String() }) {
			out = append(out, cand)
		}
	}
	return out[:min(len(out), c.trials)]
}

// fitter returns a Fit for cand.
func fitter(cand Candidate, task tabular.Task, c *config) tabular.Fit {
	s := cand.Settings
	return func(ctx context.Context, d tabular.Data) (tabular.Model, error) {
		switch cand.Family {
		case "linear":
			var opts []linear.Option
			if v, ok := s["L2"]; ok {
				opts = append(opts, linear.L2(v))
			}
			if c.workers > 0 {
				opts = append(opts, linear.Workers(c.workers))
			}
			if task == tabular.Regression {
				if _, ok := s["L2"]; !ok {
					opts = append(opts, linear.L2(1e-4))
				}
				return linear.Regression(ctx, d, opts...)
			}
			return linear.Logistic(ctx, d, append(opts, linear.Classes(c.classes))...)
		case "forest":
			opts := []tree.Option{tree.Trees(200), tree.MinLeaf(int(s["MinLeaf"]))}
			return tree.RandomForest(ctx, d, task, append(opts, treeOpts(task, c)...)...)
		default:
			opts := []tree.Option{
				tree.Trees(1000), tree.LearningRate(s["LearningRate"]), tree.MaxDepth(int(s["MaxDepth"])),
				tree.MinLeaf(int(s["MinLeaf"])), tree.L2(s["L2"]),
			}
			if v, ok := s["Subsample"]; ok {
				opts = append(opts, tree.Subsample(v))
			}
			if v, ok := s["ColSample"]; ok {
				opts = append(opts, tree.ColSample(v))
			}
			train := d
			if d.Rows() >= 50 {
				// Stop adding trees once a tenth held back stops improving.
				var val tabular.Data
				train, val = d.Split(0.1, c.seed+7)
				opts = append(opts, tree.Validation(val, 50))
			}
			return tree.GradientBoosting(ctx, train, task, append(opts, treeOpts(task, c)...)...)
		}
	}
}

func treeOpts(task tabular.Task, c *config) []tree.Option {
	var opts []tree.Option
	if task == tabular.Classification {
		opts = append(opts, tree.Classes(c.classes))
	}
	if c.workers > 0 {
		opts = append(opts, tree.Workers(c.workers))
	}
	return opts
}
