package tree

import (
	"fmt"
	"math"

	"github.com/gsoultan/mimir/tabular"
	"github.com/gsoultan/mimir/tensor"
)

// Option configures GradientBoosting, RandomForest and DecisionTree.
type Option func(*config)

type config struct {
	trees     int
	rate      float64
	maxDepth  int
	minLeaf   int
	minWeight float64
	minGain   float64
	l2        float64
	subsample float64
	colsample float64
	perNode   int
	bins      int
	classes   int
	seed      uint64
	workers   int
	val       *tabular.Data
	patience  int
	bootstrap bool
	progress  func(done, total int)
}

// Trees sets how many trees to grow: boosting rounds, or forest trees.
// The default is 100.
func Trees(n int) Option { return func(c *config) { c.trees = n } }

// LearningRate sets how much each boosted tree counts. The default is 0.1.
func LearningRate(r float64) Option { return func(c *config) { c.rate = r } }

// MaxDepth limits how many splits lie between a tree's root and its
// leaves. The default is 6 for boosting and no limit for forests and
// single trees.
func MaxDepth(d int) Option { return func(c *config) { c.maxDepth = d } }

// MinLeaf sets the fewest rows a leaf may hold. The default is 1.
func MinLeaf(n int) Option { return func(c *config) { c.minLeaf = n } }

// MinWeight sets the smallest sum of hessians a boosted leaf may hold, as
// XGBoost's min_child_weight. The default is 1.
func MinWeight(w float64) Option { return func(c *config) { c.minWeight = w } }

// MinGain sets the gain below which a node is not split. The default is 0.
func MinGain(g float64) Option { return func(c *config) { c.minGain = g } }

// L2 sets the penalty on boosted leaf values, as XGBoost's lambda. The
// default is 1.
func L2(l float64) Option { return func(c *config) { c.l2 = l } }

// Subsample sets the fraction of rows each boosted tree learns from,
// drawn afresh for each. The default is 1, every row.
func Subsample(f float64) Option { return func(c *config) { c.subsample = f } }

// ColSample sets the fraction of features each boosted tree may split on.
// The default is 1, every feature.
func ColSample(f float64) Option { return func(c *config) { c.colsample = f } }

// PerNode sets how many features a forest draws at each node to choose a
// split from. The default is the square root of the features for
// Classification and every feature for Regression, as in scikit-learn.
func PerNode(n int) Option { return func(c *config) { c.perNode = n } }

// Bins sets the most bins each feature is sorted into, from 2 to 255. The
// default is 255.
func Bins(n int) Option { return func(c *config) { c.bins = n } }

// Classes sets the number of classes, for when the last ones are missing
// from the training rows. By default it is one more than the largest
// class index in Y.
func Classes(n int) Option { return func(c *config) { c.classes = n } }

// Seed seeds the random draws of rows and features. The default is 1.
func Seed(s uint64) Option { return func(c *config) { c.seed = s } }

// Workers sets how many goroutines training uses. The default is
// GOMAXPROCS.
func Workers(n int) Option { return func(c *config) { c.workers = n } }

// Validation stops boosting once the loss on val has not improved for
// patience rounds, and keeps the trees up to its best round.
func Validation(val tabular.Data, patience int) Option {
	return func(c *config) { c.val, c.patience = &val, patience }
}

// Bootstrap sets whether each forest tree learns from rows drawn with
// replacement. The default is true for RandomForest.
func Bootstrap(b bool) Option { return func(c *config) { c.bootstrap = b } }

// Progress calls f after each tree grows with the trees grown so far and
// the most there will be. Boosting may stop early on Validation; a forest
// calls f from one goroutine at a time.
func Progress(f func(done, total int)) Option { return func(c *config) { c.progress = f } }

func (c *config) check(d tabular.Data) error {
	bad := func(name string, v any) error { return fmt.Errorf("tree: %s %v: %w", name, v, tensor.ErrShape) }
	switch {
	case c.trees < 1:
		return bad("trees", c.trees)
	case !(c.rate > 0) || math.IsInf(c.rate, 0):
		return bad("learning rate", c.rate)
	case c.maxDepth < 1:
		return bad("max depth", c.maxDepth)
	case c.minLeaf < 1:
		return bad("min leaf", c.minLeaf)
	case c.minWeight < 0 || math.IsNaN(c.minWeight):
		return bad("min weight", c.minWeight)
	case c.l2 < 0 || math.IsNaN(c.l2):
		return bad("L2", c.l2)
	case !(c.subsample > 0 && c.subsample <= 1):
		return bad("subsample", c.subsample)
	case !(c.colsample > 0 && c.colsample <= 1):
		return bad("column sample", c.colsample)
	case c.perNode < 0 || c.perNode > d.Features:
		return bad("features per node", c.perNode)
	case c.bins < 2 || c.bins > 255:
		return bad("bins", c.bins)
	case c.classes < 0 || c.classes > tabular.MaxClasses:
		return bad("classes", c.classes)
	case c.val != nil && c.val.Features != d.Features:
		return fmt.Errorf("tree: validation rows of %d features for %d: %w", c.val.Features, d.Features, tensor.ErrShape)
	case c.val != nil && c.patience < 1:
		return bad("patience", c.patience)
	}
	return nil
}

// classes returns the number of classes of d.
func (c *config) numClasses(d tabular.Data) int {
	return max(c.classes, d.Classes())
}
