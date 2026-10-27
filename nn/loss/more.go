package loss

import (
	"fmt"
	"slices"

	"github.com/gsoultan/mimir/tensor"
)

func f32(v float64) *tensor.Tensor { return tensor.Scalar(float32(v)) }

// softplus is log(1 + exp(x)), computed stably as max(x, 0) + log(1 +
// exp(-|x|)).
func softplus(x *tensor.Tensor) *tensor.Tensor {
	return x.Maximum(f32(0)).Add(f32(1).Add(x.Abs().Neg().Exp()).Log())
}

// pow returns x^e for x ≥ 0, as exp(e·log x), 0 where x is 0.
func pow(x *tensor.Tensor, e float64) *tensor.Tensor {
	if e == 1 {
		return x
	}
	x = x.Maximum(f32(0))
	return tensor.Where(x.Greater(f32(0)), x.Maximum(f32(1e-30)).Log().Mul(f32(e)).Exp(), tensor.Zeros(tensor.Float32))
}

// oneHot returns Float32 [N, C, ...] with 1 at each target's class.
func oneHot(target *tensor.Tensor, like []int) *tensor.Tensor {
	return tensor.Zeros(tensor.Float32, like...).Scatter(1, target.Unsqueeze(1), tensor.Full(tensor.Float32, 1).Expand(target.Unsqueeze(1).Shape()...))
}

func classTargets(what string, logits, target *tensor.Tensor) error {
	switch {
	case logits.Err() != nil:
		return logits.Err()
	case target.Err() != nil:
		return target.Err()
	case target.DType() != tensor.Int32:
		return fmt.Errorf("loss: %s of %s class indices, want Int32: %w", what, target.DType(), tensor.ErrDType)
	}
	ls, ts := logits.Shape(), target.Shape()
	if len(ls) < 2 || !slices.Equal(slices.Delete(slices.Clone(ls), 1, 2), ts) {
		return fmt.Errorf("loss: %s of logits %v for targets %v, want [N, C, ...] and [N, ...]: %w", what, ls, ts, tensor.ErrShape)
	}
	return nil
}

// LabelSmoothing returns CrossEntropy against targets softened by eps:
// each class gets eps/C and the true one 1-eps more, like
// torch.nn.functional.cross_entropy(label_smoothing=eps). It keeps a
// classifier from growing ever more sure of itself.
func LabelSmoothing(eps float64) Func {
	return func(logits, target *tensor.Tensor) *tensor.Tensor {
		if err := classTargets("label smoothing", logits, target); err != nil {
			return tensor.Failed(err)
		}
		logp := logits.LogSoftmax(1)
		nll := logp.Gather(1, target.Unsqueeze(1)).Mean().Neg()
		// The mean over classes, averaged over every other axis.
		uniform := logp.Mean().Neg()
		return nll.Mul(f32(1 - eps)).Add(uniform.Mul(f32(eps)))
	}
}

// Focal returns the focal loss of Lin et al. for logits [N, C, ...] and
// Int32 class indices [N, ...]: cross-entropy scaled by (1-p)^gamma, p
// the probability of the true class, so the examples a model already gets
// right count less and the rare, hard ones more, as in RetinaNet. gamma 0
// is CrossEntropy; 2 is the usual.
func Focal(gamma float64) Func {
	return func(logits, target *tensor.Tensor) *tensor.Tensor {
		if err := classTargets("focal loss", logits, target); err != nil {
			return tensor.Failed(err)
		}
		logp := logits.LogSoftmax(1).Gather(1, target.Unsqueeze(1))
		if gamma == 0 {
			return logp.Mean().Neg()
		}
		return pow(f32(1).Sub(logp.Exp()), gamma).Mul(logp).Mean().Neg()
	}
}

// BinaryFocal returns the focal loss of logits and target probabilities
// of the same shape, as RetinaNet scores each anchor: BCEWithLogits
// scaled by (1-pt)^gamma and weighted alpha for positives, 1-alpha for
// negatives, like torchvision.ops.sigmoid_focal_loss with reduction
// "mean". gamma 2 and alpha 0.25 are the usual; alpha < 0 weighs both
// alike.
func BinaryFocal(gamma, alpha float64) Func {
	return func(logits, target *tensor.Tensor) *tensor.Tensor {
		if err := sameShape("binary focal loss", logits, target); err != nil {
			return tensor.Failed(err)
		}
		ce := softplus(logits).Sub(logits.Mul(target))
		p := logits.Sigmoid()
		pt := p.Mul(target).Add(f32(1).Sub(p).Mul(f32(1).Sub(target)))
		l := ce
		if gamma != 0 {
			l = l.Mul(pow(f32(1).Sub(pt), gamma))
		}
		if alpha >= 0 {
			l = l.Mul(target.Mul(f32(alpha)).Add(f32(1).Sub(target).Mul(f32(1 - alpha))))
		}
		return l.Mean()
	}
}

// BCE returns the binary cross-entropy between probabilities pred, in
// (0, 1), and target probabilities of the same shape, like
// torch.nn.functional.binary_cross_entropy. Prefer BCEWithLogits, which
// is stable where pred is near 0 or 1.
func BCE(pred, target *tensor.Tensor) *tensor.Tensor {
	if err := sameShape("binary cross entropy", pred, target); err != nil {
		return tensor.Failed(err)
	}
	lo := f32(1e-7)
	p := pred.Maximum(lo).Minimum(f32(1 - 1e-7))
	return target.Mul(p.Log()).Add(f32(1).Sub(target).Mul(f32(1).Sub(p).Log())).Mean().Neg()
}

// KLDiv returns the Kullback-Leibler divergence of target probabilities
// from the log-probabilities logp, of the same shape [N, ...], summed over
// all but the first axis and averaged over N, like
// torch.nn.functional.kl_div(reduction="batchmean"). It is the loss of
// distillation, a student's log-softmax against a teacher's softmax.
func KLDiv(logp, target *tensor.Tensor) *tensor.Tensor {
	if err := sameShape("KL divergence", logp, target); err != nil {
		return tensor.Failed(err)
	}
	n := logp.Shape()[0]
	// target·(log target - logp), with 0·log 0 = 0.
	logt := tensor.Where(target.Greater(f32(0)), target.Maximum(f32(1e-30)).Log(), tensor.Zeros(tensor.Float32))
	return target.Mul(logt.Sub(logp)).Sum().Mul(f32(1 / float64(max(n, 1))))
}

// SmoothL1 returns the smooth L1 loss with threshold beta: half the
// squared error over beta below it, the absolute error less beta/2 above,
// like torch.nn.functional.smooth_l1_loss. Detectors regress boxes with
// it; beta 1 is Huber.
func SmoothL1(beta float64) Func {
	return func(pred, target *tensor.Tensor) *tensor.Tensor {
		if err := sameShape("smooth L1 loss", pred, target); err != nil {
			return tensor.Failed(err)
		}
		d := pred.Sub(target).Abs()
		if beta <= 0 {
			return d.Mean()
		}
		return tensor.Where(d.Less(f32(beta)), d.Mul(d).Mul(f32(0.5/beta)), d.Sub(f32(beta/2))).Mean()
	}
}

// Hinge returns the multi-class hinge loss of scores [N, C] for Int32
// classes [N]: the sum over wrong classes of max(0, 1 + score - true
// score), averaged over N·C, like torch.nn.functional.multi_margin_loss.
// It is the loss of a support vector machine.
func Hinge(scores, target *tensor.Tensor) *tensor.Tensor {
	if err := classTargets("hinge loss", scores, target); err != nil {
		return tensor.Failed(err)
	}
	if scores.Rank() != 2 {
		return tensor.Failed(fmt.Errorf("loss: hinge loss of %v, want [N, C]: %w", scores.Shape(), tensor.ErrShape))
	}
	right := scores.Gather(1, target.Unsqueeze(1))
	m := f32(1).Add(scores).Sub(right).Maximum(f32(0))
	// The true class's own margin is 1; take it out.
	return m.Sum().Sub(f32(float64(scores.Shape()[0]))).Mul(f32(1 / float64(scores.Size())))
}

// MarginRanking returns max(0, -y·(a-b) + margin) averaged, for scores a
// and b and y of the same shape holding 1 where a should rank higher and
// -1 where b should, like torch.nn.functional.margin_ranking_loss.
func MarginRanking(margin float64) func(a, b, y *tensor.Tensor) *tensor.Tensor {
	return func(a, b, y *tensor.Tensor) *tensor.Tensor {
		if err := sameShape("margin ranking loss", a, b); err != nil {
			return tensor.Failed(err)
		}
		if err := sameShape("margin ranking loss", a, y); err != nil {
			return tensor.Failed(err)
		}
		return y.Neg().Mul(a.Sub(b)).Add(f32(margin)).Maximum(f32(0)).Mean()
	}
}

// Triplet returns the triplet margin loss of embeddings [N, D]: max(0,
// |anchor-positive| - |anchor-negative| + margin) averaged over N, with
// Euclidean distances, like torch.nn.functional.triplet_margin_loss. It
// teaches embeddings where alike things are near and others far, as for
// faces or search.
func Triplet(margin float64) func(anchor, positive, negative *tensor.Tensor) *tensor.Tensor {
	return func(anchor, positive, negative *tensor.Tensor) *tensor.Tensor {
		for _, t := range []*tensor.Tensor{positive, negative} {
			if err := sameShape("triplet loss", anchor, t); err != nil {
				return tensor.Failed(err)
			}
		}
		if anchor.Rank() != 2 {
			return tensor.Failed(fmt.Errorf("loss: triplet loss of %v, want [N, D]: %w", anchor.Shape(), tensor.ErrShape))
		}
		dist := func(a, b *tensor.Tensor) *tensor.Tensor {
			d := a.Sub(b).Add(f32(1e-6))
			return d.Mul(d).Sum(1).Sqrt()
		}
		return dist(anchor, positive).Sub(dist(anchor, negative)).Add(f32(margin)).Maximum(f32(0)).Mean()
	}
}

// CosineEmbedding returns, for embeddings a and b [N, D] and y [N] of 1
// for pairs that should match and -1 for pairs that shouldn't, 1-cos
// where y is 1 and max(0, cos-margin) where it is -1, averaged, like
// torch.nn.functional.cosine_embedding_loss.
func CosineEmbedding(margin float64) func(a, b, y *tensor.Tensor) *tensor.Tensor {
	return func(a, b, y *tensor.Tensor) *tensor.Tensor {
		if err := sameShape("cosine embedding loss", a, b); err != nil {
			return tensor.Failed(err)
		}
		if err := y.Err(); err != nil {
			return y
		}
		if a.Rank() != 2 || y.Rank() != 1 || y.Shape()[0] != a.Shape()[0] {
			return tensor.Failed(fmt.Errorf("loss: cosine embedding loss of %v and labels %v, want [N, D] and [N]: %w", a.Shape(), y.Shape(), tensor.ErrShape))
		}
		norm := func(t *tensor.Tensor) *tensor.Tensor { return t.Mul(t).Sum(1).Add(f32(1e-12)).Sqrt() }
		cos := a.Mul(b).Sum(1).Div(norm(a).Mul(norm(b)))
		pos := y.Greater(f32(0))
		return tensor.Where(pos, f32(1).Sub(cos), cos.Sub(f32(margin)).Maximum(f32(0))).Mean()
	}
}

// PoissonNLL returns the negative log-likelihood of counts target under
// a Poisson distribution of log-rate logRate, exp(logRate) - target ·
// logRate averaged, like torch.nn.functional.poisson_nll_loss: for
// predicting counts, such as visits or sales a day.
func PoissonNLL(logRate, target *tensor.Tensor) *tensor.Tensor {
	if err := sameShape("Poisson loss", logRate, target); err != nil {
		return tensor.Failed(err)
	}
	return logRate.Exp().Sub(target.Mul(logRate)).Mean()
}

// GaussianNLL returns the negative log-likelihood of target under normal
// distributions of mean and variance, all the same shape, ½(log var +
// (target-mean)²/var) averaged, like torch.nn.functional.gaussian_nll_loss
// without the constant: for a model that says how sure it is.
func GaussianNLL(mean, variance, target *tensor.Tensor) *tensor.Tensor {
	if err := sameShape("Gaussian loss", mean, target); err != nil {
		return tensor.Failed(err)
	}
	if err := sameShape("Gaussian loss", variance, target); err != nil {
		return tensor.Failed(err)
	}
	v := variance.Maximum(f32(1e-6))
	d := target.Sub(mean)
	return v.Log().Add(d.Mul(d).Div(v)).Mul(f32(0.5)).Mean()
}

// Dice returns 1 minus the Dice coefficient of segmentation logits [N, C,
// ...] and Int32 class indices [N, ...], averaged over classes: the
// overlap of each class's predicted and true area, as segmentation
// models such as U-Net learn from, sum it with CrossEntropy for the
// usual mix. A class missing from both counts as a perfect overlap.
func Dice(logits, target *tensor.Tensor) *tensor.Tensor {
	if err := classTargets("dice loss", logits, target); err != nil {
		return tensor.Failed(err)
	}
	shape := logits.Shape()
	p := logits.Softmax(1)
	t := oneHot(target, shape)
	axes := []int{0}
	for i := 2; i < len(shape); i++ {
		axes = append(axes, i)
	}
	inter := p.Mul(t).Sum(axes...)
	total := p.Sum(axes...).Add(t.Sum(axes...))
	one := f32(1)
	return one.Sub(inter.Mul(f32(2)).Add(one).Div(total.Add(one)).Mean())
}

// BinaryDice returns 1 minus the Dice coefficient of mask logits and
// target masks of 0 and 1, the same shape [N, ...], for one-class
// segmentation.
func BinaryDice(logits, target *tensor.Tensor) *tensor.Tensor {
	if err := sameShape("dice loss", logits, target); err != nil {
		return tensor.Failed(err)
	}
	p := logits.Sigmoid()
	one := f32(1)
	return one.Sub(p.Mul(target).Sum().Mul(f32(2)).Add(one).Div(p.Sum().Add(target.Sum()).Add(one)))
}
