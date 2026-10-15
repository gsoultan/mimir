// Package loss has loss functions for training, like those in
// torch.nn.functional. Each returns the mean loss over all elements as a
// scalar, ready for Backward. Predictions and targets must have exactly
// the shapes documented: unlike PyTorch, nothing broadcasts, so a [N, 1]
// prediction against [N] targets is an error instead of an N×N loss.
package loss

import (
	"fmt"
	"slices"

	"github.com/gsoultan/mimir/tensor"
)

// Func is the signature every loss has.
type Func func(pred, target *tensor.Tensor) *tensor.Tensor

// CrossEntropy returns the cross-entropy between logits, shaped [N, C] or
// [N, C, d1, ...], and Int32 class indices, shaped [N] or [N, d1, ...],
// like torch.nn.functional.cross_entropy.
func CrossEntropy(logits, target *tensor.Tensor) *tensor.Tensor {
	if err := logits.Err(); err != nil {
		return logits
	}
	if logits.Rank() < 2 {
		return tensor.Failed(fmt.Errorf("loss: cross entropy of logits %v, want [N, C, ...]: %w", logits.Shape(), tensor.ErrShape))
	}
	return NLL(logits.LogSoftmax(1), target)
}

// NLL returns the negative log-likelihood of Int32 class indices target,
// shaped [N, d1, ...], under log-probabilities logp, shaped
// [N, C, d1, ...], like torch.nn.functional.nll_loss.
func NLL(logp, target *tensor.Tensor) *tensor.Tensor {
	switch {
	case logp.Err() != nil:
		return logp
	case target.Err() != nil:
		return target
	case target.DType() != tensor.Int32:
		return tensor.Failed(fmt.Errorf("loss: %s class indices, want Int32: %w", target.DType(), tensor.ErrDType))
	}
	ls, ts := logp.Shape(), target.Shape()
	if len(ls) < 2 || !slices.Equal(slices.Delete(slices.Clone(ls), 1, 2), ts) {
		return tensor.Failed(fmt.Errorf("loss: log-probabilities %v for targets %v, want [N, C, ...] and [N, ...]: %w", ls, ts, tensor.ErrShape))
	}
	return logp.Gather(1, target.Unsqueeze(1)).Mean().Neg()
}

// BCEWithLogits returns the binary cross-entropy between sigmoid(logits)
// and target probabilities of the same shape, computed stably from the
// logits, like torch.nn.functional.binary_cross_entropy_with_logits.
func BCEWithLogits(logits, target *tensor.Tensor) *tensor.Tensor {
	if err := sameShape("binary cross entropy", logits, target); err != nil {
		return tensor.Failed(err)
	}
	// max(x, 0) - x*y + log(1 + exp(-|x|))
	zero, one := tensor.Scalar[float32](0), tensor.Scalar[float32](1)
	softplus := one.Add(logits.Abs().Neg().Exp()).Log()
	return logits.Maximum(zero).Sub(logits.Mul(target)).Add(softplus).Mean()
}

// MSE returns the mean squared error, like torch.nn.functional.mse_loss.
func MSE(pred, target *tensor.Tensor) *tensor.Tensor {
	if err := sameShape("mean squared error", pred, target); err != nil {
		return tensor.Failed(err)
	}
	d := pred.Sub(target)
	return d.Mul(d).Mean()
}

// L1 returns the mean absolute error, like torch.nn.functional.l1_loss.
func L1(pred, target *tensor.Tensor) *tensor.Tensor {
	if err := sameShape("mean absolute error", pred, target); err != nil {
		return tensor.Failed(err)
	}
	return pred.Sub(target).Abs().Mean()
}

// Huber returns the Huber loss with delta 1: squared error for errors
// below 1 and absolute error above, like torch.nn.functional.huber_loss.
func Huber(pred, target *tensor.Tensor) *tensor.Tensor {
	if err := sameShape("huber loss", pred, target); err != nil {
		return tensor.Failed(err)
	}
	d := pred.Sub(target).Abs()
	one, half := tensor.Scalar[float32](1), tensor.Scalar[float32](0.5)
	return tensor.Where(d.Less(one), half.Mul(d).Mul(d), d.Sub(half)).Mean()
}

func sameShape(what string, pred, target *tensor.Tensor) error {
	if err := pred.Err(); err != nil {
		return err
	}
	if err := target.Err(); err != nil {
		return err
	}
	if !slices.Equal(pred.Shape(), target.Shape()) {
		return fmt.Errorf("loss: %s of %v and %v: %w", what, pred.Shape(), target.Shape(), tensor.ErrShape)
	}
	return nil
}
