package llm

import (
	"context"
	"fmt"
	"math"

	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/tensor"
)

// checkTokens fails unless every token is in the vocabulary.
func (m *Model) checkTokens(seqs ...[]int32) error {
	for _, s := range seqs {
		for _, t := range s {
			if t < 0 || int(t) >= m.Config.Vocab {
				return fmt.Errorf("llm: token %d of a vocabulary of %d: %w", t, m.Config.Vocab, tensor.ErrIndex)
			}
		}
	}
	return nil
}

// Hidden runs tokens, B sequences of S tokens each at positions 0 to
// S-1, Int32 [B, S], and returns the output of the last block after the
// final norm, [B, S, Embed]: what the output projection turns into
// logits. Each position sees itself and the positions before it, so a
// sequence padded at its end computes the same at its real positions as
// alone. It records the gradients of the model's trainable parameters,
// the pairs of AddLoRA; nn.Inference(m).Hidden doesn't.
func (m *Model) Hidden(tokens *tensor.Tensor) *tensor.Tensor {
	c := m.Config
	if tokens.Err() != nil {
		return tokens
	}
	if s := tokens.Shape(); len(s) != 2 || tokens.DType() != tensor.Int32 {
		return tensor.Failed(fmt.Errorf("llm: tokens of %s %v, want Int32 [B, S]: %w", tokens.DType(), s, tensor.ErrShape))
	}
	b, s := tokens.Shape()[0], tokens.Shape()[1]
	x := m.Embed.Forward(tokens.To(m.Embed.Weight.Value().Device())) // [B, S, embed]
	for _, blk := range m.Blocks {
		q, k, v := blk.project(blk.AttnNorm.Forward(x), c)
		q, k = m.Rope.Apply(q, 0), m.Rope.Apply(k, 0)
		a := tensor.ScaledDotProductAttention(q, k, v, tensor.Causal())
		x = x.Add(blk.O.Forward(a.Transpose(1, 2).Reshape(b, s, c.Heads*c.HeadDim)))
		x = x.Add(blk.mlp(blk.MLPNorm.Forward(x)))
	}
	return m.Norm.Forward(x)
}

// batch pads seqs with zeros to the length of the longest and returns
// them as Int32 [B, S], with the next token of each position, [B·S], and
// a weight of 1 for the positions whose next token is learned and 0 for
// the others, [B·S, 1]: the positions before a token whose learn is set,
// and none in the padding. n is how many are learned.
func batch(seqs [][]int32, learn [][]bool) (in, next, weight *tensor.Tensor, n int, err error) {
	if len(seqs) == 0 || len(learn) != len(seqs) {
		return nil, nil, nil, 0, fmt.Errorf("llm: %d sequences with %d masks: %w", len(seqs), len(learn), tensor.ErrShape)
	}
	s := 0
	for i, q := range seqs {
		if len(learn[i]) != len(q) {
			return nil, nil, nil, 0, fmt.Errorf("llm: sequence %d of %d tokens with a mask of %d: %w", i, len(q), len(learn[i]), tensor.ErrShape)
		}
		s = max(s, len(q)-1)
	}
	if s == 0 {
		return nil, nil, nil, 0, fmt.Errorf("llm: no sequence has two tokens: %w", tensor.ErrShape)
	}
	ids := make([]int32, len(seqs)*s)
	nexts := make([]int32, len(seqs)*s)
	ws := make([]float32, len(seqs)*s)
	for i, q := range seqs {
		for j := 0; j+1 < len(q); j++ {
			ids[i*s+j] = q[j]
			nexts[i*s+j] = q[j+1]
			if learn[i][j+1] {
				ws[i*s+j] = 1
				n++
			}
		}
	}
	return tensor.FromSlice(ids, len(seqs), s), tensor.FromSlice(nexts, len(seqs)*s), tensor.FromSlice(ws, len(seqs)*s, 1), n, nil
}

// Loss returns the mean cross-entropy of the model's prediction of each
// token of seqs whose learn is set, from the tokens before it, as a
// scalar ready for Backward: the loss of causal language modeling. A
// fine-tune on chats learns only the answers this way, and one on plain
// text every token but the first. seqs may differ in length; learn holds
// a flag per token of each. At least one flag past a sequence's first
// token must be set.
func (m *Model) Loss(seqs [][]int32, learn [][]bool) *tensor.Tensor {
	logp, next, weight, n, err := m.logProbs(seqs, learn)
	if err != nil {
		return tensor.Failed(err)
	}
	nll := logp.Gather(1, next.Unsqueeze(1)).Mul(weight).Sum().Neg()
	return nll.Div(tensor.Full(tensor.Float32, float64(n)).To(nll.Device()))
}

// logProbs returns the log-probabilities of the next token at each
// position of seqs, [B·S, Vocab], with batch's next tokens, weights and
// count of learned positions.
func (m *Model) logProbs(seqs [][]int32, learn [][]bool) (logp, next, weight *tensor.Tensor, n int, err error) {
	if err := m.checkTokens(seqs...); err != nil {
		return nil, nil, nil, 0, err
	}
	in, next, weight, n, err := batch(seqs, learn)
	if err != nil {
		return nil, nil, nil, 0, err
	}
	if n == 0 {
		return nil, nil, nil, 0, fmt.Errorf("llm: no token to learn: %w", tensor.ErrShape)
	}
	dev := m.Embed.Weight.Value().Device()
	logits := m.Forward(in).Reshape(-1, m.Config.Vocab)
	return logits.LogSoftmax(1), next.To(dev), weight.To(dev), n, nil
}

// Score is how well the model predicts the tokens of seqs whose learn is
// set: the mean cross-entropy, as Loss computes it, and how many of them
// are its likeliest next token. It records no gradients.
func (m *Model) Score(ctx context.Context, seqs [][]int32, learn [][]bool) (loss float64, right, total int, err error) {
	logp, next, weight, n, err := nn.Inference(m).logProbs(seqs, learn)
	if err != nil {
		return 0, 0, 0, err
	}
	nll := logp.Gather(1, next.Unsqueeze(1)).Mul(weight).Sum().Neg()
	hit := logp.Argmax(1).Equal(next).Cast(tensor.Float32).Unsqueeze(1).Mul(weight).Sum()
	if err := tensor.Realize(ctx, nll, hit); err != nil {
		return 0, 0, 0, err
	}
	l, err := nll.Item[float32](ctx)
	if err != nil {
		return 0, 0, 0, err
	}
	h, err := hit.Item[float32](ctx)
	if err != nil {
		return 0, 0, 0, err
	}
	return float64(l) / float64(n), int(math.Round(float64(h))), n, nil
}

// Forward returns the logits of every position of tokens, Int32 [B, S],
// as [B, S, Vocab] (see Hidden).
func (m *Model) Forward(tokens *tensor.Tensor) *tensor.Tensor {
	return m.Output.Forward(m.Hidden(tokens))
}

// Embedding returns an embedding of the text tokens: the mean of the outputs
// of the last block over its positions, scaled to length 1, which
// search ranks texts by the dot product of. It records no gradients.
func (m *Model) Embedding(ctx context.Context, tokens []int32) ([]float32, error) {
	if len(tokens) == 0 {
		return nil, fmt.Errorf("llm: no tokens to embed: %w", tensor.ErrShape)
	}
	if err := m.checkTokens(tokens); err != nil {
		return nil, err
	}
	h := nn.Inference(m).Hidden(tensor.FromSlice(tokens, 1, len(tokens))).Mean(1).Reshape(m.Config.Embed)
	v, err := h.Data[float32](ctx)
	if err != nil {
		return nil, err
	}
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum > 0 {
		inv := float32(1 / math.Sqrt(sum))
		for i := range v {
			v[i] *= inv
		}
	}
	return v, nil
}
