package llm

import (
	"context"
	"math"
	"slices"
	"sync"

	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/tensor"
)

// A decoding step, one token at a time, builds a graph of a few hundred
// small kernels, and on the CPU building, planning and launching them
// takes about as long as the products themselves. Each step's graph
// differs from the last only in the position it writes and the number of
// keys it attends to, so step makes both inputs: the position as a
// tensor the caches scatter at, the rotary angles as rows, and the keys
// as a window of the caches, a multiple of window positions long, that a
// mask cuts to those held. The graph is then the same at every position
// of a window, and step records it once and replays it, which runs its
// program again without building or planning anything.

// stepWindow is the number of positions by which a step's window of the
// caches grows: each length is recorded once, and a step attends to at
// most window-1 positions it masks.
const stepWindow = 256

// decoder holds a model's recorded decoding steps, and the products they
// fuse.
type decoder struct {
	mu    sync.Mutex
	steps map[stepKey]*recorded
	// fused holds each block's fused products, made from the parameter
	// values in made.
	fused []fused
	made  []*tensor.Tensor
}

// fused is a block's queries, keys and values as one product, and its
// gate and up projections as another, where the block's layers allow:
// one product reads the input once and runs as one kernel instead of
// three or two. A nil field leaves those layers apart.
type fused struct {
	qkv, gateUp *nn.Linear
}

// stepKey says which recording a step replays: the window of the caches
// it attends to, and their capacity.
type stepKey struct{ window, capacity int }

// recorded is a recorded step and the parameter values it read, which it
// reads again only while the model still has them.
type recorded struct {
	rec    *tensor.Recording
	params []*tensor.Tensor
}

// step returns the logits, [Vocab], of token, at the position the caches
// hold, and writes its keys and values into the caches, as Logits does,
// replaying a recorded step. It returns false, having changed nothing,
// when it cannot.
func (m *Model) step(ctx context.Context, token int32, caches []*nn.KVCache) (*tensor.Tensor, bool, error) {
	n, capacity := caches[0].Len(), caches[0].Cap()
	for _, c := range caches {
		if c.Len() != n || c.Cap() != capacity {
			return nil, false, nil
		}
	}
	if n >= capacity || m.Rope == nil || m.Rope.Dim != m.Config.HeadDim {
		return nil, false, nil
	}
	dev := m.Embed.Weight.Value().Device()
	key := stepKey{min(capacity, (n/stepWindow+1)*stepWindow), capacity}
	leaves := m.stepLeaves(token, n, key.window, caches, dev)
	dsts := make([]*tensor.Tensor, 0, 2*len(caches)+1)
	for _, c := range caches {
		k, v := c.Buffers()
		dsts = append(dsts, k, v)
	}
	dsts = append(dsts, nil)
	params := paramValues(m)
	fs, err := m.fusedFor(ctx, params)
	if err != nil {
		return nil, false, err
	}

	m.decoding.mu.Lock()
	r := m.decoding.steps[key]
	m.decoding.mu.Unlock()
	var out []*tensor.Tensor
	if r != nil && slices.Equal(r.params, params) {
		got, ok, err := r.rec.ReplayInto(ctx, leaves, dsts)
		if err != nil {
			return nil, false, err
		}
		if ok {
			out = got
		}
	}
	if out == nil {
		targets, err := m.stepGraph(leaves, key.window, caches, fs)
		if err != nil {
			return nil, false, err
		}
		rec, err := tensor.RecordInto(ctx, leaves, dsts, targets)
		if err != nil {
			for _, c := range caches {
				c.Advance(nil, nil, 0)
			}
			return nil, false, err
		}
		if rec != nil {
			m.decoding.mu.Lock()
			if m.decoding.steps == nil {
				m.decoding.steps = map[stepKey]*recorded{}
			}
			m.decoding.steps[key] = &recorded{rec: rec, params: params}
			m.decoding.mu.Unlock()
		}
		out = targets
	}
	for i, c := range caches {
		c.Advance(out[2*i], out[2*i+1], 1)
	}
	return out[len(out)-1], true, nil
}

// stepLeaves returns the inputs of a step of token at position n
// attending to a window of the caches: the token, the rotary angles of
// its position, the mask of the window, the position the caches write,
// and then each cache's keys and values.
func (m *Model) stepLeaves(token int32, n, window int, caches []*nn.KVCache, dev tensor.Device) []*tensor.Tensor {
	cos, sin := m.Rope.Turn(n, 1, dev)
	mask := make([]float32, window)
	for j := n + 1; j < window; j++ {
		mask[j] = float32(math.Inf(-1))
	}
	leaves := []*tensor.Tensor{
		tensor.FromSlice([]int32{token}, 1, 1).To(dev),
		cos, sin,
		tensor.FromSlice(mask, 1, 1, 1, window).To(dev),
		caches[0].Positions(n).To(dev),
	}
	for _, c := range caches {
		k, v := c.Buffers()
		leaves = append(leaves, k, v)
	}
	return leaves
}

// stepGraph builds the graph of a step over leaves, as stepLeaves
// returns them: each cache's keys and values with the step's written in,
// then the logits.
func (m *Model) stepGraph(leaves []*tensor.Tensor, window int, caches []*nn.KVCache, fs []fused) ([]*tensor.Tensor, error) {
	c := m.Config
	tok, cos, sin, mask, pos := leaves[0], leaves[1], leaves[2], leaves[3], leaves[4]
	x := m.Embed.Forward(tok) // [1, 1, embed]
	targets := make([]*tensor.Tensor, 0, 2*len(caches)+1)
	held := [][2]int{{0, 1}, {0, c.KVHeads}, {0, window}, {0, c.HeadDim}}
	for i, b := range m.Blocks {
		h := rmsNorm(x, b.AttnNorm)
		var q, k, v *tensor.Tensor
		if f := fs[i].qkv; f != nil {
			parts := f.Forward(h).Split(-1, c.Heads*c.HeadDim, c.KVHeads*c.HeadDim, c.KVHeads*c.HeadDim)
			q, k, v = heads(parts[0], c.Heads, c.HeadDim), heads(parts[1], c.KVHeads, c.HeadDim), heads(parts[2], c.KVHeads, c.HeadDim)
		} else {
			q, k, v = b.project(h, c)
		}
		q, k = m.Rope.Rotate(q, cos, sin), m.Rope.Rotate(k, cos, sin)
		keys, values, err := caches[i].Write(k, v, pos)
		if err != nil {
			return nil, err
		}
		targets = append(targets, keys, values)
		a := tensor.ScaledDotProductAttention(q, keys.Shrink(held), values.Shrink(held), tensor.AdditiveMask(mask))
		x = x.Add(b.O.Forward(a.Transpose(1, 2).Reshape(1, 1, c.Heads*c.HeadDim)))
		h = rmsNorm(x, b.MLPNorm)
		if f := fs[i].gateUp; f != nil {
			ff := f.Weight.Value().Shape()[0] / 2
			parts := f.Forward(h).Split(-1, ff, ff)
			x = x.Add(b.Down.Forward(parts[0].SiLU().Mul(parts[1])))
		} else {
			x = x.Add(b.mlp(h))
		}
	}
	logits := m.Output.Forward(rmsNorm(x, m.Norm)).Reshape(c.Vocab)
	return append(targets, logits), nil
}

// fusedFor returns each block's fused products, made from params, the
// values of m's parameters, which it makes again when they change.
func (m *Model) fusedFor(ctx context.Context, params []*tensor.Tensor) ([]fused, error) {
	d := &m.decoding
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.fused != nil && slices.Equal(d.made, params) {
		return d.fused, nil
	}
	fs := make([]fused, len(m.Blocks))
	for i, b := range m.Blocks {
		var err error
		if fs[i].qkv, err = fuse(ctx, b.Q, b.K, b.V); err != nil {
			return nil, err
		}
		if fs[i].gateUp, err = fuse(ctx, b.Gate, b.Up); err != nil {
			return nil, err
		}
	}
	d.fused, d.made = fs, params
	return fs, nil
}

// maxFused is the most bytes of weights fuse copies into one product. A
// fused product holds a copy of its layers' weights, so it pays only for
// small models, whose steps launch many short kernels; a large model's
// products run long enough that launching them costs little, and copying
// its weights would cost time once and memory for good.
const maxFused = 4 << 20

// fuse returns one linear layer whose outputs are those of ms, one after
// another, or nil when they are not all plain linear layers of one input
// width and weight dtype, with biases on all or none. Its weight holds
// the rows of theirs, which for a block-quantized dtype are whole blocks.
func fuse(ctx context.Context, ms ...nn.Module) (*nn.Linear, error) {
	ls := make([]*nn.Linear, len(ms))
	for i, m := range ms {
		l, ok := m.(*nn.Linear)
		if !ok {
			return nil, nil
		}
		ls[i] = l
	}
	w0 := ls[0].Weight.Value()
	in, dt, dev := w0.Shape()[1], w0.DType(), w0.Device()
	bias := ls[0].Bias != nil
	size := 0
	for _, l := range ls {
		w := l.Weight.Value()
		if w.Rank() != 2 || w.Shape()[1] != in || w.DType() != dt || w.Device() != dev || (l.Bias != nil) != bias {
			return nil, nil
		}
		b, _ := dt.Bytes(w.Shape()[0] * in)
		size += b
	}
	if size > maxFused {
		return nil, nil
	}
	data := make([]byte, 0, size)
	rows := 0
	for _, l := range ls {
		w := l.Weight.Value()
		b, err := w.Bytes(ctx)
		if err != nil {
			return nil, err
		}
		data = append(data, b...)
		rows += w.Shape()[0]
	}
	f := &nn.Linear{Weight: tensor.NewBuffer(tensor.FromBytes(dt, data, rows, in).To(dev))}
	if bias {
		bs := make([]*tensor.Tensor, len(ls))
		for i, l := range ls {
			bs[i] = l.Bias.Value()
		}
		f.Bias = tensor.NewBuffer(tensor.Cat(0, bs...))
	}
	return f, nil
}

// rmsNorm is n.Forward of x, one position [1, 1, features], in two
// kernels rather than six: the sum of squares as a product of x with
// itself, whose epilogue turns it into the scale, and the scaling.
func rmsNorm(x *tensor.Tensor, n *nn.RMSNorm) *tensor.Tensor {
	f := x.Cast(tensor.Float32)
	d := f.Shape()[2]
	ss := f.MatMul(f.Transpose(1, 2)) // [1, 1, 1]
	y := f.Mul(ss.Mul(tensor.Scalar(float32(1 / float64(d)))).Add(tensor.Scalar(float32(n.Eps))).Sqrt().Recip()).Cast(x.DType())
	if n.Weight != nil {
		y = y.Mul(n.Weight.Value())
	}
	return y
}

// paramValues returns the values of m's parameters, which a recorded step
// reads.
func paramValues(m *Model) []*tensor.Tensor {
	ps := nn.Parameters(m)
	vs := make([]*tensor.Tensor, len(ps))
	for i, p := range ps {
		vs[i] = p.Value()
	}
	return vs
}
