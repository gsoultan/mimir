package nn

import (
	"fmt"
	"math"
	"slices"

	"github.com/gsoultan/mimir/tensor"
)

// Recurrent layers run each layer as one tensor.Scan over the sequence.
// The input projection of every step is one matrix product before the
// loop, and the biases are added to it there, so a step multiplies only
// the hidden state; the gradient of W_hh is one product after the loop.
// The gate math of a step is a hand-written tensor.Func whose operations
// fuse into one kernel: an RNN step is one product with its activation
// applied as the product's epilogue, and an LSTM or GRU step a product
// and one fused kernel. An LSTM carries h and c in one [batch, 2, hidden]
// tensor, so that one kernel writes both.

// NumLayers stacks n recurrent layers, each reading the outputs of the
// one before, like num_layers. The default is 1.
func NumLayers(n int) Option { return func(o *options) { o.layers = n } }

// Bidirectional runs each recurrent layer over the sequence in both
// directions and joins the two outputs along the last dimension, the
// forward direction first, like bidirectional=True.
func Bidirectional() Option { return func(o *options) { o.bidirectional = true } }

// BatchFirst makes a recurrent layer take and return sequences laid out
// as [batch, time, features] instead of [time, batch, features], like
// batch_first=True. States keep their [layers, batch, hidden] layout.
func BatchFirst() Option { return func(o *options) { o.batchFirst = true } }

// LayerDropout makes a recurrent layer drop the outputs of each of its
// layers but the last with probability p in training mode, like the
// dropout argument of torch.nn.LSTM.
func LayerDropout(p float64) Option { return func(o *options) { o.layerDropout = p } }

// RNNReLU makes an RNN apply ReLU instead of tanh, like
// nonlinearity="relu".
func RNNReLU() Option { return func(o *options) { o.relu = true } }

// StepCheckpoint makes a recurrent layer run its loop k steps at a time
// with tensor.ScanCheckpoint: training keeps the states at every k-th
// step and recomputes the steps between during Backward, so a long
// sequence trains in the memory of k steps.
func StepCheckpoint(k int) Option { return func(o *options) { o.checkpoint = k } }

// RNN is an Elman recurrent network, like torch.nn.RNN: each layer
// computes h' = tanh(x W_ihᵀ + b_ih + h W_hhᵀ + b_hh) at every step, or
// ReLU with RNNReLU.
type RNN struct {
	recurrent `nn:"inline"`
}

// LSTM is a long short-term memory network, like torch.nn.LSTM, with the
// input, forget, cell and output gates in that order in its weights.
type LSTM struct {
	recurrent `nn:"inline"`
}

// GRU is a gated recurrent unit network, like torch.nn.GRU, with the
// reset, update and new gates in that order in its weights.
type GRU struct {
	recurrent `nn:"inline"`
}

// NewRNN returns an RNN from in features to hidden. NumLayers,
// Bidirectional, BatchFirst, LayerDropout, RNNReLU, StepCheckpoint and
// NoBias configure it. Weights and biases are drawn from
// U(-1/√hidden, 1/√hidden), PyTorch's default.
func NewRNN(in, hidden int, opts ...Option) *RNN {
	return &RNN{newRecurrent(rnnCell, in, hidden, opts)}
}

// NewLSTM returns an LSTM from in features to hidden, configured and
// initialized like NewRNN.
func NewLSTM(in, hidden int, opts ...Option) *LSTM {
	return &LSTM{newRecurrent(lstmCell, in, hidden, opts)}
}

// NewGRU returns a GRU from in features to hidden, configured and
// initialized like NewRNN.
func NewGRU(in, hidden int, opts ...Option) *GRU {
	return &GRU{newRecurrent(gruCell, in, hidden, opts)}
}

// Forward returns the outputs of the last layer at every step from zero
// initial states (see Run).
func (r *RNN) Forward(x *tensor.Tensor) *tensor.Tensor {
	out, _ := r.Run(x, nil)
	return out
}

// Run maps a sequence x, [time, batch, in] or [time, in] unbatched, from
// the initial hidden states h0, [layers*directions, batch, hidden] or nil
// for zeros, to the outputs of the last layer at every step,
// [time, batch, directions*hidden], and the final hidden state of every
// layer and direction, shaped like h0, like torch.nn.RNN.
func (r *RNN) Run(x, h0 *tensor.Tensor) (out, hn *tensor.Tensor) {
	out, s := r.run(x, h0)
	return out, s[0]
}

// Forward returns the outputs of the last layer at every step from zero
// initial states (see Run).
func (l *LSTM) Forward(x *tensor.Tensor) *tensor.Tensor {
	out, _, _ := l.Run(x, nil, nil)
	return out
}

// Run is RNN.Run with the initial cell states c0 too, shaped like h0, and
// the final cell states, like torch.nn.LSTM.
func (l *LSTM) Run(x, h0, c0 *tensor.Tensor) (out, hn, cn *tensor.Tensor) {
	out, s := l.run(x, h0, c0)
	return out, s[0], s[1]
}

// Forward returns the outputs of the last layer at every step from zero
// initial states (see Run).
func (g *GRU) Forward(x *tensor.Tensor) *tensor.Tensor {
	out, _ := g.Run(x, nil)
	return out
}

// Run is RNN.Run for a GRU, like torch.nn.GRU.
func (g *GRU) Run(x, h0 *tensor.Tensor) (out, hn *tensor.Tensor) {
	out, s := g.run(x, h0)
	return out, s[0]
}

type cellKind int

const (
	rnnCell cellKind = iota
	lstmCell
	gruCell
)

// gates returns the number of gates whose weights a cell of kind stacks.
func (k cellKind) gates() int { return [...]int{rnnCell: 1, lstmCell: 4, gruCell: 3}[k] }

// recurrent is what RNN, LSTM and GRU share: their weights, configuration
// and loop.
type recurrent struct {
	// Params holds the weights and biases under PyTorch's names:
	// weight_ih_lK [gates*hidden, in], weight_hh_lK [gates*hidden, hidden],
	// bias_ih_lK and bias_hh_lK [gates*hidden] for layer K, and the same
	// with the suffix _reverse for the backward direction of a
	// bidirectional layer. Layers after the first take
	// directions*hidden features in.
	Params map[string]*tensor.Param `nn:"inline"`

	kind          cellKind
	in, hidden    int
	layers        int
	bidirectional bool
	batchFirst    bool
	noBias        bool
	relu          bool
	checkpoint    int
	drop          *Dropout
}

func newRecurrent(kind cellKind, in, hidden int, opts []Option) recurrent {
	o := resolve(opts, options{layers: 1})
	r := recurrent{
		Params: map[string]*tensor.Param{}, kind: kind, in: in, hidden: hidden,
		layers: o.layers, bidirectional: o.bidirectional, batchFirst: o.batchFirst,
		noBias: o.noBias, relu: o.relu, checkpoint: o.checkpoint,
		drop: NewDropout(o.layerDropout, opts...),
	}
	bound := 1 / math.Sqrt(float64(max(hidden, 1)))
	g := kind.gates() * hidden
	for l := range max(r.layers, 0) {
		width := in
		if l > 0 {
			width = r.directions() * hidden
		}
		for d := range r.directions() {
			sfx := suffix(l, d)
			r.Params["weight_ih"+sfx] = uniform(o, bound, g, width)
			r.Params["weight_hh"+sfx] = uniform(o, bound, g, hidden)
			if !o.noBias {
				r.Params["bias_ih"+sfx] = uniform(o, bound, g)
				r.Params["bias_hh"+sfx] = uniform(o, bound, g)
			}
		}
	}
	return r
}

// suffix returns the suffix of the parameter names of layer l in
// direction d.
func suffix(l, d int) string {
	if d == 1 {
		return fmt.Sprintf("_l%d_reverse", l)
	}
	return fmt.Sprintf("_l%d", l)
}

func (r *recurrent) directions() int {
	if r.bidirectional {
		return 2
	}
	return 1
}

// value returns the value of the parameter name, or nil if there is none,
// as for the biases of a layer made with NoBias.
func (r *recurrent) value(name string) *tensor.Tensor {
	if p := r.Params[name]; p != nil {
		return p.Value()
	}
	return nil
}

// run checks x and the initial states, one per state the cell carries,
// runs every layer and returns the last layer's outputs and the final
// states, one tensor per state the cell carries.
func (r *recurrent) run(x *tensor.Tensor, init ...*tensor.Tensor) (*tensor.Tensor, []*tensor.Tensor) {
	failed := func(err error) (*tensor.Tensor, []*tensor.Tensor) {
		out := tensor.Failed(err)
		states := make([]*tensor.Tensor, len(init))
		for i := range states {
			states[i] = out
		}
		return out, states
	}
	if err := x.Err(); err != nil {
		return failed(err)
	}
	shape := x.Shape()
	if (len(shape) != 2 && len(shape) != 3) || shape[len(shape)-1] != r.in || r.layers < 1 || r.hidden < 1 {
		return failed(fmt.Errorf("nn: %d-layer recurrent layer from %d to %d features of %v: %w", r.layers, r.in, r.hidden, shape, tensor.ErrShape))
	}
	time := 0
	if r.batchFirst {
		time = 1
	}
	batched := len(shape) == 3
	if !batched {
		x = x.Unsqueeze(1 - time)
	}
	batch := x.Shape()[1-time]
	stateShape := []int{r.layers * r.directions(), batch, r.hidden}
	states := make([]*tensor.Tensor, len(init))
	for i, s := range init {
		if s == nil {
			continue
		}
		if !batched {
			s = s.Unsqueeze(1)
		}
		if err := s.Err(); err != nil {
			return failed(err)
		}
		if !slices.Equal(s.Shape(), stateShape) {
			return failed(fmt.Errorf("nn: recurrent state %v for %v: %w", s.Shape(), stateShape, tensor.ErrShape))
		}
		states[i] = s
	}

	finals := make([][]*tensor.Tensor, len(init)) // by state, then layer and direction
	for l := range r.layers {
		var outs []*tensor.Tensor
		for d := range r.directions() {
			at := l*r.directions() + d
			first := make([]*tensor.Tensor, len(states))
			for i, s := range states {
				if s != nil {
					first[i] = s.Shrink(stateRange(stateShape, at)).Squeeze(0)
				}
			}
			out, last := r.direction(x, time, l, d, first)
			outs = append(outs, out)
			for i, s := range last {
				finals[i] = append(finals[i], s)
			}
		}
		x = outs[0]
		if len(outs) > 1 {
			x = tensor.Cat(-1, outs...)
		}
		if l < r.layers-1 {
			x = r.drop.Forward(x)
		}
	}
	result := make([]*tensor.Tensor, len(finals))
	for i, f := range finals {
		result[i] = tensor.Stack(0, f...)
		if !batched {
			result[i] = result[i].Squeeze(1)
		}
	}
	if !batched {
		x = x.Squeeze(1 - time)
	}
	return x, result
}

// stateRange returns the ranges of a state of shape that keep the layer
// and direction at.
func stateRange(shape []int, at int) [][2]int {
	return [][2]int{{at, at + 1}, {0, shape[1]}, {0, shape[2]}}
}

// direction runs layer l in direction d over x, whose time axis is time,
// from the initial states first, nil for zeros. It returns the outputs at
// every step, along the same time axis, and the final states.
func (r *recurrent) direction(x *tensor.Tensor, time, l, d int, first []*tensor.Tensor) (*tensor.Tensor, []*tensor.Tensor) {
	sfx := suffix(l, d)
	shape := x.Shape()
	batch, h := shape[1-time], r.hidden
	g := r.kind.gates() * h

	// The input projection of every step, as one product.
	proj := x.Reshape(-1, shape[2]).MatMul(r.value("weight_ih"+sfx).Transpose(0, 1))
	bih, bhh := r.value("bias_ih"+sfx), r.value("bias_hh"+sfx)
	if bih != nil {
		if r.kind == gruCell {
			proj = proj.Add(bih)
		} else {
			proj = proj.Add(bih.Add(bhh))
		}
	}
	proj = proj.Reshape(shape[0], shape[1], g)
	dt := proj.DType()
	for i, s := range first {
		if s == nil {
			first[i] = tensor.Zeros(dt, batch, h)
		} else {
			first[i] = s.Cast(dt)
		}
	}

	// The steps read W_hh detached. Its gradient, the sum over the steps
	// of each one's gradient of its hidden product times the state it
	// read, is then one product after the loop rather than a product per
	// step and a sum of them: a hook on what each step adds to its hidden
	// product, which has the gradients of every step's at once, adds it.
	// The LSTM and RNN steps add their input projections, the GRU step
	// b_hh, or zeros.
	whh := r.value("weight_hh" + sfx)
	wt := whh.Detach().Transpose(0, 1)
	hidden := func(h *tensor.Tensor) *tensor.Tensor { return h.MatMul(wt) }
	var prev *tensor.Tensor // the states the steps read, along time
	hook := hiddenWeightHook(&prev)
	xs := []*tensor.Tensor{proj}
	if r.kind == gruCell {
		add := tensor.Zeros(dt, shape[0], shape[1], g)
		if bhh != nil {
			add = bhh.Expand(shape[0], shape[1], g)
		}
		xs = append(xs, hook.Apply(add, whh))
	} else {
		xs[0] = hook.Apply(proj, whh)
	}
	var init []*tensor.Tensor
	var step tensor.ScanStep
	var lstm lstmFuncs
	switch r.kind {
	case rnnCell:
		act := tanhFunc
		if r.relu {
			act = reluFunc
		}
		init = first
		step = func(carry, x []*tensor.Tensor) (next, y []*tensor.Tensor) {
			hn := act.Apply(hidden(carry[0]).Add(x[0]))
			return []*tensor.Tensor{hn}, []*tensor.Tensor{hn}
		}
	case lstmCell:
		lstm = newLSTMFuncs(dt, batch, h)
		init = []*tensor.Tensor{tensor.Stack(1, first[0], first[1])}
		step = func(carry, x []*tensor.Tensor) (next, y []*tensor.Tensor) {
			pre := hidden(lstm.pick.Apply(carry[0])).Reshape(batch, 4, h).Add(x[0].Reshape(batch, 4, h))
			hc := lstm.cell.Apply(lstm.gates.Apply(pre), carry[0])
			return []*tensor.Tensor{hc}, []*tensor.Tensor{lstm.pick.Apply(hc)}
		}
	case gruCell:
		f := newGRUFunc(dt, batch, h)
		init = first
		step = func(carry, x []*tensor.Tensor) (next, y []*tensor.Tensor) {
			hn := f.Apply(x[0], hidden(carry[0]).Add(x[1]), carry[0])
			return []*tensor.Tensor{hn}, []*tensor.Tensor{hn}
		}
	}
	var opts []tensor.ScanOption
	if d == 1 {
		opts = append(opts, tensor.ScanReverse())
	}
	if r.checkpoint > 0 {
		opts = append(opts, tensor.ScanCheckpoint(r.checkpoint))
	}
	carry, ys := tensor.Scan(step, init, xs, time, opts...)
	prev = shiftStates(ys[0], first[0], time, d == 1)
	if r.kind == lstmCell {
		hc := carry[0]
		return ys[0], []*tensor.Tensor{lstm.pick.Apply(hc), slot(hc, 1).Reshape(batch, h)}
	}
	return ys[0], carry
}

// hiddenWeightHook returns a Func of what the steps of a layer add to
// their hidden products, [time, batch, n] or [batch, time, n], and W_hh,
// [n, hidden], whose value is the first and whose gradient g passes on to
// it. The gradient of W_hh is gᵀ times *prev, the states the steps read
// in the same layout, which the layer sets once it has built the loop.
func hiddenWeightHook(prev **tensor.Tensor) tensor.Func {
	return tensor.Func{
		Forward: func(in []*tensor.Tensor) *tensor.Tensor { return in[0] },
		Backward: func(g *tensor.Tensor, in []*tensor.Tensor, _ *tensor.Tensor) []*tensor.Tensor {
			w := in[1].Shape()
			dw := g.Reshape(-1, w[0]).Transpose(0, 1).MatMul((*prev).Reshape(-1, w[1]).Cast(g.DType()))
			return []*tensor.Tensor{g, dw.Cast(in[1].DType())}
		},
	}
}

// shiftStates returns the states that the steps of a direction read, in
// the layout of its outputs ys, whose time axis is time: the initial
// state h0 at the first step and each step's output at the next. The
// steps run backwards if reverse.
func shiftStates(ys, h0 *tensor.Tensor, time int, reverse bool) *tensor.Tensor {
	ys, h0 = ys.Detach(), h0.Detach().Unsqueeze(time)
	n := ys.Shape()[time]
	if n == 1 {
		return h0
	}
	keep := func(lo, hi int) *tensor.Tensor {
		r := [][2]int{{0, ys.Shape()[0]}, {0, ys.Shape()[1]}, {0, ys.Shape()[2]}}
		r[time] = [2]int{lo, hi}
		return ys.Shrink(r)
	}
	if reverse {
		return tensor.Cat(time, keep(1, n), h0)
	}
	return tensor.Cat(time, h0, keep(0, n-1))
}

// slotRange returns the ranges of a [batch, n, hidden] tensor that keep
// slot k.
func slotRange(batch, n, hidden, k int) [][2]int {
	return [][2]int{{0, batch}, {k, k + 1}, {0, hidden}}
}

// slot returns slot k of x, [batch, n, hidden], as [batch, 1, hidden].
func slot(x *tensor.Tensor, k int) *tensor.Tensor {
	s := x.Shape()
	return x.Shrink(slotRange(s[0], s[1], s[2], k))
}

// slots returns a [1, len(v), 1] tensor of dtype dt holding v, which a
// step broadcasts over the slots of its gates or states. It is made of
// constants, so it runs on any device.
func slots(dt tensor.DType, v ...float64) *tensor.Tensor {
	out := tensor.Zeros(dt, 1, len(v), 1)
	for k, x := range v {
		if x != 0 {
			out = out.Add(tensor.Full(dt, x, 1, 1, 1).Pad([][2]int{{0, 0}, {k, len(v) - 1 - k}, {0, 0}}, 0))
		}
	}
	return out
}

// tanhFunc is tanh with a gradient computed from its result alone, so
// that a step keeps no other activation for Backward.
var tanhFunc = tensor.Func{
	Forward: func(in []*tensor.Tensor) *tensor.Tensor { return in[0].Tanh() },
	Backward: func(g *tensor.Tensor, _ []*tensor.Tensor, y *tensor.Tensor) []*tensor.Tensor {
		one := tensor.Full(y.DType(), 1)
		return []*tensor.Tensor{g.Mul(one.Sub(y.Mul(y)))}
	},
}

// reluFunc is ReLU with a gradient computed from its result alone.
var reluFunc = tensor.Func{
	Forward: func(in []*tensor.Tensor) *tensor.Tensor { return in[0].ReLU() },
	Backward: func(g *tensor.Tensor, _ []*tensor.Tensor, y *tensor.Tensor) []*tensor.Tensor {
		return []*tensor.Tensor{g.Mul(y.Greater(tensor.Full(y.DType(), 0)))}
	},
}

// lstmFuncs are the parts of an LSTM step over a batch of batch states
// of hidden values in dtype dt.
type lstmFuncs struct {
	// gates activates the gates, [batch, 4, hidden], in one pass:
	// a·σ(b·x)+c, which is σ(x) for the input, forget and output gates
	// and 2σ(2x)-1 = tanh(x) for the cell gate. Its gradient, from its
	// result y, is (y-c)(1-y).
	gates tensor.Func
	// cell maps the activated gates and the states, [batch, 2, hidden]
	// holding h and c, to the next states.
	cell tensor.Func
	// pick returns h, [batch, hidden], from the states. Its gradient is
	// elementwise, where a view's would be a padded copy.
	pick tensor.Func
}

func newLSTMFuncs(dt tensor.DType, batch, hidden int) lstmFuncs {
	scale, shift, unshift := slots(dt, 1, 1, 2, 1), slots(dt, 0, 0, -1, 0), slots(dt, 0, 0, 1, 0)
	isH := slots(dt, 1, 0)
	isI, isF, isO := slots(dt, 1, 0, 0, 0), slots(dt, 0, 1, 0, 0), slots(dt, 0, 0, 0, 1)
	one, zero := tensor.Full(dt, 1), tensor.Full(dt, 0)
	// Steps read slots of their inputs broadcast over the slots of their
	// results: a value computed for one slot and broadcast would be a
	// kernel of its own.
	two := func(x *tensor.Tensor) *tensor.Tensor { return x.Expand(batch, 2, hidden) }
	four := func(x *tensor.Tensor) *tensor.Tensor { return x.Expand(batch, 4, hidden) }
	return lstmFuncs{
		gates: tensor.Func{
			Forward: func(in []*tensor.Tensor) *tensor.Tensor {
				return in[0].Mul(scale).Sigmoid().Mul(scale).Add(shift)
			},
			Backward: func(g *tensor.Tensor, _ []*tensor.Tensor, y *tensor.Tensor) []*tensor.Tensor {
				return []*tensor.Tensor{g.Mul(y.Add(unshift)).Mul(one.Sub(y))}
			},
		},
		cell: tensor.Func{
			Forward: func(in []*tensor.Tensor) *tensor.Tensor {
				// Both slots compute the new cell, so that the step is one
				// kernel.
				act, s := in[0], in[1]
				c := two(slot(act, 1)).Mul(two(slot(s, 1))).Add(two(slot(act, 0)).Mul(two(slot(act, 2))))
				return tensor.Where(isH, two(slot(act, 3)).Mul(c.Tanh()), c)
			},
			Backward: func(g *tensor.Tensor, in []*tensor.Tensor, out *tensor.Tensor) []*tensor.Tensor {
				// tanh(c), a kernel of its own, then the gradients of the
				// gates and of the state, a kernel each.
				act, s := in[0], in[1]
				tc := slot(out, 1).Tanh()
				dc := func(n func(*tensor.Tensor) *tensor.Tensor) *tensor.Tensor {
					t := n(tc)
					return n(slot(g, 1)).Add(n(slot(g, 0)).Mul(n(slot(act, 3))).Mul(one.Sub(t.Mul(t))))
				}
				by := tensor.Where(isI, four(slot(act, 2)), tensor.Where(isF, four(slot(s, 1)), four(slot(act, 0))))
				dact := tensor.Where(isO, four(slot(g, 0)).Mul(four(tc)), dc(four).Mul(by))
				ds := tensor.Where(isH, zero, dc(two).Mul(two(slot(act, 1))))
				return []*tensor.Tensor{dact, ds}
			},
		},
		pick: tensor.Func{
			Forward: func(in []*tensor.Tensor) *tensor.Tensor {
				return slot(in[0], 0).Reshape(batch, hidden)
			},
			Backward: func(g *tensor.Tensor, _ []*tensor.Tensor, _ *tensor.Tensor) []*tensor.Tensor {
				return []*tensor.Tensor{tensor.Where(isH, g.Reshape(batch, 1, hidden), zero)}
			},
		},
	}
}

// newGRUFunc returns a GRU step over a batch of batch states of hidden
// values in dtype dt: from the input projection and the hidden
// projection of the step, [batch, 3*hidden] each, and the state, to the
// next state:
//
//	r = σ(x_r + h_r), z = σ(x_z + h_z), n = tanh(x_n + r h_n)
//	h' = (1-z) n + z h
//
// Its gradient recomputes r, z and n rather than keep them.
func newGRUFunc(dt tensor.DType, batch, hidden int) tensor.Func {
	isR, isZ := slots(dt, 1, 0, 0), slots(dt, 0, 1, 0)
	one := tensor.Full(dt, 1)
	gates := func(in []*tensor.Tensor) (x, hp, h, r, z, n *tensor.Tensor) {
		x, hp = in[0].Reshape(batch, 3, hidden), in[1].Reshape(batch, 3, hidden)
		h = in[2].Reshape(batch, 1, hidden)
		r = slot(x, 0).Add(slot(hp, 0)).Sigmoid()
		z = slot(x, 1).Add(slot(hp, 1)).Sigmoid()
		n = slot(x, 2).Add(r.Mul(slot(hp, 2))).Tanh()
		return x, hp, h, r, z, n
	}
	return tensor.Func{
		Forward: func(in []*tensor.Tensor) *tensor.Tensor {
			_, _, h, _, z, n := gates(in)
			return n.Add(z.Mul(h.Sub(n))).Reshape(batch, hidden)
		},
		Backward: func(g *tensor.Tensor, in []*tensor.Tensor, _ *tensor.Tensor) []*tensor.Tensor {
			_, hp, h, r, z, n := gates(in)
			g = g.Reshape(batch, 1, hidden)
			dn := g.Mul(one.Sub(z)).Mul(one.Sub(n.Mul(n)))
			dr := dn.Mul(slot(hp, 2)).Mul(r).Mul(one.Sub(r))
			dz := g.Mul(h.Sub(n)).Mul(z).Mul(one.Sub(z))
			dx := tensor.Where(isR, dr, tensor.Where(isZ, dz, dn))
			dhp := tensor.Where(isR, dr, tensor.Where(isZ, dz, dn.Mul(r)))
			return []*tensor.Tensor{dx.Reshape(batch, 3*hidden), dhp.Reshape(batch, 3*hidden), g.Mul(z).Reshape(batch, hidden)}
		},
	}
}
