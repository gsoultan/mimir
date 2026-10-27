package neighbors

import (
	"fmt"
	"slices"

	"github.com/gsoultan/mimir/tensor"
)

// Shape is what sizes a Module: enough to build an empty one that saved
// weights load into.
type Shape struct {
	Rows, Features int
	Classes        int // 0 for Regression
	K              int
	Distance       bool
	Metric         string
}

// Module is a Model as an nn.Module, so it runs, saves and loads wherever
// a network does. Forward maps x, shape [batch, Features], to [batch,
// Classes] logits whose softmax is the model's probabilities, or to
// [batch, 1] values for Regression: it finds each row's K nearest with
// TopK over the distances to every kept row.
type Module struct {
	Mean *tensor.Param // Float32 [1, Features]
	Std  *tensor.Param // Float32 [1, Features]
	X    *tensor.Param // Float32 [Rows, Features], standardized
	// Y holds each row's class as a one-hot row, [Rows, Classes], or its
	// answer, [Rows, 1].
	Y     *tensor.Param
	shape Shape
}

// NewModule returns a Module of shape s whose tables are zero, for saved
// weights to be loaded into.
func NewModule(s Shape) *Module {
	n, f, k := max(s.Rows, 1), max(s.Features, 1), max(s.Classes, 1)
	z := func(shape ...int) *tensor.Param { return tensor.NewBuffer(tensor.Zeros(tensor.Float32, shape...)) }
	return &Module{Mean: z(1, f), Std: z(1, f), X: z(n, f), Y: z(n, k), shape: s}
}

// Shape returns the shape of m's Module.
func (m *Model) Shape() Shape {
	return Shape{Rows: len(m.Y), Features: m.Features, Classes: m.Classes, K: m.K, Distance: m.Distance, Metric: m.Metric}
}

// Module returns m as a Module.
func (m *Model) Module() *Module {
	s := m.Shape()
	k := max(m.Classes, 1)
	y := make([]float32, len(m.Y)*k)
	for i, v := range m.Y {
		if m.Classes > 0 {
			y[i*k+int(v)] = 1
		} else {
			y[i] = v
		}
	}
	buf := func(v []float32, shape ...int) *tensor.Param {
		return tensor.NewBuffer(tensor.FromSlice(slices.Clone(v), shape...))
	}
	return &Module{
		Mean: buf(m.Mean, 1, m.Features), Std: buf(m.Std, 1, m.Features),
		X: buf(m.X, len(m.Y), m.Features), Y: buf(y, len(m.Y), k), shape: s,
	}
}

// Shape returns the shape mod was built with.
func (mod *Module) Shape() Shape { return mod.shape }

// Forward answers each row of x, shape [batch, Features].
func (mod *Module) Forward(x *tensor.Tensor) *tensor.Tensor {
	if err := x.Err(); err != nil {
		return x
	}
	s := mod.shape
	if x.Rank() != 2 || x.Shape()[1] != s.Features {
		return tensor.Failed(fmt.Errorf("neighbors: input %v for rows of %d features: %w", x.Shape(), s.Features, tensor.ErrShape))
	}
	f32 := func(v float64) *tensor.Tensor { return tensor.Scalar(float32(v)) }
	b := x.Shape()[0]
	z := tensor.Where(x.Equal(x), x.Sub(mod.Mean.Value()).Div(mod.Std.Value()), tensor.Zeros(tensor.Float32))
	rows := mod.X.Value()
	var d *tensor.Tensor // [batch, Rows]
	switch s.Metric {
	case Manhattan:
		d = z.Unsqueeze(1).Sub(rows.Unsqueeze(0)).Abs().Sum(2)
	case Cosine:
		nz := z.Mul(z).Sum(1).Sqrt().Unsqueeze(1)
		nr := rows.Mul(rows).Sum(1).Sqrt().Unsqueeze(0)
		cos := z.MatMulT(rows).Div(nz.Mul(nr).Maximum(f32(1e-30)))
		d = f32(1).Sub(cos)
	default:
		zz := z.Mul(z).Sum(1).Unsqueeze(1)
		rr := rows.Mul(rows).Sum(1).Unsqueeze(0)
		d = zz.Add(rr).Sub(z.MatMulT(rows).Mul(f32(2))).Maximum(f32(0)).Sqrt()
	}
	dist, idx := d.TopK(s.K, 1, false) // [batch, K]
	w := tensor.Full(tensor.Float32, 1).Expand(b, s.K)
	if s.Distance {
		w = dist.Maximum(f32(1e-9)).Recip()
	}
	cols := mod.Y.Value().Shape()[1]
	n := rows.Shape()[0]
	picked := mod.Y.Value().Unsqueeze(0).Expand(b, n, cols).Gather(1, idx.Unsqueeze(2).Expand(b, s.K, cols)) // [batch, K, cols]
	out := picked.Mul(w.Unsqueeze(2)).Sum(1).Div(w.Sum(1).Unsqueeze(1))
	if s.Classes == 0 {
		return out
	}
	return out.Add(f32(1e-12)).Log()
}
