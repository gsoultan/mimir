package svm

import (
	"fmt"
	"slices"

	"github.com/gsoultan/mimir/tabular"
	"github.com/gsoultan/mimir/tensor"
)

// Shape is what sizes a Module: enough to build an empty one that saved
// weights load into.
type Shape struct {
	Features, Support, Outputs int
	Classes                    int // 0 for Regression
	Kernel                     string
	Gamma, Coef0               float64
	Degree                     int
	YMean, YStd                float32
}

// Module is a Model as an nn.Module, so it runs, saves and loads wherever
// a network does. Forward maps x, shape [batch, Features], to [batch,
// Classes] logits whose softmax is the model's probabilities, or to
// [batch, 1] values for Regression. Its tables are buffers, not trained
// parameters.
type Module struct {
	Mean    *tensor.Param // Float32 [1, Features]
	Std     *tensor.Param // Float32 [1, Features]
	Support *tensor.Param // Float32 [Support, Features]
	Coef    *tensor.Param // Float32 [Outputs, Support]
	Bias    *tensor.Param // Float32 [1, Outputs]
	A, B    *tensor.Param // Float32 [1, Outputs], Platt's scaling
	shape   Shape
}

// NewModule returns a Module of shape s whose tables are zero, for saved
// weights to be loaded into.
func NewModule(s Shape) *Module {
	f, sv, k := max(s.Features, 1), max(s.Support, 1), max(s.Outputs, 1)
	z := func(shape ...int) *tensor.Param { return tensor.NewBuffer(tensor.Zeros(tensor.Float32, shape...)) }
	return &Module{Mean: z(1, f), Std: z(1, f), Support: z(sv, f), Coef: z(k, sv), Bias: z(1, k), A: z(1, k), B: z(1, k), shape: s}
}

// Shape returns the shape of m's Module.
func (m *Model) Shape() Shape {
	return Shape{
		Features: m.Features, Support: m.SupportVectors(), Outputs: m.Outputs, Classes: m.Classes,
		Kernel: m.Kernel, Gamma: m.Gamma, Coef0: m.Coef0, Degree: m.Degree, YMean: m.YMean, YStd: m.YStd,
	}
}

// Module returns m as a Module.
func (m *Model) Module() *Module {
	s := m.Shape()
	buf := func(v []float32, shape ...int) *tensor.Param {
		if len(v) == 0 {
			v = make([]float32, shape[0]*shape[1])
		}
		return tensor.NewBuffer(tensor.FromSlice(slices.Clone(v), shape...))
	}
	k, sv := max(m.Outputs, 1), max(s.Support, 1)
	return &Module{
		Mean: buf(m.Mean, 1, m.Features), Std: buf(m.Std, 1, m.Features),
		Support: buf(m.Support, sv, m.Features), Coef: buf(m.Coef, k, sv), Bias: buf(m.Bias, 1, k),
		A: buf(m.A, 1, k), B: buf(m.B, 1, k), shape: s,
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
		return tensor.Failed(fmt.Errorf("svm: input %v for rows of %d features: %w", x.Shape(), s.Features, tensor.ErrShape))
	}
	f32 := func(v float64) *tensor.Tensor { return tensor.Scalar(float32(v)) }
	z := x.Sub(mod.Mean.Value()).Div(mod.Std.Value())
	z = tensor.Where(x.Equal(x), z, tensor.Zeros(tensor.Float32))
	sv := mod.Support.Value()
	dot := z.MatMulT(sv) // [batch, Support]
	var k *tensor.Tensor
	switch s.Kernel {
	case RBF:
		zz := z.Mul(z).Sum(1).Unsqueeze(1)
		ss := sv.Mul(sv).Sum(1).Unsqueeze(0)
		d := zz.Add(ss).Sub(dot.Mul(f32(2))).Maximum(f32(0))
		k = d.Mul(f32(-s.Gamma)).Exp()
	case Polynomial:
		base := dot.Mul(f32(s.Gamma)).Add(f32(s.Coef0))
		k = base
		for range s.Degree - 1 {
			k = k.Mul(base)
		}
	case Sigmoid:
		k = dot.Mul(f32(s.Gamma)).Add(f32(s.Coef0)).Tanh()
	default:
		k = dot
	}
	dec := k.MatMulT(mod.Coef.Value()).Add(mod.Bias.Value()) // [batch, Outputs]
	if s.Classes == 0 {
		return dec.Mul(f32(float64(s.YStd))).Add(f32(float64(s.YMean)))
	}
	logit := dec.Mul(mod.A.Value()).Add(mod.B.Value())
	if s.Outputs == 1 {
		return tensor.Cat(1, tensor.Zeros(tensor.Float32, x.Shape()[0], 1), logit)
	}
	// log sigmoid, whose softmax is the normalized probabilities.
	neg := logit.Neg()
	softplus := neg.Maximum(f32(0)).Add(f32(1).Add(neg.Abs().Neg().Exp()).Log())
	return softplus.Neg()
}

var _ tabular.Model = (*Model)(nil)
