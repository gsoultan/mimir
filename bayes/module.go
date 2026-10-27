package bayes

import (
	"fmt"
	"slices"

	"github.com/gsoultan/mimir/tensor"
)

// Shape is what sizes a Module: enough to build an empty one that saved
// weights load into.
type Shape struct {
	Kind              string
	Features, Classes int
	Binarize          float32
}

// Module is a Model as an nn.Module, so it runs, saves and loads wherever
// a network does. Forward maps x, shape [batch, Features], to [batch,
// Classes] log-likelihoods, logits whose softmax is the model's
// probabilities. Its tables are buffers, not trained parameters.
type Module struct {
	Prior  *tensor.Param // Float32 [1, Classes]
	Square *tensor.Param // Float32 [Classes, Features]
	Linear *tensor.Param // Float32 [Classes, Features]
	Const  *tensor.Param // Float32 [Classes, Features]
	shape  Shape
}

// NewModule returns a Module of shape s whose tables are zero, for saved
// weights to be loaded into.
func NewModule(s Shape) *Module {
	k, f := max(s.Classes, 1), max(s.Features, 1)
	z := func(shape ...int) *tensor.Param { return tensor.NewBuffer(tensor.Zeros(tensor.Float32, shape...)) }
	return &Module{Prior: z(1, k), Square: z(k, f), Linear: z(k, f), Const: z(k, f), shape: s}
}

// Shape returns the shape of m's Module.
func (m *Model) Shape() Shape {
	return Shape{Kind: m.Kind, Features: m.Features, Classes: m.Classes, Binarize: m.Binarize}
}

// Module returns m as a Module.
func (m *Model) Module() *Module {
	buf := func(v []float32, shape ...int) *tensor.Param {
		return tensor.NewBuffer(tensor.FromSlice(slices.Clone(v), shape...))
	}
	k, f := m.Classes, m.Features
	return &Module{Prior: buf(m.Prior, 1, k), Square: buf(m.Square, k, f), Linear: buf(m.Linear, k, f), Const: buf(m.Const, k, f), shape: m.Shape()}
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
		return tensor.Failed(fmt.Errorf("bayes: input %v for rows of %d features: %w", x.Shape(), s.Features, tensor.ErrShape))
	}
	present := x.Equal(x)
	zero := tensor.Zeros(tensor.Float32)
	v := tensor.Where(present, x, zero)
	if s.Kind == "bernoulli" {
		v = v.Greater(tensor.Scalar(s.Binarize)).Cast(tensor.Float32)
	}
	mask := present.Cast(tensor.Float32)
	out := v.MatMulT(mod.Linear.Value()).Add(mask.MatMulT(mod.Const.Value())).Add(mod.Prior.Value())
	if s.Kind == "gaussian" {
		out = out.Add(v.Mul(v).MatMulT(mod.Square.Value()))
	}
	return out
}
