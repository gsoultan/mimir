package recommend

import (
	"fmt"
	"slices"

	"github.com/gsoultan/mimir/tensor"
)

// Shape is what sizes a Module.
type Shape struct{ Users, Items, Factors int }

// Module is a Model as an nn.Module, so it runs, saves and loads wherever
// a network does. Forward maps x, shape [batch, 1], each a user's index,
// to [batch, Items] scores, as Model.Score.
type Module struct {
	User  *tensor.Param // Float32 [Users, Factors]
	Item  *tensor.Param // Float32 [Items, Factors]
	Mean  *tensor.Param // Float32 [1, 1]
	shape Shape
}

// NewModule returns a Module of shape s whose factors are zero, for saved
// weights to be loaded into.
func NewModule(s Shape) *Module {
	return &Module{
		User:  tensor.NewBuffer(tensor.Zeros(tensor.Float32, max(s.Users, 1), max(s.Factors, 1))),
		Item:  tensor.NewBuffer(tensor.Zeros(tensor.Float32, max(s.Items, 1), max(s.Factors, 1))),
		Mean:  tensor.NewBuffer(tensor.Zeros(tensor.Float32, 1, 1)),
		shape: s,
	}
}

// Shape returns the shape of m's Module.
func (m *Model) Shape() Shape { return Shape{Users: m.Users, Items: m.Items, Factors: m.Factors} }

// Module returns m as a Module.
func (m *Model) Module() *Module {
	return &Module{
		User:  tensor.NewBuffer(tensor.FromSlice(slices.Clone(m.UserFactors), m.Users, m.Factors)),
		Item:  tensor.NewBuffer(tensor.FromSlice(slices.Clone(m.ItemFactors), m.Items, m.Factors)),
		Mean:  tensor.NewBuffer(tensor.FromSlice([]float32{m.Mean}, 1, 1)),
		shape: m.Shape(),
	}
}

// Shape returns the shape mod was built with.
func (mod *Module) Shape() Shape { return mod.shape }

// Forward scores every item for each user in x.
func (mod *Module) Forward(x *tensor.Tensor) *tensor.Tensor {
	if err := x.Err(); err != nil {
		return x
	}
	s := mod.shape
	if x.Rank() != 2 || x.Shape()[1] != 1 {
		return tensor.Failed(fmt.Errorf("recommend: input %v, want [batch, 1] user indices: %w", x.Shape(), tensor.ErrShape))
	}
	n := x.Shape()[0]
	idx := x.Cast(tensor.Int32).Expand(n, s.Factors)
	u := mod.User.Value().Gather(0, idx) // [batch, Factors]
	return u.MatMulT(mod.Item.Value()).Add(mod.Mean.Value())
}
