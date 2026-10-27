package cluster

import (
	"fmt"
	"math"
	"slices"

	"github.com/gsoultan/mimir/tensor"
)

// Module is a KMeans or Mixture grouping as an nn.Module, so it runs,
// saves and loads wherever a network does. Forward maps x, shape [batch,
// Features], to [batch, K] logits whose softmax is how likely each group
// is, and whose largest is the row's group: for k-means, minus half the
// squared standardized distance to each center; for a mixture, each
// group's log-likelihood and log weight.
type Module struct {
	Mean   *tensor.Param // Float32 [1, Features]
	Scale  *tensor.Param // Float32 [1, Features]: 1/std, or 0 for a constant feature
	Center *tensor.Param // Float32 [K, Features]
	// Prec is each group's precision per feature, 1/variance, and Bias
	// its log weight less its normalizing constant; for k-means Prec is 1
	// and Bias 0.
	Prec  *tensor.Param // Float32 [K, Features]
	Bias  *tensor.Param // Float32 [1, K]
	shape Shape
}

// Shape is what sizes a Module.
type Shape struct{ Features, K int }

// NewModule returns a Module of shape s whose tables are zero, for saved
// weights to be loaded into.
func NewModule(s Shape) *Module {
	f, k := max(s.Features, 1), max(s.K, 1)
	return &Module{
		Mean:   tensor.NewBuffer(tensor.Zeros(tensor.Float32, 1, f)),
		Scale:  tensor.NewBuffer(tensor.Zeros(tensor.Float32, 1, f)),
		Center: tensor.NewBuffer(tensor.Zeros(tensor.Float32, k, f)),
		Prec:   tensor.NewBuffer(tensor.Zeros(tensor.Float32, k, f)),
		Bias:   tensor.NewBuffer(tensor.Zeros(tensor.Float32, 1, k)),
		shape:  s,
	}
}

func module(sc []float32, std []float32, centers, prec, bias []float32, f, k int) *Module {
	scale := make([]float32, f)
	for j, s := range std {
		if s != 0 {
			scale[j] = 1 / s
		}
	}
	return &Module{
		Mean:   tensor.NewBuffer(tensor.FromSlice(slices.Clone(sc), 1, f)),
		Scale:  tensor.NewBuffer(tensor.FromSlice(scale, 1, f)),
		Center: tensor.NewBuffer(tensor.FromSlice(slices.Clone(centers), k, f)),
		Prec:   tensor.NewBuffer(tensor.FromSlice(prec, k, f)),
		Bias:   tensor.NewBuffer(tensor.FromSlice(bias, 1, k)),
		shape:  Shape{Features: f, K: k},
	}
}

// Shape returns the shape of m's Module.
func (m *Means) Shape() Shape { return Shape{Features: m.Features, K: m.K()} }

// Module returns m as a Module.
func (m *Means) Module() *Module {
	k, f := m.K(), m.Features
	prec := make([]float32, k*f)
	for i := range prec {
		prec[i] = 1
	}
	return module(m.Scaler.Mean, m.Scaler.Std, m.Centers, prec, make([]float32, k), f, k)
}

// Shape returns the shape of m's Module.
func (m *Mixture) Shape() Shape { return Shape{Features: m.Features, K: m.K()} }

// Module returns m as a Module.
func (m *Mixture) Module() *Module {
	k, f := m.K(), m.Features
	prec, bias := make([]float32, k*f), make([]float32, k)
	for a := range k {
		b := math.Log(math.Max(float64(m.Weights[a]), 1e-300))
		for j := range f {
			v := float64(m.Vars[a*f+j])
			prec[a*f+j] = float32(1 / v)
			b -= 0.5 * math.Log(2*math.Pi*v)
		}
		bias[a] = float32(b)
	}
	return module(m.Scaler.Mean, m.Scaler.Std, m.Means, prec, bias, f, k)
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
		return tensor.Failed(fmt.Errorf("cluster: input %v for rows of %d features: %w", x.Shape(), s.Features, tensor.ErrShape))
	}
	n := x.Shape()[0]
	z := x.Sub(mod.Mean.Value()).Mul(mod.Scale.Value())
	z = tensor.Where(z.Equal(z), z, tensor.Scalar[float32](0)) // a missing value is the mean
	// Σ_j p_kj (z_j - c_kj)² = z²·pᵀ - 2 z·(p∘c)ᵀ + Σ_j p_kj c_kj²
	p, c := mod.Prec.Value(), mod.Center.Value()
	quad := z.Mul(z).MatMulT(p).Sub(z.MatMulT(p.Mul(c)).Mul(tensor.Scalar[float32](2))).Add(p.Mul(c).Mul(c).Sum(1).Reshape(1, s.K))
	return mod.Bias.Value().Expand(n, s.K).Sub(quad.Mul(tensor.Scalar[float32](0.5)))
}
