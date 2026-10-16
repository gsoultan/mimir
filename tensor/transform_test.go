package tensor_test

import (
	"errors"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/internal/sched"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/nn/loss"
	"github.com/gsoultan/mimir/tensor"
)

// inputs returns the inputs of c, shifted by d, as constants.
func (c gradCase) inputs(d float32) []*tensor.Tensor {
	x := make([]*tensor.Tensor, len(c.in))
	for i, v := range c.in {
		p := slices.Clone(v)
		for j := range p {
			p[j] += d
		}
		x[i] = tensor.FromSlice(p, c.dims[i]...)
	}
	return x
}

// tangents returns a direction for each input of c, with values in
// [-1, 1).
func (c gradCase) tangents() []*tensor.Tensor {
	r := rand.New(rand.NewPCG(3, 4))
	t := make([]*tensor.Tensor, len(c.in))
	for i, v := range c.in {
		t[i] = tensor.FromSlice(vals(r, len(v), -1, 1), c.dims[i]...)
	}
	return t
}

// along returns x + h·t.
func along(x, t []*tensor.Tensor, h float32) []*tensor.Tensor {
	out := make([]*tensor.Tensor, len(x))
	for i := range x {
		out[i] = x[i].Add(t[i].Mul(tensor.Scalar(h)))
	}
	return out
}

// loss returns c's function made scalar as weighted makes it.
func (c gradCase) loss(x ...*tensor.Tensor) *tensor.Tensor {
	out := c.f(x)
	return out.Mul(tensor.Rand(99, out.Shape()...).Add(tensor.Scalar[float32](0.5))).Sum()
}

// near fails t unless got and want agree to within atol + rtol·|want|
// elementwise.
func near(t *testing.T, what string, got, want *tensor.Tensor, atol, rtol float64) {
	t.Helper()
	if !slices.Equal(got.Shape(), want.Shape()) {
		t.Fatalf("%s: shape %v, want %v", what, got.Shape(), want.Shape())
	}
	g, w := data[float32](t, got.Cast(tensor.Float32)), data[float32](t, want.Cast(tensor.Float32))
	bad := 0
	for i := range g {
		if d := math.Abs(float64(g[i] - w[i])); !(d <= atol+rtol*math.Abs(float64(w[i]))) {
			if bad++; bad <= 3 {
				t.Errorf("%s: element %d is %v, want %v", what, i, g[i], w[i])
			}
		}
	}
	if bad > 3 {
		t.Errorf("%s: %d elements differ", what, bad)
	}
}

// notDifferentiableTwice are the cases whose gradient rule gives first
// derivatives only.
var notDifferentiableTwice = []string{"custom_func"}

func TestJVPMatchesFiniteDifferences(t *testing.T) {
	const h = 1e-2
	for _, c := range gradCases() {
		t.Run(c.name, func(t *testing.T) {
			x, v := c.inputs(0), c.tangents()
			f := func(x ...*tensor.Tensor) []*tensor.Tensor { return []*tensor.Tensor{c.f(x)} }
			y, ty := tensor.JVP(f, x, v)
			near(t, "result", y[0], c.f(x), 1e-6, 1e-6)
			want := c.f(along(x, v, h)).Sub(c.f(along(x, v, -h))).Div(tensor.Scalar[float32](2 * h))
			near(t, "tangent", ty[0], want, 5e-3, 1e-2)
		})
	}
}

// TestHessianVectorProductsMatchFiniteDifferences checks the two ways to
// a second derivative, forward over reverse and reverse over reverse,
// through every differentiable operation: the product of the Hessian of
// each case's loss with a direction v is the derivative of its gradient
// along v.
func TestHessianVectorProductsMatchFiniteDifferences(t *testing.T) {
	const h = 1e-2
	for _, c := range gradCases() {
		t.Run(c.name, func(t *testing.T) {
			x, v := c.inputs(0), c.tangents()
			grad := tensor.Grad(c.loss)
			_, hvp := tensor.JVP(grad, x, v)
			if slices.Contains(notDifferentiableTwice, c.name) {
				if err := hvp[0].Err(); !errors.Is(err, tensor.ErrNotDifferentiable) {
					t.Fatalf("HVP through a first-order rule: %v, want ErrNotDifferentiable", err)
				}
				return
			}
			// Reverse over reverse: the gradient of the gradient's
			// product with v.
			xs := make([]*tensor.Tensor, len(x))
			for i := range x {
				xs[i] = x[i].Clone().RequiresGrad()
			}
			g := grad(xs...)
			var dot *tensor.Tensor
			for i := range g {
				d := g[i].Mul(v[i]).Sum()
				if dot == nil {
					dot = d
				} else {
					dot = dot.Add(d)
				}
			}
			rr := tensor.Gradients(dot, xs...)
			gp, gm := grad(along(x, v, h)...), grad(along(x, v, -h)...)
			for i := range x {
				want := gp[i].Sub(gm[i]).Div(tensor.Scalar[float32](2 * h))
				near(t, "forward over reverse", hvp[i], want, 1e-2, 2e-2)
				near(t, "reverse over reverse", rr[i], hvp[i], 1e-4, 1e-3)
			}
		})
	}
}

func TestGradMatchesBackward(t *testing.T) {
	for _, c := range gradCases() {
		t.Run(c.name, func(t *testing.T) {
			x := c.inputs(0)
			got := tensor.Grad(c.loss)(x...)
			xs := make([]*tensor.Tensor, len(x))
			for i := range x {
				xs[i] = x[i].Clone().RequiresGrad()
			}
			if err := c.loss(xs...).Backward(t.Context()); err != nil {
				t.Fatal(err)
			}
			for i := range x {
				near(t, "gradient", got[i], xs[i].Grad(), 1e-6, 1e-5)
			}
		})
	}
}

// batch returns the inputs of c for n examples, stacked along a first
// axis, and those of each example.
func (c gradCase) batch(n int) ([]*tensor.Tensor, [][]*tensor.Tensor) {
	each := make([][]*tensor.Tensor, n)
	for k := range each {
		each[k] = c.inputs(0.01 * float32(k))
	}
	stacked := make([]*tensor.Tensor, len(c.in))
	for i := range stacked {
		xs := make([]*tensor.Tensor, n)
		for k := range xs {
			xs[k] = each[k][i]
		}
		stacked[i] = tensor.Stack(0, xs...)
	}
	return stacked, each
}

// TestVMapMatchesLoop checks every operation's batching rule, and those
// of their gradients, by mapping each case and its gradient over a batch
// and comparing them with the cases computed one example at a time.
func TestVMapMatchesLoop(t *testing.T) {
	const n = 3
	for _, c := range gradCases() {
		t.Run(c.name, func(t *testing.T) {
			x, each := c.batch(n)
			f := func(x ...*tensor.Tensor) []*tensor.Tensor { return []*tensor.Tensor{c.f(x)} }
			y := tensor.VMap(f)(x...)
			gs := tensor.VMap(tensor.Grad(c.loss))(x...)
			want := make([]*tensor.Tensor, n)
			wantGrads := make([][]*tensor.Tensor, len(x))
			for k := range n {
				want[k] = c.f(each[k])
				for i, g := range tensor.Grad(c.loss)(each[k]...) {
					wantGrads[i] = append(wantGrads[i], g)
				}
			}
			near(t, "result", y[0], tensor.Stack(0, want...), 1e-5, 1e-5)
			for i := range x {
				near(t, "gradient", gs[i], tensor.Stack(0, wantGrads[i]...), 1e-5, 1e-4)
			}
		})
	}
}

// TestVMapOfHVPMatchesLoop maps a second derivative, which batches the
// gradient rules' own operations, over a batch of directions.
func TestVMapOfHVPMatchesLoop(t *testing.T) {
	for _, c := range gradCases() {
		if slices.Contains(notDifferentiableTwice, c.name) {
			continue
		}
		t.Run(c.name, func(t *testing.T) {
			x := c.inputs(0)
			v := [][]*tensor.Tensor{c.tangents(), along(c.tangents(), x, 0.5)}
			hvp := func(v ...*tensor.Tensor) []*tensor.Tensor {
				_, h := tensor.JVP(tensor.Grad(c.loss), x, v)
				return h
			}
			stacked := make([]*tensor.Tensor, len(x))
			for i := range x {
				stacked[i] = tensor.Stack(0, v[0][i], v[1][i])
			}
			got := tensor.VMap(hvp)(stacked...)
			for i := range x {
				near(t, "HVP", got[i], tensor.Stack(0, hvp(v[0]...)[i], hvp(v[1]...)[i]), 1e-4, 1e-3)
			}
		})
	}
}

func TestVMapAxesAndNesting(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 6))
	a := tensor.FromSlice(vals(r, 2*3*4, -1, 1), 2, 3, 4)
	w := tensor.FromSlice(vals(r, 4*5, -1, 1), 4, 5)
	f := func(x ...*tensor.Tensor) []*tensor.Tensor {
		return []*tensor.Tensor{x[0].MatMul(w).Tanh().Sum(0), x[0].Max()}
	}
	// Mapped over axis 1, a's three [2, 4] slices; the first result is
	// stacked along its last axis.
	got := tensor.VMap(f, tensor.InAxes(1), tensor.OutAxes(-1, 0))(a)
	var want0, want1 []*tensor.Tensor
	for k := range 3 {
		y := f(a.Shrink([][2]int{{0, 2}, {k, k + 1}, {0, 4}}).Squeeze(1))
		want0, want1 = append(want0, y[0]), append(want1, y[1])
	}
	near(t, "out axis -1", got[0], tensor.Stack(-1, want0...), 1e-6, 1e-6)
	near(t, "out axis 0", got[1], tensor.Stack(0, want1...), 1e-6, 1e-6)

	// Nested: the outer map over a's first axis, the inner over the
	// second, and an argument mapped by the outer map only.
	b := tensor.FromSlice(vals(r, 2*4, -1, 1), 2, 4)
	inner := func(x ...*tensor.Tensor) []*tensor.Tensor { return []*tensor.Tensor{x[0].Mul(x[1]).Exp().Sum()} }
	outer := func(x ...*tensor.Tensor) []*tensor.Tensor {
		return tensor.VMap(func(y ...*tensor.Tensor) []*tensor.Tensor { return inner(y[0], x[1]) })(x[0])
	}
	nested := tensor.VMap(outer)(a, b)[0]
	var rows []*tensor.Tensor
	for i := range 2 {
		var row []*tensor.Tensor
		for j := range 3 {
			row = append(row, inner(a.Shrink([][2]int{{i, i + 1}, {j, j + 1}, {0, 4}}).Reshape(4), b.Shrink([][2]int{{i, i + 1}, {0, 4}}).Reshape(4))[0])
		}
		rows = append(rows, tensor.Stack(0, row...))
	}
	near(t, "nested", nested, tensor.Stack(0, rows...), 1e-5, 1e-5)

	// The batch of a result that does not depend on the arguments is
	// broadcast.
	c := tensor.VMap(func(...*tensor.Tensor) []*tensor.Tensor { return []*tensor.Tensor{w.Sum()} })(a)[0]
	near(t, "constant", c, w.Sum().Expand(2), 0, 0)
}

// TestPerExampleGradients computes the norms of the gradients of each
// example's loss with respect to the parameters of an MLP with VMap, all
// at once, and checks them against Backward run on one example at a time.
func TestPerExampleGradients(t *testing.T) {
	ctx := t.Context()
	nn.ManualSeed(9)
	m := nn.NewSequential(nn.NewLinear(6, 16), nn.ReLU(), nn.NewLinear(16, 4))
	ps := nn.Parameters(m)
	const n = 8
	x := tensor.Rand(9, n, 6)
	labels := tensor.FromSlice([]int32{0, 1, 2, 3, 3, 2, 1, 0}, n)

	norms := func(x, label *tensor.Tensor) *tensor.Tensor {
		l := loss.CrossEntropy(m.Forward(x.Unsqueeze(0)), label.Unsqueeze(0))
		values := make([]*tensor.Tensor, len(ps))
		for i, p := range ps {
			values[i] = p.Value()
		}
		var sq *tensor.Tensor
		for _, g := range tensor.Gradients(l, values...) {
			s := g.Mul(g).Sum()
			if sq == nil {
				sq = s
			} else {
				sq = sq.Add(s)
			}
		}
		return sq.Sqrt()
	}
	got := tensor.VMap(func(in ...*tensor.Tensor) []*tensor.Tensor { return []*tensor.Tensor{norms(in[0], in[1])} })(x, labels)[0]

	want := make([]float32, n)
	for k := range n {
		xk := x.Shrink([][2]int{{k, k + 1}, {0, 6}})
		lk := labels.Shrink([][2]int{{k, k + 1}})
		if err := loss.CrossEntropy(m.Forward(xk), lk).Backward(ctx); err != nil {
			t.Fatal(err)
		}
		var sq float64
		for _, p := range ps {
			for _, v := range data[float32](t, p.Grad()) {
				sq += float64(v) * float64(v)
			}
			p.ZeroGrad()
		}
		want[k] = float32(math.Sqrt(sq))
	}
	near(t, "per-example gradient norms", got, tensor.FromSlice(want), 1e-5, 1e-4)
	for _, p := range ps {
		if p.Grad() != nil {
			t.Fatal("VMap of Gradients changed an accumulated gradient")
		}
	}
}

// TestPerExampleGradientsOfCapturedWeight takes the gradients of a
// weight the function reads, not an argument, for examples that are
// vectors and matrices, against Grad looped over the examples, and
// checks that mismatched shapes fail rather than panic.
func TestPerExampleGradientsOfCapturedWeight(t *testing.T) {
	w := tensor.Randn(1, 3, 2).RequiresGrad()
	loss := func(x ...*tensor.Tensor) *tensor.Tensor { return x[0].MatMul(w).Tanh().Sum() }
	perExample := tensor.VMap(func(x ...*tensor.Tensor) []*tensor.Tensor { return tensor.Gradients(loss(x...), w) })
	for name, batch := range map[string]*tensor.Tensor{
		"vectors":  tensor.Randn(2, 5, 3),
		"matrices": tensor.Randn(2, 5, 4, 3),
	} {
		t.Run(name, func(t *testing.T) {
			got := perExample(batch)[0]
			want := make([]*tensor.Tensor, 5)
			for k := range want {
				xk := batch.Shrink(append([][2]int{{k, k + 1}}, full(batch.Shape()[1:])...)).Squeeze(0)
				want[k] = tensor.Grad(func(v ...*tensor.Tensor) *tensor.Tensor {
					return xk.MatMul(v[0]).Tanh().Sum()
				})(w)[0]
			}
			near(t, "per-example gradients", got, tensor.Stack(0, want...), 1e-5, 1e-4)
			if w.Grad() != nil {
				t.Fatal("VMap of Gradients changed an accumulated gradient")
			}
		})
	}

	// A product of a batch with an unbatched matrix of the wrong shape, on
	// either side, fails with ErrShape.
	bad := tensor.Randn(3, 4, 2)
	for name, f := range map[string]func(x ...*tensor.Tensor) []*tensor.Tensor{
		"batch on the left": func(x ...*tensor.Tensor) []*tensor.Tensor {
			return []*tensor.Tensor{x[0].MatMul(bad)}
		},
		"batch on the right": func(x ...*tensor.Tensor) []*tensor.Tensor {
			return []*tensor.Tensor{bad.MatMul(x[0])}
		},
	} {
		if err := tensor.VMap(f)(tensor.Randn(2, 5, 4, 3))[0].Err(); !errors.Is(err, tensor.ErrShape) {
			t.Errorf("%s: %v, want ErrShape", name, err)
		}
	}
}

// full returns the ranges of the whole of each dimension of shape.
func full(shape []int) [][2]int {
	out := make([][2]int, len(shape))
	for i, n := range shape {
		out[i] = [2]int{0, n}
	}
	return out
}

func TestJacobianAndHessian(t *testing.T) {
	r := rand.New(rand.NewPCG(7, 8))
	w := tensor.FromSlice(vals(r, 3*5, -1, 1), 3, 5)
	x3 := tensor.FromSlice(vals(r, 3, -1, 1), 3)
	x5 := tensor.FromSlice(vals(r, 2*5, -1, 1), 2, 5)
	const h = 1e-2
	// fd is the Jacobian by central differences, a column at a time.
	fd := func(f func(*tensor.Tensor) *tensor.Tensor, x *tensor.Tensor) *tensor.Tensor {
		n := x.Size()
		cols := make([]*tensor.Tensor, n)
		for j := range n {
			e := make([]float32, n)
			e[j] = h
			d := tensor.FromSlice(e, x.Shape()...)
			cols[j] = f(x.Add(d)).Sub(f(x.Sub(d))).Div(tensor.Scalar[float32](2 * h))
		}
		y := f(x)
		return tensor.Stack(-1, cols...).Reshape(slices.Concat(y.Shape(), x.Shape())...)
	}
	// Forward mode: 3 inputs, 5 results.
	wide := func(x *tensor.Tensor) *tensor.Tensor { return x.Sin().MatMul(w).Exp() }
	near(t, "Jacobian by columns", tensor.Jacobian(wide, x3), fd(wide, x3), 2e-3, 1e-2)
	// Reverse mode: 10 inputs, 2 results.
	tall := func(x *tensor.Tensor) *tensor.Tensor { return x.MatMul(w.Transpose(0, 1)).Tanh().Sum(1) }
	near(t, "Jacobian by rows", tensor.Jacobian(tall, x5), fd(tall, x5), 2e-3, 1e-2)

	f := func(x *tensor.Tensor) *tensor.Tensor { return x.Sin().MatMul(w).Exp().Sum().Log() }
	hess := tensor.Hessian(f, x3)
	grad := func(x *tensor.Tensor) *tensor.Tensor {
		return tensor.Grad(func(x ...*tensor.Tensor) *tensor.Tensor { return f(x[0]) })(x)[0]
	}
	near(t, "Hessian", hess, fd(grad, x3), 2e-3, 1e-2)
	near(t, "Hessian symmetry", hess, hess.Transpose(0, 1), 1e-5, 1e-5)

	if err := tensor.Hessian(wide, x3).Err(); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("Hessian of a vector function: %v, want ErrShape", err)
	}
}

func TestTransformErrors(t *testing.T) {
	ctx := t.Context()
	x := tensor.FromSlice([]float32{1, 2, 3})
	id := func(x ...*tensor.Tensor) []*tensor.Tensor { return x }
	sum := func(x ...*tensor.Tensor) *tensor.Tensor { return x[0].Sum() }

	// Not batchable: reading values inside VMap.
	for name, f := range map[string]func(x *tensor.Tensor) error{
		"Data":     func(x *tensor.Tensor) error { _, err := x.Data[float32](ctx); return err },
		"Item":     func(x *tensor.Tensor) error { _, err := x.Sum().Item[float32](ctx); return err },
		"Realize":  func(x *tensor.Tensor) error { return tensor.Realize(ctx, x) },
		"Backward": func(x *tensor.Tensor) error { return x.Sum().Backward(ctx) },
		"Param.Set": func(x *tensor.Tensor) error {
			return tensor.NewParam(tensor.Zeros(tensor.Float32, 3)).Set(x)
		},
	} {
		var err error
		tensor.VMap(func(x ...*tensor.Tensor) []*tensor.Tensor { err = f(x[0]); return x })(x.Reshape(3, 1))
		if !errors.Is(err, tensor.ErrNotBatchable) {
			t.Errorf("%s inside VMap: %v, want ErrNotBatchable", name, err)
		}
	}
	for name, y := range map[string][]*tensor.Tensor{
		"mismatched batch":  tensor.VMap(id)(x, tensor.Ones(tensor.Float32, 2)),
		"in axes":           tensor.VMap(id, tensor.InAxes(0, 0))(x),
		"out axes":          tensor.VMap(id, tensor.OutAxes(0, 0))(x),
		"axis out of range": tensor.VMap(id, tensor.InAxes(1))(x),
		"no arguments":      tensor.VMap(func(...*tensor.Tensor) []*tensor.Tensor { return []*tensor.Tensor{x} })(),
	} {
		if err := y[0].Err(); !errors.Is(err, tensor.ErrShape) {
			t.Errorf("VMap with %s: %v, want ErrShape", name, err)
		}
	}

	// Not differentiable twice: a Func not marked Differentiable.
	g := tensor.Grad(func(x ...*tensor.Tensor) *tensor.Tensor { return square.Apply(x[0]).Sum() })
	xr := x.Clone().RequiresGrad()
	if err := tensor.Gradients(g(xr)[0].Sum(), xr)[0].Err(); !errors.Is(err, tensor.ErrNotDifferentiable) {
		t.Errorf("second derivative through a first-order Func: %v, want ErrNotDifferentiable", err)
	}
	twice := square
	twice.Differentiable = true
	h := tensor.Hessian(func(x *tensor.Tensor) *tensor.Tensor { return twice.Apply(x).Sum() }, x)
	near(t, "Hessian through a differentiable Func", h, tensor.FromSlice([]float32{2, 0, 0, 0, 2, 0, 0, 0, 2}, 3, 3), 1e-6, 0)

	// Arguments.
	if err := tensor.Grad(sum)(tensor.FromSlice([]int32{1}))[0].Err(); !errors.Is(err, tensor.ErrDType) {
		t.Errorf("Grad of an Int32 argument: %v, want ErrDType", err)
	}
	if err := tensor.Grad(func(x ...*tensor.Tensor) *tensor.Tensor { return x[0] })(x)[0].Err(); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("Grad of a vector function: %v, want ErrShape", err)
	}
	if _, ty := tensor.JVP(id, []*tensor.Tensor{x}, []*tensor.Tensor{tensor.Ones(tensor.Float32, 2)}); !errors.Is(ty[0].Err(), tensor.ErrShape) {
		t.Errorf("JVP with a tangent of another shape: %v, want ErrShape", ty[0].Err())
	}
	if _, ty := tensor.JVP(id, []*tensor.Tensor{x}, nil); !errors.Is(ty[0].Err(), tensor.ErrShape) {
		t.Errorf("JVP without tangents: %v, want ErrShape", ty[0].Err())
	}
	if _, vjp := tensor.VJP(id, x); !errors.Is(vjp()[0].Err(), tensor.ErrShape) {
		t.Errorf("VJP without gradients: %v, want ErrShape", vjp()[0].Err())
	}
	if err := tensor.Gradients(x.Sum(), x)[0].Err(); !errors.Is(err, tensor.ErrNoGrad) {
		t.Errorf("Gradients with respect to a constant: %v, want ErrNoGrad", err)
	}
	var null *tensor.Tensor
	if err := tensor.Grad(sum)(null)[0].Err(); !errors.Is(err, tensor.ErrNilTensor) {
		t.Errorf("Grad of nil: %v, want ErrNilTensor", err)
	}
	if err := tensor.Grad(func(...*tensor.Tensor) *tensor.Tensor { return nil })(x)[0].Err(); !errors.Is(err, tensor.ErrNilTensor) {
		t.Errorf("Grad of a function returning nil: %v, want ErrNilTensor", err)
	}
}

// TestTransformsDoNotChangeState checks that the transformations leave
// accumulated gradients and the graph alone: a Backward after them sees
// the graph as it was.
func TestTransformsDoNotChangeState(t *testing.T) {
	ctx := t.Context()
	w := tensor.FromSlice([]float32{1, 2, 3}).RequiresGrad()
	y := w.Mul(w).Sum()
	g := tensor.Gradients(y, w)[0]
	if w.Grad() != nil {
		t.Fatal("Gradients accumulated a gradient")
	}
	near(t, "gradient", g, tensor.FromSlice([]float32{2, 4, 6}), 0, 0)
	if err := y.Backward(ctx); err != nil {
		t.Fatal(err)
	}
	near(t, "Backward after Gradients", w.Grad(), tensor.FromSlice([]float32{2, 4, 6}), 0, 0)
	// The gradient computed before Backward released the graph is still
	// a function of w.
	if err := tensor.Gradients(g.Sum(), w)[0].Err(); err != nil {
		t.Errorf("Gradients of a gradient after Backward: %v", err)
	}
	if err := tensor.Gradients(y, w)[0].Err(); !errors.Is(err, tensor.ErrNoGrad) {
		t.Errorf("Gradients through a graph Backward released: %v, want ErrNoGrad", err)
	}
}

// TestTransformsWithScheduleCache runs a per-example HVP repeatedly with
// the schedule cache and without it: the replays the cache serves compute
// what planning anew does.
func TestTransformsWithScheduleCache(t *testing.T) {
	r := rand.New(rand.NewPCG(9, 10))
	w := tensor.FromSlice(vals(r, 4*3, -1, 1), 4, 3).RequiresGrad()
	f := func(x ...*tensor.Tensor) *tensor.Tensor { return x[0].MatMul(w).Tanh().Mul(x[0].Sum()).Sum() }
	step := func(i int) *tensor.Tensor {
		x := tensor.FromSlice(vals(rand.New(rand.NewPCG(uint64(i), 0)), 5*4, -1, 1), 5, 4)
		v := tensor.Ones(tensor.Float32, 5, 4)
		hvp := tensor.VMap(func(in ...*tensor.Tensor) []*tensor.Tensor {
			_, h := tensor.JVP(tensor.Grad(f), in[:1], in[1:])
			return h
		})(x, v)[0]
		return hvp.Add(tensor.Gradients(f(x), w)[0].Sum())
	}
	run := func(caching bool) [][]float32 {
		defer sched.SetCaching(sched.SetCaching(caching))
		var out [][]float32
		for i := range 4 {
			out = append(out, data[float32](t, step(i%2)))
		}
		return out
	}
	planned, cached := run(false), run(true)
	for i := range planned {
		if !slices.Equal(planned[i], cached[i]) {
			t.Errorf("step %d: cached schedule computes %v, planned %v", i, cached[i], planned[i])
		}
		if i >= 2 && !slices.Equal(cached[i], cached[i-2]) {
			t.Errorf("step %d: replay computes %v, first run %v", i, cached[i], cached[i-2])
		}
	}
}

// TestTransformsOfFactorizations checks what the transformations do with
// the matrix factorizations, whose gradient rules hold to first order:
// their forward derivatives come from transposing those rules, while
// second derivatives fail with ErrNotDifferentiable, and VMap, which has
// no batching rule for them, fails with ErrNotBatchable.
func TestTransformsOfFactorizations(t *testing.T) {
	r := rand.New(rand.NewPCG(15, 16))
	m := tensor.FromSlice(vals(r, 3*3, -1, 1), 3, 3)
	spd := func(x *tensor.Tensor) *tensor.Tensor {
		return x.MatMul(x.Transpose(0, 1)).Add(tensor.FromSlice([]float32{1, 0, 0, 0, 1, 0, 0, 0, 1}, 3, 3))
	}
	v := tensor.FromSlice(vals(r, 3*3, -1, 1), 3, 3)
	for name, f := range map[string]func(x *tensor.Tensor) *tensor.Tensor{
		"cholesky": func(x *tensor.Tensor) *tensor.Tensor { return spd(x).Cholesky() },
		"inv":      func(x *tensor.Tensor) *tensor.Tensor { return spd(x).Inv() },
		"det":      func(x *tensor.Tensor) *tensor.Tensor { return spd(x).Det() },
		"qr": func(x *tensor.Tensor) *tensor.Tensor {
			_, r := spd(x).QR()
			return r
		},
		"svd": func(x *tensor.Tensor) *tensor.Tensor {
			_, s, _ := spd(x).SVD()
			return s
		},
		"eigh": func(x *tensor.Tensor) *tensor.Tensor {
			w, _ := spd(x).Eigh()
			return w
		},
	} {
		t.Run(name, func(t *testing.T) {
			fs := func(x ...*tensor.Tensor) []*tensor.Tensor { return []*tensor.Tensor{f(x[0])} }
			_, ty := tensor.JVP(fs, []*tensor.Tensor{m}, []*tensor.Tensor{v})
			const h = 1e-2
			want := f(m.Add(v.Mul(tensor.Scalar[float32](h)))).Sub(f(m.Sub(v.Mul(tensor.Scalar[float32](h))))).Div(tensor.Scalar[float32](2 * h))
			// Rules that compute their gradients with the factorizations
			// themselves, which record nothing, cannot be transposed.
			if err := ty[0].Err(); err != nil {
				if !errors.Is(err, tensor.ErrNotDifferentiable) {
					t.Fatalf("JVP: %v, want a tangent or ErrNotDifferentiable", err)
				}
				t.Logf("JVP: %v", err)
			} else {
				near(t, "JVP", ty[0], want, 5e-3, 1e-2)
			}

			loss := func(x ...*tensor.Tensor) *tensor.Tensor { return f(x[0]).Sum() }
			if _, hv := tensor.JVP(tensor.Grad(loss), []*tensor.Tensor{m}, []*tensor.Tensor{v}); !errors.Is(hv[0].Err(), tensor.ErrNotDifferentiable) {
				t.Errorf("Hessian-vector product: %v, want ErrNotDifferentiable", hv[0].Err())
			}
			if err := tensor.VMap(fs)(tensor.Stack(0, m, v))[0].Err(); !errors.Is(err, tensor.ErrNotBatchable) {
				t.Errorf("VMap: %v, want ErrNotBatchable", err)
			}
			if err := tensor.VMap(tensor.Grad(loss))(tensor.Stack(0, m, v))[0].Err(); !errors.Is(err, tensor.ErrNotBatchable) {
				t.Errorf("VMap of Grad: %v, want ErrNotBatchable", err)
			}
			// Batched cotangents of an unbatched matrix.
			scaled := func(c ...*tensor.Tensor) []*tensor.Tensor {
				return tensor.Grad(func(x ...*tensor.Tensor) *tensor.Tensor { return f(x[0]).Mul(c[0]).Sum() })(m)
			}
			if err := tensor.VMap(scaled)(tensor.FromSlice([]float32{1, 2}, 2))[0].Err(); !errors.Is(err, tensor.ErrNotBatchable) {
				t.Errorf("VMap of Grad with batched cotangents: %v, want ErrNotBatchable", err)
			}
		})
	}
}

// TestTransformsThroughCheckpoint differentiates checkpointed functions
// twice and forward, nested and reading their surroundings' arguments,
// and compares with the same functions without Checkpoint.
func TestTransformsThroughCheckpoint(t *testing.T) {
	r := rand.New(rand.NewPCG(11, 12))
	w := tensor.FromSlice(vals(r, 4*4, -1, 1), 4, 4).RequiresGrad()
	x := tensor.FromSlice(vals(r, 3*4, 0.5, 1), 3, 4)
	v := tensor.FromSlice(vals(r, 3*4, -1, 1), 3, 4)
	block := func(h *tensor.Tensor) *tensor.Tensor { return h.MatMul(w).Tanh() }
	plain := func(x ...*tensor.Tensor) *tensor.Tensor { return block(block(x[0]).Mul(x[0])).Sum() }
	for name, f := range map[string]func(x ...*tensor.Tensor) *tensor.Tensor{
		"one": func(x ...*tensor.Tensor) *tensor.Tensor {
			return tensor.Checkpoint(func(in ...*tensor.Tensor) *tensor.Tensor { return block(block(in[0]).Mul(in[0])) }, x[0]).Sum()
		},
		// The inner function reads the outer one's input, and the outer
		// one reads the argument of the transformation, from closures.
		"nested capturing": func(x ...*tensor.Tensor) *tensor.Tensor {
			return tensor.Checkpoint(func(in ...*tensor.Tensor) *tensor.Tensor {
				h := tensor.Checkpoint(func(j ...*tensor.Tensor) *tensor.Tensor { return block(j[0]).Mul(in[0]) }, in[0])
				return block(h.Mul(x[0]).Div(in[0]))
			}, x[0].Mul(tensor.Scalar[float32](1))).Sum()
		},
	} {
		t.Run(name, func(t *testing.T) {
			near(t, "gradient", tensor.Grad(f)(x)[0], tensor.Grad(plain)(x)[0], 1e-6, 1e-5)
			_, got := tensor.JVP(tensor.Grad(f), []*tensor.Tensor{x}, []*tensor.Tensor{v})
			_, want := tensor.JVP(tensor.Grad(plain), []*tensor.Tensor{x}, []*tensor.Tensor{v})
			near(t, "HVP", got[0], want[0], 1e-5, 1e-4)
			fv := func(x ...*tensor.Tensor) []*tensor.Tensor { return []*tensor.Tensor{f(x...)} }
			pv := func(x ...*tensor.Tensor) []*tensor.Tensor { return []*tensor.Tensor{plain(x...)} }
			_, got = tensor.JVP(fv, []*tensor.Tensor{x}, []*tensor.Tensor{v})
			_, want = tensor.JVP(pv, []*tensor.Tensor{x}, []*tensor.Tensor{v})
			near(t, "JVP", got[0], want[0], 1e-5, 1e-4)
			// With respect to the parameter the blocks read.
			gw := tensor.Gradients(tensor.Grad(f)(x)[0].Sum(), w)[0]
			near(t, "mixed second derivative", gw, tensor.Gradients(tensor.Grad(plain)(x)[0].Sum(), w)[0], 1e-5, 1e-4)
		})
	}
}

// TestDerivativesThroughVMap differentiates the results of VMap with
// respect to its arguments and to what the function reads, with Backward
// and with Grad, and compares them with the batch computed directly.
func TestDerivativesThroughVMap(t *testing.T) {
	ctx := t.Context()
	r := rand.New(rand.NewPCG(13, 14))
	wv, xv := vals(r, 4*3, -1, 1), vals(r, 5*4, -1, 1)
	f := func(w *tensor.Tensor) func(x ...*tensor.Tensor) []*tensor.Tensor {
		return func(x ...*tensor.Tensor) []*tensor.Tensor {
			y := x[0].MatMul(w).Tanh().Mul(x[0].Sum()).Sum()
			return []*tensor.Tensor{y.Mul(y)}
		}
	}
	direct := func(x, w *tensor.Tensor) *tensor.Tensor {
		y := x.MatMul(w).Tanh().Mul(x.Sum(1).Unsqueeze(1)).Sum(1)
		return y.Mul(y).Sum()
	}
	w := tensor.FromSlice(wv, 4, 3).RequiresGrad()
	x := tensor.FromSlice(xv, 5, 4).RequiresGrad()
	if err := tensor.VMap(f(w))(x)[0].Sum().Backward(ctx); err != nil {
		t.Fatal(err)
	}
	w2 := tensor.FromSlice(wv, 4, 3).RequiresGrad()
	x2 := tensor.FromSlice(xv, 5, 4).RequiresGrad()
	if err := direct(x2, w2).Backward(ctx); err != nil {
		t.Fatal(err)
	}
	near(t, "Backward: argument", x.Grad(), x2.Grad(), 1e-5, 1e-5)
	near(t, "Backward: captured", w.Grad(), w2.Grad(), 1e-5, 1e-5)

	g := tensor.Grad(func(in ...*tensor.Tensor) *tensor.Tensor {
		return tensor.VMap(f(in[1]))(in[0])[0].Sum()
	})(x.Detach(), w.Detach())
	want := tensor.Grad(func(in ...*tensor.Tensor) *tensor.Tensor { return direct(in[0], in[1]) })(x.Detach(), w.Detach())
	near(t, "Grad: argument", g[0], want[0], 1e-5, 1e-5)
	near(t, "Grad: captured", g[1], want[1], 1e-5, 1e-5)

	// A Func inside VMap: its rule gives the derivatives taken inside,
	// and one taken outside fails.
	sq := func(x ...*tensor.Tensor) []*tensor.Tensor { return []*tensor.Tensor{square.Apply(x[0]).Sum()} }
	per := tensor.VMap(func(x ...*tensor.Tensor) []*tensor.Tensor {
		return tensor.Grad(func(x ...*tensor.Tensor) *tensor.Tensor { return sq(x...)[0] })(x...)
	})(x.Detach())[0]
	near(t, "Grad of a Func inside VMap", per, x.Detach().Mul(tensor.Scalar[float32](2)), 1e-6, 0)
	if err := tensor.VMap(sq)(x)[0].Sum().Backward(ctx); !errors.Is(err, tensor.ErrNotBatchable) {
		t.Errorf("Backward through a Func inside VMap: %v, want ErrNotBatchable", err)
	}
}
