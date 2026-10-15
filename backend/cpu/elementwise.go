package cpu

import (
	"context"
	"fmt"
	"math"

	"github.com/gsoultan/mimir/backend"
)

// elementGrain is the smallest chunk of elementwise work worth a goroutine.
const elementGrain = 1 << 15

// forRows splits the output of shape into rows along the innermost
// dimension and calls fn(i, offs, steps, n) for each: the row covers output
// elements [i, i+n), offs holds where each view's row starts and steps each
// view's innermost stride. Strided and broadcast inputs then cost one
// cursor step per row instead of per element. Dimensions are coalesced
// first, so rows are as long as the views allow, and long rows are split
// between workers.
func forRows(ctx context.Context, b *Backend, shape []int, views []backend.View, fn func(i int, offs, steps []int, n int)) error {
	shape, views = coalesce(shape, views)
	r := len(shape)
	if r == 0 {
		offs := make([]int, len(views))
		for v, view := range views {
			offs[v] = view.Offset
		}
		fn(0, offs, make([]int, len(views)), 1)
		return nil
	}
	inner := shape[r-1]
	total, _ := backend.NumElements(shape)
	if total == 0 {
		return nil
	}
	outer := make([]backend.View, len(views))
	steps := make([]int, len(views))
	for v, view := range views {
		outer[v] = backend.View{Strides: view.Strides[:r-1], Offset: view.Offset}
		steps[v] = view.Strides[r-1]
	}
	return b.parallelFor(ctx, total, elementGrain, func(lo, hi int) error {
		c := newCursor(shape[:r-1], outer, lo/inner)
		offs := make([]int, len(views))
		for i := lo; i < hi; c.next() {
			col := i % inner
			n := min(inner-col, hi-i)
			for v := range offs {
				offs[v] = c.offs[v] + col*steps[v]
			}
			fn(i, offs, steps, n)
			i += n
		}
		return nil
	})
}

// mapStrided calls fn(i, off) for every output element i of shape, where off
// is the buffer offset of element i under view v.
func mapStrided(ctx context.Context, b *Backend, shape []int, v backend.View, n int, fn func(i, off int)) error {
	if v.IsContiguous() {
		return b.parallelFor(ctx, n, elementGrain, func(lo, hi int) error {
			for i := lo; i < hi; i++ {
				fn(i, v.Offset+i)
			}
			return nil
		})
	}
	return forRows(ctx, b, shape, []backend.View{v}, func(i int, offs, steps []int, n int) {
		off, s := offs[0], steps[0]
		for j := i; j < i+n; j++ {
			fn(j, off)
			off += s
		}
	})
}

func unary[T elem](ctx context.Context, b *Backend, k *backend.Kernel, out, in []T, f func(T) T) error {
	v := k.In[0]
	if v.IsContiguous() {
		src := in[v.Offset : v.Offset+len(out)]
		return b.parallelFor(ctx, len(out), elementGrain, func(lo, hi int) error {
			for i := lo; i < hi; i++ {
				out[i] = f(src[i])
			}
			return nil
		})
	}
	return forRows(ctx, b, k.Shape, k.In, func(i int, offs, steps []int, n int) {
		off, s := offs[0], steps[0]
		for j := range out[i : i+n] {
			out[i+j] = f(in[off])
			off += s
		}
	})
}

func binary[T elem](ctx context.Context, b *Backend, k *backend.Kernel, out, x, y []T, f func(a, b T) T) error {
	vx, vy := k.In[0], k.In[1]
	if vx.IsContiguous() && vy.IsContiguous() {
		xs := x[vx.Offset : vx.Offset+len(out)]
		ys := y[vy.Offset : vy.Offset+len(out)]
		return b.parallelFor(ctx, len(out), elementGrain, func(lo, hi int) error {
			for i := lo; i < hi; i++ {
				out[i] = f(xs[i], ys[i])
			}
			return nil
		})
	}
	return forRows(ctx, b, k.Shape, k.In, func(i int, offs, steps []int, n int) {
		ox, oy, sx, sy := offs[0], offs[1], steps[0], steps[1]
		dst := out[i : i+n]
		for j := range dst {
			dst[j] = f(x[ox], y[oy])
			ox += sx
			oy += sy
		}
	})
}

func where[C, T elem](ctx context.Context, b *Backend, k *backend.Kernel, out []T, cond []C, x, y []T) error {
	return forRows(ctx, b, k.Shape, k.In, func(i int, offs, steps []int, n int) {
		oc, ox, oy := offs[0], offs[1], offs[2]
		dst := out[i : i+n]
		for j := range dst {
			if cond[oc] != 0 {
				dst[j] = x[ox]
			} else {
				dst[j] = y[oy]
			}
			oc += steps[0]
			ox += steps[1]
			oy += steps[2]
		}
	})
}

func unaryFunc[T elem](op backend.Op) (func(T) T, error) {
	if op == backend.OpNeg {
		return func(x T) T { return -x }, nil
	}
	var f func(float64) float64
	switch op {
	case backend.OpExp2:
		f = math.Exp2
	case backend.OpLog2:
		f = math.Log2
	case backend.OpSin:
		f = math.Sin
	case backend.OpSqrt:
		f = math.Sqrt
	case backend.OpRecip:
		f = func(x float64) float64 { return 1 / x }
	default:
		return nil, fmt.Errorf("no unary kernel for %s: %w", op, backend.ErrInternal)
	}
	if dtypeOf[T]() != backend.Float32 {
		return nil, backend.ErrDType
	}
	return func(x T) T { return T(f(float64(x))) }, nil
}

func binaryFunc[T elem](op backend.Op) func(a, b T) T {
	switch op {
	case backend.OpAdd:
		return func(a, b T) T { return a + b }
	case backend.OpMul:
		return func(a, b T) T { return a * b }
	case backend.OpMax:
		return maxOf[T]
	case backend.OpCmpLT:
		return func(a, b T) T { return boolTo[T](a < b) }
	case backend.OpCmpEQ:
		return func(a, b T) T { return boolTo[T](a == b) }
	default: // OpMod
		if dtypeOf[T]() == backend.Float32 {
			return func(a, b T) T { return T(math.Mod(float64(a), float64(b))) }
		}
		return func(a, b T) T { return T(int32(a) % int32(b)) }
	}
}

// maxOf returns the larger of a and b, propagating NaN like PyTorch.
func maxOf[T elem](a, b T) T {
	if a != a {
		return a
	}
	if b != b || b > a {
		return b
	}
	return a
}

func boolTo[T elem](v bool) T {
	if v {
		return 1
	}
	return 0
}

// checkNonZero rejects integer modulo by zero. Float modulo by zero is NaN.
func checkNonZero[T elem](shape []int, v backend.View, y []T) error {
	if dtypeOf[T]() == backend.Float32 {
		return nil
	}
	n, _ := backend.NumElements(shape)
	c := newCursor(shape, []backend.View{v}, 0)
	for range n {
		if y[c.offs[0]] == 0 {
			return backend.ErrDivideByZero
		}
		c.next()
	}
	return nil
}
