package cpu

import (
	"context"
	"fmt"

	"github.com/gsoultan/mimir/backend"
)

// Run executes one kernel. Inputs are read through k.In; the output is
// written contiguously. Any panic inside a kernel is reported as
// backend.ErrInternal instead of crashing the program.
func (b *Backend) Run(ctx context.Context, k *backend.Kernel, out backend.Buffer, ins []backend.Buffer) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("cpu: panic in %s: %v: %w", k.Op, r, backend.ErrInternal)
		}
	}()
	if err := ctx.Err(); err != nil {
		return err
	}
	if len(ins) != len(k.In) {
		return fmt.Errorf("cpu: %s: %d buffers for %d views: %w", k.Op, len(ins), len(k.In), backend.ErrInternal)
	}
	n, ok := backend.NumElements(k.Shape)
	if !ok || out.Len() != n || out.DType() != k.DType {
		return fmt.Errorf("cpu: %s: output %s[%d] for %s%v: %w", k.Op, out.DType(), out.Len(), k.DType, k.Shape, backend.ErrInternal)
	}
	if err := checkViews(k, ins); err != nil {
		return err
	}
	if err := b.run(ctx, k, out, ins); err != nil {
		return fmt.Errorf("cpu: %s: %w", k.Op, err)
	}
	return nil
}

// checkViews verifies that every element a view can address lies inside its
// buffer, so kernels can index without further bounds checks failing.
func checkViews(k *backend.Kernel, ins []backend.Buffer) error {
	for i, v := range k.In {
		if len(v.Shape) != len(v.Strides) {
			return fmt.Errorf("cpu: %s: input %d view rank mismatch: %w", k.Op, i, backend.ErrInternal)
		}
		lo, hi := v.Offset, v.Offset
		empty := false
		for d, size := range v.Shape {
			if size == 0 {
				empty = true
				break
			}
			span := (size - 1) * v.Strides[d]
			if span < 0 {
				lo += span
			} else {
				hi += span
			}
		}
		if !empty && (lo < 0 || hi >= ins[i].Len()) {
			return fmt.Errorf("cpu: %s: input %d view [%d, %d] outside buffer of %d: %w", k.Op, i, lo, hi, ins[i].Len(), backend.ErrInternal)
		}
	}
	return nil
}

func (b *Backend) run(ctx context.Context, k *backend.Kernel, out backend.Buffer, ins []backend.Buffer) error {
	switch k.Op {
	case backend.OpCast:
		return b.runCast(ctx, k, out, ins[0])
	case backend.OpWhere:
		if k.DType == backend.Float32 {
			return runWhere[float32](ctx, b, k, out, ins)
		}
		return runWhere[int32](ctx, b, k, out, ins)
	}
	switch k.DType {
	case backend.Float32:
		return runTyped[float32](ctx, b, k, out, ins)
	case backend.Int32:
		return runTyped[int32](ctx, b, k, out, ins)
	default:
		return backend.ErrDType
	}
}

// runTyped executes ops whose inputs share the output dtype, except for
// Int32 index inputs to Gather and ScatterAdd.
func runTyped[T elem](ctx context.Context, b *Backend, k *backend.Kernel, outBuf backend.Buffer, ins []backend.Buffer) error {
	out, err := data[T](outBuf)
	if err != nil {
		return err
	}
	in := make([][]T, len(ins))
	for i, buf := range ins {
		if (k.Op == backend.OpGather || k.Op == backend.OpScatterAdd) && i == 1 {
			continue // Int32 indices, read below
		}
		if in[i], err = data[T](buf); err != nil {
			return err
		}
	}

	switch op := k.Op; {
	case op == backend.OpConst:
		v := T(k.Attrs.Value)
		for i := range out {
			out[i] = v
		}
		return nil
	case op == backend.OpRandUniform || op == backend.OpRandNormal:
		f32, ok := any(out).([]float32)
		if !ok {
			return backend.ErrDType
		}
		return b.fillRand(ctx, f32, k.Attrs.Seed, op == backend.OpRandNormal)
	case op == backend.OpContiguous:
		return unary(ctx, b, k, out, in[0], func(x T) T { return x })
	case op.IsUnary():
		f, err := unaryFunc[T](op)
		if err != nil {
			return err
		}
		return unary(ctx, b, k, out, in[0], f)
	case op.IsBinary():
		if op == backend.OpMod {
			if err := checkNonZero(k.Shape, k.In[1], in[1]); err != nil {
				return err
			}
		}
		return binary(ctx, b, k, out, in[0], in[1], binaryFunc[T](op))
	case op == backend.OpReduceSum:
		return reduceSum(ctx, b, k, out, in[0])
	case op == backend.OpReduceMax:
		return reduce(ctx, b, k, out, in[0], maxOf[T])
	case op == backend.OpPad:
		return pad(ctx, b, k, out, in[0])
	case op == backend.OpMatMul:
		return matmul(ctx, b, k, out, in[0], in[1])
	case op == backend.OpGather:
		idx, err := data[int32](ins[1])
		if err != nil {
			return err
		}
		return gather(ctx, b, k, out, in[0], idx)
	case op == backend.OpScatterAdd:
		idx, err := data[int32](ins[1])
		if err != nil {
			return err
		}
		return scatterAdd(ctx, b, k, out, in[0], idx, in[2])
	default:
		return fmt.Errorf("no cpu kernel: %w", backend.ErrInternal)
	}
}

func (b *Backend) runCast(ctx context.Context, k *backend.Kernel, outBuf, inBuf backend.Buffer) error {
	switch inBuf.DType() {
	case backend.Float32:
		return runCastFrom[float32](ctx, b, k, outBuf, inBuf)
	case backend.Int32:
		return runCastFrom[int32](ctx, b, k, outBuf, inBuf)
	default:
		return backend.ErrDType
	}
}

func runCastFrom[F elem](ctx context.Context, b *Backend, k *backend.Kernel, outBuf, inBuf backend.Buffer) error {
	in, err := data[F](inBuf)
	if err != nil {
		return err
	}
	switch k.DType {
	case backend.Float32:
		return castTo[F, float32](ctx, b, k, outBuf, in)
	case backend.Int32:
		return castTo[F, int32](ctx, b, k, outBuf, in)
	default:
		return backend.ErrDType
	}
}

func castTo[F, T elem](ctx context.Context, b *Backend, k *backend.Kernel, outBuf backend.Buffer, in []F) error {
	out, err := data[T](outBuf)
	if err != nil {
		return err
	}
	return mapStrided(ctx, b, k.Shape, k.In[0], len(out), func(i, off int) { out[i] = T(in[off]) })
}

func runWhere[T elem](ctx context.Context, b *Backend, k *backend.Kernel, outBuf backend.Buffer, ins []backend.Buffer) error {
	out, err := data[T](outBuf)
	if err != nil {
		return err
	}
	x, err := data[T](ins[1])
	if err != nil {
		return err
	}
	y, err := data[T](ins[2])
	if err != nil {
		return err
	}
	switch ins[0].DType() {
	case backend.Float32:
		c, _ := data[float32](ins[0])
		return where(ctx, b, k, out, c, x, y)
	case backend.Int32:
		c, _ := data[int32](ins[0])
		return where(ctx, b, k, out, c, x, y)
	default:
		return backend.ErrDType
	}
}
