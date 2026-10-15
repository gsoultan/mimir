package cpu

import (
	"context"
	"fmt"
	"slices"

	"github.com/gsoultan/mimir/backend"
)

// Run executes one kernel. Inputs are read through k.In; the outputs are
// written contiguously. Any panic inside a kernel is reported as
// backend.ErrInternal instead of crashing the program.
func (b *Backend) Run(ctx context.Context, k *backend.Kernel, outs []backend.Buffer, ins []backend.Buffer) (err error) {
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
	if !ok || len(outs) != 1+len(k.Attrs.Stores) || (k.Attrs.Stores != nil && k.Op != backend.OpFused) {
		return fmt.Errorf("cpu: %s: %d outputs for %d stored values: %w", k.Op, len(outs), len(k.Attrs.Stores), backend.ErrInternal)
	}
	for i, out := range outs {
		if out.Len() != n || out.DType() != k.DType || slices.Contains(outs[:i], out) {
			return fmt.Errorf("cpu: %s: output %d of %s[%d] for %s%v: %w", k.Op, i, out.DType(), out.Len(), k.DType, k.Shape, backend.ErrInternal)
		}
	}
	if err := checkViews(k, ins); err != nil {
		return err
	}
	if err := checkAliases(k, outs, ins); err != nil {
		return err
	}
	if err := b.run(ctx, k, outs, ins); err != nil {
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

// checkAliases verifies that an output is also an input only where Run
// allows it: an elementwise kernel reading that input contiguously, as the
// output is written.
func checkAliases(k *backend.Kernel, outs, ins []backend.Buffer) error {
	for i, in := range ins {
		if !slices.Contains(outs, in) {
			continue
		}
		if v := k.In[i]; !elementwise(k.Op) || !v.IsContiguous() || v.Offset != 0 || !slices.Equal(v.Shape, k.Shape) {
			return fmt.Errorf("cpu: %s: input %d is also an output: %w", k.Op, i, backend.ErrInternal)
		}
	}
	return nil
}

// elementwise reports whether op runs as an elementwise program.
func elementwise(op backend.Op) bool {
	return op == backend.OpFused || op.IsElementwise() || op == backend.OpContiguous
}

func (b *Backend) run(ctx context.Context, k *backend.Kernel, outs []backend.Buffer, ins []backend.Buffer) error {
	out := outs[0]
	switch {
	case k.Op == backend.OpFused:
		return runProgram(ctx, b, k, k.Attrs.Program, k.Attrs.Stores, outs, ins)
	case elementwise(k.Op):
		return runProgram(ctx, b, k, singleOp(k, ins), nil, outs, ins)
	case k.Op == backend.OpMatMul:
		epi, err := epilogue(k, out, ins)
		if err != nil {
			return err
		}
		switch k.DType {
		case backend.Float32:
			return runMatMul[float32](ctx, b, k, out, ins, epi)
		case backend.Int32:
			return runMatMul[int32](ctx, b, k, out, ins, epi)
		}
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
	case op == backend.OpReduceSum:
		return reduceSum(ctx, b, k, out, in[0])
	case op == backend.OpReduceMax:
		return reduceMax(ctx, b, k, out, in[0])
	case op == backend.OpPad:
		return pad(ctx, b, k, out, in[0])
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
	case op == backend.OpUnfold:
		return unfold(ctx, b, k, out, in[0])
	case op == backend.OpFold:
		return fold(ctx, b, k, out, in[0])
	default:
		return fmt.Errorf("no cpu kernel: %w", backend.ErrInternal)
	}
}
