package cpu

import (
	"context"
	"fmt"
	"runtime"

	"github.com/gsoultan/mimir/backend"
)

// Run executes one kernel. Inputs are read through k.In; the outputs are
// written contiguously. Any panic inside a kernel is reported as
// backend.ErrInternal instead of crashing the program.
func (b *Backend) Run(ctx context.Context, k *backend.Kernel, outs []backend.Buffer, ins []backend.Buffer) (err error) {
	// A kernel works on the buffers' memory, not the buffers: one that the
	// caller drops must stay reachable until the kernel is done, or the
	// cleanup of an off-heap buffer unmaps it under the kernel.
	defer runtime.KeepAlive(outs)
	defer runtime.KeepAlive(ins)
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("cpu: panic in %s: %v: %w", k.Op, r, backend.ErrInternal)
		}
	}()
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := backend.CheckRun(k, outs, ins); err != nil {
		return fmt.Errorf("cpu: %w", err)
	}
	if err := b.run(ctx, k, outs, ins); err != nil {
		return fmt.Errorf("cpu: %s: %w", k.Op, err)
	}
	return nil
}

func (b *Backend) run(ctx context.Context, k *backend.Kernel, outs []backend.Buffer, ins []backend.Buffer) error {
	out := outs[0]
	switch {
	case k.Op == backend.OpCast && ins[0].DType().IsQuantized():
		return dequantize(ctx, b, k, out, ins[0])
	case k.Op == backend.OpGather && ins[0].DType().IsQuantized():
		return gatherQuant(ctx, b, k, out, ins)
	case k.Op == backend.OpQMatMul:
		return runQMatMul(ctx, b, k, out, ins)
	case k.Op == backend.OpFused:
		return runProgram(ctx, b, k, k.Attrs.Program, k.Attrs.Stores, outs, ins)
	case k.Op.RunsElementwise():
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
		default:
			return fmt.Errorf("%s product: %w", k.DType, backend.ErrDType)
		}
	case k.Op.IsReduce():
		return runReduce(ctx, b, k, out, ins[0])
	case k.Op == backend.OpScan:
		return runScan(ctx, b, k, out, ins[0])
	case k.Op == backend.OpAttention || k.Op == backend.OpAttentionGrad:
		o, err := data[float32](out)
		if err != nil {
			return err
		}
		if k.Op == backend.OpAttention {
			return attention(ctx, b, k, o, ins)
		}
		return attentionGrad(ctx, b, k, o, ins)
	case k.Op == backend.OpConv || k.Op == backend.OpConvWeight || k.Op == backend.OpConvInput:
		return runConv(ctx, b, k, outs, ins)

	case k.Op == backend.OpArgsort:
		return runArgsort(ctx, b, k, out, ins[0])
	case k.Op == backend.OpFFT:
		return runFFT(ctx, b, k, out, ins[0])
	case k.Op >= backend.OpCholesky && k.Op <= backend.OpEigh:
		return runLinalg(ctx, b, k, out, ins)

	case k.Op == backend.OpRandGamma || k.Op == backend.OpRandPoisson || k.Op == backend.OpGammaGrad:
		return runSample(ctx, b, k, out, ins)
	case k.Op == backend.OpSegment:
		return runSegment(ctx, b, k, out, ins)
	}
	switch k.DType {
	case backend.Float32:
		return runTyped[float32](ctx, b, k, out, ins)
	case backend.Int32:
		return runTyped[int32](ctx, b, k, out, ins)
	case backend.BFloat16:
		return runTyped[bf16](ctx, b, k, out, ins)
	case backend.Float16:
		return runTyped[f16](ctx, b, k, out, ins)
	default:
		return backend.ErrDType
	}
}

// runTyped executes ops whose inputs share the output dtype, except for
// Int32 index inputs to Gather, ScatterAdd and Scatter.
func runTyped[T elem](ctx context.Context, b *Backend, k *backend.Kernel, outBuf backend.Buffer, ins []backend.Buffer) error {
	out, err := data[T](outBuf)
	if err != nil {
		return err
	}
	in := make([][]T, len(ins))
	for i, buf := range ins {
		if (k.Op == backend.OpGather || k.Op == backend.OpScatterAdd || k.Op == backend.OpScatter) && i == 1 {
			continue // Int32 indices, read below
		}
		if in[i], err = data[T](buf); err != nil {
			return err
		}
	}

	switch op := k.Op; {
	case op == backend.OpConst:
		v := valueOf[T](k.Attrs.Value)
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
	case op == backend.OpScatter:
		idx, err := data[int32](ins[1])
		if err != nil {
			return err
		}
		return scatter(ctx, b, k, out, in[0], idx, in[2], outBuf == ins[0])
	case op == backend.OpUnfold:
		return unfold(ctx, b, k, out, in[0])
	case op == backend.OpFold:
		return fold(ctx, b, k, out, in[0])
	case op == backend.OpCat:
		return cat(ctx, b, k, out, in)
	case op == backend.OpSliceScatter:
		return sliceScatter(ctx, b, k, out, in[0], in[1], outBuf == ins[0])
	default:
		return fmt.Errorf("no cpu kernel: %w", backend.ErrInternal)
	}
}
