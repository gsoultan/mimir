package backend

import (
	"fmt"
	"slices"
)

// CheckRun verifies the arguments of Backend.Run against the contract every
// backend shares: one output per value the kernel writes, each holding the
// elements of k.Shape in k.DType; one buffer per input view, every element
// of which lies inside its buffer; and an output that is also an input
// only where an elementwise kernel reads that input as the output is
// written, or a slice scatter writes its block into input 0. Kernels can
// then index without further bounds checks.
func CheckRun(k *Kernel, outs, ins []Buffer) error {
	if len(ins) != len(k.In) {
		return fmt.Errorf("%s: %d buffers for %d views: %w", k.Op, len(ins), len(k.In), ErrInternal)
	}
	n, ok := NumElements(k.Shape)
	if !ok || len(outs) != 1+len(k.Attrs.Stores) || (k.Attrs.Stores != nil && k.Op != OpFused) {
		return fmt.Errorf("%s: %d outputs for %d stored values: %w", k.Op, len(outs), len(k.Attrs.Stores), ErrInternal)
	}
	for i, out := range outs {
		if out.Len() != n || out.DType() != k.DType || slices.Contains(outs[:i], out) {
			return fmt.Errorf("%s: output %d of %s[%d] for %s%v: %w", k.Op, i, out.DType(), out.Len(), k.DType, k.Shape, ErrInternal)
		}
	}
	for i, v := range k.In {
		if len(v.Shape) != len(v.Strides) {
			return fmt.Errorf("%s: input %d view rank mismatch: %w", k.Op, i, ErrInternal)
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
			return fmt.Errorf("%s: input %d view [%d, %d] outside buffer of %d: %w", k.Op, i, lo, hi, ins[i].Len(), ErrInternal)
		}
	}
	for i, in := range ins {
		if !slices.Contains(outs, in) {
			continue
		}
		inPlace := k.Op.RunsElementwise() || k.Op == OpSliceScatter && i == 0
		if v := k.In[i]; !inPlace || !v.IsContiguous() || v.Offset != 0 || !slices.Equal(v.Shape, k.Shape) {
			return fmt.Errorf("%s: input %d is also an output: %w", k.Op, i, ErrInternal)
		}
	}
	return nil
}

// RunsElementwise reports whether a kernel of op computes each output
// element from the input elements at the same index: OpFused, OpContiguous
// or an elementwise op.
func (o Op) RunsElementwise() bool {
	return o == OpFused || o == OpContiguous || o.IsElementwise()
}

// Coalesce merges the dimensions of shape that every view walks as one,
// and drops dimensions of size 1, so that kernels loop over as few
// dimensions as possible. Each view walks shape; the returned views walk
// the returned shape over the same elements in the same order.
func Coalesce(shape []int, views []View) ([]int, []View) {
	dims := make([]int, 0, len(shape))
	out := make([]View, len(views))
	for v, view := range views {
		out[v] = View{Strides: make([]int, 0, len(shape)), Offset: view.Offset}
	}
	for d, n := range shape {
		if n == 1 {
			continue
		}
		last := len(dims) - 1
		merge := last >= 0 && n != 0 && dims[last] != 0
		for v := range views {
			merge = merge && out[v].Strides[last] == views[v].Strides[d]*n
		}
		if merge {
			dims[last] *= n
			for v := range views {
				out[v].Strides[last] = views[v].Strides[d]
			}
			continue
		}
		dims = append(dims, n)
		for v := range views {
			out[v].Strides = append(out[v].Strides, views[v].Strides[d])
		}
	}
	for v := range out {
		out[v].Shape = dims
	}
	return dims, out
}
