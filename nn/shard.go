package nn

import (
	"fmt"
	"reflect"
	"unsafe"

	"github.com/gsoultan/mimir/tensor"
)

// ShardRule says how to lay out a parameter of a model over a mesh: it
// returns the PartitionSpec of the parameter or buffer p, named name as
// NamedParams names it, or nil to keep it whole on every device.
type ShardRule func(name string, p *tensor.Param) tensor.PartitionSpec

// Shard places every parameter and buffer of m, with its gradient, on
// mesh as rule says, or whole on every device where rule is nil or gives
// nil (see tensor.Param.Shard). The model then computes on sharded
// tensors: shard its inputs on mesh too. Kept whole and fed a batch split
// over a mesh axis, a model trains data-parallel; split as TensorParallel
// says, tensor-parallel; split by FullyShard, as fully sharded data
// parallelism and ZeRO-3 do, with its gradients and optimizer states
// split alike.
func Shard(m Module, mesh *tensor.Mesh, rule ShardRule) error {
	for name, p := range NamedParams(m) {
		var spec tensor.PartitionSpec
		if rule != nil {
			spec = rule(name, p)
		}
		if err := p.Shard(mesh, spec); err != nil {
			return fmt.Errorf("nn: shard %s as %v: %w", name, spec, err)
		}
	}
	return nil
}

// Rules returns a rule that asks each of rules in turn and takes the
// first spec that is not nil.
func Rules(rules ...ShardRule) ShardRule {
	return func(name string, p *tensor.Param) tensor.PartitionSpec {
		for _, r := range rules {
			if r == nil {
				continue
			}
			if s := r(name, p); s != nil {
				return s
			}
		}
		return nil
	}
}

// FullyShard returns the rule of fully sharded data parallelism, as
// PyTorch's FSDP and ZeRO-3 shard a model: each parameter is split along
// its first axis whose length divides evenly by the size of the mesh
// axis axis, and kept whole if none does. A product then gathers a
// weight's blocks where it is used, and the gradient's partial sums
// reduce-scatter into blocks laid out like the weight, which an optimizer
// updates in place along with its states.
func FullyShard(mesh *tensor.Mesh, axis string) ShardRule {
	k := mesh.Size(axis)
	return func(_ string, p *tensor.Param) tensor.PartitionSpec {
		if k <= 1 || p.Value().DType().IsQuantized() {
			return nil
		}
		for d, n := range p.Value().Shape() {
			if n%k == 0 {
				spec := make([]string, d+1)
				spec[d] = axis
				return tensor.P(spec...)
			}
		}
		return nil
	}
}

// TensorParallel returns the rule of Megatron-LM's tensor parallelism over
// the mesh axis axis for the transformer layers of m: in each MLP and
// GatedMLP the first projections are split by their output features,
// column-parallel, and the last by its input features, row-parallel, so
// the hidden features stay split between them and one all-reduce adds up
// the result; in each SelfAttention and MultiheadAttention the fused
// query, key and value projection is split by heads, each of its
// sections apart, and the output projection row-parallel, so every
// device attends with its own heads. A layer whose heads or hidden
// features do not divide evenly, and every other parameter, such as
// embeddings and normalizations, are kept whole.
func TensorParallel(m Module, mesh *tensor.Mesh, axis string) ShardRule {
	k := mesh.Size(axis)
	specs := map[*tensor.Param]tensor.PartitionSpec{}
	col := tensor.P(axis)
	row := tensor.P("", axis)
	linear := func(l *Linear, spec tensor.PartitionSpec, bias bool) {
		if l == nil {
			return
		}
		specs[l.Weight] = spec
		if bias && l.Bias != nil {
			specs[l.Bias] = spec[:1]
		}
	}
	divides := func(ns ...int) bool {
		for _, n := range ns {
			if k <= 0 || n%k != 0 {
				return false
			}
		}
		return k > 1
	}
	w := walker{seen: map[unsafe.Pointer]bool{}, all: true}
	w.visit(reflect.ValueOf(m), "", func(_ string, v reflect.Value) bool {
		switch l := v.Interface().(type) {
		case *MLP:
			if l.Up != nil && divides(l.Up.Weight.Value().Shape()[0]) {
				linear(l.Up, col, true)
				linear(l.Down, row, false)
			}
		case *GatedMLP:
			if l.Up != nil && divides(l.Up.Weight.Value().Shape()[0]) {
				linear(l.Gate, col, true)
				linear(l.Up, col, true)
				linear(l.Down, row, false)
			}
		case *SelfAttention:
			hq, hk := l.Heads*l.HeadDim, l.KVHeads*l.HeadDim
			if l.QKV != nil && divides(l.Heads, l.KVHeads) {
				linear(l.QKV, col.WithSections(0, hq, hk, hk), true)
				linear(l.Out, row, false)
			}
		case *MultiheadAttention:
			if l.InProjWeight != nil && divides(l.Heads) {
				e := l.InProjWeight.Value().Shape()[1]
				spec := col.WithSections(0, e, e, e)
				specs[l.InProjWeight] = spec
				if l.InProjBias != nil {
					specs[l.InProjBias] = spec
				}
				linear(l.OutProj, row, false)
			}
		}
		return true
	})
	return func(_ string, p *tensor.Param) tensor.PartitionSpec { return specs[p] }
}
