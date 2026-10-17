package train

import (
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/tensor"
)

// Sharded trains a model sharded over mesh, in the manner of jax.sharding:
// Fit places the model's parameters on mesh as rule says (see nn.Shard),
// and each batch split along its first axis over the mesh axis named
// "data", if the mesh has one and the batch divides evenly, or whole on
// every device otherwise. The rules of tensor.Shard insert the
// collectives, and the optimizer updates every parameter where its
// blocks are, with states laid out alike. So a mesh with a "data" axis
// and no rule trains data-parallel, matching DataParallel;
// nn.TensorParallel over a "model" axis trains tensor-parallel;
// nn.FullyShard over "data" trains as FSDP and ZeRO-3 do; and a mesh with
// both axes combines them, such as data-parallel replicas of a
// tensor-parallel model.
//
// Sharded does not combine with DataParallel, MicroBatch, MemoryBudget
// or Checkpoint.
func Sharded(mesh *tensor.Mesh, rule nn.ShardRule) Option {
	return func(c *config) {
		if mesh == nil {
			c.fail("sharding over a nil mesh")
		}
		c.mesh, c.shardRule = mesh, rule
	}
}

// shard returns the batch t, inputs or targets, on the mesh of Sharded:
// split over its "data" axis where it divides evenly. The copy is the
// loop's own, so it is donated to the step that reads it.
func (l *loop) shard(t *tensor.Tensor) *tensor.Tensor {
	m := l.c.mesh
	var spec tensor.PartitionSpec
	if k := m.Size("data"); k > 1 && t.Rank() > 0 && t.Shape()[0]%k == 0 {
		spec = tensor.P("data")
	}
	return tensor.Shard(t, m, spec).Donate()
}
