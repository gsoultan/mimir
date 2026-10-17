package backend

import (
	"context"
	"fmt"
	"slices"
)

// Collectives move and combine buffers across a group of devices, as NCCL
// does for a group of GPUs. A Communicator runs them over devices of one
// backend type, each device a rank numbered by its place in the group.
//
// Collectives are ordered like kernels: a device's part of a collective
// runs after the work submitted to that device before it and before the
// work submitted after it, and a backend whose kernels run after Run
// returns, as a GPU's do, may report the collective's errors at Sync. The
// ranks wait for each other inside the collective, never on the host, so
// the host can queue a collective on every device and go on.

// Reduction is how a collective combines the elements of the ranks.
type Reduction uint8

// Reductions. ReduceMean divides the sum by the number of ranks and takes
// floating-point elements only.
const (
	ReduceSum Reduction = iota
	ReduceMean
	ReduceMax
	ReduceMin
)

func (r Reduction) String() string {
	switch r {
	case ReduceSum:
		return "sum"
	case ReduceMean:
		return "mean"
	case ReduceMax:
		return "max"
	case ReduceMin:
		return "min"
	default:
		return fmt.Sprintf("Reduction(%d)", uint8(r))
	}
}

// CollectiveKind is what a collective computes.
type CollectiveKind uint8

// Collective kinds. With k the length of In[r]:
//   - AllReduce writes into every Out[r] the reduction over the ranks of
//     their In, elementwise; Out[r] has length k.
//   - Reduce writes that reduction into Out[Root] alone; Out of every
//     other rank is nil.
//   - Broadcast copies In[Root] into Out[r] of every rank. In of every
//     other rank is nil, and so may Out[Root] be, when Root's input is its
//     output already.
//   - AllGather writes the In of every rank, one after another in rank
//     order, into every Out[r], of length k times the number of ranks.
//   - ReduceScatter cuts every In[r] into as many equal parts as there are
//     ranks and writes into Out[r] the reduction over the ranks of their
//     part r; Out[r] has length k divided by the number of ranks.
//
// AllReduce, Reduce and Broadcast may write Out[r] over In[r], when they
// are the same buffer; AllGather and ReduceScatter outputs share no memory
// with any input.
const (
	AllReduce CollectiveKind = iota
	Reduce
	Broadcast
	AllGather
	ReduceScatter
)

func (k CollectiveKind) String() string {
	switch k {
	case AllReduce:
		return "all_reduce"
	case Reduce:
		return "reduce"
	case Broadcast:
		return "broadcast"
	case AllGather:
		return "all_gather"
	case ReduceScatter:
		return "reduce_scatter"
	default:
		return fmt.Sprintf("CollectiveKind(%d)", uint8(k))
	}
}

// Collective is one collective over a group: In[r] and Out[r] are the
// buffers of rank r, on that rank's device, all of one dtype.
type Collective struct {
	Kind   CollectiveKind
	Reduce Reduction // AllReduce, Reduce, ReduceScatter
	Root   int       // Reduce, Broadcast
	In     []Buffer
	Out    []Buffer
}

// Communicator runs collectives over a group of devices, which it holds
// until Close.
type Communicator interface {
	// Backends returns the devices of the group, rank by rank.
	Backends() []Backend
	// Run runs cs, in order, as one group: on each device, its parts of
	// cs run one after another, as kernels would, after the work
	// submitted to the device so far. A Run that returns an error has
	// queued nothing.
	Run(ctx context.Context, cs []Collective) error
	// Close releases the group. Collectives already queued still run.
	Close() error
}

// Grouper is a Backend whose devices form groups that run collectives.
type Grouper interface {
	Backend
	// Group returns a Communicator over devices, which are distinct
	// devices of the receiver's kind, in rank order; the receiver is
	// usually one of them.
	Group(devices []Backend) (Communicator, error)
}

// Asynchronous is a Backend that runs kernels after Run returns and
// reports their errors at the next Sync, Upload or Download, when
// Asynchronous returns true. The scheduler then returns from a schedule
// once its kernels are queued instead of waiting for them, so that the
// host can queue work for other devices meanwhile, as a data-parallel
// step does on the devices of a group. Kernel errors, such as an index out
// of range, then surface at the next read of a value from the device.
type Asynchronous interface {
	Backend
	Asynchronous() bool
}

// CheckCollectives verifies cs against the contract of Communicator.Run
// over a group of size ranks: one buffer per rank, of one dtype that
// collectives take, with the lengths the kind requires. Backends can then
// move and combine elements without further checks.
func CheckCollectives(cs []Collective, size int) error {
	for i := range cs {
		if err := checkCollective(&cs[i], size); err != nil {
			return fmt.Errorf("collective %d: %w", i, err)
		}
	}
	return nil
}

func checkCollective(c *Collective, size int) error {
	if size < 1 || len(c.In) != size || len(c.Out) != size {
		return fmt.Errorf("%s: %d inputs and %d outputs for %d ranks: %w", c.Kind, len(c.In), len(c.Out), size, ErrShape)
	}
	if c.Kind > ReduceScatter || c.Reduce > ReduceMin {
		return fmt.Errorf("%s of %s: %w", c.Kind, c.Reduce, ErrInternal)
	}
	rooted := c.Kind == Reduce || c.Kind == Broadcast
	if rooted && (c.Root < 0 || c.Root >= size) {
		return fmt.Errorf("%s from root %d of %d ranks: %w", c.Kind, c.Root, size, ErrShape)
	}
	var dt DType
	k := -1
	for r := range size {
		in, out := c.In[r], c.Out[r]
		wantIn := c.Kind != Broadcast || r == c.Root
		wantOut := c.Kind != Reduce || r == c.Root
		if (in != nil) != wantIn || (out != nil) != wantOut && !(c.Kind == Broadcast && r == c.Root) {
			return fmt.Errorf("%s: buffers of rank %d: %w", c.Kind, r, ErrInternal)
		}
		for _, b := range []Buffer{in, out} {
			if b == nil {
				continue
			}
			if dt == Invalid {
				dt = b.DType()
			}
			if b.DType() != dt {
				return fmt.Errorf("%s of %s and %s: %w", c.Kind, dt, b.DType(), ErrDType)
			}
		}
		if in != nil {
			if k >= 0 && in.Len() != k {
				return fmt.Errorf("%s of %d and %d elements: %w", c.Kind, k, in.Len(), ErrShape)
			}
			k = in.Len()
		}
	}
	switch {
	case dt.Size() == 0:
		return fmt.Errorf("%s of %s: %w", c.Kind, dt, ErrDType)
	case c.Reduce == ReduceMean && !dt.IsFloat() && (c.Kind == AllReduce || c.Kind == Reduce || c.Kind == ReduceScatter):
		return fmt.Errorf("%s mean of %s: %w", c.Kind, dt, ErrDType)
	case c.Kind == ReduceScatter && k%size != 0:
		return fmt.Errorf("%s of %d elements over %d ranks: %w", c.Kind, k, size, ErrShape)
	}
	want := k
	switch c.Kind {
	case AllGather:
		want = k * size
	case ReduceScatter:
		want = k / size
	}
	for r, out := range c.Out {
		if out == nil {
			continue
		}
		if out.Len() != want {
			return fmt.Errorf("%s: output of rank %d holds %d elements, want %d: %w", c.Kind, r, out.Len(), want, ErrShape)
		}
		for q, in := range c.In {
			if in == out && (q != r || c.Kind == AllGather || c.Kind == ReduceScatter) {
				return fmt.Errorf("%s: output of rank %d is the input of rank %d: %w", c.Kind, r, q, ErrInternal)
			}
		}
		if slices.Contains(c.Out[:r], out) {
			return fmt.Errorf("%s: ranks share an output: %w", c.Kind, ErrInternal)
		}
	}
	return nil
}
