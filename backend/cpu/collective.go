package cpu

import (
	"context"
	"fmt"
	"runtime"
	"slices"
	"sort"
	"sync"

	"github.com/gsoultan/mimir/backend"
)

// Collectives over CPU devices move and combine host memory directly.
// Every rank takes an equal share of the elements of all the collectives
// of a Run, as if they were laid end to end: it reduces its share of
// every rank's inputs, and writes the result into every rank's outputs,
// so the ranks together read and write each element once and the work
// spreads over all of them, each with its own workers. The ranks meet at
// a barrier before they start, once every rank's earlier kernels have
// run, and at another when they finish, so that no rank goes on to
// kernels that write over buffers the others still read.

var (
	_ backend.Grouper      = (*Backend)(nil)
	_ backend.Communicator = (*communicator)(nil)
)

// Group returns a Communicator over devices, each a *Backend or a
// *Stream. On a Backend, which runs kernels as Run is called, Run of the
// Communicator returns once the collectives are done; on Streams it
// queues them.
func (*Backend) Group(devices []backend.Backend) (backend.Communicator, error) {
	return newCommunicator(devices)
}

// communicator runs collectives over CPU devices.
type communicator struct {
	devices []backend.Backend
}

func newCommunicator(devices []backend.Backend) (*communicator, error) {
	if len(devices) == 0 {
		return nil, fmt.Errorf("cpu: group of no devices: %w", backend.ErrShape)
	}
	for i, d := range devices {
		switch d.(type) {
		case *Backend, *Stream:
		default:
			return nil, fmt.Errorf("cpu: group with a %s device: %w", d.Name(), backend.ErrDevice)
		}
		if slices.Contains(devices[:i], d) {
			return nil, fmt.Errorf("cpu: device %d appears twice in a group: %w", i, backend.ErrDevice)
		}
	}
	return &communicator{devices: slices.Clone(devices)}, nil
}

func (c *communicator) Backends() []backend.Backend { return slices.Clone(c.devices) }

func (c *communicator) Close() error { return nil }

// enqueue makes the queueing of one Run's parts on every Stream atomic,
// so that two Runs over the same Streams queue their parts in the same
// order on all of them: otherwise each could wait at its barrier for a
// rank that waits at the other's.
var enqueue sync.Mutex

// Run queues the ranks' parts of cs on the Streams of the group and runs
// those of Backends, which it waits for.
func (c *communicator) Run(ctx context.Context, cs []backend.Collective) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	n := len(c.devices)
	if err := backend.CheckCollectives(cs, n); err != nil {
		return fmt.Errorf("cpu: %w", err)
	}
	for i := range cs {
		for _, b := range slices.Concat(cs[i].In, cs[i].Out) {
			if b == nil {
				continue
			}
			if _, err := bytesOf(b); err != nil {
				return err
			}
		}
	}
	if len(cs) == 0 {
		return nil
	}
	g := newMeeting(slices.Clone(cs), n)
	var wg sync.WaitGroup
	errs := make([]error, n)
	enqueue.Lock()
	for r, d := range c.devices {
		switch d := d.(type) {
		case *Stream:
			d.queue(func() error { return g.part(d.be, r) })
		case *Backend:
			wg.Go(func() { errs[r] = g.part(d, r) })
		}
	}
	enqueue.Unlock()
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return err
		}
	}
	return nil
}

// meeting is one Run of collectives in progress: each rank calls part.
type meeting struct {
	cs    []backend.Collective
	ranks int
	// ends[i] is where collective i ends in the elements of all of them
	// laid end to end, and total where the last ends.
	ends  []int
	total int

	start, done barrier
}

func newMeeting(cs []backend.Collective, ranks int) *meeting {
	g := &meeting{cs: cs, ranks: ranks, ends: make([]int, len(cs))}
	for i := range cs {
		g.total += g.units(i)
		g.ends[i] = g.total
	}
	g.start.init(ranks)
	g.done.init(ranks)
	return g
}

// units returns how many elements collective i spreads over the ranks:
// those of one input, or of one output when the output is longer.
func (g *meeting) units(i int) int {
	c := &g.cs[i]
	if c.Kind == backend.Broadcast {
		return c.In[c.Root].Len()
	}
	if c.Kind == backend.AllGather {
		return c.Out[0].Len()
	}
	return c.In[0].Len()
}

// part runs rank r's share of the collectives on be, between the barriers
// that every rank reaches, whatever happens.
func (g *meeting) part(be *Backend, r int) (err error) {
	g.start.wait()
	defer g.done.wait()
	defer func() {
		if p := recover(); p != nil {
			err = fmt.Errorf("cpu: panic in collective: %v: %w", p, backend.ErrInternal)
		}
	}()
	lo, hi := g.total*r/g.ranks, g.total*(r+1)/g.ranks
	defer runtime.KeepAlive(g.cs)
	return be.parallelFor(context.Background(), hi-lo, collectiveGrain, func(a, b int) error {
		g.span(lo+a, lo+b)
		return nil
	})
}

// collectiveGrain is the fewest elements a worker moves or combines at a
// time, and collectiveBlock the most it combines in one pass over the
// ranks, which its accumulator holds.
const (
	collectiveGrain = 1 << 14
	collectiveBlock = 1 << 11
)

// span runs elements [lo, hi) of the collectives laid end to end.
func (g *meeting) span(lo, hi int) {
	var acc accumulator
	i := sort.SearchInts(g.ends, lo+1)
	for lo < hi {
		start := g.ends[i] - g.units(i)
		end := min(hi, g.ends[i])
		g.run(&g.cs[i], &acc, lo-start, end-start)
		lo = end
		i++
	}
}

// run runs elements [lo, hi) of c, in blocks that stay within one rank's
// part of an AllGather or ReduceScatter.
func (g *meeting) run(c *backend.Collective, acc *accumulator, lo, hi int) {
	for lo < hi {
		end := min(hi, lo+collectiveBlock)
		switch c.Kind {
		case backend.AllReduce, backend.Reduce:
			acc.reduce(c.In, c.Reduce, lo, end-lo, g.ranks)
			for r, out := range c.Out {
				if out != nil && (c.Kind == backend.AllReduce || r == c.Root) {
					acc.store(out, lo)
				}
			}
		case backend.Broadcast:
			src := c.In[c.Root]
			for _, out := range c.Out {
				if out != nil && out != src {
					copyElements(out, lo, src, lo, end-lo)
				}
			}
		case backend.AllGather:
			k := c.In[0].Len()
			q, j := lo/k, lo%k
			end = min(end, (q+1)*k)
			for _, out := range c.Out {
				copyElements(out, lo, c.In[q], j, end-lo)
			}
		case backend.ReduceScatter:
			k := c.Out[0].Len()
			p, j := lo/k, lo%k
			end = min(end, (p+1)*k)
			acc.reduce(c.In, c.Reduce, lo, end-lo, g.ranks)
			acc.store(c.Out[p], j)
		}
		lo = end
	}
}

// copyElements copies n elements of src from element from into dst from
// element at; both hold elements of one dtype.
func copyElements(dst backend.Buffer, at int, src backend.Buffer, from, n int) {
	size := dst.DType().Size()
	d, _ := bytesOf(dst)
	s, _ := bytesOf(src)
	copy(d[at*size:(at+n)*size], s[from*size:(from+n)*size])
}

// accumulator holds a block of reduced elements: in floats for the float
// dtypes, in ints for Int32.
type accumulator struct {
	f  []float32
	i  []int32
	dt backend.DType
}

// reduce sets the accumulator to the reduction of elements [at, at+n) of
// the buffers ins, in their order.
func (a *accumulator) reduce(ins []backend.Buffer, red backend.Reduction, at, n, ranks int) {
	a.dt = ins[0].DType()
	if a.dt == backend.Int32 {
		a.i = grow(a.i, n)
		copy(a.i, ins[0].(*buffer[int32]).data[at:at+n])
		for _, in := range ins[1:] {
			combineInto(a.i, in.(*buffer[int32]).data[at:at+n], red)
		}
		return
	}
	a.f = grow(a.f, n)
	loadFloats(a.f, ins[0], at)
	var tmp [collectiveBlock]float32
	for _, in := range ins[1:] {
		loadFloats(tmp[:n], in, at)
		combineInto(a.f, tmp[:n], red)
	}
	if red == backend.ReduceMean {
		inv := 1 / float32(ranks)
		for j := range a.f {
			a.f[j] *= inv
		}
	}
}

// store writes the accumulator into out from element at.
func (a *accumulator) store(out backend.Buffer, at int) {
	switch out := out.(type) {
	case *buffer[int32]:
		copy(out.data[at:], a.i)
	case *buffer[float32]:
		copy(out.data[at:], a.f)
	case *buffer[bf16]:
		for j, v := range a.f {
			out.data[at+j] = bf16(f32ToBF16(v))
		}
	case *buffer[f16]:
		for j, v := range a.f {
			out.data[at+j] = f16(f32ToF16(v))
		}
	}
}

// loadFloats reads elements [at, at+len(dst)) of the float buffer b as float32.
func loadFloats(dst []float32, b backend.Buffer, at int) {
	switch b := b.(type) {
	case *buffer[float32]:
		copy(dst, b.data[at:])
	case *buffer[bf16]:
		for j := range dst {
			dst[j] = bf16ToF32(uint16(b.data[at+j]))
		}
	case *buffer[f16]:
		for j := range dst {
			dst[j] = f16ToF32(uint16(b.data[at+j]))
		}
	}
}

// combineInto sets acc[j] to the reduction of acc[j] and x[j]. The mean sums;
// its division comes after the last rank.
func combineInto[T number](acc, x []T, red backend.Reduction) {
	x = x[:len(acc)]
	switch red {
	case backend.ReduceSum, backend.ReduceMean:
		for j, v := range x {
			acc[j] += v
		}
	case backend.ReduceMax:
		for j, v := range x {
			// NaN propagates, as in OpReduceMax.
			if v > acc[j] || v != v {
				acc[j] = v
			}
		}
	case backend.ReduceMin:
		for j, v := range x {
			if v < acc[j] || v != v {
				acc[j] = v
			}
		}
	}
}

// barrier is a one-use barrier for a fixed number of goroutines.
type barrier struct {
	mu      sync.Mutex
	waiting int
	open    chan struct{}
}

func (b *barrier) init(n int) {
	b.waiting = n
	b.open = make(chan struct{})
}

// wait blocks until every goroutine has called wait.
func (b *barrier) wait() {
	b.mu.Lock()
	b.waiting--
	if b.waiting == 0 {
		close(b.open)
	}
	b.mu.Unlock()
	<-b.open
}
