package cpu

import (
	"context"
	"fmt"
	"math"
	"slices"

	"github.com/gsoultan/mimir/backend"
)

// An argsort sorts each line, the elements that differ only in their
// position along the axis, by a key of 64 bits: the element's order in
// the high 32 bits, where unsigned order is the order of the sort, and
// its position in the low 32. Keys are distinct, so any sort of them is
// the stable sort of the elements, and the positions are the keys' low
// halves.
//
// A full sort of a line is an LSD radix sort of the keys' high halves,
// which keeps the keys in the order of their positions where the halves
// tie, and a top-k keeps the k smallest keys in a heap. Workers take
// whole lines; when there are fewer lines than workers, each long line is
// cut into a part for every worker, which sorts it, and the sorted parts
// are merged.

// sortSmall is the longest line sorted by insertion.
const sortSmall = 48

// sortSplit is the shortest line that is cut into parts for the workers
// when there are too few lines to share.
const sortSplit = 1 << 15

// sortKey returns the order of a float32 as an unsigned key: NaN is the
// largest value, and -0 equals +0.
func sortKey(f float32) uint32 {
	bits := math.Float32bits(f)
	switch {
	case f != f:
		bits = 0x7fc00000
	case f == 0:
		bits = 0
	}
	if bits&(1<<31) != 0 {
		return ^bits
	}
	return bits | 1<<31
}

// runArgsort runs an argsort of in into the Int32 out.
func runArgsort(ctx context.Context, b *Backend, k *backend.Kernel, outBuf, in backend.Buffer) error {
	out, err := data[int32](outBuf)
	if err != nil {
		return err
	}
	axis, v := k.Attrs.Axis, k.In[0]
	if k.DType != backend.Int32 || axis < 0 || axis >= len(k.Shape) || len(v.Shape) != len(k.Shape) || k.Shape[axis] > v.Shape[axis] {
		return fmt.Errorf("argsort of %v along axis %d to %v: %w", v.Shape, axis, k.Shape, backend.ErrInternal)
	}
	var key func(off int) uint32
	switch in := in.(type) {
	case *buffer[float32]:
		key = func(off int) uint32 { return sortKey(in.data[off]) }
	case *buffer[int32]:
		key = func(off int) uint32 { return uint32(in.data[off]) ^ 1<<31 }
	case *buffer[bf16]:
		key = func(off int) uint32 { return sortKey(conv[float32](in.data[off])) }
	case *buffer[f16]:
		key = func(off int) uint32 { return sortKey(conv[float32](in.data[off])) }
	default:
		return fmt.Errorf("foreign buffer %T: %w", in, backend.ErrDType)
	}
	if k.Attrs.Descending {
		up := key
		key = func(off int) uint32 { return ^up(off) }
	}
	if len(out) == 0 {
		return nil
	}
	n, kk := v.Shape[axis], k.Shape[axis]
	outStrides := backend.ContiguousStrides(k.Shape)
	drop := func(s []int) []int { return slices.Delete(slices.Clone(s), axis, axis+1) }
	rest, views := backend.Coalesce(drop(k.Shape), []backend.View{
		{Strides: drop(v.Strides), Offset: v.Offset},
		{Strides: drop(outStrides)},
	})
	lines, _ := backend.NumElements(rest)
	l := sortLine{key: key, n: n, k: kk, inStep: v.Strides[axis], outStep: outStrides[axis], out: out}
	if lines < b.workers && n >= sortSplit {
		c := newCursor(rest, views, 0)
		for range lines {
			if err := l.sortParallel(ctx, b, c.offs[0], c.offs[1]); err != nil {
				return err
			}
			c.next()
		}
		return nil
	}
	return b.parallelFor(ctx, lines, max(1, elementGrain/max(n, 1)), func(lo, hi int) error {
		c := newCursor(rest, views, lo)
		var keys, tmp []uint64
		for range hi - lo {
			keys = l.load(keys, c.offs[0], 0, n)
			keys, tmp = sortKeys(keys, tmp, l.k)
			l.store(keys, c.offs[1])
			c.next()
		}
		return nil
	})
}

// sortLine sorts lines of n elements, read inStep apart through key, and
// writes the positions of the first k into out, outStep apart.
type sortLine struct {
	key             func(off int) uint32
	n, k            int
	inStep, outStep int
	out             []int32
}

// load sets keys to the keys of positions [lo, hi) of the line that
// starts at offset src.
func (l *sortLine) load(keys []uint64, src, lo, hi int) []uint64 {
	keys = grow(keys, hi-lo)
	off := src + lo*l.inStep
	for j := range keys {
		keys[j] = uint64(l.key(off))<<32 | uint64(lo+j)
		off += l.inStep
	}
	return keys
}

// store writes the positions of the first k of keys into the line of the
// output at dst.
func (l *sortLine) store(keys []uint64, dst int) {
	for j, key := range keys[:l.k] {
		l.out[dst+j*l.outStep] = int32(uint32(key))
	}
}

// sortParallel sorts the line at src into dst with every worker: each
// sorts a part of the line, or keeps its k smallest keys, and the parts
// are merged.
func (l *sortLine) sortParallel(ctx context.Context, b *Backend, src, dst int) error {
	parts := b.workers
	size := (l.n + parts - 1) / parts
	runs := make([][]uint64, parts)
	err := b.parallelFor(ctx, parts, 1, func(lo, hi int) error {
		for p := lo; p < hi; p++ {
			from, to := min(p*size, l.n), min((p+1)*size, l.n)
			keys := l.load(nil, src, from, to)
			sorted, _ := sortKeys(keys, nil, min(l.k, len(keys)))
			runs[p] = sorted[:min(l.k, len(sorted))]
		}
		return nil
	})
	if err != nil {
		return err
	}
	// Merge pairs of runs in rounds, each pair by a worker of its own.
	for len(runs) > 1 {
		merged := make([][]uint64, (len(runs)+1)/2)
		err := b.parallelFor(ctx, len(merged), 1, func(lo, hi int) error {
			for p := lo; p < hi; p++ {
				if 2*p+1 == len(runs) {
					merged[p] = runs[2*p]
					continue
				}
				merged[p] = mergeKeys(runs[2*p], runs[2*p+1], l.k)
			}
			return nil
		})
		if err != nil {
			return err
		}
		runs = merged
	}
	l.store(runs[0], dst)
	return nil
}

// mergeKeys returns the first k of the merge of the sorted keys a and b.
func mergeKeys(a, b []uint64, k int) []uint64 {
	out := make([]uint64, min(k, len(a)+len(b)))
	i, j := 0, 0
	for o := range out {
		if j == len(b) || i < len(a) && a[i] < b[j] {
			out[o] = a[i]
			i++
		} else {
			out[o] = b[j]
			j++
		}
	}
	return out
}

// sortKeys sorts keys, or, when k is a small part of them, moves their k
// smallest to the front in order, using tmp as scratch. It returns the
// keys sorted at least as far as k, which may be tmp, and the other
// buffer.
func sortKeys(keys, tmp []uint64, k int) ([]uint64, []uint64) {
	n := len(keys)
	switch {
	case n <= sortSmall:
		insertionSort(keys)
		return keys, tmp
	case k*8 <= n:
		smallest(keys, k)
		return keys, tmp
	}
	return radixSort(keys, tmp)
}

func insertionSort(keys []uint64) {
	for i := 1; i < len(keys); i++ {
		x := keys[i]
		j := i
		for ; j > 0 && keys[j-1] > x; j-- {
			keys[j] = keys[j-1]
		}
		keys[j] = x
	}
}

// smallest moves the k smallest keys to the front of keys, in order. A
// max-heap of the k smallest so far rejects most keys with one
// comparison against its top.
func smallest(keys []uint64, k int) {
	if k == 0 {
		return
	}
	heap := keys[:k]
	for i := k/2 - 1; i >= 0; i-- {
		siftDown(heap, i)
	}
	for _, x := range keys[k:] {
		if x < heap[0] {
			heap[0] = x
			siftDown(heap, 0)
		}
	}
	slices.Sort(heap)
}

// siftDown restores the max-heap order of h below node i.
func siftDown(h []uint64, i int) {
	n := len(h)
	x := h[i]
	for {
		c := 2*i + 1
		if c >= n {
			break
		}
		if c+1 < n && h[c+1] > h[c] {
			c++
		}
		if h[c] <= x {
			break
		}
		h[i] = h[c]
		i = c
	}
	h[i] = x
}

// radixSort sorts keys by their high 32 bits, a byte at a time from the
// lowest, keeping keys whose bytes tie in their order: since keys start
// in the order of their low halves, that is the order of the keys. A
// byte that every key shares takes no pass. It returns the sorted keys,
// in keys or tmp, and the other.
func radixSort(keys, tmp []uint64) ([]uint64, []uint64) {
	var counts [4][256]int
	for _, x := range keys {
		h := uint32(x >> 32)
		counts[0][h&0xff]++
		counts[1][h>>8&0xff]++
		counts[2][h>>16&0xff]++
		counts[3][h>>24]++
	}
	tmp = grow(tmp, len(keys))
	for pass := range 4 {
		c := &counts[pass]
		shift := 32 + 8*uint(pass)
		if c[keys[0]>>shift&0xff] == len(keys) {
			continue
		}
		at := 0
		for i, n := range c {
			c[i] = at
			at += n
		}
		for _, x := range keys {
			d := x >> shift & 0xff
			tmp[c[d]] = x
			c[d]++
		}
		keys, tmp = tmp, keys
	}
	return keys, tmp
}
