package sched

import (
	"context"
	"fmt"
	"slices"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/internal/graph"
)

// A realized node can be donated, as jax.jit donates arguments: its
// owner is done with it, so the next schedule that reads it may take its
// memory, as a batch of inputs that a training step reads once. The
// schedule gives the donated buffer to the pool once its last reader has
// run, so that later kernels of the same schedule take it for their
// results and temporaries, and schedules after it, or the next upload
// (see Upload), take it if they do not. The storage, and so every node
// realized into it, is stale from the moment the schedule runs, and a
// read of one reports graph.ErrDonated.
//
// A schedule takes the memory of a donated node when the node is
// realized on its device into memory from the backend (Storage.Pooled),
// is the only node realized into it, and is not a target or a
// destination, nor viewed by a target, which keep the memory. Nor does it
// take memory that a node keep lists reads (see RealizeInto), such as
// running statistics computed from a batch, which must stay computable.
// Whether a leaf is donated is part of the schedule's key, so a step that
// donates its batch replays a program of its own.

// Donate marks the realized node n, whose owner is done with it once the
// schedule that reads it next has run, as donated. A node that is not
// realized is left alone: the schedule that computes it frees its memory
// anyway when it is not a target.
func Donate(n *graph.Node) {
	mu.Lock()
	defer mu.Unlock()
	if n.Realized() {
		n.Donated = true
	}
}

// donates reports whether a schedule on be that reads the realized node
// n may take its memory, as far as n alone tells.
func donates(n *graph.Node, be backend.Backend) bool {
	m := n.Mem
	return n.Donated && m.Pooled && m.Dev == be && m.Refs == 1 && !n.Stale()
}

// withhold keeps the schedule that w walked, of targets, from taking memory
// that the nodes keep returns read, through nodes other than targets:
// it drops their leaves from w.gifts and clears the key's mark of them.
// It calls keep only if the walk found a donated leaf.
func (w *walker) withhold(keep func() []*graph.Node, targets []*graph.Node) {
	if len(w.gifts) == 0 || keep == nil {
		return
	}
	mems := spared(keep(), targets)
	if len(mems) == 0 {
		return
	}
	w.gifts = slices.DeleteFunc(w.gifts, func(g gift) bool {
		if !mems[w.leaves[g.leaf].Mem] {
			return false
		}
		w.key[g.at] &^= 1 << 49
		return true
	})
}

// spared returns the storages that the nodes of keep read through
// unrealized nodes other than targets, leaving out those that are
// targets or can no longer be computed: memory that a schedule of
// targets must not take, so that they stay computable after it.
func spared(keep, targets []*graph.Node) map[*graph.Storage]bool {
	if len(keep) == 0 {
		return nil
	}
	isTarget := make(map[*graph.Node]bool, len(targets))
	for _, t := range targets {
		isTarget[t] = true
	}
	mems := map[*graph.Storage]bool{}
	seen := map[*graph.Node]bool{}
	for _, k := range keep {
		if k.Realized() || isTarget[k] || seen[k] {
			continue
		}
		seen[k] = true
		for stack := []*graph.Node{k}; len(stack) > 0; {
			n := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			for _, in := range n.Inputs {
				switch {
				case in.Realized():
					mems[in.Mem] = true
				case !seen[in] && !isTarget[in]:
					seen[in] = true
					stack = append(stack, in)
				}
			}
		}
	}
	return mems
}

// Upload returns a realized node of shape on be holding data, the bytes
// of its elements of dt. Its buffer comes from be's pool if the pool
// holds a free one of that size, as it does once a donated input of the
// same size has gone back to it, so that a loop that uploads a batch,
// and donates it to the step that reads it, uploads into the same buffer
// every time. Otherwise it comes from the backend, and stays off the
// pool's account until it is donated or released.
func Upload(ctx context.Context, be backend.Backend, dt backend.DType, data []byte, shape []int) (*graph.Node, error) {
	n, ok := backend.NumElements(shape)
	if !ok {
		return nil, fmt.Errorf("sched: upload of shape %v: %w", shape, backend.ErrShape)
	}
	buf, err := uploadBuffer(be, dt, n)
	if err != nil {
		return nil, err
	}
	if err := be.Upload(ctx, buf, data); err != nil {
		giveBack(be, buf)
		return nil, err
	}
	node, err := graph.Load(be, buf, shape)
	if err != nil {
		giveBack(be, buf)
		return nil, err
	}
	node.Mem.Pooled = true
	return node, nil
}

// uploadBuffer returns a buffer of n elements of dt for an upload: a free
// one from be's pool, or else a new one from be.
func uploadBuffer(be backend.Backend, dt backend.DType, n int) (backend.Buffer, error) {
	mu.Lock()
	p := poolOf(be)
	key := poolKey{dt, n}
	if bufs := p.free[key]; len(bufs) > 0 {
		buf := bufs[len(bufs)-1].buf
		p.free[key] = bufs[:len(bufs)-1]
		mu.Unlock()
		return buf, nil
	}
	mu.Unlock()
	return be.Alloc(dt, n)
}

// giveBack returns buf, which uploadBuffer returned, to be's pool.
func giveBack(be backend.Backend, buf backend.Buffer) {
	mu.Lock()
	defer mu.Unlock()
	p := poolOf(be)
	p.track(buf)
	p.release(buf)
}
