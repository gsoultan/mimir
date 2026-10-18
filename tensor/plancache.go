package tensor

import (
	"fmt"
	"slices"

	"github.com/gsoultan/mimir/internal/graph"
	"github.com/gsoultan/mimir/internal/sched"
)

// A training step computes the same graph every step, over new values:
// the layouts that a plan's analysis gives its nodes depend only on the
// graph's structure, the layouts of the sharded tensors it reads and the
// constraints on its nodes. A mesh keeps the analyses of its recent
// graphs under a key of those, so that a step whose graph it has seen
// skips the rules of layout.go.

// maxPlans is the most analyses a mesh keeps: a training loop computes a
// few graphs over and over, the step, the optimizer's update, an
// evaluation.
const maxPlans = 32

// checkPlans makes every analysis found in the cache run anew too, and
// fail if the two differ, for tests.
var checkPlans bool

// planCache holds a mesh's recent analyses. It is guarded by meshMu.
type planCache struct {
	byHash map[uint64][]*analysis
	n      int
	clock  uint64
	hits   int // for tests
}

// analysis is what analyze gave the nodes of a graph, by their place in
// the plan's order, and the leaves, by their place in the key.
type analysis struct {
	key       []uint64
	used      uint64
	lays      []layout
	reqs      [][]layout
	depths    []int
	lasts     []int
	leafLasts []int
}

func (c *planCache) lookup(hash uint64, key []uint64) *analysis {
	for _, a := range c.byHash[hash] {
		if slices.Equal(a.key, key) {
			c.clock++
			a.used = c.clock
			c.hits++
			return a
		}
	}
	return nil
}

func (c *planCache) add(hash uint64, a *analysis) {
	if c.byHash == nil {
		c.byHash = map[uint64][]*analysis{}
	}
	if c.n >= maxPlans {
		// Drop the analysis used longest ago.
		var oldest *analysis
		var at uint64
		for h, as := range c.byHash {
			for _, x := range as {
				if oldest == nil || x.used < oldest.used {
					oldest, at = x, h
				}
			}
		}
		c.byHash[at] = slices.DeleteFunc(c.byHash[at], func(x *analysis) bool { return x == oldest })
		if len(c.byHash[at]) == 0 {
			delete(c.byHash, at)
		}
		c.n--
	}
	c.clock++
	a.used = c.clock
	c.byHash[hash] = append(c.byHash[hash], a)
	c.n++
}

// Key words of a reference to an input that is a leaf: leafRef with its
// number, and firstRead too where it is first read, followed by what the
// analysis reads of it.
const (
	leafRef   = 1 << 63
	firstRead = 1 << 62
)

// structure returns the key of the graph that analyze walked, before it
// gives its nodes layouts, and its leaves, numbered as the key numbers
// them. It reports false if the graph has no key.
func (p *plan) structure() ([]uint64, []*graph.Node, bool) {
	for i, n := range p.order {
		p.nodes[n].num = i
	}
	key := make([]uint64, 0, 8*len(p.order))
	var leaves []*graph.Node
	for _, n := range p.order {
		pn := p.nodes[n]
		// The rules read whether a constant or a padding is zero, not
		// what it is, so that a step's changing scalars, such as a
		// learning rate, keep its key.
		var zero uint64
		if n.Attrs.Value == 0 {
			zero = 1
		}
		key = append(key, uint64(n.Op)<<40|uint64(n.DType)<<32|uint64(len(pn.ins))<<16|uint64(len(n.Shape))<<1|zero)
		for _, d := range n.Shape {
			key = append(key, uint64(d))
		}
		for _, x := range pn.ins {
			px := p.nodes[x]
			switch {
			case !px.leaf:
				key = append(key, uint64(px.num))
			case px.keyed:
				key = append(key, leafRef|uint64(px.num))
			default:
				px.num, px.keyed = len(leaves), true
				leaves = append(leaves, x)
				var held uint64
				if px.val != nil {
					held = 1
				}
				key = append(key, leafRef|firstRead|uint64(px.num), uint64(x.DType)<<32|uint64(len(x.Shape))<<1|held)
				for _, d := range x.Shape {
					key = append(key, uint64(d))
				}
				key = px.lay.appendWords(key)
			}
		}
		var ok bool
		if key, ok = sched.AppendAttrs(key, &n.Attrs); !ok {
			return nil, nil, false
		}
		if l, ok := constraints.get(n); ok {
			key = l.appendWords(append(key, 1))
		} else {
			key = append(key, 0)
		}
	}
	return key, leaves, true
}

// record returns the analysis of p's nodes and leaves, under key.
func (p *plan) record(key []uint64, leaves []*graph.Node) *analysis {
	a := &analysis{
		key:       key,
		lays:      make([]layout, len(p.order)),
		reqs:      make([][]layout, len(p.order)),
		depths:    make([]int, len(p.order)),
		lasts:     make([]int, len(p.order)),
		leafLasts: make([]int, len(leaves)),
	}
	for i, n := range p.order {
		pn := p.nodes[n]
		a.lays[i], a.reqs[i], a.depths[i], a.lasts[i] = pn.lay, pn.req, pn.depth, pn.last
	}
	for j, x := range leaves {
		a.leafLasts[j] = p.nodes[x].last
	}
	return a
}

// apply gives p's nodes and leaves the analysis a.
func (p *plan) apply(a *analysis, leaves []*graph.Node) {
	for i, n := range p.order {
		pn := p.nodes[n]
		pn.lay, pn.req, pn.depth, pn.last = a.lays[i], a.reqs[i], a.depths[i], a.lasts[i]
	}
	for j, x := range leaves {
		p.nodes[x].last = a.leafLasts[j]
	}
}

// same returns an error unless a is the analysis of p's nodes and leaves.
func (p *plan) same(a *analysis, leaves []*graph.Node) error {
	b := p.record(a.key, leaves)
	for i, n := range p.order {
		if !a.lays[i].equal(b.lays[i]) || a.depths[i] != b.depths[i] || a.lasts[i] != b.lasts[i] ||
			!slices.EqualFunc(a.reqs[i], b.reqs[i], layout.equal) {
			return fmt.Errorf("tensor: sharded %s: the cached analysis differs from the plan's: %w", n.Op, ErrInternal)
		}
	}
	if !slices.Equal(a.leafLasts, b.leafLasts) {
		return fmt.Errorf("tensor: the cached analysis of a sharded graph reads its leaves differently: %w", ErrInternal)
	}
	return nil
}

// appendWords appends to k words that equal layouts share, and that say
// how many follow.
func (l layout) appendWords(k []uint64) []uint64 {
	var max uint64
	if l.max {
		max = 1
	}
	k = append(k, uint64(len(l.dims))<<32|uint64(len(l.packed))<<16|uint64(l.partial)<<1|max)
	for _, d := range l.dims {
		k = append(k, uint64(uint8(d.axis))<<32|uint64(len(d.sec)))
		for _, s := range d.sec {
			k = append(k, uint64(s))
		}
	}
	for _, part := range l.packed {
		k = append(k, uint64(part.off), uint64(len(part.shape)))
		for _, s := range part.shape {
			k = append(k, uint64(s))
		}
		k = part.lay.appendWords(k)
	}
	return k
}
