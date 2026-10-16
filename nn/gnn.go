package nn

import (
	"context"
	"fmt"
	"math"
	"slices"
	"sync"

	"github.com/gsoultan/mimir/tensor"
)

// Graph is a directed graph that graph layers pass messages over, like
// the edge_index of PyTorch Geometric: edge e runs from node src[e] to
// node dst[e], and messages flow along it, from source to target.
//
// Node features are tensors with a row per node, [nodes, F...], and
// edge features have a row per edge, [edges, F...]. Source and Target
// read the features of each edge's nodes, and Sum, Mean and Max
// aggregate edge features into the target nodes, like the scatter
// operations of PyTorch Geometric. Adjacency combines the features of
// neighbours directly, without a tensor per edge, which is how the
// convolutions here run.
//
// The graph layers GCNConv, SAGEConv and GATConv are not Modules, since
// their Convolve takes a graph along with the features; a model that
// holds its graph, and calls them on it in its Forward, is one.
//
// A Graph builds the tensor.Patterns it multiplies by the first time a
// layer needs them, and keeps them; it is safe for concurrent use.
type Graph struct {
	nodes    int
	src, dst []int32

	once                         [4]sync.Once // the patterns and the graph with self loops
	source, target, adj          *tensor.Pattern
	errSource, errTarget, errAdj error
	loops                        *Graph

	mu    sync.Mutex
	norms map[tensor.Device]*tensor.Tensor // see degreeNorm
}

// NewGraph returns the graph of nodes nodes with an edge from src[e] to
// dst[e] for each e. Repeated edges count as often as they appear.
// Endpoints outside [0, nodes) fail with tensor.ErrIndex.
func NewGraph(nodes int, src, dst []int32) (*Graph, error) {
	if nodes < 0 || nodes >= math.MaxInt32 || len(src) != len(dst) || len(src) >= math.MaxInt32 {
		return nil, fmt.Errorf("nn: graph of %d nodes and %d sources for %d targets: %w", nodes, len(src), len(dst), tensor.ErrShape)
	}
	for e := range src {
		if s, d := src[e], dst[e]; s < 0 || int(s) >= nodes || d < 0 || int(d) >= nodes {
			return nil, fmt.Errorf("nn: edge %d from %d to %d in a graph of %d nodes: %w", e, s, d, nodes, tensor.ErrIndex)
		}
	}
	return &Graph{nodes: nodes, src: slices.Clone(src), dst: slices.Clone(dst)}, nil
}

// GraphFromEdgeIndex returns the graph of nodes nodes whose edges an
// Int32 edge index of shape [2, edges] lists, sources in its first row
// and targets in its second, as PyTorch Geometric stores them.
func GraphFromEdgeIndex(ctx context.Context, nodes int, edgeIndex *tensor.Tensor) (*Graph, error) {
	if err := edgeIndex.Err(); err != nil {
		return nil, err
	}
	if s := edgeIndex.Shape(); len(s) != 2 || s[0] != 2 || edgeIndex.DType() != tensor.Int32 {
		return nil, fmt.Errorf("nn: edge index of %s %v, want int32 [2, edges]: %w", edgeIndex.DType(), s, tensor.ErrShape)
	}
	v, err := edgeIndex.Data[int32](ctx)
	if err != nil {
		return nil, err
	}
	e := len(v) / 2
	return NewGraph(nodes, v[:e], v[e:])
}

// Nodes returns the number of nodes of g.
func (g *Graph) Nodes() int { return g.nodes }

// Edges returns the number of edges of g.
func (g *Graph) Edges() int { return len(g.src) }

// WithSelfLoops returns g with exactly one edge from every node to
// itself: the loops g has are left out, and one is added for each node
// after the other edges, like PyTorch Geometric's remove_self_loops
// followed by add_self_loops. The result is kept, so later calls return
// the same graph.
func (g *Graph) WithSelfLoops() *Graph {
	g.once[3].Do(func() {
		src := make([]int32, 0, len(g.src)+g.nodes)
		dst := make([]int32, 0, len(g.src)+g.nodes)
		for e := range g.src {
			if g.src[e] != g.dst[e] {
				src, dst = append(src, g.src[e]), append(dst, g.dst[e])
			}
		}
		for i := range int32(g.nodes) {
			src, dst = append(src, i), append(dst, i)
		}
		g.loops = &Graph{nodes: g.nodes, src: src, dst: dst}
	})
	return g.loops
}

// edgePattern returns the edges × nodes pattern with an entry at
// (e, end[e]) for each edge e.
func (g *Graph) edgePattern(end []int32) (*tensor.Pattern, error) {
	row := make([]int32, len(end))
	for e := range row {
		row[e] = int32(e)
	}
	return tensor.NewPattern(len(end), g.nodes, row, end)
}

func (g *Graph) sourcePattern() (*tensor.Pattern, error) {
	g.once[0].Do(func() { g.source, g.errSource = g.edgePattern(g.src) })
	return g.source, g.errSource
}

func (g *Graph) targetPattern() (*tensor.Pattern, error) {
	g.once[1].Do(func() { g.target, g.errTarget = g.edgePattern(g.dst) })
	return g.target, g.errTarget
}

// Adjacency returns the nodes × nodes pattern with an entry at (dst[e],
// src[e]) for each edge e: its product with node features x sums, for
// each node, the features of the sources of its incoming edges, which
// Sum(Source(x)) computes through a tensor per edge.
func (g *Graph) Adjacency() *tensor.Pattern {
	p, _ := g.adjacency()
	return p
}

func (g *Graph) adjacency() (*tensor.Pattern, error) {
	g.once[2].Do(func() { g.adj, g.errAdj = tensor.NewPattern(g.nodes, g.nodes, g.dst, g.src) })
	return g.adj, g.errAdj
}

// apply returns f(p, x) for the pattern get returns, or a failed tensor.
func apply(get func() (*tensor.Pattern, error), x *tensor.Tensor, f func(*tensor.Pattern, *tensor.Tensor) *tensor.Tensor) *tensor.Tensor {
	p, err := get()
	if err != nil {
		return tensor.Failed(err)
	}
	return f(p, x)
}

// Source returns the features of each edge's source node, [edges, F...],
// from node features x, [nodes, F...]: the x_j of PyTorch Geometric's
// message functions.
func (g *Graph) Source(x *tensor.Tensor) *tensor.Tensor {
	return apply(g.sourcePattern, x, (*tensor.Pattern).MatMul)
}

// Target returns the features of each edge's target node, [edges, F...],
// from node features x, [nodes, F...]: the x_i of PyTorch Geometric.
func (g *Graph) Target(x *tensor.Tensor) *tensor.Tensor {
	return apply(g.targetPattern, x, (*tensor.Pattern).MatMul)
}

// Sum adds the edge features m, [edges, F...], into the target of each
// edge, [nodes, F...], in the order of the edges; a node without
// incoming edges gets zeros.
func (g *Graph) Sum(m *tensor.Tensor) *tensor.Tensor {
	return apply(g.targetPattern, m, func(p *tensor.Pattern, m *tensor.Tensor) *tensor.Tensor { return p.T().MatMul(m) })
}

// Mean averages the edge features m, [edges, F...], over the incoming
// edges of each node, [nodes, F...]; a node without any gets zeros.
func (g *Graph) Mean(m *tensor.Tensor) *tensor.Tensor {
	return apply(g.targetPattern, m, func(p *tensor.Pattern, m *tensor.Tensor) *tensor.Tensor { return p.T().Mean(m) })
}

// Max takes the elementwise maximum of the edge features m, [edges,
// F...], over the incoming edges of each node, [nodes, F...]; a node
// without any gets zeros.
func (g *Graph) Max(m *tensor.Tensor) *tensor.Tensor {
	return apply(g.targetPattern, m, func(p *tensor.Pattern, m *tensor.Tensor) *tensor.Tensor { return p.T().Max(m) })
}

// Softmax normalizes the edge scores m, [edges, F...], over the incoming
// edges of each node, like PyTorch Geometric's softmax(src, index): an
// edge's result is e^m divided by the sum of e^m over the edges into its
// target, computed after subtracting the target's largest score.
func (g *Graph) Softmax(m *tensor.Tensor) *tensor.Tensor {
	// The result does not depend on the shift, so no gradient flows
	// through it.
	shift := g.Target(g.Max(m.Detach()))
	e := m.Sub(shift).Exp()
	return e.Div(g.Target(g.Sum(e)))
}

// degreeNorm returns D^-1/2, the inverse square roots of the
// in-degrees of g, at least 1, as a [nodes, 1] tensor on d, computed
// once per device.
func (g *Graph) degreeNorm(d tensor.Device) *tensor.Tensor {
	g.mu.Lock()
	defer g.mu.Unlock()
	if t, ok := g.norms[d]; ok {
		return t
	}
	deg := make([]float32, g.nodes)
	for _, i := range g.dst {
		deg[i]++
	}
	for i, n := range deg {
		deg[i] = float32(1 / math.Sqrt(math.Max(float64(n), 1)))
	}
	t := tensor.FromSlice(deg, g.nodes, 1).To(d)
	if err := t.Err(); err != nil {
		return t
	}
	if g.norms == nil {
		g.norms = map[tensor.Device]*tensor.Tensor{}
	}
	g.norms[d] = t
	return t
}

// GCNConv is the graph convolution of Kipf and Welling, like PyTorch
// Geometric's GCNConv: it adds a self loop to every node and computes
// D^-1/2 Â D^-1/2 x Wᵀ + b, where Â is the adjacency matrix with the
// loops and D its in-degrees, as one sum over the neighbours of each
// node.
type GCNConv struct {
	Weight *tensor.Param // [out, in]
	Bias   *tensor.Param // [out], or nil
}

// NewGCNConv returns a graph convolution from in to out features, with
// Glorot-uniform weights and a zero bias, PyTorch Geometric's defaults.
func NewGCNConv(in, out int, opts ...Option) *GCNConv {
	o := resolve(opts, options{})
	c := &GCNConv{Weight: glorot(o, out, in)}
	if !o.noBias {
		c.Bias = tensor.NewParam(tensor.Zeros(tensor.Float32, out))
	}
	return c
}

// Convolve maps node features x, [nodes, in], over g to [nodes, out].
func (c *GCNConv) Convolve(g *Graph, x *tensor.Tensor) *tensor.Tensor {
	loops := g.WithSelfLoops()
	adj, err := loops.adjacency()
	if err != nil {
		return tensor.Failed(err)
	}
	norm := loops.degreeNorm(x.Device())
	h := x.MatMulT(c.Weight.Value()).Mul(norm)
	y := adj.MatMul(h).Mul(norm)
	if c.Bias != nil {
		y = y.Add(c.Bias.Value())
	}
	return y
}

// SAGEConv is GraphSAGE with the mean aggregator, like PyTorch
// Geometric's SAGEConv: each node maps the mean of its neighbours'
// features and its own, x_i Wrᵀ + mean_j(x_j) Wlᵀ + b.
type SAGEConv struct {
	Neighbors *Linear // the mean of the neighbours', with the bias
	Root      *Linear // the node's own, without a bias
}

// NewSAGEConv returns a GraphSAGE layer from in to out features, with
// Linear's default initialization.
func NewSAGEConv(in, out int, opts ...Option) *SAGEConv {
	return &SAGEConv{Neighbors: NewLinear(in, out, opts...), Root: NewLinear(in, out, append(slices.Clone(opts), NoBias())...)}
}

// Convolve maps node features x, [nodes, in], over g to [nodes, out].
func (c *SAGEConv) Convolve(g *Graph, x *tensor.Tensor) *tensor.Tensor {
	adj, err := g.adjacency()
	if err != nil {
		return tensor.Failed(err)
	}
	return c.Neighbors.Forward(adj.Mean(x)).Add(c.Root.Forward(x))
}

// GATConv is the graph attention layer of Veličković et al., like
// PyTorch Geometric's GATConv: each of Heads heads maps the features
// x Wᵀ, scores each edge j→i by LeakyReLU(a_src·h_j + a_dst·h_i),
// normalizes the scores over the edges into each node with a softmax,
// and sums the neighbours' features weighted by them. Every node
// attends to itself as well. The heads' results are concatenated or,
// with MeanHeads, averaged, and the bias added.
type GATConv struct {
	Weight *tensor.Param // [heads*out, in]
	AttSrc *tensor.Param // [heads, out]
	AttDst *tensor.Param // [heads, out]
	Bias   *tensor.Param // [heads*out], or [out] averaged; or nil

	heads, out int
	concat     bool
	slope      float64
}

// NewGATConv returns a graph attention layer from in to out features
// per head, with Glorot-uniform weights and a zero bias. The scores'
// LeakyReLU has a negative slope of 0.2 unless NegativeSlope sets it.
func NewGATConv(in, out, heads int, opts ...Option) *GATConv {
	o := resolve(opts, options{slope: 0.2})
	heads = max(heads, 1)
	c := &GATConv{
		Weight: glorot(o, heads*out, in),
		AttSrc: glorot(o, heads, out),
		AttDst: glorot(o, heads, out),
		heads:  heads, out: out, concat: !o.meanHeads, slope: o.slope,
	}
	if !o.noBias {
		n := out
		if c.concat {
			n *= heads
		}
		c.Bias = tensor.NewParam(tensor.Zeros(tensor.Float32, n))
	}
	return c
}

// Convolve maps node features x, [nodes, in], over g to [nodes,
// heads*out], or [nodes, out] with MeanHeads.
func (c *GATConv) Convolve(g *Graph, x *tensor.Tensor) *tensor.Tensor {
	loops := g.WithSelfLoops()
	n := g.Nodes()
	h := x.MatMulT(c.Weight.Value()).Reshape(n, c.heads, c.out)
	src := h.Mul(c.AttSrc.Value()).Sum(-1) // [nodes, heads]
	dst := h.Mul(c.AttDst.Value()).Sum(-1)
	s := loops.Source(src).Add(loops.Target(dst))
	s = tensor.Where(s.Greater(scalar(0)), s, s.Mul(scalar(c.slope))) // LeakyReLU
	alpha := loops.Softmax(s)                                         // [edges, heads]
	y := loops.Sum(loops.Source(h).Mul(alpha.Unsqueeze(-1)))
	if c.concat {
		y = y.Reshape(n, c.heads*c.out)
	} else {
		y = y.Mean(1)
	}
	if c.Bias != nil {
		y = y.Add(c.Bias.Value())
	}
	return y
}

// glorot returns a parameter of shape [fanOut, fanIn] drawn from the
// Glorot-uniform distribution U(-√(6/(fanIn+fanOut)), √(6/(fanIn+fanOut))).
func glorot(o options, fanOut, fanIn int) *tensor.Param {
	return uniform(o, math.Sqrt(6/float64(max(fanIn+fanOut, 1))), fanOut, fanIn)
}
