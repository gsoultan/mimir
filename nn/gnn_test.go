package nn_test

import (
	"errors"
	"maps"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/internal/goldentest"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/tensor"
)

// gnnCases computes the cases of gnn_cases in tools/golden/gen.py from
// their inputs: the features or messages first, then the edge index,
// then the layer's parameters. A layer's parameters are set from the
// inputs, and their gradients come from the parameters.
var gnnCases = map[string]func(t *testing.T, g *nn.Graph, in []*tensor.Tensor, grads map[int]goldentest.Grader) *tensor.Tensor{
	"aggregate_sum": func(_ *testing.T, g *nn.Graph, in []*tensor.Tensor, _ map[int]goldentest.Grader) *tensor.Tensor {
		return g.Sum(in[0])
	},
	"aggregate_mean": func(_ *testing.T, g *nn.Graph, in []*tensor.Tensor, _ map[int]goldentest.Grader) *tensor.Tensor {
		return g.Mean(in[0])
	},
	"aggregate_max": func(_ *testing.T, g *nn.Graph, in []*tensor.Tensor, _ map[int]goldentest.Grader) *tensor.Tensor {
		return g.Max(in[0])
	},
	"edge_softmax": func(_ *testing.T, g *nn.Graph, in []*tensor.Tensor, _ map[int]goldentest.Grader) *tensor.Tensor {
		return g.Softmax(in[0])
	},
	"gcn": func(t *testing.T, g *nn.Graph, in []*tensor.Tensor, grads map[int]goldentest.Grader) *tensor.Tensor {
		c := nn.NewGCNConv(5, 4)
		setGrads(t, in, grads, 2, c.Weight, c.Bias)
		return c.Convolve(g, in[0])
	},
	"sage": func(t *testing.T, g *nn.Graph, in []*tensor.Tensor, grads map[int]goldentest.Grader) *tensor.Tensor {
		c := nn.NewSAGEConv(5, 4)
		setGrads(t, in, grads, 2, c.Neighbors.Weight, c.Neighbors.Bias, c.Root.Weight)
		return c.Convolve(g, in[0])
	},
	"gat_concat": func(t *testing.T, g *nn.Graph, in []*tensor.Tensor, grads map[int]goldentest.Grader) *tensor.Tensor {
		c := nn.NewGATConv(5, 3, 2)
		setGrads(t, in, grads, 2, c.Weight, c.AttSrc, c.AttDst, c.Bias)
		return c.Convolve(g, in[0])
	},
	"gat_mean": func(t *testing.T, g *nn.Graph, in []*tensor.Tensor, grads map[int]goldentest.Grader) *tensor.Tensor {
		c := nn.NewGATConv(5, 3, 2, nn.MeanHeads())
		setGrads(t, in, grads, 2, c.Weight, c.AttSrc, c.AttDst, c.Bias)
		return c.Convolve(g, in[0])
	},
}

// setGrads sets ps from the inputs from the first on, and takes their
// gradients from the parameters.
func setGrads(t *testing.T, in []*tensor.Tensor, grads map[int]goldentest.Grader, first int, ps ...*tensor.Param) {
	t.Helper()
	if len(in) != first+len(ps) {
		t.Fatalf("%d inputs for %d parameters after %d", len(in), len(ps), first)
	}
	for i, p := range ps {
		set(t, p, in[first+i].Detach())
		grads[first+i] = p
	}
}

func TestGraphGolden(t *testing.T) {
	cases, meta := goldentest.Load(t, "gnn")
	t.Logf("golden oracle: %s", meta["oracle"])
	for _, name := range slices.Sorted(maps.Keys(gnnCases)) {
		if _, ok := cases[name]; !ok {
			t.Errorf("case %q has no golden data; regenerate with tools/golden/gen.py", name)
		}
	}
	tol := goldentest.Tol{Atol: 1e-5, Rtol: 1e-4}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			run, ok := gnnCases[name]
			if !ok {
				t.Fatalf("golden case %q has no Go counterpart", name)
			}
			in := goldentest.Inputs(t, c)
			g, err := nn.GraphFromEdgeIndex(t.Context(), 7, in[1]) // GNN_NODES
			if err != nil {
				t.Fatal(err)
			}
			grads := map[int]goldentest.Grader{0: in[0]}
			goldentest.CheckGradients(t, c, run(t, g, in, grads), grads, tol)
		})
	}
}

func TestGraphErrors(t *testing.T) {
	if _, err := nn.NewGraph(3, []int32{0, 3}, []int32{1, 2}); !errors.Is(err, tensor.ErrIndex) {
		t.Errorf("edge to a node outside: %v", err)
	}
	if _, err := nn.NewGraph(3, []int32{0}, []int32{1, 2}); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("lists of different lengths: %v", err)
	}
	if _, err := nn.GraphFromEdgeIndex(t.Context(), 3, tensor.Zeros(tensor.Int32, 3, 2)); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("edge index of 3 rows: %v", err)
	}
	g, err := nn.NewGraph(3, []int32{0, 1}, []int32{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	if err := nn.NewGCNConv(4, 2).Convolve(g, tensor.Ones(tensor.Float32, 5, 4)).Err(); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("features of 5 nodes for a graph of 3: %v", err)
	}
	if err := g.Sum(tensor.Ones(tensor.Float32, 3, 2)).Err(); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("messages of 3 edges for a graph of 2: %v", err)
	}
	loops := g.WithSelfLoops()
	if loops.Edges() != 5 || loops.Nodes() != 3 || g.WithSelfLoops() != loops {
		t.Errorf("with self loops: %d edges, %d nodes", loops.Edges(), loops.Nodes())
	}
}

func TestGraphSourceAndTargetGatherEdges(t *testing.T) {
	g, err := nn.NewGraph(3, []int32{2, 0, 2}, []int32{0, 1, 1})
	if err != nil {
		t.Fatal(err)
	}
	x := tensor.FromSlice([]float32{10, 20, 30}, 3, 1)
	if got := data(t, g.Source(x)); !slices.Equal(got, []float32{30, 10, 30}) {
		t.Errorf("sources %v", got)
	}
	if got := data(t, g.Target(x)); !slices.Equal(got, []float32{10, 20, 20}) {
		t.Errorf("targets %v", got)
	}
	// The adjacency sums neighbours as Sum of Source does.
	if got, want := data(t, g.Adjacency().MatMul(x)), data(t, g.Sum(g.Source(x))); !slices.Equal(got, want) {
		t.Errorf("adjacency product %v, sum of sources %v", got, want)
	}
}

// BenchmarkGCNConv times a GCN layer's forward and backward pass over a
// random graph of 100,000 nodes and a million edges, 64 features in and
// out.
func BenchmarkGCNConv(b *testing.B) {
	const nodes, edges, features = 100_000, 1_000_000, 64
	r := rand.New(rand.NewPCG(1, 2))
	src, dst := make([]int32, edges), make([]int32, edges)
	for e := range src {
		src[e], dst[e] = int32(r.IntN(nodes)), int32(r.IntN(nodes))
	}
	g, err := nn.NewGraph(nodes, src, dst)
	if err != nil {
		b.Fatal(err)
	}
	conv := nn.NewGCNConv(features, features)
	x := tensor.Randn(1, nodes, features)
	if err := x.Realize(b.Context()); err != nil {
		b.Fatal(err)
	}
	step := func() {
		if err := conv.Convolve(g, x).Sum().Backward(b.Context()); err != nil {
			b.Fatal(err)
		}
		conv.Weight.ZeroGrad()
		conv.Bias.ZeroGrad()
	}
	for range 3 {
		step()
	}
	for b.Loop() {
		step()
	}
}
