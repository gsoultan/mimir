package main

import (
	"context"
	"fmt"
	"math/rand/v2"
	"slices"

	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/tensor"
)

// Citations sizes a generated citation graph.
type Citations struct {
	Papers, Links, Words, Topics int
	// Homophily is the share of citations within a topic, and Focus the
	// share of a paper's words drawn from its topic's vocabulary.
	Homophily, Focus float64
	// WordsPerPaper is the mean number of distinct words of a paper.
	WordsPerPaper int
	// Train labelled papers per topic, and Val and Test papers to
	// validate and test on.
	Train, Val, Test int
}

// cora has the numbers of the Cora citation graph.
var cora = Citations{
	Papers: 2708, Links: 5278, Words: 1433, Topics: 7,
	Homophily: 0.75, Focus: 0.3, WordsPerPaper: 18,
	Train: 20, Val: 500, Test: 1000,
}

// citations is a generated citation graph: its graph, with an edge each
// way for every citation, the papers' bag-of-words features, normalized
// to sum to 1, their topics and how they are split.
type citations struct {
	graph            *nn.Graph
	features         *tensor.Tensor // [papers, words]
	labels           []int32
	train, val, test []int32
	// pick is the sparse matrix that picks the rows of train.
	pick *tensor.Pattern
}

// generate returns a random citation graph shaped by c, the same for the
// same seed.
func generate(c Citations, seed uint64) (*citations, error) {
	if c.Papers < c.Topics*c.Train+c.Val+c.Test || c.Topics < 2 || c.Words < c.Topics {
		return nil, fmt.Errorf("gnn: %d papers of %d topics cannot split %d, %d and %d ways", c.Papers, c.Topics, c.Train, c.Val, c.Test)
	}
	rng := rand.New(rand.NewPCG(seed, 0x9e3779b97f4a7c15))
	labels := make([]int32, c.Papers)
	byTopic := make([][]int32, c.Topics)
	for p := range labels {
		labels[p] = int32(rng.IntN(c.Topics))
		byTopic[labels[p]] = append(byTopic[labels[p]], int32(p))
	}
	// Citations, mostly within a topic, each one an edge either way.
	src, dst := make([]int32, 0, 2*c.Links), make([]int32, 0, 2*c.Links)
	for len(src) < 2*c.Links {
		a := int32(rng.IntN(c.Papers))
		var b int32
		if same := byTopic[labels[a]]; rng.Float64() < c.Homophily {
			b = same[rng.IntN(len(same))]
		} else {
			b = int32(rng.IntN(c.Papers))
		}
		if a != b {
			src, dst = append(src, a, b), append(dst, b, a)
		}
	}
	g, err := nn.NewGraph(c.Papers, src, dst)
	if err != nil {
		return nil, err
	}
	// Words: each topic has a vocabulary of its own words, and a paper
	// draws a share Focus of its words from its topic's.
	vocab := c.Words / c.Topics
	features := make([]float32, c.Papers*c.Words)
	for p, topic := range labels {
		row := features[p*c.Words : (p+1)*c.Words]
		n := 1 + rng.IntN(2*c.WordsPerPaper)
		for range n {
			w := rng.IntN(c.Words)
			if rng.Float64() < c.Focus {
				w = int(topic)*vocab + rng.IntN(vocab)
			}
			row[w] = 1
		}
		var sum float32
		for _, v := range row {
			sum += v
		}
		for w := range row {
			row[w] /= sum
		}
	}
	cite := &citations{
		graph: g, features: tensor.FromSlice(features, c.Papers, c.Words), labels: labels,
	}
	// Train on the first Train papers of each topic, and validate and
	// test on others, chosen at random.
	taken := make([]bool, c.Papers)
	for _, papers := range byTopic {
		for _, p := range papers[:min(c.Train, len(papers))] {
			cite.train = append(cite.train, p)
			taken[p] = true
		}
	}
	rest := slices.DeleteFunc(rng.Perm(c.Papers), func(p int) bool { return taken[p] })
	for _, p := range rest[:c.Val] {
		cite.val = append(cite.val, int32(p))
	}
	for _, p := range rest[c.Val : c.Val+c.Test] {
		cite.test = append(cite.test, int32(p))
	}
	at := make([]int32, len(cite.train))
	for i := range at {
		at[i] = int32(i)
	}
	if cite.pick, err = tensor.NewPattern(len(at), c.Papers, at, cite.train); err != nil {
		return nil, err
	}
	return cite, nil
}

// labelsOf returns the topics of papers, as an Int32 tensor.
func (c *citations) labelsOf(papers []int32) *tensor.Tensor {
	y := make([]int32, len(papers))
	for i, p := range papers {
		y[i] = c.labels[p]
	}
	return tensor.FromSlice(y)
}

// trainRows returns the rows of x, [papers, F], of the training papers,
// as a sparse product that passes the gradient back to them alone.
func (c *citations) trainRows(x *tensor.Tensor) *tensor.Tensor { return c.pick.MatMul(x) }

// accuracy returns the share of papers whose largest logit is their
// topic's.
func (c *citations) accuracy(ctx context.Context, papers []int32, logits *tensor.Tensor) (float64, error) {
	pred, err := logits.Argmax(1).Data[int32](ctx)
	if err != nil {
		return 0, err
	}
	right := 0
	for _, p := range papers {
		if pred[p] == c.labels[p] {
			right++
		}
	}
	return float64(right) / float64(len(papers)), nil
}
