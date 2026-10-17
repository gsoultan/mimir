// Command gnn trains a two-layer graph convolutional network to classify
// the nodes of a citation graph, as Kipf and Welling did on Cora.
//
//	go run ./examples/gnn -epochs 200 -device auto
//
// The graph is generated in code, so nothing is downloaded: papers of
// seven topics cite papers of their own topic four times in five, and
// each paper's words are drawn mostly from its topic's vocabulary, with
// Cora's numbers of papers, citations and words. As on Cora, the network
// trains on 20 labelled papers per topic, picks the epoch that does best
// on 500 others and reports its accuracy on 1,000 more.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"time"

	"github.com/gsoultan/mimir"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/nn/loss"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
)

func main() {
	var (
		epochs = flag.Int("epochs", 200, "full-graph training steps")
		hidden = flag.Int("hidden", 16, "features of the hidden layer")
		lr     = flag.Float64("lr", 0.01, "Adam learning rate")
		seed   = flag.Uint64("seed", 1, "seed of the generated graph")
		device = flag.String("device", "cpu", "cpu, gpu or auto")
	)
	flag.Parse()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	dev, err := pick(*device)
	if err != nil {
		log.Fatal(err)
	}
	cfg := config{Citations: cora, Epochs: *epochs, Hidden: *hidden, LR: *lr, Seed: *seed}
	res, err := run(ctx, dev, cfg, os.Stdout)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("test accuracy %.1f%% at epoch %d (best validation %.1f%%), %v\n",
		100*res.Test, res.Epoch, 100*res.Val, res.Time.Round(time.Millisecond))
}

// pick returns the device that name selects.
func pick(name string) (mimir.Device, error) {
	switch name {
	case "cpu":
		return mimir.CPU(), nil
	case "gpu":
		return mimir.GPU()
	case "auto":
		return mimir.Auto(), nil
	}
	return mimir.Device{}, fmt.Errorf("unknown device %q, want cpu, gpu or auto", name)
}

// config sets up a run.
type config struct {
	Citations
	Epochs, Hidden int
	LR             float64
	Seed           uint64
}

// result is how a run went: the accuracy on the test papers at the epoch
// with the best validation accuracy, and the time training took.
type result struct {
	Test, Val float64
	Epoch     int
	Time      time.Duration
}

// GCN is the two-layer network of Kipf and Welling over one graph:
// convolution, ReLU and dropout, then a convolution to the class logits.
// Kipf and Welling also drop input features, which on these dense
// features costs more than the rest of a step and changes little.
type GCN struct {
	Conv1, Conv2 *nn.GCNConv
	Drop         *nn.Dropout
	graph        *nn.Graph
}

// Forward maps the features of every node to class logits.
func (m *GCN) Forward(x *tensor.Tensor) *tensor.Tensor {
	h := m.Conv1.Convolve(m.graph, x).ReLU()
	return m.Conv2.Convolve(m.graph, m.Drop.Forward(h))
}

func run(ctx context.Context, dev mimir.Device, cfg config, progress *os.File) (result, error) {
	cite, err := generate(cfg.Citations, cfg.Seed)
	if err != nil {
		return result{}, err
	}
	nn.ManualSeed(cfg.Seed)
	model := &GCN{
		Conv1: nn.NewGCNConv(cfg.Words, cfg.Hidden), Conv2: nn.NewGCNConv(cfg.Hidden, cfg.Topics),
		Drop:  nn.NewDropout(0.5),
		graph: cite.graph,
	}
	if err := nn.To(model, dev); err != nil {
		return result{}, err
	}
	// Weight decay on the first layer only, as Kipf and Welling had it.
	opt := optim.Adam([]*tensor.Param{model.Conv1.Weight, model.Conv1.Bias}, optim.LR(cfg.LR), optim.WeightDecay(5e-4))
	opt2 := optim.Adam([]*tensor.Param{model.Conv2.Weight, model.Conv2.Bias}, optim.LR(cfg.LR))
	x := cite.features.To(dev)
	trainY := cite.labelsOf(cite.train).To(dev)
	if progress != nil {
		fmt.Fprintf(progress, "%d papers, %d citations, %d words, %d topics; training on %d on %v\n",
			cfg.Papers, cite.graph.Edges()/2, cfg.Words, cfg.Topics, len(cite.train), dev)
	}

	var best result
	start := time.Now()
	for epoch := 1; epoch <= cfg.Epochs; epoch++ {
		nn.Train(model)
		l := loss.CrossEntropy(cite.trainRows(model.Forward(x)), trainY)
		if err := l.Backward(ctx); err != nil {
			return result{}, err
		}
		for _, o := range []*optim.Optimizer{opt, opt2} {
			if err := o.Step(ctx); err != nil {
				return result{}, err
			}
			o.ZeroGrad()
		}
		nn.Eval(model)
		logits := model.Forward(x)
		val, err := cite.accuracy(ctx, cite.val, logits)
		if err != nil {
			return result{}, err
		}
		if val > best.Val || best.Epoch == 0 {
			test, err := cite.accuracy(ctx, cite.test, logits)
			if err != nil {
				return result{}, err
			}
			best = result{Test: test, Val: val, Epoch: epoch}
		}
		if progress != nil && (epoch%20 == 0 || epoch == cfg.Epochs) {
			lv, err := l.Item[float32](ctx)
			if err != nil {
				return result{}, err
			}
			fmt.Fprintf(progress, "epoch %3d  loss %.4f  validation %.1f%%\n", epoch, lv, 100*val)
		}
	}
	best.Time = time.Since(start)
	return best, nil
}
