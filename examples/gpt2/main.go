// Command gpt2 trains a small GPT-2 from scratch on a text file with
// mimir, then generates text from it.
//
//	go run ./examples/gpt2 -data input.txt -model small -steps 2000 -checkpoint gpt2.safetensors
//	go run ./examples/gpt2 -generate -checkpoint gpt2.safetensors -prompt "ROMEO:" -temperature 0.8 -topk 40
//
// The tokenizer is byte-level: every byte is a token. Training uses AdamW
// with a linear warmup and a cosine decay of the learning rate, clips the
// gradients' norm, and on a GPU keeps the weights in bfloat16 with float32
// master copies (-precision). It prints the loss and the tokens trained
// per second, and checkpoints to -checkpoint, from which it resumes.
// Generation samples with a temperature and top-k through the KV cache.
package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"io/fs"
	"log"
	"math/rand/v2"
	"os"
	"os/signal"
	"time"

	"github.com/gsoultan/mimir"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/tensor"
)

func main() {
	var (
		dataPath  = flag.String("data", "", "text file to train on")
		size      = flag.String("model", "small", "model size: tiny, small, base or gpt2 (GPT-2 small)")
		context_  = flag.Int("context", 128, "positions per sequence")
		batch     = flag.Int("batch", 16, "sequences per step")
		steps     = flag.Int("steps", 1000, "optimizer steps")
		lr        = flag.Float64("lr", 1e-3, "peak learning rate")
		warmup    = flag.Int("warmup", 100, "steps of learning-rate warmup")
		clip      = flag.Float64("clip", 1, "largest gradient norm, or 0")
		decay     = flag.Float64("weight-decay", 0.1, "AdamW weight decay")
		device    = flag.String("device", "cpu", "cpu, gpu or auto")
		precision = flag.String("precision", "auto", "f32, bf16, f16, or auto: bf16 on a GPU, f32 on the CPU")
		ckpt      = flag.String("checkpoint", "", "file to checkpoint to and resume from, or to generate from")
		every     = flag.Int("log", 50, "steps between progress lines and checkpoints")
		evalIter  = flag.Int("eval", 8, "validation batches per progress line")
		seed      = flag.Uint64("seed", 1, "seed of the weights, batches and sampling")
		generate  = flag.Bool("generate", false, "generate from -checkpoint instead of training")
		prompt    = flag.String("prompt", "\n", "text to continue")
		tokens    = flag.Int("tokens", 500, "tokens to generate")
		temp      = flag.Float64("temperature", 0.8, "sampling temperature; 0 picks the likeliest token")
		topK      = flag.Int("topk", 40, "sample among the k likeliest tokens; 0 for all")
	)
	flag.Parse()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	dev, err := pick(*device)
	if err != nil {
		log.Fatal(err)
	}
	tensor.SetDefaultDevice(dev)
	sampler := &Sampler{Temperature: *temp, TopK: *topK, Rand: rand.New(rand.NewPCG(*seed, 2))}
	if *generate {
		var g *GPT
		if *ckpt == "" {
			log.Fatal("no -checkpoint to generate from")
		}
		if g, err = LoadGPT(*ckpt); err == nil {
			err = sample(ctx, g, []byte(*prompt), *tokens, sampler)
		}
	} else {
		c, ok := sizes[*size]
		if !ok {
			log.Fatalf("unknown model size %q", *size)
		}
		c.Vocab, c.Context = byteVocab, *context_
		var half tensor.DType
		if half, err = precisionOf(*precision, dev); err != nil {
			log.Fatal(err)
		}
		cfg := TrainConfig{
			Batch: *batch, Steps: *steps, LR: *lr, MinLR: *lr / 10, Warmup: *warmup, Clip: *clip,
			Decay: *decay, Half: half, Seed: *seed, Log: max(*every, 1), EvalIter: *evalIter,
		}
		var g *GPT
		g, err = trainModel(ctx, *dataPath, c, cfg, *ckpt, dev)
		if err == nil && *tokens > 0 {
			fmt.Println("sample:")
			if err = nn.Cast(g, tensor.Float32); err == nil {
				err = sample(ctx, g, []byte(*prompt), min(*tokens, 200), sampler)
			}
		}
	}
	if err != nil {
		log.Fatal(err)
	}
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
	return mimir.Device{}, fmt.Errorf("device %q, want cpu, gpu or auto: %w", name, tensor.ErrDevice)
}

// precisionOf returns the dtype weights train in for the -precision flag
// on dev. On the CPU half types save memory but not time, as operations
// compute in float32 and round, so auto picks float32 there.
func precisionOf(name string, dev mimir.Device) (tensor.DType, error) {
	switch name {
	case "f32":
		return tensor.Float32, nil
	case "bf16":
		return tensor.BFloat16, nil
	case "f16":
		return tensor.Float16, nil
	case "auto":
		if dev.Equal(mimir.CPU()) {
			return tensor.Float32, nil
		}
		return tensor.BFloat16, nil
	}
	return 0, fmt.Errorf("precision %q, want f32, bf16, f16 or auto: %w", name, tensor.ErrDType)
}

// trainModel trains a model of shape c on the text at path and returns
// it.
func trainModel(ctx context.Context, path string, c Config, cfg TrainConfig, ckpt string, dev mimir.Device) (*GPT, error) {
	if path == "" {
		return nil, fmt.Errorf("no -data file to train on; for tiny Shakespeare run\n\tcurl -o input.txt %s\n%w", shakespeareURL, fs.ErrNotExist)
	}
	text, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	model, err := NewGPT(c, cfg.Seed)
	if err != nil {
		return nil, err
	}
	tr, err := NewTrainer(model, encode(text), cfg)
	if err != nil {
		return nil, err
	}
	if ckpt != "" {
		ok, err := tr.Resume(ckpt)
		if err != nil {
			return nil, err
		}
		if ok {
			fmt.Printf("resumed from %s at step %d\n", ckpt, tr.Opt.Steps())
		}
	}
	fmt.Printf("training a GPT-2 of %d layers, %d features and %d heads (%.2fM parameters) in %s on %v\n",
		c.Layers, c.Embed, c.Heads, float64(model.NumParams())/1e6, cfg.Half, dev)
	fmt.Printf("%d tokens of %s, %d sequences of %d per step\n", len(text), path, cfg.Batch, c.Context)
	return model, tr.Run(ctx, os.Stdout, ckpt)
}

// shakespeareURL is where Andrej Karpathy's char-rnn keeps tiny
// Shakespeare, about a megabyte of text.
const shakespeareURL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

// sample prints prompt and n tokens that g generates after it.
func sample(ctx context.Context, g *GPT, prompt []byte, n int, s *Sampler) error {
	w := bufio.NewWriter(os.Stdout)
	defer w.Flush()
	w.Write(prompt)
	start := time.Now()
	_, err := Generate(ctx, g, encode(prompt), n, s, true, func(tok int32) error {
		w.WriteByte(byte(tok))
		return w.Flush()
	})
	if err != nil {
		return err
	}
	fmt.Fprintf(w, "\n(%d tokens at %.0f tokens/s)\n", n, float64(n)/time.Since(start).Seconds())
	return nil
}
