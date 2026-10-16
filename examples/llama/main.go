// Command llama generates text with a Llama-family model from a GGUF
// file, such as Llama 2 or 3, TinyLlama or SmolLM, quantized or not.
//
//	go run ./examples/llama -model Llama-3.2-1B-Instruct-Q4_K_M.gguf -prompt "The capital of France is"
//
// It builds the model from the file's metadata, keeps the weights in the
// file's types (Q8_0, Q4_K, Q6_K, F16, BF16 or F32) and reads them from a
// mapping of the file, tokenizes the prompt with the file's own
// vocabulary, runs the whole prompt in one batched pass and then
// generates one token at a time with a KV cache, sampling greedily or
// with a temperature, top-k and top-p. It prints the speed of the prompt
// and of the generation separately.
package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log"
	"math/rand/v2"
	"os"
	"os/signal"
	"runtime/pprof"
	"time"

	"github.com/gsoultan/mimir"
	"github.com/gsoultan/mimir/io/gguf"
	"github.com/gsoultan/mimir/tensor"
)

func main() {
	var (
		path    = flag.String("model", "", "GGUF file of a llama-architecture model")
		prompt  = flag.String("prompt", "Once upon a time", "text to continue")
		n       = flag.Int("n", 128, "tokens to generate at most")
		temp    = flag.Float64("temperature", 0.8, "sampling temperature; 0 picks the likeliest token")
		topK    = flag.Int("topk", 40, "sample among the k likeliest tokens; 0 for all")
		topP    = flag.Float64("topp", 0.95, "sample among the likeliest tokens whose probability adds up to p")
		seed    = flag.Uint64("seed", 1, "sampling seed")
		device  = flag.String("device", "cpu", "cpu, gpu or auto")
		raw     = flag.Bool("raw", false, "do not add the vocabulary's beginning-of-text token")
		profile = flag.String("cpuprofile", "", "write a CPU profile of the generation to this file")
	)
	flag.Parse()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	if *path == "" {
		log.Fatal("no -model: give a GGUF file of a llama-architecture model")
	}
	dev, err := pick(*device)
	if err != nil {
		log.Fatal(err)
	}
	s := &Sampler{Temperature: *temp, TopK: *topK, TopP: *topP, Rand: rand.New(rand.NewPCG(*seed, 3))}
	if err := run(ctx, dev, *path, *prompt, *n, s, !*raw, *profile); err != nil {
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

func run(ctx context.Context, dev tensor.Device, path, prompt string, n int, s *Sampler, addSpecial bool, profile string) error {
	start := time.Now()
	f, err := gguf.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	tok, err := NewTokenizer(f)
	if err != nil {
		return err
	}
	m, err := Load(f, dev)
	if err != nil {
		return err
	}
	c := m.Config
	name, _ := f.Metadata("general.name")
	fmt.Fprintf(os.Stderr, "%v: %d layers, %d features, %d heads (%d key heads), vocabulary of %d, on %v, loaded in %v\n",
		name, c.Layers, c.Embed, c.Heads, c.KVHeads, c.Vocab, dev, time.Since(start).Round(time.Millisecond))
	if tok.Len() != c.Vocab {
		return fmt.Errorf("llama: %d tokens in the vocabulary for an embedding of %d: %w", tok.Len(), c.Vocab, errTokenizer)
	}
	if profile != "" {
		pf, err := os.Create(profile)
		if err != nil {
			return err
		}
		defer pf.Close()
		if err := pprof.StartCPUProfile(pf); err != nil {
			return err
		}
		defer pprof.StopCPUProfile()
	}
	w := bufio.NewWriter(os.Stdout)
	defer w.Flush()
	fmt.Fprint(w, prompt)
	w.Flush()
	_, st, err := Generate(ctx, m, tok.Encode(prompt, addSpecial), n, s, tok.IsEOG, func(id int32) error {
		w.Write(tok.Piece(id))
		return w.Flush()
	})
	fmt.Fprintln(w)
	w.Flush()
	fmt.Fprintln(os.Stderr, st)
	return err
}
