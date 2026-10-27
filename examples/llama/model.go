package main

import (
	"math"

	mimfile "github.com/gsoultan/mimir-model"

	"github.com/gsoultan/mimir/io/mim"
	"github.com/gsoultan/mimir/llm"
	"github.com/gsoultan/mimir/tensor"
)

// Model is a Llama-family decoder: Llama 2 and 3, TinyLlama, SmolLM and
// others of the "llama" architecture of llama.cpp, and Qwen 2. Package llm
// builds it, and fine-tunes it with LoRA.
type Model = llm.Model

// Load reads the model in f, whose metadata is md, to dev. Its weights
// keep the file's types, quantized or not; on the CPU they read the
// file's mapping in place.
func Load(f *mim.Model, md mimfile.Metadata, dev tensor.Device) (*Model, error) {
	return llm.Load(f, md, dev)
}

// metaInt returns md[key] as an integer, which the metadata holds as an
// int64 or a uint64, and whether it is one.
func metaInt(md mimfile.Metadata, key string) (int64, bool) {
	switch v := md[key].(type) {
	case int64:
		return v, true
	case uint64:
		return int64(v), v <= math.MaxInt64
	}
	return 0, false
}
