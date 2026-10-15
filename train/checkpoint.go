package train

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"github.com/gsoultan/mimir/io/safetensors"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
)

const checkpointFile = "checkpoint.safetensors"

// save writes the model's parameters as "model.<name>", the optimizer's
// state as "optim.<key>" and the progress as metadata, replacing the
// previous checkpoint atomically.
func save(ctx context.Context, dir string, model nn.Module, opt *optim.Optimizer, epochs int) error {
	ts := map[string]safetensors.Tensor{}
	for name, p := range nn.NamedParams(model) {
		t, err := nn.Encode(ctx, p.Value())
		if err != nil {
			return fmt.Errorf("train: checkpoint %s: %w", name, err)
		}
		ts["model."+name] = t
	}
	state, steps := opt.Snapshot()
	for key, v := range state {
		t, err := nn.Encode(ctx, v)
		if err != nil {
			return fmt.Errorf("train: checkpoint optimizer %s: %w", key, err)
		}
		ts["optim."+key] = t
	}
	meta := map[string]string{
		"epochs": strconv.Itoa(epochs),
		"steps":  strconv.Itoa(steps),
		"lr":     strconv.FormatFloat(opt.LR(), 'g', -1, 64),
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".checkpoint-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name()) // no-op after the rename
	err = safetensors.Write(tmp, ts, meta)
	if serr := tmp.Sync(); err == nil {
		err = serr
	}
	if cerr := tmp.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return fmt.Errorf("train: checkpoint: %w", err)
	}
	return os.Rename(tmp.Name(), filepath.Join(dir, checkpointFile))
}

// resume loads dir's checkpoint, if any, into model and opt and returns
// the number of epochs it had completed.
func resume(dir string, model nn.Module, opt *optim.Optimizer) (int, error) {
	b, err := os.ReadFile(filepath.Join(dir, checkpointFile))
	if errors.Is(err, fs.ErrNotExist) {
		return 0, nil
	}
	if err != nil {
		return 0, err
	}
	f, err := safetensors.Parse(b)
	if err != nil {
		return 0, fmt.Errorf("train: checkpoint: %w", err)
	}
	epochs, errE := strconv.Atoi(f.Metadata["epochs"])
	steps, errS := strconv.Atoi(f.Metadata["steps"])
	lr, errL := strconv.ParseFloat(f.Metadata["lr"], 64)
	if errE != nil || errS != nil || errL != nil {
		return 0, fmt.Errorf("train: checkpoint metadata %v: %w", f.Metadata, safetensors.ErrFormat)
	}
	if err := nn.LoadPrefixed(f, "model.", model); err != nil {
		return 0, fmt.Errorf("train: checkpoint: %w", err)
	}
	state := map[string]*tensor.Tensor{}
	for _, name := range f.Names() {
		key, ok := strings.CutPrefix(name, "optim.")
		if !ok {
			continue
		}
		st, _ := f.Tensor(name)
		if state[key], err = nn.Decode(st); err != nil {
			return 0, fmt.Errorf("train: checkpoint: %w", err)
		}
	}
	if err := opt.Restore(state, steps); err != nil {
		return 0, fmt.Errorf("train: checkpoint: %w", err)
	}
	opt.SetLR(lr)
	return epochs, nil
}

func sortedKeys(m map[string]float64) []string { return slices.Sorted(maps.Keys(m)) }
