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

	mimfile "github.com/gsoultan/mim"

	"github.com/gsoultan/mimir/io/mim"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
)

const checkpointFile = "checkpoint.mim"

// save writes the model's parameters as "model.<name>" and the optimizer's
// state as "optim.<key>", with the progress in the file's training
// section, replacing the previous checkpoint atomically.
func save(ctx context.Context, dir string, model nn.Module, opt *optim.Optimizer, epochs int) error {
	ts := map[string]*tensor.Tensor{}
	for name, p := range nn.NamedParams(model) {
		ts["model."+name] = p.Value()
	}
	state, steps := opt.Snapshot()
	tr := &mimfile.Training{
		Counters: []mimfile.TrainCounter{{Name: "epochs", Value: uint64(epochs)}, {Name: "steps", Value: uint64(steps)}},
		Hyper:    []mimfile.TrainHyper{{Name: "lr", Value: opt.LR()}},
	}
	for _, key := range slices.Sorted(maps.Keys(state)) {
		ts["optim."+key] = state[key]
		tr.Extras = append(tr.Extras, mimfile.TrainExtra{Name: key, Tensor: "optim." + key})
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".checkpoint-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name()) // no-op after the rename
	err = mim.WriteTensors(ctx, tmp, ts, &mim.SaveOptions{Training: tr})
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
	f, err := mim.Open(filepath.Join(dir, checkpointFile))
	if errors.Is(err, fs.ErrNotExist) {
		return 0, nil
	}
	if err != nil {
		return 0, fmt.Errorf("train: checkpoint: %w", err)
	}
	defer f.Close()
	tr, err := f.Training()
	if err != nil {
		return 0, fmt.Errorf("train: checkpoint: %w", err)
	}
	if tr == nil {
		return 0, fmt.Errorf("train: checkpoint without a training section: %w", mim.ErrFormat)
	}
	counters := map[string]uint64{}
	for _, c := range tr.Counters {
		counters[c.Name] = c.Value
	}
	epochs, okE := counters["epochs"]
	steps, okS := counters["steps"]
	i := slices.IndexFunc(tr.Hyper, func(h mimfile.TrainHyper) bool { return h.Name == "lr" })
	if !okE || !okS || i < 0 {
		return 0, fmt.Errorf("train: checkpoint training section %+v: %w", tr, mim.ErrFormat)
	}
	if err := f.LoadPrefixed("model.", model); err != nil {
		return 0, fmt.Errorf("train: checkpoint: %w", err)
	}
	state := map[string]*tensor.Tensor{}
	for _, e := range tr.Extras {
		t, err := f.Tensor(e.Tensor)
		if err != nil {
			return 0, fmt.Errorf("train: checkpoint: %w", err)
		}
		state[e.Name] = t.To(tensor.DefaultDevice())
	}
	if err := opt.Restore(state, int(steps)); err != nil {
		return 0, fmt.Errorf("train: checkpoint: %w", err)
	}
	opt.SetLR(tr.Hyper[i].Value)
	return int(epochs), nil
}

func sortedKeys(m map[string]float64) []string { return slices.Sorted(maps.Keys(m)) }
