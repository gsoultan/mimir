package mim

import (
	"context"
	"errors"
	"fmt"
	"io"
	"maps"
	"slices"

	mimfile "github.com/gsoultan/mim"

	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/tensor"
)

// SaveOptions are the optional parts of a file that Save or WriteTensors
// writes.
type SaveOptions struct {
	// Metadata is written as the file's metadata, if not empty.
	Metadata mimfile.Metadata
	// Training is written as the file's training section, if not nil: what
	// resuming training needs besides the weights, such as the optimizer
	// state, which it names among the tensors.
	Training *mimfile.Training
}

// Save writes every parameter and buffer of m to w as a file of weights
// alone, named by nn.NamedParams. Model.Load reads them back.
func Save(ctx context.Context, w io.Writer, m nn.Module, opts *SaveOptions) error {
	ts := map[string]*tensor.Tensor{}
	for name, p := range nn.NamedParams(m) {
		ts[name] = p.Value()
	}
	return WriteTensors(ctx, w, ts, opts)
}

// WriteTensors computes ts, together, and writes them to w by name as a
// file of weights alone. Tensors of dtypes Float32, Float16, BFloat16,
// Int32 and the block types are written as they are.
func WriteTensors(ctx context.Context, w io.Writer, ts map[string]*tensor.Tensor, opts *SaveOptions) error {
	names := slices.Sorted(maps.Keys(ts))
	all := make([]*tensor.Tensor, len(names))
	for i, name := range names {
		all[i] = ts[name]
	}
	if err := tensor.Realize(ctx, all...); err != nil {
		return fmt.Errorf("mim: save: %w", err)
	}
	fw := mimfile.NewWriter()
	for i, name := range names {
		t := all[i]
		dt, err := mimDType(t.DType())
		if err != nil {
			return fmt.Errorf("mim: save %s: %w", name, err)
		}
		b, err := t.Bytes(ctx)
		if err != nil {
			return fmt.Errorf("mim: save %s: %w", name, err)
		}
		shape := make([]uint64, t.Rank())
		for j, d := range t.Shape() {
			shape[j] = uint64(d)
		}
		if err := fw.AddTensor(name, dt, shape, b); err != nil {
			return fmt.Errorf("mim: save %s: %w", name, err)
		}
	}
	if opts != nil && len(opts.Metadata) > 0 {
		md, err := opts.Metadata.Encode()
		if err != nil {
			return fmt.Errorf("mim: save metadata: %w", err)
		}
		if err := fw.AddSection(mimfile.SectionMetadata, false, md); err != nil {
			return fmt.Errorf("mim: save metadata: %w", err)
		}
	}
	if opts != nil && opts.Training != nil {
		tr, err := opts.Training.Encode()
		if err != nil {
			return fmt.Errorf("mim: save training state: %w", err)
		}
		if err := fw.AddSection(mimfile.SectionTraining, false, tr); err != nil {
			return fmt.Errorf("mim: save training state: %w", err)
		}
	}
	if _, err := fw.WriteTo(w); err != nil {
		return fmt.Errorf("mim: save: %w", err)
	}
	return nil
}

// Names returns the names of the file's tensors, sorted, or nil once the
// model is closed.
func (m *Model) Names() []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil
	}
	names := make([]string, m.f.NumTensors())
	for i := range names {
		names[i] = m.f.TensorAt(i).Name
	}
	return names
}

// Tensor returns the file's tensor named name, on the CPU. It reads the
// file in place where mimir holds the tensor's dtype, as weights do (see
// Open), and holds the mapping for as long as it does.
func (m *Model) Tensor(name string) (*tensor.Tensor, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil, errors.New("mim: tensor of a closed model")
	}
	info, ok := m.f.Tensor(name)
	if !ok {
		return nil, fmt.Errorf("mim: no tensor %q: %w", name, tensor.ErrShape)
	}
	t, err := m.tensorOf(info)
	if err != nil {
		return nil, fmt.Errorf("mim: tensor %q: %w", name, err)
	}
	return t, nil
}

// Training returns the file's training section, or nil if it has none.
func (m *Model) Training() (*mimfile.Training, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil, errors.New("mim: training state of a closed model")
	}
	if _, ok := m.f.Section(mimfile.SectionTraining); !ok {
		return nil, nil
	}
	tr, err := m.f.Training()
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrFormat, err)
	}
	return tr, nil
}

// Load sets every parameter and buffer of mod from the file's tensor of
// its name, as Save writes them, which must have its shape. The values
// live on the default device. A float tensor of another dtype than its
// parameter, as a model saved after nn.Cast holds, is cast to the
// parameter's dtype; any other dtype difference is an error.
//
// On the CPU, a parameter reads its values in the file's mapping, without
// a copy of its own, and the file's pages stream in from disk as the
// values are read; an optimizer step then writes over the parameter's
// private copy of its pages, never the file. The parameters hold the
// mapping, so the model may be closed once Load returns.
func (m *Model) Load(mod nn.Module) error { return m.LoadPrefixed("", mod) }

// LoadPrefixed is Load for files that name mod's parameters prefix+name.
func (m *Model) LoadPrefixed(prefix string, mod nn.Module) error {
	for name, p := range nn.NamedParams(mod) {
		v, err := m.Tensor(prefix + name)
		if err != nil {
			return fmt.Errorf("mim: load %s: %w", name, err)
		}
		v = v.To(tensor.DefaultDevice())
		if cur := p.Value().DType(); v.DType() != cur && v.DType().IsFloat() && cur.IsFloat() {
			v = v.Cast(cur)
		}
		if err := p.Set(v); err != nil {
			return fmt.Errorf("mim: load %s: %w", name, err)
		}
	}
	return nil
}
