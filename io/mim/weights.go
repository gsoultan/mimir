package mim

import (
	"context"
	"errors"
	"fmt"
	"io"
	"maps"
	"slices"
	"unicode/utf8"

	mimfile "github.com/gsoultan/mimir-model"

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
	// Hash writes the SHA-256 hash of every section, which a reader may
	// check (see mimfile.File.VerifyHashes), at the cost of reading every
	// tensor twice: once for the hash, which comes first in the file, and
	// once to write it.
	Hash bool
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
// Int32 and the block types are written as they are. Each tensor's bytes
// are copied from its device as they are written, so writing takes the
// memory of the largest tensor, not of all of them.
func WriteTensors(ctx context.Context, w io.Writer, ts map[string]*tensor.Tensor, opts *SaveOptions) error {
	if opts == nil {
		opts = &SaveOptions{}
	}
	names := slices.Sorted(maps.Keys(ts))
	all := make([]*tensor.Tensor, len(names))
	for i, name := range names {
		all[i] = ts[name]
	}
	if err := tensor.Realize(ctx, all...); err != nil {
		return fmt.Errorf("mim: save: %w", err)
	}
	fts := make([]fileTensor, len(names))
	for i, name := range names {
		if name == "" || len(name) > mimfile.MaxNameLength || !utf8.ValidString(name) {
			return fmt.Errorf("mim: save: tensor name %q: %w", name, ErrFormat)
		}
		t := all[i]
		dt, err := mimDType(t.DType())
		if err != nil {
			return fmt.Errorf("mim: save %s: %w", name, err)
		}
		shape := make([]uint64, t.Rank())
		for j, d := range t.Shape() {
			shape[j] = uint64(d)
		}
		if len(shape) > mimfile.MaxRank {
			return fmt.Errorf("mim: save %s: rank %d: %w", name, len(shape), tensor.ErrShape)
		}
		n, err := mimfile.ByteSize(dt, shape)
		if err != nil {
			return fmt.Errorf("mim: save %s: %w", name, err)
		}
		fts[i] = fileTensor{name: name, t: t, dtype: dt, shape: shape, length: n}
	}
	var sections []fileSection
	if len(opts.Metadata) > 0 {
		md, err := opts.Metadata.Encode()
		if err != nil {
			return fmt.Errorf("mim: save metadata: %w", err)
		}
		sections = append(sections, fileSection{mimfile.SectionMetadata, md})
	}
	if opts.Training != nil {
		tr, err := opts.Training.Encode()
		if err != nil {
			return fmt.Errorf("mim: save training state: %w", err)
		}
		sections = append(sections, fileSection{mimfile.SectionTraining, tr})
	}
	if err := writeFile(ctx, w, fts, sections, opts.Hash); err != nil {
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
// its name, as Save writes them, on the default device. It is LoadOn with
// tensor.DefaultDevice().
func (m *Model) Load(mod nn.Module) error { return m.LoadOn(mod, tensor.DefaultDevice()) }

// LoadPrefixed is Load for files that name mod's parameters prefix+name.
func (m *Model) LoadPrefixed(prefix string, mod nn.Module) error {
	return m.LoadPrefixedOn(prefix, mod, tensor.DefaultDevice())
}

// LoadOn sets every parameter and buffer of mod from the file's tensor of
// its name, as Save writes them, which must have its shape, with the
// values on d. A float tensor of another dtype than its parameter, as a
// model saved after nn.Cast holds, is cast to the parameter's dtype; a
// trainable parameter takes float values only.
//
// LoadOn changes all of mod's parameters or none: it finds, checks and
// computes every value on d before it sets the first, so a missing
// tensor, a shape or dtype that does not fit, or a failure to copy the
// values to d leaves mod as it was.
//
// On the CPU, a parameter reads its values in the file's mapping, without
// a copy of its own, and the file's pages stream in from disk as the
// values are read; an optimizer step then writes over the parameter's
// private copy of its pages, never the file. The parameters hold the
// mapping, so the model may be closed once LoadOn returns.
func (m *Model) LoadOn(mod nn.Module, d tensor.Device) error { return m.LoadPrefixedOn("", mod, d) }

// LoadPrefixedOn is LoadOn for files that name mod's parameters
// prefix+name.
func (m *Model) LoadPrefixedOn(prefix string, mod nn.Module, d tensor.Device) error {
	type load struct {
		name     string
		p        *tensor.Param
		old, new *tensor.Tensor
	}
	var loads []load
	for name, p := range nn.NamedParams(mod) {
		v, err := m.Tensor(prefix + name)
		if err != nil {
			return fmt.Errorf("mim: load %s: %w", name, err)
		}
		cur := p.Value()
		if err := cur.Err(); err != nil {
			return fmt.Errorf("mim: load %s: %w", name, err)
		}
		if !slices.Equal(v.Shape(), cur.Shape()) {
			return fmt.Errorf("mim: load %s: %s%v into a %s%v parameter: %w", name, v.DType(), v.Shape(), cur.DType(), cur.Shape(), tensor.ErrShape)
		}
		if v.DType() != cur.DType() && v.DType().IsFloat() && cur.DType().IsFloat() {
			v = v.Cast(cur.DType())
		}
		if p.Trainable() && !v.DType().IsFloat() {
			return fmt.Errorf("mim: load %s: %s values into a trainable parameter: %w", name, v.DType(), tensor.ErrDType)
		}
		loads = append(loads, load{name: name, p: p, old: cur, new: v.To(d)})
	}
	vs := make([]*tensor.Tensor, len(loads))
	for i, l := range loads {
		vs[i] = l.new
	}
	// Values the file maps on the CPU are computed already, and stay
	// mapped; copies and casts are computed now, so a failure to make
	// them, such as running out of memory on d, sets nothing.
	if err := tensor.Realize(context.Background(), vs...); err != nil {
		return fmt.Errorf("mim: load on %v: %w", d, err)
	}
	for i, l := range loads {
		if err := l.p.Set(l.new); err != nil {
			// Only a change to mod during LoadOn gets here: put back the
			// values set before it.
			for _, done := range slices.Backward(loads[:i]) {
				_ = done.p.Set(done.old)
			}
			return fmt.Errorf("mim: load %s: %w", l.name, err)
		}
	}
	return nil
}
