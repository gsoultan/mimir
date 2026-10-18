package datasets

import (
	"fmt"
	"slices"

	"github.com/gsoultan/mimir/data"
	"github.com/gsoultan/mimir/io/mim"
	"github.com/gsoultan/mimir/tensor"
)

// Mapped returns a dataset read from the Mimir Model file at path, in
// which the tensor named x holds the inputs and the tensor named y the
// targets, each with the examples along its first dimension. f32 tensors
// load as Float32 and i32 tensors as Int32; any other dtype is an error.
// The file is mapped into memory rather than read, and batches are views
// of the mapping, or gathers from it when shuffling, so a dataset larger
// than memory streams from disk through the page cache.
//
// The mapping lives as long as anything reads it: the dataset, and every
// batch, since a batch is a view of it or a gather from it. It is
// released once they are all garbage. On a default device other than the
// CPU the examples are copied to the device instead.
//
// Such a file is written with mim.WriteTensors:
//
//	err := mim.WriteTensors(ctx, w, map[string]*tensor.Tensor{
//		"x": tensor.FromSlice(pixels, n, 1, 28, 28),
//		"y": tensor.FromSlice(labels, n),
//	}, nil)
func Mapped(path, x, y string, opts ...data.Option) (*data.Tensors, error) {
	f, err := mim.Open(path)
	if err != nil {
		return nil, fmt.Errorf("datasets: %w", err)
	}
	// The tensors that read the mapping hold it, so the file's own hold
	// goes once they are made.
	defer f.Close()
	xt, err := mappedTensor(f, path, x)
	if err != nil {
		return nil, err
	}
	yt, err := mappedTensor(f, path, y)
	if err != nil {
		return nil, err
	}
	return data.FromTensors(xt, yt, opts...)
}

// mappedTensor returns the tensor of f named name, on the default device:
// over the mapping on the CPU, and as a copy elsewhere.
func mappedTensor(f *mim.Model, path, name string) (*tensor.Tensor, error) {
	if !slices.Contains(f.Names(), name) {
		return nil, fmt.Errorf("datasets: %s has no tensor %q: %w", path, name, ErrFormat)
	}
	t, err := f.Tensor(name)
	if err != nil {
		return nil, fmt.Errorf("datasets: %s: %w", path, err)
	}
	if dt := t.DType(); dt != tensor.Float32 && dt != tensor.Int32 {
		return nil, fmt.Errorf("datasets: %s: tensor %q has dtype %s, want f32 or i32: %w", path, name, dt, tensor.ErrDType)
	}
	t = t.To(tensor.DefaultDevice())
	if err := t.Err(); err != nil {
		return nil, fmt.Errorf("datasets: %s: tensor %q: %w", path, name, err)
	}
	return t, nil
}
