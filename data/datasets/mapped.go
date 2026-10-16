package datasets

import (
	"fmt"

	"github.com/gsoultan/mimir/data"
	"github.com/gsoultan/mimir/io/safetensors"
	"github.com/gsoultan/mimir/tensor"
)

// Mapped returns a dataset read from the safetensors file at path, in
// which the tensor named x holds the inputs and the tensor named y the
// targets, each with the examples along its first dimension. F32 tensors
// load as Float32 and I32 tensors as Int32; any other dtype is an error.
// The file is mapped into memory rather than read, and batches are views
// of the mapping, or gathers from it when shuffling, so a dataset larger
// than memory streams from disk through the page cache.
//
// The mapping lives as long as anything reads it: the dataset, and every
// batch, since a batch is a view of it or a gather from it. It is
// released once they are all garbage. On a default device other than the
// CPU the examples are copied to the device instead. The tensors must
// start at a multiple of 4 bytes, as safetensors.Write lays them out, to
// be read in place; others are copied.
//
// Such a file is written with safetensors.Write from Float32Tensor and
// Int32Tensor:
//
//	err := safetensors.Write(w, map[string]safetensors.Tensor{
//		"x": safetensors.Float32Tensor([]int{n, 1, 28, 28}, pixels),
//		"y": safetensors.Int32Tensor([]int{n}, labels),
//	}, nil)
func Mapped(path, x, y string, opts ...data.Option) (*data.Tensors, error) {
	f, err := safetensors.Open(path)
	if err != nil {
		return nil, fmt.Errorf("datasets: %w", err)
	}
	// The tensors that read the mapping hold it, so the file's own hold
	// goes once they are made.
	defer f.Close()
	return fromFile(f, path, x, y, opts)
}

// fromFile builds the dataset of the tensors x and y of f on the default
// device.
func fromFile(f *safetensors.File, path, x, y string, opts []data.Option) (*data.Tensors, error) {
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
func mappedTensor(f *safetensors.File, path, name string) (*tensor.Tensor, error) {
	st, ok := f.Tensor(name)
	if !ok {
		return nil, fmt.Errorf("datasets: %s has no tensor %q: %w", path, name, ErrFormat)
	}
	var dt tensor.DType
	switch st.DType {
	case "F32":
		dt = tensor.Float32
	case "I32":
		dt = tensor.Int32
	default:
		return nil, fmt.Errorf("datasets: %s: tensor %q has dtype %s, want F32 or I32: %w", path, name, st.DType, tensor.ErrDType)
	}
	t := tensor.FromBytesFunc(dt, st.Data, f.Hold(), st.Shape...).To(tensor.DefaultDevice())
	if err := t.Err(); err != nil {
		return nil, fmt.Errorf("datasets: %s: tensor %q: %w", path, name, err)
	}
	return t, nil
}
