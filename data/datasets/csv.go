package datasets

import (
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"slices"
	"strconv"

	"github.com/gsoultan/mimir/data"
	"github.com/gsoultan/mimir/tensor"
)

// Table is a dataset read from a table of numbers.
type Table struct {
	*data.Tensors
	Features []string // column names of X, in order
	// Classes names the class of each index in Y when the target column
	// holds class labels; it is nil for a numeric target.
	Classes []string
}

// CSVOption configures CSV.
type CSVOption func(*csvConfig)

type csvConfig struct {
	classes bool
	names   []string
	with    bool // With was given
	data    []data.Option
}

// Classes reads the target column as class labels: Y holds Int32 indices
// into Table.Classes. With names, Table.Classes is names and a label not
// among them is an error; without, it lists the distinct labels in sorted
// order. Without Classes a numeric target becomes a Float32 Y for
// regression, and a non-numeric target is read as classes. CSVShards
// needs the names.
func Classes(names ...string) CSVOption {
	return func(c *csvConfig) { c.classes, c.names = true, names }
}

// With applies data options, such as data.BatchSize, to the dataset of
// CSV. CSVShards rejects it with data.ErrOption: the data options of a
// stream go to data.NewStream.
func With(opts ...data.Option) CSVOption {
	return func(c *csvConfig) { c.with, c.data = true, append(c.data, opts...) }
}

// classIndex returns the index of each class name, or an error for a
// duplicate.
func classIndex(names []string) (map[string]int32, error) {
	idx := make(map[string]int32, len(names))
	for i, name := range names {
		if _, dup := idx[name]; dup {
			return nil, fmt.Errorf("datasets: class %q listed twice: %w", name, data.ErrOption)
		}
		idx[name] = int32(i)
	}
	return idx, nil
}

// CSV reads a CSV table with a header row. The column named target becomes
// Y and every other column, which must be numeric, becomes a Float32
// feature in X [N, F].
func CSV(r io.Reader, target string, opts ...CSVOption) (*Table, error) {
	var cfg csvConfig
	for _, o := range opts {
		o(&cfg)
	}
	cr := csv.NewReader(r)
	cr.ReuseRecord = true
	header, err := cr.Read()
	if err != nil {
		return nil, fmt.Errorf("datasets: csv header: %w", err)
	}
	header = slices.Clone(header)
	ti := slices.Index(header, target)
	if ti < 0 {
		return nil, fmt.Errorf("datasets: csv has no column %q in %v: %w", target, header, ErrFormat)
	}
	features := slices.Delete(slices.Clone(header), ti, ti+1)
	var x []float32
	var labels []string
	for line := 2; ; line++ {
		rec, err := cr.Read()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("datasets: csv: %w", err)
		}
		for i, field := range rec {
			if i == ti {
				labels = append(labels, field)
				continue
			}
			v, err := strconv.ParseFloat(field, 32)
			if err != nil {
				return nil, fmt.Errorf("datasets: csv line %d column %q: %q is not a number: %w", line, header[i], field, ErrFormat)
			}
			x = append(x, float32(v))
		}
	}
	n := len(labels)
	if n == 0 {
		return nil, fmt.Errorf("datasets: csv has no rows: %w", ErrFormat)
	}
	t := &Table{Features: features}
	var y *tensor.Tensor
	if nums, ok := parseFloats(labels); ok && !cfg.classes {
		y = tensor.FromSlice(nums)
	} else if cfg.names != nil {
		index, err := classIndex(cfg.names)
		if err != nil {
			return nil, err
		}
		t.Classes = slices.Clone(cfg.names)
		idx := make([]int32, n)
		for i, l := range labels {
			c, ok := index[l]
			if !ok {
				return nil, fmt.Errorf("datasets: csv line %d column %q: unknown class %q: %w", i+2, target, l, ErrFormat)
			}
			idx[i] = c
		}
		y = tensor.FromSlice(idx)
	} else {
		t.Classes = slices.Compact(slices.Sorted(slices.Values(labels)))
		idx := make([]int32, n)
		for i, l := range labels {
			c, _ := slices.BinarySearch(t.Classes, l)
			idx[i] = int32(c)
		}
		y = tensor.FromSlice(idx)
	}
	if t.Tensors, err = data.FromTensors(tensor.FromSlice(x, n, len(features)), y, cfg.data...); err != nil {
		return nil, err
	}
	return t, nil
}

func parseFloats(s []string) ([]float32, bool) {
	out := make([]float32, len(s))
	for i, v := range s {
		f, err := strconv.ParseFloat(v, 32)
		if err != nil {
			return nil, false
		}
		out[i] = float32(f)
	}
	return out, true
}
