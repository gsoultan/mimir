// Command onnx runs an ONNX model and prints its outputs and how long a
// run takes.
//
//	go run ./examples/onnx -model io/onnx/testdata/transformer.onnx -dim batch=4 -dim seq=16 -intmax 50
//
// It feeds the model random inputs, normal floats and integers in [0,
// -intmax), or the tensors of a safetensors file named after the inputs
// (-inputs), with symbolic dimensions set by -dim and 1 otherwise. It
// prints each output's shape and first values, and the time of the
// first run, which plans the schedule, and of the runs after it, which
// replay it.
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"math/rand/v2"
	"os"
	"os/signal"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/gsoultan/mimir"
	"github.com/gsoultan/mimir/io/onnx"
	"github.com/gsoultan/mimir/io/safetensors"
	"github.com/gsoultan/mimir/tensor"
)

// dims collects -dim name=size flags.
type dims map[string]int

func (d dims) String() string { return fmt.Sprint(map[string]int(d)) }

func (d dims) Set(s string) error {
	name, size, ok := strings.Cut(s, "=")
	n, err := strconv.Atoi(size)
	if !ok || err != nil || n < 0 {
		return fmt.Errorf("dimension %q, want name=size", s)
	}
	d[name] = n
	return nil
}

// config is what a run of the command does.
type config struct {
	model  string
	inputs string // safetensors file of inputs, or ""
	dims   dims
	runs   int
	intMax int
	seed   uint64
	dev    tensor.Device
}

func main() {
	c := config{dims: dims{}}
	flag.StringVar(&c.model, "model", "", "ONNX model file")
	flag.StringVar(&c.inputs, "inputs", "", "safetensors file of the inputs, by name; random inputs otherwise")
	flag.Var(c.dims, "dim", "size of a symbolic dimension, as name=size; repeatable")
	flag.IntVar(&c.runs, "runs", 20, "runs to time after the first")
	flag.IntVar(&c.intMax, "intmax", 2, "random integer inputs lie in [0, intmax)")
	flag.Uint64Var(&c.seed, "seed", 1, "seed of the random inputs")
	device := flag.String("device", "cpu", "cpu, gpu or auto")
	flag.Parse()
	if c.model == "" {
		log.Fatal("no -model: give an ONNX file")
	}
	var err error
	if c.dev, err = pick(*device); err != nil {
		log.Fatal(err)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	if err := run(ctx, os.Stdout, c); err != nil {
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

func run(ctx context.Context, w io.Writer, c config) error {
	start := time.Now()
	m, err := onnx.Open(c.model)
	if err != nil {
		return err
	}
	defer m.Close()
	fmt.Fprintf(w, "%s: opset %d, written by %q, opened in %v\n", c.model, m.Opset(), m.Producer(), time.Since(start).Round(time.Microsecond))
	inputs, err := makeInputs(m, c)
	if err != nil {
		return err
	}
	for _, in := range m.Inputs() {
		fmt.Fprintf(w, "input  %v, given %v\n", in, inputs[in.Name].Shape())
	}
	start = time.Now()
	out, err := m.Run(ctx, inputs)
	if err != nil {
		return err
	}
	first := time.Since(start)
	for _, o := range m.Outputs() {
		if err := describe(ctx, w, o.Name, out[o.Name]); err != nil {
			return err
		}
	}
	var total, best time.Duration
	for i := range c.runs {
		start := time.Now()
		if _, err := m.Run(ctx, inputs); err != nil {
			return err
		}
		d := time.Since(start)
		total += d
		if i == 0 || d < best {
			best = d
		}
	}
	fmt.Fprintf(w, "first run %v", first.Round(time.Microsecond))
	if c.runs > 0 {
		fmt.Fprintf(w, ", then %v a run on average over %d, %v at best", (total / time.Duration(c.runs)).Round(time.Microsecond), c.runs, best.Round(time.Microsecond))
	}
	fmt.Fprintln(w)
	return nil
}

// makeInputs returns the model's inputs on c's device: from c's file, or
// random.
func makeInputs(m *onnx.Model, c config) (map[string]*tensor.Tensor, error) {
	var f *safetensors.File
	if c.inputs != "" {
		var err error
		if f, err = safetensors.Open(c.inputs); err != nil {
			return nil, err
		}
		defer f.Close()
	}
	r := rand.New(rand.NewPCG(c.seed, 1))
	inputs := map[string]*tensor.Tensor{}
	for i, in := range m.Inputs() {
		var x *tensor.Tensor
		if f != nil {
			st, ok := f.Tensor(in.Name)
			if !ok {
				st, ok = f.Tensor("in/" + in.Name)
			}
			if !ok {
				return nil, fmt.Errorf("%s has no tensor %q", c.inputs, in.Name)
			}
			var err error
			if x, err = fromFile(st, in.DType()); err != nil {
				return nil, fmt.Errorf("input %q: %w", in.Name, err)
			}
		} else {
			shape := make([]int, len(in.Shape))
			for d, dim := range in.Shape {
				shape[d] = dim.Size
				if dim.Size < 0 {
					shape[d] = 1
					if n, ok := c.dims[dim.Name]; ok {
						shape[d] = n
					}
				}
			}
			if in.DType().IsFloat() {
				x = tensor.Randn(c.seed+uint64(i), shape...).Cast(in.DType())
			} else {
				n := 1
				for _, d := range shape {
					n *= d
				}
				v := make([]int32, n)
				for j := range v {
					v[j] = int32(r.IntN(max(c.intMax, 1)))
				}
				x = tensor.FromSlice(v, shape...)
			}
		}
		x = x.To(c.dev)
		if err := x.Realize(context.Background()); err != nil {
			return nil, err
		}
		inputs[in.Name] = x
	}
	return inputs, nil
}

// fromFile returns tensor st of a safetensors file as a tensor of dt.
func fromFile(st safetensors.Tensor, dt tensor.DType) (*tensor.Tensor, error) {
	if dt.IsFloat() {
		v, err := st.Float32s()
		if err != nil {
			return nil, err
		}
		return tensor.FromSlice(v, st.Shape...).Cast(dt), nil
	}
	v, err := st.Int32s()
	if err != nil {
		return nil, err
	}
	return tensor.FromSlice(v, st.Shape...), nil
}

// describe prints an output's dtype, shape and first values, and for a
// float output the position of the largest value of its first row.
func describe(ctx context.Context, w io.Writer, name string, t *tensor.Tensor) error {
	shape := t.Shape()
	if !t.DType().IsFloat() {
		v, err := t.Data[int32](ctx)
		if err != nil {
			return err
		}
		fmt.Fprintf(w, "output %s: %s%v %v\n", name, t.DType(), shape, v[:min(len(v), 8)])
		return nil
	}
	v, err := t.Cast(tensor.Float32).Data[float32](ctx)
	if err != nil {
		return err
	}
	fmt.Fprintf(w, "output %s: %s%v %.4g", name, t.DType(), shape, v[:min(len(v), 8)])
	if len(shape) > 0 && shape[len(shape)-1] > 1 && len(v) > 0 {
		row := v[:shape[len(shape)-1]]
		fmt.Fprintf(w, ", first row's largest at %d", slices.Index(row, slices.Max(row)))
	}
	fmt.Fprintln(w)
	return nil
}
