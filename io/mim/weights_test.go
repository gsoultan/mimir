package mim_test

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"maps"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"testing"
	"time"

	mimfile "github.com/gsoultan/mim"

	"github.com/gsoultan/mimir/io/mim"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
)

// values returns x's elements, through float32 for a half tensor.
func values(t *testing.T, x *tensor.Tensor) []float32 {
	t.Helper()
	v, err := x.Cast(tensor.Float32).Data[float32](t.Context())
	if err != nil {
		t.Fatal(err)
	}
	return v
}

// save writes m to a file in a fresh directory and returns its path.
func save(t *testing.T, m nn.Module, opts *mim.SaveOptions) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "model.mim")
	out, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := mim.Save(t.Context(), out, m, opts); err != nil {
		t.Fatal(err)
	}
	if err := out.Close(); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestSaveLoadRoundTrips(t *testing.T) {
	nn.ManualSeed(1)
	a := nn.NewSequential(nn.NewConv2d(1, 2, 3), nn.NewBatchNorm2d(2), nn.Flatten(), nn.NewLinear(8, 3))
	nn.ManualSeed(2)
	b := nn.NewSequential(nn.NewConv2d(1, 2, 3), nn.NewBatchNorm2d(2), nn.Flatten(), nn.NewLinear(8, 3))
	x := tensor.Randn(3, 2, 1, 4, 4)
	nn.Eval(a)
	nn.Eval(b)
	if slices.Equal(values(t, a.Forward(x)), values(t, b.Forward(x))) {
		t.Fatal("different seeds built identical models")
	}
	var buf bytes.Buffer
	md := mimfile.Metadata{"note": "two seeds"}
	if err := mim.Save(t.Context(), &buf, a, &mim.SaveOptions{Metadata: md}); err != nil {
		t.Fatal(err)
	}
	f, err := mim.Parse(buf.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := f.Load(b); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(values(t, a.Forward(x)), values(t, b.Forward(x))) {
		t.Error("loaded model computes differently")
	}
	var names []string
	for name := range nn.NamedParams(a) {
		names = append(names, name)
	}
	slices.Sort(names)
	if got := f.Names(); !slices.Equal(got, names) {
		t.Errorf("tensors %v, want the parameters %v", got, names)
	}
	if got, err := f.Metadata(); err != nil || got["note"] != "two seeds" {
		t.Errorf("metadata %v, %v", got, err)
	}
	if len(f.Inputs()) != 0 {
		t.Errorf("a file of weights has inputs %v", f.Inputs())
	}
	if _, err := f.Run(t.Context(), nil); !errors.Is(err, mim.ErrNoGraph) {
		t.Errorf("run of a file of weights: %v, want ErrNoGraph", err)
	}
	if err := f.Load(nn.NewLinear(2, 2)); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("loading into another architecture: %v, want ErrShape", err)
	}
}

func TestSaveWritesTheParametersAsTheyAre(t *testing.T) {
	for dt, fdt := range map[tensor.DType]mimfile.DType{tensor.Float32: mimfile.F32, tensor.BFloat16: mimfile.BF16, tensor.Float16: mimfile.F16} {
		nn.ManualSeed(3)
		a := nn.NewLinear(4, 2)
		if err := nn.Cast(a, dt); err != nil {
			t.Fatal(err)
		}
		var buf bytes.Buffer
		if err := mim.Save(t.Context(), &buf, a, nil); err != nil {
			t.Fatal(err)
		}
		f, err := mim.Parse(buf.Bytes())
		if err != nil {
			t.Fatal(err)
		}
		file, err := mimfile.Parse(buf.Bytes())
		if err != nil {
			t.Fatal(err)
		}
		for name, p := range nn.NamedParams(a) {
			info, _ := file.Tensor(name)
			want, err := p.Value().Bytes(t.Context())
			if err != nil {
				t.Fatal(err)
			}
			if info.DType != fdt || !bytes.Equal(file.TensorData(info), want) {
				t.Errorf("%s saved as %s%v, want the bytes of a %s tensor", name, info.DType, info.Shape, dt)
			}
		}
		// A model of the file's dtype takes the values as they are.
		nn.ManualSeed(4)
		b := nn.NewLinear(4, 2)
		if err := nn.Cast(b, dt); err != nil {
			t.Fatal(err)
		}
		if err := f.Load(b); err != nil {
			t.Fatal(err)
		}
		x := tensor.Randn(1, 3, 4)
		if b.Weight.Value().DType() != dt || !slices.Equal(values(t, a.Forward(x)), values(t, b.Forward(x))) {
			t.Errorf("%s model loaded as %s computes differently", dt, b.Weight.Value().DType())
		}
		// A float32 model keeps its dtype and takes the values widened.
		c := nn.NewLinear(4, 2)
		if err := f.Load(c); err != nil {
			t.Fatal(err)
		}
		if c.Weight.Value().DType() != tensor.Float32 || !slices.Equal(values(t, c.Weight.Value()), values(t, a.Weight.Value())) {
			t.Errorf("%s model loaded into a float32 one: dtype %s", dt, c.Weight.Value().DType())
		}
		f.Close()
	}
}

func TestWriteTensorsKeepsTrainingState(t *testing.T) {
	w := tensor.FromSlice([]float32{1, 2, 3}, 3)
	m := tensor.FromSlice([]float32{0.5, 0.25, 0}, 3)
	step := tensor.FromSlice([]int32{7})
	tr := &mimfile.Training{
		Optimizer: "adam",
		Counters:  []mimfile.TrainCounter{{Name: "step", Value: 7}},
		Params:    []mimfile.TrainParam{{Tensor: "w", States: []mimfile.TrainState{{Slot: "m", Tensor: "w.m"}}}},
	}
	var buf bytes.Buffer
	if err := mim.WriteTensors(t.Context(), &buf, map[string]*tensor.Tensor{"w": w, "w.m": m, "step": step}, &mim.SaveOptions{Training: tr}); err != nil {
		t.Fatal(err)
	}
	f, err := mim.Parse(buf.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	got, err := f.Training()
	if err != nil || got.Optimizer != "adam" || got.Counters[0].Value != 7 || got.Params[0].States[0].Tensor != "w.m" {
		t.Fatalf("training state %+v, %v", got, err)
	}
	for name, want := range map[string]*tensor.Tensor{"w.m": m, "step": step} {
		v, err := f.Tensor(name)
		if err != nil || v.DType() != want.DType() || !slices.Equal(values(t, v), values(t, want)) {
			t.Errorf("tensor %s: %v, %v", name, v, err)
		}
	}
	if _, err := f.Tensor("w.v"); !errors.Is(err, tensor.ErrShape) {
		t.Errorf("missing tensor: %v, want ErrShape", err)
	}
	f.Close()
	if _, err := f.Tensor("w"); err == nil {
		t.Error("tensor of a closed model")
	}
}

func TestLoadFromAnOpenedFileReadsItInPlace(t *testing.T) {
	ctx := t.Context()
	nn.ManualSeed(3)
	a := nn.NewSequential(nn.NewLinear(4, 3), nn.NewBatchNorm1d(3))
	nn.ManualSeed(4)
	b := nn.NewSequential(nn.NewLinear(4, 3), nn.NewBatchNorm1d(3))
	nn.Eval(a)
	nn.Eval(b)
	path := save(t, a, nil)
	saved, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	f, err := mim.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := f.Load(b); err != nil {
		t.Fatal(err)
	}
	x := tensor.Randn(5, 6, 4)
	if want, got := values(t, a.Forward(x)), values(t, b.Forward(x)); !slices.Equal(got, want) {
		t.Errorf("loaded model computes %v, want %v", got, want)
	}
	bias := b.Layers[0].(*nn.Linear).Bias
	if !bias.Value().Device().Equal(tensor.DefaultDevice()) {
		t.Errorf("parameter loaded onto %v, want the default device %v", bias.Value().Device(), tensor.DefaultDevice())
	}
	if mapped, known := mapsFile(t, path); known && tensor.DefaultDevice().Equal(tensor.CPU()) && !mapped {
		t.Error("the parameters do not read the mapping of the file")
	}

	// A training step writes the parameters in place, but never the file.
	nn.Train(b)
	if err := b.Forward(x).Sum().Backward(ctx); err != nil {
		t.Fatal(err)
	}
	if err := optim.SGD(nn.Parameters(b), optim.LR(0.1)).Step(ctx); err != nil {
		t.Fatal(err)
	}
	if got := values(t, bias.Value()); slices.Equal(got, values(t, a.Layers[0].(*nn.Linear).Bias.Value())) {
		t.Error("the training step changed no bias")
	}
	if now, err := os.ReadFile(path); err != nil || !bytes.Equal(now, saved) {
		t.Errorf("the file changed after a training step (%v)", err)
	}
}

// mapsFile reports whether the process maps the file at path, and whether
// that can be seen at all.
func mapsFile(t *testing.T, path string) (mapped, known bool) {
	t.Helper()
	if runtime.GOOS != "linux" {
		return false, false
	}
	maps, err := os.ReadFile("/proc/self/maps")
	if err != nil {
		t.Fatal(err)
	}
	return bytes.Contains(maps, []byte(path)), true
}

// TestLoadedModelOutlivesTheFile checks that the parameters of a model
// loaded from an opened file hold the mapping: the file may be closed once
// Load returns, and is unmapped once the model is garbage.
func TestLoadedModelOutlivesTheFile(t *testing.T) {
	nn.ManualSeed(5)
	a := nn.NewLinear(64, 32)
	path := save(t, a, nil)
	x := tensor.Randn(6, 8, 64)
	want := values(t, a.Forward(x))

	load := func() *nn.Linear {
		f, err := mim.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		b := nn.NewLinear(64, 32)
		if err := f.Load(b); err != nil {
			t.Fatal(err)
		}
		if err := f.Close(); err != nil {
			t.Fatal(err)
		}
		return b
	}
	b := load()
	runtime.GC()
	runtime.GC()
	if got := values(t, b.Forward(x)); !slices.Equal(got, want) {
		t.Errorf("model loaded from a closed file computes %v, want %v", got, want)
	}
	inPlace := tensor.DefaultDevice().Equal(tensor.CPU())
	if mapped, known := mapsFile(t, path); known && inPlace && !mapped {
		t.Error("the file was unmapped while the model reads it")
	}
	runtime.KeepAlive(b)
	b = nil
	deadline := time.Now().Add(10 * time.Second)
	for {
		runtime.GC()
		mapped, known := mapsFile(t, path)
		if !known || !mapped {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("the file is still mapped after the model is garbage")
		}
		time.Sleep(time.Millisecond)
	}
}

func TestSaveRejectsWhatItCannotWrite(t *testing.T) {
	var buf bytes.Buffer
	nan := tensor.FromSlice([]float32{float32(math.NaN())})
	if err := mim.WriteTensors(t.Context(), &buf, map[string]*tensor.Tensor{"": nan}, nil); err == nil {
		t.Error("a tensor without a name saved")
	}
	failed := tensor.FromSlice([]float32{1, 2}, 2).Reshape(3)
	if err := mim.WriteTensors(t.Context(), &buf, map[string]*tensor.Tensor{"x": failed}, nil); err == nil {
		t.Error("a failed tensor saved")
	}
}

// benchTensors returns n Float32 tensors of size elements each, computed.
func benchTensors(b *testing.B, n, size int) map[string]*tensor.Tensor {
	b.Helper()
	ts := map[string]*tensor.Tensor{}
	for i := range n {
		ts[fmt.Sprintf("layers.%d.weight", i)] = tensor.Randn(uint64(i), size)
	}
	all := slices.Collect(maps.Values(ts))
	if err := tensor.Realize(b.Context(), all...); err != nil {
		b.Fatal(err)
	}
	return ts
}

var benchShapes = []struct {
	name    string
	n, size int
}{{"4x16MiB", 4, 4 << 20}, {"20000x1KiB", 20000, 256}}

func BenchmarkWriteTensors(b *testing.B) {
	for _, s := range benchShapes {
		b.Run(s.name, func(b *testing.B) {
			ts := benchTensors(b, s.n, s.size)
			b.SetBytes(int64(4 * s.n * s.size))
			for b.Loop() {
				if err := mim.WriteTensors(b.Context(), io.Discard, ts, nil); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkOpenAndRead(b *testing.B) {
	for _, s := range benchShapes {
		b.Run(s.name, func(b *testing.B) {
			path := filepath.Join(b.TempDir(), "model.mim")
			out, err := os.Create(path)
			if err != nil {
				b.Fatal(err)
			}
			if err := mim.WriteTensors(b.Context(), out, benchTensors(b, s.n, s.size), nil); err != nil {
				b.Fatal(err)
			}
			out.Close()
			b.SetBytes(int64(4 * s.n * s.size))
			for b.Loop() {
				f, err := mim.Open(path)
				if err != nil {
					b.Fatal(err)
				}
				var sums []*tensor.Tensor
				for _, name := range f.Names() {
					t, err := f.Tensor(name)
					if err != nil {
						b.Fatal(err)
					}
					sums = append(sums, t.Sum())
				}
				f.Close()
				if err := tensor.Realize(b.Context(), sums...); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
