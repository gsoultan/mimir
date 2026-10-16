package nn_test

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"testing"
	"time"

	"github.com/gsoultan/mimir/io/safetensors"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
)

func TestLoadFromMappedFileReadsItInPlace(t *testing.T) {
	ctx := t.Context()
	nn.ManualSeed(3)
	a := nn.NewSequential(nn.NewLinear(4, 3), nn.NewBatchNorm1d(3))
	nn.ManualSeed(4)
	b := nn.NewSequential(nn.NewLinear(4, 3), nn.NewBatchNorm1d(3))
	nn.Eval(a)
	nn.Eval(b)
	path := filepath.Join(t.TempDir(), "model.safetensors")
	out, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := nn.Save(ctx, out, a); err != nil {
		t.Fatal(err)
	}
	if err := out.Close(); err != nil {
		t.Fatal(err)
	}
	saved, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	f, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if !f.Mapped() && (runtime.GOOS == "linux" || runtime.GOOS == "darwin") {
		t.Errorf("the file is not mapped on %s", runtime.GOOS)
	}
	if err := nn.Load(f, b); err != nil {
		t.Fatal(err)
	}
	x := tensor.Randn(5, 6, 4)
	if want, got := data(t, a.Forward(x)), data(t, b.Forward(x)); !slices.Equal(got, want) {
		t.Errorf("loaded model computes %v, want %v", got, want)
	}
	bias := b.Layers[0].(*nn.Linear).Bias
	if !bias.Value().Device().Equal(tensor.DefaultDevice()) {
		t.Errorf("parameter loaded onto %v, want the default device %v", bias.Value().Device(), tensor.DefaultDevice())
	}

	// On the CPU the parameters read the mapping: a change to its bytes
	// shows in the model.
	if f.Mapped() && tensor.DefaultDevice().Equal(tensor.CPU()) {
		st, _ := f.Tensor("Layers.0.Bias")
		binary.LittleEndian.PutUint32(st.Data, math.Float32bits(100))
		if got := data(t, bias.Value()); got[0] != 100 {
			t.Errorf("bias after writing its bytes = %v, want 100 first", got)
		}
	}

	// A training step writes the parameters in place, but never the file.
	nn.Train(b)
	if err := b.Forward(x).Sum().Backward(ctx); err != nil {
		t.Fatal(err)
	}
	if err := optim.SGD(nn.Parameters(b), optim.LR(0.1)).Step(ctx); err != nil {
		t.Fatal(err)
	}
	if err := b.Forward(x).Err(); err != nil {
		t.Fatal(err)
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
// loaded from a mapped file hold the mapping: the file may be closed once
// Load returns, and is unmapped once the model is garbage.
func TestLoadedModelOutlivesTheFile(t *testing.T) {
	ctx := t.Context()
	nn.ManualSeed(5)
	a := nn.NewLinear(64, 32)
	path := filepath.Join(t.TempDir(), "model.safetensors")
	out, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := nn.Save(ctx, out, a); err != nil {
		t.Fatal(err)
	}
	if err := out.Close(); err != nil {
		t.Fatal(err)
	}
	x := tensor.Randn(6, 8, 64)
	want := data(t, a.Forward(x))

	load := func() *nn.Linear {
		f, err := safetensors.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		b := nn.NewLinear(64, 32)
		if err := nn.Load(f, b); err != nil {
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
	if got := data(t, b.Forward(x)); !slices.Equal(got, want) {
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
