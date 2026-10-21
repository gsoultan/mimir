package mim

import (
	"bytes"
	"context"
	"math"
	"math/rand/v2"
	"path/filepath"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/internal/sched"
	"github.com/gsoultan/mimir/tensor"
)

// testInputs builds inputs for m's run: every symbolic dimension of a
// size, floats drawn at random and small integers.
func testInputs(t testing.TB, m *Model, size int, seed uint64) map[string]*tensor.Tensor {
	t.Helper()
	rng := rand.New(rand.NewPCG(seed, 7))
	in := map[string]*tensor.Tensor{}
	for _, info := range m.Inputs() {
		shape := make([]int, len(info.Shape))
		n := 1
		for i, d := range info.Shape {
			shape[i] = d.Size
			if d.Size < 0 {
				shape[i] = size
				if d.Bound >= 0 && d.Bound < size {
					shape[i] = d.Bound
				}
			}
			n *= shape[i]
		}
		switch info.DType() {
		case tensor.Float32:
			v := make([]float32, n)
			for i := range v {
				v[i] = rng.Float32()*2 - 1
			}
			in[info.Name] = tensor.FromSlice(v, shape...)
		default:
			v := make([]int32, n)
			for i := range v {
				v[i] = int32(i % 4)
			}
			in[info.Name] = tensor.FromSlice(v, shape...)
		}
	}
	return in
}

// bits returns the outputs of a run as bytes, in the order of the
// model's outputs, for a comparison bit for bit.
func bits(t testing.TB, m *Model, out map[string]*tensor.Tensor) [][]byte {
	t.Helper()
	ctx := context.Background()
	var all [][]byte
	for _, info := range m.Outputs() {
		o := out[info.Name]
		var b []byte
		switch o.DType() {
		case tensor.Float32:
			v, err := o.Data[float32](ctx)
			if err != nil {
				t.Fatal(err)
			}
			b = make([]byte, 4*len(v))
			for i, x := range v {
				u := math.Float32bits(x)
				b[4*i], b[4*i+1], b[4*i+2], b[4*i+3] = byte(u), byte(u>>8), byte(u>>16), byte(u>>24)
			}
		default:
			v, err := o.Data[int32](ctx)
			if err != nil {
				t.Fatal(err)
			}
			b = make([]byte, 4*len(v))
			for i, x := range v {
				b[4*i], b[4*i+1], b[4*i+2], b[4*i+3] = byte(x), byte(x>>8), byte(x>>16), byte(x>>24)
			}
		}
		all = append(all, b)
	}
	return all
}

// The second run of a model with inputs of the shapes of the first
// replays the first's program, compiling nothing, and computes bit for
// bit what building the graph again would, variables included.
func TestRunsReplayTheFirst(t *testing.T) {
	files, err := filepath.Glob(filepath.Join("testdata", "*", "*.mim"))
	if err != nil || len(files) == 0 {
		t.Fatalf("testdata models: %v, %v", files, err)
	}
	ctx := context.Background()
	for _, file := range files {
		t.Run(filepath.Base(file), func(t *testing.T) {
			want := func() [][][]byte {
				replaying = false
				defer func() { replaying = true }()
				m, err := Open(file)
				if err != nil {
					t.Fatal(err)
				}
				defer m.Close()
				s := m.NewSession()
				var runs [][][]byte
				for i := range 3 {
					out, err := s.Run(ctx, testInputs(t, m, 4, uint64(i)))
					if err != nil {
						t.Fatal(err)
					}
					runs = append(runs, bits(t, m, out))
				}
				return runs
			}()
			m, err := Open(file)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			s := m.NewSession()
			var compiled int
			var replays int64
			for i := range 3 {
				if i == 2 {
					compiled, replays = sched.Compiled(), replayed.Load()
				}
				out, err := s.Run(ctx, testInputs(t, m, 4, uint64(i)))
				if err != nil {
					t.Fatal(err)
				}
				got := bits(t, m, out)
				for j := range got {
					if !bytes.Equal(got[j], want[i][j]) {
						t.Fatalf("run %d: output %d of the replay differs from the run it stands for", i, j)
					}
				}
			}
			m.mu.Lock()
			var recorded, known int
			for _, recs := range m.recorded {
				for _, r := range recs {
					known++
					if r != nil {
						recorded++
					}
				}
			}
			m.mu.Unlock()
			if known == 0 {
				t.Fatal("no run was recorded or found unrecordable")
			}
			if (recorded > 0) == m.dataDependent {
				t.Fatalf("%d runs recorded, while the model's graph is data-dependent: %v", recorded, m.dataDependent)
			}
			// The first run reads the variables as zeros, laid out as a
			// fill; the second reads what the first left and is recorded
			// too; the third replays it.
			if replayed.Load() != replays+1 && !m.dataDependent {
				t.Error("the third run did not replay")
			}
			if sched.Compiled() != compiled {
				t.Error("the third run compiled a program")
			}
		})
	}
}

// A run with inputs of other shapes is recorded on its own, and a run
// whose inputs do not fit a recording, as transposed ones, builds its
// graph.
func TestRecordingsPerBinding(t *testing.T) {
	ctx := context.Background()
	m, err := Open(filepath.Join("testdata", "onnx", "tiny-cnn.mim"))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	name := m.Inputs()[0].Name
	for _, n := range []int{1, 3, 1} {
		if _, err := m.Run(ctx, testInputs(t, m, n, 1)); err != nil {
			t.Fatal(err)
		}
	}
	if len(m.recorded) != 2 {
		t.Fatalf("%d bindings recorded after runs of two", len(m.recorded))
	}
	x := testInputs(t, m, 3, 2)[name]
	want, err := m.Run(ctx, map[string]*tensor.Tensor{name: x})
	if err != nil {
		t.Fatal(err)
	}
	// The same values, laid out as a transposed tensor's are: the run
	// builds its graph and computes the same outputs.
	shape := x.Shape()
	odd := x.Transpose(0, 1).Clone().Transpose(0, 1)
	if !slices.Equal(odd.Shape(), shape) {
		t.Fatal(odd.Shape())
	}
	got, err := m.Run(ctx, map[string]*tensor.Tensor{name: odd})
	if err != nil {
		t.Fatal(err)
	}
	for j, b := range bits(t, m, got) {
		if !bytes.Equal(b, bits(t, m, want)[j]) {
			t.Fatalf("output %d over a transposed input differs", j)
		}
	}
}
