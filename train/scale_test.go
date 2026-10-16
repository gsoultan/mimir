package train_test

import (
	"fmt"
	"os"
	"runtime"
	"runtime/debug"
	"runtime/metrics"
	"strconv"
	"testing"
	"time"

	"github.com/gsoultan/mimir/backend/cpu"
	"github.com/gsoultan/mimir/data"
	"github.com/gsoultan/mimir/data/datasets"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/nn/loss"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
	"github.com/gsoultan/mimir/train"
)

// scaleRows is the number of rows TestFlatMemoryOverManyRows trains
// over unless MIMIR_SCALE_ROWS says otherwise.
const scaleRows = 1_000_000

// memoryMargin is how much the memory held may grow over the epoch.
const memoryMargin = 64 << 20

// TestFlatMemoryOverManyRows trains a small model over a synthetic stream
// of rows and asserts that memory stays flat however many rows pass: the
// memory held after the last tenth of the epoch is at most memoryMargin
// above the memory held after the first tenth. MIMIR_SCALE_ROWS sets the
// number of rows; the default of a million keeps the test short, and
//
//	MIMIR_SCALE_ROWS=1000000000 go test -run FlatMemory -timeout 3h -v ./train
//
// is the billion-row run that the scale job of CI performs.
func TestFlatMemoryOverManyRows(t *testing.T) {
	rows := int64(scaleRows)
	if s := os.Getenv("MIMIR_SCALE_ROWS"); s != "" {
		v, err := strconv.ParseInt(s, 10, 64)
		if err != nil || v < 1 {
			t.Fatalf("MIMIR_SCALE_ROWS=%q, want a positive number of rows", s)
		}
		rows = v
	}
	const features, batch, shards = 32, 8192, 256
	steps := int(rows / batch)
	if steps < 10 {
		t.Fatalf("%d rows make %d batches of %d, want at least 10", rows, steps, batch)
	}
	ds, err := data.NewStream(datasets.Synthetic(rows, features, shards, 1),
		data.BatchSize(batch), data.Shuffle(1), data.DropLast(), data.ClassTargets())
	if err != nil {
		t.Fatal(err)
	}
	nn.ManualSeed(1)
	m := nn.NewSequential(nn.NewLinear(features, 64), nn.ReLU(), nn.NewLinear(64, 2))
	opt := optim.Adam(nn.Parameters(m), optim.LR(1e-3))

	// Memory is sampled a tenth of the way in, once the pools are warm,
	// and then three more times up to the last batch.
	sampleAt := map[int]bool{steps / 10: true, steps * 4 / 10: true, steps * 7 / 10: true, steps - 1: true}
	var samples []memorySample
	var epoch train.Epoch
	began := time.Now()
	err = train.Fit(t.Context(), m, opt, ds,
		train.Loss(loss.CrossEntropy),
		train.OnBatch(func(b train.Batch) error {
			if sampleAt[b.Index] {
				samples = append(samples, measureMemory(b.Index))
			}
			return nil
		}),
		train.OnEpoch(func(e train.Epoch) error { epoch = e; return nil }))
	if err != nil {
		t.Fatal(err)
	}
	elapsed := time.Since(began)
	t.Logf("%d rows in %d steps of %d took %s: %.0f rows/s, mean loss %.3f",
		rows, steps, batch, elapsed.Round(time.Millisecond), float64(steps*batch)/elapsed.Seconds(), epoch.Loss)
	for _, s := range samples {
		t.Log(s)
	}
	if epoch.Steps != steps {
		t.Errorf("took %d steps, want %d", epoch.Steps, steps)
	}
	if epoch.Loss > 0.5 {
		t.Errorf("mean loss %.3f, want below 0.5 on separable data", epoch.Loss)
	}
	base := samples[0]
	for _, s := range samples[1:] {
		if s.held > base.held+memoryMargin {
			t.Errorf("Go memory held grew from %s at step %d to %s at step %d, want flat within %s",
				mib(base.held), base.step, mib(s.held), s.step, mib(memoryMargin))
		}
		if tensor.DefaultDevice().Equal(tensor.CPU()) && s.resident > base.resident+memoryMargin {
			t.Errorf("resident memory grew from %s at step %d to %s at step %d, want flat within %s",
				mib(base.resident), base.step, mib(s.resident), s.step, mib(memoryMargin))
		}
	}
}

// memorySample is the memory held at one step.
type memorySample struct {
	step     int
	held     uint64 // bytes the Go runtime holds from the operating system
	resident uint64 // the process's resident set, or 0 where it is not read
}

func (s memorySample) String() string {
	return fmt.Sprintf("step %d: Go holds %s, resident %s", s.step, mib(s.held), mib(s.resident))
}

func mib(bytes uint64) string { return fmt.Sprintf("%.1f MiB", float64(bytes)/(1<<20)) }

// measureMemory collects garbage, returns what memory it can to the
// operating system and samples what remains held.
func measureMemory(step int) memorySample {
	runtime.GC()
	settleOffHeap()
	debug.FreeOSMemory()
	ms := []metrics.Sample{{Name: "/memory/classes/total:bytes"}, {Name: "/memory/classes/heap/released:bytes"}}
	metrics.Read(ms)
	return memorySample{step: step, held: ms[0].Value.Uint64() - ms[1].Value.Uint64(), resident: residentBytes()}
}

// settleOffHeap waits for the cleanups of dropped off-heap buffers, which
// run on a goroutine of their own after a collection, to unmap them: it
// returns once the bytes held off the heap have not fallen for a few
// samples, or after a bound.
func settleOffHeap() {
	last, stable := cpu.OffHeapBytes(), 0
	for range 200 {
		time.Sleep(5 * time.Millisecond)
		now := cpu.OffHeapBytes()
		if now < last {
			stable = 0
		} else if stable++; stable == 3 {
			return
		}
		last = now
	}
}
