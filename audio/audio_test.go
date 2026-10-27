package audio_test

import (
	"bytes"
	"context"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/gsoultan/mimir/audio"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/nn/loss"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
)

func tone(freq float64, rate, n int) []float32 {
	out := make([]float32, n)
	for i := range n {
		out[i] = float32(0.5 * math.Sin(2*math.Pi*freq*float64(i)/float64(rate)))
	}
	return out
}

func TestWAVRoundTrip(t *testing.T) {
	c := audio.Clip{Samples: tone(440, 8000, 800), Rate: 8000}
	var b bytes.Buffer
	if err := audio.WriteWAV(&b, c); err != nil {
		t.Fatal(err)
	}
	back, err := audio.ReadWAV(&b)
	if err != nil {
		t.Fatal(err)
	}
	if back.Rate != 8000 || len(back.Samples) != 800 {
		t.Fatalf("read %d samples at %d", len(back.Samples), back.Rate)
	}
	for i, s := range c.Samples {
		if math.Abs(float64(s-back.Samples[i])) > 1.0/32767 {
			t.Fatalf("sample %d: %v, wrote %v", i, back.Samples[i], s)
		}
	}
	if _, err := audio.ReadWAV(bytes.NewReader([]byte("RIFF0000AVI "))); err == nil {
		t.Error("read an AVI as a WAV")
	}
}

func TestResample(t *testing.T) {
	c := audio.Clip{Samples: tone(300, 16000, 16000), Rate: 16000}
	down := c.Resample(8000)
	if len(down.Samples) != 8000 || down.Rate != 8000 {
		t.Fatalf("%d samples at %d", len(down.Samples), down.Rate)
	}
	want := tone(300, 8000, 8000)
	for i := 100; i < 7900; i++ {
		if math.Abs(float64(down.Samples[i]-want[i])) > 0.01 {
			t.Fatalf("sample %d: %v, want %v", i, down.Samples[i], want[i])
		}
	}
	// 7 kHz can't be held at 8 kHz: it is filtered out, not folded to 1 kHz.
	high := audio.Clip{Samples: tone(7000, 32000, 32000), Rate: 32000}.Resample(8000)
	var energy float64
	for _, s := range high.Samples[200:7800] {
		energy += float64(s * s)
	}
	if rms := math.Sqrt(energy / 7600); rms > 0.02 {
		t.Errorf("a 7 kHz tone resampled to 8 kHz has rms %v", rms)
	}
	if got := len(c.Fit(100).Samples) + len(c.Fit(20000).Samples); got != 20100 {
		t.Errorf("fit lengths %d", got)
	}
}

func TestSpectrogramPeaks(t *testing.T) {
	ctx := context.Background()
	m := audio.Mel{Rate: 16000, FFT: 512, Hop: 128}
	x := tensor.FromSlice(tone(1000, 16000, 4000), 1, 4000)
	spec := audio.Spectrogram(x, m)
	sh := spec.Shape()
	if sh[1] != 257 || sh[2] != m.Frames(4000) {
		t.Fatalf("spectrogram %v, want [1 257 %d]", sh, m.Frames(4000))
	}
	// Frame 10's strongest bin is 1000 Hz: bin 1000/16000·512 = 32.
	col := spec.Shrink([][2]int{{0, 1}, {0, 257}, {10, 11}}).Reshape(257)
	top, err := col.Argmax(0).Cast(tensor.Int32).Item[int32](ctx)
	if err != nil {
		t.Fatal(err)
	}
	if top != 32 {
		t.Errorf("peak at bin %d, want 32", top)
	}
}

func TestFilterbank(t *testing.T) {
	m := audio.Mel{Rate: 16000, FFT: 400, Mels: 40}
	fb := audio.Filterbank(m)
	if len(fb) != 201*40 {
		t.Fatalf("%d filter weights", len(fb))
	}
	for k := range 40 {
		var top float32
		for b := range 201 {
			v := fb[b*40+k]
			if v < 0 || v > 1 {
				t.Fatalf("filter %d at bin %d is %v", k, b, v)
			}
			top = max(top, v)
		}
		if k > 2 && top < 0.5 {
			t.Errorf("filter %d peaks at %v", k, top)
		}
	}
}

func TestMelAndMFCCShapes(t *testing.T) {
	ctx := context.Background()
	mel := audio.NewMelSpectrogram(audio.Mel{Rate: 8000, FFT: 256, Mels: 32})
	x := tensor.Randn(1, 2, 8000)
	lm := mel.Forward(x)
	mf := audio.MFCC(lm, 13)
	if err := tensor.Realize(ctx, lm, mf); err != nil {
		t.Fatal(err)
	}
	frames := mel.Config().Frames(8000)
	if s := lm.Shape(); s[0] != 2 || s[1] != 32 || s[2] != frames {
		t.Errorf("log mel %v", s)
	}
	if s := mf.Shape(); s[1] != 13 || s[2] != frames {
		t.Errorf("mfcc %v", s)
	}
}

// TestClassifierLearns tells rising from falling chirps and noise.
func TestClassifierLearns(t *testing.T) {
	if testing.Short() {
		t.Skip("trains a sound classifier")
	}
	ctx := context.Background()
	r := rand.New(rand.NewPCG(1, 1))
	const rate, n = 8000, 4000
	clip := func(c int) []float32 {
		out := make([]float32, n)
		f0 := 300 + r.Float64()*200
		for i := range n {
			tt := float64(i) / rate
			var v float64
			switch c {
			case 0: // rising
				v = math.Sin(2 * math.Pi * (f0*tt + 1500*tt*tt))
			case 1: // falling
				v = math.Sin(2 * math.Pi * ((f0+1500)*tt - 1500*tt*tt))
			default:
				v = r.NormFloat64() * 0.5
			}
			out[i] = float32(v*0.5 + r.NormFloat64()*0.05)
		}
		return out
	}
	batch := func(k int) (*tensor.Tensor, *tensor.Tensor) {
		var x []float32
		var y []int32
		for i := range k {
			x = append(x, clip(i%3)...)
			y = append(y, int32(i%3))
		}
		return tensor.FromSlice(x, k, n), tensor.FromSlice(y, k)
	}
	m := audio.NewClassifier(3, audio.NewMelSpectrogram(audio.Mel{Rate: rate, FFT: 256, Mels: 32}), 0.25)
	opt := optim.Adam(nn.Parameters(m), optim.LR(3e-3))
	for range 40 {
		x, y := batch(12)
		l := loss.CrossEntropy(m.Forward(x), y)
		if err := l.Backward(ctx); err != nil {
			t.Fatal(err)
		}
		if err := opt.Step(ctx); err != nil {
			t.Fatal(err)
		}
		opt.ZeroGrad()
	}
	nn.Eval(m)
	x, y := batch(30)
	pred, err := m.Forward(x).Argmax(1).Cast(tensor.Int32).Data[int32](ctx)
	if err != nil {
		t.Fatal(err)
	}
	want, _ := y.Data[int32](ctx)
	right := 0
	for i := range pred {
		if pred[i] == want[i] {
			right++
		}
	}
	if right < 27 {
		t.Errorf("%d of 30 sounds right after training", right)
	}
	m.SetHead(5)
	if out := m.Forward(x); out.Shape()[1] != 5 {
		t.Errorf("new head gives %v", out.Shape())
	}
}
