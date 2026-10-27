package audio

import (
	"fmt"
	"math"

	"github.com/gsoultan/mimir/tensor"
)

// Mel configures a mel spectrogram. Zero fields take the defaults.
type Mel struct {
	Rate int // samples a second; the default is 16,000
	FFT  int // samples in each frame's Fourier transform; the default is 400, 25 ms at 16 kHz
	Hop  int // samples between frames; the default is FFT/4, 10 ms at 16 kHz
	Mels int // mel bands; the default is 64
	// FMin and FMax bound the bands, in hertz; the defaults are 0 and
	// Rate/2.
	FMin, FMax float64
}

func (m Mel) resolve() Mel {
	if m.Rate <= 0 {
		m.Rate = 16000
	}
	if m.FFT <= 0 {
		m.FFT = 400
	}
	if m.Hop <= 0 {
		m.Hop = max(m.FFT/4, 1)
	}
	if m.Mels <= 0 {
		m.Mels = 64
	}
	if m.FMax <= 0 {
		m.FMax = float64(m.Rate) / 2
	}
	return m
}

// Frames returns how many frames a clip of n samples has: one every Hop
// samples, the clip padded by FFT/2 at each end, as torchaudio's
// center=True.
func (m Mel) Frames(n int) int { return 1 + n/m.resolve().Hop }

// hann returns the periodic Hann window of n samples, torch.hann_window's.
func hann(n int) []float32 {
	w := make([]float32, n)
	for i := range n {
		w[i] = float32(0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(n)))
	}
	return w
}

// reflect pads the last axis of x, [N, T], by p samples at each end,
// mirrored without repeating the edge, as torch's reflect padding.
func reflect(x *tensor.Tensor, p int) *tensor.Tensor {
	n, t := x.Shape()[0], x.Shape()[1]
	if p == 0 {
		return x
	}
	if t <= p {
		return x.Pad([][2]int{{0, 0}, {p, p}}, 0)
	}
	left := x.Shrink([][2]int{{0, n}, {1, p + 1}}).Flip(1)
	right := x.Shrink([][2]int{{0, n}, {t - p - 1, t - 1}}).Flip(1)
	return tensor.Cat(1, left, x, right)
}

// Spectrogram returns the power spectrogram of waveforms x, [N, T]: for
// each frame, a Hann-windowed Fourier transform's squared magnitudes,
// [N, FFT/2+1, frames], like torchaudio.transforms.Spectrogram with
// n_fft FFT, hop_length Hop and power 2.
func Spectrogram(x *tensor.Tensor, m Mel) *tensor.Tensor {
	if err := x.Err(); err != nil {
		return x
	}
	m = m.resolve()
	if x.Rank() != 2 {
		return tensor.Failed(fmt.Errorf("audio: spectrogram of %v, want [batch, samples]: %w", x.Shape(), tensor.ErrShape))
	}
	if x.DType() != tensor.Float32 {
		x = x.Cast(tensor.Float32)
	}
	frames := reflect(x, m.FFT/2).Unfold(1, m.FFT, m.Hop) // [N, frames, FFT]
	frames = frames.Mul(tensor.FromSlice(hann(m.FFT), m.FFT))
	spec := frames.RFFT(2)         // [N, frames, bins, 2]
	power := spec.Mul(spec).Sum(3) // [N, frames, bins]
	return power.Transpose(1, 2)   // [N, bins, frames]
}

// hz and mel convert between hertz and the HTK mel scale.
func mel(f float64) float64 { return 2595 * math.Log10(1+f/700) }
func hz(m float64) float64  { return 700 * (math.Pow(10, m/2595) - 1) }

// Filterbank returns the triangular mel filters, [FFT/2+1, Mels], as
// torchaudio.functional.melscale_fbanks with the HTK scale and no
// normalization.
func Filterbank(m Mel) []float32 {
	m = m.resolve()
	bins := m.FFT/2 + 1
	lo, hi := mel(m.FMin), mel(m.FMax)
	pts := make([]float64, m.Mels+2)
	for i := range pts {
		pts[i] = hz(lo + (hi-lo)*float64(i)/float64(m.Mels+1))
	}
	out := make([]float32, bins*m.Mels)
	for b := range bins {
		f := float64(m.Rate/2) * float64(b) / float64(bins-1)
		for k := range m.Mels {
			down := (f - pts[k]) / (pts[k+1] - pts[k])
			up := (pts[k+2] - f) / (pts[k+2] - pts[k+1])
			out[b*m.Mels+k] = float32(max(0, min(down, up)))
		}
	}
	return out
}

// MelSpectrogram is the log mel spectrogram of waveforms as an
// nn.Module: Forward maps [N, samples] to [N, Mels, frames], the natural
// log of each band's power plus 1e-6, like torchaudio's MelSpectrogram
// followed by a log. Its window and filters are buffers.
type MelSpectrogram struct {
	Filters *tensor.Param // Float32 [FFT/2+1, Mels]
	cfg     Mel
}

// NewMelSpectrogram returns the log mel spectrogram of m.
func NewMelSpectrogram(m Mel) *MelSpectrogram {
	m = m.resolve()
	return &MelSpectrogram{Filters: tensor.NewBuffer(tensor.FromSlice(Filterbank(m), m.FFT/2+1, m.Mels)), cfg: m}
}

// Config returns the spectrogram's settings, the defaults filled in.
func (s *MelSpectrogram) Config() Mel { return s.cfg }

// Forward maps [N, samples] to [N, Mels, frames].
func (s *MelSpectrogram) Forward(x *tensor.Tensor) *tensor.Tensor {
	p := Spectrogram(x, s.cfg)
	if err := p.Err(); err != nil {
		return p
	}
	// [N, bins, frames] → [N, frames, bins] · [bins, mels] → [N, mels, frames]
	melp := p.Transpose(1, 2).MatMul(s.Filters.Value()).Transpose(1, 2)
	return melp.Add(tensor.Scalar[float32](1e-6)).Log()
}

// MFCC returns the first n mel-frequency cepstral coefficients of the
// log mel spectrogram logMel, [N, Mels, frames], as [N, n, frames]: its
// orthonormal DCT-II along the bands, as torchaudio's MFCC takes of its
// decibels.
func MFCC(logMel *tensor.Tensor, n int) *tensor.Tensor {
	if err := logMel.Err(); err != nil {
		return logMel
	}
	if logMel.Rank() != 3 || n < 1 || n > logMel.Shape()[1] {
		return tensor.Failed(fmt.Errorf("audio: %d coefficients of %v, want [N, mels, frames] and at most mels: %w", n, logMel.Shape(), tensor.ErrShape))
	}
	mels := logMel.Shape()[1]
	dct := make([]float32, n*mels)
	for k := range n {
		scale := math.Sqrt(2 / float64(mels))
		if k == 0 {
			scale = math.Sqrt(1 / float64(mels))
		}
		for j := range mels {
			dct[k*mels+j] = float32(scale * math.Cos(math.Pi/float64(mels)*(float64(j)+0.5)*float64(k)))
		}
	}
	return tensor.FromSlice(dct, n, mels).MatMul(logMel)
}
