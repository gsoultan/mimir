// Package audio turns sound into what networks read, as torchaudio does:
// WAV files in and out, spectrograms, mel spectrograms and MFCCs, and a
// network that classifies sounds from their mel spectrogram, for spoken
// commands, bird calls or machine noises.
//
//	clip, err := audio.ReadWAV(f)            // samples in [-1, 1], one channel
//	x := tensor.FromSlice(clip.Resample(16000).Samples, 1, n)
//	mel := audio.NewMelSpectrogram(audio.Mel{Rate: 16000})
//	m := audio.NewClassifier(10, mel)        // ten kinds of sound
//	logits := m.Forward(x)                   // x is [batch, samples]
//
// The features are computed with tensor operations, the Fourier
// transform among them, so they run on every device and inside a model's
// Forward, and a saved classifier carries its features with it.
package audio

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
)

// Clip is sound: Samples between -1 and 1, Rate of them a second, one
// channel.
type Clip struct {
	Samples []float32
	Rate    int
}

// Seconds returns the clip's length.
func (c Clip) Seconds() float64 { return float64(len(c.Samples)) / float64(max(c.Rate, 1)) }

// ErrWAV means a file is not a WAV file ReadWAV reads.
var ErrWAV = errors.New("audio: not a PCM or float WAV file")

// ReadWAV reads a WAV file of 8, 16, 24 or 32-bit integer or 32 or 64-bit
// float samples, averaging its channels into one.
func ReadWAV(r io.Reader) (Clip, error) {
	b, err := io.ReadAll(io.LimitReader(r, 1<<31))
	if err != nil {
		return Clip{}, err
	}
	if len(b) < 12 || string(b[:4]) != "RIFF" || string(b[8:12]) != "WAVE" {
		return Clip{}, ErrWAV
	}
	var format, channels, bits int
	var rate int
	var data []byte
	for p := 12; p+8 <= len(b); {
		id, size := string(b[p:p+4]), int(binary.LittleEndian.Uint32(b[p+4:p+8]))
		body := b[p+8 : min(p+8+size, len(b))]
		switch id {
		case "fmt ":
			if len(body) < 16 {
				return Clip{}, ErrWAV
			}
			format = int(binary.LittleEndian.Uint16(body[0:2]))
			channels = int(binary.LittleEndian.Uint16(body[2:4]))
			rate = int(binary.LittleEndian.Uint32(body[4:8]))
			bits = int(binary.LittleEndian.Uint16(body[14:16]))
			if format == 0xFFFE && len(body) >= 26 { // WAVE_FORMAT_EXTENSIBLE: the sub-format's first two bytes
				format = int(binary.LittleEndian.Uint16(body[24:26]))
			}
		case "data":
			data = body
		}
		p += 8 + size + size%2
	}
	if channels < 1 || rate < 1 || data == nil {
		return Clip{}, ErrWAV
	}
	width := bits / 8
	if (format != 1 && format != 3) || width < 1 || format == 1 && width > 4 || format == 3 && width != 4 && width != 8 {
		return Clip{}, fmt.Errorf("%w: format %d of %d bits", ErrWAV, format, bits)
	}
	frames := len(data) / (width * channels)
	out := make([]float32, frames)
	for i := range frames {
		var sum float64
		for c := range channels {
			at := (i*channels + c) * width
			s := data[at : at+width]
			var v float64
			switch {
			case format == 3 && width == 4:
				v = float64(math.Float32frombits(binary.LittleEndian.Uint32(s)))
			case format == 3:
				v = math.Float64frombits(binary.LittleEndian.Uint64(s))
			case width == 1:
				v = (float64(s[0]) - 128) / 128 // 8-bit WAV is unsigned
			case width == 2:
				v = float64(int16(binary.LittleEndian.Uint16(s))) / 32768
			case width == 3:
				v = float64(int32(uint32(s[0])<<8|uint32(s[1])<<16|uint32(s[2])<<24)>>8) / (1 << 23)
			default:
				v = float64(int32(binary.LittleEndian.Uint32(s))) / (1 << 31)
			}
			sum += v
		}
		out[i] = float32(sum / float64(channels))
	}
	return Clip{Samples: out, Rate: rate}, nil
}

// WriteWAV writes c as a 16-bit mono WAV file.
func WriteWAV(w io.Writer, c Clip) error {
	var b bytes.Buffer
	n := len(c.Samples)
	le := binary.LittleEndian
	b.WriteString("RIFF")
	binary.Write(&b, le, uint32(36+2*n))
	b.WriteString("WAVEfmt ")
	binary.Write(&b, le, uint32(16))
	binary.Write(&b, le, uint16(1))
	binary.Write(&b, le, uint16(1))
	binary.Write(&b, le, uint32(c.Rate))
	binary.Write(&b, le, uint32(2*c.Rate))
	binary.Write(&b, le, uint16(2))
	binary.Write(&b, le, uint16(16))
	b.WriteString("data")
	binary.Write(&b, le, uint32(2*n))
	for _, s := range c.Samples {
		v := math.Round(float64(min(max(s, -1), 1)) * 32767)
		binary.Write(&b, le, int16(v))
	}
	_, err := w.Write(b.Bytes())
	return err
}

// Resample returns c at rate samples a second, by band-limited
// (windowed sinc) interpolation, so lowering the rate doesn't fold high
// frequencies into low ones.
func (c Clip) Resample(rate int) Clip {
	if rate == c.Rate || rate < 1 || len(c.Samples) == 0 {
		return Clip{Samples: append([]float32(nil), c.Samples...), Rate: max(rate, c.Rate)}
	}
	ratio := float64(rate) / float64(c.Rate)
	cutoff := min(1, ratio) // of the input's Nyquist frequency
	const half = 16         // taps a side, at the cutoff
	reach := float64(half) / cutoff
	n := int(math.Floor(float64(len(c.Samples)) * ratio))
	out := make([]float32, n)
	for i := range n {
		t := float64(i) / ratio // the position in the input
		lo, hi := int(math.Ceil(t-reach)), int(math.Floor(t+reach))
		var s, wsum float64
		for j := max(lo, 0); j <= min(hi, len(c.Samples)-1); j++ {
			d := (t - float64(j)) * cutoff
			w := sinc(d) * cutoff
			// A Hann window over the taps.
			w *= 0.5 + 0.5*math.Cos(math.Pi*(t-float64(j))/reach)
			s += w * float64(c.Samples[j])
			wsum += w
		}
		if wsum != 0 {
			s /= wsum // so a constant stays itself
		}
		out[i] = float32(s)
	}
	return Clip{Samples: out, Rate: rate}
}

func sinc(x float64) float64 {
	if x == 0 {
		return 1
	}
	return math.Sin(math.Pi*x) / (math.Pi * x)
}

// Fit cuts or pads c with silence to n samples, centred, so clips of
// different lengths batch together.
func (c Clip) Fit(n int) Clip {
	out := make([]float32, n)
	if len(c.Samples) >= n {
		start := (len(c.Samples) - n) / 2
		copy(out, c.Samples[start:start+n])
	} else {
		copy(out[(n-len(c.Samples))/2:], c.Samples)
	}
	return Clip{Samples: out, Rate: c.Rate}
}
