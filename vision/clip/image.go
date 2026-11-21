// Copyright (c) 2026 Gembit Soultan Shirazi
// SPDX-License-Identifier: Apache-2.0
//
// This file is part of mimir, a pure-Go machine learning framework,
// created and owned by Gembit Soultan Shirazi.
// It is licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License in the LICENSE file at the root
// of this repository or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

package clip

import (
	"fmt"
	"image"
	"image/color"
	"math"

	"github.com/gsoultan/mimir/tensor"
)

// maxPixels bounds the images Prepare reads, 100 megapixels.
const maxPixels = 100 << 20

// Prepare returns img as the image encoder reads it, [3, ImageSize,
// ImageSize] in channel order red, green, blue: the shorter side resized
// to Resize with a bicubic filter, rounding to 8 bits after each pass as
// Pillow does, the middle ImageSize square cut out, and each channel
// scaled to [0, 1] then by the model's mean and deviation.
func (m *Model) Prepare(img image.Image) ([]float32, error) {
	c := m.Config
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	if w < 1 || h < 1 || w*h > maxPixels {
		return nil, fmt.Errorf("clip: an image of %d×%d: %w", w, h, tensor.ErrShape)
	}
	rgb := make([]float64, 3*w*h) // [h, w, 3]
	for y := range h {
		for x := range w {
			r, g, bl, _ := color.NRGBAModel.Convert(img.At(b.Min.X+x, b.Min.Y+y)).RGBA()
			i := 3 * (y*w + x)
			rgb[i], rgb[i+1], rgb[i+2] = float64(r>>8), float64(g>>8), float64(bl>>8)
		}
	}
	// The shorter side to Resize, the other in proportion, rounded down,
	// as Hugging Face's get_resize_output_image_size.
	nw, nh := c.Resize, c.Resize
	if w < h {
		nh = int(float64(c.Resize) * float64(h) / float64(w))
	} else if h < w {
		nw = int(float64(c.Resize) * float64(w) / float64(h))
	}
	if nw != w {
		rgb = resample(rgb, h, w, nw, true)
	}
	if nh != h {
		rgb = resample(rgb, h, nw, nh, false)
	}
	s := c.ImageSize
	top, left := (nh-s)/2, (nw-s)/2
	out := make([]float32, 3*s*s)
	for ch := range 3 {
		for y := range s {
			for x := range s {
				v := rgb[3*((top+y)*nw+left+x)+ch] / 255
				out[ch*s*s+y*s+x] = float32((v - c.Mean[ch]) / c.Std[ch])
			}
		}
	}
	return out, nil
}

// bicubic is Pillow's bicubic filter, a = -0.5, of support 2.
func bicubic(x float64) float64 {
	const a = -0.5
	x = math.Abs(x)
	switch {
	case x < 1:
		return ((a+2)*x-(a+3))*x*x + 1
	case x < 2:
		return (((x-5)*x+8)*x - 4) * a
	}
	return 0
}

// resample resizes the rows (horizontal) or columns of rgb, [h, w, 3],
// to n pixels with Pillow's resampling: a bicubic filter widened by the
// scale when shrinking, so every source pixel counts, and each result
// rounded and clipped to 8 bits.
func resample(rgb []float64, h, w, n int, horizontal bool) []float64 {
	in := w
	if !horizontal {
		in = h
	}
	scale := float64(in) / float64(n)
	fscale := max(scale, 1)
	support := 2 * fscale
	type taps struct {
		lo int
		w  []float64
	}
	coeffs := make([]taps, n)
	for i := range n {
		center := (float64(i) + 0.5) * scale
		lo := max(int(center-support+0.5), 0)
		hi := min(int(center+support+0.5), in)
		ws := make([]float64, hi-lo)
		var total float64
		for j := range ws {
			ws[j] = bicubic((float64(j+lo) - center + 0.5) / fscale)
			total += ws[j]
		}
		if total != 0 {
			for j := range ws {
				ws[j] /= total
			}
		}
		coeffs[i] = taps{lo, ws}
	}
	clip8 := func(v float64) float64 { return min(max(math.Round(v), 0), 255) }
	if horizontal {
		out := make([]float64, 3*h*n)
		for y := range h {
			for x, t := range coeffs {
				for ch := range 3 {
					var s float64
					for j, wt := range t.w {
						s += wt * rgb[3*(y*w+t.lo+j)+ch]
					}
					out[3*(y*n+x)+ch] = clip8(s)
				}
			}
		}
		return out
	}
	out := make([]float64, 3*n*w)
	for y, t := range coeffs {
		for x := range w {
			for ch := range 3 {
				var s float64
				for j, wt := range t.w {
					s += wt * rgb[3*((t.lo+j)*w+x)+ch]
				}
				out[3*(y*w+x)+ch] = clip8(s)
			}
		}
	}
	return out
}
