// Package grid writes batches of grayscale images as one PNG of tiles,
// and clouds of 2-D points as a scatter plot, for the generative examples.
package grid

import (
	"context"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"

	"github.com/gsoultan/mimir/tensor"
)

// ErrShape reports images whose size does not match the tile size.
var ErrShape = errors.New("grid: images do not fit the tiles")

// Write saves images, a tensor of n tiles of h×w pixels in any shape with
// n·h·w elements, to path as a PNG of cols tiles a row with a pixel of
// border. Pixels map from [lo, hi] to black through white; values outside
// are clamped.
func Write(ctx context.Context, path string, images *tensor.Tensor, h, w, cols int, lo, hi float64) error {
	px, err := images.Cast(tensor.Float32).Data[float32](ctx)
	if err != nil {
		return err
	}
	if h <= 0 || w <= 0 || cols <= 0 || len(px)%(h*w) != 0 || !(hi > lo) {
		return fmt.Errorf("grid: %d pixels in %d×%d tiles, %d a row, range [%g, %g]: %w", len(px), h, w, cols, lo, hi, ErrShape)
	}
	n := len(px) / (h * w)
	rows := (n + cols - 1) / cols
	img := image.NewGray(image.Rect(0, 0, cols*(w+1)+1, rows*(h+1)+1))
	for i := range img.Pix {
		img.Pix[i] = 64
	}
	for k := range n {
		x0, y0 := 1+(k%cols)*(w+1), 1+(k/cols)*(h+1)
		for y := range h {
			for x := range w {
				v := (float64(px[k*h*w+y*w+x]) - lo) / (hi - lo)
				if math.IsNaN(v) {
					v = 0
				}
				img.SetGray(x0+x, y0+y, color.Gray{Y: uint8(math.Round(255 * min(max(v, 0), 1)))})
			}
		}
	}
	return save(path, img)
}

// Scatter saves a size×size PNG of the 2-D points of each tensor of sets,
// [n, 2] with x and y in [lo, hi], each set a lighter gray than the one
// before; points outside are left out.
func Scatter(ctx context.Context, path string, size int, lo, hi float64, sets ...*tensor.Tensor) error {
	if size <= 0 || !(hi > lo) {
		return fmt.Errorf("grid: scatter of %d pixels over [%g, %g]: %w", size, lo, hi, ErrShape)
	}
	img := image.NewGray(image.Rect(0, 0, size, size))
	for k, set := range sets {
		if s := set.Shape(); len(s) != 2 || s[1] != 2 {
			return fmt.Errorf("grid: scatter of points %v, want [n, 2]: %w", s, ErrShape)
		}
		xy, err := set.Cast(tensor.Float32).Data[float32](ctx)
		if err != nil {
			return err
		}
		shade := color.Gray{Y: uint8(255 * (k + 1) / len(sets))}
		for i := 0; i+1 < len(xy); i += 2 {
			x := (float64(xy[i]) - lo) / (hi - lo) * float64(size)
			y := (hi - float64(xy[i+1])) / (hi - lo) * float64(size)
			if x >= 0 && x < float64(size) && y >= 0 && y < float64(size) {
				img.SetGray(int(x), int(y), shade)
			}
		}
	}
	return save(path, img)
}

func save(path string, img image.Image) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := png.Encode(f, img); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
