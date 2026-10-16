// Package datasets loads standard datasets as data.Datasets, and streams
// CSV shards too large for memory as a data.Source.
package datasets

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"os"
	"path/filepath"

	"github.com/gsoultan/mimir/data"
	"github.com/gsoultan/mimir/tensor"
)

// ErrFormat reports a malformed dataset file.
var ErrFormat = errors.New("datasets: invalid file")

// mnistFiles are the four MNIST files with the SHA-256 of their gzipped
// form, which matches the MD5 sums torchvision publishes.
var mnistFiles = []struct{ name, sha256 string }{
	{"train-images-idx3-ubyte.gz", "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609"},
	{"train-labels-idx1-ubyte.gz", "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c"},
	{"t10k-images-idx3-ubyte.gz", "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6"},
	{"t10k-labels-idx1-ubyte.gz", "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6"},
}

// mnistMirrors serve the MNIST files.
var mnistMirrors = []string{
	"https://ossci-datasets.s3.amazonaws.com/mnist/",
	"https://storage.googleapis.com/cvdf-datasets/mnist/",
}

// MNIST loads the MNIST handwritten digits from dir: 60,000 training and
// 10,000 test images of 28×28 pixels. X is Float32 [N, 1, 28, 28] scaled
// to [0, 1] and Y is the Int32 digit [N]. The training set uses opts; the
// test set uses them too but keeps its order.
//
// dir must hold the four standard files, gzipped or not, as
// DownloadMNIST leaves them.
func MNIST(dir string, opts ...data.Option) (train, test *data.Tensors, err error) {
	load := func(images, labels string, opts []data.Option) (*data.Tensors, error) {
		x, n, err := readIDXImages(filepath.Join(dir, images))
		if err != nil {
			return nil, err
		}
		y, err := readIDXLabels(filepath.Join(dir, labels), n)
		if err != nil {
			return nil, err
		}
		return data.FromTensors(x, y, opts...)
	}
	if train, err = load("train-images-idx3-ubyte", "train-labels-idx1-ubyte", opts); err != nil {
		return nil, nil, err
	}
	if test, err = load("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", append(opts, data.InOrder())); err != nil {
		return nil, nil, err
	}
	return train, test, nil
}

// DownloadMNIST fetches any MNIST file missing from dir, creating dir if
// needed, and checks each download's SHA-256 before keeping it.
func DownloadMNIST(ctx context.Context, dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	for _, f := range mnistFiles {
		path := filepath.Join(dir, f.name)
		if _, err := os.Stat(path); err == nil {
			continue
		}
		if _, err := os.Stat(path[:len(path)-len(".gz")]); err == nil {
			continue
		}
		var errs []error
		for _, mirror := range mnistMirrors {
			err := download(ctx, mirror+f.name, path, f.sha256)
			if err == nil {
				errs = nil
				break
			}
			errs = append(errs, err)
			if ctx.Err() != nil {
				break
			}
		}
		if errs != nil {
			return fmt.Errorf("datasets: download %s: %w", f.name, errors.Join(errs...))
		}
	}
	return nil
}

// download saves url to path if its SHA-256 matches sum.
func download(ctx context.Context, url, path, sum string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%s: %s", url, resp.Status)
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".download-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name()) // no-op after the rename
	h := sha256.New()
	_, err = io.Copy(io.MultiWriter(tmp, h), io.LimitReader(resp.Body, 64<<20))
	if cerr := tmp.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return err
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != sum {
		return fmt.Errorf("%s: SHA-256 %s, want %s: %w", url, got, sum, ErrFormat)
	}
	return os.Rename(tmp.Name(), path)
}

// openMaybeGzipped opens path, or path.gz, decompressing as needed.
func openMaybeGzipped(path string) ([]byte, error) {
	b, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		b, err = os.ReadFile(path + ".gz")
		if errors.Is(err, fs.ErrNotExist) {
			return nil, fmt.Errorf("datasets: %s missing (DownloadMNIST fetches it): %w", path, err)
		}
	}
	if err != nil {
		return nil, err
	}
	if !bytes.HasPrefix(b, []byte{0x1f, 0x8b}) {
		return b, nil
	}
	zr, err := gzip.NewReader(bytes.NewReader(b))
	if err != nil {
		return nil, fmt.Errorf("datasets: %s: %v: %w", path, err, ErrFormat)
	}
	defer zr.Close()
	out, err := io.ReadAll(bufio.NewReader(io.LimitReader(zr, 1<<30)))
	if err != nil {
		return nil, fmt.Errorf("datasets: %s: %v: %w", path, err, ErrFormat)
	}
	return out, nil
}

// readIDX parses an IDX file of unsigned bytes with the given number of
// dimensions and returns its dimensions and data.
func readIDX(path string, rank int) ([]int, []byte, error) {
	b, err := openMaybeGzipped(path)
	if err != nil {
		return nil, nil, err
	}
	head := 4 + 4*rank
	if len(b) < head || binary.BigEndian.Uint32(b) != uint32(0x0800|rank) {
		return nil, nil, fmt.Errorf("datasets: %s is not a rank-%d IDX file of bytes: %w", path, rank, ErrFormat)
	}
	dims := make([]int, rank)
	n := 1
	for i := range dims {
		dims[i] = int(binary.BigEndian.Uint32(b[4+4*i:]))
		if dims[i] > 1<<26 {
			return nil, nil, fmt.Errorf("datasets: %s has dimension %d: %w", path, dims[i], ErrFormat)
		}
		n *= dims[i]
	}
	if len(b)-head != n {
		return nil, nil, fmt.Errorf("datasets: %s holds %d bytes for dimensions %v: %w", path, len(b)-head, dims, ErrFormat)
	}
	return dims, b[head:], nil
}

func readIDXImages(path string) (*tensor.Tensor, int, error) {
	dims, px, err := readIDX(path, 3)
	if err != nil {
		return nil, 0, err
	}
	x := make([]float32, len(px))
	for i, p := range px {
		x[i] = float32(p) / 255
	}
	return tensor.FromSlice(x, dims[0], 1, dims[1], dims[2]), dims[0], nil
}

func readIDXLabels(path string, n int) (*tensor.Tensor, error) {
	dims, raw, err := readIDX(path, 1)
	if err != nil {
		return nil, err
	}
	if dims[0] != n {
		return nil, fmt.Errorf("datasets: %s has %d labels for %d images: %w", path, dims[0], n, ErrFormat)
	}
	y := make([]int32, n)
	for i, l := range raw {
		y[i] = int32(l)
	}
	return tensor.FromSlice(y), nil
}
