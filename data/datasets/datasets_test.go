package datasets

import (
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/gsoultan/mimir/data"
)

// idx encodes an IDX file of bytes.
func idx(dims []int, body []byte) []byte {
	b := binary.BigEndian.AppendUint32(nil, uint32(0x0800|len(dims)))
	for _, d := range dims {
		b = binary.BigEndian.AppendUint32(b, uint32(d))
	}
	return append(b, body...)
}

func gz(t *testing.T, b []byte) []byte {
	var buf bytes.Buffer
	w := gzip.NewWriter(&buf)
	if _, err := w.Write(b); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// writeMNIST writes a tiny MNIST of n 2×2 images labelled 0..n-1, gzipping
// the training files and leaving the test files raw.
func writeMNIST(t *testing.T, dir string, n int) {
	px := make([]byte, 4*n)
	labels := make([]byte, n)
	for i := range n {
		px[4*i] = byte(255 * i / max(n-1, 1))
		labels[i] = byte(i)
	}
	files := map[string][]byte{
		"train-images-idx3-ubyte.gz": gz(t, idx([]int{n, 2, 2}, px)),
		"train-labels-idx1-ubyte.gz": gz(t, idx([]int{n}, labels)),
		"t10k-images-idx3-ubyte":     idx([]int{n, 2, 2}, px),
		"t10k-labels-idx1-ubyte":     idx([]int{n}, labels),
	}
	for name, b := range files {
		if err := os.WriteFile(filepath.Join(dir, name), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

func TestMNISTReadsIDX(t *testing.T) {
	dir := t.TempDir()
	writeMNIST(t, dir, 5)
	train, test, err := MNIST(dir, data.BatchSize(2), data.Shuffle(1))
	if err != nil {
		t.Fatal(err)
	}
	if train.Len() != 5 || test.Len() != 5 || train.Batches() != 3 {
		t.Fatalf("lengths %d and %d, %d batches", train.Len(), test.Len(), train.Batches())
	}
	var ys []int32
	for b, err := range test.All(t.Context()) {
		if err != nil {
			t.Fatal(err)
		}
		if got := b.X.Shape(); got[1] != 1 || got[2] != 2 || got[3] != 2 {
			t.Fatalf("image batch shape %v", got)
		}
		y, err := b.Y.Data[int32](t.Context())
		if err != nil {
			t.Fatal(err)
		}
		ys = append(ys, y...)
		x, err := b.X.Data[float32](t.Context())
		if err != nil {
			t.Fatal(err)
		}
		for i, label := range y {
			if want := float32(byte(255*int(label)/4)) / 255; x[4*i] != want {
				t.Errorf("image %d pixel %v, want %v", label, x[4*i], want)
			}
		}
	}
	if !slices.Equal(ys, []int32{0, 1, 2, 3, 4}) {
		t.Errorf("test labels %v, want them in order", ys)
	}
}

func TestMNISTRejectsBadFiles(t *testing.T) {
	dir := t.TempDir()
	if _, _, err := MNIST(dir); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("missing files: %v", err)
	}
	writeMNIST(t, dir, 3)
	bad := idx([]int{4}, []byte{0, 1, 2, 3}) // 4 labels for 3 images
	if err := os.WriteFile(filepath.Join(dir, "t10k-labels-idx1-ubyte"), bad, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := MNIST(dir); !errors.Is(err, ErrFormat) {
		t.Errorf("label count mismatch: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "t10k-labels-idx1-ubyte"), []byte{0, 0, 8}, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := MNIST(dir); !errors.Is(err, ErrFormat) {
		t.Errorf("truncated header: %v", err)
	}
}

func TestDownloadMNISTVerifiesAndFallsBack(t *testing.T) {
	good := []byte("mnist bytes")
	sum := sha256.Sum256(good)
	files, mirrors := mnistFiles, mnistMirrors
	t.Cleanup(func() { mnistFiles, mnistMirrors = files, mirrors })
	mnistFiles = []struct{ name, sha256 string }{{"a.gz", hex.EncodeToString(sum[:])}}

	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		switch {
		case strings.HasPrefix(r.URL.Path, "/down/"):
			http.NotFound(w, r)
		case strings.HasPrefix(r.URL.Path, "/corrupt/"):
			w.Write([]byte("tampered"))
		default:
			w.Write(good)
		}
	}))
	defer srv.Close()

	dir := filepath.Join(t.TempDir(), "mnist")
	mnistMirrors = []string{srv.URL + "/corrupt/"}
	if err := DownloadMNIST(t.Context(), dir); !errors.Is(err, ErrFormat) {
		t.Errorf("corrupt download: %v, want ErrFormat", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "a.gz")); !errors.Is(err, os.ErrNotExist) {
		t.Error("kept a download whose checksum did not match")
	}
	mnistMirrors = []string{srv.URL + "/down/", srv.URL + "/ok/"}
	if err := DownloadMNIST(t.Context(), dir); err != nil {
		t.Fatal(err)
	}
	if b, _ := os.ReadFile(filepath.Join(dir, "a.gz")); !bytes.Equal(b, good) {
		t.Errorf("saved %q", b)
	}
	before := hits.Load()
	if err := DownloadMNIST(t.Context(), dir); err != nil || hits.Load() != before {
		t.Errorf("second download fetched again (%v)", err)
	}
	if entries, _ := os.ReadDir(dir); len(entries) != 1 {
		t.Errorf("left temporary files: %v", entries)
	}
}

func TestCSV(t *testing.T) {
	const in = "sepal,petal,species\n5.1,1.4,setosa\n7.0,4.7,versicolor\n6.3,6.0,virginica\n4.9,1.4,setosa\n"
	tab, err := CSV(strings.NewReader(in), "species", With(data.BatchSize(4)))
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(tab.Features, []string{"sepal", "petal"}) || !slices.Equal(tab.Classes, []string{"setosa", "versicolor", "virginica"}) {
		t.Errorf("features %v, classes %v", tab.Features, tab.Classes)
	}
	for b, err := range tab.All(t.Context()) {
		if err != nil {
			t.Fatal(err)
		}
		y, _ := b.Y.Data[int32](t.Context())
		x, _ := b.X.Data[float32](t.Context())
		if !slices.Equal(y, []int32{0, 1, 2, 0}) || x[2] != 7 || x[3] != 4.7 {
			t.Errorf("x %v, y %v", x, y)
		}
	}

	reg, err := CSV(strings.NewReader("a,price\n1,10.5\n2,20\n"), "price")
	if err != nil {
		t.Fatal(err)
	}
	if reg.Classes != nil {
		t.Errorf("numeric target read as classes %v", reg.Classes)
	}
	cls, err := CSV(strings.NewReader("a,digit\n1,3\n2,7\n"), "digit", Classes())
	if err != nil || !slices.Equal(cls.Classes, []string{"3", "7"}) {
		t.Errorf("Classes(): %v, %v", cls, err)
	}

	for name, bad := range map[string]string{
		"no target":   "a,b\n1,2\n",
		"not numeric": "a,y\nx,1\n",
		"no rows":     "a,y\n",
		"NaN":         "a,y\n1,2\nnan,1\n",
		"Inf":         "a,y\n1,2\n3,4\n-Infinity,1\n",
		"Inf target":  "a,y\n1,2\n3,inf\n",
		"too large":   "a,y\n1e39,1\n",
	} {
		if _, err := CSV(strings.NewReader(bad), "y"); !errors.Is(err, ErrFormat) {
			t.Errorf("%s: %v, want ErrFormat", name, err)
		}
	}
	// The line and column of a value that is not finite.
	if _, err := CSV(strings.NewReader("a,y\n1,2\n3,4\n-Infinity,1\n"), "y"); err == nil || !strings.Contains(err.Error(), `line 4 column "a"`) || !strings.Contains(err.Error(), "finite") {
		t.Errorf("Inf feature: %v", err)
	}
	if _, err := CSV(strings.NewReader("a,y\n1,2\n3,NaN\n"), "y"); err == nil || !strings.Contains(err.Error(), `line 3 column "y"`) {
		t.Errorf("NaN target: %v", err)
	}
	// As class labels, such strings are labels like any other.
	if tab, err := CSV(strings.NewReader("a,y\n1,nan\n2,inf\n"), "y", Classes()); err != nil || !slices.Equal(tab.Classes, []string{"inf", "nan"}) {
		t.Errorf("labels nan and inf: %v", err)
	}
}
