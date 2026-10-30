//go:build !purego

package tree

//go:generate sh -c "cd ../tools/asmgen && go run ./hist -out ../../tree/hist_amd64.s -stubs ../../tree/hist_stub_amd64.go -pkg tree"

// avx2 reports whether the CPU and OS run AVX2.
var avx2 = func() bool {
	maxID, _, _, _ := cpuid(0, 0)
	if maxID < 7 {
		return false
	}
	_, _, ecx1, _ := cpuid(1, 0)
	const osxsave, avx = 1 << 27, 1 << 28
	if ecx1&osxsave == 0 || ecx1&avx == 0 {
		return false
	}
	if xcr0, _ := xgetbv(); xcr0&0x6 != 0x6 {
		return false
	}
	_, ebx7, _, _ := cpuid(7, 0)
	return ebx7&(1<<5) != 0
}()

// addRows adds the sz numbers gh[r*sz:] of each row r of rows into hist
// at the row's bin of each feature of feats, whose bins start at offs,
// with AVX2 when it can. The kernels check no bounds: gh holds a stride
// for each of b's rows, rows are all below b.n, and hist holds every bin.
func addRows(hist []float64, b *binned, feats, offs []int, rows []int32, gh []float32, sz int) {
	if !avx2 || (sz != 4 && sz != 8) || len(rows) == 0 || len(feats) == 0 {
		addRowsGo(hist, b, feats, offs, rows, gh, sz)
		return
	}
	_ = gh[b.n*sz-1]
	_ = hist[b.total*sz-1]
	if sz == 4 {
		addRows4AVX2(&hist[0], &b.bins[0], b.f, &feats[0], &offs[0], len(feats), &rows[0], &gh[0], len(rows))
	} else {
		addRows8AVX2(&hist[0], &b.bins[0], b.f, &feats[0], &offs[0], len(feats), &rows[0], &gh[0], len(rows))
	}
}
