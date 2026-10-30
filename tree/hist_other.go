//go:build !amd64 || purego

package tree

// addRows adds the sz numbers gh[r*sz:] of each row r of rows into hist
// at the row's bin of each feature of feats, whose bins start at offs.
func addRows(hist []float64, b *binned, feats, offs []int, rows []int32, gh []float32, sz int) {
	addRowsGo(hist, b, feats, offs, rows, gh, sz)
}
