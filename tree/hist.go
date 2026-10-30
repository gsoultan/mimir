package tree

// addRowsGo is addRows in Go: it adds the sz numbers gh[r*sz:] of each
// row r of rows into hist at the row's bin of each feature of feats, whose
// bins start at offs.
func addRowsGo(hist []float64, b *binned, feats, offs []int, rows []int32, gh []float32, sz int) {
	f := b.f
	v := make([]float64, sz)
	for _, r := range rows {
		row := b.bins[int(r)*f : int(r)*f+f]
		for o, x := range gh[int(r)*sz : int(r)*sz+sz] {
			v[o] = float64(x)
		}
		for fi, j := range feats {
			e := (offs[fi] + int(row[j])) * sz
			c := hist[e : e+sz : e+sz]
			for o, x := range v {
				c[o] += x
			}
		}
	}
}
