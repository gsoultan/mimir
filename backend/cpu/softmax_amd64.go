//go:build !purego

package cpu

//go:generate sh -c "cd ../../tools/asmgen && go run ./softmax -out ../../backend/cpu/softmax_amd64.s -stubs ../../backend/cpu/softmax_stub_amd64.go -pkg cpu"

var (
	rowAVX512 = rowRoutines{lanes: 16, max: rowMaxAVX512, expShift: expShiftAVX512, scale: scaleRowAVX512, softmaxGrad: softmaxGradAVX512, dot: dotAVX512, exp2: exp2AVX512}
	rowAVX2   = rowRoutines{lanes: 8, max: rowMaxAVX2, expShift: expShiftAVX2, scale: scaleRowAVX2, softmaxGrad: softmaxGradAVX2, dot: dotAVX2, exp2: exp2AVX2}
)

func init() {
	switch f := x86Features(); {
	case f.avx512:
		rowVector = &rowAVX512
	case f.avx2 && f.fma:
		rowVector = &rowAVX2
	}
}

// rowRoutineSets returns every set of row routines this CPU supports, by
// name, for tests.
func rowRoutineSets() map[string]*rowRoutines {
	sets := map[string]*rowRoutines{}
	f := x86Features()
	if f.avx512 {
		sets["avx512"] = &rowAVX512
	}
	if f.avx2 && f.fma {
		sets["avx2"] = &rowAVX2
	}
	return sets
}
