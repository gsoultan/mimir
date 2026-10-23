//go:build !purego

package cpu

//go:generate sh -c "cd ../../tools/asmgen && go run ./softmax -out ../../backend/cpu/softmax_amd64.s -stubs ../../backend/cpu/softmax_stub_amd64.go -pkg cpu"

var (
	rowAVX512 = rowRoutines{lanes: 16, max: rowMaxAVX512, expShift: expShiftAVX512, scale: scaleRowAVX512, softmaxGrad: softmaxGradAVX512, dot4: dot4AVX512, exp2: exp2AVX512,
		addVV: addVVAVX512, mulVV: mulVVAVX512, addVS: addVSAVX512, mulVS: mulVSAVX512, recip: recipAVX512, sqrt: sqrtAVX512}
	rowAVX2 = rowRoutines{lanes: 8, max: rowMaxAVX2, expShift: expShiftAVX2, scale: scaleRowAVX2, softmaxGrad: softmaxGradAVX2, dot4: dot4AVX2, exp2: exp2AVX2,
		addVV: addVVAVX2, mulVV: mulVVAVX2, addVS: addVSAVX2, mulVS: mulVSAVX2, recip: recipAVX2, sqrt: sqrtAVX2}
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
