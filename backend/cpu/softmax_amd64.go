//go:build !purego

package cpu

//go:generate sh -c "cd ../../tools/asmgen && go run ./softmax -out ../../backend/cpu/softmax_amd64.s -stubs ../../backend/cpu/softmax_stub_amd64.go -pkg cpu"

var (
	rowAVX512 = rowRoutines{lanes: 16, max: rowMaxAVX512, expShift: expShiftAVX512, scale: scaleRowAVX512, softmaxGrad: softmaxGradAVX512}
	rowAVX2   = rowRoutines{lanes: 8, max: rowMaxAVX2, expShift: expShiftAVX2, scale: scaleRowAVX2, softmaxGrad: softmaxGradAVX2}
)

func init() {
	switch f := x86Features(); {
	case f.avx512:
		rowVector = &rowAVX512
	case f.avx2 && f.fma:
		rowVector = &rowAVX2
	}
}
