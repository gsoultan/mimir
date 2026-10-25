//go:build !purego

package cpu

//go:generate sh -c "cd ../../tools/asmgen && go run ./winograd -out ../../backend/cpu/winograd_amd64.s -stubs ../../backend/cpu/winograd_stub_amd64.go -pkg cpu"

func init() {
	if x86Features().avx512 {
		winoInputRow, winoGradRow, winoOutputRow = winoInputAVX512, winoGradAVX512, winoOutputAVX512
		winoInputChannels, winoGradChannels = winoInputChannelsAVX512, winoGradChannelsAVX512
	}
}
