package main

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/gsoultan/mimir/tensor"
)

func TestRun(t *testing.T) {
	for _, tc := range []struct {
		model, inputs string
		want          []string
	}{
		{"cnn", "", []string{"input  x: float[batch 3 16 16], given [1 3 16 16]", "output probs: float32[1 10]"}},
		{"transformer", "", []string{"given [2 5]", "output logits: float32[2 5 50]"}},
		{"decoder", "../../io/onnx/testdata/decoder.safetensors", []string{"output image: float32[1 2 20 20]", "output smooth: float32[1 4 7 7]"}},
	} {
		t.Run(tc.model, func(t *testing.T) {
			var out bytes.Buffer
			c := config{model: "../../io/onnx/testdata/" + tc.model + ".onnx", inputs: tc.inputs,
				dims: dims{"batch": 2, "seq": 5}, runs: 2, intMax: 50, dev: tensor.CPU()}
			if tc.model == "cnn" {
				c.dims = dims{}
			}
			if err := run(context.Background(), &out, c); err != nil {
				t.Fatal(err)
			}
			for _, w := range append(tc.want, "first run") {
				if !strings.Contains(out.String(), w) {
					t.Errorf("output lacks %q:\n%s", w, out.String())
				}
			}
		})
	}
}
