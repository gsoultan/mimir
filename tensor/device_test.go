package tensor_test

import (
	"errors"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/backend/cpu"
	"github.com/gsoultan/mimir/tensor"
)

// Two CPU backends stand in for two devices.

func TestToMovesValuesAndGradients(t *testing.T) {
	other := tensor.DeviceOn(cpu.New())
	x := tensor.FromSlice([]float32{1, -2, 3}).RequiresGrad()
	y := x.To(other)
	if !y.Device().Equal(other) || !x.Device().Equal(tensor.DefaultDevice()) {
		t.Fatalf("devices %v and %v after To", x.Device(), y.Device())
	}
	if y.To(other) != y {
		t.Error("To the tensor's own device made a copy")
	}
	loss := y.Mul(y).Sum()
	if !loss.Device().Equal(other) {
		t.Errorf("a tensor computed from one on %v is on %v", other, loss.Device())
	}
	if err := loss.Backward(t.Context()); err != nil {
		t.Fatal(err)
	}
	g, err := x.Grad().Data[float32](t.Context())
	if err != nil || !slices.Equal(g, []float32{2, -4, 6}) {
		t.Errorf("gradient through To = %v, %v; want [2 -4 6]", g, err)
	}
	if !x.Grad().Device().Equal(x.Device()) {
		t.Errorf("gradient of a tensor on %v is on %v", x.Device(), x.Grad().Device())
	}
}

func TestCombiningDevicesFails(t *testing.T) {
	other := tensor.DeviceOn(cpu.New())
	x := tensor.FromSlice([]float32{1, 2})
	y := tensor.FromSlice([]float32{3, 4}).To(other)
	if err := x.Add(y).Err(); !errors.Is(err, tensor.ErrDevice) {
		t.Errorf("adding tensors on two devices: got %v, want ErrDevice", err)
	}
	// Constants and random values take the device of what they meet.
	z := y.Add(tensor.Ones(tensor.Float32, 2)).Mul(tensor.Rand(1, 2).Add(tensor.Full(tensor.Float32, 1)))
	if err := z.Realize(t.Context()); err != nil || !z.Device().Equal(other) {
		t.Errorf("constants with a tensor on %v: %v on %v", other, err, z.Device())
	}
	// Even once computed on another device.
	c := tensor.Full(tensor.Float32, 10, 2)
	if err := c.Realize(t.Context()); err != nil {
		t.Fatal(err)
	}
	got, err := y.Add(c).Data[float32](t.Context())
	if err != nil || !slices.Equal(got, []float32{13, 14}) {
		t.Errorf("adding a constant computed on the default device = %v, %v; want [13 14]", got, err)
	}
}

func TestRealizeComputesEachDeviceSeparately(t *testing.T) {
	other := tensor.DeviceOn(cpu.New())
	x := tensor.FromSlice([]float32{1, 2}).Neg()
	y := tensor.FromSlice([]float32{3, 4}).To(other).Neg()
	if err := tensor.Realize(t.Context(), x, y); err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		t    *tensor.Tensor
		want []float32
	}{{x, []float32{-1, -2}}, {y, []float32{-3, -4}}} {
		if got, err := c.t.Data[float32](t.Context()); err != nil || !slices.Equal(got, c.want) {
			t.Errorf("got %v, %v; want %v", got, err, c.want)
		}
	}
}

func TestDevicesListTheCPUFirst(t *testing.T) {
	ds := tensor.Devices()
	if len(ds) == 0 || !ds[0].Equal(tensor.CPU()) || ds[0].Name() != "cpu" {
		t.Fatalf("Devices() = %v, want the CPU first", ds)
	}
	if _, err := tensor.GPU(); (err == nil) != (len(ds) == 2) {
		t.Errorf("GPU() error %v with devices %v", err, ds)
	}
	if a := tensor.Auto(); len(ds) == 1 && !a.Equal(tensor.CPU()) {
		t.Errorf("Auto() = %v without a GPU, want the CPU", a)
	}
}

func TestArgmaxRunsOnTheTensorsDevice(t *testing.T) {
	other := tensor.DeviceOn(cpu.New())
	y := tensor.FromSlice([]float32{3, 7, 7, 1}, 2, 2).To(other)
	got, err := y.Argmax(1).Data[int32](t.Context())
	if err != nil || !slices.Equal(got, []int32{1, 0}) {
		t.Errorf("Argmax on another device = %v, %v; want [1 0]", got, err)
	}
}
