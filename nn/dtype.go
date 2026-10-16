package nn

import (
	"context"
	"fmt"

	"github.com/gsoultan/mimir/io/safetensors"
	"github.com/gsoultan/mimir/tensor"
)

// DTypeName returns the safetensors dtype that Save writes tensors of dt
// as: "F32", "I32", "BF16" or "F16".
func DTypeName(dt tensor.DType) string {
	switch dt {
	case tensor.Float32:
		return "F32"
	case tensor.Int32:
		return "I32"
	case tensor.BFloat16:
		return "BF16"
	case tensor.Float16:
		return "F16"
	}
	return ""
}

// DTypeOf returns the dtype that Decode gives a safetensors tensor of
// dtype name, and whether it decodes that name.
func DTypeOf(name string) (tensor.DType, bool) {
	for _, dt := range []tensor.DType{tensor.Float32, tensor.Int32, tensor.BFloat16, tensor.Float16} {
		if DTypeName(dt) == name {
			return dt, true
		}
	}
	return 0, false
}

// Cast converts every trainable parameter of m to dt, a float dtype, in
// place: a model cast to BFloat16 or Float16 holds its weights in half the
// memory and computes with them as the tensor package describes under
// mixed precision. Buffers, such as batch normalization's running
// statistics, keep their dtype: they accumulate small changes over many
// batches, which a half type would lose. Each parameter keeps its
// accumulated gradient. Optimizers update half parameters in their own
// precision unless given optim.MasterWeights.
func Cast(m Module, dt tensor.DType) error {
	if !dt.IsFloat() {
		return fmt.Errorf("nn: cast parameters to %s: %w", dt, tensor.ErrDType)
	}
	var names []string
	var params []*tensor.Param
	var values []*tensor.Tensor
	for name, p := range NamedParams(m) {
		if !p.Trainable() || p.Value().DType() == dt {
			continue
		}
		names, params = append(names, name), append(params, p)
		values = append(values, p.Value().Detach().Cast(dt))
	}
	if err := tensor.Realize(context.Background(), values...); err != nil {
		return fmt.Errorf("nn: cast to %s: %w", dt, err)
	}
	for i, p := range params {
		if err := p.Set(values[i]); err != nil {
			return fmt.Errorf("nn: cast %s to %s: %w", names[i], dt, err)
		}
	}
	return nil
}

// encodeHalf returns t, a BFloat16 or Float16 tensor, as a safetensors
// tensor of its dtype, through the float32 values of its elements.
func encodeHalf(ctx context.Context, t *tensor.Tensor) (safetensors.Tensor, error) {
	v, err := t.Cast(tensor.Float32).Data[float32](ctx)
	if t.DType() == tensor.BFloat16 {
		return safetensors.BFloat16Tensor(t.Shape(), v), err
	}
	return safetensors.Float16Tensor(t.Shape(), v), err
}

// decodeHalf returns a BF16 or F16 safetensors tensor as a computed tensor
// of BFloat16 or Float16: its elements are float32 values, which the cast
// stores as they are.
func decodeHalf(st safetensors.Tensor) (*tensor.Tensor, error) {
	v, err := st.Float32s()
	if err != nil {
		return nil, err
	}
	dt, _ := DTypeOf(st.DType)
	x := tensor.FromSlice(v, st.Shape...).Cast(dt)
	if err := x.Realize(context.Background()); err != nil {
		return nil, err
	}
	return x, nil
}
