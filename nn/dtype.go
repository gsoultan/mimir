package nn

import (
	"context"
	"fmt"

	"github.com/gsoultan/mimir/tensor"
)

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
