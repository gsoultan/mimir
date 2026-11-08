// Copyright (c) 2026 Gembit Soultan Shirazi
// SPDX-License-Identifier: Apache-2.0
//
// This file is part of mimir, a pure-Go machine learning framework,
// created and owned by Gembit Soultan Shirazi.
// It is licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License in the LICENSE file at the root
// of this repository or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

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
