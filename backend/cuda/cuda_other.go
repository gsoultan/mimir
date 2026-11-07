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

//go:build !(linux && (amd64 || arm64)) || nocuda

package cuda

import (
	"context"
	"fmt"

	"github.com/gsoultan/mimir/backend"
)

// Backend runs kernels on a CUDA device. This build has none: the
// platform is not Linux on amd64 or arm64, or the nocuda build tag is
// set, and New always fails.
type Backend struct{}

var _ backend.Backend = (*Backend)(nil)

// New reports ErrUnavailable: the backend is not built in.
func New() (*Backend, error) {
	return nil, fmt.Errorf("cuda: not built in (needs Linux on amd64 or arm64, without the nocuda tag): %w", ErrUnavailable)
}

// NewDevice reports ErrUnavailable: the backend is not built in.
func NewDevice(int) (*Backend, error) { return New() }

// DeviceCount returns 0.
func DeviceCount() int { return 0 }

// Name returns "cuda".
func (*Backend) Name() string { return "cuda" }

// DeviceName returns the empty string.
func (*Backend) DeviceName() string { return "" }

func (*Backend) Alloc(backend.DType, int) (backend.Buffer, error) { return nil, ErrUnavailable }
func (*Backend) Free(backend.Buffer)                              {}
func (*Backend) Upload(context.Context, backend.Buffer, []byte) error {
	return ErrUnavailable
}
func (*Backend) Download(context.Context, []byte, backend.Buffer) error {
	return ErrUnavailable
}
func (*Backend) Run(context.Context, *backend.Kernel, []backend.Buffer, []backend.Buffer) error {
	return ErrUnavailable
}
func (*Backend) Sync(context.Context) error { return ErrUnavailable }
