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

// Package metal implements the mimir backend for Apple GPUs through
// Metal, without cgo.
//
// It calls the Objective-C runtime and Metal through purego and runs every
// kernel as Metal Shading Language source that it generates and compiles
// at runtime, once per kernel structure. Buffers live in memory the CPU
// and GPU share, so uploads and downloads are copies within RAM.
//
// The backend is available on Apple silicon Macs. Elsewhere New reports
// ErrUnavailable.
package metal

import "errors"

// ErrUnavailable reports that no Metal device can be used.
var ErrUnavailable = errors.New("metal unavailable")
