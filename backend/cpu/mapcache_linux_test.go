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

package cpu

import (
	"syscall"
	"testing"

	"github.com/gsoultan/mimir/backend"
)

func TestReusedMappingsTakeNoPageFaults(t *testing.T) {
	b := New()
	const n = 16 << 20 / 4
	faults := func() int64 {
		var ru syscall.Rusage
		if err := syscall.Getrusage(syscall.RUSAGE_SELF, &ru); err != nil {
			t.Fatal(err)
		}
		return ru.Minflt
	}
	touch := func() int64 {
		before := faults()
		buf, err := b.Alloc(backend.Float32, n)
		if err != nil {
			t.Fatal(err)
		}
		_, data := dataOf(buf)
		for i := 0; i < len(data); i += pageSize / 4 {
			data[i] = 1
		}
		b.Free(buf)
		return faults() - before
	}
	touch()
	reused := touch()
	t.Logf("%d page faults to allocate and fill %d MiB from the cache", reused, 4*n>>20)
	if pages := int64(4 * n / pageSize); reused > pages/8 {
		t.Errorf("%d page faults over a reused mapping of %d pages", reused, pages)
	}
}
