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

package diskcache

import "testing"

// TestCacheRoundTrips checks that a compiled kernel stored in the cache
// comes back as it was.
func TestCacheRoundTrips(t *testing.T) {
	t.Setenv("MIMIR_CACHE_DIR", t.TempDir())
	c := Open("cuda")
	if c == nil {
		t.Fatal("no cache")
	}
	key := c.Key("nvrtc 12.9", []string{"--gpu-architecture=sm_75"}, "kernel")
	if _, ok := c.Get(key); ok {
		t.Fatal("entry before put")
	}
	c.Put(key, []byte("image"))
	if got, ok := c.Get(key); !ok || string(got) != "image" {
		t.Fatalf("got %q, %t", got, ok)
	}
	if other := c.Key("nvrtc 12.9", []string{"--gpu-architecture=sm_86"}, "kernel"); other == key {
		t.Error("the key ignores the options")
	}
	if _, ok := Open("rocm").Get(key); ok {
		t.Error("another backend's cache holds the entry")
	}
}

// TestNilCacheHoldsNothing checks that a cache that could not be opened
// misses and drops what it is given.
func TestNilCacheHoldsNothing(t *testing.T) {
	var c *Cache
	c.Put("k", []byte("image"))
	if _, ok := c.Get("k"); ok {
		t.Error("a nil cache holds an entry")
	}
}
