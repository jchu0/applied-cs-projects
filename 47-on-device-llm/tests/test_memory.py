"""Tests for the memory module: MemoryPool, KVCache, TensorCache.

These cover the public memory-management API documented in the README
(aligned bump allocator, sliding-window KV cache, LRU tensor cache) that
was previously only exercised indirectly via fixtures.
"""

import numpy as np
import pytest

from on_device_llm.memory import MemoryPool, KVCache, TensorCache


# =============================================================================
# MemoryPool
# =============================================================================


class TestMemoryPool:
    def test_allocate_returns_view_of_requested_size(self):
        pool = MemoryPool(size_bytes=4096, alignment=64)
        buf = pool.allocate(128, name="a")
        assert buf.nbytes == 128
        assert pool.used_bytes >= 128

    def test_allocation_is_aligned(self):
        pool = MemoryPool(size_bytes=4096, alignment=64)
        pool.allocate(1, name="a")  # leaves next_offset at 1, off the boundary
        pool.allocate(1, name="b")  # must round up to the next 64 boundary
        block_b = [b for b in pool._blocks if b.name == "b"][0]
        assert block_b.offset % pool.alignment == 0
        assert block_b.offset >= 64

    def test_allocate_tensor_shape_and_dtype(self):
        pool = MemoryPool(size_bytes=1 << 16)
        t = pool.allocate_tensor((4, 8), dtype=np.float32, name="t")
        assert t.shape == (4, 8)
        assert t.dtype == np.float32

    def test_exhaustion_raises_memory_error(self):
        pool = MemoryPool(size_bytes=256, alignment=1)
        pool.allocate(200, name="big")
        with pytest.raises(MemoryError):
            pool.allocate(200, name="too_big")

    def test_free_and_reuse_from_free_list(self):
        pool = MemoryPool(size_bytes=256, alignment=1)
        pool.allocate(200, name="big")
        assert pool.free("big") is True
        # Now the freed block should satisfy a request that would otherwise
        # exceed the remaining bump space.
        reused = pool.allocate(150, name="reused")
        assert reused.nbytes == 150

    def test_free_unknown_name_returns_false(self):
        pool = MemoryPool(size_bytes=256)
        assert pool.free("does-not-exist") is False

    def test_reset_clears_state(self):
        pool = MemoryPool(size_bytes=1024)
        pool.allocate(128, name="a")
        pool.reset()
        assert pool.used_bytes == 0
        assert pool.free_bytes == 1024
        assert pool.stats()["total_blocks"] == 0

    def test_peak_usage_and_stats(self):
        pool = MemoryPool(size_bytes=1024, alignment=1)
        pool.allocate(100, name="a")
        pool.allocate(100, name="b")
        stats = pool.stats()
        assert stats["total_bytes"] == 1024
        assert stats["active_blocks"] == 2
        assert stats["total_allocations"] == 2
        assert pool.peak_usage >= 200


# =============================================================================
# KVCache
# =============================================================================


class TestKVCache:
    def _cache(self, **kw):
        params = dict(num_layers=2, num_heads=2, head_dim=4, max_seq_len=8)
        params.update(kw)
        return KVCache(**params)

    def test_append_advances_length_after_last_layer(self):
        cache = self._cache()
        k = np.ones((2, 4), dtype=np.float32)
        v = np.ones((2, 4), dtype=np.float32) * 2
        cache.append(0, k, v)
        assert cache.length == 0  # not last layer yet
        cache.append(1, k, v)
        assert cache.length == 1  # last layer -> advance

    def test_get_returns_correct_length_and_values(self):
        cache = self._cache()
        for step in range(3):
            for layer in range(2):
                k = np.full((2, 4), step, dtype=np.float32)
                v = np.full((2, 4), step + 10, dtype=np.float32)
                cache.append(layer, k, v)
        keys, values = cache.get(0)
        assert keys.shape == (3, 2, 4)
        assert values.shape == (3, 2, 4)
        np.testing.assert_array_equal(keys[2], np.full((2, 4), 2))
        np.testing.assert_array_equal(values[0], np.full((2, 4), 10))

    def test_sliding_window_caps_and_shifts(self):
        cache = self._cache(window_size=2)
        for step in range(4):
            for layer in range(2):
                k = np.full((2, 4), step, dtype=np.float32)
                cache.append(layer, k, k)
        assert cache.effective_length == 2
        keys, _ = cache.get(0)
        assert keys.shape[0] == 2
        # Window keeps the most recent entries (steps 2 and 3).
        np.testing.assert_array_equal(keys[-1], np.full((2, 4), 3))

    def test_clear_resets_length_and_data(self):
        cache = self._cache()
        cache.append(0, np.ones((2, 4), dtype=np.float32), np.ones((2, 4), dtype=np.float32))
        cache.append(1, np.ones((2, 4), dtype=np.float32), np.ones((2, 4), dtype=np.float32))
        cache.clear()
        assert cache.length == 0
        assert np.all(cache.k_cache == 0)

    def test_memory_usage_matches_arrays(self):
        cache = self._cache()
        assert cache.memory_usage() == cache.k_cache.nbytes + cache.v_cache.nbytes

    def test_resize_grow_preserves_data(self):
        cache = self._cache(max_seq_len=4)
        k = np.full((2, 4), 7, dtype=np.float32)
        cache.append(0, k, k)
        cache.append(1, k, k)
        cache.resize(16)
        assert cache.max_seq_len == 16
        keys, _ = cache.get(0)
        np.testing.assert_array_equal(keys[0], np.full((2, 4), 7))

    def test_resize_shrink_updates_limit(self):
        cache = self._cache(max_seq_len=8)
        for layer in range(2):
            cache.append(layer, np.ones((2, 4), dtype=np.float32), np.ones((2, 4), dtype=np.float32))
        cache.resize(1)
        assert cache.max_seq_len == 1
        assert cache.length <= 1

    def test_get_layer_slice(self):
        cache = self._cache()
        for step in range(3):
            for layer in range(2):
                k = np.full((2, 4), step, dtype=np.float32)
                cache.append(layer, k, k)
        keys, values = cache.get_layer_slice(0, 1, 3)
        assert keys.shape == (2, 2, 4)
        np.testing.assert_array_equal(keys[0], np.full((2, 4), 1))


# =============================================================================
# TensorCache (LRU)
# =============================================================================


class TestTensorCache:
    def test_put_and_get(self):
        cache = TensorCache(max_entries=4)
        t = np.arange(10, dtype=np.float32)
        cache.put("a", t)
        got = cache.get("a")
        np.testing.assert_array_equal(got, t)

    def test_get_missing_returns_none(self):
        cache = TensorCache()
        assert cache.get("missing") is None

    def test_lru_eviction_by_entry_count(self):
        cache = TensorCache(max_entries=2)
        cache.put("a", np.zeros(1, dtype=np.float32))
        cache.put("b", np.zeros(1, dtype=np.float32))
        # Touch "a" so "b" becomes the least-recently-used entry.
        cache.get("a")
        cache.put("c", np.zeros(1, dtype=np.float32))
        assert cache.get("b") is None
        assert cache.get("a") is not None
        assert cache.get("c") is not None

    def test_byte_budget_eviction(self):
        # Each tensor is 400 bytes; cap at 800 -> only two fit.
        cache = TensorCache(max_entries=100, max_bytes=800)
        cache.put("a", np.zeros(100, dtype=np.float32))
        cache.put("b", np.zeros(100, dtype=np.float32))
        cache.put("c", np.zeros(100, dtype=np.float32))
        stats = cache.stats()
        assert stats["total_bytes"] <= 800
        assert stats["entries"] == 2

    def test_reput_same_key_updates_without_double_counting(self):
        # Regression: re-putting an existing key previously duplicated the
        # key in the access order and double-counted total_bytes.
        cache = TensorCache(max_entries=10)
        t = np.zeros(100, dtype=np.float32)  # 400 bytes
        cache.put("a", t)
        cache.put("a", t)
        stats = cache.stats()
        assert stats["entries"] == 1
        assert stats["total_bytes"] == 400
        assert cache._access_order == ["a"]

    def test_reput_with_larger_tensor_tracks_bytes(self):
        cache = TensorCache(max_entries=10)
        cache.put("a", np.zeros(10, dtype=np.float32))   # 40 bytes
        cache.put("a", np.zeros(100, dtype=np.float32))  # 400 bytes
        assert cache.stats()["total_bytes"] == 400
        assert cache.get("a").shape == (100,)

    def test_clear(self):
        cache = TensorCache()
        cache.put("a", np.zeros(10, dtype=np.float32))
        cache.clear()
        assert cache.get("a") is None
        assert cache.stats()["entries"] == 0
        assert cache.stats()["total_bytes"] == 0
