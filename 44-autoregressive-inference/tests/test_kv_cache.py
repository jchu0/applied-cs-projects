"""Tests for KV cache functionality.

This module tests:
- KVCacheConfig creation and validation
- KVCacheBlock operations (append, get, eviction)
- PagedKVCacheManager (allocation, freeing, memory tracking)
- SlidingWindowCache (circular buffer operations)
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from autoregressive_inference.kv_cache import (
    KVCacheConfig,
    KVCacheBlock,
    PagedKVCacheManager,
    SlidingWindowCache,
)

# Try to import torch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestKVCacheConfig:
    """Tests for KVCacheConfig."""

    def test_config_creation(self, small_kv_config):
        """Test basic config creation."""
        assert small_kv_config.num_layers == 2
        assert small_kv_config.num_heads == 4
        assert small_kv_config.head_dim == 64
        assert small_kv_config.block_size == 16

    def test_block_memory_calculation(self, small_kv_config):
        """Test memory calculation per block."""
        # 2 layers * 2 (KV) * 4 heads * 16 tokens * 64 dim * 4 bytes (float32)
        expected = 2 * 2 * 4 * 16 * 64 * 4
        assert small_kv_config.get_block_memory_bytes() == expected

    def test_config_float16_memory(self):
        """Test memory calculation with float16."""
        config = KVCacheConfig(
            num_layers=2,
            num_heads=4,
            head_dim=64,
            max_seq_len=256,
            block_size=16,
            dtype="float16"
        )
        # 2 layers * 2 (KV) * 4 heads * 16 tokens * 64 dim * 2 bytes (float16)
        expected = 2 * 2 * 4 * 16 * 64 * 2
        assert config.get_block_memory_bytes() == expected


class TestKVCacheBlock:
    """Tests for KVCacheBlock operations."""

    def test_block_creation(self, kv_cache_block, small_kv_config):
        """Test block initialization."""
        assert kv_cache_block.block_id == 0
        assert kv_cache_block.num_tokens == 0
        assert kv_cache_block.ref_count == 0
        assert not kv_cache_block.is_full()

    def test_block_data_shape(self, kv_cache_block, small_kv_config):
        """Test that block data has correct shape."""
        expected_shape = (
            small_kv_config.num_layers,
            2,  # K and V
            small_kv_config.num_heads,
            small_kv_config.block_size,
            small_kv_config.head_dim
        )
        assert kv_cache_block.data.shape == expected_shape

    def test_append_single_token(self, kv_cache_block, numpy_kv_tensors, small_kv_config):
        """Test appending a single token's KV."""
        k, v = numpy_kv_tensors

        # Append to all layers
        for layer_idx in range(small_kv_config.num_layers):
            pos = kv_cache_block.append(layer_idx, k, v)
            if layer_idx == small_kv_config.num_layers - 1:
                assert pos == 0  # First token

        assert kv_cache_block.num_tokens == 1

    def test_append_multiple_tokens(self, kv_cache_block, numpy_kv_tensors, small_kv_config):
        """Test appending multiple tokens."""
        k, v = numpy_kv_tensors

        for token_idx in range(5):
            for layer_idx in range(small_kv_config.num_layers):
                kv_cache_block.append(layer_idx, k, v)

        assert kv_cache_block.num_tokens == 5

    def test_append_until_full(self, kv_cache_block, numpy_kv_tensors, small_kv_config):
        """Test appending tokens until block is full."""
        k, v = numpy_kv_tensors

        for token_idx in range(small_kv_config.block_size):
            for layer_idx in range(small_kv_config.num_layers):
                kv_cache_block.append(layer_idx, k, v)

        assert kv_cache_block.is_full()
        assert kv_cache_block.num_tokens == small_kv_config.block_size

    def test_append_to_full_block_raises(self, kv_cache_block, numpy_kv_tensors, small_kv_config):
        """Test that appending to full block raises error."""
        k, v = numpy_kv_tensors

        # Fill the block
        for token_idx in range(small_kv_config.block_size):
            for layer_idx in range(small_kv_config.num_layers):
                kv_cache_block.append(layer_idx, k, v)

        # Try to append one more
        with pytest.raises(RuntimeError, match="Block is full"):
            kv_cache_block.append(0, k, v)

    def test_get_kv(self, kv_cache_block, numpy_kv_tensors, small_kv_config):
        """Test retrieving KV from block."""
        k, v = numpy_kv_tensors

        # Append 3 tokens
        for token_idx in range(3):
            for layer_idx in range(small_kv_config.num_layers):
                kv_cache_block.append(layer_idx, k, v)

        # Get KV for layer 0
        k_out, v_out = kv_cache_block.get_kv(0)

        assert k_out.shape == (small_kv_config.num_heads, 3, small_kv_config.head_dim)
        assert v_out.shape == (small_kv_config.num_heads, 3, small_kv_config.head_dim)

    def test_get_kv_at_position(self, kv_cache_block, small_kv_config):
        """Test retrieving KV at specific position."""
        np.random.seed(42)

        # Append tokens with different values
        for token_idx in range(3):
            k = np.full((small_kv_config.num_heads, small_kv_config.head_dim),
                       token_idx, dtype=np.float32)
            v = np.full((small_kv_config.num_heads, small_kv_config.head_dim),
                       token_idx * 10, dtype=np.float32)

            for layer_idx in range(small_kv_config.num_layers):
                kv_cache_block.append(layer_idx, k, v)

        # Check position 1
        k_out, v_out = kv_cache_block.get_kv_at_position(0, 1)
        assert np.allclose(k_out, 1.0)
        assert np.allclose(v_out, 10.0)

    def test_block_reset(self, kv_cache_block, numpy_kv_tensors, small_kv_config):
        """Test resetting a block."""
        k, v = numpy_kv_tensors

        # Fill with some data
        for layer_idx in range(small_kv_config.num_layers):
            kv_cache_block.append(layer_idx, k, v)

        kv_cache_block.ref_count = 2

        # Reset
        kv_cache_block.reset()

        assert kv_cache_block.num_tokens == 0
        assert kv_cache_block.ref_count == 0
        assert not kv_cache_block.is_full()

    def test_append_token_all_layers(self, kv_cache_block, small_kv_config):
        """Test appending a full token across all layers at once."""
        k_by_layer = [
            np.random.randn(small_kv_config.num_heads, small_kv_config.head_dim).astype(np.float32)
            for _ in range(small_kv_config.num_layers)
        ]
        v_by_layer = [
            np.random.randn(small_kv_config.num_heads, small_kv_config.head_dim).astype(np.float32)
            for _ in range(small_kv_config.num_layers)
        ]

        pos = kv_cache_block.append_token(k_by_layer, v_by_layer)

        assert pos == 0
        assert kv_cache_block.num_tokens == 1


class TestPagedKVCacheManager:
    """Tests for PagedKVCacheManager."""

    def test_manager_creation(self, paged_kv_cache):
        """Test manager initialization."""
        assert paged_kv_cache.num_blocks == 10
        assert len(paged_kv_cache.free_blocks) == 10
        assert len(paged_kv_cache.blocks) == 10
        assert len(paged_kv_cache.request_blocks) == 0

    def test_allocate_blocks(self, paged_kv_cache):
        """Test block allocation."""
        blocks = paged_kv_cache.allocate_blocks("req-1", 3)

        assert len(blocks) == 3
        assert len(paged_kv_cache.free_blocks) == 7
        assert "req-1" in paged_kv_cache.request_blocks
        assert len(paged_kv_cache.request_blocks["req-1"]) == 3

    def test_allocate_multiple_requests(self, paged_kv_cache):
        """Test allocating blocks for multiple requests."""
        blocks1 = paged_kv_cache.allocate_blocks("req-1", 3)
        blocks2 = paged_kv_cache.allocate_blocks("req-2", 2)

        assert len(blocks1) == 3
        assert len(blocks2) == 2
        assert len(paged_kv_cache.free_blocks) == 5

        # Verify blocks are different
        assert set(blocks1).isdisjoint(set(blocks2))

    def test_allocate_insufficient_blocks(self, paged_kv_cache):
        """Test allocation when not enough blocks available."""
        # Allocate most blocks
        paged_kv_cache.allocate_blocks("req-1", 8)

        # Try to allocate more than available
        blocks = paged_kv_cache.allocate_blocks("req-2", 5)

        assert blocks == []  # Should return empty list
        assert len(paged_kv_cache.free_blocks) == 2

    def test_allocate_single_block(self, paged_kv_cache):
        """Test single block allocation."""
        block_id = paged_kv_cache.allocate_single_block("req-1")

        assert block_id is not None
        assert len(paged_kv_cache.request_blocks["req-1"]) == 1

    def test_free_blocks(self, paged_kv_cache):
        """Test freeing blocks for a request."""
        paged_kv_cache.allocate_blocks("req-1", 3)
        initial_free = len(paged_kv_cache.free_blocks)

        freed_count = paged_kv_cache.free_blocks_for_request("req-1")

        assert freed_count == 3
        assert len(paged_kv_cache.free_blocks) == initial_free + 3
        assert "req-1" not in paged_kv_cache.request_blocks

    def test_free_nonexistent_request(self, paged_kv_cache):
        """Test freeing blocks for nonexistent request."""
        freed_count = paged_kv_cache.free_blocks_for_request("nonexistent")
        assert freed_count == 0

    def test_get_block(self, paged_kv_cache):
        """Test getting a block by ID."""
        blocks = paged_kv_cache.allocate_blocks("req-1", 1)
        block = paged_kv_cache.get_block(blocks[0])

        assert block.block_id == blocks[0]
        assert block.ref_count == 1

    def test_get_blocks_for_request(self, paged_kv_cache):
        """Test getting all blocks for a request."""
        allocated = paged_kv_cache.allocate_blocks("req-1", 3)
        retrieved = paged_kv_cache.get_blocks_for_request("req-1")

        assert retrieved == allocated

    def test_can_allocate(self, paged_kv_cache):
        """Test can_allocate check."""
        assert paged_kv_cache.can_allocate(5)
        assert paged_kv_cache.can_allocate(10)
        assert not paged_kv_cache.can_allocate(11)

        paged_kv_cache.allocate_blocks("req-1", 8)
        assert paged_kv_cache.can_allocate(2)
        assert not paged_kv_cache.can_allocate(3)

    def test_append_to_cache(self, paged_kv_cache, numpy_kv_tensors, small_kv_config):
        """Test appending KV to cache."""
        k, v = numpy_kv_tensors

        # Allocate blocks first
        paged_kv_cache.allocate_blocks("req-1", 2)

        # Append to first layer
        success = paged_kv_cache.append_to_cache("req-1", 0, k, v)
        assert success

    def test_append_auto_allocate(self, paged_kv_cache, numpy_kv_tensors):
        """Test appending to cache auto-allocates if needed."""
        k, v = numpy_kv_tensors

        # Don't allocate first - should auto-allocate
        success = paged_kv_cache.append_to_cache("req-1", 0, k, v)
        assert success
        assert "req-1" in paged_kv_cache.request_blocks

    def test_append_needs_new_block(self, paged_kv_cache, numpy_kv_tensors, small_kv_config):
        """Test that new blocks are allocated when current is full."""
        k, v = numpy_kv_tensors

        # Allocate one block
        paged_kv_cache.allocate_blocks("req-1", 1)

        # Fill the block
        for token_idx in range(small_kv_config.block_size):
            for layer_idx in range(small_kv_config.num_layers):
                paged_kv_cache.append_to_cache("req-1", layer_idx, k, v)

        # Add one more - should allocate new block
        initial_blocks = len(paged_kv_cache.request_blocks["req-1"])
        paged_kv_cache.append_to_cache("req-1", 0, k, v)

        assert len(paged_kv_cache.request_blocks["req-1"]) == initial_blocks + 1

    def test_get_kv_for_request(self, paged_kv_cache, small_kv_config):
        """Test retrieving concatenated KV for a request."""
        np.random.seed(42)

        # Allocate and fill with known data
        paged_kv_cache.allocate_blocks("req-1", 2)

        for token_idx in range(5):
            k = np.full((small_kv_config.num_heads, small_kv_config.head_dim),
                       token_idx, dtype=np.float32)
            v = np.full((small_kv_config.num_heads, small_kv_config.head_dim),
                       token_idx * 10, dtype=np.float32)

            for layer_idx in range(small_kv_config.num_layers):
                paged_kv_cache.append_to_cache("req-1", layer_idx, k, v)

        # Get KV
        k_out, v_out = paged_kv_cache.get_kv_for_request("req-1", 0)

        assert k_out is not None
        assert v_out is not None
        assert k_out.shape == (small_kv_config.num_heads, 5, small_kv_config.head_dim)

    def test_get_kv_nonexistent_request(self, paged_kv_cache):
        """Test getting KV for nonexistent request."""
        k, v = paged_kv_cache.get_kv_for_request("nonexistent", 0)
        assert k is None
        assert v is None

    def test_get_total_tokens(self, paged_kv_cache, numpy_kv_tensors, small_kv_config):
        """Test getting total cached tokens for a request."""
        k, v = numpy_kv_tensors

        paged_kv_cache.allocate_blocks("req-1", 2)

        for token_idx in range(7):
            for layer_idx in range(small_kv_config.num_layers):
                paged_kv_cache.append_to_cache("req-1", layer_idx, k, v)

        total = paged_kv_cache.get_total_tokens_for_request("req-1")
        assert total == 7

    def test_memory_usage_stats(self, paged_kv_cache):
        """Test memory usage statistics."""
        stats = paged_kv_cache.get_memory_usage()

        assert stats['total_blocks'] == 10
        assert stats['free_blocks'] == 10
        assert stats['used_blocks'] == 0
        assert stats['utilization'] == 0.0

        paged_kv_cache.allocate_blocks("req-1", 5)
        stats = paged_kv_cache.get_memory_usage()

        assert stats['used_blocks'] == 5
        assert stats['free_blocks'] == 5
        assert stats['utilization'] == 0.5

    def test_ref_count_management(self, paged_kv_cache):
        """Test reference count is properly managed."""
        blocks = paged_kv_cache.allocate_blocks("req-1", 3)

        for block_id in blocks:
            assert paged_kv_cache.blocks[block_id].ref_count == 1

        paged_kv_cache.free_blocks_for_request("req-1")

        for block_id in blocks:
            assert paged_kv_cache.blocks[block_id].ref_count == 0


class TestSlidingWindowCache:
    """Tests for SlidingWindowCache."""

    def test_cache_creation(self, sliding_window_cache, small_kv_config):
        """Test sliding window cache initialization."""
        assert sliding_window_cache.window_size == 32
        assert sliding_window_cache.position == 0
        assert sliding_window_cache.length == 0
        assert not sliding_window_cache.is_full()

    def test_cache_shape(self, sliding_window_cache, small_kv_config):
        """Test cache data shape."""
        expected_shape = (
            small_kv_config.num_layers,
            small_kv_config.num_heads,
            32,  # window_size
            small_kv_config.head_dim
        )
        assert sliding_window_cache.k_cache.shape == expected_shape
        assert sliding_window_cache.v_cache.shape == expected_shape

    def test_append_single_token(self, sliding_window_cache, numpy_kv_tensors, small_kv_config):
        """Test appending a single token."""
        k, v = numpy_kv_tensors

        for layer_idx in range(small_kv_config.num_layers):
            sliding_window_cache.append(layer_idx, k, v)

        assert sliding_window_cache.length == 1
        assert sliding_window_cache.position == 1

    def test_append_multiple_tokens(self, sliding_window_cache, numpy_kv_tensors, small_kv_config):
        """Test appending multiple tokens."""
        k, v = numpy_kv_tensors

        for token_idx in range(10):
            for layer_idx in range(small_kv_config.num_layers):
                sliding_window_cache.append(layer_idx, k, v)

        assert sliding_window_cache.length == 10
        assert sliding_window_cache.position == 10

    def test_window_overflow(self, sliding_window_cache, numpy_kv_tensors, small_kv_config):
        """Test that cache wraps around when full."""
        k, v = numpy_kv_tensors
        window_size = sliding_window_cache.window_size

        # Fill beyond window size
        for token_idx in range(window_size + 10):
            for layer_idx in range(small_kv_config.num_layers):
                sliding_window_cache.append(layer_idx, k, v)

        assert sliding_window_cache.length == window_size
        assert sliding_window_cache.position == window_size + 10
        assert sliding_window_cache.is_full()

    def test_get_kv_before_full(self, sliding_window_cache, small_kv_config):
        """Test getting KV before window is full."""
        np.random.seed(42)

        for token_idx in range(5):
            k = np.full((small_kv_config.num_heads, small_kv_config.head_dim),
                       token_idx, dtype=np.float32)
            v = np.full((small_kv_config.num_heads, small_kv_config.head_dim),
                       token_idx * 10, dtype=np.float32)

            for layer_idx in range(small_kv_config.num_layers):
                sliding_window_cache.append(layer_idx, k, v)

        k_out, v_out = sliding_window_cache.get_kv(0)

        assert k_out.shape == (small_kv_config.num_heads, 5, small_kv_config.head_dim)
        # Check values are in order
        assert np.allclose(k_out[:, 0, :], 0.0)
        assert np.allclose(k_out[:, 4, :], 4.0)

    def test_get_kv_after_wrap(self, sliding_window_cache, small_kv_config):
        """Test getting KV after window wraps around."""
        window_size = sliding_window_cache.window_size

        # Fill beyond window size with indexed values
        for token_idx in range(window_size + 5):
            k = np.full((small_kv_config.num_heads, small_kv_config.head_dim),
                       token_idx, dtype=np.float32)
            v = np.full((small_kv_config.num_heads, small_kv_config.head_dim),
                       token_idx, dtype=np.float32)

            for layer_idx in range(small_kv_config.num_layers):
                sliding_window_cache.append(layer_idx, k, v)

        k_out, v_out = sliding_window_cache.get_kv(0)

        assert k_out.shape == (small_kv_config.num_heads, window_size, small_kv_config.head_dim)

        # First token should be token 5 (after wrap)
        assert np.allclose(k_out[:, 0, :], 5.0)
        # Last token should be token window_size + 4
        assert np.allclose(k_out[:, -1, :], window_size + 4)

    def test_get_length(self, sliding_window_cache, numpy_kv_tensors, small_kv_config):
        """Test getting current length."""
        k, v = numpy_kv_tensors

        assert sliding_window_cache.get_length() == 0

        for token_idx in range(5):
            for layer_idx in range(small_kv_config.num_layers):
                sliding_window_cache.append(layer_idx, k, v)

        assert sliding_window_cache.get_length() == 5

    def test_get_total_position(self, sliding_window_cache, numpy_kv_tensors, small_kv_config):
        """Test getting total position (including evicted)."""
        k, v = numpy_kv_tensors
        window_size = sliding_window_cache.window_size

        for token_idx in range(window_size + 10):
            for layer_idx in range(small_kv_config.num_layers):
                sliding_window_cache.append(layer_idx, k, v)

        assert sliding_window_cache.get_total_position() == window_size + 10
        assert sliding_window_cache.get_length() == window_size

    def test_reset(self, sliding_window_cache, numpy_kv_tensors, small_kv_config):
        """Test resetting the cache."""
        k, v = numpy_kv_tensors

        for token_idx in range(10):
            for layer_idx in range(small_kv_config.num_layers):
                sliding_window_cache.append(layer_idx, k, v)

        sliding_window_cache.reset()

        assert sliding_window_cache.position == 0
        assert sliding_window_cache.length == 0
        assert not sliding_window_cache.is_full()

    def test_append_token_all_layers(self, sliding_window_cache, small_kv_config):
        """Test appending a full token across all layers."""
        k_by_layer = [
            np.random.randn(small_kv_config.num_heads, small_kv_config.head_dim).astype(np.float32)
            for _ in range(small_kv_config.num_layers)
        ]
        v_by_layer = [
            np.random.randn(small_kv_config.num_heads, small_kv_config.head_dim).astype(np.float32)
            for _ in range(small_kv_config.num_layers)
        ]

        sliding_window_cache.append_token(k_by_layer, v_by_layer)

        assert sliding_window_cache.length == 1
        assert sliding_window_cache.position == 1


class TestKVCacheEviction:
    """Tests for KV cache eviction scenarios."""

    def test_eviction_order(self, paged_kv_cache, numpy_kv_tensors, small_kv_config):
        """Test that oldest data is evicted first in sliding window."""
        # This is implicitly tested in sliding window but let's be explicit
        cache = SlidingWindowCache(small_kv_config, window_size=4)

        # Add tokens 0-5
        for token_idx in range(6):
            k = np.full((small_kv_config.num_heads, small_kv_config.head_dim),
                       token_idx, dtype=np.float32)
            v = np.full((small_kv_config.num_heads, small_kv_config.head_dim),
                       token_idx, dtype=np.float32)

            for layer_idx in range(small_kv_config.num_layers):
                cache.append(layer_idx, k, v)

        # Should have tokens 2, 3, 4, 5 (oldest evicted)
        k_out, v_out = cache.get_kv(0)

        assert np.allclose(k_out[:, 0, :], 2.0)  # First is token 2
        assert np.allclose(k_out[:, -1, :], 5.0)  # Last is token 5

    def test_memory_pressure_handling(self, small_kv_config):
        """Test behavior under memory pressure."""
        # Create a small cache
        cache = PagedKVCacheManager(small_kv_config, num_blocks=3)

        # Allocate for first request
        blocks1 = cache.allocate_blocks("req-1", 2)
        assert len(blocks1) == 2

        # Try to allocate more than available
        blocks2 = cache.allocate_blocks("req-2", 2)
        assert blocks2 == []  # Should fail

        # Free first request
        cache.free_blocks_for_request("req-1")

        # Now allocation should succeed
        blocks3 = cache.allocate_blocks("req-2", 2)
        assert len(blocks3) == 2


class TestKVCacheConcurrency:
    """Tests for thread-safety of KV cache operations."""

    def test_concurrent_allocations(self, small_kv_config):
        """Test concurrent block allocations."""
        import threading

        cache = PagedKVCacheManager(small_kv_config, num_blocks=100)
        results = []
        errors = []

        def allocate_blocks(request_id):
            try:
                blocks = cache.allocate_blocks(request_id, 5)
                results.append((request_id, blocks))
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(10):
            t = threading.Thread(target=allocate_blocks, args=(f"req-{i}",))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10

        # Verify no duplicate blocks
        all_blocks = []
        for _, blocks in results:
            all_blocks.extend(blocks)

        assert len(all_blocks) == len(set(all_blocks))

    def test_concurrent_free(self, small_kv_config):
        """Test concurrent block freeing."""
        import threading

        cache = PagedKVCacheManager(small_kv_config, num_blocks=100)

        # Allocate blocks for multiple requests
        for i in range(10):
            cache.allocate_blocks(f"req-{i}", 5)

        errors = []

        def free_blocks(request_id):
            try:
                cache.free_blocks_for_request(request_id)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(10):
            t = threading.Thread(target=free_blocks, args=(f"req-{i}",))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(cache.free_blocks) == 100
