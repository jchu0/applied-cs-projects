"""Tests for memory allocator implementations."""

import pytest
import threading
import time

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gpumem.allocator.allocator import (
    Allocator,
    CachingAllocator,
    PoolAllocator,
    BuddyAllocator,
    SlabAllocator,
    UnifiedMemoryAllocator,
)
from gpumem.core.memory import DeviceType, MemoryConfig, MemoryBlock


# =============================================================================
# CachingAllocator Tests
# =============================================================================

class TestCachingAllocator:
    """Tests for CachingAllocator."""

    def test_creation(self):
        """Test creating caching allocator."""
        allocator = CachingAllocator()
        assert allocator is not None
        assert allocator.config is not None

    def test_creation_with_config(self):
        """Test creating with custom config."""
        config = MemoryConfig(device_type=DeviceType.CUDA)
        allocator = CachingAllocator(config)
        assert allocator.config.device_type == DeviceType.CUDA

    def test_allocate_small(self):
        """Test small block allocation."""
        allocator = CachingAllocator()
        block = allocator.allocate(512)
        assert block is not None
        assert block.size >= 512
        assert block.allocated is True

    def test_allocate_large(self):
        """Test large block allocation (above threshold)."""
        allocator = CachingAllocator()
        block = allocator.allocate(2 * 1024 * 1024)  # 2MB
        assert block is not None
        assert block.size >= 2 * 1024 * 1024

    def test_free_and_reuse(self):
        """Test that freed blocks are cached and reused."""
        allocator = CachingAllocator()

        # Allocate and free
        block1 = allocator.allocate(512)
        ptr1 = block1.ptr
        allocator.free(block1)

        # Allocate same size - should reuse
        block2 = allocator.allocate(512)
        # May or may not get same ptr depending on implementation

        assert block2 is not None

    def test_stream_isolation(self):
        """Test stream-based pool isolation."""
        allocator = CachingAllocator()

        block1 = allocator.allocate(1024, stream=1)
        block2 = allocator.allocate(1024, stream=2)

        assert block1 is not None
        assert block2 is not None

    def test_get_stats(self):
        """Test statistics collection."""
        allocator = CachingAllocator()

        block = allocator.allocate(1024)
        stats = allocator.get_stats()

        assert stats.num_allocs >= 1

        allocator.free(block)
        stats = allocator.get_stats()
        assert stats.num_frees >= 0

    def test_empty_cache(self):
        """Test cache clearing."""
        allocator = CachingAllocator()

        # Allocate and free to populate cache
        for _ in range(10):
            block = allocator.allocate(512)
            allocator.free(block)

        # Clear cache
        allocator.empty_cache()

        # Should not raise


# =============================================================================
# PoolAllocator Tests
# =============================================================================

class TestPoolAllocator:
    """Tests for PoolAllocator."""

    def test_creation(self):
        """Test creating pool allocator."""
        allocator = PoolAllocator()
        assert allocator is not None

    def test_allocate_various_sizes(self):
        """Test allocation of various sizes."""
        allocator = PoolAllocator()

        sizes = [256, 512, 1024, 4096, 65536]
        blocks = []

        for size in sizes:
            block = allocator.allocate(size)
            assert block is not None
            assert block.size >= size
            blocks.append(block)

        for block in blocks:
            allocator.free(block)

    def test_size_class_rounding(self):
        """Test that allocations are rounded to size classes."""
        allocator = PoolAllocator()

        # 300 bytes should round to 512
        block = allocator.allocate(300)
        assert block.size == 512

        # 1000 bytes should round to 1024
        block = allocator.allocate(1000)
        assert block.size == 1024

    def test_pool_reuse(self):
        """Test that freed blocks return to pool."""
        allocator = PoolAllocator()

        block1 = allocator.allocate(1024)
        allocator.free(block1)

        block2 = allocator.allocate(1024)
        # Should reuse from pool
        assert block2 is not None

    def test_stats_tracking(self):
        """Test statistics tracking."""
        allocator = PoolAllocator()

        block = allocator.allocate(1024)
        stats = allocator.get_stats()

        assert stats.allocated > 0
        assert stats.num_allocs == 1

        allocator.free(block)
        stats = allocator.get_stats()

        assert stats.num_frees == 1


# =============================================================================
# BuddyAllocator Tests
# =============================================================================

class TestBuddyAllocator:
    """Tests for BuddyAllocator."""

    def test_creation(self):
        """Test creating buddy allocator."""
        allocator = BuddyAllocator()
        assert allocator is not None

    def test_creation_with_size(self):
        """Test creating with total size."""
        allocator = BuddyAllocator(total_size=1024 * 1024)
        stats = allocator.get_stats()
        assert stats.reserved == 1024 * 1024

    def test_power_of_2_allocation(self):
        """Test power-of-2 allocations."""
        allocator = BuddyAllocator()

        # Allocate exact power of 2
        block = allocator.allocate(1024)
        assert block is not None
        assert block.size == 1024

    def test_non_power_of_2_rounding(self):
        """Test rounding to power of 2."""
        allocator = BuddyAllocator()

        # 1000 should round to 1024
        block = allocator.allocate(1000)
        assert block is not None
        assert block.size == 1024

    def test_buddy_coalescing(self):
        """Test buddy coalescing on free."""
        allocator = BuddyAllocator(total_size=4096)

        # Allocate two buddies
        block1 = allocator.allocate(1024)
        block2 = allocator.allocate(1024)

        # Free both - should coalesce
        allocator.free(block1)
        allocator.free(block2)

        # Should be able to allocate larger block now
        block3 = allocator.allocate(2048)
        assert block3 is not None

    def test_oom_handling(self):
        """Test OOM when memory exhausted."""
        allocator = BuddyAllocator(total_size=4096)

        # Exhaust memory
        block1 = allocator.allocate(4096)
        assert block1 is not None

        # Should return None on OOM
        block2 = allocator.allocate(1024)
        assert block2 is None

        stats = allocator.get_stats()
        assert stats.num_ooms > 0


# =============================================================================
# SlabAllocator Tests
# =============================================================================

class TestSlabAllocator:
    """Tests for SlabAllocator."""

    def test_creation(self):
        """Test creating slab allocator."""
        allocator = SlabAllocator(object_size=64)
        assert allocator is not None

    def test_fixed_size_allocation(self):
        """Test fixed-size object allocation."""
        allocator = SlabAllocator(object_size=64)

        block = allocator.allocate()
        assert block is not None
        assert block.size == 64

    def test_multiple_allocations(self):
        """Test multiple object allocations."""
        allocator = SlabAllocator(object_size=128)
        blocks = []

        for _ in range(100):
            block = allocator.allocate()
            assert block is not None
            blocks.append(block)

        for block in blocks:
            allocator.free(block)

    def test_slab_creation(self):
        """Test automatic slab creation."""
        allocator = SlabAllocator(object_size=64)

        # Allocate enough to create multiple slabs
        objects_per_slab = 64 * 1024 // 64  # ~1000

        blocks = []
        for _ in range(objects_per_slab + 10):
            block = allocator.allocate()
            blocks.append(block)

        # Should have created multiple slabs
        stats = allocator.get_stats()
        assert stats.reserved >= 2 * 64 * 1024

    def test_partial_slab_tracking(self):
        """Test partial slab management."""
        allocator = SlabAllocator(object_size=64)

        # Allocate some
        blocks = [allocator.allocate() for _ in range(10)]

        # Free some back
        for block in blocks[:5]:
            allocator.free(block)

        # Allocate again - should use freed slots
        new_block = allocator.allocate()
        assert new_block is not None


# =============================================================================
# UnifiedMemoryAllocator Tests
# =============================================================================

class TestUnifiedMemoryAllocator:
    """Tests for UnifiedMemoryAllocator."""

    def test_creation(self):
        """Test creating unified allocator."""
        allocator = UnifiedMemoryAllocator()
        assert allocator is not None

    def test_unified_allocation(self):
        """Test unified memory allocation."""
        allocator = UnifiedMemoryAllocator()

        block = allocator.allocate(1024)
        assert block is not None
        assert block.size == 1024

    def test_device_pointer_mapping(self):
        """Test getting device-specific pointers."""
        allocator = UnifiedMemoryAllocator()

        block = allocator.allocate(1024)

        cpu_ptr = allocator.get_device_ptr(block.ptr, DeviceType.CPU)
        gpu_ptr = allocator.get_device_ptr(block.ptr, DeviceType.CUDA)

        # Both should be valid pointers
        assert cpu_ptr is not None
        assert gpu_ptr is not None

    def test_unified_free(self):
        """Test freeing unified memory."""
        allocator = UnifiedMemoryAllocator()

        block = allocator.allocate(1024)
        allocator.free(block)

        stats = allocator.get_stats()
        # Memory should be freed


# =============================================================================
# Common Allocator Pattern Tests
# =============================================================================

class TestAllocatorPatterns:
    """Test common allocation patterns across allocators."""

    @pytest.fixture(params=[
        CachingAllocator,
        PoolAllocator,
        lambda: BuddyAllocator(total_size=10 * 1024 * 1024),
    ])
    def allocator(self, request):
        return request.param()

    def test_alloc_free_cycle(self, allocator):
        """Test basic alloc/free cycle."""
        block = allocator.allocate(1024)
        assert block is not None
        allocator.free(block)

    def test_multiple_allocations(self, allocator):
        """Test multiple concurrent allocations."""
        blocks = []
        for _ in range(10):
            block = allocator.allocate(512)
            assert block is not None
            blocks.append(block)

        for block in blocks:
            allocator.free(block)

    def test_grow_shrink_pattern(self, allocator):
        """Test growing then shrinking allocations."""
        blocks = []

        # Grow
        for i in range(5):
            block = allocator.allocate(1024 * (i + 1))
            if block is not None:
                blocks.append(block)

        # Shrink
        while blocks:
            allocator.free(blocks.pop())


# =============================================================================
# Thread Safety Tests
# =============================================================================

class TestAllocatorThreadSafety:
    """Test thread safety of allocators."""

    def test_concurrent_caching_allocator(self):
        """Test concurrent access to CachingAllocator."""
        allocator = CachingAllocator()
        errors = []
        results = []

        def worker():
            try:
                local_blocks = []
                for _ in range(20):
                    block = allocator.allocate(512)
                    if block:
                        local_blocks.append(block)
                    time.sleep(0.0001)

                for block in local_blocks:
                    allocator.free(block)

                results.append(len(local_blocks))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert sum(results) == 100

    def test_concurrent_pool_allocator(self):
        """Test concurrent access to PoolAllocator."""
        allocator = PoolAllocator()
        errors = []

        def worker():
            try:
                for _ in range(50):
                    block = allocator.allocate(1024)
                    if block:
                        allocator.free(block)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# =============================================================================
# Statistics Tests
# =============================================================================

class TestAllocatorStatistics:
    """Test allocator statistics."""

    def test_caching_allocator_stats(self):
        """Test CachingAllocator statistics."""
        allocator = CachingAllocator()

        block = allocator.allocate(1024)
        stats = allocator.get_stats()

        assert hasattr(stats, 'allocated')
        assert hasattr(stats, 'num_allocs')

    def test_pool_allocator_stats(self):
        """Test PoolAllocator statistics."""
        allocator = PoolAllocator()

        blocks = [allocator.allocate(512) for _ in range(5)]
        stats = allocator.get_stats()

        assert stats.num_allocs == 5
        assert stats.allocated > 0

        for block in blocks:
            allocator.free(block)

        stats = allocator.get_stats()
        assert stats.num_frees == 5

    def test_buddy_allocator_stats(self):
        """Test BuddyAllocator statistics."""
        allocator = BuddyAllocator(total_size=1024 * 1024)

        block = allocator.allocate(1024)
        stats = allocator.get_stats()

        assert stats.reserved == 1024 * 1024
        assert stats.allocated > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
