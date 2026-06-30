"""Tests for core memory management primitives."""

import pytest
import threading
import time

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gpumem.core.memory import (
    DeviceType, MemoryConfig, MemoryStats, MemoryBlock,
    MemoryRegion, AllocationRequest, MemoryPool
)


# =============================================================================
# DeviceType Tests
# =============================================================================

class TestDeviceType:
    """Tests for DeviceType enum."""

    def test_device_types_defined(self):
        """Test all device types are defined."""
        expected_types = ['CPU', 'CUDA', 'ROCM', 'METAL', 'VULKAN']
        for device_name in expected_types:
            assert hasattr(DeviceType, device_name)

    def test_device_type_unique_values(self):
        """Test device type enum values are unique."""
        values = set()
        for device_type in DeviceType:
            assert device_type.value not in values
            values.add(device_type.value)

    def test_device_type_iteration(self):
        """Test iterating over device types."""
        types = list(DeviceType)
        assert len(types) == 5


# =============================================================================
# MemoryConfig Tests
# =============================================================================

class TestMemoryConfig:
    """Tests for MemoryConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MemoryConfig()

        assert config.device_type == DeviceType.CPU
        assert config.device_id == 0
        assert config.max_memory == 0  # Unlimited
        assert config.min_split_size == 512
        assert config.max_split_size == 200 * 1024 * 1024
        assert config.garbage_collection_threshold == 0.8
        assert config.expandable_segments is True
        assert config.release_threshold == 0.95
        assert config.alignment == 512
        assert config.round_up_power2 is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = MemoryConfig(
            device_type=DeviceType.CUDA,
            device_id=2,
            max_memory=4 * 1024 * 1024 * 1024,  # 4GB
            min_split_size=1024,
            alignment=256
        )

        assert config.device_type == DeviceType.CUDA
        assert config.device_id == 2
        assert config.max_memory == 4 * 1024 * 1024 * 1024
        assert config.min_split_size == 1024
        assert config.alignment == 256

    def test_config_thresholds(self):
        """Test configuration threshold values."""
        config = MemoryConfig(
            garbage_collection_threshold=0.5,
            release_threshold=0.7
        )

        assert config.garbage_collection_threshold == 0.5
        assert config.release_threshold == 0.7


# =============================================================================
# MemoryStats Tests
# =============================================================================

class TestMemoryStats:
    """Tests for MemoryStats."""

    def test_initial_stats(self):
        """Test initial statistics values."""
        stats = MemoryStats()

        assert stats.allocated == 0
        assert stats.reserved == 0
        assert stats.active == 0
        assert stats.inactive == 0
        assert stats.freed == 0
        assert stats.num_allocs == 0
        assert stats.num_frees == 0
        assert stats.num_ooms == 0
        assert stats.peak_allocated == 0
        assert stats.peak_reserved == 0

    def test_stats_update(self):
        """Test updating statistics."""
        stats = MemoryStats()

        # Simulate allocation
        alloc_size = 1024 * 1024  # 1MB
        stats.allocated += alloc_size
        stats.active += alloc_size
        stats.num_allocs += 1

        assert stats.allocated == alloc_size
        assert stats.active == alloc_size
        assert stats.num_allocs == 1

        # Update peak
        if stats.allocated > stats.peak_allocated:
            stats.peak_allocated = stats.allocated

        assert stats.peak_allocated == alloc_size

        # Simulate free
        stats.allocated -= alloc_size
        stats.active -= alloc_size
        stats.freed += alloc_size
        stats.num_frees += 1

        assert stats.allocated == 0
        assert stats.freed == alloc_size
        assert stats.num_frees == 1

    def test_stats_repr(self):
        """Test statistics string representation."""
        stats = MemoryStats(allocated=1000000, reserved=2000000, active=500000)
        repr_str = repr(stats)

        assert 'MemoryStats' in repr_str
        assert 'allocated' in repr_str


# =============================================================================
# MemoryBlock Tests
# =============================================================================

class TestMemoryBlock:
    """Tests for MemoryBlock."""

    def test_block_creation(self):
        """Test creating memory blocks."""
        block = MemoryBlock(
            ptr=12345,
            size=1024,
            device=DeviceType.CUDA,
            device_id=0
        )

        assert block.ptr == 12345
        assert block.size == 1024
        assert block.device == DeviceType.CUDA
        assert block.device_id == 0
        assert block.allocated is True
        assert block.prev is None
        assert block.next is None
        assert block.pool is None

    def test_block_with_stream(self):
        """Test block with stream assignment."""
        block = MemoryBlock(
            ptr=1000,
            size=512,
            device=DeviceType.CUDA,
            device_id=0,
            stream=5
        )

        assert block.stream == 5

    def test_block_split(self):
        """Test splitting a memory block."""
        block = MemoryBlock(
            ptr=1000,
            size=1024,
            device=DeviceType.CPU,
            device_id=0,
            allocated=False
        )

        # Split block
        remaining = block.split(256)

        assert block.size == 256
        assert remaining is not None
        assert remaining.size == 768
        assert remaining.ptr == 1256
        assert block.next == remaining
        assert remaining.prev == block

    def test_block_split_invalid(self):
        """Test splitting with invalid size."""
        block = MemoryBlock(
            ptr=1000,
            size=256,
            device=DeviceType.CPU,
            device_id=0
        )

        # Can't split to larger size
        remaining = block.split(512)
        assert remaining is None

    def test_block_merge(self):
        """Test merging adjacent blocks."""
        block1 = MemoryBlock(
            ptr=1000,
            size=512,
            device=DeviceType.CPU,
            device_id=0,
            allocated=False
        )
        block2 = MemoryBlock(
            ptr=1512,
            size=512,
            device=DeviceType.CPU,
            device_id=0,
            allocated=False
        )

        # Set up adjacency
        block1.next = block2
        block2.prev = block1

        # Merge
        success = block1.merge(block2)

        assert success is True
        assert block1.size == 1024
        assert block1.next is None

    def test_block_merge_non_adjacent(self):
        """Test merging non-adjacent blocks fails."""
        block1 = MemoryBlock(
            ptr=1000,
            size=512,
            device=DeviceType.CPU,
            device_id=0
        )
        block2 = MemoryBlock(
            ptr=2000,  # Not adjacent
            size=512,
            device=DeviceType.CPU,
            device_id=0
        )

        success = block1.merge(block2)
        assert success is False

    def test_block_can_merge_with(self):
        """Test can_merge_with check."""
        block1 = MemoryBlock(
            ptr=1000,
            size=512,
            device=DeviceType.CPU,
            device_id=0,
            allocated=False,
            stream=0
        )
        block2 = MemoryBlock(
            ptr=1512,
            size=512,
            device=DeviceType.CPU,
            device_id=0,
            allocated=False,
            stream=0
        )

        assert block1.can_merge_with(block2) is True

        # Can't merge if allocated
        block2.allocated = True
        assert block1.can_merge_with(block2) is False

    def test_block_hash_and_equality(self):
        """Test block hashing and equality."""
        block1 = MemoryBlock(ptr=1000, size=512, device=DeviceType.CPU, device_id=0)
        block2 = MemoryBlock(ptr=1000, size=1024, device=DeviceType.CUDA, device_id=1)
        block3 = MemoryBlock(ptr=2000, size=512, device=DeviceType.CPU, device_id=0)

        # Same ptr = equal
        assert block1 == block2
        assert hash(block1) == hash(block2)

        # Different ptr = not equal
        assert block1 != block3

    def test_block_metadata(self):
        """Test block metadata fields."""
        block = MemoryBlock(
            ptr=1000,
            size=512,
            device=DeviceType.CPU,
            device_id=0,
            tag="tensor_weights"
        )

        assert block.tag == "tensor_weights"
        assert block.timestamp > 0


# =============================================================================
# MemoryRegion Tests
# =============================================================================

class TestMemoryRegion:
    """Tests for MemoryRegion."""

    def test_region_creation(self):
        """Test creating memory region."""
        region = MemoryRegion(
            start=10000,
            size=1024 * 1024,
            device=DeviceType.CUDA,
            device_id=0
        )

        assert region.start == 10000
        assert region.size == 1024 * 1024
        assert region.device == DeviceType.CUDA
        assert region.device_id == 0
        assert region.free_size == 1024 * 1024
        assert len(region.blocks) == 0

    def test_region_with_blocks(self):
        """Test region with blocks."""
        region = MemoryRegion(
            start=10000,
            size=1024,
            device=DeviceType.CPU,
            device_id=0
        )

        block = MemoryBlock(ptr=10000, size=512, device=DeviceType.CPU, device_id=0)
        region.blocks.append(block)

        assert len(region.blocks) == 1


# =============================================================================
# AllocationRequest Tests
# =============================================================================

class TestAllocationRequest:
    """Tests for AllocationRequest."""

    def test_request_creation(self):
        """Test creating allocation request."""
        request = AllocationRequest(
            size=1024,
            device=DeviceType.CUDA,
            device_id=1,
            stream=5,
            tag="weights"
        )

        assert request.size == 1024
        assert request.device == DeviceType.CUDA
        assert request.device_id == 1
        assert request.stream == 5
        assert request.tag == "weights"

    def test_request_defaults(self):
        """Test request default values."""
        request = AllocationRequest(size=512)

        assert request.size == 512
        assert request.device == DeviceType.CPU
        assert request.device_id == 0
        assert request.stream == 0
        assert request.alignment == 512
        assert request.zero_init is False
        assert request.persistent is False

    def test_request_preferences(self):
        """Test request preferences."""
        request = AllocationRequest(
            size=1024,
            alignment=256,
            zero_init=True,
            persistent=True
        )

        assert request.alignment == 256
        assert request.zero_init is True
        assert request.persistent is True


# =============================================================================
# MemoryPool Tests
# =============================================================================

class TestMemoryPool:
    """Tests for MemoryPool."""

    def test_pool_creation(self):
        """Test creating memory pool."""
        config = MemoryConfig(device_type=DeviceType.CUDA)
        pool = MemoryPool(config)

        assert pool.config == config
        stats = pool.get_stats()
        assert stats.allocated == 0
        assert stats.num_allocs == 0

    def test_allocation(self):
        """Test memory allocation from pool."""
        pool = MemoryPool(MemoryConfig())
        request = AllocationRequest(size=1024)

        block = pool.allocate(request)

        assert block is not None
        assert block.size >= 1024
        assert block.allocated is True

        stats = pool.get_stats()
        assert stats.allocated >= 1024
        assert stats.num_allocs == 1

    def test_free(self):
        """Test memory deallocation."""
        pool = MemoryPool(MemoryConfig())
        request = AllocationRequest(size=1024)

        block = pool.allocate(request)
        pool.free(block)

        stats = pool.get_stats()
        assert stats.num_frees == 1

    def test_round_size(self):
        """Test size rounding logic."""
        config = MemoryConfig(alignment=512)
        pool = MemoryPool(config)

        assert pool._round_size(500) == 512
        assert pool._round_size(513) == 1024
        assert pool._round_size(1024) == 1024

    def test_size_class(self):
        """Test size class assignment."""
        pool = MemoryPool(MemoryConfig())

        assert pool._get_size_class(500) == 512
        assert pool._get_size_class(600) == 1024
        assert pool._get_size_class(1024) == 1024

    def test_multiple_allocations(self):
        """Test multiple allocations."""
        pool = MemoryPool(MemoryConfig())

        blocks = []
        for i in range(5):
            request = AllocationRequest(size=1024 * (i + 1))
            block = pool.allocate(request)
            assert block is not None
            blocks.append(block)

        stats = pool.get_stats()
        assert stats.num_allocs == 5

        for block in blocks:
            pool.free(block)

        stats = pool.get_stats()
        assert stats.num_frees == 5

    def test_block_reuse(self):
        """Test that freed blocks are reused."""
        pool = MemoryPool(MemoryConfig())
        request = AllocationRequest(size=1024)

        # Allocate and free
        block1 = pool.allocate(request)
        pool.free(block1)

        # Allocate again - should reuse
        block2 = pool.allocate(request)
        assert block2 is not None

    def test_memory_limit(self):
        """Test memory limit enforcement."""
        # Very small limit to ensure we hit OOM
        config = MemoryConfig(max_memory=1024)  # 1KB limit
        pool = MemoryPool(config)

        # First allocation creates a region (at least 2MB by default)
        # but the limit prevents any allocation
        request = AllocationRequest(size=2048)  # Request more than limit
        block = pool.allocate(request)

        # Should fail or record OOM
        stats = pool.get_stats()
        assert stats.num_ooms > 0 or block is None

    def test_stream_aware_allocation(self):
        """Test stream-aware allocation."""
        pool = MemoryPool(MemoryConfig())

        request1 = AllocationRequest(size=1024, stream=1)
        request2 = AllocationRequest(size=1024, stream=2)

        block1 = pool.allocate(request1)
        block2 = pool.allocate(request2)

        assert block1 is not None
        assert block2 is not None
        assert block1.stream == 1
        assert block2.stream == 2

    def test_empty_cache(self):
        """Test cache clearing."""
        pool = MemoryPool(MemoryConfig())

        # Allocate and free to populate cache
        blocks = []
        for _ in range(5):
            request = AllocationRequest(size=1024)
            block = pool.allocate(request)
            blocks.append(block)

        for block in blocks:
            pool.free(block)

        # Clear cache
        pool.empty_cache()

        stats = pool.get_stats()
        assert stats.inactive == 0

    def test_trim(self):
        """Test pool trimming."""
        pool = MemoryPool(MemoryConfig())

        # Allocate and free to create inactive memory
        blocks = []
        for _ in range(10):
            request = AllocationRequest(size=1024)
            block = pool.allocate(request)
            blocks.append(block)

        for block in blocks:
            pool.free(block)

        # Trim pool
        pool.trim(target_size=0)

    def test_reset_stats(self):
        """Test statistics reset."""
        pool = MemoryPool(MemoryConfig())

        request = AllocationRequest(size=1024)
        block = pool.allocate(request)
        pool.free(block)

        pool.reset_stats()

        stats = pool.get_stats()
        assert stats.num_allocs == 0
        assert stats.num_frees == 0

    def test_block_splitting(self):
        """Test block splitting on allocation."""
        config = MemoryConfig(min_split_size=512)
        pool = MemoryPool(config)

        # Allocate small block from a larger region
        request = AllocationRequest(size=512)
        block = pool.allocate(request)

        assert block is not None
        # Block size should be close to requested

    def test_coalescing(self):
        """Test block coalescing on free."""
        pool = MemoryPool(MemoryConfig())

        # Allocate adjacent blocks
        blocks = []
        for _ in range(3):
            request = AllocationRequest(size=1024)
            blocks.append(pool.allocate(request))

        # Free all - should coalesce
        for block in blocks:
            pool.free(block)


# =============================================================================
# Thread Safety Tests
# =============================================================================

class TestMemoryPoolThreadSafety:
    """Test thread safety of memory pool."""

    def test_concurrent_allocation(self):
        """Test concurrent allocations."""
        pool = MemoryPool(MemoryConfig())
        results = []
        errors = []

        def worker():
            try:
                local_blocks = []
                for _ in range(10):
                    request = AllocationRequest(size=1024)
                    block = pool.allocate(request)
                    if block:
                        local_blocks.append(block)
                    time.sleep(0.001)

                for block in local_blocks:
                    pool.free(block)

                results.append(len(local_blocks))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert sum(results) == 50

    def test_concurrent_free(self):
        """Test concurrent frees."""
        pool = MemoryPool(MemoryConfig())
        blocks = []

        # Pre-allocate blocks
        for _ in range(20):
            request = AllocationRequest(size=512)
            blocks.append(pool.allocate(request))

        errors = []

        def worker(thread_blocks):
            try:
                for block in thread_blocks:
                    pool.free(block)
            except Exception as e:
                errors.append(e)

        # Split blocks between threads
        mid = len(blocks) // 2
        t1 = threading.Thread(target=worker, args=(blocks[:mid],))
        t2 = threading.Thread(target=worker, args=(blocks[mid:],))

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0

        stats = pool.get_stats()
        assert stats.num_frees == 20


# =============================================================================
# Edge Cases
# =============================================================================

class TestMemoryEdgeCases:
    """Test edge cases in memory management."""

    def test_zero_size_allocation(self):
        """Test allocation with zero size."""
        pool = MemoryPool(MemoryConfig())
        request = AllocationRequest(size=0)

        # Should round up to minimum alignment
        block = pool.allocate(request)
        assert block is not None

    def test_large_allocation(self):
        """Test large allocation."""
        pool = MemoryPool(MemoryConfig())
        request = AllocationRequest(size=100 * 1024 * 1024)  # 100MB

        block = pool.allocate(request)
        assert block is not None
        assert block.size >= 100 * 1024 * 1024

    def test_rapid_alloc_free(self):
        """Test rapid allocation and free cycles."""
        pool = MemoryPool(MemoryConfig())

        for _ in range(100):
            request = AllocationRequest(size=1024)
            block = pool.allocate(request)
            pool.free(block)

        stats = pool.get_stats()
        assert stats.num_allocs == 100
        assert stats.num_frees == 100

    def test_free_unknown_block(self):
        """Test freeing unknown block."""
        pool = MemoryPool(MemoryConfig())

        # Create block not from this pool
        fake_block = MemoryBlock(ptr=99999, size=512, device=DeviceType.CPU, device_id=0)

        # Should not crash
        pool.free(fake_block)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
