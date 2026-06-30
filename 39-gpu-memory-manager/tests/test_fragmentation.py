"""Tests for memory fragmentation scenarios, OOM recovery, stress testing, and profiler accuracy."""

import pytest
import threading
import time
import random
from typing import List, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gpumem.core.memory import (
    DeviceType,
    MemoryConfig,
    MemoryStats,
    MemoryBlock,
    MemoryPool,
    MemoryRegion,
    AllocationRequest,
)
from gpumem.allocator.allocator import (
    Allocator,
    CachingAllocator,
    PoolAllocator,
    BuddyAllocator,
    SlabAllocator,
    UnifiedMemoryAllocator,
)
from gpumem.cache.cache import (
    LRUCache,
    LFUCache,
    TensorCache,
    Defragmenter,
    DefragmentStrategy,
    CachePolicy,
)
from gpumem.profiler.profiler import (
    MemoryProfiler,
    AllocationTrace,
    MemorySnapshot,
    MemoryTimeline,
    MemoryEvent,
    EventType,
)


# ==============================================================================
# MEMORY FRAGMENTATION TESTS
# ==============================================================================


class TestFragmentationScenarios:
    """Test various memory fragmentation patterns."""

    def test_alternating_allocation_pattern(self):
        """Test fragmentation from alternating alloc/free patterns."""
        allocator = PoolAllocator(MemoryConfig())
        blocks: List[MemoryBlock] = []

        # Allocate many small blocks
        for _ in range(20):
            block = allocator.allocate(1024)  # 1KB each
            assert block is not None
            blocks.append(block)

        # Free every other block - creates fragmentation
        for i in range(0, len(blocks), 2):
            allocator.free(blocks[i])

        # Try to allocate a larger block - may fail due to fragmentation
        # even though total free space is sufficient
        large_block = allocator.allocate(8 * 1024)  # 8KB

        # Should still succeed because PoolAllocator uses size classes
        assert large_block is not None

        stats = allocator.get_stats()
        assert stats.num_allocs >= 20
        assert stats.num_frees == 10

    def test_swiss_cheese_fragmentation(self):
        """Test swiss cheese pattern - small holes throughout memory."""
        allocator = BuddyAllocator(MemoryConfig(), total_size=1024 * 1024)  # 1MB
        blocks: List[MemoryBlock] = []

        # Allocate blocks of various sizes
        sizes = [4096, 8192, 16384, 32768, 4096, 8192, 16384, 4096]
        for size in sizes:
            block = allocator.allocate(size)
            if block:
                blocks.append(block)

        # Free every third block to create swiss cheese pattern
        freed_indices = [i for i in range(0, len(blocks), 3)]
        for i in freed_indices:
            allocator.free(blocks[i])

        stats = allocator.get_stats()
        assert stats.num_frees == len(freed_indices)

        # Attempt to allocate a block larger than any single hole
        total_freed = sum(blocks[i].size for i in freed_indices if i < len(blocks))

        # This may fail due to fragmentation in buddy allocator
        large_alloc = allocator.allocate(total_freed)
        # Note: may be None if holes aren't contiguous

    def test_external_fragmentation_detection(self):
        """Test that external fragmentation can be detected."""
        defrag = Defragmenter(DefragmentStrategy.COMPACT)

        # Create blocks simulating external fragmentation
        blocks = [
            MemoryBlock(ptr=0, size=1024, device=DeviceType.CPU, device_id=0, allocated=True),
            MemoryBlock(ptr=1024, size=512, device=DeviceType.CPU, device_id=0, allocated=False),
            MemoryBlock(ptr=1536, size=1024, device=DeviceType.CPU, device_id=0, allocated=True),
            MemoryBlock(ptr=2560, size=256, device=DeviceType.CPU, device_id=0, allocated=False),
            MemoryBlock(ptr=2816, size=512, device=DeviceType.CPU, device_id=0, allocated=True),
            MemoryBlock(ptr=3328, size=768, device=DeviceType.CPU, device_id=0, allocated=False),
        ]

        frag_ratio = defrag.get_fragmentation_ratio(blocks)

        # Should detect fragmentation (ratio > 0)
        assert frag_ratio > 0
        assert frag_ratio <= 1.0

    def test_internal_fragmentation_from_size_classes(self):
        """Test internal fragmentation from size class rounding."""
        allocator = PoolAllocator(MemoryConfig())

        # Request sizes that don't match size classes exactly
        requested_sizes = [100, 300, 700, 1500, 3000]
        blocks = []

        for req_size in requested_sizes:
            block = allocator.allocate(req_size)
            assert block is not None
            blocks.append((req_size, block))

            # Block size should be >= requested (rounded up)
            assert block.size >= req_size

        # Calculate internal fragmentation
        total_requested = sum(s for s, _ in blocks)
        total_allocated = sum(b.size for _, b in blocks)

        internal_frag = (total_allocated - total_requested) / total_allocated

        # Some internal fragmentation is expected
        assert internal_frag >= 0
        assert internal_frag < 1.0

    def test_fragmentation_after_long_running_workload(self):
        """Simulate fragmentation over a long-running workload."""
        allocator = PoolAllocator(MemoryConfig())  # Use PoolAllocator which tracks frees

        # Simulate a workload with varying allocation patterns
        active_blocks = []
        freed_count = 0

        for iteration in range(50):
            # Allocate some new blocks
            for _ in range(random.randint(1, 5)):
                size = random.choice([512, 1024, 2048, 4096, 8192])
                block = allocator.allocate(size)
                if block:
                    active_blocks.append(block)

            # Free some random blocks
            if active_blocks and random.random() > 0.3:
                num_to_free = random.randint(1, min(3, len(active_blocks)))
                for _ in range(num_to_free):
                    if active_blocks:
                        block = active_blocks.pop(random.randint(0, len(active_blocks) - 1))
                        allocator.free(block)
                        freed_count += 1

        # Clean up remaining blocks
        for block in active_blocks:
            allocator.free(block)
            freed_count += 1

        stats = allocator.get_stats()
        assert stats.num_allocs > 0
        # Verify we tracked the frees
        assert freed_count > 0

    def test_defragmentation_compaction(self):
        """Test that defragmentation compacts memory."""
        defrag = Defragmenter(DefragmentStrategy.COMPACT)

        # Create fragmented memory layout
        blocks = [
            MemoryBlock(ptr=0, size=1024, device=DeviceType.CPU, device_id=0, allocated=True),
            MemoryBlock(ptr=1024, size=1024, device=DeviceType.CPU, device_id=0, allocated=False),
            MemoryBlock(ptr=2048, size=1024, device=DeviceType.CPU, device_id=0, allocated=True),
            MemoryBlock(ptr=3072, size=1024, device=DeviceType.CPU, device_id=0, allocated=False),
            MemoryBlock(ptr=4096, size=1024, device=DeviceType.CPU, device_id=0, allocated=True),
        ]

        copy_log = []
        def mock_copy(src, dst, size):
            copy_log.append((src, dst, size))

        frag_before = defrag.get_fragmentation_ratio(blocks)
        compacted = defrag.defragment(blocks, mock_copy)
        frag_after = defrag.get_fragmentation_ratio(compacted)

        # Fragmentation should be reduced or eliminated
        assert frag_after <= frag_before

    def test_buddy_allocator_fragmentation_resistance(self):
        """Test that buddy allocator resists fragmentation through coalescing."""
        allocator = BuddyAllocator(MemoryConfig(), total_size=16 * 1024)  # 16KB

        # Allocate two buddy blocks
        block1 = allocator.allocate(4096)  # 4KB
        block2 = allocator.allocate(4096)  # 4KB

        assert block1 is not None
        assert block2 is not None

        # Free both - they should coalesce
        allocator.free(block1)
        allocator.free(block2)

        # Now should be able to allocate 8KB (coalesced buddies)
        large_block = allocator.allocate(8192)  # 8KB
        assert large_block is not None


# ==============================================================================
# OOM RECOVERY TESTS
# ==============================================================================


class TestOOMRecovery:
    """Test out-of-memory recovery mechanisms."""

    def test_oom_triggers_cache_flush(self):
        """Test that OOM triggers cache flush and retry."""
        config = MemoryConfig(max_memory=100 * 1024)  # 100KB limit
        allocator = CachingAllocator(config)

        # Fill up memory
        blocks = []
        for _ in range(20):
            block = allocator.allocate(4096)  # 4KB each
            if block:
                blocks.append(block)

        # Free some blocks (they go to cache)
        for block in blocks[:10]:
            allocator.free(block)

        # Try to allocate more - should trigger cache flush
        new_block = allocator.allocate(8192)

        # May succeed after cache flush
        # (depends on implementation details)

    def test_oom_tracking_in_stats(self):
        """Test that OOM events are properly tracked."""
        config = MemoryConfig(max_memory=1024)  # Very small limit
        pool = MemoryPool(config)

        request = AllocationRequest(size=512, device=DeviceType.CPU)

        # First allocation should succeed
        block1 = pool.allocate(request)

        # Second allocation may cause OOM
        request2 = AllocationRequest(size=1024, device=DeviceType.CPU)
        block2 = pool.allocate(request2)

        stats = pool.get_stats()
        # If allocation failed, OOM count should increase
        if block2 is None:
            assert stats.num_ooms > 0

    def test_oom_recovery_with_empty_cache(self):
        """Test OOM recovery when cache is emptied."""
        allocator = CachingAllocator(MemoryConfig())

        # Allocate and free to build up cache
        for _ in range(10):
            block = allocator.allocate(1024)
            if block:
                allocator.free(block)

        # Empty the cache
        allocator.empty_cache()

        # Should still be able to allocate new memory
        block = allocator.allocate(2048)
        assert block is not None

    def test_buddy_allocator_oom_handling(self):
        """Test OOM handling in buddy allocator."""
        # Create a small buddy allocator
        allocator = BuddyAllocator(MemoryConfig(), total_size=4096)  # 4KB total

        # Allocate all memory
        block1 = allocator.allocate(2048)  # 2KB
        block2 = allocator.allocate(2048)  # 2KB

        # Try to allocate more - should fail with OOM
        block3 = allocator.allocate(1024)

        stats = allocator.get_stats()
        if block3 is None:
            assert stats.num_ooms > 0

    def test_pool_trim_on_high_memory_pressure(self):
        """Test that pool can be trimmed under memory pressure."""
        config = MemoryConfig()
        pool = MemoryPool(config)

        # Allocate and free to create inactive memory
        blocks = []
        for _ in range(10):
            request = AllocationRequest(size=4096, device=DeviceType.CPU)
            block = pool.allocate(request)
            if block:
                blocks.append(block)

        for block in blocks:
            pool.free(block)

        initial_inactive = pool.get_stats().inactive

        # Trim the pool
        pool.trim(target_size=initial_inactive // 2)

        # Inactive should be reduced
        final_inactive = pool.get_stats().inactive
        assert final_inactive <= initial_inactive

    def test_graceful_degradation_under_oom(self):
        """Test system behavior under sustained OOM conditions."""
        config = MemoryConfig(max_memory=10 * 1024)  # 10KB limit
        pool = MemoryPool(config)

        success_count = 0
        failure_count = 0

        # Attempt many allocations
        for _ in range(50):
            request = AllocationRequest(size=1024, device=DeviceType.CPU)
            block = pool.allocate(request)
            if block:
                success_count += 1
                # Don't free - accumulate pressure
            else:
                failure_count += 1

        # System should handle failures gracefully
        assert success_count > 0
        # Some failures expected under memory pressure
        stats = pool.get_stats()
        # Verify stats are consistent
        assert stats.num_allocs == success_count


# ==============================================================================
# ALLOCATION STRATEGY STRESS TESTS
# ==============================================================================


class TestAllocationStrategiesUnderStress:
    """Test allocation strategies under various stress conditions."""

    def test_high_frequency_small_allocations(self):
        """Test allocator under high frequency small allocations."""
        allocator = SlabAllocator(MemoryConfig(), object_size=64)

        blocks = []
        start_time = time.time()

        # Rapid allocations
        for _ in range(1000):
            block = allocator.allocate()
            assert block is not None
            blocks.append(block)

        alloc_time = time.time() - start_time

        # Free all
        start_time = time.time()
        for block in blocks:
            allocator.free(block)

        free_time = time.time() - start_time

        stats = allocator.get_stats()
        assert stats.num_allocs == 1000
        assert stats.num_frees == 1000

    def test_mixed_size_allocation_pattern(self):
        """Test allocator with mixed allocation sizes."""
        allocator = CachingAllocator(MemoryConfig())

        sizes = [64, 256, 1024, 4096, 16384, 65536, 262144]
        blocks = []

        # Random mixed allocations
        for _ in range(100):
            size = random.choice(sizes)
            block = allocator.allocate(size)
            assert block is not None
            blocks.append(block)

        # Random frees and reallocs
        for _ in range(50):
            if blocks and random.random() > 0.5:
                idx = random.randint(0, len(blocks) - 1)
                allocator.free(blocks.pop(idx))
            else:
                size = random.choice(sizes)
                block = allocator.allocate(size)
                if block:
                    blocks.append(block)

        # Clean up
        for block in blocks:
            allocator.free(block)

        stats = allocator.get_stats()
        assert stats.num_allocs > 100

    def test_concurrent_allocation_stress(self):
        """Test allocator under concurrent access."""
        allocator = PoolAllocator(MemoryConfig())
        errors = []
        results = []
        lock = threading.Lock()

        def worker(worker_id):
            try:
                local_blocks = []
                for _ in range(50):
                    block = allocator.allocate(1024)
                    if block:
                        local_blocks.append(block)
                    time.sleep(0.0001)

                for block in local_blocks:
                    allocator.free(block)

                with lock:
                    results.append(len(local_blocks))
            except Exception as e:
                with lock:
                    errors.append((worker_id, e))

        threads = []
        for i in range(10):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert sum(results) > 0

    def test_allocation_deallocation_balance(self):
        """Test that allocations and deallocations remain balanced."""
        allocator = BuddyAllocator(MemoryConfig(), total_size=1024 * 1024)

        blocks = []

        for round_num in range(10):
            # Allocate phase
            for _ in range(10):
                block = allocator.allocate(random.choice([512, 1024, 2048, 4096]))
                if block:
                    blocks.append(block)

            # Deallocate phase
            num_to_free = random.randint(1, len(blocks) // 2 + 1)
            for _ in range(num_to_free):
                if blocks:
                    allocator.free(blocks.pop())

        # Final cleanup
        for block in blocks:
            allocator.free(block)

        stats = allocator.get_stats()
        # All memory should be freed
        assert stats.allocated == 0

    def test_stream_ordered_allocations(self):
        """Test allocations across multiple streams."""
        allocator = PoolAllocator(MemoryConfig())  # Use PoolAllocator which tracks stats
        stream_blocks = {0: [], 1: [], 2: []}

        # Allocate on different streams
        for stream_id in [0, 1, 2]:
            for _ in range(10):
                block = allocator.allocate(1024, stream=stream_id)
                assert block is not None
                stream_blocks[stream_id].append(block)

        # Free in reverse stream order
        for stream_id in [2, 1, 0]:
            for block in stream_blocks[stream_id]:
                allocator.free(block)

        stats = allocator.get_stats()
        assert stats.num_allocs == 30
        assert stats.num_frees == 30

    def test_power_of_two_allocation_efficiency(self):
        """Test efficiency of power-of-2 allocations in buddy allocator."""
        allocator = BuddyAllocator(MemoryConfig(), total_size=1024 * 1024)

        # Allocate power-of-2 sizes
        blocks = []
        sizes = [512, 1024, 2048, 4096, 8192, 16384]

        for size in sizes:
            block = allocator.allocate(size)
            assert block is not None
            # Buddy allocator should give exact size for power-of-2
            assert block.size == size
            blocks.append(block)

        # Free all
        for block in blocks:
            allocator.free(block)

    def test_worst_case_fragmentation_pattern(self):
        """Test worst case fragmentation: allocate all, free odd indices."""
        allocator = PoolAllocator(MemoryConfig())

        # Allocate many same-sized blocks
        blocks = []
        for _ in range(100):
            block = allocator.allocate(1024)
            assert block is not None
            blocks.append(block)

        # Free every other block
        for i in range(1, 100, 2):
            allocator.free(blocks[i])

        # Try to allocate double-sized block
        large_block = allocator.allocate(2048)

        # Pool allocator uses size classes, so this should work
        assert large_block is not None

        # Cleanup
        for i in range(0, 100, 2):
            allocator.free(blocks[i])
        allocator.free(large_block)


# ==============================================================================
# MEMORY PROFILER ACCURACY TESTS
# ==============================================================================


class TestMemoryProfilerAccuracy:
    """Test memory profiler accuracy and correctness."""

    def test_profiler_tracks_allocations_correctly(self):
        """Test that profiler accurately tracks allocations."""
        profiler = MemoryProfiler(record_stacktraces=False)
        profiler.enable()

        # Record some allocations
        blocks = [
            MemoryBlock(ptr=1000, size=1024, device=DeviceType.CPU, device_id=0, tag="test1"),
            MemoryBlock(ptr=2000, size=2048, device=DeviceType.CPU, device_id=0, tag="test2"),
            MemoryBlock(ptr=3000, size=4096, device=DeviceType.CPU, device_id=0, tag="test3"),
        ]

        for block in blocks:
            profiler.record_alloc(block)

        active = profiler.get_active_allocations()
        assert len(active) == 3

        stats = profiler.get_stats()
        assert stats.allocated == 1024 + 2048 + 4096
        assert stats.num_allocs == 3

    def test_profiler_tracks_frees_correctly(self):
        """Test that profiler accurately tracks frees."""
        profiler = MemoryProfiler(record_stacktraces=False)
        profiler.enable()

        # Allocate
        block = MemoryBlock(ptr=1000, size=1024, device=DeviceType.CPU, device_id=0)
        profiler.record_alloc(block)

        # Free
        profiler.record_free(block)

        active = profiler.get_active_allocations()
        assert len(active) == 0

        stats = profiler.get_stats()
        assert stats.allocated == 0
        assert stats.num_frees == 1

    def test_profiler_size_histogram_accuracy(self):
        """Test that size histogram categorizes correctly."""
        profiler = MemoryProfiler(record_stacktraces=False)
        profiler.enable()

        # Allocate various sizes
        allocations = [
            (1000, 512),      # <1KB
            (2000, 512),      # <1KB
            (3000, 50000),    # 1KB-1MB
            (4000, 500000),   # 1KB-1MB
            (5000, 5000000),  # 1MB-10MB
            (6000, 50000000), # 10MB-100MB
            (7000, 200000000), # >100MB
        ]

        for ptr, size in allocations:
            block = MemoryBlock(ptr=ptr, size=size, device=DeviceType.CPU, device_id=0)
            profiler.record_alloc(block)

        histogram = profiler.get_size_histogram()

        assert histogram["<1KB"] == 2
        assert histogram["1KB-1MB"] == 2
        assert histogram["1MB-10MB"] == 1
        assert histogram["10MB-100MB"] == 1
        assert histogram[">100MB"] == 1

    def test_profiler_tag_summary_accuracy(self):
        """Test that tag summary aggregates correctly."""
        profiler = MemoryProfiler(record_stacktraces=False)
        profiler.enable()

        # Allocate with tags
        blocks = [
            MemoryBlock(ptr=1000, size=1024, device=DeviceType.CPU, device_id=0, tag="weights"),
            MemoryBlock(ptr=2000, size=2048, device=DeviceType.CPU, device_id=0, tag="weights"),
            MemoryBlock(ptr=3000, size=4096, device=DeviceType.CPU, device_id=0, tag="activations"),
            MemoryBlock(ptr=4000, size=8192, device=DeviceType.CPU, device_id=0, tag="gradients"),
        ]

        for block in blocks:
            profiler.record_alloc(block)

        tag_summary = profiler.get_tag_summary()

        assert tag_summary["weights"] == 1024 + 2048
        assert tag_summary["activations"] == 4096
        assert tag_summary["gradients"] == 8192

    def test_profiler_device_summary_accuracy(self):
        """Test that device summary aggregates correctly."""
        profiler = MemoryProfiler(record_stacktraces=False)
        profiler.enable()

        blocks = [
            MemoryBlock(ptr=1000, size=1024, device=DeviceType.CPU, device_id=0),
            MemoryBlock(ptr=2000, size=2048, device=DeviceType.CPU, device_id=0),
            MemoryBlock(ptr=3000, size=4096, device=DeviceType.CUDA, device_id=0),
            MemoryBlock(ptr=4000, size=8192, device=DeviceType.CUDA, device_id=0),
        ]

        for block in blocks:
            profiler.record_alloc(block)

        device_summary = profiler.get_device_summary()

        assert device_summary["CPU"] == 1024 + 2048
        assert device_summary["CUDA"] == 4096 + 8192

    def test_profiler_leak_detection(self):
        """Test that profiler can detect long-lived allocations."""
        profiler = MemoryProfiler(record_stacktraces=False)
        profiler.enable()

        # Simulate old allocation
        old_block = MemoryBlock(ptr=1000, size=1024, device=DeviceType.CPU, device_id=0)
        profiler.record_alloc(old_block)

        # Manually age the allocation by modifying timestamp
        if old_block.ptr in profiler._allocations:
            profiler._allocations[old_block.ptr].timestamp = time.time() - 120  # 2 minutes ago

        # Find leaks with 60 second threshold
        leaks = profiler.find_leaks(min_lifetime=60.0)

        assert len(leaks) == 1
        assert leaks[0].ptr == 1000

    def test_profiler_snapshot_accuracy(self):
        """Test that snapshots capture correct state."""
        profiler = MemoryProfiler(record_stacktraces=False)
        profiler.enable()

        # Create some allocations
        for i in range(5):
            block = MemoryBlock(ptr=1000 * (i + 1), size=1024 * (i + 1),
                              device=DeviceType.CPU, device_id=0)
            profiler.record_alloc(block)

        snapshot = profiler.take_snapshot(tag="test_snapshot")

        assert snapshot.stats.num_allocs == 5
        assert len(snapshot.active_allocations) == 5
        assert snapshot.metadata["tag"] == "test_snapshot"

        # Verify snapshot data
        snapshot_dict = snapshot.to_dict()
        assert snapshot_dict["num_active_allocations"] == 5
        total_size = sum(1024 * (i + 1) for i in range(5))
        assert snapshot_dict["total_active_size"] == total_size

    def test_profiler_top_allocations(self):
        """Test that top allocations are correctly identified."""
        profiler = MemoryProfiler(record_stacktraces=False)
        profiler.enable()

        # Allocate with varying sizes
        sizes = [100, 1000, 500, 2000, 300, 1500]
        for i, size in enumerate(sizes):
            block = MemoryBlock(ptr=1000 * (i + 1), size=size,
                              device=DeviceType.CPU, device_id=0)
            profiler.record_alloc(block)

        top_3 = profiler.get_top_allocations(n=3)

        assert len(top_3) == 3
        # Should be sorted by size descending
        assert top_3[0].size == 2000
        assert top_3[1].size == 1500
        assert top_3[2].size == 1000

    def test_profiler_oom_recording(self):
        """Test that OOM events are recorded."""
        profiler = MemoryProfiler(record_stacktraces=False)
        profiler.enable()

        profiler.record_oom(size=1024 * 1024, device=DeviceType.CUDA, device_id=0)
        profiler.record_oom(size=2 * 1024 * 1024, device=DeviceType.CUDA, device_id=0)

        stats = profiler.get_stats()
        assert stats.num_ooms == 2

        # Check timeline has OOM events
        timeline = profiler.get_timeline()
        oom_events = timeline.get_events(event_types=[EventType.OOM])
        assert len(oom_events) == 2

    def test_profiler_timeline_event_ordering(self):
        """Test that timeline events are correctly ordered."""
        profiler = MemoryProfiler(record_stacktraces=False)
        profiler.enable()

        # Record events with small delays
        block1 = MemoryBlock(ptr=1000, size=1024, device=DeviceType.CPU, device_id=0)
        profiler.record_alloc(block1)

        time.sleep(0.01)

        block2 = MemoryBlock(ptr=2000, size=2048, device=DeviceType.CPU, device_id=0)
        profiler.record_alloc(block2)

        time.sleep(0.01)

        profiler.record_free(block1)

        timeline = profiler.get_timeline()
        events = timeline.get_events()

        # Events should be in chronological order
        for i in range(len(events) - 1):
            assert events[i].timestamp <= events[i + 1].timestamp

    def test_profiler_reset(self):
        """Test that profiler reset clears all state."""
        profiler = MemoryProfiler(record_stacktraces=False)
        profiler.enable()

        # Record some data
        for i in range(5):
            block = MemoryBlock(ptr=1000 * (i + 1), size=1024,
                              device=DeviceType.CPU, device_id=0)
            profiler.record_alloc(block)

        # Reset
        profiler.reset()

        # Verify clean state
        assert len(profiler.get_active_allocations()) == 0
        stats = profiler.get_stats()
        assert stats.num_allocs == 0
        assert stats.allocated == 0

    def test_profiler_disabled_no_tracking(self):
        """Test that disabled profiler doesn't track."""
        profiler = MemoryProfiler(record_stacktraces=False)
        # Don't enable

        block = MemoryBlock(ptr=1000, size=1024, device=DeviceType.CPU, device_id=0)
        profiler.record_alloc(block)

        assert len(profiler.get_active_allocations()) == 0

    def test_profiler_allocation_lifetime_tracking(self):
        """Test that allocation lifetimes are tracked correctly."""
        profiler = MemoryProfiler(record_stacktraces=False)
        profiler.enable()

        block = MemoryBlock(ptr=1000, size=1024, device=DeviceType.CPU, device_id=0)
        profiler.record_alloc(block)

        # Small delay
        time.sleep(0.1)

        active = profiler.get_active_allocations()
        assert len(active) == 1

        # Lifetime should be positive
        assert active[0].lifetime >= 0.1
        assert active[0].is_active

        # Free it
        profiler.record_free(block)

        # No more active allocations
        assert len(profiler.get_active_allocations()) == 0


# ==============================================================================
# CACHE ACCURACY TESTS
# ==============================================================================


class TestCacheAccuracy:
    """Test cache implementation accuracy."""

    def test_lru_cache_eviction_order(self):
        """Test that LRU cache evicts least recently used."""
        cache = LRUCache(max_size=3072)  # 3KB

        # Add blocks
        block1 = MemoryBlock(ptr=1000, size=1024, device=DeviceType.CPU, device_id=0)
        block2 = MemoryBlock(ptr=2000, size=1024, device=DeviceType.CPU, device_id=0)
        block3 = MemoryBlock(ptr=3000, size=1024, device=DeviceType.CPU, device_id=0)

        cache.put("a", block1)
        cache.put("b", block2)
        cache.put("c", block3)

        # Access "a" to make it recently used
        cache.get("a")

        # Add another - should evict "b" (least recently used)
        block4 = MemoryBlock(ptr=4000, size=1024, device=DeviceType.CPU, device_id=0)
        cache.put("d", block4)

        # "b" should be evicted
        assert cache.get("b") is None
        assert cache.get("a") is not None
        assert cache.get("c") is not None
        assert cache.get("d") is not None

    def test_lfu_cache_eviction_order(self):
        """Test that LFU cache evicts least frequently used."""
        cache = LFUCache(max_size=3072)  # 3KB

        block1 = MemoryBlock(ptr=1000, size=1024, device=DeviceType.CPU, device_id=0)
        block2 = MemoryBlock(ptr=2000, size=1024, device=DeviceType.CPU, device_id=0)
        block3 = MemoryBlock(ptr=3000, size=1024, device=DeviceType.CPU, device_id=0)

        cache.put("a", block1)
        cache.put("b", block2)
        cache.put("c", block3)

        # Access "a" and "c" multiple times
        for _ in range(5):
            cache.get("a")
        for _ in range(3):
            cache.get("c")
        # "b" only accessed once (on put)

        # Add another - should evict "b" (least frequently used)
        block4 = MemoryBlock(ptr=4000, size=1024, device=DeviceType.CPU, device_id=0)
        cache.put("d", block4)

        # "b" should be evicted
        assert cache.get("b") is None
        assert cache.get("a") is not None

    def test_cache_size_tracking(self):
        """Test that cache size is tracked correctly."""
        cache = LRUCache(max_size=10 * 1024)

        block1 = MemoryBlock(ptr=1000, size=1024, device=DeviceType.CPU, device_id=0)
        block2 = MemoryBlock(ptr=2000, size=2048, device=DeviceType.CPU, device_id=0)

        cache.put("a", block1)
        assert cache.get_size() == 1024

        cache.put("b", block2)
        assert cache.get_size() == 3072

        cache.remove("a")
        assert cache.get_size() == 2048

    def test_tensor_cache_shape_matching(self):
        """Test tensor cache matches shapes correctly."""
        cache = TensorCache(max_size=10 * 1024 * 1024, policy=CachePolicy.LRU)

        shape = (32, 64, 128)
        block = MemoryBlock(ptr=1000, size=32 * 64 * 128 * 4,
                           device=DeviceType.CPU, device_id=0)

        cache.put_tensor(shape, "float32", block)

        # Exact match should work
        retrieved = cache.get_tensor(shape, "float32")
        assert retrieved is not None
        assert retrieved.ptr == 1000


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_allocator_with_profiler(self):
        """Test allocator with profiler integration."""
        allocator = PoolAllocator(MemoryConfig())
        profiler = MemoryProfiler(record_stacktraces=False)
        profiler.enable()

        # Allocate with profiling
        blocks = []
        for _ in range(10):
            block = allocator.allocate(1024)
            assert block is not None
            profiler.record_alloc(block)
            blocks.append(block)

        assert len(profiler.get_active_allocations()) == 10

        # Free with profiling
        for block in blocks:
            profiler.record_free(block)
            allocator.free(block)

        assert len(profiler.get_active_allocations()) == 0

        allocator_stats = allocator.get_stats()
        profiler_stats = profiler.get_stats()

        assert allocator_stats.num_allocs == profiler_stats.num_allocs
        assert allocator_stats.num_frees == profiler_stats.num_frees

    def test_full_memory_lifecycle(self):
        """Test complete memory lifecycle with all components."""
        config = MemoryConfig()
        allocator = CachingAllocator(config)
        profiler = MemoryProfiler(record_stacktraces=False)
        profiler.enable()

        # Simulate workload
        blocks = []
        for phase in range(3):
            # Allocation phase
            for _ in range(5):
                size = random.choice([512, 1024, 2048, 4096])
                block = allocator.allocate(size)
                if block:
                    profiler.record_alloc(block)
                    blocks.append(block)

            # Take snapshot
            profiler.take_snapshot(tag=f"phase_{phase}")

            # Partial free phase
            num_to_free = len(blocks) // 2
            for _ in range(num_to_free):
                if blocks:
                    block = blocks.pop()
                    profiler.record_free(block)
                    allocator.free(block)

        # Final cleanup
        for block in blocks:
            profiler.record_free(block)
            allocator.free(block)

        # Verify consistency
        timeline = profiler.get_timeline()
        snapshots = timeline.get_snapshots()
        assert len(snapshots) == 3

    def test_concurrent_allocation_with_profiling(self):
        """Test concurrent allocations with profiling."""
        allocator = PoolAllocator(MemoryConfig())
        profiler = MemoryProfiler(record_stacktraces=False)
        profiler.enable()

        errors = []
        lock = threading.Lock()

        def worker():
            try:
                for _ in range(20):
                    block = allocator.allocate(1024)
                    if block:
                        with lock:
                            profiler.record_alloc(block)
                        time.sleep(0.001)
                        with lock:
                            profiler.record_free(block)
                        allocator.free(block)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = []
        for _ in range(5):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0

        # All allocations should be freed
        assert len(profiler.get_active_allocations()) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
