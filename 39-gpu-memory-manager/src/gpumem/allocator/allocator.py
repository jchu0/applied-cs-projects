"""Memory allocator implementations."""

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from collections import defaultdict

from ..core.memory import (
    MemoryBlock,
    MemoryPool,
    MemoryConfig,
    MemoryStats,
    AllocationRequest,
    DeviceType,
)

logger = logging.getLogger(__name__)


class Allocator(ABC):
    """Base class for memory allocators."""

    @abstractmethod
    def allocate(self, size: int, stream: int = 0) -> Optional[MemoryBlock]:
        """Allocate memory block."""
        pass

    @abstractmethod
    def free(self, block: MemoryBlock):
        """Free memory block."""
        pass

    @abstractmethod
    def get_stats(self) -> MemoryStats:
        """Get memory statistics."""
        pass

    @abstractmethod
    def empty_cache(self):
        """Release cached memory."""
        pass


class CachingAllocator(Allocator):
    """
    CUDA-style caching allocator.

    Features:
    - Caches freed blocks for reuse
    - Stream-ordered allocation
    - Automatic coalescing
    - OOM handling with cache flush
    """

    def __init__(self, config: MemoryConfig = None):
        self.config = config or MemoryConfig()
        self._lock = threading.RLock()
        self._pools: Dict[int, MemoryPool] = {}  # stream -> pool
        self._default_pool = MemoryPool(self.config)

        # Allocator-level high-water marks across all pools
        self._peak_allocated = 0
        self._peak_reserved = 0

        # Small/large allocation split
        self._small_threshold = 1024 * 1024  # 1MB
        self._small_pools: Dict[int, List[MemoryBlock]] = defaultdict(list)

    def allocate(self, size: int, stream: int = 0) -> Optional[MemoryBlock]:
        """Allocate memory block."""
        with self._lock:
            request = AllocationRequest(
                size=size,
                device=self.config.device_type,
                device_id=self.config.device_id,
                stream=stream
            )

            # Try small block allocation
            if size <= self._small_threshold:
                block = self._allocate_small(size, stream)
                if block:
                    self._update_peak_stats()
                    return block

            # Use pool allocation
            pool = self._get_pool(stream)
            block = pool.allocate(request)

            if block is None:
                # OOM - try to free cache
                logger.info("OOM, attempting cache flush")
                self.empty_cache()
                block = pool.allocate(request)

            if block is not None:
                self._update_peak_stats()

            return block

    def free(self, block: MemoryBlock):
        """Free memory block."""
        with self._lock:
            if block.size <= self._small_threshold:
                self._free_small(block)
            else:
                pool = self._get_pool(block.stream)
                pool.free(block)

    def _allocate_small(self, size: int, stream: int) -> Optional[MemoryBlock]:
        """Allocate from small block cache."""
        size_class = self._small_size_class(size)
        key = (stream, size_class)

        if key in self._small_pools and self._small_pools[key]:
            block = self._small_pools[key].pop()
            block.allocated = True
            block.timestamp = time.time()
            return block

        return None

    def _free_small(self, block: MemoryBlock):
        """Return small block to cache."""
        size_class = self._small_size_class(block.size)
        key = (block.stream, size_class)

        block.allocated = False
        self._small_pools[key].append(block)

    def _small_size_class(self, size: int) -> int:
        """Get size class for small allocations."""
        if size <= 512:
            return 512
        elif size <= 1024:
            return 1024
        elif size <= 2048:
            return 2048
        else:
            # Round to nearest 4KB
            return ((size + 4095) // 4096) * 4096

    def _get_pool(self, stream: int) -> MemoryPool:
        """Get or create pool for stream."""
        if stream not in self._pools:
            self._pools[stream] = MemoryPool(self.config)
        return self._pools[stream]

    def _update_peak_stats(self):
        """Update high-water marks from current totals across all pools."""
        allocated = 0
        reserved = 0
        for pool in list(self._pools.values()) + [self._default_pool]:
            pool_stats = pool.get_stats()
            allocated += pool_stats.allocated
            reserved += pool_stats.reserved

        self._peak_allocated = max(self._peak_allocated, allocated)
        self._peak_reserved = max(self._peak_reserved, reserved)

    def get_stats(self) -> MemoryStats:
        """Get combined statistics."""
        stats = MemoryStats()

        # Combine pool stats
        for pool in self._pools.values():
            pool_stats = pool.get_stats()
            stats.allocated += pool_stats.allocated
            stats.reserved += pool_stats.reserved
            stats.active += pool_stats.active
            stats.inactive += pool_stats.inactive
            stats.num_allocs += pool_stats.num_allocs
            stats.num_frees += pool_stats.num_frees
            stats.num_ooms += pool_stats.num_ooms

        # Default pool
        default_stats = self._default_pool.get_stats()
        stats.allocated += default_stats.allocated
        stats.reserved += default_stats.reserved
        stats.active += default_stats.active
        stats.inactive += default_stats.inactive

        stats.peak_allocated = max(
            self._peak_allocated,
            default_stats.peak_allocated
        )
        stats.peak_reserved = max(
            self._peak_reserved,
            default_stats.peak_reserved
        )

        return stats

    def empty_cache(self):
        """Release all cached memory."""
        with self._lock:
            for pool in self._pools.values():
                pool.empty_cache()
            self._default_pool.empty_cache()
            self._small_pools.clear()


class PoolAllocator(Allocator):
    """
    Simple pool-based allocator.

    Pre-allocates fixed-size pools for different size classes.
    """

    def __init__(self, config: MemoryConfig = None):
        self.config = config or MemoryConfig()
        self._lock = threading.Lock()

        # Size class pools
        self._size_classes = [
            256, 512, 1024, 2048, 4096,  # Small
            8192, 16384, 32768, 65536,   # Medium
            131072, 262144, 524288, 1048576,  # Large
        ]
        self._pools: Dict[int, List[MemoryBlock]] = {
            sc: [] for sc in self._size_classes
        }
        self._allocated: Dict[int, MemoryBlock] = {}
        self._stats = MemoryStats()
        self._next_ptr = 0x200000

    def allocate(self, size: int, stream: int = 0) -> Optional[MemoryBlock]:
        """Allocate from pool."""
        with self._lock:
            size_class = self._get_size_class(size)

            # Try to get from pool
            if self._pools[size_class]:
                block = self._pools[size_class].pop()
            else:
                # Allocate new block
                block = MemoryBlock(
                    ptr=self._next_ptr,
                    size=size_class,
                    device=self.config.device_type,
                    device_id=self.config.device_id,
                    stream=stream
                )
                self._next_ptr += size_class
                self._stats.reserved += size_class
                self._stats.peak_reserved = max(
                    self._stats.peak_reserved,
                    self._stats.reserved
                )

            block.allocated = True
            block.timestamp = time.time()
            self._allocated[block.ptr] = block

            self._stats.allocated += size_class
            self._stats.active += size_class
            self._stats.num_allocs += 1
            self._stats.peak_allocated = max(
                self._stats.peak_allocated,
                self._stats.allocated
            )

            return block

    def free(self, block: MemoryBlock):
        """Return block to pool."""
        with self._lock:
            if block.ptr not in self._allocated:
                return

            del self._allocated[block.ptr]
            block.allocated = False

            size_class = self._get_size_class(block.size)
            self._pools[size_class].append(block)

            self._stats.allocated -= block.size
            self._stats.active -= block.size
            self._stats.inactive += block.size
            self._stats.num_frees += 1

    def _get_size_class(self, size: int) -> int:
        """Get appropriate size class."""
        for sc in self._size_classes:
            if sc >= size:
                return sc
        # Larger than any class - round up
        return ((size + self._size_classes[-1] - 1) //
                self._size_classes[-1]) * self._size_classes[-1]

    def get_stats(self) -> MemoryStats:
        return self._stats

    def empty_cache(self):
        with self._lock:
            for size_class in self._size_classes:
                self._pools[size_class].clear()
            self._stats.inactive = 0


class BuddyAllocator(Allocator):
    """
    Buddy system allocator.

    Efficient for power-of-2 allocations with fast coalescing.
    """

    def __init__(self, config: MemoryConfig = None, total_size: int = 1024 * 1024 * 1024):
        self.config = config or MemoryConfig()
        self._lock = threading.Lock()
        self._total_size = total_size

        # Min/max block sizes (powers of 2)
        self._min_order = 9   # 512 bytes
        self._max_order = 30  # 1GB

        # Free lists for each order
        self._free_lists: Dict[int, List[Tuple[int, int]]] = {
            order: [] for order in range(self._min_order, self._max_order + 1)
        }

        # Initialize with single max-order block
        max_order = self._get_order(total_size)
        self._free_lists[max_order].append((0, total_size))

        self._allocated: Dict[int, Tuple[int, int]] = {}  # ptr -> (size, order)
        self._stats = MemoryStats()
        self._stats.reserved = total_size
        self._stats.peak_reserved = total_size

    def allocate(self, size: int, stream: int = 0) -> Optional[MemoryBlock]:
        """Allocate using buddy system."""
        with self._lock:
            order = self._get_order(size)
            block_size = 2 ** order

            # Find free block
            ptr = self._find_block(order)
            if ptr is None:
                self._stats.num_ooms += 1
                return None

            block = MemoryBlock(
                ptr=ptr,
                size=block_size,
                device=self.config.device_type,
                device_id=self.config.device_id,
                stream=stream
            )

            self._allocated[ptr] = (block_size, order)
            self._stats.allocated += block_size
            self._stats.active += block_size
            self._stats.num_allocs += 1
            self._stats.peak_allocated = max(
                self._stats.peak_allocated,
                self._stats.allocated
            )

            return block

    def free(self, block: MemoryBlock):
        """Free block and coalesce buddies."""
        with self._lock:
            if block.ptr not in self._allocated:
                return

            size, order = self._allocated[block.ptr]
            del self._allocated[block.ptr]

            # Coalesce with buddy
            self._coalesce(block.ptr, order)

            self._stats.allocated -= size
            self._stats.active -= size
            self._stats.inactive += size
            self._stats.num_frees += 1

    def _get_order(self, size: int) -> int:
        """Get order (power of 2) for size."""
        order = self._min_order
        while (2 ** order) < size:
            order += 1
        return min(order, self._max_order)

    def _find_block(self, order: int) -> Optional[int]:
        """Find or split block of given order."""
        # Look for exact match
        if self._free_lists[order]:
            ptr, _ = self._free_lists[order].pop()
            return ptr

        # Split larger block
        for larger_order in range(order + 1, self._max_order + 1):
            if self._free_lists[larger_order]:
                return self._split_block(larger_order, order)

        return None

    def _split_block(self, from_order: int, to_order: int) -> int:
        """Split block from larger order to smaller."""
        ptr, size = self._free_lists[from_order].pop()

        # Split recursively
        current_order = from_order
        while current_order > to_order:
            current_order -= 1
            buddy_size = 2 ** current_order

            # Add buddy to free list
            buddy_ptr = ptr + buddy_size
            self._free_lists[current_order].append((buddy_ptr, buddy_size))

        return ptr

    def _coalesce(self, ptr: int, order: int):
        """Coalesce block with its buddy."""
        while order < self._max_order:
            block_size = 2 ** order
            buddy_ptr = ptr ^ block_size  # XOR to find buddy

            # Check if buddy is free
            buddy_found = False
            for i, (bptr, bsize) in enumerate(self._free_lists[order]):
                if bptr == buddy_ptr:
                    self._free_lists[order].pop(i)
                    buddy_found = True
                    break

            if not buddy_found:
                break

            # Merge with buddy
            ptr = min(ptr, buddy_ptr)
            order += 1

        # Add coalesced block to free list
        self._free_lists[order].append((ptr, 2 ** order))

    def get_stats(self) -> MemoryStats:
        return self._stats

    def empty_cache(self):
        pass  # No caching in buddy allocator


class SlabAllocator(Allocator):
    """
    Slab allocator for fixed-size objects.

    Efficient for allocating many objects of the same size.
    """

    def __init__(self, config: MemoryConfig = None, object_size: int = 64):
        self.config = config or MemoryConfig()
        self._lock = threading.Lock()
        self._object_size = object_size

        # Slab parameters
        self._slab_size = 64 * 1024  # 64KB slabs
        self._objects_per_slab = self._slab_size // object_size

        # Slab management
        self._slabs: List[List[int]] = []  # List of slabs (list of free object ptrs)
        self._partial_slabs: List[int] = []  # Indices of partial slabs
        self._full_slabs: List[int] = []  # Indices of full slabs

        self._allocated: Dict[int, int] = {}  # ptr -> slab_index
        self._stats = MemoryStats()
        self._next_ptr = 0x300000

    def allocate(self, size: int = 0, stream: int = 0) -> Optional[MemoryBlock]:
        """Allocate object from slab."""
        with self._lock:
            # Find partial slab
            if self._partial_slabs:
                slab_idx = self._partial_slabs[0]
                ptr = self._slabs[slab_idx].pop()

                if not self._slabs[slab_idx]:
                    # Slab is now full
                    self._partial_slabs.remove(slab_idx)
                    self._full_slabs.append(slab_idx)
            else:
                # Allocate new slab
                slab_idx = self._allocate_slab()
                ptr = self._slabs[slab_idx].pop()

            block = MemoryBlock(
                ptr=ptr,
                size=self._object_size,
                device=self.config.device_type,
                device_id=self.config.device_id,
                stream=stream
            )

            self._allocated[ptr] = slab_idx
            self._stats.allocated += self._object_size
            self._stats.active += self._object_size
            self._stats.num_allocs += 1
            self._stats.peak_allocated = max(
                self._stats.peak_allocated,
                self._stats.allocated
            )

            return block

    def free(self, block: MemoryBlock):
        """Return object to slab."""
        with self._lock:
            if block.ptr not in self._allocated:
                return

            slab_idx = self._allocated[block.ptr]
            del self._allocated[block.ptr]

            # Return to slab
            was_full = slab_idx in self._full_slabs
            self._slabs[slab_idx].append(block.ptr)

            if was_full:
                # Slab is now partial
                self._full_slabs.remove(slab_idx)
                self._partial_slabs.append(slab_idx)

            self._stats.allocated -= self._object_size
            self._stats.active -= self._object_size
            self._stats.inactive += self._object_size
            self._stats.num_frees += 1

    def _allocate_slab(self) -> int:
        """Allocate a new slab."""
        slab_idx = len(self._slabs)
        slab = []

        # Create free list
        base_ptr = self._next_ptr
        for i in range(self._objects_per_slab):
            slab.append(base_ptr + i * self._object_size)

        self._slabs.append(slab)
        self._partial_slabs.append(slab_idx)
        self._next_ptr += self._slab_size
        self._stats.reserved += self._slab_size
        self._stats.peak_reserved = max(
            self._stats.peak_reserved,
            self._stats.reserved
        )

        return slab_idx

    def get_stats(self) -> MemoryStats:
        return self._stats

    def empty_cache(self):
        pass  # Slabs are not cached


class UnifiedMemoryAllocator(Allocator):
    """
    Unified memory allocator for CPU and GPU.

    Provides automatic data migration between devices.
    """

    def __init__(self, config: MemoryConfig = None):
        self.config = config or MemoryConfig()
        self._lock = threading.Lock()
        self._allocators: Dict[DeviceType, Allocator] = {
            DeviceType.CPU: CachingAllocator(config),
            DeviceType.CUDA: CachingAllocator(config),
        }
        self._mappings: Dict[int, Tuple[int, DeviceType]] = {}  # unified_ptr -> (device_ptr, device)

    def allocate(self, size: int, stream: int = 0) -> Optional[MemoryBlock]:
        """Allocate unified memory."""
        with self._lock:
            # Allocate on both devices
            cpu_block = self._allocators[DeviceType.CPU].allocate(size, stream)
            gpu_block = self._allocators[DeviceType.CUDA].allocate(size, stream)

            if cpu_block is None or gpu_block is None:
                return None

            # Create unified block (use CPU ptr as unified ptr)
            block = MemoryBlock(
                ptr=cpu_block.ptr,
                size=size,
                device=DeviceType.CPU,
                device_id=0,
                stream=stream
            )

            # Store mappings
            self._mappings[block.ptr] = {
                DeviceType.CPU: cpu_block.ptr,
                DeviceType.CUDA: gpu_block.ptr,
            }

            return block

    def free(self, block: MemoryBlock):
        """Free unified memory."""
        with self._lock:
            if block.ptr not in self._mappings:
                return

            mappings = self._mappings[block.ptr]
            del self._mappings[block.ptr]

            # Free on both devices
            for device_type, ptr in mappings.items():
                device_block = MemoryBlock(
                    ptr=ptr,
                    size=block.size,
                    device=device_type,
                    device_id=0
                )
                self._allocators[device_type].free(device_block)

    def get_device_ptr(self, unified_ptr: int, device: DeviceType) -> int:
        """Get device-specific pointer."""
        if unified_ptr in self._mappings:
            return self._mappings[unified_ptr].get(device, unified_ptr)
        return unified_ptr

    def get_stats(self) -> MemoryStats:
        """Get combined statistics."""
        stats = MemoryStats()
        for allocator in self._allocators.values():
            alloc_stats = allocator.get_stats()
            stats.allocated += alloc_stats.allocated
            stats.reserved += alloc_stats.reserved
            stats.active += alloc_stats.active
        return stats

    def empty_cache(self):
        for allocator in self._allocators.values():
            allocator.empty_cache()
