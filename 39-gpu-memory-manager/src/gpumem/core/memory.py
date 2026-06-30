"""Core memory management primitives."""

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Device types for memory allocation."""
    CPU = auto()
    CUDA = auto()
    ROCM = auto()
    METAL = auto()
    VULKAN = auto()


@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    device_type: DeviceType = DeviceType.CPU
    device_id: int = 0
    max_memory: int = 0  # 0 = unlimited
    min_split_size: int = 512  # Minimum size to split blocks
    max_split_size: int = 200 * 1024 * 1024  # 200MB
    garbage_collection_threshold: float = 0.8  # Trigger GC at 80%
    expandable_segments: bool = True
    release_threshold: float = 0.95  # Release memory at 95%
    alignment: int = 512  # Memory alignment in bytes
    round_up_power2: bool = True  # Round allocations to power of 2


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    allocated: int = 0
    reserved: int = 0
    active: int = 0
    inactive: int = 0
    freed: int = 0

    num_allocs: int = 0
    num_frees: int = 0
    num_ooms: int = 0

    peak_allocated: int = 0
    peak_reserved: int = 0

    def __repr__(self):
        return (
            f"MemoryStats(allocated={self.allocated / 1e6:.1f}MB, "
            f"reserved={self.reserved / 1e6:.1f}MB, "
            f"active={self.active / 1e6:.1f}MB)"
        )


@dataclass
class MemoryBlock:
    """A block of allocated memory."""
    ptr: int  # Memory address (simulated)
    size: int
    device: DeviceType
    device_id: int

    allocated: bool = True
    stream: int = 0  # CUDA stream
    timestamp: float = field(default_factory=time.time)

    # Block management
    prev: Optional['MemoryBlock'] = None
    next: Optional['MemoryBlock'] = None
    pool: Optional['MemoryPool'] = None

    # Metadata
    tag: str = ""
    stacktrace: str = ""

    def __hash__(self):
        return hash(self.ptr)

    def __eq__(self, other):
        if not isinstance(other, MemoryBlock):
            return False
        return self.ptr == other.ptr

    def split(self, size: int) -> Optional['MemoryBlock']:
        """Split this block, return the remaining portion."""
        if size >= self.size:
            return None

        remaining_size = self.size - size
        remaining = MemoryBlock(
            ptr=self.ptr + size,
            size=remaining_size,
            device=self.device,
            device_id=self.device_id,
            allocated=False,
            stream=self.stream,
            pool=self.pool
        )

        self.size = size

        # Update linked list
        remaining.next = self.next
        remaining.prev = self
        if self.next:
            self.next.prev = remaining
        self.next = remaining

        return remaining

    def merge(self, other: 'MemoryBlock') -> bool:
        """Merge with adjacent block."""
        if other.ptr != self.ptr + self.size:
            return False

        self.size += other.size
        self.next = other.next
        if other.next:
            other.next.prev = self

        return True

    def can_merge_with(self, other: 'MemoryBlock') -> bool:
        """Check if can merge with adjacent block."""
        return (
            not other.allocated and
            other.stream == self.stream and
            other.ptr == self.ptr + self.size
        )


@dataclass
class MemoryRegion:
    """A contiguous region of device memory."""
    start: int
    size: int
    device: DeviceType
    device_id: int

    blocks: List[MemoryBlock] = field(default_factory=list)
    free_size: int = 0

    def __post_init__(self):
        self.free_size = self.size


@dataclass
class AllocationRequest:
    """Request for memory allocation."""
    size: int
    device: DeviceType = DeviceType.CPU
    device_id: int = 0
    stream: int = 0
    tag: str = ""

    # Preferences
    alignment: int = 512
    zero_init: bool = False
    persistent: bool = False  # Don't cache


class MemoryPool:
    """
    Pool of memory blocks for efficient allocation.

    Features:
    - Block coalescing
    - Size-class bucketing
    - Stream-aware allocation
    """

    def __init__(self, config: MemoryConfig):
        self.config = config
        self._lock = threading.Lock()
        self._regions: List[MemoryRegion] = []
        self._free_blocks: Dict[int, List[MemoryBlock]] = {}  # size_class -> blocks
        self._allocated_blocks: Dict[int, MemoryBlock] = {}  # ptr -> block
        self._stats = MemoryStats()
        self._next_ptr = 0x100000  # Simulated memory addresses

        # Size classes (powers of 2)
        self._size_classes = [2**i for i in range(9, 31)]  # 512B to 1GB

    def _get_size_class(self, size: int) -> int:
        """Get size class for allocation size."""
        for sc in self._size_classes:
            if sc >= size:
                return sc
        return size

    def _round_size(self, size: int) -> int:
        """Round size to alignment."""
        alignment = self.config.alignment
        return ((size + alignment - 1) // alignment) * alignment

    def allocate(self, request: AllocationRequest) -> Optional[MemoryBlock]:
        """Allocate memory block."""
        with self._lock:
            size = self._round_size(request.size)
            size_class = self._get_size_class(size)

            # Try to find free block
            block = self._find_free_block(size_class, request.stream)

            if block is None:
                # Allocate new region
                block = self._allocate_new_region(size, request)
                if block is None:
                    self._stats.num_ooms += 1
                    return None

            # Split block if too large
            if block.size > size + self.config.min_split_size:
                remaining = block.split(size)
                if remaining:
                    self._add_free_block(remaining)

            block.allocated = True
            block.tag = request.tag
            block.timestamp = time.time()

            self._allocated_blocks[block.ptr] = block
            self._update_stats_alloc(block.size)

            return block

    def free(self, block: MemoryBlock):
        """Free memory block."""
        with self._lock:
            if block.ptr not in self._allocated_blocks:
                logger.warning(f"Attempting to free unknown block: {block.ptr}")
                return

            del self._allocated_blocks[block.ptr]
            block.allocated = False
            block.timestamp = time.time()

            # Coalesce with neighbors
            self._coalesce(block)

            # Add to free list
            self._add_free_block(block)
            self._update_stats_free(block.size)

    def _find_free_block(self, size_class: int, stream: int) -> Optional[MemoryBlock]:
        """Find a free block of appropriate size."""
        # Look in exact size class first
        if size_class in self._free_blocks:
            blocks = self._free_blocks[size_class]
            for i, block in enumerate(blocks):
                if block.stream == stream or stream == 0:
                    blocks.pop(i)
                    return block

        # Look in larger size classes
        for sc in self._size_classes:
            if sc > size_class and sc in self._free_blocks:
                blocks = self._free_blocks[sc]
                for i, block in enumerate(blocks):
                    if block.stream == stream or stream == 0:
                        blocks.pop(i)
                        return block

        return None

    def _add_free_block(self, block: MemoryBlock):
        """Add block to free list."""
        size_class = self._get_size_class(block.size)
        if size_class not in self._free_blocks:
            self._free_blocks[size_class] = []
        self._free_blocks[size_class].append(block)

    def _allocate_new_region(
        self,
        size: int,
        request: AllocationRequest
    ) -> Optional[MemoryBlock]:
        """Allocate a new memory region."""
        # Check memory limits
        if (self.config.max_memory > 0 and
            self._stats.reserved + size > self.config.max_memory):
            return None

        # Create new region
        region_size = max(size, 2 * 1024 * 1024)  # At least 2MB
        region = MemoryRegion(
            start=self._next_ptr,
            size=region_size,
            device=request.device,
            device_id=request.device_id
        )
        self._next_ptr += region_size
        self._regions.append(region)

        # Create block
        block = MemoryBlock(
            ptr=region.start,
            size=region_size,
            device=request.device,
            device_id=request.device_id,
            allocated=False,
            stream=request.stream,
            pool=self
        )
        region.blocks.append(block)

        self._stats.reserved += region_size
        self._stats.peak_reserved = max(
            self._stats.peak_reserved,
            self._stats.reserved
        )

        return block

    def _coalesce(self, block: MemoryBlock):
        """Coalesce block with free neighbors."""
        # Merge with next block
        if block.next and block.can_merge_with(block.next):
            next_block = block.next
            # Remove from free list
            self._remove_from_free_list(next_block)
            block.merge(next_block)

        # Merge with previous block
        if block.prev and not block.prev.allocated:
            prev_block = block.prev
            if prev_block.can_merge_with(block):
                self._remove_from_free_list(prev_block)
                prev_block.merge(block)
                # Use merged block
                return prev_block

        return block

    def _remove_from_free_list(self, block: MemoryBlock):
        """Remove block from free list."""
        size_class = self._get_size_class(block.size)
        if size_class in self._free_blocks:
            blocks = self._free_blocks[size_class]
            if block in blocks:
                blocks.remove(block)

    def _update_stats_alloc(self, size: int):
        """Update stats for allocation."""
        self._stats.allocated += size
        self._stats.active += size
        self._stats.num_allocs += 1
        self._stats.peak_allocated = max(
            self._stats.peak_allocated,
            self._stats.allocated
        )

    def _update_stats_free(self, size: int):
        """Update stats for free."""
        self._stats.allocated -= size
        self._stats.active -= size
        self._stats.inactive += size
        self._stats.num_frees += 1

    def get_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        return self._stats

    def empty_cache(self):
        """Release all cached memory."""
        with self._lock:
            self._free_blocks.clear()
            self._stats.inactive = 0

    def trim(self, target_size: int = 0):
        """Trim pool to target size by releasing unused memory."""
        with self._lock:
            if self._stats.inactive <= target_size:
                return

            to_free = self._stats.inactive - target_size
            freed = 0

            # Free largest blocks first
            for size_class in sorted(self._free_blocks.keys(), reverse=True):
                if freed >= to_free:
                    break

                blocks = self._free_blocks[size_class]
                while blocks and freed < to_free:
                    block = blocks.pop()
                    freed += block.size
                    self._stats.inactive -= block.size
                    self._stats.reserved -= block.size

    def reset_stats(self):
        """Reset statistics."""
        with self._lock:
            self._stats = MemoryStats(
                reserved=self._stats.reserved,
                allocated=self._stats.allocated,
                active=self._stats.active,
                inactive=self._stats.inactive
            )
