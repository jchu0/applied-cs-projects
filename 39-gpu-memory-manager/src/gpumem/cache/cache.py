"""Memory caching and defragmentation."""

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Callable
from abc import ABC, abstractmethod
from collections import OrderedDict
from enum import Enum, auto
from dataclasses import dataclass

from ..core.memory import MemoryBlock, MemoryConfig, DeviceType

logger = logging.getLogger(__name__)


class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = auto()   # Least recently used
    LFU = auto()   # Least frequently used
    FIFO = auto()  # First in, first out
    TTL = auto()   # Time to live


class DefragmentStrategy(Enum):
    """Defragmentation strategies."""
    NONE = auto()
    COMPACT = auto()      # Compact to one end
    BEST_FIT = auto()     # Move to best-fit locations
    INCREMENTAL = auto()  # Defragment incrementally


@dataclass
class CacheEntry:
    """Entry in memory cache."""
    key: Any
    block: MemoryBlock
    size: int
    access_count: int = 0
    last_access: float = 0
    created_at: float = 0
    ttl: float = 0  # 0 = no expiration


class MemoryCache(ABC):
    """Base class for memory caches."""

    @abstractmethod
    def get(self, key: Any) -> Optional[MemoryBlock]:
        """Get cached memory block."""
        pass

    @abstractmethod
    def put(self, key: Any, block: MemoryBlock, ttl: float = 0):
        """Cache memory block."""
        pass

    @abstractmethod
    def remove(self, key: Any) -> Optional[MemoryBlock]:
        """Remove from cache."""
        pass

    @abstractmethod
    def clear(self):
        """Clear all cache entries."""
        pass

    @abstractmethod
    def get_size(self) -> int:
        """Get total cached size."""
        pass


class LRUCache(MemoryCache):
    """
    Least Recently Used cache for memory blocks.

    Features:
    - O(1) get/put operations
    - Automatic eviction when full
    - TTL support
    """

    def __init__(self, max_size: int = 1024 * 1024 * 1024):  # 1GB default
        self._max_size = max_size
        self._current_size = 0
        self._lock = threading.Lock()
        self._cache: OrderedDict[Any, CacheEntry] = OrderedDict()

    def get(self, key: Any) -> Optional[MemoryBlock]:
        """Get cached block, updating LRU order."""
        with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]

            # Check TTL
            if entry.ttl > 0 and time.time() - entry.created_at > entry.ttl:
                self._remove_entry(key)
                return None

            # Update access info
            entry.access_count += 1
            entry.last_access = time.time()

            # Move to end (most recently used)
            self._cache.move_to_end(key)

            return entry.block

    def put(self, key: Any, block: MemoryBlock, ttl: float = 0):
        """Cache block, evicting if necessary."""
        with self._lock:
            # Remove existing entry
            if key in self._cache:
                self._remove_entry(key)

            # Evict until enough space
            while self._current_size + block.size > self._max_size:
                if not self._cache:
                    logger.warning("Cannot cache block: too large")
                    return
                # Remove least recently used (first item)
                oldest_key = next(iter(self._cache))
                self._remove_entry(oldest_key)

            # Add new entry
            entry = CacheEntry(
                key=key,
                block=block,
                size=block.size,
                access_count=1,
                last_access=time.time(),
                created_at=time.time(),
                ttl=ttl
            )
            self._cache[key] = entry
            self._current_size += block.size

    def remove(self, key: Any) -> Optional[MemoryBlock]:
        """Remove entry from cache."""
        with self._lock:
            if key not in self._cache:
                return None
            entry = self._cache[key]
            self._remove_entry(key)
            return entry.block

    def _remove_entry(self, key: Any):
        """Internal remove without lock."""
        entry = self._cache.pop(key)
        self._current_size -= entry.size

    def clear(self):
        """Clear all entries."""
        with self._lock:
            self._cache.clear()
            self._current_size = 0

    def get_size(self) -> int:
        """Get total cached size."""
        return self._current_size

    def cleanup_expired(self) -> int:
        """Remove expired entries."""
        with self._lock:
            current_time = time.time()
            expired = []

            for key, entry in self._cache.items():
                if entry.ttl > 0 and current_time - entry.created_at > entry.ttl:
                    expired.append(key)

            for key in expired:
                self._remove_entry(key)

            return len(expired)


class LFUCache(MemoryCache):
    """
    Least Frequently Used cache.

    Evicts based on access frequency with aging.
    """

    def __init__(self, max_size: int = 1024 * 1024 * 1024):
        self._max_size = max_size
        self._current_size = 0
        self._lock = threading.Lock()
        self._cache: Dict[Any, CacheEntry] = {}
        self._freq_map: Dict[int, List[Any]] = {}  # frequency -> keys
        self._min_freq = 0

    def get(self, key: Any) -> Optional[MemoryBlock]:
        """Get cached block, updating frequency."""
        with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]

            # Update frequency
            old_freq = entry.access_count
            entry.access_count += 1
            entry.last_access = time.time()

            # Update frequency map
            self._freq_map[old_freq].remove(key)
            if not self._freq_map[old_freq]:
                del self._freq_map[old_freq]
                if self._min_freq == old_freq:
                    self._min_freq += 1

            new_freq = entry.access_count
            if new_freq not in self._freq_map:
                self._freq_map[new_freq] = []
            self._freq_map[new_freq].append(key)

            return entry.block

    def put(self, key: Any, block: MemoryBlock, ttl: float = 0):
        """Cache block, evicting if necessary."""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)

            # Evict least frequently used
            while self._current_size + block.size > self._max_size:
                if not self._cache:
                    return
                self._evict_lfu()

            # Add entry
            entry = CacheEntry(
                key=key,
                block=block,
                size=block.size,
                access_count=1,
                last_access=time.time(),
                created_at=time.time(),
                ttl=ttl
            )
            self._cache[key] = entry
            self._current_size += block.size

            # Update frequency map
            if 1 not in self._freq_map:
                self._freq_map[1] = []
            self._freq_map[1].append(key)
            self._min_freq = 1

    def _evict_lfu(self):
        """Evict least frequently used entry."""
        if self._min_freq not in self._freq_map:
            return

        key = self._freq_map[self._min_freq][0]
        self._remove_entry(key)

    def _remove_entry(self, key: Any):
        """Remove entry from cache."""
        entry = self._cache.pop(key)
        self._current_size -= entry.size

        freq = entry.access_count
        if freq in self._freq_map:
            self._freq_map[freq].remove(key)
            if not self._freq_map[freq]:
                del self._freq_map[freq]

    def remove(self, key: Any) -> Optional[MemoryBlock]:
        with self._lock:
            if key not in self._cache:
                return None
            entry = self._cache[key]
            self._remove_entry(key)
            return entry.block

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._freq_map.clear()
            self._current_size = 0
            self._min_freq = 0

    def get_size(self) -> int:
        return self._current_size


class TensorCache:
    """
    Cache for tensor memory with shape-based hashing.

    Features:
    - Shape and dtype matching
    - Automatic resizing
    - Memory pooling
    """

    def __init__(self, max_size: int = 1024 * 1024 * 1024, policy: CachePolicy = CachePolicy.LRU):
        if policy == CachePolicy.LRU:
            self._cache = LRUCache(max_size)
        elif policy == CachePolicy.LFU:
            self._cache = LFUCache(max_size)
        else:
            self._cache = LRUCache(max_size)

        self._lock = threading.Lock()
        self._shape_map: Dict[Tuple, List[Any]] = {}  # shape -> keys

    def get_tensor(
        self,
        shape: Tuple[int, ...],
        dtype: str = "float32"
    ) -> Optional[MemoryBlock]:
        """Get cached tensor memory."""
        with self._lock:
            key = (shape, dtype)

            # Check exact match
            block = self._cache.get(key)
            if block:
                return block

            # Check for larger compatible tensor
            dtype_size = self._dtype_size(dtype)
            required_size = self._tensor_size(shape, dtype_size)

            for cached_shape in list(self._shape_map.keys()):
                cached_size = self._tensor_size(cached_shape, dtype_size)
                if cached_size >= required_size:
                    cached_key = (cached_shape, dtype)
                    block = self._cache.get(cached_key)
                    if block:
                        # Remove from shape map
                        self._shape_map[cached_shape].remove(cached_key)
                        if not self._shape_map[cached_shape]:
                            del self._shape_map[cached_shape]
                        return block

            return None

    def put_tensor(
        self,
        shape: Tuple[int, ...],
        dtype: str,
        block: MemoryBlock,
        ttl: float = 0
    ):
        """Cache tensor memory."""
        with self._lock:
            key = (shape, dtype)
            self._cache.put(key, block, ttl)

            # Update shape map
            if shape not in self._shape_map:
                self._shape_map[shape] = []
            self._shape_map[shape].append(key)

    def _tensor_size(self, shape: Tuple[int, ...], dtype_size: int) -> int:
        """Calculate tensor size in bytes."""
        import math
        return math.prod(shape) * dtype_size

    def _dtype_size(self, dtype: str) -> int:
        """Get size of dtype in bytes."""
        sizes = {
            "float16": 2,
            "float32": 4,
            "float64": 8,
            "int8": 1,
            "int16": 2,
            "int32": 4,
            "int64": 8,
            "bool": 1,
        }
        return sizes.get(dtype, 4)

    def clear(self):
        """Clear cache."""
        with self._lock:
            self._cache.clear()
            self._shape_map.clear()

    def get_size(self) -> int:
        """Get cached size."""
        return self._cache.get_size()


class Defragmenter:
    """
    Memory defragmentation utility.

    Consolidates free memory blocks.
    """

    def __init__(self, strategy: DefragmentStrategy = DefragmentStrategy.COMPACT):
        self.strategy = strategy
        self._lock = threading.Lock()

    def defragment(
        self,
        blocks: List[MemoryBlock],
        copy_fn: Callable[[int, int, int], None] = None
    ) -> List[MemoryBlock]:
        """
        Defragment memory blocks.

        Args:
            blocks: List of memory blocks
            copy_fn: Function to copy memory (src, dst, size)

        Returns:
            Defragmented block list
        """
        with self._lock:
            if self.strategy == DefragmentStrategy.NONE:
                return blocks
            elif self.strategy == DefragmentStrategy.COMPACT:
                return self._compact(blocks, copy_fn)
            elif self.strategy == DefragmentStrategy.BEST_FIT:
                return self._best_fit(blocks, copy_fn)
            elif self.strategy == DefragmentStrategy.INCREMENTAL:
                return self._incremental(blocks, copy_fn)
            else:
                return blocks

    def _compact(
        self,
        blocks: List[MemoryBlock],
        copy_fn: Callable
    ) -> List[MemoryBlock]:
        """Compact all allocated blocks to start of memory."""
        if not blocks:
            return blocks

        # Sort by allocation status and address
        allocated = [b for b in blocks if b.allocated]
        free = [b for b in blocks if not b.allocated]

        if not allocated:
            return blocks

        # Compact allocated blocks
        base_ptr = min(b.ptr for b in blocks)
        current_ptr = base_ptr

        new_blocks = []
        for block in sorted(allocated, key=lambda b: b.ptr):
            if block.ptr != current_ptr:
                # Need to move block
                if copy_fn:
                    copy_fn(block.ptr, current_ptr, block.size)
                block.ptr = current_ptr

            new_blocks.append(block)
            current_ptr += block.size

        # Create single free block
        total_free = sum(b.size for b in free)
        if total_free > 0:
            free_block = MemoryBlock(
                ptr=current_ptr,
                size=total_free,
                device=blocks[0].device,
                device_id=blocks[0].device_id,
                allocated=False
            )
            new_blocks.append(free_block)

        return new_blocks

    def _best_fit(
        self,
        blocks: List[MemoryBlock],
        copy_fn: Callable
    ) -> List[MemoryBlock]:
        """Move blocks to best-fit locations."""
        # Simplified: just compact
        return self._compact(blocks, copy_fn)

    def _incremental(
        self,
        blocks: List[MemoryBlock],
        copy_fn: Callable
    ) -> List[MemoryBlock]:
        """Incremental defragmentation - move one block at a time."""
        # Find smallest free block adjacent to allocated
        sorted_blocks = sorted(blocks, key=lambda b: b.ptr)

        for i, block in enumerate(sorted_blocks[:-1]):
            next_block = sorted_blocks[i + 1]

            if not block.allocated and next_block.allocated:
                # Can move next_block into free space
                if copy_fn:
                    copy_fn(next_block.ptr, block.ptr, next_block.size)

                # Update pointers
                old_ptr = next_block.ptr
                next_block.ptr = block.ptr
                block.ptr = block.ptr + next_block.size

                # Only do one move per call
                break

        return blocks

    def get_fragmentation_ratio(self, blocks: List[MemoryBlock]) -> float:
        """Calculate fragmentation ratio (0 = none, 1 = max)."""
        if not blocks:
            return 0.0

        free_blocks = [b for b in blocks if not b.allocated]
        if not free_blocks:
            return 0.0

        total_free = sum(b.size for b in free_blocks)
        largest_free = max(b.size for b in free_blocks)

        if total_free == 0:
            return 0.0

        return 1.0 - (largest_free / total_free)
