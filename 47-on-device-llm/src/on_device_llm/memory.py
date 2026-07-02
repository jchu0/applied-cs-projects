"""Memory management utilities for on-device LLM inference.

This module provides memory pool and KV cache implementations optimized
for on-device inference with minimal memory footprint.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class MemoryBlock:
    """A block of memory in the pool."""

    offset: int
    size: int
    in_use: bool = False
    name: Optional[str] = None


class MemoryPool:
    """
    Pre-allocated memory pool for efficient tensor allocation.

    Provides a simple bump allocator with optional free list for
    memory reuse. Designed for inference workloads where memory
    patterns are predictable.
    """

    def __init__(self, size_bytes: int, alignment: int = 64):
        """
        Initialize memory pool.

        Args:
            size_bytes: Total pool size in bytes
            alignment: Memory alignment (default 64 for cache lines)
        """
        self.size_bytes = size_bytes
        self.alignment = alignment

        # Pre-allocate the memory buffer
        self._buffer = np.zeros(size_bytes, dtype=np.uint8)

        # Track allocations
        self._blocks: List[MemoryBlock] = []
        self._next_offset = 0

        # Statistics
        self._peak_usage = 0
        self._total_allocations = 0

    def allocate(self, size: int, name: Optional[str] = None) -> np.ndarray:
        """
        Allocate memory from the pool.

        Args:
            size: Size in bytes
            name: Optional name for debugging

        Returns:
            View into the memory buffer
        """
        # Align the offset
        aligned_offset = (self._next_offset + self.alignment - 1) // self.alignment * self.alignment

        if aligned_offset + size > self.size_bytes:
            # Try to find a free block
            for block in self._blocks:
                if not block.in_use and block.size >= size:
                    block.in_use = True
                    block.name = name
                    self._total_allocations += 1
                    return self._buffer[block.offset:block.offset + size]

            raise MemoryError(
                f"Memory pool exhausted. Requested {size} bytes, "
                f"available {self.size_bytes - aligned_offset}"
            )

        # Create new block
        block = MemoryBlock(
            offset=aligned_offset,
            size=size,
            in_use=True,
            name=name
        )
        self._blocks.append(block)

        self._next_offset = aligned_offset + size
        self._peak_usage = max(self._peak_usage, self._next_offset)
        self._total_allocations += 1

        return self._buffer[aligned_offset:aligned_offset + size]

    def allocate_tensor(self, shape: Tuple[int, ...], dtype: np.dtype = np.float32,
                        name: Optional[str] = None) -> np.ndarray:
        """
        Allocate a tensor from the pool.

        Args:
            shape: Tensor shape
            dtype: Data type
            name: Optional name for debugging

        Returns:
            Tensor view into the memory buffer
        """
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        buffer = self.allocate(size, name)
        return np.frombuffer(buffer, dtype=dtype).reshape(shape)

    def free(self, name: str) -> bool:
        """
        Free a named allocation.

        Args:
            name: Name of the allocation to free

        Returns:
            True if found and freed, False otherwise
        """
        for block in self._blocks:
            if block.name == name and block.in_use:
                block.in_use = False
                return True
        return False

    def reset(self) -> None:
        """Reset the pool, freeing all allocations."""
        self._next_offset = 0
        self._blocks.clear()
        # Zero out the buffer
        self._buffer.fill(0)

    @property
    def used_bytes(self) -> int:
        """Current memory usage in bytes."""
        return self._next_offset

    @property
    def free_bytes(self) -> int:
        """Available memory in bytes."""
        return self.size_bytes - self._next_offset

    @property
    def peak_usage(self) -> int:
        """Peak memory usage in bytes."""
        return self._peak_usage

    def stats(self) -> Dict[str, int]:
        """Get memory pool statistics."""
        active_blocks = sum(1 for b in self._blocks if b.in_use)
        return {
            "total_bytes": self.size_bytes,
            "used_bytes": self._next_offset,
            "free_bytes": self.free_bytes,
            "peak_usage": self._peak_usage,
            "total_allocations": self._total_allocations,
            "active_blocks": active_blocks,
            "total_blocks": len(self._blocks),
        }


@dataclass
class KVCache:
    """
    CPU-optimized KV cache with sliding window support.

    Stores key-value pairs for all layers to enable efficient
    autoregressive generation without recomputing past attention.
    """

    num_layers: int
    num_heads: int
    head_dim: int
    max_seq_len: int
    window_size: Optional[int] = None

    # Cache storage - initialized in __post_init__
    k_cache: np.ndarray = field(default=None, repr=False)
    v_cache: np.ndarray = field(default=None, repr=False)

    # Current sequence length
    length: int = field(default=0, init=False)

    def __post_init__(self):
        """Initialize cache arrays."""
        if self.k_cache is None:
            self.k_cache = np.zeros(
                (self.num_layers, self.max_seq_len, self.num_heads, self.head_dim),
                dtype=np.float32
            )
        if self.v_cache is None:
            self.v_cache = np.zeros_like(self.k_cache)

    def append(self, layer_idx: int, k: np.ndarray, v: np.ndarray) -> None:
        """
        Append key-value pair to cache.

        Args:
            layer_idx: Layer index
            k: Key tensor [num_heads, head_dim]
            v: Value tensor [num_heads, head_dim]
        """
        if self.window_size and self.length >= self.window_size:
            # Shift window - move all entries back by 1
            self.k_cache[layer_idx, :-1] = self.k_cache[layer_idx, 1:]
            self.v_cache[layer_idx, :-1] = self.v_cache[layer_idx, 1:]
            pos = self.window_size - 1
        else:
            pos = self.length

        self.k_cache[layer_idx, pos] = k
        self.v_cache[layer_idx, pos] = v

        # Only increment length after processing last layer
        if layer_idx == self.num_layers - 1:
            if not self.window_size or self.length < self.window_size:
                self.length += 1

    def get(self, layer_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get cached key-value pairs for a layer.

        Args:
            layer_idx: Layer index

        Returns:
            Tuple of (keys, values) with shape [length, num_heads, head_dim]
        """
        if self.window_size:
            length = min(self.length, self.window_size)
        else:
            length = self.length

        return (
            self.k_cache[layer_idx, :length],
            self.v_cache[layer_idx, :length]
        )

    def get_layer_slice(self, layer_idx: int, start: int, end: int
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a slice of cached key-value pairs.

        Args:
            layer_idx: Layer index
            start: Start position
            end: End position

        Returns:
            Tuple of (keys, values) slices
        """
        return (
            self.k_cache[layer_idx, start:end],
            self.v_cache[layer_idx, start:end]
        )

    def clear(self) -> None:
        """Clear the cache."""
        self.k_cache.fill(0)
        self.v_cache.fill(0)
        self.length = 0

    def memory_usage(self) -> int:
        """Get memory usage in bytes."""
        return self.k_cache.nbytes + self.v_cache.nbytes

    def resize(self, new_max_seq_len: int) -> None:
        """
        Resize the cache to a new maximum sequence length.

        Args:
            new_max_seq_len: New maximum sequence length
        """
        if new_max_seq_len <= self.max_seq_len:
            # Just update the limit, no reallocation needed
            self.max_seq_len = new_max_seq_len
            self.length = min(self.length, new_max_seq_len)
            return

        # Allocate new cache
        new_k_cache = np.zeros(
            (self.num_layers, new_max_seq_len, self.num_heads, self.head_dim),
            dtype=np.float32
        )
        new_v_cache = np.zeros_like(new_k_cache)

        # Copy existing data
        copy_len = min(self.length, new_max_seq_len)
        new_k_cache[:, :copy_len] = self.k_cache[:, :copy_len]
        new_v_cache[:, :copy_len] = self.v_cache[:, :copy_len]

        self.k_cache = new_k_cache
        self.v_cache = new_v_cache
        self.max_seq_len = new_max_seq_len

    @property
    def effective_length(self) -> int:
        """Get effective cache length considering window size."""
        if self.window_size:
            return min(self.length, self.window_size)
        return self.length


class TensorCache:
    """
    LRU cache for tensor data.

    Useful for caching dequantized tensors to avoid repeated
    dequantization during inference.
    """

    def __init__(self, max_entries: int = 100, max_bytes: Optional[int] = None):
        """
        Initialize tensor cache.

        Args:
            max_entries: Maximum number of cached tensors
            max_bytes: Maximum total cache size in bytes
        """
        self.max_entries = max_entries
        self.max_bytes = max_bytes

        self._cache: Dict[str, np.ndarray] = {}
        self._access_order: List[str] = []
        self._total_bytes = 0

    def get(self, key: str) -> Optional[np.ndarray]:
        """
        Get a cached tensor.

        Args:
            key: Cache key

        Returns:
            Cached tensor or None
        """
        if key in self._cache:
            # Move to end of access order (most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None

    def put(self, key: str, tensor: np.ndarray) -> None:
        """
        Cache a tensor.

        Args:
            key: Cache key
            tensor: Tensor to cache
        """
        tensor_bytes = tensor.nbytes

        # If the key already exists, drop the old entry first so byte
        # accounting and access order stay consistent (treat as an update).
        if key in self._cache:
            self._total_bytes -= self._cache[key].nbytes
            self._access_order.remove(key)
            del self._cache[key]

        # Evict if necessary
        while len(self._cache) >= self.max_entries:
            self._evict_oldest()

        if self.max_bytes:
            while self._total_bytes + tensor_bytes > self.max_bytes and self._cache:
                self._evict_oldest()

        self._cache[key] = tensor
        self._access_order.append(key)
        self._total_bytes += tensor_bytes

    def _evict_oldest(self) -> None:
        """Evict the least recently used entry."""
        if not self._access_order:
            return

        oldest_key = self._access_order.pop(0)
        if oldest_key in self._cache:
            self._total_bytes -= self._cache[oldest_key].nbytes
            del self._cache[oldest_key]

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()
        self._total_bytes = 0

    def stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "entries": len(self._cache),
            "total_bytes": self._total_bytes,
            "max_entries": self.max_entries,
            "max_bytes": self.max_bytes or 0,
        }
