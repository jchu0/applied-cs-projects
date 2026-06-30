"""KV Cache management for autoregressive inference.

This module implements paged KV cache management for efficient memory utilization,
including block allocation, sliding window, and copy-on-write support.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import threading

# Try to import torch, fall back to numpy for testing
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import numpy as np


def _to_torch_if_needed(x, target):
    """Convert x to match target's type (torch tensor or numpy array)."""
    if HAS_TORCH and isinstance(target, torch.Tensor):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(target.device, target.dtype)
        return x
    return x


@dataclass
class KVCacheConfig:
    """Configuration for KV cache."""
    num_layers: int
    num_heads: int
    head_dim: int
    max_seq_len: int
    block_size: int = 16  # Tokens per block
    dtype: str = "float16"
    device: str = "cpu"

    def get_block_memory_bytes(self) -> int:
        """Calculate memory per block in bytes."""
        dtype_size = 2 if self.dtype == "float16" else 4
        return (
            self.num_layers * 2 * self.num_heads *
            self.block_size * self.head_dim * dtype_size
        )


class KVCacheBlock:
    """A single block of KV cache memory."""

    def __init__(self, block_id: int, config: KVCacheConfig):
        """Initialize a KV cache block.

        Args:
            block_id: Unique identifier for this block.
            config: KV cache configuration.
        """
        self.block_id = block_id
        self.config = config

        # Allocate memory: [num_layers, 2, num_heads, block_size, head_dim]
        # 2 is for K and V
        if HAS_TORCH:
            dtype = torch.float16 if config.dtype == "float16" else torch.float32
            self.data = torch.zeros(
                config.num_layers,
                2,  # K, V
                config.num_heads,
                config.block_size,
                config.head_dim,
                dtype=dtype,
                device=config.device
            )
        else:
            dtype = np.float16 if config.dtype == "float16" else np.float32
            self.data = np.zeros(
                (config.num_layers, 2, config.num_heads,
                 config.block_size, config.head_dim),
                dtype=dtype
            )

        self.num_tokens = 0
        self.ref_count = 0  # For copy-on-write

    def append(self, layer_idx: int, k: Any, v: Any) -> int:
        """Append KV to block at specific layer.

        Args:
            layer_idx: The layer index.
            k: Key tensor of shape [num_heads, head_dim].
            v: Value tensor of shape [num_heads, head_dim].

        Returns:
            Position within the block where data was stored.

        Raises:
            RuntimeError: If block is full.
        """
        if self.num_tokens >= self.config.block_size:
            raise RuntimeError("Block is full")

        pos = self.num_tokens
        self.data[layer_idx, 0, :, pos, :] = _to_torch_if_needed(k, self.data)
        self.data[layer_idx, 1, :, pos, :] = _to_torch_if_needed(v, self.data)

        # Only increment token count on last layer
        if layer_idx == self.config.num_layers - 1:
            self.num_tokens += 1

        return pos

    def append_token(self, k_by_layer: List[Any], v_by_layer: List[Any]) -> int:
        """Append a full token's KV across all layers.

        Args:
            k_by_layer: List of key tensors, one per layer.
            v_by_layer: List of value tensors, one per layer.

        Returns:
            Position within the block where data was stored.

        Raises:
            RuntimeError: If block is full.
        """
        if self.num_tokens >= self.config.block_size:
            raise RuntimeError("Block is full")

        pos = self.num_tokens
        for layer_idx in range(self.config.num_layers):
            self.data[layer_idx, 0, :, pos, :] = _to_torch_if_needed(k_by_layer[layer_idx], self.data)
            self.data[layer_idx, 1, :, pos, :] = _to_torch_if_needed(v_by_layer[layer_idx], self.data)

        self.num_tokens += 1
        return pos

    def get_kv(self, layer_idx: int) -> Tuple[Any, Any]:
        """Get K and V tensors for a layer.

        Args:
            layer_idx: The layer index.

        Returns:
            Tuple of (K, V) tensors.
        """
        return (
            self.data[layer_idx, 0, :, :self.num_tokens, :],
            self.data[layer_idx, 1, :, :self.num_tokens, :]
        )

    def get_kv_at_position(self, layer_idx: int, position: int) -> Tuple[Any, Any]:
        """Get K and V at a specific position.

        Args:
            layer_idx: The layer index.
            position: The position within the block.

        Returns:
            Tuple of (K, V) tensors at the position.
        """
        return (
            self.data[layer_idx, 0, :, position, :],
            self.data[layer_idx, 1, :, position, :]
        )

    def is_full(self) -> bool:
        """Check if block is full."""
        return self.num_tokens >= self.config.block_size

    def reset(self) -> None:
        """Reset the block to empty state."""
        self.num_tokens = 0
        self.ref_count = 0
        if HAS_TORCH:
            self.data.zero_()
        else:
            self.data.fill(0)


class PagedKVCacheManager:
    """
    Paged KV cache manager for efficient memory utilization.

    Implements virtual memory-style paging for KV cache,
    allowing non-contiguous allocation and copy-on-write.
    """

    def __init__(self, config: KVCacheConfig, num_blocks: int):
        """Initialize paged KV cache manager.

        Args:
            config: KV cache configuration.
            num_blocks: Total number of blocks to pre-allocate.
        """
        self.config = config
        self.num_blocks = num_blocks

        # Pre-allocate all blocks
        self.blocks = [
            KVCacheBlock(i, config) for i in range(num_blocks)
        ]

        # Free block list
        self.free_blocks: List[int] = list(range(num_blocks))

        # Request to block mapping
        self.request_blocks: Dict[str, List[int]] = {}

        self._lock = threading.Lock()

    def allocate_blocks(self, request_id: str, num_blocks: int) -> List[int]:
        """Allocate blocks for a request.

        Args:
            request_id: The request identifier.
            num_blocks: Number of blocks to allocate.

        Returns:
            List of allocated block IDs, empty if not enough memory.
        """
        with self._lock:
            if len(self.free_blocks) < num_blocks:
                return []  # Not enough memory

            allocated = []
            for _ in range(num_blocks):
                block_id = self.free_blocks.pop()
                allocated.append(block_id)
                self.blocks[block_id].ref_count = 1

            if request_id in self.request_blocks:
                self.request_blocks[request_id].extend(allocated)
            else:
                self.request_blocks[request_id] = allocated

            return allocated

    def allocate_single_block(self, request_id: str) -> Optional[int]:
        """Allocate a single block for a request.

        Args:
            request_id: The request identifier.

        Returns:
            Block ID if successful, None if out of memory.
        """
        blocks = self.allocate_blocks(request_id, 1)
        return blocks[0] if blocks else None

    def free_blocks_for_request(self, request_id: str) -> int:
        """Free all blocks for a request.

        Args:
            request_id: The request identifier.

        Returns:
            Number of blocks freed.
        """
        with self._lock:
            if request_id not in self.request_blocks:
                return 0

            freed_count = 0
            for block_id in self.request_blocks[request_id]:
                block = self.blocks[block_id]
                block.ref_count -= 1

                if block.ref_count == 0:
                    block.reset()
                    self.free_blocks.append(block_id)
                    freed_count += 1

            del self.request_blocks[request_id]
            return freed_count

    def get_block(self, block_id: int) -> KVCacheBlock:
        """Get a block by ID.

        Args:
            block_id: The block identifier.

        Returns:
            The KVCacheBlock.
        """
        return self.blocks[block_id]

    def get_blocks_for_request(self, request_id: str) -> List[int]:
        """Get all block IDs for a request.

        Args:
            request_id: The request identifier.

        Returns:
            List of block IDs.
        """
        with self._lock:
            return list(self.request_blocks.get(request_id, []))

    def append_to_cache(
        self,
        request_id: str,
        layer_idx: int,
        k: Any,
        v: Any
    ) -> bool:
        """Append KV to the cache for a request.

        Args:
            request_id: The request identifier.
            layer_idx: The layer index.
            k: Key tensor.
            v: Value tensor.

        Returns:
            True if successful, False if out of memory.
        """
        with self._lock:
            blocks = self.request_blocks.get(request_id, [])
            if not blocks:
                # Allocate first block
                if not self.free_blocks:
                    return False
                block_id = self.free_blocks.pop()
                self.blocks[block_id].ref_count = 1
                self.request_blocks[request_id] = [block_id]
                blocks = [block_id]

            # Find block with space
            for block_id in blocks:
                block = self.blocks[block_id]
                if not block.is_full():
                    block.append(layer_idx, k, v)
                    return True

            # Need new block
            if not self.free_blocks:
                return False

            new_block_id = self.free_blocks.pop()
            self.blocks[new_block_id].ref_count = 1
            self.request_blocks[request_id].append(new_block_id)
            self.blocks[new_block_id].append(layer_idx, k, v)
            return True

    def get_kv_for_request(
        self,
        request_id: str,
        layer_idx: int
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """Get concatenated KV cache for a request at a layer.

        Args:
            request_id: The request identifier.
            layer_idx: The layer index.

        Returns:
            Tuple of (K, V) tensors, or (None, None) if not found.
        """
        blocks = self.request_blocks.get(request_id, [])

        if not blocks:
            return None, None

        k_list = []
        v_list = []

        for block_id in blocks:
            k, v = self.blocks[block_id].get_kv(layer_idx)
            if HAS_TORCH:
                if k.shape[1] > 0:  # Has tokens
                    k_list.append(k)
                    v_list.append(v)
            else:
                if k.shape[1] > 0:  # Has tokens
                    k_list.append(k)
                    v_list.append(v)

        if not k_list:
            return None, None

        if HAS_TORCH:
            return torch.cat(k_list, dim=1), torch.cat(v_list, dim=1)
        else:
            return np.concatenate(k_list, axis=1), np.concatenate(v_list, axis=1)

    def get_total_tokens_for_request(self, request_id: str) -> int:
        """Get total number of cached tokens for a request.

        Args:
            request_id: The request identifier.

        Returns:
            Total token count.
        """
        with self._lock:
            blocks = self.request_blocks.get(request_id, [])
            return sum(self.blocks[bid].num_tokens for bid in blocks)

    def can_allocate(self, num_blocks: int) -> bool:
        """Check if blocks can be allocated.

        Args:
            num_blocks: Number of blocks requested.

        Returns:
            True if enough blocks are available.
        """
        with self._lock:
            return len(self.free_blocks) >= num_blocks

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics.

        Returns:
            Dictionary of memory statistics.
        """
        used_blocks = self.num_blocks - len(self.free_blocks)
        block_size_bytes = self.config.get_block_memory_bytes()

        return {
            'total_blocks': self.num_blocks,
            'used_blocks': used_blocks,
            'free_blocks': len(self.free_blocks),
            'utilization': used_blocks / self.num_blocks if self.num_blocks > 0 else 0,
            'memory_used_bytes': used_blocks * block_size_bytes,
            'memory_used_gb': used_blocks * block_size_bytes / 1e9,
            'total_memory_gb': self.num_blocks * block_size_bytes / 1e9,
        }


class SlidingWindowCache:
    """KV cache with sliding window for long sequences.

    This cache maintains a fixed-size window of the most recent tokens,
    useful for models with sliding window attention.
    """

    def __init__(self, config: KVCacheConfig, window_size: int):
        """Initialize sliding window cache.

        Args:
            config: KV cache configuration.
            window_size: Maximum number of tokens to keep.
        """
        self.config = config
        self.window_size = window_size

        # Allocate window buffer
        if HAS_TORCH:
            dtype = torch.float16 if config.dtype == "float16" else torch.float32
            self.k_cache = torch.zeros(
                config.num_layers,
                config.num_heads,
                window_size,
                config.head_dim,
                dtype=dtype,
                device=config.device
            )
            self.v_cache = torch.zeros_like(self.k_cache)
        else:
            dtype = np.float16 if config.dtype == "float16" else np.float32
            self.k_cache = np.zeros(
                (config.num_layers, config.num_heads, window_size, config.head_dim),
                dtype=dtype
            )
            self.v_cache = np.zeros_like(self.k_cache)

        self.position = 0  # Current write position (circular)
        self.length = 0    # Actual cached length

    def append(self, layer_idx: int, k: Any, v: Any) -> None:
        """Append to sliding window cache.

        Args:
            layer_idx: The layer index.
            k: Key tensor of shape [num_heads, head_dim].
            v: Value tensor of shape [num_heads, head_dim].
        """
        pos = self.position % self.window_size
        self.k_cache[layer_idx, :, pos, :] = _to_torch_if_needed(k, self.k_cache)
        self.v_cache[layer_idx, :, pos, :] = _to_torch_if_needed(v, self.v_cache)

        if layer_idx == self.config.num_layers - 1:
            self.position += 1
            self.length = min(self.length + 1, self.window_size)

    def append_token(self, k_by_layer: List[Any], v_by_layer: List[Any]) -> None:
        """Append a full token's KV across all layers.

        Args:
            k_by_layer: List of key tensors, one per layer.
            v_by_layer: List of value tensors, one per layer.
        """
        pos = self.position % self.window_size
        for layer_idx in range(self.config.num_layers):
            self.k_cache[layer_idx, :, pos, :] = _to_torch_if_needed(k_by_layer[layer_idx], self.k_cache)
            self.v_cache[layer_idx, :, pos, :] = _to_torch_if_needed(v_by_layer[layer_idx], self.v_cache)

        self.position += 1
        self.length = min(self.length + 1, self.window_size)

    def get_kv(self, layer_idx: int) -> Tuple[Any, Any]:
        """Get KV in correct order.

        Args:
            layer_idx: The layer index.

        Returns:
            Tuple of (K, V) tensors in sequence order.
        """
        if self.length < self.window_size:
            return (
                self.k_cache[layer_idx, :, :self.length, :],
                self.v_cache[layer_idx, :, :self.length, :]
            )

        # Need to reorder for circular buffer
        start = self.position % self.window_size
        if HAS_TORCH:
            indices = torch.cat([
                torch.arange(start, self.window_size),
                torch.arange(0, start)
            ])
            return (
                self.k_cache[layer_idx, :, indices, :],
                self.v_cache[layer_idx, :, indices, :]
            )
        else:
            indices = np.concatenate([
                np.arange(start, self.window_size),
                np.arange(0, start)
            ])
            # Use two-step indexing to avoid numpy dimension transposition
            # with mixed basic/advanced indexing
            k_layer = self.k_cache[layer_idx]  # (num_heads, window_size, head_dim)
            v_layer = self.v_cache[layer_idx]
            return (
                k_layer[:, indices, :],
                v_layer[:, indices, :]
            )

    def get_length(self) -> int:
        """Get current number of cached tokens."""
        return self.length

    def get_total_position(self) -> int:
        """Get total number of tokens that have been processed."""
        return self.position

    def reset(self) -> None:
        """Reset the cache to empty state."""
        self.position = 0
        self.length = 0
        if HAS_TORCH:
            self.k_cache.zero_()
            self.v_cache.zero_()
        else:
            self.k_cache.fill(0)
            self.v_cache.fill(0)

    def is_full(self) -> bool:
        """Check if the window is full."""
        return self.length >= self.window_size
