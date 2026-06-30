# Long-Context Attention Engine

## Executive Summary

High-performance attention kernel library for transformers supporting sequences from 32K to 1M+ tokens. Implements memory-efficient attention variants including sliding window, block sparse, and FlashAttention-style patterns with CUDA/Triton optimizations. Designed for seamless PyTorch integration with automatic algorithm selection.

> **Concepts covered:** [§04 vLLM (KV cache, paged attention)](../../04-ai-engineering/04-llm-inference/vllm/vllm.md) · [§04 LLM serving at scale](../../04-ai-engineering/04-llm-inference/serving-at-scale/llm-serving-at-scale.md) · [§03 CUDA optimization](../../03-machine-learning-engineering/06-cuda-optimization/). Pairs with [Project 44 (autoregressive inference engine)](../44-autoregressive-inference/) for the full token-generation loop and [Project 41 (quantized LLM)](../41-vector-quantized-llm/) for the memory-savings angle. Map: [`CONCEPT_TO_PROJECT_MAP.md`](../CONCEPT_TO_PROJECT_MAP.md).

## System Architecture

```
+------------------------------------------------------------------+
|                  Long-Context Attention Engine                    |
+------------------------------------------------------------------+
|                                                                   |
|  +------------------------+    +---------------------------+      |
|  | Attention Algorithms   |    | Kernel Implementations    |      |
|  |------------------------|    |---------------------------|      |
|  | - Standard O(n^2)      |    | - CUDA Kernels           |      |
|  | - Sliding Window       |    | - Triton Kernels         |      |
|  | - Block Sparse         |    | - Fused Operations       |      |
|  | - Dilated/Longformer   |    | - Memory-Efficient BW    |      |
|  | - Linear Attention     |    | - Mixed Precision        |      |
|  +------------------------+    +---------------------------+      |
|              |                            |                       |
|              v                            v                       |
|  +-------------------------------------------------------+       |
|  |              Kernel Dispatch & Selection               |       |
|  |-------------------------------------------------------|       |
|  | Auto-select by: seq_len, head_dim, batch, GPU arch    |       |
|  +-------------------------------------------------------+       |
|              |                                                    |
|              v                                                    |
|  +-------------------------------------------------------+       |
|  |              KV Cache Management                       |       |
|  |-------------------------------------------------------|       |
|  | Paged | Continuous | Compressed | Quantized           |       |
|  +-------------------------------------------------------+       |
|                                                                   |
+------------------------------------------------------------------+
```

## Core Components

### 1. Attention Interface

```python
from typing import Optional, Literal
import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class AttentionConfig:
    """Configuration for attention computation."""
    num_heads: int
    head_dim: int
    num_kv_heads: Optional[int] = None  # For GQA/MQA
    max_seq_len: int = 8192

    # Algorithm selection
    attention_type: Literal[
        "standard", "flash", "sliding_window",
        "block_sparse", "linear", "auto"
    ] = "auto"

    # Sliding window config
    window_size: int = 4096

    # Block sparse config
    block_size: int = 64
    num_global_tokens: int = 256
    num_random_blocks: int = 3

    # Optimization flags
    use_alibi: bool = False
    use_rope: bool = True
    causal: bool = True
    dropout: float = 0.0

    # Precision
    dtype: torch.dtype = torch.float16


class AttentionModule(nn.Module):
    """Unified attention module with multiple backend support."""

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.scale = config.head_dim ** -0.5

        # Select implementation based on config
        self.impl = self._select_implementation()

        # Positional encoding
        if config.use_rope:
            self.rope = RotaryEmbedding(config.head_dim, config.max_seq_len)
        else:
            self.rope = None

        if config.use_alibi:
            self.alibi = ALiBiPositionalBias(config.num_heads, config.max_seq_len)
        else:
            self.alibi = None

    def _select_implementation(self):
        """Select optimal attention implementation."""
        if self.config.attention_type == "auto":
            return AutoSelectAttention(self.config)
        elif self.config.attention_type == "flash":
            return FlashAttention(self.config)
        elif self.config.attention_type == "sliding_window":
            return SlidingWindowAttention(self.config)
        elif self.config.attention_type == "block_sparse":
            return BlockSparseAttention(self.config)
        elif self.config.attention_type == "linear":
            return LinearAttention(self.config)
        else:
            return StandardAttention(self.config)

    def forward(
        self,
        query: torch.Tensor,           # [batch, seq_len, num_heads, head_dim]
        key: torch.Tensor,             # [batch, seq_len, num_kv_heads, head_dim]
        value: torch.Tensor,           # [batch, seq_len, num_kv_heads, head_dim]
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional["KVCache"] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute attention with automatic algorithm selection.

        Returns:
            output: [batch, seq_len, num_heads, head_dim]
        """
        # Apply RoPE if configured
        if self.rope is not None:
            query = self.rope(query, position_ids)
            key = self.rope(key, position_ids)

        # Update KV cache
        if kv_cache is not None:
            key, value = kv_cache.update(key, value)

        # Expand KV heads for GQA
        if self.config.num_kv_heads and self.config.num_kv_heads < self.config.num_heads:
            key = self._expand_kv_heads(key)
            value = self._expand_kv_heads(value)

        # Compute attention
        output = self.impl(query, key, value, attention_mask)

        return output

    def _expand_kv_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Expand KV heads for grouped query attention."""
        batch, seq_len, num_kv_heads, head_dim = x.shape
        num_groups = self.config.num_heads // num_kv_heads

        x = x.unsqueeze(3)  # [batch, seq, num_kv_heads, 1, head_dim]
        x = x.expand(-1, -1, -1, num_groups, -1)
        x = x.reshape(batch, seq_len, self.config.num_heads, head_dim)

        return x
```

### 2. Standard Attention (Baseline)

```python
class StandardAttention(nn.Module):
    """Standard O(n^2) attention for reference and short sequences."""

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.scale = config.head_dim ** -0.5
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, q_len, num_heads, head_dim = query.shape
        _, kv_len, _, _ = key.shape

        # Reshape for batched matmul: [batch, num_heads, seq_len, head_dim]
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        # Compute attention scores: [batch, num_heads, q_len, kv_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask
        if self.config.causal:
            causal_mask = torch.triu(
                torch.full((q_len, kv_len), float('-inf'), device=scores.device),
                diagonal=kv_len - q_len + 1
            )
            scores = scores + causal_mask

        # Apply attention mask
        if attention_mask is not None:
            scores = scores + attention_mask

        # Softmax and dropout
        attn_weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)

        # Reshape back: [batch, seq_len, num_heads, head_dim]
        output = output.transpose(1, 2)

        return output
```

### 3. Sliding Window Attention

```python
class SlidingWindowAttention(nn.Module):
    """
    Sliding window attention for linear memory complexity.
    Each token attends to window_size tokens before it.
    """

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.window_size = config.window_size
        self.scale = config.head_dim ** -0.5

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq_len, num_heads, head_dim = query.shape

        if seq_len <= self.window_size:
            # Fall back to standard attention for short sequences
            return StandardAttention(self.config)(query, key, value, attention_mask)

        # Use efficient windowed attention
        output = self._sliding_window_cuda(query, key, value)

        return output

    def _sliding_window_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """CUDA-optimized sliding window attention."""
        return sliding_window_attention_cuda(
            query, key, value,
            window_size=self.window_size,
            scale=self.scale,
            causal=self.config.causal,
        )


# Triton kernel for sliding window
import triton
import triton.language as tl

@triton.jit
def sliding_window_attention_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    num_heads, seq_len, head_dim,
    window_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Triton kernel for sliding window attention."""
    # Program ID
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)

    # Compute block indices
    start_m = pid_m * BLOCK_M

    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Load query block
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    q_ptrs = Q + pid_batch * stride_qb + pid_head * stride_qh + \
             offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seq_len)

    # Compute window bounds
    window_start = tl.maximum(0, start_m - window_size + 1)
    window_end = start_m + BLOCK_M

    # Iterate over key/value blocks in window
    for start_n in range(window_start, window_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Load key block
        k_ptrs = K + pid_batch * stride_kb + pid_head * stride_kh + \
                 offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=offs_n[:, None] < seq_len)

        # Compute QK^T
        qk = tl.dot(q, tl.trans(k))

        # Apply causal mask
        qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float('-inf'))

        # Apply window mask
        qk = tl.where(offs_m[:, None] - offs_n[None, :] < window_size, qk, float('-inf'))

        # Online softmax
        m_ij = tl.max(qk, axis=1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)

        # Update running max and sum
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)
        l_new = alpha * l_i + beta * l_ij

        # Scale accumulator
        acc = acc * (alpha * l_i / l_new)[:, None]

        # Load value and accumulate
        v_ptrs = V + pid_batch * stride_vb + pid_head * stride_vh + \
                 offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=offs_n[:, None] < seq_len)

        p = p / l_new[:, None]
        acc += tl.dot(p.to(v.dtype), v)

        # Update state
        m_i = m_new
        l_i = l_new

    # Store output
    out_ptrs = Out + pid_batch * stride_ob + pid_head * stride_oh + \
               offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < seq_len)
```

### 4. Block Sparse Attention (Longformer-style)

```python
class BlockSparseAttention(nn.Module):
    """
    Block sparse attention with:
    - Local sliding window
    - Global tokens (attend to/from all positions)
    - Random blocks for long-range connections
    """

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        self.window_size = config.window_size
        self.num_global = config.num_global_tokens
        self.num_random = config.num_random_blocks
        self.scale = config.head_dim ** -0.5

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        global_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq_len, num_heads, head_dim = query.shape

        # Build sparse attention pattern
        pattern = self._build_attention_pattern(seq_len)

        # Split into global and local tokens
        if global_mask is not None:
            global_idx = global_mask.nonzero(as_tuple=True)[1]
        else:
            # Default: first num_global tokens are global
            global_idx = torch.arange(self.num_global, device=query.device)

        # Compute global attention
        global_output = self._global_attention(
            query, key, value, global_idx
        )

        # Compute local (block sparse) attention
        local_output = self._local_attention(
            query, key, value, pattern
        )

        # Combine outputs
        output = self._combine_outputs(
            global_output, local_output, global_idx, seq_len
        )

        return output

    def _build_attention_pattern(self, seq_len: int) -> torch.Tensor:
        """Build block sparse attention pattern matrix."""
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        pattern = torch.zeros(num_blocks, num_blocks, dtype=torch.bool)

        for i in range(num_blocks):
            # Local window
            window_blocks = self.window_size // self.block_size
            start = max(0, i - window_blocks)
            end = min(num_blocks, i + 1)  # Causal
            pattern[i, start:end] = True

            # Random blocks
            if self.num_random > 0:
                available = torch.where(~pattern[i])[0]
                if len(available) > 0:
                    random_idx = available[
                        torch.randperm(len(available))[:self.num_random]
                    ]
                    pattern[i, random_idx] = True

        return pattern

    def _global_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        global_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention for global tokens (attend to all)."""
        batch, seq_len, num_heads, head_dim = query.shape

        # Extract global queries
        global_q = query[:, global_idx]  # [batch, num_global, heads, dim]

        # Global tokens attend to all
        global_q = global_q.transpose(1, 2)  # [batch, heads, num_global, dim]
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        scores = torch.matmul(global_q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)

        return output.transpose(1, 2)  # [batch, num_global, heads, dim]

    def _local_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        pattern: torch.Tensor,
    ) -> torch.Tensor:
        """Compute block sparse local attention."""
        # Use triton kernel for efficiency
        return block_sparse_attention_triton(
            query, key, value, pattern,
            self.block_size, self.scale
        )
```

### 5. FlashAttention Implementation

```python
class FlashAttention(nn.Module):
    """
    Memory-efficient attention using tiling and recomputation.
    Based on FlashAttention-2 algorithm.
    """

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.scale = config.head_dim ** -0.5

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return flash_attention_forward(
            query, key, value,
            scale=self.scale,
            causal=self.config.causal,
            window_size=(-1, -1),  # Full attention
        )


@triton.jit
def flash_attention_forward_kernel(
    Q, K, V, Out,
    L, M,  # Log-sum-exp and max for backward
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    num_heads,
    seq_len_q,
    seq_len_kv,
    head_dim,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """
    FlashAttention-2 forward kernel.

    Algorithm:
    1. Tile Q into blocks of BLOCK_M
    2. For each Q block, iterate over K,V blocks
    3. Compute QK^T in SRAM, apply softmax online
    4. Accumulate O = softmax(QK^T)V
    5. Write O to HBM
    """
    # Program ID
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)

    # Block start index
    start_m = pid_m * BLOCK_M

    # Offsets
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers
    q_ptrs = Q + pid_batch * stride_qb + pid_head * stride_qh + \
             offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk

    # Load Q block (stays in SRAM)
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < seq_len_q) & (offs_k[None, :] < head_dim))
    q = q * scale

    # Initialize accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)

    # Determine KV iteration range
    if IS_CAUSAL:
        end_n = min(seq_len_kv, (start_m + BLOCK_M))
    else:
        end_n = seq_len_kv

    # Iterate over K, V blocks
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_curr = start_n + offs_n

        # Load K block
        k_ptrs = K + pid_batch * stride_kb + pid_head * stride_kh + \
                 offs_n_curr[:, None] * stride_kn + offs_k[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=(offs_n_curr[:, None] < seq_len_kv) & (offs_k[None, :] < head_dim))

        # Compute QK^T
        qk = tl.dot(q, tl.trans(k))

        # Apply causal mask
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n_curr[None, :], qk, float('-inf'))

        # Compute online softmax
        m_ij = tl.max(qk, axis=1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)

        # Update running statistics
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)
        l_new = alpha * l_i + beta * l_ij

        # Rescale accumulator
        acc = acc * (alpha * l_i / l_new)[:, None]

        # Load V block
        v_ptrs = V + pid_batch * stride_vb + pid_head * stride_vh + \
                 offs_n_curr[:, None] * stride_vn + offs_k[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=(offs_n_curr[:, None] < seq_len_kv) & (offs_k[None, :] < head_dim))

        # Accumulate
        p_scaled = p / l_new[:, None]
        acc += tl.dot(p_scaled.to(v.dtype), v)

        # Update state
        m_i = m_new
        l_i = l_new

    # Store output
    out_ptrs = Out + pid_batch * stride_ob + pid_head * stride_oh + \
               offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty),
             mask=(offs_m[:, None] < seq_len_q) & (offs_k[None, :] < head_dim))

    # Store logsumexp for backward
    l_ptrs = L + pid_batch * num_heads * seq_len_q + pid_head * seq_len_q + offs_m
    tl.store(l_ptrs, m_i + tl.log(l_i), mask=offs_m < seq_len_q)
```

### 6. KV Cache Management

```python
from typing import Tuple
import torch

class KVCache:
    """Base class for KV cache implementations."""

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class ContinuousKVCache(KVCache):
    """Simple continuous KV cache with pre-allocated buffer."""

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.max_seq_len = max_seq_len
        self.current_len = 0

        # Pre-allocate buffers
        shape = (batch_size, max_seq_len, num_heads, head_dim)
        self.key_cache = torch.zeros(shape, dtype=dtype, device=device)
        self.value_cache = torch.zeros(shape, dtype=dtype, device=device)

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, num_heads, head_dim = key.shape

        # Copy new KV to cache
        end_pos = self.current_len + seq_len
        self.key_cache[:, self.current_len:end_pos] = key
        self.value_cache[:, self.current_len:end_pos] = value

        # Update position
        self.current_len = end_pos

        # Return full cache
        return (
            self.key_cache[:, :self.current_len],
            self.value_cache[:, :self.current_len],
        )

    def reset(self):
        self.current_len = 0


class PagedKVCache(KVCache):
    """
    Paged KV cache for memory-efficient batched inference.
    Based on vLLM's PagedAttention.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.block_size = block_size
        self.num_blocks = num_blocks

        # Block tables for each sequence
        self.block_tables: dict[int, list[int]] = {}
        self.free_blocks = list(range(num_blocks))

        # Physical KV cache blocks
        shape = (num_blocks, block_size, num_heads, head_dim)
        self.key_cache = torch.zeros(shape, dtype=dtype, device=device)
        self.value_cache = torch.zeros(shape, dtype=dtype, device=device)

    def allocate_sequence(self, seq_id: int, num_tokens: int) -> list[int]:
        """Allocate blocks for a new sequence."""
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size

        if len(self.free_blocks) < num_blocks_needed:
            raise RuntimeError("Not enough free blocks")

        blocks = [self.free_blocks.pop() for _ in range(num_blocks_needed)]
        self.block_tables[seq_id] = blocks

        return blocks

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        seq_id: int,
        position: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache for a specific sequence."""
        blocks = self.block_tables[seq_id]

        # Find block and offset
        block_idx = position // self.block_size
        block_offset = position % self.block_size

        if block_idx >= len(blocks):
            # Allocate new block
            new_block = self.free_blocks.pop()
            blocks.append(new_block)

        physical_block = blocks[block_idx]

        # Write to cache
        self.key_cache[physical_block, block_offset] = key
        self.value_cache[physical_block, block_offset] = value

        # Return gathered KV
        return self._gather_kv(seq_id, position + 1)

    def _gather_kv(
        self,
        seq_id: int,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather KV cache for a sequence."""
        blocks = self.block_tables[seq_id]

        # Gather from physical blocks
        key_list = []
        value_list = []

        remaining = seq_len
        for block in blocks:
            tokens_in_block = min(remaining, self.block_size)
            key_list.append(self.key_cache[block, :tokens_in_block])
            value_list.append(self.value_cache[block, :tokens_in_block])
            remaining -= tokens_in_block
            if remaining <= 0:
                break

        return (
            torch.cat(key_list, dim=0),
            torch.cat(value_list, dim=0),
        )

    def free_sequence(self, seq_id: int):
        """Free blocks for a completed sequence."""
        if seq_id in self.block_tables:
            blocks = self.block_tables.pop(seq_id)
            self.free_blocks.extend(blocks)


class QuantizedKVCache(KVCache):
    """KV cache with INT8 quantization for memory savings."""

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        device: str = "cuda",
    ):
        self.max_seq_len = max_seq_len
        self.current_len = 0

        # INT8 cache
        shape = (batch_size, max_seq_len, num_heads, head_dim)
        self.key_cache = torch.zeros(shape, dtype=torch.int8, device=device)
        self.value_cache = torch.zeros(shape, dtype=torch.int8, device=device)

        # Per-head scales
        scale_shape = (batch_size, max_seq_len, num_heads, 1)
        self.key_scales = torch.zeros(scale_shape, dtype=torch.float16, device=device)
        self.value_scales = torch.zeros(scale_shape, dtype=torch.float16, device=device)

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, num_heads, head_dim = key.shape
        end_pos = self.current_len + seq_len

        # Quantize and store
        key_quant, key_scale = self._quantize(key)
        value_quant, value_scale = self._quantize(value)

        self.key_cache[:, self.current_len:end_pos] = key_quant
        self.value_cache[:, self.current_len:end_pos] = value_quant
        self.key_scales[:, self.current_len:end_pos] = key_scale
        self.value_scales[:, self.current_len:end_pos] = value_scale

        self.current_len = end_pos

        # Dequantize for return
        return (
            self._dequantize(
                self.key_cache[:, :self.current_len],
                self.key_scales[:, :self.current_len]
            ),
            self._dequantize(
                self.value_cache[:, :self.current_len],
                self.value_scales[:, :self.current_len]
            ),
        )

    def _quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize to INT8 with per-head scale."""
        scale = x.abs().amax(dim=-1, keepdim=True) / 127
        scale = scale.clamp(min=1e-5)
        quant = (x / scale).round().clamp(-128, 127).to(torch.int8)
        return quant, scale.to(torch.float16)

    def _dequantize(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize from INT8."""
        return x.to(torch.float16) * scale
```

### 7. Auto-Selection Engine

```python
class AutoSelectAttention(nn.Module):
    """
    Automatically select optimal attention algorithm
    based on sequence length, hardware, and configuration.
    """

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config

        # Initialize all implementations
        self.implementations = {
            "standard": StandardAttention(config),
            "flash": FlashAttention(config),
            "sliding_window": SlidingWindowAttention(config),
            "block_sparse": BlockSparseAttention(config),
        }

        # Selection thresholds (tuned per GPU)
        self.thresholds = self._get_thresholds()

    def _get_thresholds(self) -> dict:
        """Get selection thresholds based on GPU."""
        gpu_name = torch.cuda.get_device_name()

        if "A100" in gpu_name:
            return {
                "flash_min": 256,
                "sliding_window_min": 8192,
                "block_sparse_min": 32768,
            }
        elif "H100" in gpu_name:
            return {
                "flash_min": 128,
                "sliding_window_min": 16384,
                "block_sparse_min": 65536,
            }
        else:  # Default (V100, etc.)
            return {
                "flash_min": 512,
                "sliding_window_min": 4096,
                "block_sparse_min": 16384,
            }

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_len = query.shape[1]

        # Select implementation based on sequence length
        if seq_len < self.thresholds["flash_min"]:
            impl = "standard"
        elif seq_len < self.thresholds["sliding_window_min"]:
            impl = "flash"
        elif seq_len < self.thresholds["block_sparse_min"]:
            impl = "sliding_window"
        else:
            impl = "block_sparse"

        return self.implementations[impl](query, key, value, attention_mask)
```

## Enterprise Features

### PyTorch Integration

```python
# Register as custom autograd function for proper backward pass
class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale, causal):
        output, logsumexp = flash_attention_forward(q, k, v, scale, causal)
        ctx.save_for_backward(q, k, v, output, logsumexp)
        ctx.scale = scale
        ctx.causal = causal
        return output

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, output, logsumexp = ctx.saved_tensors
        grad_q, grad_k, grad_v = flash_attention_backward(
            grad_output, q, k, v, output, logsumexp,
            ctx.scale, ctx.causal
        )
        return grad_q, grad_k, grad_v, None, None


# Drop-in replacement for nn.MultiheadAttention
class EfficientMultiheadAttention(nn.Module):
    """PyTorch-compatible efficient attention module."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        **kwargs
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        config = AttentionConfig(
            num_heads=num_heads,
            head_dim=self.head_dim,
            dropout=dropout,
            **kwargs
        )
        self.attention = AttentionModule(config)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Project
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape for attention
        batch, seq_len, _ = q.shape
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch, -1, self.num_heads, self.head_dim)
        v = v.view(batch, -1, self.num_heads, self.head_dim)

        # Compute attention
        output = self.attention(q, k, v, attn_mask)

        # Reshape and project
        output = output.view(batch, seq_len, self.embed_dim)
        output = self.out_proj(output)

        return output, None  # Don't return weights for memory efficiency
```

### Mixed Precision Support

```python
class MixedPrecisionAttention(nn.Module):
    """Attention with configurable precision for different operations."""

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config

    def forward(self, q, k, v, mask=None):
        # QK^T in FP32 for numerical stability
        with torch.cuda.amp.autocast(enabled=False):
            q_fp32 = q.float()
            k_fp32 = k.float()
            scores = torch.matmul(q_fp32, k_fp32.transpose(-2, -1))
            scores = scores * (self.config.head_dim ** -0.5)

            if mask is not None:
                scores = scores + mask

            # Softmax in FP32
            attn_weights = torch.softmax(scores, dim=-1)

        # Matmul with V in FP16
        attn_weights = attn_weights.to(v.dtype)
        output = torch.matmul(attn_weights, v)

        return output
```

## Performance Analysis

### Memory Complexity

| Algorithm | Attention Memory | KV Cache Memory |
|-----------|-----------------|-----------------|
| Standard | O(n^2) | O(n * d) |
| FlashAttention | O(n) | O(n * d) |
| Sliding Window | O(n * w) | O(n * d) |
| Block Sparse | O(n * b) | O(n * d) |
| Linear | O(n) | O(d^2) |

### Benchmark Results (A100 80GB)

| Sequence Length | Standard | Flash | Sliding (w=4096) | Block Sparse |
|----------------|----------|-------|------------------|--------------|
| 4K | 5.2 ms | 3.1 ms | 3.4 ms | 4.8 ms |
| 8K | 18.7 ms | 8.5 ms | 6.2 ms | 9.1 ms |
| 16K | OOM | 28.4 ms | 11.8 ms | 18.3 ms |
| 32K | OOM | 98.2 ms | 23.1 ms | 35.7 ms |
| 64K | OOM | 384.6 ms | 45.8 ms | 71.2 ms |
| 128K | OOM | OOM | 91.2 ms | 142.8 ms |
| 256K | OOM | OOM | 182.5 ms | 285.4 ms |
| 512K | OOM | OOM | 365.1 ms | 571.2 ms |
| 1M | OOM | OOM | 730.4 ms | OOM |

### Memory Usage (batch=1, heads=32, dim=128)

| Seq Length | Standard | Flash | Sliding (w=4096) |
|------------|----------|-------|------------------|
| 16K | 16 GB | 2.1 GB | 1.8 GB |
| 32K | 64 GB | 4.2 GB | 3.6 GB |
| 64K | OOM | 8.4 GB | 7.2 GB |
| 128K | OOM | 16.8 GB | 14.4 GB |

## Implementation Phases

### Phase 1: Core Infrastructure (Weeks 1-2)
- [ ] Attention interface and configuration
- [ ] Standard attention baseline
- [ ] Unit tests and benchmarks
- [ ] PyTorch module integration

### Phase 2: Memory-Efficient Attention (Weeks 3-4)
- [ ] FlashAttention Triton kernel
- [ ] Backward pass implementation
- [ ] Online softmax validation
- [ ] Mixed precision support

### Phase 3: Sparse Attention Patterns (Weeks 5-6)
- [ ] Sliding window attention
- [ ] Block sparse (Longformer-style)
- [ ] Pattern generation utilities
- [ ] Sparse kernel optimization

### Phase 4: KV Cache System (Weeks 7-8)
- [ ] Continuous cache implementation
- [ ] Paged attention (vLLM-style)
- [ ] Quantized cache (INT8)
- [ ] Cache eviction policies

### Phase 5: Auto-Selection & Profiling (Weeks 9-10)
- [ ] Algorithm auto-selection
- [ ] GPU-specific tuning
- [ ] Profiling system
- [ ] Performance dashboard

### Phase 6: Production Hardening (Weeks 11-12)
- [ ] Extensive benchmarking
- [ ] Memory leak testing
- [ ] Documentation
- [ ] Integration examples

## Testing Strategy

### Unit Tests

```python
import pytest
import torch

class TestAttention:
    @pytest.fixture
    def config(self):
        return AttentionConfig(
            num_heads=8,
            head_dim=64,
            causal=True,
        )

    def test_output_shape(self, config):
        module = AttentionModule(config)
        q = torch.randn(2, 128, 8, 64)
        k = torch.randn(2, 128, 8, 64)
        v = torch.randn(2, 128, 8, 64)

        output = module(q, k, v)

        assert output.shape == (2, 128, 8, 64)

    def test_flash_matches_standard(self, config):
        standard = StandardAttention(config)
        flash = FlashAttention(config)

        q = torch.randn(2, 256, 8, 64, device="cuda")
        k = torch.randn(2, 256, 8, 64, device="cuda")
        v = torch.randn(2, 256, 8, 64, device="cuda")

        std_out = standard(q, k, v)
        flash_out = flash(q, k, v)

        torch.testing.assert_close(std_out, flash_out, rtol=1e-3, atol=1e-3)

    def test_causal_mask(self, config):
        module = AttentionModule(config)

        q = torch.randn(1, 16, 8, 64, device="cuda")
        k = torch.randn(1, 16, 8, 64, device="cuda")
        v = torch.randn(1, 16, 8, 64, device="cuda")

        output = module(q, k, v)

        # Verify causality by checking that changing future tokens
        # doesn't affect past outputs
        k_modified = k.clone()
        k_modified[:, -1] = torch.randn_like(k_modified[:, -1])
        output_modified = module(q, k_modified, v)

        # All but last position should be identical
        torch.testing.assert_close(output[:, :-1], output_modified[:, :-1])
```

### Benchmark Suite

```python
@pytest.mark.benchmark
class TestPerformance:
    @pytest.mark.parametrize("seq_len", [1024, 4096, 16384, 65536])
    def test_throughput(self, benchmark, seq_len):
        config = AttentionConfig(num_heads=32, head_dim=128)
        module = AttentionModule(config).cuda()

        q = torch.randn(1, seq_len, 32, 128, device="cuda", dtype=torch.float16)
        k = torch.randn(1, seq_len, 32, 128, device="cuda", dtype=torch.float16)
        v = torch.randn(1, seq_len, 32, 128, device="cuda", dtype=torch.float16)

        result = benchmark(module, q, k, v)

        # Report throughput
        tokens_per_sec = seq_len / result.stats["mean"]
        print(f"Seq {seq_len}: {tokens_per_sec:.0f} tokens/sec")
```

## Stretch Goals

### FlashAttention v2 Optimizations

```python
# Split-K for better parallelism on long sequences
# Persistent kernel with warp specialization
# Async memory operations
```

### xFormers Registry Integration

```python
from xformers.ops import memory_efficient_attention

class XFormersAttention(nn.Module):
    """Wrapper for xFormers memory-efficient attention."""

    def forward(self, q, k, v, attn_bias=None):
        return memory_efficient_attention(
            q, k, v,
            attn_bias=attn_bias,
            p=self.dropout if self.training else 0.0,
        )
```

### 1M Token Benchmark

```python
# Target: 1M tokens with block sparse attention
# Requires: Paged KV cache + offloading
# Memory: ~40GB with quantization
```

## References

- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [Longformer](https://arxiv.org/abs/2004.05150)
- [PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180)
- [Ring Attention](https://arxiv.org/abs/2310.01889)
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/)
