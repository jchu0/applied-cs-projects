"""Core autoregressive generation for LLM inference."""

import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 100
    min_new_tokens: int = 0
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    do_sample: bool = True
    num_beams: int = 1
    early_stopping: bool = True
    eos_token_id: int = 2
    pad_token_id: int = 0


@dataclass
class GenerationOutput:
    """Output from generation."""
    sequences: np.ndarray          # Generated token IDs
    scores: Optional[List[np.ndarray]] = None  # Per-step logits
    attentions: Optional[List[np.ndarray]] = None


class KVCache:
    """
    Key-Value cache for efficient autoregressive generation.

    Caches past key/value states to avoid recomputation.
    """

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        max_length: int,
        dtype: np.dtype = np.float16
    ):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_length = max_length
        self.dtype = dtype

        # Pre-allocate cache
        shape = (num_layers, 2, batch_size, num_heads, max_length, head_dim)
        self.cache = np.zeros(shape, dtype=dtype)
        self.seq_lens = np.zeros(batch_size, dtype=np.int32)

    def update(
        self,
        layer_idx: int,
        key: np.ndarray,
        value: np.ndarray,
        batch_idx: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update cache with new key/value.

        Args:
            layer_idx: Layer index
            key: New key states (batch, heads, 1, dim)
            value: New value states (batch, heads, 1, dim)
            batch_idx: Batch indices to update

        Returns:
            Full cached key/value states
        """
        if batch_idx is None:
            batch_idx = np.arange(self.batch_size)

        for i, b in enumerate(batch_idx):
            pos = self.seq_lens[b]
            self.cache[layer_idx, 0, b, :, pos:pos+1, :] = key[i:i+1]
            self.cache[layer_idx, 1, b, :, pos:pos+1, :] = value[i:i+1]

        # Return full cache
        full_key = []
        full_value = []
        for i, b in enumerate(batch_idx):
            seq_len = self.seq_lens[b] + 1
            full_key.append(self.cache[layer_idx, 0, b, :, :seq_len, :])
            full_value.append(self.cache[layer_idx, 1, b, :, :seq_len, :])

        return full_key, full_value

    def increment_seq_lens(self, batch_idx: Optional[np.ndarray] = None):
        """Increment sequence lengths after generation step."""
        if batch_idx is None:
            self.seq_lens += 1
        else:
            self.seq_lens[batch_idx] += 1

    def reset(self, batch_idx: Optional[np.ndarray] = None):
        """Reset cache for specified batches."""
        if batch_idx is None:
            self.seq_lens.fill(0)
        else:
            self.seq_lens[batch_idx] = 0

    def get_length(self, batch_idx: int) -> int:
        """Get current sequence length for a batch."""
        return int(self.seq_lens[batch_idx])

    @property
    def memory_usage(self) -> int:
        """Memory usage in bytes."""
        return self.cache.nbytes


class PagedKVCache:
    """
    Paged KV cache for memory-efficient inference.

    Uses fixed-size pages for dynamic memory allocation.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        page_size: int = 16,
        max_pages: int = 1000,
        dtype: np.dtype = np.float16
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.max_pages = max_pages
        self.dtype = dtype

        # Page pool
        page_shape = (num_layers, 2, num_heads, page_size, head_dim)
        self.pages = [np.zeros(page_shape, dtype=dtype) for _ in range(max_pages)]
        self.free_pages = list(range(max_pages))

        # Sequence to page mapping
        self.seq_page_tables = {}  # seq_id -> [page_indices]
        self.seq_offsets = {}      # seq_id -> offset in last page

    def allocate_sequence(self, seq_id: int):
        """Allocate initial page for a sequence."""
        if len(self.free_pages) == 0:
            raise RuntimeError("No free pages available")

        page_idx = self.free_pages.pop()
        self.seq_page_tables[seq_id] = [page_idx]
        self.seq_offsets[seq_id] = 0

    def update(
        self,
        seq_id: int,
        layer_idx: int,
        key: np.ndarray,
        value: np.ndarray
    ):
        """Update cache for a sequence."""
        if seq_id not in self.seq_page_tables:
            self.allocate_sequence(seq_id)

        pages = self.seq_page_tables[seq_id]
        offset = self.seq_offsets[seq_id]

        # Check if we need a new page
        if offset >= self.page_size:
            if len(self.free_pages) == 0:
                raise RuntimeError("No free pages available")
            new_page = self.free_pages.pop()
            pages.append(new_page)
            offset = 0

        # Write to page
        page_idx = pages[-1]
        self.pages[page_idx][layer_idx, 0, :, offset, :] = key
        self.pages[page_idx][layer_idx, 1, :, offset, :] = value

        self.seq_offsets[seq_id] = offset + 1

    def get_kv(
        self,
        seq_id: int,
        layer_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get full KV cache for a sequence."""
        if seq_id not in self.seq_page_tables:
            return np.array([]), np.array([])

        pages = self.seq_page_tables[seq_id]
        keys = []
        values = []

        for i, page_idx in enumerate(pages):
            if i == len(pages) - 1:
                # Last page - only up to offset
                length = self.seq_offsets[seq_id]
            else:
                length = self.page_size

            keys.append(self.pages[page_idx][layer_idx, 0, :, :length, :])
            values.append(self.pages[page_idx][layer_idx, 1, :, :length, :])

        if not keys:
            return np.array([]), np.array([])

        return np.concatenate(keys, axis=1), np.concatenate(values, axis=1)

    def free_sequence(self, seq_id: int):
        """Free pages for a completed sequence."""
        if seq_id in self.seq_page_tables:
            self.free_pages.extend(self.seq_page_tables[seq_id])
            del self.seq_page_tables[seq_id]
            del self.seq_offsets[seq_id]


class AttentionMask:
    """Attention mask utilities."""

    @staticmethod
    def causal_mask(seq_len: int, dtype: np.dtype = np.float32) -> np.ndarray:
        """Create causal (lower triangular) mask."""
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        mask = mask * -1e9
        return mask.astype(dtype)

    @staticmethod
    def padding_mask(
        lengths: np.ndarray,
        max_len: int,
        dtype: np.dtype = np.float32
    ) -> np.ndarray:
        """Create padding mask from sequence lengths."""
        batch_size = len(lengths)
        mask = np.zeros((batch_size, max_len), dtype=dtype)
        for i, length in enumerate(lengths):
            mask[i, length:] = -1e9
        return mask


def scaled_dot_product_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Scaled dot-product attention.

    Args:
        query: (batch, heads, seq_q, dim)
        key: (batch, heads, seq_k, dim)
        value: (batch, heads, seq_k, dim)
        mask: Optional attention mask

    Returns:
        Attention output (batch, heads, seq_q, dim)
    """
    dim = query.shape[-1]
    scores = np.matmul(query, key.transpose(0, 1, 3, 2)) / np.sqrt(dim)

    if mask is not None:
        scores = scores + mask

    # Softmax
    scores = scores - scores.max(axis=-1, keepdims=True)
    attn_weights = np.exp(scores)
    attn_weights = attn_weights / attn_weights.sum(axis=-1, keepdims=True)

    return np.matmul(attn_weights, value)


class TransformerBlock:
    """Single transformer block for inference."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.intermediate_size = intermediate_size

        # Initialize random weights (placeholder)
        self.q_proj = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
        self.k_proj = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
        self.v_proj = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
        self.o_proj = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02

        self.gate_proj = np.random.randn(hidden_size, intermediate_size).astype(np.float32) * 0.02
        self.up_proj = np.random.randn(hidden_size, intermediate_size).astype(np.float32) * 0.02
        self.down_proj = np.random.randn(intermediate_size, hidden_size).astype(np.float32) * 0.02

    def forward(
        self,
        hidden: np.ndarray,
        past_key: Optional[np.ndarray] = None,
        past_value: Optional[np.ndarray] = None,
        attention_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass through transformer block.

        Returns:
            Tuple of (output, key, value)
        """
        batch_size, seq_len, _ = hidden.shape

        # Self-attention
        q = hidden @ self.q_proj
        k = hidden @ self.k_proj
        v = hidden @ self.v_proj

        # Reshape for multi-head
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Concatenate with past
        if past_key is not None:
            k = np.concatenate([past_key, k], axis=2)
            v = np.concatenate([past_value, v], axis=2)

        # Attention
        attn_out = scaled_dot_product_attention(q, k, v, attention_mask)

        # Reshape back
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_size)
        attn_out = attn_out @ self.o_proj

        # Residual
        hidden = hidden + attn_out

        # FFN
        gate = hidden @ self.gate_proj
        up = hidden @ self.up_proj
        gate = gate * (1 / (1 + np.exp(-gate)))  # SiLU
        ffn_out = (gate * up) @ self.down_proj

        # Residual
        hidden = hidden + ffn_out

        return hidden, k, v


class AutoregressiveModel:
    """
    Simple autoregressive transformer model.

    For inference demonstration.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        intermediate_size: int = 3072
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Embedding
        self.embedding = np.random.randn(vocab_size, hidden_size).astype(np.float32) * 0.02

        # Transformer blocks
        self.blocks = [
            TransformerBlock(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ]

        # LM head
        self.lm_head = np.random.randn(hidden_size, vocab_size).astype(np.float32) * 0.02

    def forward(
        self,
        input_ids: np.ndarray,
        kv_cache: Optional[KVCache] = None,
        attention_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
        """
        Forward pass.

        Returns:
            Tuple of (logits, new_kv_states)
        """
        batch_size, seq_len = input_ids.shape

        # Embedding
        hidden = self.embedding[input_ids]

        # Process blocks
        new_kv_states = []
        for i, block in enumerate(self.blocks):
            # Get past KV if available
            past_key = None
            past_value = None
            if kv_cache is not None and kv_cache.seq_lens[0] > 0:
                # Simplified - assumes all batches same length
                past_len = kv_cache.get_length(0)
                past_key = kv_cache.cache[i, 0, :batch_size, :, :past_len, :]
                past_value = kv_cache.cache[i, 1, :batch_size, :, :past_len, :]

            hidden, k, v = block.forward(hidden, past_key, past_value, attention_mask)
            new_kv_states.append((k, v))

        # LM head
        logits = hidden @ self.lm_head

        return logits, new_kv_states
