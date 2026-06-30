"""Runtime engine for on-device LLM execution."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
import numpy as np
import time
from contextlib import contextmanager

from ..core.model import (
    LLMWeights, ModelConfig, QuantizedTensor, QuantizationType,
    DeviceType, TransformerLayer
)


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    model_memory_mb: float
    kv_cache_mb: float
    activation_mb: float
    peak_mb: float
    available_mb: float


class MemoryPool:
    """Memory pool for efficient allocation."""

    def __init__(self, max_size_mb: float = 512):
        self.max_size = int(max_size_mb * 1024 * 1024)
        self.used = 0
        self.allocations: dict[int, tuple[int, np.ndarray]] = {}
        self._next_id = 0
        self.peak_usage = 0

    def allocate(self, shape: tuple[int, ...], dtype: np.dtype = np.float32) -> tuple[int, np.ndarray]:
        """Allocate memory from pool."""
        size = int(np.prod(shape) * np.dtype(dtype).itemsize)

        if self.used + size > self.max_size:
            raise MemoryError(f"Out of memory: need {size}, available {self.max_size - self.used}")

        array = np.zeros(shape, dtype=dtype)
        alloc_id = self._next_id
        self._next_id += 1

        self.allocations[alloc_id] = (size, array)
        self.used += size
        self.peak_usage = max(self.peak_usage, self.used)

        return alloc_id, array

    def free(self, alloc_id: int) -> None:
        """Free allocation."""
        if alloc_id in self.allocations:
            size, _ = self.allocations[alloc_id]
            self.used -= size
            del self.allocations[alloc_id]

    def get_stats(self) -> dict[str, float]:
        """Get memory statistics."""
        return {
            "used_mb": self.used / (1024 * 1024),
            "peak_mb": self.peak_usage / (1024 * 1024),
            "available_mb": (self.max_size - self.used) / (1024 * 1024),
            "num_allocations": len(self.allocations)
        }

    def reset(self) -> None:
        """Reset pool."""
        self.allocations.clear()
        self.used = 0


class KVCache:
    """Key-value cache for autoregressive generation."""

    def __init__(
        self,
        config: ModelConfig,
        max_length: int,
        pool: MemoryPool | None = None
    ):
        self.config = config
        self.max_length = max_length
        self.pool = pool or MemoryPool()

        head_dim = config.hidden_size // config.num_heads
        cache_shape = (
            config.num_layers,
            2,  # key and value
            max_length,
            config.num_kv_heads,
            head_dim
        )

        self._cache_id, self.cache = self.pool.allocate(cache_shape, np.float16)
        self.position = 0

    def update(
        self,
        layer_idx: int,
        key: np.ndarray,
        value: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Update cache and return full k/v."""
        seq_len = key.shape[0]
        end_pos = self.position + seq_len

        # Store new k/v
        self.cache[layer_idx, 0, self.position:end_pos] = key
        self.cache[layer_idx, 1, self.position:end_pos] = value

        # Return full cache up to current position
        return (
            self.cache[layer_idx, 0, :end_pos],
            self.cache[layer_idx, 1, :end_pos]
        )

    def advance(self, n: int = 1) -> None:
        """Advance position by n tokens."""
        self.position = min(self.position + n, self.max_length)

    def reset(self) -> None:
        """Reset cache."""
        self.cache.fill(0)
        self.position = 0

    @property
    def memory_mb(self) -> float:
        """Get cache memory in MB."""
        return self.cache.nbytes / (1024 * 1024)


class ExecutionContext:
    """Execution context for a generation run."""

    def __init__(
        self,
        config: ModelConfig,
        batch_size: int = 1,
        max_length: int = 512,
        memory_limit_mb: float = 512
    ):
        self.config = config
        self.batch_size = batch_size
        self.max_length = max_length

        self.pool = MemoryPool(memory_limit_mb)
        self.kv_cache = KVCache(config, max_length, self.pool)

        # Preallocate activation buffers
        self._alloc_ids: list[int] = []
        self._setup_buffers()

    def _setup_buffers(self) -> None:
        """Set up activation buffers."""
        h = self.config.hidden_size
        ff = self.config.intermediate_size

        # Hidden states
        alloc_id, self.hidden = self.pool.allocate(
            (self.batch_size, self.max_length, h), np.float16
        )
        self._alloc_ids.append(alloc_id)

        # Attention intermediates
        alloc_id, self.query = self.pool.allocate(
            (self.batch_size, self.config.num_heads, self.max_length, h // self.config.num_heads),
            np.float16
        )
        self._alloc_ids.append(alloc_id)

        # FFN intermediates
        alloc_id, self.ffn_gate = self.pool.allocate(
            (self.batch_size, self.max_length, ff), np.float16
        )
        self._alloc_ids.append(alloc_id)

    def get_memory_stats(self) -> MemoryStats:
        """Get memory statistics."""
        pool_stats = self.pool.get_stats()
        return MemoryStats(
            model_memory_mb=0,  # Loaded separately
            kv_cache_mb=self.kv_cache.memory_mb,
            activation_mb=pool_stats["used_mb"] - self.kv_cache.memory_mb,
            peak_mb=pool_stats["peak_mb"],
            available_mb=pool_stats["available_mb"]
        )

    def reset(self) -> None:
        """Reset context for new generation."""
        self.kv_cache.reset()


class RuntimeConfig:
    """Configuration for runtime execution."""

    def __init__(
        self,
        device: DeviceType = DeviceType.CPU,
        num_threads: int = 4,
        use_mmap: bool = True,
        memory_limit_mb: float = 1024,
        batch_size: int = 1,
        context_length: int = 512
    ):
        self.device = device
        self.num_threads = num_threads
        self.use_mmap = use_mmap
        self.memory_limit_mb = memory_limit_mb
        self.batch_size = batch_size
        self.context_length = context_length


class Operator:
    """Base class for operators."""

    def __init__(self, name: str):
        self.name = name
        self.total_time = 0.0
        self.call_count = 0

    def __call__(self, *args, **kwargs) -> np.ndarray:
        start = time.perf_counter()
        result = self.forward(*args, **kwargs)
        self.total_time += time.perf_counter() - start
        self.call_count += 1
        return result

    def forward(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError


class MatMulOp(Operator):
    """Matrix multiplication operator."""

    def __init__(self):
        super().__init__("matmul")

    def forward(
        self,
        x: np.ndarray,
        weight: QuantizedTensor,
        bias: np.ndarray | None = None
    ) -> np.ndarray:
        # Dequantize weight
        w = weight.dequantize()

        # Matrix multiply
        result = np.matmul(x, w.T)

        if bias is not None:
            result += bias

        return result


class RMSNormOp(Operator):
    """RMS normalization operator."""

    def __init__(self, eps: float = 1e-5):
        super().__init__("rmsnorm")
        self.eps = eps

    def forward(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        # RMS norm
        variance = np.mean(x ** 2, axis=-1, keepdims=True)
        x_norm = x * np.reciprocal(np.sqrt(variance + self.eps))
        return x_norm * weight


class RoPEOp(Operator):
    """Rotary position embedding operator."""

    def __init__(self, config: ModelConfig):
        super().__init__("rope")
        self.config = config
        self._init_freqs()

    def _init_freqs(self) -> None:
        """Initialize rotation frequencies."""
        head_dim = self.config.hidden_size // self.config.num_heads
        freqs = 1.0 / (self.config.rope_theta ** (
            np.arange(0, head_dim, 2) / head_dim
        ))
        positions = np.arange(self.config.max_position)
        freqs_cis = np.outer(positions, freqs)
        self.cos = np.cos(freqs_cis).astype(np.float16)
        self.sin = np.sin(freqs_cis).astype(np.float16)

    def forward(
        self,
        x: np.ndarray,
        position: int
    ) -> np.ndarray:
        seq_len = x.shape[-2]
        cos = self.cos[position:position + seq_len]
        sin = self.sin[position:position + seq_len]

        # Rotate pairs
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        rotated = np.empty_like(x)
        rotated[..., ::2] = x1 * cos - x2 * sin
        rotated[..., 1::2] = x1 * sin + x2 * cos

        return rotated


class SoftmaxOp(Operator):
    """Softmax operator with numerical stability."""

    def __init__(self):
        super().__init__("softmax")

    def forward(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class SiLUOp(Operator):
    """SiLU activation operator."""

    def __init__(self):
        super().__init__("silu")

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x * (1 / (1 + np.exp(-x)))


class DeviceRuntime:
    """Runtime for executing LLM on device."""

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.operators: dict[str, Operator] = {}
        self._setup_operators()

    def _setup_operators(self) -> None:
        """Set up execution operators."""
        self.operators["matmul"] = MatMulOp()
        self.operators["rmsnorm"] = RMSNormOp()
        self.operators["softmax"] = SoftmaxOp()
        self.operators["silu"] = SiLUOp()

    def create_context(self, model_config: ModelConfig) -> ExecutionContext:
        """Create execution context."""
        return ExecutionContext(
            model_config,
            batch_size=self.config.batch_size,
            max_length=self.config.context_length,
            memory_limit_mb=self.config.memory_limit_mb
        )

    def execute_attention(
        self,
        ctx: ExecutionContext,
        layer: TransformerLayer,
        hidden: np.ndarray,
        layer_idx: int
    ) -> np.ndarray:
        """Execute attention computation."""
        # Project Q, K, V
        q = self.operators["matmul"](hidden, layer.q_proj)
        k = self.operators["matmul"](hidden, layer.k_proj)
        v = self.operators["matmul"](hidden, layer.v_proj)

        # Reshape for multi-head attention
        batch, seq, _ = hidden.shape
        head_dim = ctx.config.hidden_size // ctx.config.num_heads

        q = q.reshape(batch, seq, ctx.config.num_heads, head_dim)
        k = k.reshape(batch, seq, ctx.config.num_kv_heads, head_dim)
        v = v.reshape(batch, seq, ctx.config.num_kv_heads, head_dim)

        # Apply RoPE (simplified)
        rope_op = RoPEOp(ctx.config)
        q = rope_op(q.transpose(0, 2, 1, 3), ctx.kv_cache.position)
        k = rope_op(k.transpose(0, 2, 1, 3), ctx.kv_cache.position)

        # Update KV cache
        k_cache, v_cache = ctx.kv_cache.update(
            layer_idx,
            k.transpose(0, 2, 1, 3)[0],
            v.transpose(0, 2, 1, 3)[0]
        )

        # Expand KV for GQA
        if ctx.config.num_kv_heads < ctx.config.num_heads:
            repeat = ctx.config.num_heads // ctx.config.num_kv_heads
            k_cache = np.repeat(k_cache, repeat, axis=1)
            v_cache = np.repeat(v_cache, repeat, axis=1)

        # Attention scores
        scale = 1.0 / np.sqrt(head_dim)
        scores = np.matmul(q, k_cache.transpose(0, 1, 3, 2)) * scale

        # Causal mask
        mask = np.triu(np.full((seq, k_cache.shape[2]), -np.inf), k=1)
        scores = scores + mask

        # Softmax and output
        attn_weights = self.operators["softmax"](scores)
        attn_output = np.matmul(attn_weights, v_cache)

        # Reshape and project output
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch, seq, -1)
        output = self.operators["matmul"](attn_output, layer.o_proj)

        return output

    def execute_ffn(
        self,
        ctx: ExecutionContext,
        layer: TransformerLayer,
        hidden: np.ndarray
    ) -> np.ndarray:
        """Execute feed-forward network."""
        # SwiGLU FFN
        gate = self.operators["matmul"](hidden, layer.gate_proj)
        gate = self.operators["silu"](gate)
        up = self.operators["matmul"](hidden, layer.up_proj)
        hidden_ffn = gate * up
        output = self.operators["matmul"](hidden_ffn, layer.down_proj)
        return output

    def execute_layer(
        self,
        ctx: ExecutionContext,
        layer: TransformerLayer,
        hidden: np.ndarray,
        layer_idx: int
    ) -> np.ndarray:
        """Execute single transformer layer."""
        # Input norm
        normed = self.operators["rmsnorm"](hidden, layer.input_norm)

        # Attention
        attn_output = self.execute_attention(ctx, layer, normed, layer_idx)
        hidden = hidden + attn_output

        # Post-attention norm
        normed = self.operators["rmsnorm"](hidden, layer.post_attn_norm)

        # FFN
        ffn_output = self.execute_ffn(ctx, layer, normed)
        hidden = hidden + ffn_output

        return hidden

    def get_profiling_stats(self) -> dict[str, dict[str, float]]:
        """Get operator profiling statistics."""
        stats = {}
        for name, op in self.operators.items():
            if op.call_count > 0:
                stats[name] = {
                    "total_time_ms": op.total_time * 1000,
                    "call_count": op.call_count,
                    "avg_time_ms": (op.total_time / op.call_count) * 1000
                }
        return stats

    def reset_profiling(self) -> None:
        """Reset profiling counters."""
        for op in self.operators.values():
            op.total_time = 0.0
            op.call_count = 0
