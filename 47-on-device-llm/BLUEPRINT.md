# Project 46: On-Device LLM Runtime

## Executive Summary

A lightweight LLM inference runtime optimized for on-device deployment using memory mapping, vector quantization, and SIMD kernels. This project enables efficient execution of quantized models on consumer hardware (CPU/GPU/Metal/Vulkan) with minimal memory footprint and maximum throughput using techniques pioneered by llama.cpp and similar projects.

> **Concepts covered:** [§04 Quantization](../../04-ai-engineering/04-llm-inference/quantization/quantization.md) · [§04 LLM serving at scale (edge tier)](../../04-ai-engineering/04-llm-inference/serving-at-scale/llm-serving-at-scale.md). Pairs with [Project 41 (full quantization pipeline)](../41-vector-quantized-llm/) for the model-prep side. For server-side inference, contrast with [Project 44 (autoregressive engine)](../44-autoregressive-inference/) and [Project 22 (attention kernels)](../22-long-context-attention/). Map: [`CONCEPT_TO_PROJECT_MAP.md`](../CONCEPT_TO_PROJECT_MAP.md).

## Architecture Overview

### System Design

```
+------------------------------------------------------------------+
|                    On-Device LLM Runtime                          |
+------------------------------------------------------------------+
|                                                                    |
|  +-------------------+     +-------------------+     +-----------+ |
|  | Model Loader      |     | Inference Engine  |     | Sampler   | |
|  | (GGUF/mmap)       |---->| (SIMD Kernels)    |---->| (Top-k/p) | |
|  +-------------------+     +-------------------+     +-----------+ |
|         |                          |                       |       |
|         v                          v                       v       |
|  +-------------------+     +-------------------+     +-----------+ |
|  | Quantization      |     | Attention         |     | KV Cache  | |
|  | (Q4/Q8)           |     | (Fused/Sliding)   |     | (CPU-opt) | |
|  +-------------------+     +-------------------+     +-----------+ |
|                                    |                               |
|  +----------------------------------------------------------+     |
|  |                    Backend Abstraction                    |     |
|  |  +--------+  +--------+  +--------+  +--------+           |     |
|  |  | CPU    |  | CUDA   |  | Metal  |  | Vulkan |           |     |
|  |  | (SIMD) |  |        |  | (MPS)  |  |        |           |     |
|  |  +--------+  +--------+  +--------+  +--------+           |     |
|  +----------------------------------------------------------+     |
+------------------------------------------------------------------+
```

### Core Components

#### 1. Model Loader with Memory Mapping

```python
import mmap
import struct
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, BinaryIO
from enum import IntEnum

class GGMLType(IntEnum):
    """GGML quantization types."""
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15

@dataclass
class TensorInfo:
    """Metadata for a tensor in GGUF file."""
    name: str
    shape: List[int]
    dtype: GGMLType
    offset: int
    size_bytes: int

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int  # For GQA
    intermediate_size: int
    max_seq_len: int
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

class GGUFLoader:
    """
    Load GGUF format models with memory mapping.

    Memory mapping allows:
    - Lazy loading (only load what's accessed)
    - Shared memory across processes
    - No full copy into RAM
    """

    GGUF_MAGIC = 0x46554747  # "GGUF"

    def __init__(self, path: str):
        self.path = path
        self.file: Optional[BinaryIO] = None
        self.mmap: Optional[mmap.mmap] = None

        self.metadata: Dict[str, any] = {}
        self.tensors: Dict[str, TensorInfo] = {}
        self.config: Optional[ModelConfig] = None

        self._load_header()

    def _load_header(self):
        """Parse GGUF header and tensor info."""
        self.file = open(self.path, 'rb')

        # Read magic
        magic = struct.unpack('<I', self.file.read(4))[0]
        if magic != self.GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF magic: {magic:08x}")

        # Read version
        version = struct.unpack('<I', self.file.read(4))[0]
        if version < 2:
            raise ValueError(f"GGUF version {version} not supported")

        # Read counts
        n_tensors = struct.unpack('<Q', self.file.read(8))[0]
        n_metadata = struct.unpack('<Q', self.file.read(8))[0]

        # Read metadata
        for _ in range(n_metadata):
            key, value = self._read_metadata_kv()
            self.metadata[key] = value

        # Parse config from metadata
        self.config = self._parse_config()

        # Read tensor infos
        tensor_data_offset = 0
        for _ in range(n_tensors):
            info = self._read_tensor_info(tensor_data_offset)
            self.tensors[info.name] = info
            tensor_data_offset = info.offset + info.size_bytes

        # Create memory map
        self.mmap = mmap.mmap(
            self.file.fileno(), 0,
            access=mmap.ACCESS_READ
        )

    def _read_metadata_kv(self):
        """Read a metadata key-value pair."""
        # Read key
        key_len = struct.unpack('<Q', self.file.read(8))[0]
        key = self.file.read(key_len).decode('utf-8')

        # Read value type and value
        value_type = struct.unpack('<I', self.file.read(4))[0]

        if value_type == 4:  # UINT32
            value = struct.unpack('<I', self.file.read(4))[0]
        elif value_type == 5:  # INT32
            value = struct.unpack('<i', self.file.read(4))[0]
        elif value_type == 6:  # FLOAT32
            value = struct.unpack('<f', self.file.read(4))[0]
        elif value_type == 8:  # STRING
            str_len = struct.unpack('<Q', self.file.read(8))[0]
            value = self.file.read(str_len).decode('utf-8')
        elif value_type == 10:  # UINT64
            value = struct.unpack('<Q', self.file.read(8))[0]
        else:
            value = None

        return key, value

    def _read_tensor_info(self, base_offset: int) -> TensorInfo:
        """Read tensor metadata."""
        # Name
        name_len = struct.unpack('<Q', self.file.read(8))[0]
        name = self.file.read(name_len).decode('utf-8')

        # Dimensions
        n_dims = struct.unpack('<I', self.file.read(4))[0]
        shape = []
        for _ in range(n_dims):
            dim = struct.unpack('<Q', self.file.read(8))[0]
            shape.append(dim)

        # Type
        dtype = GGMLType(struct.unpack('<I', self.file.read(4))[0])

        # Offset
        offset = struct.unpack('<Q', self.file.read(8))[0]

        # Calculate size
        size_bytes = self._calc_tensor_size(shape, dtype)

        return TensorInfo(
            name=name,
            shape=shape,
            dtype=dtype,
            offset=base_offset + offset,
            size_bytes=size_bytes
        )

    def _calc_tensor_size(self, shape: List[int], dtype: GGMLType) -> int:
        """Calculate tensor size in bytes."""
        n_elements = 1
        for dim in shape:
            n_elements *= dim

        if dtype == GGMLType.F32:
            return n_elements * 4
        elif dtype == GGMLType.F16:
            return n_elements * 2
        elif dtype in (GGMLType.Q4_0, GGMLType.Q4_1):
            # 4-bit: 32 elements per block
            n_blocks = (n_elements + 31) // 32
            block_size = 18 if dtype == GGMLType.Q4_0 else 20
            return n_blocks * block_size
        elif dtype == GGMLType.Q8_0:
            # 8-bit: 32 elements per block
            n_blocks = (n_elements + 31) // 32
            return n_blocks * 34

        return n_elements  # Fallback

    def _parse_config(self) -> ModelConfig:
        """Extract model config from metadata."""
        return ModelConfig(
            vocab_size=self.metadata.get('llama.vocab_size', 32000),
            hidden_size=self.metadata.get('llama.embedding_length', 4096),
            num_layers=self.metadata.get('llama.block_count', 32),
            num_heads=self.metadata.get('llama.attention.head_count', 32),
            num_kv_heads=self.metadata.get('llama.attention.head_count_kv', 32),
            intermediate_size=self.metadata.get('llama.feed_forward_length', 11008),
            max_seq_len=self.metadata.get('llama.context_length', 2048),
            rope_theta=self.metadata.get('llama.rope.freq_base', 10000.0),
            norm_eps=self.metadata.get('llama.attention.layer_norm_rms_epsilon', 1e-5)
        )

    def get_tensor(self, name: str) -> np.ndarray:
        """
        Get tensor data via memory mapping.

        Uses lazy loading - data is only read from disk when accessed.
        """
        if name not in self.tensors:
            raise KeyError(f"Tensor not found: {name}")

        info = self.tensors[name]

        # Read from mmap
        data = self.mmap[info.offset:info.offset + info.size_bytes]

        # Dequantize if needed
        if info.dtype == GGMLType.F32:
            return np.frombuffer(data, dtype=np.float32).reshape(info.shape)
        elif info.dtype == GGMLType.F16:
            return np.frombuffer(data, dtype=np.float16).reshape(info.shape)
        elif info.dtype in (GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q8_0):
            return self._dequantize(data, info)

        return np.frombuffer(data, dtype=np.uint8)

    def _dequantize(self, data: bytes, info: TensorInfo) -> np.ndarray:
        """Dequantize tensor data."""
        # Implementation depends on quantization type
        # This is a simplified version
        if info.dtype == GGMLType.Q8_0:
            return self._dequantize_q8_0(data, info.shape)
        elif info.dtype == GGMLType.Q4_0:
            return self._dequantize_q4_0(data, info.shape)

        raise NotImplementedError(f"Dequantization for {info.dtype}")

    def _dequantize_q8_0(self, data: bytes, shape: List[int]) -> np.ndarray:
        """Dequantize Q8_0 format."""
        block_size = 32
        n_elements = 1
        for dim in shape:
            n_elements *= dim

        n_blocks = (n_elements + block_size - 1) // block_size
        result = np.zeros(n_elements, dtype=np.float32)

        offset = 0
        for i in range(n_blocks):
            # Scale (float16)
            scale = np.frombuffer(data[offset:offset+2], dtype=np.float16)[0]
            offset += 2

            # Quantized values (int8)
            values = np.frombuffer(data[offset:offset+32], dtype=np.int8)
            offset += 32

            # Dequantize
            start = i * block_size
            end = min(start + block_size, n_elements)
            result[start:end] = values[:end-start].astype(np.float32) * float(scale)

        return result.reshape(shape)

    def _dequantize_q4_0(self, data: bytes, shape: List[int]) -> np.ndarray:
        """Dequantize Q4_0 format."""
        # 4-bit quantization with block scale
        block_size = 32
        n_elements = 1
        for dim in shape:
            n_elements *= dim

        n_blocks = (n_elements + block_size - 1) // block_size
        result = np.zeros(n_elements, dtype=np.float32)

        offset = 0
        for i in range(n_blocks):
            # Scale (float16)
            scale = np.frombuffer(data[offset:offset+2], dtype=np.float16)[0]
            offset += 2

            # 4-bit values packed into bytes
            packed = np.frombuffer(data[offset:offset+16], dtype=np.uint8)
            offset += 16

            # Unpack 4-bit values
            values = np.zeros(32, dtype=np.float32)
            for j in range(16):
                values[j*2] = (packed[j] & 0xF) - 8
                values[j*2+1] = (packed[j] >> 4) - 8

            # Dequantize
            start = i * block_size
            end = min(start + block_size, n_elements)
            result[start:end] = values[:end-start] * float(scale)

        return result.reshape(shape)

    def close(self):
        """Close file and memory map."""
        if self.mmap:
            self.mmap.close()
        if self.file:
            self.file.close()
```

#### 2. SIMD-Optimized Inference Kernels

```python
import numpy as np
from typing import Optional

# Try to import Numba for SIMD
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

if HAS_NUMBA:
    @njit(parallel=True, fastmath=True)
    def matmul_f32(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Matrix multiplication with SIMD optimization.

        A: [M, K]
        B: [K, N]
        Returns: [M, N]
        """
        M, K = A.shape
        K_, N = B.shape
        assert K == K_

        C = np.zeros((M, N), dtype=np.float32)

        # Tiled multiplication for cache efficiency
        tile_size = 32

        for i in prange(0, M, tile_size):
            for j in range(0, N, tile_size):
                for k in range(0, K, tile_size):
                    # Tile boundaries
                    i_end = min(i + tile_size, M)
                    j_end = min(j + tile_size, N)
                    k_end = min(k + tile_size, K)

                    # Compute tile
                    for ii in range(i, i_end):
                        for jj in range(j, j_end):
                            acc = 0.0
                            for kk in range(k, k_end):
                                acc += A[ii, kk] * B[kk, jj]
                            C[ii, jj] += acc

        return C

    @njit(fastmath=True)
    def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """RMS normalization."""
        # Compute RMS
        rms = 0.0
        for i in range(len(x)):
            rms += x[i] * x[i]
        rms = np.sqrt(rms / len(x) + eps)

        # Normalize and scale
        out = np.empty_like(x)
        for i in range(len(x)):
            out[i] = (x[i] / rms) * weight[i]

        return out

    @njit(parallel=True, fastmath=True)
    def softmax(x: np.ndarray) -> np.ndarray:
        """Softmax with numerical stability."""
        # Find max for stability
        max_val = x[0]
        for i in range(1, len(x)):
            if x[i] > max_val:
                max_val = x[i]

        # Compute exp and sum
        out = np.empty_like(x)
        sum_exp = 0.0
        for i in prange(len(x)):
            out[i] = np.exp(x[i] - max_val)
            sum_exp += out[i]

        # Normalize
        for i in prange(len(x)):
            out[i] /= sum_exp

        return out

    @njit(fastmath=True)
    def rope_embed(x: np.ndarray, pos: int, theta: float = 10000.0) -> np.ndarray:
        """Apply rotary position embeddings."""
        dim = len(x)
        out = np.empty_like(x)

        for i in range(0, dim, 2):
            freq = 1.0 / (theta ** (i / dim))
            cos_val = np.cos(pos * freq)
            sin_val = np.sin(pos * freq)

            out[i] = x[i] * cos_val - x[i+1] * sin_val
            out[i+1] = x[i] * sin_val + x[i+1] * cos_val

        return out

else:
    # Fallback implementations without SIMD
    def matmul_f32(A, B):
        return A @ B

    def rmsnorm(x, weight, eps=1e-5):
        rms = np.sqrt(np.mean(x ** 2) + eps)
        return (x / rms) * weight

    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def rope_embed(x, pos, theta=10000.0):
        dim = len(x)
        out = np.empty_like(x)
        for i in range(0, dim, 2):
            freq = 1.0 / (theta ** (i / dim))
            cos_val = np.cos(pos * freq)
            sin_val = np.sin(pos * freq)
            out[i] = x[i] * cos_val - x[i+1] * sin_val
            out[i+1] = x[i] * sin_val + x[i+1] * cos_val
        return out
```

#### 3. KV Cache for CPU

```python
@dataclass
class KVCache:
    """CPU-optimized KV cache with sliding window support."""

    num_layers: int
    num_heads: int
    head_dim: int
    max_seq_len: int

    # Cache storage
    k_cache: np.ndarray = None  # [num_layers, max_seq_len, num_heads, head_dim]
    v_cache: np.ndarray = None

    # Current length
    length: int = 0

    # Sliding window
    window_size: Optional[int] = None

    def __post_init__(self):
        self.k_cache = np.zeros(
            (self.num_layers, self.max_seq_len, self.num_heads, self.head_dim),
            dtype=np.float32
        )
        self.v_cache = np.zeros_like(self.k_cache)

    def append(self,
               layer_idx: int,
               k: np.ndarray,
               v: np.ndarray) -> None:
        """
        Append KV to cache.

        k, v: [num_heads, head_dim]
        """
        if self.window_size and self.length >= self.window_size:
            # Shift window
            self.k_cache[layer_idx, :-1] = self.k_cache[layer_idx, 1:]
            self.v_cache[layer_idx, :-1] = self.v_cache[layer_idx, 1:]
            pos = self.window_size - 1
        else:
            pos = self.length

        self.k_cache[layer_idx, pos] = k
        self.v_cache[layer_idx, pos] = v

        if layer_idx == self.num_layers - 1:
            if not self.window_size or self.length < self.window_size:
                self.length += 1

    def get(self, layer_idx: int) -> tuple:
        """Get KV cache for a layer."""
        if self.window_size:
            length = min(self.length, self.window_size)
        else:
            length = self.length

        return (
            self.k_cache[layer_idx, :length],  # [length, num_heads, head_dim]
            self.v_cache[layer_idx, :length]
        )

    def clear(self):
        """Clear cache."""
        self.k_cache.fill(0)
        self.v_cache.fill(0)
        self.length = 0

    def memory_usage(self) -> int:
        """Get memory usage in bytes."""
        return self.k_cache.nbytes + self.v_cache.nbytes
```

#### 4. Transformer Inference

```python
class TransformerInference:
    """On-device transformer inference engine."""

    def __init__(self, loader: GGUFLoader):
        self.loader = loader
        self.config = loader.config

        # Initialize KV cache
        self.kv_cache = KVCache(
            num_layers=self.config.num_layers,
            num_heads=self.config.num_kv_heads,
            head_dim=self.config.hidden_size // self.config.num_heads,
            max_seq_len=self.config.max_seq_len
        )

        # Pre-load frequently used tensors
        self._load_embedding()

    def _load_embedding(self):
        """Load token embedding table."""
        self.embed_tokens = self.loader.get_tensor('token_embd.weight')

    def forward(self, token_id: int, position: int) -> np.ndarray:
        """
        Forward pass for single token.

        Returns logits [vocab_size].
        """
        # Embed token
        hidden = self.embed_tokens[token_id].copy()

        # Process each layer
        for layer_idx in range(self.config.num_layers):
            hidden = self._forward_layer(hidden, layer_idx, position)

        # Final norm
        norm_weight = self.loader.get_tensor('output_norm.weight')
        hidden = rmsnorm(hidden, norm_weight, self.config.norm_eps)

        # Output projection
        lm_head = self.loader.get_tensor('output.weight')
        logits = matmul_f32(hidden.reshape(1, -1), lm_head.T).flatten()

        return logits

    def _forward_layer(self,
                       hidden: np.ndarray,
                       layer_idx: int,
                       position: int) -> np.ndarray:
        """Forward pass for one transformer layer."""
        # Input norm
        norm_weight = self.loader.get_tensor(
            f'blk.{layer_idx}.attn_norm.weight'
        )
        normed = rmsnorm(hidden, norm_weight, self.config.norm_eps)

        # Self-attention
        attn_out = self._attention(normed, layer_idx, position)

        # Residual
        hidden = hidden + attn_out

        # FFN norm
        ffn_norm = self.loader.get_tensor(
            f'blk.{layer_idx}.ffn_norm.weight'
        )
        normed = rmsnorm(hidden, ffn_norm, self.config.norm_eps)

        # FFN
        ffn_out = self._ffn(normed, layer_idx)

        # Residual
        hidden = hidden + ffn_out

        return hidden

    def _attention(self,
                   x: np.ndarray,
                   layer_idx: int,
                   position: int) -> np.ndarray:
        """Multi-head attention with GQA support."""
        head_dim = self.config.hidden_size // self.config.num_heads
        num_heads = self.config.num_heads
        num_kv_heads = self.config.num_kv_heads

        # Load weights
        wq = self.loader.get_tensor(f'blk.{layer_idx}.attn_q.weight')
        wk = self.loader.get_tensor(f'blk.{layer_idx}.attn_k.weight')
        wv = self.loader.get_tensor(f'blk.{layer_idx}.attn_v.weight')
        wo = self.loader.get_tensor(f'blk.{layer_idx}.attn_output.weight')

        # Project Q, K, V
        q = matmul_f32(x.reshape(1, -1), wq.T).flatten()
        k = matmul_f32(x.reshape(1, -1), wk.T).flatten()
        v = matmul_f32(x.reshape(1, -1), wv.T).flatten()

        # Reshape to heads
        q = q.reshape(num_heads, head_dim)
        k = k.reshape(num_kv_heads, head_dim)
        v = v.reshape(num_kv_heads, head_dim)

        # Apply RoPE
        for h in range(num_heads):
            q[h] = rope_embed(q[h], position, self.config.rope_theta)
        for h in range(num_kv_heads):
            k[h] = rope_embed(k[h], position, self.config.rope_theta)

        # Update KV cache
        self.kv_cache.append(layer_idx, k, v)

        # Get cached KV
        k_cache, v_cache = self.kv_cache.get(layer_idx)

        # Compute attention
        output = np.zeros(num_heads * head_dim, dtype=np.float32)

        # GQA: repeat KV heads if needed
        kv_repeat = num_heads // num_kv_heads

        for h in range(num_heads):
            kv_h = h // kv_repeat  # KV head index

            # Attention scores
            scores = np.zeros(len(k_cache), dtype=np.float32)
            for i in range(len(k_cache)):
                scores[i] = np.dot(q[h], k_cache[i, kv_h]) / np.sqrt(head_dim)

            # Softmax
            probs = softmax(scores)

            # Weighted sum of values
            head_out = np.zeros(head_dim, dtype=np.float32)
            for i in range(len(v_cache)):
                head_out += probs[i] * v_cache[i, kv_h]

            output[h * head_dim:(h + 1) * head_dim] = head_out

        # Output projection
        return matmul_f32(output.reshape(1, -1), wo.T).flatten()

    def _ffn(self, x: np.ndarray, layer_idx: int) -> np.ndarray:
        """Feed-forward network (SwiGLU)."""
        # Load weights
        w_gate = self.loader.get_tensor(f'blk.{layer_idx}.ffn_gate.weight')
        w_up = self.loader.get_tensor(f'blk.{layer_idx}.ffn_up.weight')
        w_down = self.loader.get_tensor(f'blk.{layer_idx}.ffn_down.weight')

        # Gate and up projections
        gate = matmul_f32(x.reshape(1, -1), w_gate.T).flatten()
        up = matmul_f32(x.reshape(1, -1), w_up.T).flatten()

        # SiLU activation on gate
        gate = gate * (1.0 / (1.0 + np.exp(-gate)))

        # Element-wise product
        hidden = gate * up

        # Down projection
        return matmul_f32(hidden.reshape(1, -1), w_down.T).flatten()


class Sampler:
    """Token sampling strategies."""

    def __init__(self,
                 temperature: float = 1.0,
                 top_k: int = 40,
                 top_p: float = 0.9):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

    def sample(self, logits: np.ndarray) -> int:
        """Sample next token from logits."""
        # Temperature
        if self.temperature != 1.0:
            logits = logits / self.temperature

        # Top-k
        if self.top_k > 0:
            indices = np.argsort(logits)[-self.top_k:]
            mask = np.ones_like(logits) * float('-inf')
            mask[indices] = logits[indices]
            logits = mask

        # Softmax
        probs = softmax(logits)

        # Top-p
        if self.top_p < 1.0:
            sorted_indices = np.argsort(probs)[::-1]
            cumsum = np.cumsum(probs[sorted_indices])
            cutoff_idx = np.searchsorted(cumsum, self.top_p) + 1
            mask = np.zeros_like(probs)
            mask[sorted_indices[:cutoff_idx]] = probs[sorted_indices[:cutoff_idx]]
            probs = mask / mask.sum()

        # Sample
        return np.random.choice(len(probs), p=probs)
```

### Enterprise Features

#### Multi-Backend Support

```python
from abc import ABC, abstractmethod
from enum import Enum

class Backend(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    METAL = "metal"
    VULKAN = "vulkan"

class ComputeBackend(ABC):
    """Abstract backend for compute operations."""

    @abstractmethod
    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def softmax(self, x: np.ndarray) -> np.ndarray:
        pass


class CPUBackend(ComputeBackend):
    """CPU backend with SIMD."""

    def matmul(self, A, B):
        return matmul_f32(A, B)

    def softmax(self, x):
        return softmax(x)


class CUDABackend(ComputeBackend):
    """CUDA GPU backend."""

    def __init__(self):
        import cupy as cp
        self.cp = cp

    def matmul(self, A, B):
        A_gpu = self.cp.asarray(A)
        B_gpu = self.cp.asarray(B)
        return self.cp.asnumpy(A_gpu @ B_gpu)

    def softmax(self, x):
        x_gpu = self.cp.asarray(x)
        exp_x = self.cp.exp(x_gpu - self.cp.max(x_gpu))
        return self.cp.asnumpy(exp_x / exp_x.sum())


class ThreadPoolInference:
    """Inference with threadpool parallelization."""

    def __init__(self,
                 model: TransformerInference,
                 num_threads: int = 4):
        from concurrent.futures import ThreadPoolExecutor
        self.model = model
        self.executor = ThreadPoolExecutor(max_workers=num_threads)

    def batch_forward(self,
                      token_ids: list,
                      positions: list) -> list:
        """Process multiple tokens in parallel."""
        futures = []
        for tid, pos in zip(token_ids, positions):
            future = self.executor.submit(self.model.forward, tid, pos)
            futures.append(future)

        return [f.result() for f in futures]
```

## API Reference

### Load and Run Model

```python
# Load model
loader = GGUFLoader('model-q4.gguf')
model = TransformerInference(loader)
sampler = Sampler(temperature=0.7, top_k=40)

# Generate text
prompt_tokens = [1, 15043, 29892]  # Tokenized prompt
output_tokens = []

for i, token in enumerate(prompt_tokens[:-1]):
    model.forward(token, i)  # Fill KV cache

# Generate
position = len(prompt_tokens) - 1
token = prompt_tokens[-1]

for _ in range(100):
    logits = model.forward(token, position)
    token = sampler.sample(logits)
    output_tokens.append(token)
    position += 1

    if token == 2:  # EOS
        break
```

## Implementation Phases

### Phase 1: GGUF Loader (Weeks 1-2)
- Parse GGUF header
- Memory mapping
- Tensor access
- Q4/Q8 dequantization

### Phase 2: SIMD Kernels (Weeks 3-4)
- Matrix multiplication
- RMSNorm
- Softmax
- RoPE embeddings

### Phase 3: Inference Engine (Weeks 5-6)
- Transformer forward pass
- KV cache
- Attention with GQA
- FFN (SwiGLU)

### Phase 4: Optimization (Weeks 7-9)
- Kernel tuning
- Memory optimization
- Batch processing
- Sliding window

### Phase 5: Multi-Backend (Weeks 10-14)
- CUDA backend
- Metal backend
- Vulkan compute shaders
- Speculative decoding

## Performance Targets

| Metric | Target | Hardware |
|--------|--------|----------|
| Tokens/sec | >20 | M2 CPU, 7B Q4 |
| Tokens/sec | >100 | RTX 4090, 7B Q4 |
| Memory | <4GB | 7B Q4 model |
| Startup | <2s | With mmap |

## Dependencies

- NumPy
- Numba (optional, for SIMD)
- CuPy (optional, for CUDA)

## References

- llama.cpp
- GGML/GGUF format
- LLaMA architecture
