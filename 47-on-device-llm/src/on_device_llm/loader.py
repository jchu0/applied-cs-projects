"""GGUF model loader with memory mapping support.

This module provides functionality to load GGUF format models with
memory mapping for efficient on-device inference.
"""

import mmap
import struct
from dataclasses import dataclass
from typing import Any, BinaryIO, Dict, List, Optional, Tuple

import numpy as np

from on_device_llm.quantization import GGMLType, dequantize_q4_0, dequantize_q8_0, calc_tensor_size


@dataclass
class TensorInfo:
    """Metadata for a tensor in GGUF file."""

    name: str
    shape: Tuple[int, ...]
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

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.hidden_size // self.num_heads


class GGUFLoader:
    """
    Load GGUF format models with memory mapping.

    Memory mapping allows:
    - Lazy loading (only load what's accessed)
    - Shared memory across processes
    - No full copy into RAM
    """

    GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian

    # GGUF value types
    VALUE_TYPE_UINT8 = 0
    VALUE_TYPE_INT8 = 1
    VALUE_TYPE_UINT16 = 2
    VALUE_TYPE_INT16 = 3
    VALUE_TYPE_UINT32 = 4
    VALUE_TYPE_INT32 = 5
    VALUE_TYPE_FLOAT32 = 6
    VALUE_TYPE_BOOL = 7
    VALUE_TYPE_STRING = 8
    VALUE_TYPE_ARRAY = 9
    VALUE_TYPE_UINT64 = 10
    VALUE_TYPE_INT64 = 11
    VALUE_TYPE_FLOAT64 = 12

    def __init__(self, path: str):
        """
        Initialize GGUF loader.

        Args:
            path: Path to GGUF file
        """
        self.path = path
        self.file: Optional[BinaryIO] = None
        self.mmap_obj: Optional[mmap.mmap] = None

        self.metadata: Dict[str, Any] = {}
        self.tensors: Dict[str, TensorInfo] = {}
        self.config: Optional[ModelConfig] = None

        self._data_offset = 0  # Offset where tensor data begins

        self._load_header()

    def _load_header(self) -> None:
        """Parse GGUF header and tensor info."""
        self.file = open(self.path, 'rb')

        # Read magic
        magic = struct.unpack('<I', self.file.read(4))[0]
        if magic != self.GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF magic: {magic:08x}, expected {self.GGUF_MAGIC:08x}")

        # Read version
        version = struct.unpack('<I', self.file.read(4))[0]
        if version < 2:
            raise ValueError(f"GGUF version {version} not supported (minimum: 2)")

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
        tensor_infos: List[Tuple[str, List[int], GGMLType]] = []
        for _ in range(n_tensors):
            name, shape, dtype = self._read_tensor_header()
            tensor_infos.append((name, shape, dtype))

        # Calculate data offset (aligned to 32 bytes)
        current_offset = self.file.tell()
        self._data_offset = (current_offset + 31) // 32 * 32

        # Calculate tensor offsets and sizes
        offset = self._data_offset
        for name, shape, dtype in tensor_infos:
            # Align to 32 bytes
            offset = (offset + 31) // 32 * 32
            size_bytes = calc_tensor_size(tuple(shape), dtype)

            self.tensors[name] = TensorInfo(
                name=name,
                shape=tuple(shape),
                dtype=dtype,
                offset=offset,
                size_bytes=size_bytes
            )
            offset += size_bytes

        # Create memory map
        self.mmap_obj = mmap.mmap(
            self.file.fileno(), 0,
            access=mmap.ACCESS_READ
        )

    def _read_string(self) -> str:
        """Read a length-prefixed string."""
        length = struct.unpack('<Q', self.file.read(8))[0]
        return self.file.read(length).decode('utf-8')

    def _read_metadata_kv(self) -> Tuple[str, Any]:
        """Read a metadata key-value pair."""
        # Read key
        key = self._read_string()

        # Read value type and value
        value_type = struct.unpack('<I', self.file.read(4))[0]
        value = self._read_value(value_type)

        return key, value

    def _read_value(self, value_type: int) -> Any:
        """Read a value of the given type."""
        if value_type == self.VALUE_TYPE_UINT8:
            return struct.unpack('<B', self.file.read(1))[0]
        elif value_type == self.VALUE_TYPE_INT8:
            return struct.unpack('<b', self.file.read(1))[0]
        elif value_type == self.VALUE_TYPE_UINT16:
            return struct.unpack('<H', self.file.read(2))[0]
        elif value_type == self.VALUE_TYPE_INT16:
            return struct.unpack('<h', self.file.read(2))[0]
        elif value_type == self.VALUE_TYPE_UINT32:
            return struct.unpack('<I', self.file.read(4))[0]
        elif value_type == self.VALUE_TYPE_INT32:
            return struct.unpack('<i', self.file.read(4))[0]
        elif value_type == self.VALUE_TYPE_FLOAT32:
            return struct.unpack('<f', self.file.read(4))[0]
        elif value_type == self.VALUE_TYPE_BOOL:
            return struct.unpack('<B', self.file.read(1))[0] != 0
        elif value_type == self.VALUE_TYPE_STRING:
            return self._read_string()
        elif value_type == self.VALUE_TYPE_ARRAY:
            return self._read_array()
        elif value_type == self.VALUE_TYPE_UINT64:
            return struct.unpack('<Q', self.file.read(8))[0]
        elif value_type == self.VALUE_TYPE_INT64:
            return struct.unpack('<q', self.file.read(8))[0]
        elif value_type == self.VALUE_TYPE_FLOAT64:
            return struct.unpack('<d', self.file.read(8))[0]
        else:
            raise ValueError(f"Unknown value type: {value_type}")

    def _read_array(self) -> List[Any]:
        """Read an array value."""
        element_type = struct.unpack('<I', self.file.read(4))[0]
        length = struct.unpack('<Q', self.file.read(8))[0]

        return [self._read_value(element_type) for _ in range(length)]

    def _read_tensor_header(self) -> Tuple[str, List[int], GGMLType]:
        """Read tensor metadata."""
        name = self._read_string()

        # Dimensions
        n_dims = struct.unpack('<I', self.file.read(4))[0]
        shape = []
        for _ in range(n_dims):
            dim = struct.unpack('<Q', self.file.read(8))[0]
            shape.append(dim)

        # Type
        dtype = GGMLType(struct.unpack('<I', self.file.read(4))[0])

        # Offset (relative to data section)
        _ = struct.unpack('<Q', self.file.read(8))[0]

        return name, shape, dtype

    def _parse_config(self) -> ModelConfig:
        """Extract model config from metadata."""
        # Try different naming conventions
        def get_value(keys: List[str], default: Any) -> Any:
            for key in keys:
                if key in self.metadata:
                    return self.metadata[key]
            return default

        return ModelConfig(
            vocab_size=get_value([
                'llama.vocab_size',
                'vocab_size',
                'tokenizer.ggml.vocab_size'
            ], 32000),
            hidden_size=get_value([
                'llama.embedding_length',
                'embedding_length',
                'hidden_size'
            ], 4096),
            num_layers=get_value([
                'llama.block_count',
                'block_count',
                'num_hidden_layers'
            ], 32),
            num_heads=get_value([
                'llama.attention.head_count',
                'attention.head_count',
                'num_attention_heads'
            ], 32),
            num_kv_heads=get_value([
                'llama.attention.head_count_kv',
                'attention.head_count_kv',
                'num_key_value_heads'
            ], 32),
            intermediate_size=get_value([
                'llama.feed_forward_length',
                'feed_forward_length',
                'intermediate_size'
            ], 11008),
            max_seq_len=get_value([
                'llama.context_length',
                'context_length',
                'max_position_embeddings'
            ], 2048),
            rope_theta=get_value([
                'llama.rope.freq_base',
                'rope.freq_base',
                'rope_theta'
            ], 10000.0),
            norm_eps=get_value([
                'llama.attention.layer_norm_rms_epsilon',
                'attention.layer_norm_rms_epsilon',
                'rms_norm_eps'
            ], 1e-5)
        )

    def get_tensor(self, name: str) -> np.ndarray:
        """
        Get tensor data via memory mapping.

        Uses lazy loading - data is only read from disk when accessed.

        Args:
            name: Tensor name

        Returns:
            Dequantized tensor as float32 ndarray
        """
        if name not in self.tensors:
            raise KeyError(f"Tensor not found: {name}")

        info = self.tensors[name]

        # Read from mmap
        data = self.mmap_obj[info.offset:info.offset + info.size_bytes]

        # Dequantize if needed
        if info.dtype == GGMLType.F32:
            return np.frombuffer(data, dtype=np.float32).reshape(info.shape).copy()
        elif info.dtype == GGMLType.F16:
            return np.frombuffer(data, dtype=np.float16).astype(np.float32).reshape(info.shape)
        elif info.dtype == GGMLType.Q8_0:
            return dequantize_q8_0(data, info.shape)
        elif info.dtype == GGMLType.Q4_0:
            return dequantize_q4_0(data, info.shape)
        else:
            raise NotImplementedError(f"Dequantization for {info.dtype} not implemented")

    def get_tensor_info(self, name: str) -> Optional[TensorInfo]:
        """Get tensor metadata."""
        return self.tensors.get(name)

    def list_tensors(self) -> List[str]:
        """List all tensor names."""
        return list(self.tensors.keys())

    def close(self) -> None:
        """Close file and memory map."""
        if self.mmap_obj:
            self.mmap_obj.close()
            self.mmap_obj = None
        if self.file:
            self.file.close()
            self.file = None

    def __enter__(self) -> 'GGUFLoader':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


class MockGGUFLoader:
    """
    Mock GGUF loader for testing without actual model files.

    Creates synthetic model weights with proper shapes based on config.
    """

    def __init__(
        self,
        config: ModelConfig = None,
        dtype: GGMLType = GGMLType.F32,
        *,
        vocab_size: int = None,
        hidden_size: int = None,
        num_layers: int = None,
        num_heads: int = None,
        num_kv_heads: int = None,
        intermediate_size: int = None,
        max_seq_len: int = 2048,
    ):
        """
        Initialize mock loader.

        Args:
            config: Model configuration (alternative to individual params)
            dtype: Simulated quantization type
            vocab_size: Vocabulary size (if not using config)
            hidden_size: Hidden dimension (if not using config)
            num_layers: Number of transformer layers (if not using config)
            num_heads: Number of attention heads (if not using config)
            num_kv_heads: Number of KV heads for GQA (defaults to num_heads)
            intermediate_size: FFN intermediate size (defaults to 4*hidden_size)
            max_seq_len: Maximum sequence length
        """
        if config is not None:
            self.config = config
        else:
            # Build config from individual parameters
            if num_kv_heads is None:
                num_kv_heads = num_heads
            if intermediate_size is None:
                intermediate_size = hidden_size * 4 if hidden_size else 256
            self.config = ModelConfig(
                vocab_size=vocab_size or 1000,
                hidden_size=hidden_size or 64,
                num_layers=num_layers or 4,
                num_heads=num_heads or 4,
                num_kv_heads=num_kv_heads,
                intermediate_size=intermediate_size,
                max_seq_len=max_seq_len,
            )
        self.dtype = dtype
        self.metadata: Dict[str, Any] = {}
        self.tensors: Dict[str, TensorInfo] = {}

        self._tensor_data: Dict[str, np.ndarray] = {}
        self._generate_tensors()

    def _generate_tensors(self) -> None:
        """Generate synthetic tensor data."""
        cfg = self.config

        # Embedding table
        self._add_tensor(
            'token_embd.weight',
            (cfg.vocab_size, cfg.hidden_size)
        )

        # Layers
        for i in range(cfg.num_layers):
            prefix = f'blk.{i}'

            # Attention norm
            self._add_tensor(f'{prefix}.attn_norm.weight', (cfg.hidden_size,))

            # Attention weights
            self._add_tensor(
                f'{prefix}.attn_q.weight',
                (cfg.num_heads * cfg.head_dim, cfg.hidden_size)
            )
            self._add_tensor(
                f'{prefix}.attn_k.weight',
                (cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size)
            )
            self._add_tensor(
                f'{prefix}.attn_v.weight',
                (cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size)
            )
            self._add_tensor(
                f'{prefix}.attn_output.weight',
                (cfg.hidden_size, cfg.num_heads * cfg.head_dim)
            )

            # FFN norm
            self._add_tensor(f'{prefix}.ffn_norm.weight', (cfg.hidden_size,))

            # FFN weights (SwiGLU)
            self._add_tensor(
                f'{prefix}.ffn_gate.weight',
                (cfg.intermediate_size, cfg.hidden_size)
            )
            self._add_tensor(
                f'{prefix}.ffn_up.weight',
                (cfg.intermediate_size, cfg.hidden_size)
            )
            self._add_tensor(
                f'{prefix}.ffn_down.weight',
                (cfg.hidden_size, cfg.intermediate_size)
            )

        # Output norm and head
        self._add_tensor('output_norm.weight', (cfg.hidden_size,))
        self._add_tensor('output.weight', (cfg.vocab_size, cfg.hidden_size))

    def _add_tensor(self, name: str, shape: Tuple[int, ...]) -> None:
        """Add a synthetic tensor."""
        # Use small random values (Xavier-like init)
        fan_in = shape[-1] if len(shape) > 1 else shape[0]
        std = 1.0 / np.sqrt(fan_in)

        data = np.random.randn(*shape).astype(np.float32) * std
        self._tensor_data[name] = data

        self.tensors[name] = TensorInfo(
            name=name,
            shape=shape,
            dtype=self.dtype,
            offset=0,
            size_bytes=data.nbytes
        )

    def get_tensor(self, name: str) -> np.ndarray:
        """Get tensor data."""
        if name not in self._tensor_data:
            raise KeyError(f"Tensor not found: {name}")
        return self._tensor_data[name].copy()

    def get_tensor_info(self, name: str) -> Optional[TensorInfo]:
        """Get tensor metadata."""
        return self.tensors.get(name)

    def list_tensors(self) -> List[str]:
        """List all tensor names."""
        return list(self.tensors.keys())

    def close(self) -> None:
        """No-op for mock loader."""
        pass
