"""Core model representation for on-device LLM."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import numpy as np
import struct
import io


class QuantizationType(Enum):
    """Quantization types for mobile."""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"
    GGML_Q4_0 = "q4_0"  # 4-bit with block scaling
    GGML_Q4_1 = "q4_1"  # 4-bit with block scaling + min
    GGML_Q8_0 = "q8_0"  # 8-bit with block scaling


class DeviceType(Enum):
    """Target device types."""
    CPU = "cpu"
    GPU_METAL = "metal"  # Apple Metal
    GPU_VULKAN = "vulkan"  # Cross-platform
    NPU = "npu"  # Neural processing unit
    DSP = "dsp"  # Digital signal processor


@dataclass
class TensorInfo:
    """Tensor metadata."""
    name: str
    shape: tuple[int, ...]
    dtype: QuantizationType
    offset: int  # Offset in weight file
    size_bytes: int
    block_size: int = 32  # For block quantization


@dataclass
class ModelConfig:
    """LLM model configuration."""
    vocab_size: int = 32000
    hidden_size: int = 2048
    num_layers: int = 22
    num_heads: int = 32
    num_kv_heads: int = 4  # Grouped query attention
    intermediate_size: int = 5632
    max_position: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    tie_embeddings: bool = True


@dataclass
class ModelMetadata:
    """Model metadata for serialization."""
    name: str
    version: str
    config: ModelConfig
    quantization: QuantizationType
    tensor_count: int
    param_count: int
    file_size: int


class QuantizedTensor:
    """Quantized tensor for efficient storage."""

    def __init__(
        self,
        data: np.ndarray,
        scales: np.ndarray | None = None,
        zeros: np.ndarray | None = None,
        qtype: QuantizationType = QuantizationType.FP16,
        block_size: int = 32
    ):
        self.data = data
        self.scales = scales
        self.zeros = zeros
        self.qtype = qtype
        self.block_size = block_size
        self._shape = data.shape

    @property
    def shape(self) -> tuple[int, ...]:
        if self.qtype in [QuantizationType.INT4, QuantizationType.GGML_Q4_0]:
            # Packed 4-bit: actual elements = packed * 2
            shape = list(self._shape)
            shape[-1] *= 2
            return tuple(shape)
        return self._shape

    def dequantize(self) -> np.ndarray:
        """Dequantize to FP32."""
        if self.qtype == QuantizationType.FP32:
            return self.data.astype(np.float32)

        if self.qtype == QuantizationType.FP16:
            return self.data.astype(np.float32)

        if self.qtype == QuantizationType.INT8:
            if self.scales is not None:
                return (self.data.astype(np.float32) - 128) * self.scales
            return self.data.astype(np.float32)

        if self.qtype in [QuantizationType.INT4, QuantizationType.GGML_Q4_0]:
            # Unpack 4-bit values
            low = self.data & 0x0F
            high = (self.data >> 4) & 0x0F
            unpacked = np.stack([low, high], axis=-1).reshape(self.shape)

            if self.scales is not None:
                # Block dequantization
                scale_shape = [1] * (len(unpacked.shape) - 1) + [-1]
                scales = self.scales.reshape(scale_shape)
                return (unpacked.astype(np.float32) - 8) * scales

            return unpacked.astype(np.float32)

        return self.data.astype(np.float32)

    @classmethod
    def quantize(
        cls,
        data: np.ndarray,
        qtype: QuantizationType,
        block_size: int = 32
    ) -> "QuantizedTensor":
        """Quantize FP32 tensor."""
        if qtype == QuantizationType.FP32:
            return cls(data, qtype=qtype)

        if qtype == QuantizationType.FP16:
            return cls(data.astype(np.float16), qtype=qtype)

        if qtype == QuantizationType.INT8:
            # Per-tensor quantization
            absmax = np.abs(data).max()
            scale = absmax / 127 if absmax > 0 else 1.0
            quantized = np.round(data / scale).astype(np.int8)
            return cls(quantized, scales=np.array([scale]), qtype=qtype)

        if qtype in [QuantizationType.INT4, QuantizationType.GGML_Q4_0]:
            # Block-wise 4-bit quantization
            flat = data.flatten()
            n_blocks = (len(flat) + block_size - 1) // block_size
            padded = np.pad(flat, (0, n_blocks * block_size - len(flat)))
            blocks = padded.reshape(n_blocks, block_size)

            # Compute scales per block
            scales = np.abs(blocks).max(axis=1) / 7
            scales = np.where(scales == 0, 1.0, scales)

            # Quantize to 4-bit
            quantized = np.round(blocks / scales[:, None]).astype(np.int8)
            quantized = np.clip(quantized, -8, 7) + 8  # Shift to unsigned

            # Pack two 4-bit values into one byte
            packed = (quantized[:, ::2] | (quantized[:, 1::2] << 4)).astype(np.uint8)

            return cls(
                packed.reshape(data.shape[:-1] + (-1,)),
                scales=scales.astype(np.float32),
                qtype=qtype,
                block_size=block_size
            )

        return cls(data, qtype=qtype)

    def nbytes(self) -> int:
        """Get memory size in bytes."""
        total = self.data.nbytes
        if self.scales is not None:
            total += self.scales.nbytes
        if self.zeros is not None:
            total += self.zeros.nbytes
        return total


@dataclass
class TransformerLayer:
    """Single transformer layer weights."""
    # Attention
    q_proj: QuantizedTensor
    k_proj: QuantizedTensor
    v_proj: QuantizedTensor
    o_proj: QuantizedTensor

    # MLP
    gate_proj: QuantizedTensor
    up_proj: QuantizedTensor
    down_proj: QuantizedTensor

    # Norms
    input_norm: np.ndarray  # Usually FP32
    post_attn_norm: np.ndarray


@dataclass
class LLMWeights:
    """Complete LLM weights."""
    config: ModelConfig
    embed_tokens: QuantizedTensor
    layers: list[TransformerLayer]
    norm: np.ndarray
    lm_head: QuantizedTensor | None  # None if tied with embeddings

    def param_count(self) -> int:
        """Count total parameters."""
        count = np.prod(self.embed_tokens.shape)
        for layer in self.layers:
            count += np.prod(layer.q_proj.shape)
            count += np.prod(layer.k_proj.shape)
            count += np.prod(layer.v_proj.shape)
            count += np.prod(layer.o_proj.shape)
            count += np.prod(layer.gate_proj.shape)
            count += np.prod(layer.up_proj.shape)
            count += np.prod(layer.down_proj.shape)
            count += len(layer.input_norm)
            count += len(layer.post_attn_norm)
        count += len(self.norm)
        if self.lm_head:
            count += np.prod(self.lm_head.shape)
        return int(count)

    def memory_size(self) -> int:
        """Get total memory size in bytes."""
        size = self.embed_tokens.nbytes()
        for layer in self.layers:
            size += layer.q_proj.nbytes()
            size += layer.k_proj.nbytes()
            size += layer.v_proj.nbytes()
            size += layer.o_proj.nbytes()
            size += layer.gate_proj.nbytes()
            size += layer.up_proj.nbytes()
            size += layer.down_proj.nbytes()
            size += layer.input_norm.nbytes
            size += layer.post_attn_norm.nbytes
        size += self.norm.nbytes
        if self.lm_head:
            size += self.lm_head.nbytes()
        return size


class ModelSerializer:
    """Serialize/deserialize models to/from files."""

    MAGIC = b"DLLM"
    VERSION = 1

    def __init__(self):
        self.tensor_infos: list[TensorInfo] = []

    def save(self, weights: LLMWeights, filepath: str) -> None:
        """Save model to file."""
        with open(filepath, 'wb') as f:
            # Write header
            f.write(self.MAGIC)
            f.write(struct.pack('<I', self.VERSION))

            # Write config
            config_data = self._serialize_config(weights.config)
            f.write(struct.pack('<I', len(config_data)))
            f.write(config_data)

            # Collect all tensors
            tensors = self._collect_tensors(weights)

            # Write tensor count
            f.write(struct.pack('<I', len(tensors)))

            # Write tensor metadata
            data_offset = f.tell() + sum(
                4 + len(name.encode()) + 4 + 4 * len(shape) + 4 + 8 + 4
                for name, tensor, shape in tensors
            )

            for name, tensor, shape in tensors:
                name_bytes = name.encode('utf-8')
                f.write(struct.pack('<I', len(name_bytes)))
                f.write(name_bytes)
                f.write(struct.pack('<I', len(shape)))
                for dim in shape:
                    f.write(struct.pack('<I', dim))
                f.write(struct.pack('<I', tensor.qtype.value.encode()[0] if isinstance(tensor.qtype.value, str) else 0))
                f.write(struct.pack('<Q', data_offset))
                tensor_bytes = self._tensor_to_bytes(tensor)
                f.write(struct.pack('<I', len(tensor_bytes)))
                data_offset += len(tensor_bytes)

            # Write tensor data
            for name, tensor, shape in tensors:
                f.write(self._tensor_to_bytes(tensor))

    def load(self, filepath: str) -> LLMWeights:
        """Load model from file."""
        with open(filepath, 'rb') as f:
            # Read header
            magic = f.read(4)
            if magic != self.MAGIC:
                raise ValueError("Invalid model file")

            version = struct.unpack('<I', f.read(4))[0]
            if version != self.VERSION:
                raise ValueError(f"Unsupported version: {version}")

            # Read config
            config_len = struct.unpack('<I', f.read(4))[0]
            config_data = f.read(config_len)
            config = self._deserialize_config(config_data)

            # Read tensor count
            tensor_count = struct.unpack('<I', f.read(4))[0]

            # Read tensor metadata
            tensor_infos = []
            for _ in range(tensor_count):
                name_len = struct.unpack('<I', f.read(4))[0]
                name = f.read(name_len).decode('utf-8')
                ndims = struct.unpack('<I', f.read(4))[0]
                shape = tuple(struct.unpack('<I', f.read(4))[0] for _ in range(ndims))
                qtype_byte = struct.unpack('<I', f.read(4))[0]
                offset = struct.unpack('<Q', f.read(8))[0]
                size = struct.unpack('<I', f.read(4))[0]

                tensor_infos.append(TensorInfo(
                    name=name,
                    shape=shape,
                    dtype=QuantizationType.FP16,  # Simplified
                    offset=offset,
                    size_bytes=size
                ))

            # Read tensors and reconstruct weights
            # Simplified: would need full implementation
            return self._reconstruct_weights(f, config, tensor_infos)

    def _serialize_config(self, config: ModelConfig) -> bytes:
        """Serialize config to bytes."""
        buffer = io.BytesIO()
        buffer.write(struct.pack('<I', config.vocab_size))
        buffer.write(struct.pack('<I', config.hidden_size))
        buffer.write(struct.pack('<I', config.num_layers))
        buffer.write(struct.pack('<I', config.num_heads))
        buffer.write(struct.pack('<I', config.num_kv_heads))
        buffer.write(struct.pack('<I', config.intermediate_size))
        buffer.write(struct.pack('<I', config.max_position))
        buffer.write(struct.pack('<f', config.rope_theta))
        buffer.write(struct.pack('<f', config.rms_norm_eps))
        buffer.write(struct.pack('<?', config.tie_embeddings))
        return buffer.getvalue()

    def _deserialize_config(self, data: bytes) -> ModelConfig:
        """Deserialize config from bytes."""
        buffer = io.BytesIO(data)
        return ModelConfig(
            vocab_size=struct.unpack('<I', buffer.read(4))[0],
            hidden_size=struct.unpack('<I', buffer.read(4))[0],
            num_layers=struct.unpack('<I', buffer.read(4))[0],
            num_heads=struct.unpack('<I', buffer.read(4))[0],
            num_kv_heads=struct.unpack('<I', buffer.read(4))[0],
            intermediate_size=struct.unpack('<I', buffer.read(4))[0],
            max_position=struct.unpack('<I', buffer.read(4))[0],
            rope_theta=struct.unpack('<f', buffer.read(4))[0],
            rms_norm_eps=struct.unpack('<f', buffer.read(4))[0],
            tie_embeddings=struct.unpack('<?', buffer.read(1))[0]
        )

    def _collect_tensors(self, weights: LLMWeights) -> list[tuple[str, QuantizedTensor, tuple]]:
        """Collect all tensors with names."""
        tensors = []
        tensors.append(("embed_tokens", weights.embed_tokens, weights.embed_tokens.shape))

        for i, layer in enumerate(weights.layers):
            prefix = f"layers.{i}"
            tensors.append((f"{prefix}.q_proj", layer.q_proj, layer.q_proj.shape))
            tensors.append((f"{prefix}.k_proj", layer.k_proj, layer.k_proj.shape))
            tensors.append((f"{prefix}.v_proj", layer.v_proj, layer.v_proj.shape))
            tensors.append((f"{prefix}.o_proj", layer.o_proj, layer.o_proj.shape))
            tensors.append((f"{prefix}.gate_proj", layer.gate_proj, layer.gate_proj.shape))
            tensors.append((f"{prefix}.up_proj", layer.up_proj, layer.up_proj.shape))
            tensors.append((f"{prefix}.down_proj", layer.down_proj, layer.down_proj.shape))

        if weights.lm_head:
            tensors.append(("lm_head", weights.lm_head, weights.lm_head.shape))

        return tensors

    def _tensor_to_bytes(self, tensor: QuantizedTensor) -> bytes:
        """Convert tensor to bytes."""
        buffer = io.BytesIO()
        buffer.write(tensor.data.tobytes())
        if tensor.scales is not None:
            buffer.write(tensor.scales.tobytes())
        if tensor.zeros is not None:
            buffer.write(tensor.zeros.tobytes())
        return buffer.getvalue()

    def _reconstruct_weights(
        self,
        f: Any,
        config: ModelConfig,
        tensor_infos: list[TensorInfo]
    ) -> LLMWeights:
        """Reconstruct weights from file."""
        # Simplified placeholder
        # Would need to read each tensor and reconstruct full structure
        embed = QuantizedTensor(
            np.zeros((config.vocab_size, config.hidden_size), dtype=np.float16),
            qtype=QuantizationType.FP16
        )

        layers = []
        for _ in range(config.num_layers):
            layer = TransformerLayer(
                q_proj=QuantizedTensor(np.zeros((config.hidden_size, config.hidden_size), dtype=np.float16), qtype=QuantizationType.FP16),
                k_proj=QuantizedTensor(np.zeros((config.hidden_size, config.hidden_size // config.num_heads * config.num_kv_heads), dtype=np.float16), qtype=QuantizationType.FP16),
                v_proj=QuantizedTensor(np.zeros((config.hidden_size, config.hidden_size // config.num_heads * config.num_kv_heads), dtype=np.float16), qtype=QuantizationType.FP16),
                o_proj=QuantizedTensor(np.zeros((config.hidden_size, config.hidden_size), dtype=np.float16), qtype=QuantizationType.FP16),
                gate_proj=QuantizedTensor(np.zeros((config.hidden_size, config.intermediate_size), dtype=np.float16), qtype=QuantizationType.FP16),
                up_proj=QuantizedTensor(np.zeros((config.hidden_size, config.intermediate_size), dtype=np.float16), qtype=QuantizationType.FP16),
                down_proj=QuantizedTensor(np.zeros((config.intermediate_size, config.hidden_size), dtype=np.float16), qtype=QuantizationType.FP16),
                input_norm=np.ones(config.hidden_size, dtype=np.float32),
                post_attn_norm=np.ones(config.hidden_size, dtype=np.float32)
            )
            layers.append(layer)

        return LLMWeights(
            config=config,
            embed_tokens=embed,
            layers=layers,
            norm=np.ones(config.hidden_size, dtype=np.float32),
            lm_head=None
        )


def create_test_model(
    hidden_size: int = 512,
    num_layers: int = 4,
    qtype: QuantizationType = QuantizationType.INT4
) -> LLMWeights:
    """Create a small test model."""
    config = ModelConfig(
        vocab_size=1000,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=8,
        num_kv_heads=2,
        intermediate_size=hidden_size * 4,
        max_position=512
    )

    # Create random weights
    embed = QuantizedTensor.quantize(
        np.random.randn(config.vocab_size, hidden_size).astype(np.float32) * 0.02,
        qtype
    )

    layers = []
    for _ in range(num_layers):
        layer = TransformerLayer(
            q_proj=QuantizedTensor.quantize(np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02, qtype),
            k_proj=QuantizedTensor.quantize(np.random.randn(hidden_size, hidden_size // 4).astype(np.float32) * 0.02, qtype),
            v_proj=QuantizedTensor.quantize(np.random.randn(hidden_size, hidden_size // 4).astype(np.float32) * 0.02, qtype),
            o_proj=QuantizedTensor.quantize(np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02, qtype),
            gate_proj=QuantizedTensor.quantize(np.random.randn(hidden_size, hidden_size * 4).astype(np.float32) * 0.02, qtype),
            up_proj=QuantizedTensor.quantize(np.random.randn(hidden_size, hidden_size * 4).astype(np.float32) * 0.02, qtype),
            down_proj=QuantizedTensor.quantize(np.random.randn(hidden_size * 4, hidden_size).astype(np.float32) * 0.02, qtype),
            input_norm=np.ones(hidden_size, dtype=np.float32),
            post_attn_norm=np.ones(hidden_size, dtype=np.float32)
        )
        layers.append(layer)

    return LLMWeights(
        config=config,
        embed_tokens=embed,
        layers=layers,
        norm=np.ones(hidden_size, dtype=np.float32),
        lm_head=None
    )
