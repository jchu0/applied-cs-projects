"""On-Device LLM Runtime - Lightweight inference for edge deployment."""

from on_device_llm.quantization import (
    GGMLType,
    QuantizedTensor,
    quantize_q4_0,
    quantize_q8_0,
    dequantize_q4_0,
    dequantize_q8_0,
)
from on_device_llm.operators import (
    matmul_f32,
    rmsnorm,
    softmax,
    rope_embed,
    silu,
    gelu,
    attention_scores,
    weighted_sum,
)
from on_device_llm.memory import MemoryPool, KVCache
from on_device_llm.loader import GGUFLoader, TensorInfo, ModelConfig
from on_device_llm.inference import TransformerInference, Sampler, GenerationConfig, BatchInference
from on_device_llm.backend import (
    Backend,
    ComputeBackend,
    CPUBackend,
    CUDABackend,
    MetalBackend,
    VulkanBackend,
    NPUComputeBackend,
    get_backend,
    get_best_backend,
    get_npu_backend,
    list_available_backends,
)
from on_device_llm.npu import (
    NPUType,
    NPUCapability,
    NPUDeviceInfo,
    Int8Tensor,
    Int8TensorBlock,
    NPUBackend,
    NNAPIBackend,
    CoreMLBackend,
    HexagonBackend,
    SimulatedNPUBackend,
    NPUManager,
    int8_matmul_native,
    int8_matmul_with_bias,
    convert_ggml_to_int8,
    optimize_for_npu,
    estimate_npu_performance,
)
from on_device_llm.speculative import (
    SpeculativeConfig,
    SpeculativeStats,
    SpeculativeDecoder,
    SelfSpeculativeDecoder,
    LookaheadDecoder,
)

__version__ = "0.1.0"

__all__ = [
    # Quantization
    "GGMLType",
    "QuantizedTensor",
    "quantize_q4_0",
    "quantize_q8_0",
    "dequantize_q4_0",
    "dequantize_q8_0",
    # Operators
    "matmul_f32",
    "rmsnorm",
    "softmax",
    "rope_embed",
    "silu",
    "gelu",
    "attention_scores",
    "weighted_sum",
    # Memory
    "MemoryPool",
    "KVCache",
    # Loader
    "GGUFLoader",
    "TensorInfo",
    "ModelConfig",
    # Inference
    "TransformerInference",
    "Sampler",
    "GenerationConfig",
    "BatchInference",
    # Backend
    "Backend",
    "ComputeBackend",
    "CPUBackend",
    "CUDABackend",
    "MetalBackend",
    "VulkanBackend",
    "NPUComputeBackend",
    "get_backend",
    "get_best_backend",
    "get_npu_backend",
    "list_available_backends",
    # NPU
    "NPUType",
    "NPUCapability",
    "NPUDeviceInfo",
    "Int8Tensor",
    "Int8TensorBlock",
    "NPUBackend",
    "NNAPIBackend",
    "CoreMLBackend",
    "HexagonBackend",
    "SimulatedNPUBackend",
    "NPUManager",
    "int8_matmul_native",
    "int8_matmul_with_bias",
    "convert_ggml_to_int8",
    "optimize_for_npu",
    "estimate_npu_performance",
    # Speculative Decoding
    "SpeculativeConfig",
    "SpeculativeStats",
    "SpeculativeDecoder",
    "SelfSpeculativeDecoder",
    "LookaheadDecoder",
]
