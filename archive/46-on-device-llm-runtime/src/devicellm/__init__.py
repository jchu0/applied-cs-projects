"""On-device LLM runtime."""

from .core import (
    QuantizationType, DeviceType, TensorInfo, ModelConfig, ModelMetadata,
    QuantizedTensor, TransformerLayer, LLMWeights, ModelSerializer,
    create_test_model
)
from .runtime import (
    MemoryStats, MemoryPool, KVCache, ExecutionContext, RuntimeConfig,
    Operator, DeviceRuntime
)
from .inference import (
    GenerationConfig, TokenOutput, GenerationStats, Sampler,
    LLMEngine, StreamingEngine, BatchEngine, ContinuousEngine, create_engine
)
from .optimize import (
    OptimizationLevel, ConversionConfig, ConversionResult, ProfilingResult,
    ModelConverter, CalibratedConverter, ModelOptimizer, ModelProfiler,
    ModelExporter, convert_model, profile_and_optimize
)

__all__ = [
    # Core
    "QuantizationType", "DeviceType", "TensorInfo", "ModelConfig", "ModelMetadata",
    "QuantizedTensor", "TransformerLayer", "LLMWeights", "ModelSerializer",
    "create_test_model",
    # Runtime
    "MemoryStats", "MemoryPool", "KVCache", "ExecutionContext", "RuntimeConfig",
    "Operator", "DeviceRuntime",
    # Inference
    "GenerationConfig", "TokenOutput", "GenerationStats", "Sampler",
    "LLMEngine", "StreamingEngine", "BatchEngine", "ContinuousEngine", "create_engine",
    # Optimize
    "OptimizationLevel", "ConversionConfig", "ConversionResult", "ProfilingResult",
    "ModelConverter", "CalibratedConverter", "ModelOptimizer", "ModelProfiler",
    "ModelExporter", "convert_model", "profile_and_optimize",
]
