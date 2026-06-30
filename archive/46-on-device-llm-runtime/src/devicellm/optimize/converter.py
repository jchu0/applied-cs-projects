"""Model conversion and optimization for on-device deployment."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import numpy as np
import time
import json

from ..core.model import (
    LLMWeights, ModelConfig, QuantizedTensor, QuantizationType,
    TransformerLayer, ModelSerializer, DeviceType
)


class OptimizationLevel(Enum):
    """Optimization levels."""
    NONE = 0
    BASIC = 1  # Basic quantization
    AGGRESSIVE = 2  # Aggressive pruning + quantization
    EXTREME = 3  # Maximum compression


@dataclass
class ConversionConfig:
    """Configuration for model conversion."""
    target_quantization: QuantizationType = QuantizationType.INT4
    optimization_level: OptimizationLevel = OptimizationLevel.BASIC
    calibration_samples: int = 100
    prune_threshold: float = 0.01
    target_device: DeviceType = DeviceType.CPU
    target_memory_mb: float | None = None


@dataclass
class ConversionResult:
    """Result of model conversion."""
    original_size_mb: float
    converted_size_mb: float
    compression_ratio: float
    accuracy_delta: float  # Estimated accuracy change
    conversion_time_s: float


@dataclass
class ProfilingResult:
    """Result of model profiling."""
    total_time_ms: float
    layer_times_ms: list[float]
    operator_times: dict[str, float]
    memory_peak_mb: float
    bottleneck_layers: list[int]


class ModelConverter:
    """Convert models for on-device deployment."""

    def __init__(self, config: ConversionConfig):
        self.config = config

    def convert(self, weights: LLMWeights) -> tuple[LLMWeights, ConversionResult]:
        """Convert model to target quantization."""
        start_time = time.perf_counter()
        original_size = weights.memory_size() / (1024 * 1024)

        # Convert embeddings
        new_embed = self._quantize_tensor(
            weights.embed_tokens.dequantize(),
            self.config.target_quantization
        )

        # Convert layers
        new_layers = []
        for layer in weights.layers:
            new_layer = self._convert_layer(layer)
            new_layers.append(new_layer)

        # Convert LM head
        new_lm_head = None
        if weights.lm_head:
            new_lm_head = self._quantize_tensor(
                weights.lm_head.dequantize(),
                self.config.target_quantization
            )

        converted = LLMWeights(
            config=weights.config,
            embed_tokens=new_embed,
            layers=new_layers,
            norm=weights.norm,
            lm_head=new_lm_head
        )

        converted_size = converted.memory_size() / (1024 * 1024)
        conversion_time = time.perf_counter() - start_time

        result = ConversionResult(
            original_size_mb=original_size,
            converted_size_mb=converted_size,
            compression_ratio=original_size / converted_size if converted_size > 0 else 1.0,
            accuracy_delta=self._estimate_accuracy_loss(),
            conversion_time_s=conversion_time
        )

        return converted, result

    def _convert_layer(self, layer: TransformerLayer) -> TransformerLayer:
        """Convert a single transformer layer."""
        qtype = self.config.target_quantization

        return TransformerLayer(
            q_proj=self._quantize_tensor(layer.q_proj.dequantize(), qtype),
            k_proj=self._quantize_tensor(layer.k_proj.dequantize(), qtype),
            v_proj=self._quantize_tensor(layer.v_proj.dequantize(), qtype),
            o_proj=self._quantize_tensor(layer.o_proj.dequantize(), qtype),
            gate_proj=self._quantize_tensor(layer.gate_proj.dequantize(), qtype),
            up_proj=self._quantize_tensor(layer.up_proj.dequantize(), qtype),
            down_proj=self._quantize_tensor(layer.down_proj.dequantize(), qtype),
            input_norm=layer.input_norm,
            post_attn_norm=layer.post_attn_norm
        )

    def _quantize_tensor(
        self,
        tensor: np.ndarray,
        qtype: QuantizationType
    ) -> QuantizedTensor:
        """Quantize a tensor to target type."""
        if self.config.optimization_level >= OptimizationLevel.AGGRESSIVE:
            # Apply pruning before quantization
            tensor = self._prune_weights(tensor)

        return QuantizedTensor.quantize(tensor, qtype)

    def _prune_weights(self, tensor: np.ndarray) -> np.ndarray:
        """Prune small weights to zero."""
        threshold = self.config.prune_threshold * np.abs(tensor).max()
        tensor = np.where(np.abs(tensor) < threshold, 0, tensor)
        return tensor

    def _estimate_accuracy_loss(self) -> float:
        """Estimate accuracy loss from conversion."""
        # Simple heuristic based on quantization type
        loss_map = {
            QuantizationType.FP32: 0.0,
            QuantizationType.FP16: 0.001,
            QuantizationType.INT8: 0.01,
            QuantizationType.INT4: 0.03,
            QuantizationType.GGML_Q4_0: 0.025,
        }
        base_loss = loss_map.get(self.config.target_quantization, 0.05)

        # Adjust for optimization level
        if self.config.optimization_level >= OptimizationLevel.AGGRESSIVE:
            base_loss *= 1.5

        return base_loss


class CalibratedConverter(ModelConverter):
    """Converter with calibration for better accuracy."""

    def __init__(
        self,
        config: ConversionConfig,
        calibration_data: list[np.ndarray] | None = None
    ):
        super().__init__(config)
        self.calibration_data = calibration_data or []
        self.scale_factors: dict[str, np.ndarray] = {}

    def calibrate(self, weights: LLMWeights) -> None:
        """Run calibration to determine optimal scales."""
        if not self.calibration_data:
            return

        # Collect activation statistics for each layer
        for i, layer in enumerate(weights.layers):
            # Simplified: just use weight statistics
            for name, tensor in [
                ("q_proj", layer.q_proj),
                ("k_proj", layer.k_proj),
                ("v_proj", layer.v_proj),
                ("o_proj", layer.o_proj),
                ("gate_proj", layer.gate_proj),
                ("up_proj", layer.up_proj),
                ("down_proj", layer.down_proj),
            ]:
                key = f"layer.{i}.{name}"
                data = tensor.dequantize()
                # Use percentile for robustness
                scale = np.percentile(np.abs(data), 99.9)
                self.scale_factors[key] = np.array([scale])

    def convert(self, weights: LLMWeights) -> tuple[LLMWeights, ConversionResult]:
        """Convert with calibration."""
        self.calibrate(weights)
        return super().convert(weights)


class ModelOptimizer:
    """Optimize model for specific device targets."""

    def __init__(self, target_device: DeviceType):
        self.target_device = target_device

    def optimize(self, weights: LLMWeights) -> LLMWeights:
        """Apply device-specific optimizations."""
        if self.target_device == DeviceType.CPU:
            return self._optimize_for_cpu(weights)
        elif self.target_device == DeviceType.GPU_METAL:
            return self._optimize_for_metal(weights)
        elif self.target_device == DeviceType.NPU:
            return self._optimize_for_npu(weights)
        return weights

    def _optimize_for_cpu(self, weights: LLMWeights) -> LLMWeights:
        """Optimize for CPU execution."""
        # Layout optimization for cache efficiency
        # Simplified: just return as-is
        return weights

    def _optimize_for_metal(self, weights: LLMWeights) -> LLMWeights:
        """Optimize for Apple Metal GPU."""
        # Metal prefers certain memory alignments
        return weights

    def _optimize_for_npu(self, weights: LLMWeights) -> LLMWeights:
        """Optimize for Neural Processing Unit."""
        # NPU often requires int8 quantization
        return weights


class ModelProfiler:
    """Profile model performance."""

    def __init__(self):
        self.results: list[ProfilingResult] = []

    def profile(
        self,
        weights: LLMWeights,
        input_length: int = 128,
        num_iterations: int = 10
    ) -> ProfilingResult:
        """Profile model execution."""
        from ..runtime.runtime import DeviceRuntime, RuntimeConfig
        from ..inference.generate import LLMEngine

        config = RuntimeConfig(
            memory_limit_mb=weights.memory_size() / (1024 * 1024) + 256
        )
        runtime = DeviceRuntime(config)
        engine = LLMEngine(weights, config)
        ctx = engine._ensure_context(input_length + 100)

        # Warmup
        dummy_input = np.random.randint(0, 100, size=input_length)
        engine.forward(dummy_input, ctx)
        ctx.reset()

        # Profile
        layer_times = [0.0] * len(weights.layers)
        total_times = []

        for _ in range(num_iterations):
            ctx.reset()
            start = time.perf_counter()

            hidden = engine.embed(dummy_input)[np.newaxis, ...]

            for i, layer in enumerate(weights.layers):
                layer_start = time.perf_counter()
                hidden = runtime.execute_layer(ctx, layer, hidden, i)
                layer_times[i] += (time.perf_counter() - layer_start) * 1000

            total_times.append((time.perf_counter() - start) * 1000)

        # Average times
        avg_total = sum(total_times) / len(total_times)
        avg_layer_times = [t / num_iterations for t in layer_times]

        # Get operator times
        op_times = {}
        for name, op in runtime.operators.items():
            if op.call_count > 0:
                op_times[name] = op.total_time * 1000 / num_iterations

        # Find bottleneck layers
        threshold = np.mean(avg_layer_times) * 1.5
        bottlenecks = [i for i, t in enumerate(avg_layer_times) if t > threshold]

        memory_stats = ctx.get_memory_stats()

        result = ProfilingResult(
            total_time_ms=avg_total,
            layer_times_ms=avg_layer_times,
            operator_times=op_times,
            memory_peak_mb=memory_stats.peak_mb,
            bottleneck_layers=bottlenecks
        )

        self.results.append(result)
        return result

    def compare_profiles(self) -> dict[str, Any]:
        """Compare multiple profiling runs."""
        if len(self.results) < 2:
            return {}

        return {
            "speedup": self.results[0].total_time_ms / self.results[-1].total_time_ms,
            "memory_reduction": 1 - (self.results[-1].memory_peak_mb / self.results[0].memory_peak_mb)
        }


class ModelExporter:
    """Export models to various formats."""

    def __init__(self):
        self.serializer = ModelSerializer()

    def export_native(self, weights: LLMWeights, filepath: str) -> None:
        """Export to native format."""
        self.serializer.save(weights, filepath)

    def export_ggml(self, weights: LLMWeights, filepath: str) -> None:
        """Export to GGML format."""
        # Simplified GGML export
        with open(filepath, 'wb') as f:
            # Write GGML magic
            f.write(b'GGML')

            # Write hyperparams
            config = weights.config
            f.write(np.array([
                config.vocab_size,
                config.hidden_size,
                config.num_layers,
                config.num_heads,
                config.num_kv_heads
            ], dtype=np.int32).tobytes())

            # Write tensors
            # Simplified: would need full GGML tensor format

    def export_metadata(self, weights: LLMWeights, filepath: str) -> None:
        """Export model metadata as JSON."""
        metadata = {
            "config": {
                "vocab_size": weights.config.vocab_size,
                "hidden_size": weights.config.hidden_size,
                "num_layers": weights.config.num_layers,
                "num_heads": weights.config.num_heads,
                "num_kv_heads": weights.config.num_kv_heads,
                "intermediate_size": weights.config.intermediate_size,
                "max_position": weights.config.max_position,
            },
            "param_count": weights.param_count(),
            "memory_mb": weights.memory_size() / (1024 * 1024),
        }

        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)


def convert_model(
    weights: LLMWeights,
    target_qtype: QuantizationType = QuantizationType.INT4,
    optimize_for: DeviceType = DeviceType.CPU
) -> tuple[LLMWeights, ConversionResult]:
    """Convenience function to convert model."""
    config = ConversionConfig(
        target_quantization=target_qtype,
        target_device=optimize_for
    )
    converter = ModelConverter(config)
    converted, result = converter.convert(weights)

    optimizer = ModelOptimizer(optimize_for)
    optimized = optimizer.optimize(converted)

    return optimized, result


def profile_and_optimize(
    weights: LLMWeights,
    target_time_ms: float | None = None
) -> tuple[LLMWeights, list[ProfilingResult]]:
    """Profile model and apply optimizations."""
    profiler = ModelProfiler()

    # Initial profile
    initial = profiler.profile(weights)

    if target_time_ms is None or initial.total_time_ms <= target_time_ms:
        return weights, [initial]

    # Try different optimization levels
    results = [initial]

    for level in [OptimizationLevel.BASIC, OptimizationLevel.AGGRESSIVE]:
        config = ConversionConfig(
            optimization_level=level,
            target_quantization=QuantizationType.INT4
        )
        converter = ModelConverter(config)
        optimized, _ = converter.convert(weights)

        result = profiler.profile(optimized)
        results.append(result)

        if result.total_time_ms <= target_time_ms:
            return optimized, results

    return weights, results
