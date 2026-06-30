"""NPU (Neural Processing Unit) backend for on-device LLM inference.

This module provides NPU acceleration support for mobile and edge devices:
- Android NNAPI via ONNX Runtime
- iOS/macOS CoreML via ONNX Runtime
- Qualcomm Hexagon DSP
- Int8-native quantized operations (no dequantization overhead)

NPUs excel at int8 matrix multiplication and provide significant power efficiency
improvements over CPU/GPU for inference workloads.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, Dict, Any, List
import platform
import struct

import numpy as np

from on_device_llm.quantization import GGMLType, BLOCK_SIZES


class NPUType(Enum):
    """Types of NPU hardware."""
    NNAPI = "nnapi"           # Android Neural Networks API
    COREML = "coreml"         # Apple CoreML (Neural Engine)
    HEXAGON = "hexagon"       # Qualcomm Hexagon DSP
    MEDIATEK = "mediatek"     # MediaTek APU
    SIMULATED = "simulated"   # Simulated NPU for testing


class NPUCapability(Enum):
    """NPU capability flags."""
    INT8_MATMUL = "int8_matmul"           # Native int8 matrix multiplication
    INT4_MATMUL = "int4_matmul"           # Native int4 matrix multiplication
    FP16_COMPUTE = "fp16_compute"         # FP16 computation
    DYNAMIC_SHAPES = "dynamic_shapes"     # Dynamic tensor shapes
    BATCH_INFERENCE = "batch_inference"   # Batch processing support
    QUANTIZE_ON_LOAD = "quantize_on_load" # Quantize during model loading


@dataclass
class NPUDeviceInfo:
    """Information about an NPU device."""
    npu_type: NPUType
    name: str
    capabilities: List[NPUCapability]
    max_batch_size: int = 1
    max_sequence_length: int = 2048
    memory_mb: int = 0
    power_efficient: bool = True

    def supports(self, capability: NPUCapability) -> bool:
        """Check if device supports a capability."""
        return capability in self.capabilities


@dataclass
class Int8TensorBlock:
    """A block of int8 quantized values with scale factor.

    This format is optimized for NPU int8 matmul:
    - Values stored as int8 (no packing like Q4_0)
    - Single scale factor per block
    - Zero-point for asymmetric quantization
    """
    values: np.ndarray  # int8 values
    scale: float        # Scale factor
    zero_point: int = 0 # Zero point for asymmetric quantization

    def dequantize(self) -> np.ndarray:
        """Dequantize to float32."""
        return (self.values.astype(np.float32) - self.zero_point) * self.scale


@dataclass
class Int8Tensor:
    """NPU-optimized int8 tensor representation.

    Uses per-channel or per-tensor quantization for efficient NPU execution.
    """
    data: np.ndarray           # int8 data
    scales: np.ndarray         # Per-channel or per-tensor scales
    zero_points: np.ndarray    # Per-channel or per-tensor zero points
    shape: Tuple[int, ...]     # Original shape
    per_channel: bool = False  # Whether quantization is per-channel
    channel_axis: int = 0      # Which axis has per-channel quantization

    @classmethod
    def from_float32(cls, x: np.ndarray, per_channel: bool = False,
                     channel_axis: int = 0, symmetric: bool = True) -> "Int8Tensor":
        """Quantize float32 tensor to int8.

        Args:
            x: Input float32 tensor
            per_channel: Use per-channel quantization
            channel_axis: Axis for per-channel quantization
            symmetric: Use symmetric quantization (zero_point=0)

        Returns:
            Int8Tensor with quantized data
        """
        if per_channel:
            # Per-channel quantization
            n_channels = x.shape[channel_axis]
            scales = np.zeros(n_channels, dtype=np.float32)
            zero_points = np.zeros(n_channels, dtype=np.int8)

            data = np.zeros(x.shape, dtype=np.int8)

            for i in range(n_channels):
                # Select channel slice
                slices = [slice(None)] * len(x.shape)
                slices[channel_axis] = i
                channel_data = x[tuple(slices)]

                if symmetric:
                    # Symmetric quantization
                    amax = np.max(np.abs(channel_data))
                    scale = amax / 127.0 if amax > 0 else 1.0
                    quantized = np.clip(np.round(channel_data / scale), -128, 127)
                    scales[i] = scale
                    zero_points[i] = 0
                else:
                    # Asymmetric quantization
                    vmin, vmax = channel_data.min(), channel_data.max()
                    scale = (vmax - vmin) / 255.0 if vmax > vmin else 1.0
                    zero_point = int(np.round(-vmin / scale))
                    zero_point = np.clip(zero_point, 0, 255) - 128  # Center at 0
                    quantized = np.clip(np.round(channel_data / scale) + zero_point, -128, 127)
                    scales[i] = scale
                    zero_points[i] = zero_point

                data[tuple(slices)] = quantized.astype(np.int8)
        else:
            # Per-tensor quantization
            if symmetric:
                amax = np.max(np.abs(x))
                scale = amax / 127.0 if amax > 0 else 1.0
                data = np.clip(np.round(x / scale), -128, 127).astype(np.int8)
                scales = np.array([scale], dtype=np.float32)
                zero_points = np.array([0], dtype=np.int8)
            else:
                vmin, vmax = x.min(), x.max()
                scale = (vmax - vmin) / 255.0 if vmax > vmin else 1.0
                zero_point = int(np.round(-vmin / scale))
                zero_point = np.clip(zero_point, 0, 255) - 128
                data = np.clip(np.round(x / scale) + zero_point, -128, 127).astype(np.int8)
                scales = np.array([scale], dtype=np.float32)
                zero_points = np.array([zero_point], dtype=np.int8)

        return cls(
            data=data,
            scales=scales,
            zero_points=zero_points,
            shape=x.shape,
            per_channel=per_channel,
            channel_axis=channel_axis
        )

    def dequantize(self) -> np.ndarray:
        """Dequantize to float32."""
        if self.per_channel:
            result = np.zeros(self.shape, dtype=np.float32)
            n_channels = self.shape[self.channel_axis]

            for i in range(n_channels):
                slices = [slice(None)] * len(self.shape)
                slices[self.channel_axis] = i
                channel_data = self.data[tuple(slices)].astype(np.float32)
                result[tuple(slices)] = (channel_data - self.zero_points[i]) * self.scales[i]

            return result
        else:
            return (self.data.astype(np.float32) - self.zero_points[0]) * self.scales[0]


def int8_matmul_native(A: Int8Tensor, B: Int8Tensor) -> np.ndarray:
    """Perform int8 matrix multiplication with int32 accumulation.

    This is the native NPU operation - int8 inputs, int32 accumulation,
    then scale to float32 output.

    Args:
        A: First int8 tensor [M, K]
        B: Second int8 tensor [K, N]

    Returns:
        Float32 result [M, N]
    """
    # Int8 matmul with int32 accumulation (simulates NPU behavior)
    # NPUs typically do: int8 x int8 -> int32 accumulate -> scale to float
    result_int32 = np.matmul(
        A.data.astype(np.int32),
        B.data.astype(np.int32)
    )

    # Apply scales
    # For per-tensor quantization: output_scale = A_scale * B_scale
    if not A.per_channel and not B.per_channel:
        output_scale = A.scales[0] * B.scales[0]
        result = result_int32.astype(np.float32) * output_scale
    else:
        # For per-channel, need more complex dequantization
        # Simplified: just use the first scale
        output_scale = A.scales[0] * B.scales[0]
        result = result_int32.astype(np.float32) * output_scale

    return result


def int8_matmul_with_bias(A: Int8Tensor, B: Int8Tensor,
                          bias: Optional[np.ndarray] = None) -> np.ndarray:
    """Int8 matmul with optional float32 bias addition.

    Args:
        A: First int8 tensor [M, K]
        B: Second int8 tensor [K, N]
        bias: Optional float32 bias [N]

    Returns:
        Float32 result [M, N]
    """
    result = int8_matmul_native(A, B)
    if bias is not None:
        result = result + bias
    return result


class NPUBackend(ABC):
    """Abstract base class for NPU backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name."""
        pass

    @property
    @abstractmethod
    def npu_type(self) -> NPUType:
        """NPU type enum."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available."""
        pass

    @abstractmethod
    def get_device_info(self) -> NPUDeviceInfo:
        """Get NPU device information."""
        pass

    @abstractmethod
    def matmul_int8(self, A: Int8Tensor, B: Int8Tensor) -> np.ndarray:
        """Int8 matrix multiplication."""
        pass

    @abstractmethod
    def matmul_f32(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Float32 matrix multiplication (may use NPU or fall back)."""
        pass

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax (typically CPU fallback)."""
        x = x.astype(np.float32)
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def rmsnorm(self, x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """RMS normalization."""
        x = x.astype(np.float32)
        rms = np.sqrt(np.mean(x ** 2) + eps)
        return (x / rms) * weight.astype(np.float32)

    def synchronize(self) -> None:
        """Synchronize any async operations."""
        pass

    def memory_info(self) -> dict:
        """Get memory usage information."""
        return {"total": 0, "used": 0, "free": 0}


class NNAPIBackend(NPUBackend):
    """Android NNAPI backend using ONNX Runtime.

    NNAPI provides access to Android's Neural Networks API, which delegates
    to hardware accelerators (NPU, DSP, GPU) when available.
    """

    def __init__(self):
        """Initialize NNAPI backend."""
        self._available = False
        self._session = None
        self._ort = None

        try:
            import onnxruntime as ort
            self._ort = ort

            # Check if NNAPI execution provider is available
            providers = ort.get_available_providers()
            if "NnapiExecutionProvider" in providers:
                self._available = True
        except ImportError:
            pass

    @property
    def name(self) -> str:
        return "NNAPI (Android)"

    @property
    def npu_type(self) -> NPUType:
        return NPUType.NNAPI

    def is_available(self) -> bool:
        return self._available

    def get_device_info(self) -> NPUDeviceInfo:
        capabilities = [
            NPUCapability.INT8_MATMUL,
            NPUCapability.FP16_COMPUTE,
        ]
        return NPUDeviceInfo(
            npu_type=NPUType.NNAPI,
            name="Android NNAPI",
            capabilities=capabilities,
            max_batch_size=8,
            max_sequence_length=2048,
            power_efficient=True
        )

    def matmul_int8(self, A: Int8Tensor, B: Int8Tensor) -> np.ndarray:
        """Int8 matmul using NNAPI."""
        if not self._available:
            raise RuntimeError("NNAPI backend not available")

        # Use native int8 matmul simulation
        # In production, this would create an ONNX graph and run via NNAPI
        return int8_matmul_native(A, B)

    def matmul_f32(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Float32 matmul - quantize and use int8 path for NPU efficiency."""
        if not self._available:
            raise RuntimeError("NNAPI backend not available")

        # Quantize inputs
        A_int8 = Int8Tensor.from_float32(A.astype(np.float32))
        B_int8 = Int8Tensor.from_float32(B.astype(np.float32))

        return self.matmul_int8(A_int8, B_int8)


class CoreMLBackend(NPUBackend):
    """Apple CoreML backend for Neural Engine acceleration.

    CoreML provides access to Apple's Neural Engine on M1/M2/M3 chips
    and A-series chips in iPhones/iPads.
    """

    def __init__(self):
        """Initialize CoreML backend."""
        self._available = False
        self._ort = None

        # Check if we're on macOS/iOS
        if platform.system() not in ("Darwin",):
            return

        try:
            import onnxruntime as ort
            self._ort = ort

            # Check if CoreML execution provider is available
            providers = ort.get_available_providers()
            if "CoreMLExecutionProvider" in providers:
                self._available = True
        except ImportError:
            pass

    @property
    def name(self) -> str:
        return "CoreML (Apple Neural Engine)"

    @property
    def npu_type(self) -> NPUType:
        return NPUType.COREML

    def is_available(self) -> bool:
        return self._available

    def get_device_info(self) -> NPUDeviceInfo:
        capabilities = [
            NPUCapability.INT8_MATMUL,
            NPUCapability.FP16_COMPUTE,
            NPUCapability.DYNAMIC_SHAPES,
            NPUCapability.BATCH_INFERENCE,
        ]
        return NPUDeviceInfo(
            npu_type=NPUType.COREML,
            name="Apple Neural Engine",
            capabilities=capabilities,
            max_batch_size=32,
            max_sequence_length=4096,
            power_efficient=True
        )

    def matmul_int8(self, A: Int8Tensor, B: Int8Tensor) -> np.ndarray:
        """Int8 matmul using CoreML."""
        if not self._available:
            raise RuntimeError("CoreML backend not available")

        return int8_matmul_native(A, B)

    def matmul_f32(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Float32 matmul via CoreML."""
        if not self._available:
            raise RuntimeError("CoreML backend not available")

        # For CoreML, we can use FP16 which is well-supported
        A_int8 = Int8Tensor.from_float32(A.astype(np.float32))
        B_int8 = Int8Tensor.from_float32(B.astype(np.float32))

        return self.matmul_int8(A_int8, B_int8)


class HexagonBackend(NPUBackend):
    """Qualcomm Hexagon DSP backend.

    Hexagon DSP provides efficient int8 inference on Snapdragon SoCs.
    Uses Qualcomm Neural Processing SDK (SNPE) or QNN.
    """

    def __init__(self):
        """Initialize Hexagon backend."""
        self._available = False

        # Hexagon is only available on Qualcomm devices
        # Check for SNPE or QNN availability
        try:
            # In production, this would check for libsnpe.so or libqnn.so
            # For now, we simulate availability check
            pass
        except Exception:
            pass

    @property
    def name(self) -> str:
        return "Qualcomm Hexagon DSP"

    @property
    def npu_type(self) -> NPUType:
        return NPUType.HEXAGON

    def is_available(self) -> bool:
        return self._available

    def get_device_info(self) -> NPUDeviceInfo:
        capabilities = [
            NPUCapability.INT8_MATMUL,
            NPUCapability.INT4_MATMUL,
        ]
        return NPUDeviceInfo(
            npu_type=NPUType.HEXAGON,
            name="Qualcomm Hexagon DSP",
            capabilities=capabilities,
            max_batch_size=4,
            max_sequence_length=2048,
            power_efficient=True
        )

    def matmul_int8(self, A: Int8Tensor, B: Int8Tensor) -> np.ndarray:
        if not self._available:
            raise RuntimeError("Hexagon backend not available")
        return int8_matmul_native(A, B)

    def matmul_f32(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        if not self._available:
            raise RuntimeError("Hexagon backend not available")
        A_int8 = Int8Tensor.from_float32(A.astype(np.float32))
        B_int8 = Int8Tensor.from_float32(B.astype(np.float32))
        return self.matmul_int8(A_int8, B_int8)


class SimulatedNPUBackend(NPUBackend):
    """Simulated NPU backend for testing and development.

    Provides the same interface as real NPU backends but runs on CPU.
    Useful for testing NPU-specific code paths without hardware.
    """

    def __init__(self, simulate_latency: bool = False):
        """Initialize simulated NPU backend.

        Args:
            simulate_latency: Whether to add simulated processing delays
        """
        self._simulate_latency = simulate_latency
        self._ops_count = 0

    @property
    def name(self) -> str:
        return "Simulated NPU"

    @property
    def npu_type(self) -> NPUType:
        return NPUType.SIMULATED

    def is_available(self) -> bool:
        return True  # Always available

    def get_device_info(self) -> NPUDeviceInfo:
        return NPUDeviceInfo(
            npu_type=NPUType.SIMULATED,
            name="Simulated NPU (CPU)",
            capabilities=[
                NPUCapability.INT8_MATMUL,
                NPUCapability.INT4_MATMUL,
                NPUCapability.FP16_COMPUTE,
                NPUCapability.DYNAMIC_SHAPES,
                NPUCapability.BATCH_INFERENCE,
            ],
            max_batch_size=128,
            max_sequence_length=8192,
            power_efficient=False
        )

    def matmul_int8(self, A: Int8Tensor, B: Int8Tensor) -> np.ndarray:
        """Int8 matmul simulation."""
        self._ops_count += 1

        if self._simulate_latency:
            import time
            # Simulate ~1ms per 1M MACs
            macs = A.shape[0] * A.shape[1] * B.shape[1]
            time.sleep(macs / 1e9)

        return int8_matmul_native(A, B)

    def matmul_f32(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Float32 matmul via quantization."""
        A_int8 = Int8Tensor.from_float32(A.astype(np.float32))
        B_int8 = Int8Tensor.from_float32(B.astype(np.float32))
        return self.matmul_int8(A_int8, B_int8)

    @property
    def ops_count(self) -> int:
        """Number of operations executed."""
        return self._ops_count

    def reset_stats(self) -> None:
        """Reset operation statistics."""
        self._ops_count = 0


class NPUManager:
    """Manager for NPU backend selection and usage.

    Provides automatic NPU detection and fallback to CPU.
    """

    def __init__(self):
        """Initialize NPU manager."""
        self._backends: Dict[NPUType, NPUBackend] = {}
        self._detect_backends()

    def _detect_backends(self) -> None:
        """Detect available NPU backends."""
        # Check each backend type
        backends_to_check = [
            (NPUType.COREML, CoreMLBackend),
            (NPUType.NNAPI, NNAPIBackend),
            (NPUType.HEXAGON, HexagonBackend),
            (NPUType.SIMULATED, SimulatedNPUBackend),
        ]

        for npu_type, backend_class in backends_to_check:
            try:
                backend = backend_class()
                if backend.is_available():
                    self._backends[npu_type] = backend
            except Exception:
                pass

        # Always have simulated backend available
        if NPUType.SIMULATED not in self._backends:
            self._backends[NPUType.SIMULATED] = SimulatedNPUBackend()

    def get_best_backend(self) -> NPUBackend:
        """Get the best available NPU backend.

        Preference order: CoreML > NNAPI > Hexagon > Simulated
        """
        preference_order = [
            NPUType.COREML,
            NPUType.NNAPI,
            NPUType.HEXAGON,
            NPUType.SIMULATED,
        ]

        for npu_type in preference_order:
            if npu_type in self._backends and npu_type != NPUType.SIMULATED:
                return self._backends[npu_type]

        return self._backends[NPUType.SIMULATED]

    def get_backend(self, npu_type: NPUType) -> Optional[NPUBackend]:
        """Get a specific NPU backend."""
        return self._backends.get(npu_type)

    def list_available(self) -> List[Tuple[NPUType, str]]:
        """List available NPU backends."""
        return [
            (npu_type, backend.name)
            for npu_type, backend in self._backends.items()
        ]

    def has_hardware_npu(self) -> bool:
        """Check if a real hardware NPU is available (not simulated)."""
        return any(
            npu_type != NPUType.SIMULATED
            for npu_type in self._backends.keys()
        )


# Utility functions for NPU optimization

def convert_ggml_to_int8(data: bytes, shape: Tuple[int, ...],
                          dtype: GGMLType) -> Int8Tensor:
    """Convert GGML quantized tensor to NPU-friendly Int8Tensor.

    Args:
        data: GGML quantized data bytes
        shape: Tensor shape
        dtype: GGML quantization type

    Returns:
        Int8Tensor optimized for NPU
    """
    if dtype == GGMLType.Q8_0:
        # Q8_0 is already int8 - just need to restructure
        block_size = 32
        n_elements = 1
        for dim in shape:
            n_elements *= dim

        n_blocks = (n_elements + block_size - 1) // block_size

        # Extract scales and values from Q8_0 format
        values = np.zeros(n_blocks * block_size, dtype=np.int8)
        scales = []

        offset = 0
        for i in range(n_blocks):
            # Read scale (float16)
            scale = np.frombuffer(data[offset:offset + 2], dtype=np.float16)[0]
            scales.append(float(scale))
            offset += 2

            # Read int8 values
            block_values = np.frombuffer(data[offset:offset + 32], dtype=np.int8)
            values[i * block_size:(i + 1) * block_size] = block_values
            offset += 32

        # Average scale for per-tensor quantization
        avg_scale = np.mean(scales)

        return Int8Tensor(
            data=values[:n_elements].reshape(shape),
            scales=np.array([avg_scale], dtype=np.float32),
            zero_points=np.array([0], dtype=np.int8),
            shape=shape,
            per_channel=False
        )

    elif dtype == GGMLType.Q4_0:
        # Q4_0 needs dequantization then requantization to int8
        # First dequantize to float32
        from on_device_llm.quantization import dequantize_q4_0
        float_data = dequantize_q4_0(data, shape)

        # Then quantize to int8
        return Int8Tensor.from_float32(float_data)

    else:
        raise NotImplementedError(f"Conversion from {dtype} not implemented")


def optimize_for_npu(weights: Dict[str, np.ndarray],
                     per_channel: bool = True) -> Dict[str, Int8Tensor]:
    """Convert model weights to NPU-optimized format.

    Args:
        weights: Dictionary of weight tensors
        per_channel: Use per-channel quantization for better accuracy

    Returns:
        Dictionary of Int8Tensors
    """
    optimized = {}

    for name, weight in weights.items():
        # For 2D weights (linear layers), use per-channel on output dimension
        if weight.ndim == 2 and per_channel:
            optimized[name] = Int8Tensor.from_float32(
                weight,
                per_channel=True,
                channel_axis=0
            )
        else:
            optimized[name] = Int8Tensor.from_float32(weight)

    return optimized


def estimate_npu_performance(shape_a: Tuple[int, ...],
                             shape_b: Tuple[int, ...],
                             device_info: NPUDeviceInfo) -> Dict[str, float]:
    """Estimate performance metrics for NPU matmul.

    Args:
        shape_a: Shape of first matrix
        shape_b: Shape of second matrix
        device_info: NPU device information

    Returns:
        Dictionary with estimated metrics
    """
    M, K = shape_a[-2], shape_a[-1]
    K2, N = shape_b[-2], shape_b[-1]

    assert K == K2, "Matrix dimensions must match"

    # MACs (multiply-accumulate operations)
    macs = M * K * N

    # Rough estimates based on typical NPU performance
    # Real NPUs vary widely, these are ballpark figures
    if device_info.npu_type == NPUType.COREML:
        # Apple Neural Engine: ~15 TOPS int8
        tops = 15.0
    elif device_info.npu_type == NPUType.NNAPI:
        # Typical Android NPU: ~5-10 TOPS
        tops = 7.0
    elif device_info.npu_type == NPUType.HEXAGON:
        # Hexagon DSP: ~10-15 TOPS
        tops = 12.0
    else:
        # Simulated: based on CPU
        tops = 0.1

    # Estimated time in milliseconds
    estimated_ms = (macs / (tops * 1e12)) * 1000

    # Power estimate (mW)
    power_mw = 500 if device_info.power_efficient else 5000

    return {
        "macs": macs,
        "estimated_ms": estimated_ms,
        "tops": tops,
        "power_mw": power_mw,
        "energy_mj": estimated_ms * power_mw / 1000,  # millijoules
    }
