"""Tests for NPU (Neural Processing Unit) backend."""

import pytest
import numpy as np

from on_device_llm.npu import (
    NPUType,
    NPUCapability,
    NPUDeviceInfo,
    Int8Tensor,
    Int8TensorBlock,
    SimulatedNPUBackend,
    NNAPIBackend,
    CoreMLBackend,
    HexagonBackend,
    NPUManager,
    int8_matmul_native,
    int8_matmul_with_bias,
    convert_ggml_to_int8,
    optimize_for_npu,
    estimate_npu_performance,
)
from on_device_llm.quantization import GGMLType, quantize_q8_0, quantize_q4_0
from on_device_llm.backend import Backend, NPUComputeBackend, get_backend, list_available_backends


class TestInt8Tensor:
    """Tests for Int8Tensor quantization."""

    def test_from_float32_symmetric(self):
        """Test symmetric quantization."""
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        tensor = Int8Tensor.from_float32(x, symmetric=True)

        assert tensor.data.dtype == np.int8
        assert tensor.shape == x.shape
        assert tensor.zero_points[0] == 0  # Symmetric has zero_point=0
        assert tensor.scales[0] > 0

    def test_from_float32_asymmetric(self):
        """Test asymmetric quantization."""
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        tensor = Int8Tensor.from_float32(x, symmetric=False)

        assert tensor.data.dtype == np.int8
        assert tensor.shape == x.shape
        assert tensor.scales[0] > 0

    def test_from_float32_per_channel(self):
        """Test per-channel quantization."""
        x = np.random.randn(4, 8).astype(np.float32)
        tensor = Int8Tensor.from_float32(x, per_channel=True, channel_axis=0)

        assert tensor.per_channel is True
        assert tensor.channel_axis == 0
        assert len(tensor.scales) == 4

    def test_dequantize_roundtrip(self):
        """Test quantize then dequantize preserves values approximately."""
        x = np.random.randn(16, 32).astype(np.float32)
        tensor = Int8Tensor.from_float32(x)
        reconstructed = tensor.dequantize()

        # Int8 quantization should have reasonable error
        mae = np.mean(np.abs(x - reconstructed))
        assert mae < 0.1 * np.max(np.abs(x))  # Less than 10% of max value

    def test_dequantize_per_channel(self):
        """Test per-channel dequantization."""
        x = np.random.randn(4, 8).astype(np.float32) * 10
        tensor = Int8Tensor.from_float32(x, per_channel=True, channel_axis=0)
        reconstructed = tensor.dequantize()

        # Per-channel should have lower error
        mae = np.mean(np.abs(x - reconstructed))
        assert mae < 0.15 * np.max(np.abs(x))


class TestInt8TensorBlock:
    """Tests for Int8TensorBlock."""

    def test_dequantize(self):
        """Test block dequantization."""
        values = np.array([10, 20, 30, 40], dtype=np.int8)
        block = Int8TensorBlock(values=values, scale=0.5, zero_point=0)

        result = block.dequantize()
        expected = np.array([5.0, 10.0, 15.0, 20.0], dtype=np.float32)
        np.testing.assert_allclose(result, expected)

    def test_dequantize_with_zero_point(self):
        """Test block dequantization with non-zero zero_point."""
        values = np.array([10, 20, 30, 40], dtype=np.int8)
        block = Int8TensorBlock(values=values, scale=0.5, zero_point=5)

        result = block.dequantize()
        expected = np.array([2.5, 7.5, 12.5, 17.5], dtype=np.float32)
        np.testing.assert_allclose(result, expected)


class TestInt8Matmul:
    """Tests for int8 matrix multiplication."""

    def test_int8_matmul_native_basic(self):
        """Test basic int8 matmul."""
        A = np.array([[1, 2], [3, 4]], dtype=np.float32)
        B = np.array([[5, 6], [7, 8]], dtype=np.float32)

        A_int8 = Int8Tensor.from_float32(A)
        B_int8 = Int8Tensor.from_float32(B)

        result = int8_matmul_native(A_int8, B_int8)
        expected = A @ B

        # Should be close to float matmul result
        np.testing.assert_allclose(result, expected, rtol=0.2)

    def test_int8_matmul_larger(self):
        """Test int8 matmul on larger matrices."""
        A = np.random.randn(32, 64).astype(np.float32)
        B = np.random.randn(64, 32).astype(np.float32)

        A_int8 = Int8Tensor.from_float32(A)
        B_int8 = Int8Tensor.from_float32(B)

        result = int8_matmul_native(A_int8, B_int8)
        expected = A @ B

        # Relative error should be reasonable
        rel_error = np.mean(np.abs(result - expected)) / np.mean(np.abs(expected))
        assert rel_error < 0.3

    def test_int8_matmul_with_bias(self):
        """Test int8 matmul with bias."""
        A = np.random.randn(8, 16).astype(np.float32)
        B = np.random.randn(16, 8).astype(np.float32)
        bias = np.random.randn(8).astype(np.float32)

        A_int8 = Int8Tensor.from_float32(A)
        B_int8 = Int8Tensor.from_float32(B)

        result = int8_matmul_with_bias(A_int8, B_int8, bias)
        expected = (A @ B) + bias

        # Should include bias
        rel_error = np.mean(np.abs(result - expected)) / np.mean(np.abs(expected))
        assert rel_error < 0.35


class TestNPUDeviceInfo:
    """Tests for NPU device information."""

    def test_device_info_creation(self):
        """Test NPUDeviceInfo creation."""
        info = NPUDeviceInfo(
            npu_type=NPUType.SIMULATED,
            name="Test NPU",
            capabilities=[NPUCapability.INT8_MATMUL, NPUCapability.FP16_COMPUTE],
            max_batch_size=16,
        )

        assert info.npu_type == NPUType.SIMULATED
        assert info.name == "Test NPU"
        assert info.max_batch_size == 16

    def test_supports_capability(self):
        """Test capability checking."""
        info = NPUDeviceInfo(
            npu_type=NPUType.SIMULATED,
            name="Test",
            capabilities=[NPUCapability.INT8_MATMUL],
        )

        assert info.supports(NPUCapability.INT8_MATMUL)
        assert not info.supports(NPUCapability.INT4_MATMUL)


class TestSimulatedNPUBackend:
    """Tests for simulated NPU backend."""

    def test_is_available(self):
        """Test simulated backend is always available."""
        backend = SimulatedNPUBackend()
        assert backend.is_available() is True

    def test_name(self):
        """Test backend name."""
        backend = SimulatedNPUBackend()
        assert "Simulated" in backend.name

    def test_npu_type(self):
        """Test NPU type."""
        backend = SimulatedNPUBackend()
        assert backend.npu_type == NPUType.SIMULATED

    def test_device_info(self):
        """Test device info retrieval."""
        backend = SimulatedNPUBackend()
        info = backend.get_device_info()

        assert info.npu_type == NPUType.SIMULATED
        assert NPUCapability.INT8_MATMUL in info.capabilities

    def test_matmul_int8(self):
        """Test int8 matmul on simulated backend."""
        backend = SimulatedNPUBackend()

        A = np.random.randn(8, 16).astype(np.float32)
        B = np.random.randn(16, 8).astype(np.float32)

        A_int8 = Int8Tensor.from_float32(A)
        B_int8 = Int8Tensor.from_float32(B)

        result = backend.matmul_int8(A_int8, B_int8)
        expected = A @ B

        # Check shape
        assert result.shape == expected.shape

    def test_matmul_f32(self):
        """Test float32 matmul (internally quantizes)."""
        backend = SimulatedNPUBackend()

        A = np.random.randn(8, 16).astype(np.float32)
        B = np.random.randn(16, 8).astype(np.float32)

        result = backend.matmul_f32(A, B)
        expected = A @ B

        assert result.shape == expected.shape

    def test_ops_count(self):
        """Test operation counting."""
        backend = SimulatedNPUBackend()
        backend.reset_stats()

        A = np.random.randn(8, 16).astype(np.float32)
        B = np.random.randn(16, 8).astype(np.float32)
        A_int8 = Int8Tensor.from_float32(A)
        B_int8 = Int8Tensor.from_float32(B)

        assert backend.ops_count == 0
        backend.matmul_int8(A_int8, B_int8)
        assert backend.ops_count == 1
        backend.matmul_int8(A_int8, B_int8)
        assert backend.ops_count == 2

        backend.reset_stats()
        assert backend.ops_count == 0

    def test_softmax(self):
        """Test softmax operation."""
        backend = SimulatedNPUBackend()
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        result = backend.softmax(x)
        assert np.isclose(result.sum(), 1.0)
        assert all(result >= 0)

    def test_rmsnorm(self):
        """Test RMS normalization."""
        backend = SimulatedNPUBackend()
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        weight = np.ones(4, dtype=np.float32)

        result = backend.rmsnorm(x, weight)
        assert result.shape == x.shape


class TestNPUManager:
    """Tests for NPU manager."""

    def test_manager_creation(self):
        """Test NPU manager initialization."""
        manager = NPUManager()
        # Should always have simulated backend
        assert manager.get_backend(NPUType.SIMULATED) is not None

    def test_list_available(self):
        """Test listing available backends."""
        manager = NPUManager()
        available = manager.list_available()

        # Should have at least simulated
        assert len(available) >= 1
        npu_types = [npu_type for npu_type, name in available]
        assert NPUType.SIMULATED in npu_types

    def test_get_best_backend(self):
        """Test getting best backend."""
        manager = NPUManager()
        backend = manager.get_best_backend()

        assert backend is not None
        assert backend.is_available()

    def test_has_hardware_npu(self):
        """Test hardware NPU detection."""
        manager = NPUManager()
        # Result depends on system, just check it doesn't error
        result = manager.has_hardware_npu()
        assert isinstance(result, bool)


class TestNPUComputeBackend:
    """Tests for NPU compute backend wrapper."""

    def test_backend_creation(self):
        """Test NPU compute backend creation."""
        backend = NPUComputeBackend()
        assert backend.is_available()  # Simulated always available

    def test_backend_type(self):
        """Test backend type."""
        backend = NPUComputeBackend()
        assert backend.backend_type == Backend.NPU

    def test_matmul(self):
        """Test matmul through NPU backend wrapper."""
        backend = NPUComputeBackend()

        A = np.random.randn(8, 16).astype(np.float32)
        B = np.random.randn(16, 8).astype(np.float32)

        result = backend.matmul(A, B)
        expected = A @ B

        assert result.shape == expected.shape

    def test_softmax(self):
        """Test softmax through NPU backend wrapper."""
        backend = NPUComputeBackend()
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        result = backend.softmax(x)
        assert np.isclose(result.sum(), 1.0)

    def test_rmsnorm(self):
        """Test rmsnorm through NPU backend wrapper."""
        backend = NPUComputeBackend()
        x = np.random.randn(16).astype(np.float32)
        weight = np.ones(16, dtype=np.float32)

        result = backend.rmsnorm(x, weight)
        assert result.shape == x.shape

    def test_silu(self):
        """Test SiLU activation."""
        backend = NPUComputeBackend()
        x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)

        result = backend.silu(x)
        # SiLU(x) = x * sigmoid(x)
        expected = x * (1.0 / (1.0 + np.exp(-x)))
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestGGMLConversion:
    """Tests for GGML to Int8Tensor conversion."""

    def test_convert_q8_0(self):
        """Test converting Q8_0 to Int8Tensor."""
        original = np.random.randn(64).astype(np.float32)
        data, dtype = quantize_q8_0(original)

        int8_tensor = convert_ggml_to_int8(data, original.shape, dtype)

        assert int8_tensor.data.dtype == np.int8
        assert int8_tensor.shape == original.shape

    def test_convert_q4_0(self):
        """Test converting Q4_0 to Int8Tensor."""
        original = np.random.randn(64).astype(np.float32)
        data, dtype = quantize_q4_0(original)

        int8_tensor = convert_ggml_to_int8(data, original.shape, dtype)

        assert int8_tensor.data.dtype == np.int8
        assert int8_tensor.shape == original.shape


class TestOptimizeForNPU:
    """Tests for NPU optimization utilities."""

    def test_optimize_weights(self):
        """Test optimizing model weights for NPU."""
        weights = {
            "layer1.weight": np.random.randn(64, 128).astype(np.float32),
            "layer2.weight": np.random.randn(128, 64).astype(np.float32),
            "layer1.bias": np.random.randn(64).astype(np.float32),
        }

        optimized = optimize_for_npu(weights, per_channel=True)

        assert len(optimized) == 3
        for name, tensor in optimized.items():
            assert isinstance(tensor, Int8Tensor)
            assert tensor.data.dtype == np.int8

    def test_optimize_per_tensor(self):
        """Test per-tensor optimization."""
        weights = {
            "weight": np.random.randn(32, 64).astype(np.float32),
        }

        optimized = optimize_for_npu(weights, per_channel=False)

        tensor = optimized["weight"]
        assert tensor.per_channel is False


class TestPerformanceEstimation:
    """Tests for NPU performance estimation."""

    def test_estimate_basic(self):
        """Test basic performance estimation."""
        info = NPUDeviceInfo(
            npu_type=NPUType.SIMULATED,
            name="Test",
            capabilities=[NPUCapability.INT8_MATMUL],
        )

        metrics = estimate_npu_performance((8, 16), (16, 8), info)

        assert "macs" in metrics
        assert "estimated_ms" in metrics
        assert "tops" in metrics
        assert "power_mw" in metrics
        assert "energy_mj" in metrics

        assert metrics["macs"] == 8 * 16 * 8  # M * K * N

    def test_estimate_coreml(self):
        """Test CoreML performance estimation."""
        info = NPUDeviceInfo(
            npu_type=NPUType.COREML,
            name="Apple Neural Engine",
            capabilities=[NPUCapability.INT8_MATMUL],
        )

        metrics = estimate_npu_performance((32, 128), (128, 32), info)

        # CoreML should have higher TOPS estimate
        assert metrics["tops"] > 10


class TestBackendIntegration:
    """Tests for NPU integration with backend system."""

    def test_npu_in_backend_enum(self):
        """Test NPU is in Backend enum."""
        assert hasattr(Backend, "NPU")
        assert Backend.NPU.value == "npu"

    def test_get_backend_npu(self):
        """Test getting NPU backend."""
        backend = get_backend(Backend.NPU)
        assert backend.backend_type == Backend.NPU

    def test_list_backends_includes_npu(self):
        """Test NPU appears in available backends."""
        available = list_available_backends()
        backend_types = [bt for bt, name in available]

        # NPU should be available (at least simulated)
        assert Backend.NPU in backend_types


class TestNNAPIBackend:
    """Tests for NNAPI backend."""

    def test_backend_creation(self):
        """Test NNAPI backend creation."""
        backend = NNAPIBackend()
        # May or may not be available depending on system
        assert backend.npu_type == NPUType.NNAPI

    def test_device_info(self):
        """Test device info."""
        backend = NNAPIBackend()
        if backend.is_available():
            info = backend.get_device_info()
            assert info.npu_type == NPUType.NNAPI


class TestCoreMLBackend:
    """Tests for CoreML backend."""

    def test_backend_creation(self):
        """Test CoreML backend creation."""
        backend = CoreMLBackend()
        assert backend.npu_type == NPUType.COREML

    def test_device_info(self):
        """Test device info."""
        backend = CoreMLBackend()
        if backend.is_available():
            info = backend.get_device_info()
            assert info.npu_type == NPUType.COREML
            assert NPUCapability.FP16_COMPUTE in info.capabilities


class TestHexagonBackend:
    """Tests for Hexagon backend."""

    def test_backend_creation(self):
        """Test Hexagon backend creation."""
        backend = HexagonBackend()
        assert backend.npu_type == NPUType.HEXAGON

    def test_device_info(self):
        """Test device info."""
        backend = HexagonBackend()
        if backend.is_available():
            info = backend.get_device_info()
            assert info.npu_type == NPUType.HEXAGON
