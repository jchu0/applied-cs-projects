"""Tests for compute backends."""

import numpy as np
import pytest

from on_device_llm.backend import (
    Backend,
    ComputeBackend,
    CPUBackend,
    CUDABackend,
    MetalBackend,
    VulkanBackend,
    get_backend,
    get_best_backend,
    list_available_backends,
)


class TestBackendEnum:
    """Tests for Backend enum."""

    def test_backend_values(self):
        """Test backend enum values."""
        assert Backend.CPU.value == "cpu"
        assert Backend.CUDA.value == "cuda"
        assert Backend.METAL.value == "metal"
        assert Backend.VULKAN.value == "vulkan"


class TestCPUBackend:
    """Tests for CPUBackend."""

    @pytest.fixture
    def backend(self):
        """Create CPU backend fixture."""
        return CPUBackend()

    def test_backend_name(self, backend):
        """Test backend name."""
        assert "CPU" in backend.name

    def test_backend_type(self, backend):
        """Test backend type."""
        assert backend.backend_type == Backend.CPU

    def test_is_available(self, backend):
        """Test that CPU is always available."""
        assert backend.is_available() is True

    def test_matmul(self, backend):
        """Test matrix multiplication."""
        A = np.array([[1, 2], [3, 4]], dtype=np.float32)
        B = np.array([[5, 6], [7, 8]], dtype=np.float32)
        C = backend.matmul(A, B)

        expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
        np.testing.assert_array_almost_equal(C, expected)

    def test_softmax(self, backend):
        """Test softmax."""
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = backend.softmax(x)

        # Sum should be 1
        assert abs(result.sum() - 1.0) < 1e-5
        # Should be sorted (higher input = higher prob)
        assert result[2] > result[1] > result[0]

    def test_rmsnorm(self, backend):
        """Test RMS normalization."""
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        weight = np.ones(4, dtype=np.float32)
        result = backend.rmsnorm(x, weight)

        # Result should be normalized
        assert result.shape == x.shape
        # RMS of result should be close to 1 (scaled by weights)
        rms = np.sqrt(np.mean(result ** 2))
        assert abs(rms - 1.0) < 0.1

    def test_rope_embed(self, backend):
        """Test rotary position embeddings."""
        x = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        result = backend.rope_embed(x, pos=0)

        # At position 0, should be unchanged
        np.testing.assert_array_almost_equal(result, x, decimal=5)

        # At position 1, should be rotated
        result_pos1 = backend.rope_embed(x, pos=1)
        assert not np.allclose(result_pos1, x)

    def test_silu(self, backend):
        """Test SiLU activation."""
        x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        result = backend.silu(x)

        # silu(0) = 0
        assert abs(result[1]) < 1e-5
        # silu(1) = 1 * sigmoid(1) ≈ 0.731
        assert 0.7 < result[2] < 0.8
        # silu(-1) = -1 * sigmoid(-1) ≈ -0.269
        assert -0.3 < result[0] < -0.2


class TestCUDABackend:
    """Tests for CUDABackend."""

    @pytest.fixture
    def backend(self):
        """Create CUDA backend fixture."""
        return CUDABackend()

    def test_backend_type(self, backend):
        """Test backend type."""
        assert backend.backend_type == Backend.CUDA

    def test_availability_check(self, backend):
        """Test availability check doesn't crash."""
        # Should return bool without error
        assert isinstance(backend.is_available(), bool)

    @pytest.mark.skipif(
        not CUDABackend().is_available(),
        reason="CUDA not available"
    )
    def test_matmul_cuda(self, backend):
        """Test CUDA matmul."""
        A = np.random.randn(32, 64).astype(np.float32)
        B = np.random.randn(64, 32).astype(np.float32)
        result = backend.matmul(A, B)

        expected = A @ B
        np.testing.assert_array_almost_equal(result, expected, decimal=4)


class TestMetalBackend:
    """Tests for MetalBackend."""

    @pytest.fixture
    def backend(self):
        """Create Metal backend fixture."""
        return MetalBackend()

    def test_backend_type(self, backend):
        """Test backend type."""
        assert backend.backend_type == Backend.METAL

    def test_availability_check(self, backend):
        """Test availability check doesn't crash."""
        assert isinstance(backend.is_available(), bool)

    def test_name_indicates_status(self, backend):
        """Test that name indicates availability."""
        if backend.is_available():
            assert "MLX" in backend.name
        else:
            assert "not available" in backend.name.lower()

    @pytest.mark.skipif(
        not MetalBackend().is_available(),
        reason="Metal/MLX not available"
    )
    def test_matmul_metal(self, backend):
        """Test Metal matmul."""
        A = np.random.randn(32, 64).astype(np.float32)
        B = np.random.randn(64, 32).astype(np.float32)
        result = backend.matmul(A, B)

        expected = A @ B
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    @pytest.mark.skipif(
        not MetalBackend().is_available(),
        reason="Metal/MLX not available"
    )
    def test_softmax_metal(self, backend):
        """Test Metal softmax."""
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = backend.softmax(x)

        assert abs(result.sum() - 1.0) < 1e-5

    @pytest.mark.skipif(
        not MetalBackend().is_available(),
        reason="Metal/MLX not available"
    )
    def test_silu_metal(self, backend):
        """Test Metal SiLU."""
        x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        result = backend.silu(x)

        cpu_result = CPUBackend().silu(x)
        np.testing.assert_array_almost_equal(result, cpu_result, decimal=4)


class TestVulkanBackend:
    """Tests for VulkanBackend."""

    @pytest.fixture
    def backend(self):
        """Create Vulkan backend fixture."""
        return VulkanBackend()

    def test_backend_type(self, backend):
        """Test backend type."""
        assert backend.backend_type == Backend.VULKAN

    def test_availability_check(self, backend):
        """Test availability check doesn't crash."""
        assert isinstance(backend.is_available(), bool)

    @pytest.mark.skipif(
        not VulkanBackend().is_available(),
        reason="Vulkan not available"
    )
    def test_matmul_vulkan(self, backend):
        """Test Vulkan matmul."""
        A = np.random.randn(32, 64).astype(np.float32)
        B = np.random.randn(64, 32).astype(np.float32)
        result = backend.matmul(A, B)

        expected = A @ B
        np.testing.assert_array_almost_equal(result, expected, decimal=4)


class TestBackendFactory:
    """Tests for backend factory functions."""

    def test_get_cpu_backend(self):
        """Test getting CPU backend."""
        backend = get_backend(Backend.CPU)
        assert isinstance(backend, CPUBackend)

    def test_get_cuda_backend_unavailable(self):
        """Test getting CUDA backend when unavailable."""
        if not CUDABackend().is_available():
            with pytest.raises(RuntimeError):
                get_backend(Backend.CUDA)

    def test_get_metal_backend_unavailable(self):
        """Test getting Metal backend when unavailable."""
        if not MetalBackend().is_available():
            with pytest.raises(RuntimeError):
                get_backend(Backend.METAL)

    def test_get_vulkan_backend_unavailable(self):
        """Test getting Vulkan backend when unavailable."""
        if not VulkanBackend().is_available():
            with pytest.raises(RuntimeError):
                get_backend(Backend.VULKAN)

    def test_get_best_backend(self):
        """Test getting best available backend."""
        backend = get_best_backend()
        assert isinstance(backend, ComputeBackend)
        assert backend.is_available()

    def test_list_available_backends(self):
        """Test listing available backends."""
        available = list_available_backends()

        # CPU should always be available
        cpu_found = any(b[0] == Backend.CPU for b in available)
        assert cpu_found

        # All listed backends should be available
        for backend_type, name in available:
            backend = get_backend(backend_type)
            assert backend.is_available()


class TestBackendComparison:
    """Tests comparing different backends produce similar results."""

    def test_matmul_consistency(self):
        """Test that all backends produce consistent matmul results."""
        A = np.random.randn(16, 32).astype(np.float32)
        B = np.random.randn(32, 16).astype(np.float32)

        cpu_result = CPUBackend().matmul(A, B)

        # Compare with any available accelerated backend
        for backend in [CUDABackend(), MetalBackend(), VulkanBackend()]:
            if backend.is_available():
                result = backend.matmul(A, B)
                np.testing.assert_array_almost_equal(
                    result, cpu_result, decimal=4,
                    err_msg=f"{backend.name} matmul differs from CPU"
                )

    def test_softmax_consistency(self):
        """Test softmax consistency across backends."""
        x = np.random.randn(100).astype(np.float32)

        cpu_result = CPUBackend().softmax(x)

        for backend in [CUDABackend(), MetalBackend(), VulkanBackend()]:
            if backend.is_available():
                result = backend.softmax(x)
                np.testing.assert_array_almost_equal(
                    result, cpu_result, decimal=4,
                    err_msg=f"{backend.name} softmax differs from CPU"
                )

    def test_rmsnorm_consistency(self):
        """Test RMSNorm consistency across backends."""
        x = np.random.randn(64).astype(np.float32)
        weight = np.random.randn(64).astype(np.float32)

        cpu_result = CPUBackend().rmsnorm(x, weight)

        for backend in [CUDABackend(), MetalBackend(), VulkanBackend()]:
            if backend.is_available():
                result = backend.rmsnorm(x, weight)
                np.testing.assert_array_almost_equal(
                    result, cpu_result, decimal=4,
                    err_msg=f"{backend.name} rmsnorm differs from CPU"
                )

    def test_silu_consistency(self):
        """Test SiLU consistency across backends."""
        x = np.random.randn(64).astype(np.float32)

        cpu_result = CPUBackend().silu(x)

        for backend in [CUDABackend(), MetalBackend(), VulkanBackend()]:
            if backend.is_available():
                result = backend.silu(x)
                np.testing.assert_array_almost_equal(
                    result, cpu_result, decimal=4,
                    err_msg=f"{backend.name} silu differs from CPU"
                )
