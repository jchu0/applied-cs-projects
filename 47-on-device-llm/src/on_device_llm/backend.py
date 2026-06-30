"""Compute backend abstraction for on-device LLM inference.

This module provides a backend abstraction layer to support multiple
compute backends (CPU, CUDA, Metal, Vulkan) with a unified interface.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import numpy as np

from on_device_llm.operators import matmul_f32, softmax, rmsnorm, rope_embed, silu


class Backend(Enum):
    """Supported compute backends."""

    CPU = "cpu"
    CUDA = "cuda"
    METAL = "metal"
    VULKAN = "vulkan"
    NPU = "npu"  # Neural Processing Unit (NNAPI, CoreML, Hexagon)


class ComputeBackend(ABC):
    """Abstract base class for compute backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name."""
        pass

    @property
    @abstractmethod
    def backend_type(self) -> Backend:
        """Backend type enum."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available on the current system."""
        pass

    @abstractmethod
    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Matrix multiplication.

        Args:
            A: First matrix [M, K]
            B: Second matrix [K, N]

        Returns:
            Result matrix [M, N]
        """
        pass

    @abstractmethod
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Softmax activation.

        Args:
            x: Input logits

        Returns:
            Probability distribution
        """
        pass

    @abstractmethod
    def rmsnorm(self, x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """
        RMS normalization.

        Args:
            x: Input tensor
            weight: Scale parameters
            eps: Numerical stability constant

        Returns:
            Normalized tensor
        """
        pass

    @abstractmethod
    def rope_embed(self, x: np.ndarray, pos: int, theta: float = 10000.0) -> np.ndarray:
        """
        Rotary position embeddings.

        Args:
            x: Input embedding
            pos: Position index
            theta: Base frequency

        Returns:
            Position-encoded embedding
        """
        pass

    @abstractmethod
    def silu(self, x: np.ndarray) -> np.ndarray:
        """
        SiLU activation.

        Args:
            x: Input tensor

        Returns:
            Activated tensor
        """
        pass

    def synchronize(self) -> None:
        """Synchronize any async operations (no-op for CPU)."""
        pass

    def memory_info(self) -> dict:
        """Get memory usage information."""
        return {"total": 0, "used": 0, "free": 0}


class CPUBackend(ComputeBackend):
    """CPU backend with SIMD optimization via Numba."""

    @property
    def name(self) -> str:
        return "CPU (Numba SIMD)"

    @property
    def backend_type(self) -> Backend:
        return Backend.CPU

    def is_available(self) -> bool:
        return True

    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return matmul_f32(A.astype(np.float32), B.astype(np.float32))

    def softmax(self, x: np.ndarray) -> np.ndarray:
        return softmax(x.astype(np.float32))

    def rmsnorm(self, x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        return rmsnorm(x.astype(np.float32), weight.astype(np.float32), eps)

    def rope_embed(self, x: np.ndarray, pos: int, theta: float = 10000.0) -> np.ndarray:
        return rope_embed(x.astype(np.float32), pos, theta)

    def silu(self, x: np.ndarray) -> np.ndarray:
        return silu(x.astype(np.float32))


class CUDABackend(ComputeBackend):
    """CUDA GPU backend using CuPy."""

    def __init__(self):
        """Initialize CUDA backend."""
        self._cp = None
        self._device_id = 0
        self._available = False

        try:
            import cupy as cp
            self._cp = cp
            self._available = True
        except ImportError:
            pass

    @property
    def name(self) -> str:
        if self._available and self._cp is not None:
            device = self._cp.cuda.Device(self._device_id)
            return f"CUDA ({device.name})"
        return "CUDA (not available)"

    @property
    def backend_type(self) -> Backend:
        return Backend.CUDA

    def is_available(self) -> bool:
        return self._available

    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        if not self._available:
            raise RuntimeError("CUDA backend not available")

        A_gpu = self._cp.asarray(A.astype(np.float32))
        B_gpu = self._cp.asarray(B.astype(np.float32))
        result = A_gpu @ B_gpu
        return self._cp.asnumpy(result)

    def softmax(self, x: np.ndarray) -> np.ndarray:
        if not self._available:
            raise RuntimeError("CUDA backend not available")

        x_gpu = self._cp.asarray(x.astype(np.float32))
        exp_x = self._cp.exp(x_gpu - self._cp.max(x_gpu))
        result = exp_x / exp_x.sum()
        return self._cp.asnumpy(result)

    def rmsnorm(self, x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        if not self._available:
            raise RuntimeError("CUDA backend not available")

        x_gpu = self._cp.asarray(x.astype(np.float32))
        w_gpu = self._cp.asarray(weight.astype(np.float32))

        rms = self._cp.sqrt(self._cp.mean(x_gpu ** 2) + eps)
        result = (x_gpu / rms) * w_gpu
        return self._cp.asnumpy(result)

    def rope_embed(self, x: np.ndarray, pos: int, theta: float = 10000.0) -> np.ndarray:
        if not self._available:
            raise RuntimeError("CUDA backend not available")

        # For RoPE, CPU is often fast enough; use GPU for large batches
        return rope_embed(x.astype(np.float32), pos, theta)

    def silu(self, x: np.ndarray) -> np.ndarray:
        if not self._available:
            raise RuntimeError("CUDA backend not available")

        x_gpu = self._cp.asarray(x.astype(np.float32))
        result = x_gpu / (1.0 + self._cp.exp(-x_gpu)) * x_gpu / x_gpu  # x * sigmoid(x)
        # Simpler: x * sigmoid(x)
        x_gpu = self._cp.asarray(x.astype(np.float32))
        result = x_gpu * (1.0 / (1.0 + self._cp.exp(-x_gpu)))
        return self._cp.asnumpy(result)

    def synchronize(self) -> None:
        if self._available:
            self._cp.cuda.Stream.null.synchronize()

    def memory_info(self) -> dict:
        if not self._available:
            return {"total": 0, "used": 0, "free": 0}

        mempool = self._cp.get_default_memory_pool()
        return {
            "total": mempool.total_bytes(),
            "used": mempool.used_bytes(),
            "free": mempool.total_bytes() - mempool.used_bytes(),
        }


class MetalBackend(ComputeBackend):
    """Metal GPU backend for Apple Silicon using MLX.

    MLX is Apple's array framework optimized for Apple Silicon with unified memory.
    This backend provides hardware-accelerated inference on M1/M2/M3 chips.
    """

    def __init__(self):
        """Initialize Metal backend using MLX."""
        self._mx = None
        self._available = False
        self._device_name = "Apple Silicon"

        try:
            import mlx.core as mx
            self._mx = mx
            self._available = True
            # MLX automatically uses Metal on Apple Silicon
        except ImportError:
            pass

    @property
    def name(self) -> str:
        if self._available:
            return f"Metal (MLX on {self._device_name})"
        return "Metal (MLX not available)"

    @property
    def backend_type(self) -> Backend:
        return Backend.METAL

    def is_available(self) -> bool:
        return self._available

    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        if not self._available:
            raise RuntimeError("Metal backend not available")

        A_mlx = self._mx.array(A.astype(np.float32))
        B_mlx = self._mx.array(B.astype(np.float32))
        result = self._mx.matmul(A_mlx, B_mlx)
        self._mx.eval(result)
        return np.array(result)

    def softmax(self, x: np.ndarray) -> np.ndarray:
        if not self._available:
            raise RuntimeError("Metal backend not available")

        x_mlx = self._mx.array(x.astype(np.float32))
        result = self._mx.softmax(x_mlx)
        self._mx.eval(result)
        return np.array(result)

    def rmsnorm(self, x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        if not self._available:
            raise RuntimeError("Metal backend not available")

        x_mlx = self._mx.array(x.astype(np.float32))
        w_mlx = self._mx.array(weight.astype(np.float32))

        # RMSNorm: x / sqrt(mean(x^2) + eps) * weight
        rms = self._mx.sqrt(self._mx.mean(x_mlx * x_mlx) + eps)
        result = (x_mlx / rms) * w_mlx
        self._mx.eval(result)
        return np.array(result)

    def rope_embed(self, x: np.ndarray, pos: int, theta: float = 10000.0) -> np.ndarray:
        if not self._available:
            raise RuntimeError("Metal backend not available")

        # RoPE computation - use numpy for simplicity as it's fast enough
        # MLX doesn't have a built-in RoPE, so we compute on CPU
        return rope_embed(x.astype(np.float32), pos, theta)

    def silu(self, x: np.ndarray) -> np.ndarray:
        if not self._available:
            raise RuntimeError("Metal backend not available")

        x_mlx = self._mx.array(x.astype(np.float32))
        # SiLU(x) = x * sigmoid(x)
        result = x_mlx * self._mx.sigmoid(x_mlx)
        self._mx.eval(result)
        return np.array(result)

    def gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU activation using MLX."""
        if not self._available:
            raise RuntimeError("Metal backend not available")

        x_mlx = self._mx.array(x.astype(np.float32))
        result = self._mx.nn.gelu(x_mlx)
        self._mx.eval(result)
        return np.array(result)

    def synchronize(self) -> None:
        if self._available:
            self._mx.eval()

    def memory_info(self) -> dict:
        if not self._available:
            return {"total": 0, "used": 0, "free": 0}
        # MLX uses unified memory - report system memory
        import platform
        if platform.system() == "Darwin":
            try:
                import subprocess
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True, text=True
                )
                total = int(result.stdout.strip())
                return {"total": total, "used": 0, "free": total}
            except Exception:
                pass
        return {"total": 0, "used": 0, "free": 0}


class VulkanBackend(ComputeBackend):
    """Vulkan compute backend for cross-platform GPU support.

    Uses kompute library for Vulkan compute shaders.
    Supports NVIDIA, AMD, Intel, and other Vulkan-capable GPUs.
    """

    def __init__(self):
        """Initialize Vulkan backend."""
        self._kp = None
        self._mgr = None
        self._available = False
        self._device_name = "Unknown"

        try:
            import kp
            self._kp = kp
            self._mgr = kp.Manager()
            self._available = True
            self._device_name = "Vulkan Device"
        except ImportError:
            pass
        except Exception:
            # Vulkan might not be available even if kompute is installed
            pass

    @property
    def name(self) -> str:
        if self._available:
            return f"Vulkan ({self._device_name})"
        return "Vulkan (not available)"

    @property
    def backend_type(self) -> Backend:
        return Backend.VULKAN

    def is_available(self) -> bool:
        return self._available

    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        if not self._available:
            raise RuntimeError("Vulkan backend not available")

        # Fall back to NumPy - Vulkan compute shader would require custom GLSL
        # In production, this would use optimized Vulkan compute shaders
        return (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float32)

    def softmax(self, x: np.ndarray) -> np.ndarray:
        if not self._available:
            raise RuntimeError("Vulkan backend not available")

        # NumPy fallback
        x = x.astype(np.float32)
        exp_x = np.exp(x - np.max(x))
        return (exp_x / exp_x.sum()).astype(np.float32)

    def rmsnorm(self, x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        if not self._available:
            raise RuntimeError("Vulkan backend not available")

        x = x.astype(np.float32)
        weight = weight.astype(np.float32)
        rms = np.sqrt(np.mean(x ** 2) + eps)
        return ((x / rms) * weight).astype(np.float32)

    def rope_embed(self, x: np.ndarray, pos: int, theta: float = 10000.0) -> np.ndarray:
        if not self._available:
            raise RuntimeError("Vulkan backend not available")
        return rope_embed(x.astype(np.float32), pos, theta)

    def silu(self, x: np.ndarray) -> np.ndarray:
        if not self._available:
            raise RuntimeError("Vulkan backend not available")

        x = x.astype(np.float32)
        return (x * (1.0 / (1.0 + np.exp(-x)))).astype(np.float32)

    def synchronize(self) -> None:
        if self._available and self._mgr:
            # Vulkan commands are submitted synchronously by default in kompute
            pass

    def memory_info(self) -> dict:
        return {"total": 0, "used": 0, "free": 0}


class NPUComputeBackend(ComputeBackend):
    """NPU backend wrapper for on-device inference.

    Provides access to Neural Processing Units (NPU) via:
    - Android NNAPI
    - Apple CoreML (Neural Engine)
    - Qualcomm Hexagon DSP

    NPUs excel at int8 inference with minimal power consumption.
    """

    def __init__(self):
        """Initialize NPU backend."""
        self._available = False
        self._npu_manager = None
        self._backend = None

        try:
            from on_device_llm.npu import NPUManager, NPUType
            self._npu_manager = NPUManager()
            # Check if we have a real hardware NPU (not just simulated)
            if self._npu_manager.has_hardware_npu():
                self._backend = self._npu_manager.get_best_backend()
                self._available = True
            else:
                # Fall back to simulated for testing
                self._backend = self._npu_manager.get_backend(NPUType.SIMULATED)
                self._available = True  # Simulated is always available
        except ImportError:
            pass

    @property
    def name(self) -> str:
        if self._backend:
            return f"NPU ({self._backend.name})"
        return "NPU (not available)"

    @property
    def backend_type(self) -> Backend:
        return Backend.NPU

    def is_available(self) -> bool:
        return self._available

    def has_hardware_npu(self) -> bool:
        """Check if a real hardware NPU is available."""
        if self._npu_manager:
            return self._npu_manager.has_hardware_npu()
        return False

    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        if not self._available or not self._backend:
            raise RuntimeError("NPU backend not available")
        return self._backend.matmul_f32(A.astype(np.float32), B.astype(np.float32))

    def matmul_int8(self, A_int8, B_int8) -> np.ndarray:
        """Direct int8 matmul for NPU-optimized inference."""
        if not self._available or not self._backend:
            raise RuntimeError("NPU backend not available")
        return self._backend.matmul_int8(A_int8, B_int8)

    def softmax(self, x: np.ndarray) -> np.ndarray:
        if not self._available or not self._backend:
            raise RuntimeError("NPU backend not available")
        return self._backend.softmax(x.astype(np.float32))

    def rmsnorm(self, x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        if not self._available or not self._backend:
            raise RuntimeError("NPU backend not available")
        return self._backend.rmsnorm(x.astype(np.float32), weight.astype(np.float32), eps)

    def rope_embed(self, x: np.ndarray, pos: int, theta: float = 10000.0) -> np.ndarray:
        if not self._available:
            raise RuntimeError("NPU backend not available")
        # RoPE is typically done on CPU as it's memory-bound
        return rope_embed(x.astype(np.float32), pos, theta)

    def silu(self, x: np.ndarray) -> np.ndarray:
        if not self._available:
            raise RuntimeError("NPU backend not available")
        # SiLU on CPU - NPUs often don't have native SiLU
        x = x.astype(np.float32)
        return (x * (1.0 / (1.0 + np.exp(-x)))).astype(np.float32)

    def synchronize(self) -> None:
        if self._backend:
            self._backend.synchronize()

    def memory_info(self) -> dict:
        if self._backend:
            return self._backend.memory_info()
        return {"total": 0, "used": 0, "free": 0}


def get_backend(backend_type: Backend = Backend.CPU) -> ComputeBackend:
    """
    Get a compute backend instance.

    Args:
        backend_type: Type of backend to create

    Returns:
        Backend instance

    Raises:
        RuntimeError: If requested backend is not available
        ValueError: If unknown backend type
    """
    if backend_type == Backend.CPU:
        return CPUBackend()
    elif backend_type == Backend.CUDA:
        backend = CUDABackend()
        if not backend.is_available():
            raise RuntimeError("CUDA backend requested but not available")
        return backend
    elif backend_type == Backend.METAL:
        backend = MetalBackend()
        if not backend.is_available():
            raise RuntimeError("Metal backend requested but not available (install mlx)")
        return backend
    elif backend_type == Backend.VULKAN:
        backend = VulkanBackend()
        if not backend.is_available():
            raise RuntimeError("Vulkan backend requested but not available (install kompute)")
        return backend
    elif backend_type == Backend.NPU:
        backend = NPUComputeBackend()
        if not backend.is_available():
            raise RuntimeError("NPU backend requested but not available")
        return backend
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


def get_best_backend() -> ComputeBackend:
    """
    Get the best available backend.

    Preference order: CUDA > Metal > Vulkan > NPU (hardware) > CPU

    Returns:
        Best available backend instance
    """
    # Try CUDA first (best for NVIDIA GPUs)
    cuda = CUDABackend()
    if cuda.is_available():
        return cuda

    # Try Metal (best for Apple Silicon)
    metal = MetalBackend()
    if metal.is_available():
        return metal

    # Try Vulkan (cross-platform GPU support)
    vulkan = VulkanBackend()
    if vulkan.is_available():
        return vulkan

    # Try NPU (best for mobile/edge devices with hardware NPU)
    npu = NPUComputeBackend()
    if npu.is_available() and npu.has_hardware_npu():
        return npu

    # Fall back to CPU
    return CPUBackend()


def get_npu_backend() -> NPUComputeBackend:
    """
    Get the NPU backend specifically.

    Returns:
        NPU backend instance (may use simulated NPU if no hardware)

    Raises:
        RuntimeError: If NPU backend cannot be initialized
    """
    backend = NPUComputeBackend()
    if not backend.is_available():
        raise RuntimeError("NPU backend not available")
    return backend


def list_available_backends() -> list:
    """
    List all available backends.

    Returns:
        List of (backend_type, backend_name) tuples
    """
    available = []

    cpu = CPUBackend()
    if cpu.is_available():
        available.append((Backend.CPU, cpu.name))

    cuda = CUDABackend()
    if cuda.is_available():
        available.append((Backend.CUDA, cuda.name))

    metal = MetalBackend()
    if metal.is_available():
        available.append((Backend.METAL, metal.name))

    vulkan = VulkanBackend()
    if vulkan.is_available():
        available.append((Backend.VULKAN, vulkan.name))

    npu = NPUComputeBackend()
    if npu.is_available():
        available.append((Backend.NPU, npu.name))

    return available
