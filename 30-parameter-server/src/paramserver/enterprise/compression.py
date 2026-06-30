"""Gradient compression for bandwidth reduction.

Implements various compression techniques:
- Quantization: Reduce precision (1-8 bits)
- Top-K: Keep only largest K% gradients
- Random-K: Randomly sample K% gradients
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Tuple
import numpy as np


class CompressionType(Enum):
    """Type of gradient compression."""
    NONE = "none"
    QUANTIZE = "quantize"
    TOP_K = "top_k"
    RANDOM_K = "random_k"


class GradientCompressor(ABC):
    """Abstract base class for gradient compressors."""

    @abstractmethod
    def compress(
        self,
        gradient: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """Compress a gradient.

        Args:
            gradient: Original gradient array.

        Returns:
            Tuple of (compressed_data, metadata for decompression).
        """
        pass

    @abstractmethod
    def decompress(
        self,
        compressed: np.ndarray,
        metadata: dict,
    ) -> np.ndarray:
        """Decompress a gradient.

        Args:
            compressed: Compressed gradient data.
            metadata: Metadata from compression.

        Returns:
            Decompressed gradient array.
        """
        pass

    @property
    @abstractmethod
    def compression_type(self) -> CompressionType:
        """Get the compression type."""
        pass

    def get_compression_ratio(
        self,
        original: np.ndarray,
        compressed: np.ndarray,
    ) -> float:
        """Calculate compression ratio.

        Args:
            original: Original array.
            compressed: Compressed array.

        Returns:
            Compression ratio (original size / compressed size).
        """
        original_bytes = original.nbytes
        compressed_bytes = compressed.nbytes
        if compressed_bytes == 0:
            return float("inf")
        return original_bytes / compressed_bytes


class QuantizationCompressor(GradientCompressor):
    """Quantize gradients to lower precision.

    Reduces gradient precision to 1-8 bits, significantly
    reducing bandwidth at the cost of some accuracy.

    Attributes:
        bits: Number of bits for quantization (1-8).
        use_error_feedback: Whether to accumulate quantization errors.
    """

    def __init__(
        self,
        bits: int = 8,
        use_error_feedback: bool = True,
    ):
        """Initialize quantization compressor.

        Args:
            bits: Quantization bits (1-8).
            use_error_feedback: Accumulate errors for next round.
        """
        if not 1 <= bits <= 8:
            raise ValueError(f"bits must be 1-8, got {bits}")

        self.bits = bits
        self.use_error_feedback = use_error_feedback
        self.num_levels = 2 ** bits
        self._error_buffer: Optional[np.ndarray] = None

    @property
    def compression_type(self) -> CompressionType:
        return CompressionType.QUANTIZE

    def compress(
        self,
        gradient: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """Quantize gradient to lower precision."""
        original_shape = gradient.shape
        original_dtype = gradient.dtype
        flat = gradient.flatten().astype(np.float32)

        # Add accumulated error if using error feedback
        if self.use_error_feedback and self._error_buffer is not None:
            if self._error_buffer.shape == flat.shape:
                flat = flat + self._error_buffer

        # Find min/max for scaling
        g_min = float(flat.min())
        g_max = float(flat.max())

        if g_max == g_min:
            # All zeros or constant
            quantized = np.zeros(len(flat), dtype=np.uint8)
            metadata = {
                "shape": original_shape,
                "dtype": str(original_dtype),
                "min": g_min,
                "max": g_max,
                "bits": self.bits,
            }
            return quantized, metadata

        # Normalize to [0, 1] then quantize
        scale = (g_max - g_min) / (self.num_levels - 1)
        normalized = (flat - g_min) / (g_max - g_min)
        quantized = np.round(normalized * (self.num_levels - 1)).astype(np.uint8)

        # Calculate error for feedback
        if self.use_error_feedback:
            dequantized = quantized.astype(np.float32) * scale + g_min
            self._error_buffer = flat - dequantized

        metadata = {
            "shape": original_shape,
            "dtype": str(original_dtype),
            "min": g_min,
            "max": g_max,
            "bits": self.bits,
        }

        return quantized, metadata

    def decompress(
        self,
        compressed: np.ndarray,
        metadata: dict,
    ) -> np.ndarray:
        """Dequantize to original precision."""
        g_min = metadata["min"]
        g_max = metadata["max"]
        shape = metadata["shape"]
        dtype = np.dtype(metadata["dtype"])

        if g_max == g_min:
            return np.full(shape, g_min, dtype=dtype)

        # Dequantize
        scale = (g_max - g_min) / (self.num_levels - 1)
        decompressed = compressed.astype(np.float32) * scale + g_min

        return decompressed.reshape(shape).astype(dtype)

    def reset_error_buffer(self) -> None:
        """Reset accumulated error."""
        self._error_buffer = None


class TopKCompressor(GradientCompressor):
    """Keep only top-K% largest gradients by magnitude.

    Sparsifies gradients by keeping only the most significant
    values. Uses error feedback to preserve small gradients.

    Attributes:
        k_percent: Percentage of gradients to keep (0-100).
        use_error_feedback: Whether to accumulate dropped gradients.
    """

    def __init__(
        self,
        k_percent: float = 10.0,
        use_error_feedback: bool = True,
    ):
        """Initialize top-K compressor.

        Args:
            k_percent: Percentage to keep (0-100).
            use_error_feedback: Accumulate dropped gradients.
        """
        if not 0 < k_percent <= 100:
            raise ValueError(f"k_percent must be (0, 100], got {k_percent}")

        self.k_percent = k_percent
        self.use_error_feedback = use_error_feedback
        self._error_buffer: Optional[np.ndarray] = None

    @property
    def compression_type(self) -> CompressionType:
        return CompressionType.TOP_K

    def compress(
        self,
        gradient: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """Compress by keeping top-K largest values."""
        original_shape = gradient.shape
        original_dtype = gradient.dtype
        flat = gradient.flatten().astype(np.float32)

        # Add accumulated error
        if self.use_error_feedback and self._error_buffer is not None:
            if self._error_buffer.shape == flat.shape:
                flat = flat + self._error_buffer

        # Calculate K (number of elements to keep)
        k = max(1, int(len(flat) * self.k_percent / 100))

        # Find top-K indices by magnitude
        abs_vals = np.abs(flat)
        if k >= len(flat):
            # Keep all
            indices = np.arange(len(flat), dtype=np.int32)
            values = flat
        else:
            # Partial sort to find threshold
            partition_idx = len(flat) - k
            threshold = np.partition(abs_vals, partition_idx)[partition_idx]
            mask = abs_vals >= threshold
            indices = np.where(mask)[0].astype(np.int32)
            values = flat[indices]

        # Store error for feedback
        if self.use_error_feedback:
            self._error_buffer = flat.copy()
            self._error_buffer[indices] = 0

        metadata = {
            "shape": original_shape,
            "dtype": str(original_dtype),
            "size": len(flat),
        }

        # Pack indices and values
        compressed = np.concatenate([
            indices.view(np.uint8),
            values.view(np.uint8),
        ])

        return compressed, metadata

    def decompress(
        self,
        compressed: np.ndarray,
        metadata: dict,
    ) -> np.ndarray:
        """Decompress sparse gradient."""
        shape = metadata["shape"]
        dtype = np.dtype(metadata["dtype"])
        size = metadata["size"]

        # Unpack indices and values
        # indices are int32 (4 bytes each)
        idx_bytes = compressed.nbytes // 2  # Half for indices, half for values
        n_elements = idx_bytes // 4

        indices = compressed[:idx_bytes].view(np.int32)
        values = compressed[idx_bytes:].view(np.float32)

        # Reconstruct sparse array
        result = np.zeros(size, dtype=np.float32)
        result[indices] = values

        return result.reshape(shape).astype(dtype)

    def reset_error_buffer(self) -> None:
        """Reset accumulated error."""
        self._error_buffer = None


class RandomKCompressor(GradientCompressor):
    """Randomly sample K% of gradients.

    Unlike top-K, this samples uniformly at random,
    which can be faster but less accurate.

    Attributes:
        k_percent: Percentage of gradients to keep (0-100).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        k_percent: float = 10.0,
        seed: Optional[int] = None,
    ):
        """Initialize random-K compressor.

        Args:
            k_percent: Percentage to keep (0-100).
            seed: Random seed.
        """
        if not 0 < k_percent <= 100:
            raise ValueError(f"k_percent must be (0, 100], got {k_percent}")

        self.k_percent = k_percent
        self._rng = np.random.default_rng(seed)

    @property
    def compression_type(self) -> CompressionType:
        return CompressionType.RANDOM_K

    def compress(
        self,
        gradient: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """Compress by random sampling."""
        original_shape = gradient.shape
        original_dtype = gradient.dtype
        flat = gradient.flatten().astype(np.float32)

        # Calculate K
        k = max(1, int(len(flat) * self.k_percent / 100))
        k = min(k, len(flat))

        # Random sample without replacement
        indices = self._rng.choice(len(flat), k, replace=False).astype(np.int32)
        indices.sort()  # Sort for better cache locality
        values = flat[indices]

        # Scale values to account for sampling
        scale_factor = len(flat) / k
        values = values * scale_factor

        metadata = {
            "shape": original_shape,
            "dtype": str(original_dtype),
            "size": len(flat),
        }

        # Pack indices and values
        compressed = np.concatenate([
            indices.view(np.uint8),
            values.view(np.uint8),
        ])

        return compressed, metadata

    def decompress(
        self,
        compressed: np.ndarray,
        metadata: dict,
    ) -> np.ndarray:
        """Decompress randomly sampled gradient."""
        shape = metadata["shape"]
        dtype = np.dtype(metadata["dtype"])
        size = metadata["size"]

        # Unpack
        idx_bytes = compressed.nbytes // 2
        indices = compressed[:idx_bytes].view(np.int32)
        values = compressed[idx_bytes:].view(np.float32)

        # Reconstruct
        result = np.zeros(size, dtype=np.float32)
        result[indices] = values

        return result.reshape(shape).astype(dtype)
