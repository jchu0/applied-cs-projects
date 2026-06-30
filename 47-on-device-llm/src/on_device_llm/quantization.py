"""Quantization utilities for on-device LLM inference.

This module provides quantization and dequantization functions for various
GGML quantization formats (Q4_0, Q8_0, etc.) used in on-device inference.
"""

import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Tuple

import numpy as np


class GGMLType(IntEnum):
    """GGML quantization types."""
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15


# Block sizes for each quantization type
BLOCK_SIZES = {
    GGMLType.Q4_0: 32,
    GGMLType.Q4_1: 32,
    GGMLType.Q8_0: 32,
    GGMLType.Q8_1: 32,
}

# Bytes per block for each quantization type
BYTES_PER_BLOCK = {
    GGMLType.Q4_0: 18,  # 2 bytes scale + 16 bytes data (32 4-bit values)
    GGMLType.Q4_1: 20,  # 2 bytes scale + 2 bytes min + 16 bytes data
    GGMLType.Q8_0: 34,  # 2 bytes scale + 32 bytes data (32 8-bit values)
    GGMLType.Q8_1: 36,  # 2 bytes scale + 2 bytes sum + 32 bytes data
}


@dataclass
class QuantizedTensor:
    """A quantized tensor with metadata."""

    data: bytes
    shape: Tuple[int, ...]
    dtype: GGMLType

    @property
    def n_elements(self) -> int:
        """Total number of elements in the tensor."""
        result = 1
        for dim in self.shape:
            result *= dim
        return result

    @property
    def n_blocks(self) -> int:
        """Number of quantization blocks."""
        block_size = BLOCK_SIZES.get(self.dtype, 32)
        return (self.n_elements + block_size - 1) // block_size

    def dequantize(self) -> np.ndarray:
        """Dequantize the tensor to float32."""
        if self.dtype == GGMLType.Q4_0:
            return dequantize_q4_0(self.data, self.shape)
        elif self.dtype == GGMLType.Q8_0:
            return dequantize_q8_0(self.data, self.shape)
        elif self.dtype == GGMLType.F32:
            return np.frombuffer(self.data, dtype=np.float32).reshape(self.shape)
        elif self.dtype == GGMLType.F16:
            return np.frombuffer(self.data, dtype=np.float16).astype(np.float32).reshape(self.shape)
        else:
            raise NotImplementedError(f"Dequantization not implemented for {self.dtype}")


def quantize_q8_0(x: np.ndarray) -> Tuple[bytes, GGMLType]:
    """
    Quantize a float32 tensor to Q8_0 format.

    Q8_0 format:
    - Block size: 32 elements
    - Per block: 1 float16 scale + 32 int8 values
    - Total: 34 bytes per block

    Args:
        x: Input tensor (will be flattened)

    Returns:
        Tuple of (quantized bytes, dtype)
    """
    x_flat = x.flatten().astype(np.float32)
    n_elements = len(x_flat)
    block_size = 32
    n_blocks = (n_elements + block_size - 1) // block_size

    # Pad to block boundary
    if n_elements % block_size != 0:
        x_flat = np.pad(x_flat, (0, block_size - n_elements % block_size))

    result = bytearray()

    for i in range(n_blocks):
        start = i * block_size
        block = x_flat[start:start + block_size]

        # Compute scale as max absolute value / 127
        amax = np.max(np.abs(block))
        scale = amax / 127.0 if amax > 0 else 1.0

        # Quantize to int8
        quantized = np.clip(np.round(block / scale), -128, 127).astype(np.int8)

        # Pack: scale (float16) + values (int8 x 32)
        result.extend(np.array([scale], dtype=np.float16).tobytes())
        result.extend(quantized.tobytes())

    return bytes(result), GGMLType.Q8_0


def dequantize_q8_0(data: bytes, shape: Tuple[int, ...]) -> np.ndarray:
    """
    Dequantize Q8_0 format to float32.

    Args:
        data: Quantized bytes
        shape: Target output shape

    Returns:
        Dequantized float32 tensor
    """
    block_size = 32
    n_elements = 1
    for dim in shape:
        n_elements *= dim

    n_blocks = (n_elements + block_size - 1) // block_size
    result = np.zeros(n_blocks * block_size, dtype=np.float32)

    offset = 0
    for i in range(n_blocks):
        # Read scale (float16)
        scale = np.frombuffer(data[offset:offset + 2], dtype=np.float16)[0]
        offset += 2

        # Read quantized values (int8)
        values = np.frombuffer(data[offset:offset + 32], dtype=np.int8)
        offset += 32

        # Dequantize
        start = i * block_size
        result[start:start + block_size] = values.astype(np.float32) * float(scale)

    return result[:n_elements].reshape(shape)


def quantize_q4_0(x: np.ndarray) -> Tuple[bytes, GGMLType]:
    """
    Quantize a float32 tensor to Q4_0 format.

    Q4_0 format:
    - Block size: 32 elements
    - Per block: 1 float16 scale + 16 bytes (32 4-bit values packed)
    - Total: 18 bytes per block
    - Values are centered at -8 (range: -8 to 7)

    Args:
        x: Input tensor (will be flattened)

    Returns:
        Tuple of (quantized bytes, dtype)
    """
    x_flat = x.flatten().astype(np.float32)
    n_elements = len(x_flat)
    block_size = 32
    n_blocks = (n_elements + block_size - 1) // block_size

    # Pad to block boundary
    if n_elements % block_size != 0:
        x_flat = np.pad(x_flat, (0, block_size - n_elements % block_size))

    result = bytearray()

    for i in range(n_blocks):
        start = i * block_size
        block = x_flat[start:start + block_size]

        # Compute scale as max absolute value / 7
        amax = np.max(np.abs(block))
        scale = amax / 7.0 if amax > 0 else 1.0

        # Quantize to 4-bit signed (-8 to 7)
        quantized = np.clip(np.round(block / scale), -8, 7).astype(np.int8)

        # Pack two 4-bit values per byte
        packed = np.zeros(16, dtype=np.uint8)
        for j in range(16):
            low = (quantized[j * 2] + 8) & 0xF  # Convert from [-8,7] to [0,15]
            high = (quantized[j * 2 + 1] + 8) & 0xF
            packed[j] = low | (high << 4)

        # Pack: scale (float16) + packed values (16 bytes)
        result.extend(np.array([scale], dtype=np.float16).tobytes())
        result.extend(packed.tobytes())

    return bytes(result), GGMLType.Q4_0


def dequantize_q4_0(data: bytes, shape: Tuple[int, ...]) -> np.ndarray:
    """
    Dequantize Q4_0 format to float32.

    Args:
        data: Quantized bytes
        shape: Target output shape

    Returns:
        Dequantized float32 tensor
    """
    block_size = 32
    n_elements = 1
    for dim in shape:
        n_elements *= dim

    n_blocks = (n_elements + block_size - 1) // block_size
    result = np.zeros(n_blocks * block_size, dtype=np.float32)

    offset = 0
    for i in range(n_blocks):
        # Read scale (float16)
        scale = np.frombuffer(data[offset:offset + 2], dtype=np.float16)[0]
        offset += 2

        # Read packed 4-bit values (16 bytes = 32 values)
        packed = np.frombuffer(data[offset:offset + 16], dtype=np.uint8)
        offset += 16

        # Unpack 4-bit values (cast to int to avoid uint8 underflow)
        values = np.zeros(32, dtype=np.float32)
        for j in range(16):
            values[j * 2] = int(packed[j] & 0xF) - 8  # Convert from [0,15] to [-8,7]
            values[j * 2 + 1] = int(packed[j] >> 4) - 8

        # Dequantize
        start = i * block_size
        result[start:start + block_size] = values * float(scale)

    return result[:n_elements].reshape(shape)


def compute_quantization_error(original: np.ndarray, quantized_data: bytes,
                                dtype: GGMLType) -> Tuple[float, float]:
    """
    Compute the quantization error metrics.

    Args:
        original: Original float32 tensor
        quantized_data: Quantized bytes
        dtype: Quantization type

    Returns:
        Tuple of (mean absolute error, max absolute error)
    """
    tensor = QuantizedTensor(quantized_data, original.shape, dtype)
    reconstructed = tensor.dequantize()

    mae = np.mean(np.abs(original.flatten() - reconstructed.flatten()))
    max_error = np.max(np.abs(original.flatten() - reconstructed.flatten()))

    return float(mae), float(max_error)


def get_type_size(dtype: GGMLType) -> int:
    """Get the size in bytes per element for a given type."""
    if dtype == GGMLType.F32:
        return 4
    elif dtype == GGMLType.F16:
        return 2
    elif dtype in (GGMLType.Q4_0, GGMLType.Q4_1):
        # 4 bits per element = 0.5 bytes, but stored in blocks
        return 0  # Use calc_tensor_size instead
    elif dtype in (GGMLType.Q8_0, GGMLType.Q8_1):
        return 0  # Use calc_tensor_size instead
    return 0


def calc_tensor_size(shape: Tuple[int, ...], dtype: GGMLType) -> int:
    """Calculate tensor size in bytes including quantization overhead."""
    n_elements = 1
    for dim in shape:
        n_elements *= dim

    if dtype == GGMLType.F32:
        return n_elements * 4
    elif dtype == GGMLType.F16:
        return n_elements * 2
    elif dtype in BLOCK_SIZES:
        block_size = BLOCK_SIZES[dtype]
        bytes_per_block = BYTES_PER_BLOCK[dtype]
        n_blocks = (n_elements + block_size - 1) // block_size
        return n_blocks * bytes_per_block

    return n_elements  # Fallback
