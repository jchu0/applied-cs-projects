"""Core quantization types and data structures."""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto


class QuantType(Enum):
    """Quantization bit widths."""
    INT8 = 8
    INT4 = 4
    INT2 = 2
    FP16 = 16
    FP8 = 8
    FP8_E4M3 = "fp8_e4m3"  # FP8 with 4-bit exponent, 3-bit mantissa (forward)
    FP8_E5M2 = "fp8_e5m2"  # FP8 with 5-bit exponent, 2-bit mantissa (backward)
    NF4 = 4  # Normal float 4-bit
    GPTQ = "gptq"  # GPTQ quantization method
    AWQ = "awq"  # Activation-aware Weight Quantization


class ScaleType(Enum):
    """Scale granularity."""
    PER_TENSOR = auto()
    PER_CHANNEL = auto()
    PER_GROUP = auto()
    PER_TOKEN = auto()


@dataclass
class QuantConfig:
    """Configuration for quantization."""
    bits: int = 8
    quant_type: QuantType = QuantType.INT8
    scale_type: ScaleType = ScaleType.PER_CHANNEL
    group_size: int = 128
    symmetric: bool = True
    desc_act: bool = False  # Activation description
    zero_point: bool = False  # Whether to use zero points

    # Calibration
    num_calibration_samples: int = 128
    calibration_method: str = "minmax"

    # Advanced
    use_double_quant: bool = False
    fp16_logits: bool = True
    model_seqlen: int = 2048

    # GPTQ specific
    block_size: int = 128
    dampening: float = 0.01
    use_multi_gpu: bool = False


@dataclass
class QuantizedTensor:
    """Quantized tensor representation."""
    data: np.ndarray          # Quantized weights (int8/int4)
    scales: np.ndarray = None        # Scaling factors (alternative name)
    zeros: Optional[np.ndarray] = None  # Zero points

    bits: int = 8
    scale_type: ScaleType = ScaleType.PER_CHANNEL
    group_size: int = 128

    original_shape: Tuple[int, ...] = ()

    # Alternative attribute names for compatibility
    scale: np.ndarray = None         # Alias for scales
    zero_point: np.ndarray = None    # Alias for zeros
    config: 'QuantConfig' = None     # Config reference
    activation_scale: np.ndarray = None  # AWQ activation scales

    def __post_init__(self):
        """Handle alternative attribute names."""
        # Support both 'scales' and 'scale'
        if self.scale is not None and self.scales is None:
            self.scales = self.scale
        elif self.scales is not None and self.scale is None:
            self.scale = self.scales
        # Support both 'zeros' and 'zero_point'
        if self.zero_point is not None and self.zeros is None:
            self.zeros = self.zero_point
        elif self.zeros is not None and self.zero_point is None:
            self.zero_point = self.zeros

    def dequantize(self) -> np.ndarray:
        """Dequantize back to float."""
        if self.bits == 4:
            # Unpack INT4
            data = unpack_int4(self.data)
        else:
            data = self.data.astype(np.float32)

        # Apply scales based on type
        if self.scale_type == ScaleType.PER_TENSOR:
            if self.zeros is not None:
                z = np.float32(self.zeros) if np.isscalar(self.zeros) else self.zeros
                data = data - z
            return data * self.scales
        elif self.scale_type == ScaleType.PER_CHANNEL:
            if self.zeros is not None:
                data = data - self.zeros.reshape(-1, 1)
            return data * self.scales.reshape(-1, 1)
        elif self.scale_type == ScaleType.PER_GROUP:
            # Reshape for group scaling
            num_groups = data.shape[1] // self.group_size
            if num_groups == 0:
                # Fallback to per-channel if not enough columns for groups
                if self.zeros is not None:
                    data = data - self.zeros.reshape(-1, 1)
                return data * self.scales.reshape(-1, 1)
            data = data.reshape(data.shape[0], num_groups, self.group_size)
            scales = self.scales.reshape(data.shape[0], num_groups, 1)
            if self.zeros is not None:
                zeros = self.zeros.reshape(data.shape[0], num_groups, 1)
                data = data - zeros
            result = data * scales
            return result.reshape(self.original_shape)
        else:
            if self.zeros is not None:
                data = data - self.zeros
            return data * self.scales

    @property
    def dtype(self) -> np.dtype:
        """Get the dtype of the quantized data."""
        return self.data.dtype

    @property
    def packed_data(self) -> np.ndarray:
        """Alias for data - used for INT4 packed representation."""
        return self.data

    @property
    def nbytes(self) -> int:
        """Get total size in bytes."""
        data_bytes = self.data.nbytes
        scale_bytes = self.scales.nbytes if self.scales is not None else 0
        zero_bytes = self.zeros.nbytes if self.zeros is not None else 0
        return data_bytes + scale_bytes + zero_bytes

    def memory_usage(self) -> int:
        """Get memory usage in bytes (alias for nbytes)."""
        return self.nbytes

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the original shape of the tensor."""
        return self.original_shape if self.original_shape else self.data.shape

    @property
    def is_quantized(self) -> bool:
        """Check if tensor is quantized."""
        return True

    def save(self, path: str):
        """Save quantized tensor to file."""
        np.savez(
            path,
            data=self.data,
            scales=self.scales,
            zeros=self.zeros,
            bits=self.bits,
            scale_type=self.scale_type.value,
            group_size=self.group_size,
            original_shape=self.original_shape
        )

    @classmethod
    def load(cls, path: str) -> 'QuantizedTensor':
        """Load quantized tensor from file."""
        data = np.load(path, allow_pickle=True)
        return cls(
            data=data['data'],
            scales=data['scales'],
            zeros=data['zeros'] if data['zeros'] is not None else None,
            bits=int(data['bits']),
            scale_type=ScaleType(data['scale_type']),
            group_size=int(data['group_size']),
            original_shape=tuple(data['original_shape'])
        )


class QuantizedLinear:
    """Quantized linear layer for efficient inference."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: QuantConfig = None
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        self.config = config or QuantConfig()

        # Quantized weight
        self.qweight: Optional[QuantizedTensor] = None
        self.bias_data: Optional[np.ndarray] = None

    def quantize_weight(self, weight: np.ndarray):
        """Quantize weight matrix."""
        config = self.config

        if config.bits == 8:
            qweight, scales, zeros = quantize_int8(
                weight,
                config.scale_type,
                config.group_size,
                config.symmetric
            )
        elif config.bits == 4:
            qweight, scales, zeros = quantize_int4(
                weight,
                config.scale_type,
                config.group_size,
                config.symmetric
            )
        else:
            raise ValueError(f"Unsupported bits: {config.bits}")

        self.qweight = QuantizedTensor(
            data=qweight,
            scales=scales,
            zeros=zeros,
            bits=config.bits,
            scale_type=config.scale_type,
            group_size=config.group_size,
            original_shape=weight.shape
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with quantized weights."""
        if self.qweight is None:
            raise RuntimeError("Weights not quantized")

        # Dequantize weight
        weight = self.qweight.dequantize()

        # Matrix multiply
        output = x @ weight.T

        if self.has_bias and self.bias_data is not None:
            output = output + self.bias_data

        return output

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


def quantize_int8(
    tensor: np.ndarray,
    scale_type: ScaleType,
    group_size: int = 128,
    symmetric: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize tensor to INT8."""
    if scale_type == ScaleType.PER_TENSOR:
        if symmetric:
            scale = np.abs(tensor).max() / 127.0
            if scale == 0:
                scale = 1.0
            qtensor = np.clip(np.round(tensor / scale), -128, 127).astype(np.int8)
            return qtensor, np.float32(scale), np.int8(0)
        else:
            min_val = tensor.min()
            max_val = tensor.max()
            scale = (max_val - min_val) / 255.0
            if scale == 0:
                scale = 1.0
            zero = np.round(-min_val / scale).astype(np.int8)
            qtensor = np.clip(np.round(tensor / scale) + zero, 0, 255).astype(np.uint8)
            return qtensor, np.float32(scale), zero

    elif scale_type == ScaleType.PER_CHANNEL:
        if symmetric:
            scales = np.abs(tensor).max(axis=1) / 127.0
            scales = np.where(scales == 0, 1, scales)
            qtensor = np.clip(np.round(tensor / scales.reshape(-1, 1)), -128, 127).astype(np.int8)
            zeros = np.zeros(scales.shape, dtype=np.int8)
            return qtensor, scales, zeros
        else:
            min_vals = tensor.min(axis=1)
            max_vals = tensor.max(axis=1)
            scales = (max_vals - min_vals) / 255.0
            scales = np.where(scales == 0, 1, scales)
            zeros = np.round(-min_vals / scales).astype(np.int8)
            qtensor = np.clip(
                np.round(tensor / scales.reshape(-1, 1)) + zeros.reshape(-1, 1),
                0, 255
            ).astype(np.uint8)
            return qtensor, scales, zeros

    elif scale_type == ScaleType.PER_GROUP:
        out_features, in_features = tensor.shape
        num_groups = in_features // group_size
        tensor_grouped = tensor.reshape(out_features, num_groups, group_size)

        if symmetric:
            scales = np.abs(tensor_grouped).max(axis=2) / 127.0
            scales = np.where(scales == 0, 1, scales)
            qtensor = np.clip(
                np.round(tensor_grouped / scales[:, :, np.newaxis]),
                -128, 127
            ).astype(np.int8)
            zeros = np.zeros(scales.shape, dtype=np.int8)
            return qtensor.reshape(out_features, in_features), scales, zeros
        else:
            min_vals = tensor_grouped.min(axis=2)
            max_vals = tensor_grouped.max(axis=2)
            scales = (max_vals - min_vals) / 255.0
            scales = np.where(scales == 0, 1, scales)
            zeros = np.round(-min_vals / scales).astype(np.int8)
            qtensor = np.clip(
                np.round(tensor_grouped / scales[:, :, np.newaxis]) + zeros[:, :, np.newaxis],
                0, 255
            ).astype(np.uint8)
            return qtensor.reshape(out_features, in_features), scales, zeros

    else:
        raise ValueError(f"Unsupported scale type: {scale_type}")


def quantize_int4(
    tensor: np.ndarray,
    scale_type: ScaleType,
    group_size: int = 128,
    symmetric: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize tensor to INT4."""
    if scale_type == ScaleType.PER_GROUP:
        out_features, in_features = tensor.shape
        num_groups = in_features // group_size
        tensor_grouped = tensor.reshape(out_features, num_groups, group_size)

        if symmetric:
            scales = np.abs(tensor_grouped).max(axis=2) / 7.0
            scales = np.where(scales == 0, 1, scales)
            qtensor = np.clip(
                np.round(tensor_grouped / scales[:, :, np.newaxis]),
                -8, 7
            ).astype(np.int8)
            zeros = np.zeros(scales.shape, dtype=np.int8)
        else:
            min_vals = tensor_grouped.min(axis=2)
            max_vals = tensor_grouped.max(axis=2)
            scales = (max_vals - min_vals) / 15.0
            scales = np.where(scales == 0, 1, scales)
            zeros = np.round(-min_vals / scales).astype(np.int8)
            qtensor = np.clip(
                np.round(tensor_grouped / scales[:, :, np.newaxis]) + zeros[:, :, np.newaxis],
                0, 15
            ).astype(np.uint8)

        # Pack INT4
        packed = pack_int4(qtensor.reshape(out_features, in_features))
        return packed, scales, zeros

    else:
        # Per-channel INT4
        if symmetric:
            scales = np.abs(tensor).max(axis=1) / 7.0
            scales = np.where(scales == 0, 1, scales)
            qtensor = np.clip(
                np.round(tensor / scales.reshape(-1, 1)),
                -8, 7
            ).astype(np.int8)
            zeros = np.zeros(scales.shape, dtype=np.int8)
        else:
            min_vals = tensor.min(axis=1)
            max_vals = tensor.max(axis=1)
            scales = (max_vals - min_vals) / 15.0
            scales = np.where(scales == 0, 1, scales)
            zeros = np.round(-min_vals / scales).astype(np.int8)
            qtensor = np.clip(
                np.round(tensor / scales.reshape(-1, 1)) + zeros.reshape(-1, 1),
                0, 15
            ).astype(np.uint8)

        packed = pack_int4(qtensor)
        return packed, scales, zeros


def pack_int4(tensor: np.ndarray) -> np.ndarray:
    """Pack two INT4 values into one INT8."""
    flat = tensor.flatten()
    if len(flat) % 2 != 0:
        flat = np.append(flat, 0)

    # Pack pairs
    packed = (flat[0::2] & 0xF) | ((flat[1::2] & 0xF) << 4)
    return packed.astype(np.uint8).reshape(tensor.shape[0], -1)


def unpack_int4(packed: np.ndarray, signed: bool = True) -> np.ndarray:
    """Unpack INT8 to two INT4 values.

    Args:
        packed: Packed INT4 data
        signed: If True, treat values as signed [-8, 7], else unsigned [0, 15]
    """
    flat = packed.flatten()

    # Unpack
    low = flat & 0xF
    high = (flat >> 4) & 0xF

    # Convert to signed if needed (values >= 8 become negative)
    if signed:
        low = np.where(low >= 8, low.astype(np.int8) - 16, low)
        high = np.where(high >= 8, high.astype(np.int8) - 16, high)

    # Interleave
    unpacked = np.empty(len(flat) * 2, dtype=np.float32)
    unpacked[0::2] = low
    unpacked[1::2] = high

    return unpacked.reshape(packed.shape[0], -1)


# FP8 Format constants
# E4M3: 4-bit exponent (bias=7), 3-bit mantissa, range: [-448, 448], no inf/nan
# E5M2: 5-bit exponent (bias=15), 2-bit mantissa, range: [-57344, 57344], has inf/nan
FP8_E4M3_MAX = 448.0
FP8_E4M3_MIN = -448.0
FP8_E5M2_MAX = 57344.0
FP8_E5M2_MIN = -57344.0


def float_to_fp8_e4m3(value: float) -> int:
    """Convert a float to FP8 E4M3 format (4-bit exp, 3-bit mantissa).

    E4M3 has no inf/nan, bias=7, max=448.
    """
    if value == 0:
        return 0

    sign = 1 if value < 0 else 0
    value = abs(value)

    # Clamp to representable range
    value = min(value, FP8_E4M3_MAX)

    # Handle denormals (exponent = 0)
    if value < 2**-6:  # Smallest normal is 2^-6
        # Denormalized: mantissa represents value / 2^-6
        mantissa = int(round(value / (2**-9)))  # 2^-9 is smallest denormal step
        mantissa = min(mantissa, 7)
        return (sign << 7) | mantissa

    # Normal numbers
    exp = int(np.floor(np.log2(value)))
    exp_biased = exp + 7  # bias = 7

    if exp_biased < 1:
        exp_biased = 0
        mantissa = int(round(value / (2**-9)))
        mantissa = min(mantissa, 7)
    elif exp_biased > 14:
        exp_biased = 15  # Max exponent
        # Max value is 448 = 1.75 * 256 = (1 + 6/8) * 2^8, so mantissa = 6
        mantissa = 6
    else:
        # Compute mantissa (3 bits)
        mantissa_float = (value / (2**exp)) - 1.0
        mantissa = int(round(mantissa_float * 8))
        mantissa = min(max(mantissa, 0), 7)

    return (sign << 7) | (exp_biased << 3) | mantissa


def fp8_e4m3_to_float(fp8: int) -> float:
    """Convert FP8 E4M3 to float."""
    sign = (fp8 >> 7) & 1
    exp = (fp8 >> 3) & 0xF
    mantissa = fp8 & 0x7

    if exp == 0:
        # Denormalized
        value = mantissa * (2**-9)
    else:
        # Normalized
        value = (1.0 + mantissa / 8.0) * (2**(exp - 7))

    # Clamp to max representable value
    value = min(value, FP8_E4M3_MAX)

    return -value if sign else value


def float_to_fp8_e5m2(value: float) -> int:
    """Convert a float to FP8 E5M2 format (5-bit exp, 2-bit mantissa).

    E5M2 has inf/nan support, bias=15.
    """
    if value == 0:
        return 0

    if np.isnan(value):
        return 0x7F  # NaN representation

    sign = 1 if value < 0 else 0
    value = abs(value)

    if np.isinf(value):
        return (sign << 7) | 0x7C  # Inf representation

    # Clamp to representable range
    value = min(value, FP8_E5M2_MAX)

    # Handle denormals
    if value < 2**-14:  # Smallest normal
        mantissa = int(round(value / (2**-16)))
        mantissa = min(mantissa, 3)
        return (sign << 7) | mantissa

    # Normal numbers
    exp = int(np.floor(np.log2(value)))
    exp_biased = exp + 15  # bias = 15

    if exp_biased < 1:
        exp_biased = 0
        mantissa = int(round(value / (2**-16)))
        mantissa = min(mantissa, 3)
    elif exp_biased > 30:
        exp_biased = 31
        mantissa = 0  # Infinity
    else:
        mantissa_float = (value / (2**exp)) - 1.0
        mantissa = int(round(mantissa_float * 4))
        mantissa = min(max(mantissa, 0), 3)

    return (sign << 7) | (exp_biased << 2) | mantissa


def fp8_e5m2_to_float(fp8: int) -> float:
    """Convert FP8 E5M2 to float."""
    sign = (fp8 >> 7) & 1
    exp = (fp8 >> 2) & 0x1F
    mantissa = fp8 & 0x3

    if exp == 31:
        if mantissa == 0:
            return float('-inf') if sign else float('inf')
        else:
            return float('nan')

    if exp == 0:
        # Denormalized
        value = mantissa * (2**-16)
    else:
        # Normalized
        value = (1.0 + mantissa / 4.0) * (2**(exp - 15))

    return -value if sign else value


def quantize_fp8_e4m3(
    tensor: np.ndarray,
    scale_type: ScaleType = ScaleType.PER_TENSOR
) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize tensor to FP8 E4M3 format.

    Args:
        tensor: Input tensor to quantize
        scale_type: Scaling granularity

    Returns:
        Tuple of (quantized_data, scales)
    """
    if scale_type == ScaleType.PER_TENSOR:
        # Compute scale to fit values in FP8 range
        abs_max = np.abs(tensor).max()
        scale = abs_max / FP8_E4M3_MAX if abs_max > 0 else 1.0
        scaled = tensor / scale

        # Vectorized conversion
        flat = scaled.flatten()
        quantized = np.zeros(len(flat), dtype=np.uint8)
        for i, val in enumerate(flat):
            quantized[i] = float_to_fp8_e4m3(float(val))

        return quantized.reshape(tensor.shape), np.array([scale], dtype=np.float32)

    elif scale_type == ScaleType.PER_CHANNEL:
        out_features = tensor.shape[0]
        abs_max = np.abs(tensor).max(axis=1)
        scales = np.where(abs_max > 0, abs_max / FP8_E4M3_MAX, 1.0)

        scaled = tensor / scales.reshape(-1, 1)
        quantized = np.zeros_like(tensor, dtype=np.uint8)

        for i in range(out_features):
            for j in range(tensor.shape[1]):
                quantized[i, j] = float_to_fp8_e4m3(float(scaled[i, j]))

        return quantized, scales.astype(np.float32)

    else:
        raise ValueError(f"Unsupported scale type for FP8: {scale_type}")


def dequantize_fp8_e4m3(
    quantized: np.ndarray,
    scales: np.ndarray,
    scale_type: ScaleType = ScaleType.PER_TENSOR
) -> np.ndarray:
    """Dequantize FP8 E4M3 back to float."""
    flat = quantized.flatten()
    dequantized = np.array([fp8_e4m3_to_float(int(v)) for v in flat], dtype=np.float32)
    dequantized = dequantized.reshape(quantized.shape)

    if scale_type == ScaleType.PER_TENSOR:
        return dequantized * scales[0]
    elif scale_type == ScaleType.PER_CHANNEL:
        return dequantized * scales.reshape(-1, 1)
    else:
        return dequantized * scales


def quantize_fp8_e5m2(
    tensor: np.ndarray,
    scale_type: ScaleType = ScaleType.PER_TENSOR
) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize tensor to FP8 E5M2 format.

    Args:
        tensor: Input tensor to quantize
        scale_type: Scaling granularity

    Returns:
        Tuple of (quantized_data, scales)
    """
    if scale_type == ScaleType.PER_TENSOR:
        abs_max = np.abs(tensor).max()
        scale = abs_max / FP8_E5M2_MAX if abs_max > 0 else 1.0
        scaled = tensor / scale

        flat = scaled.flatten()
        quantized = np.zeros(len(flat), dtype=np.uint8)
        for i, val in enumerate(flat):
            quantized[i] = float_to_fp8_e5m2(float(val))

        return quantized.reshape(tensor.shape), np.array([scale], dtype=np.float32)

    elif scale_type == ScaleType.PER_CHANNEL:
        out_features = tensor.shape[0]
        abs_max = np.abs(tensor).max(axis=1)
        scales = np.where(abs_max > 0, abs_max / FP8_E5M2_MAX, 1.0)

        scaled = tensor / scales.reshape(-1, 1)
        quantized = np.zeros_like(tensor, dtype=np.uint8)

        for i in range(out_features):
            for j in range(tensor.shape[1]):
                quantized[i, j] = float_to_fp8_e5m2(float(scaled[i, j]))

        return quantized, scales.astype(np.float32)

    else:
        raise ValueError(f"Unsupported scale type for FP8: {scale_type}")


def dequantize_fp8_e5m2(
    quantized: np.ndarray,
    scales: np.ndarray,
    scale_type: ScaleType = ScaleType.PER_TENSOR
) -> np.ndarray:
    """Dequantize FP8 E5M2 back to float."""
    flat = quantized.flatten()
    dequantized = np.array([fp8_e5m2_to_float(int(v)) for v in flat], dtype=np.float32)
    dequantized = dequantized.reshape(quantized.shape)

    if scale_type == ScaleType.PER_TENSOR:
        return dequantized * scales[0]
    elif scale_type == ScaleType.PER_CHANNEL:
        return dequantized * scales.reshape(-1, 1)
    else:
        return dequantized * scales


@dataclass
class FP8QuantizedTensor:
    """FP8 quantized tensor representation."""
    data: np.ndarray  # FP8 encoded data (uint8)
    scales: np.ndarray
    format: str  # "e4m3" or "e5m2"
    scale_type: ScaleType = ScaleType.PER_TENSOR
    original_shape: Tuple[int, ...] = ()

    def dequantize(self) -> np.ndarray:
        """Dequantize back to float32."""
        if self.format == "e4m3":
            return dequantize_fp8_e4m3(self.data, self.scales, self.scale_type)
        else:
            return dequantize_fp8_e5m2(self.data, self.scales, self.scale_type)

    @property
    def nbytes(self) -> int:
        """Get total size in bytes."""
        return self.data.nbytes + self.scales.nbytes

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the original shape of the tensor."""
        return self.original_shape if self.original_shape else self.data.shape
