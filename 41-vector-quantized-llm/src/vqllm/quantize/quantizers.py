"""Quantization method implementations."""

import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple, Callable
from abc import ABC, abstractmethod

from ..core.types import (
    QuantConfig,
    QuantType,
    QuantizedTensor,
    QuantizedLinear,
    ScaleType,
    quantize_int8,
    quantize_int4,
    quantize_fp8_e4m3,
    quantize_fp8_e5m2,
    FP8QuantizedTensor,
)

logger = logging.getLogger(__name__)


class Quantizer(ABC):
    """Base quantizer class."""

    def __init__(self, config: QuantConfig = None):
        self.config = config or QuantConfig()
        self._calibration_stats: Dict[str, Any] = {}

    @abstractmethod
    def quantize_weight(
        self,
        weight: np.ndarray,
        name: str = ""
    ) -> QuantizedTensor:
        """Quantize a weight tensor."""
        pass

    def _validate_weight(self, weight: np.ndarray) -> np.ndarray:
        """Validate a weight tensor for quantization.

        All quantization paths (per-tensor, per-channel, per-group) operate on
        a 2-D ``[out_features, in_features]`` matrix. This surfaces a clear
        error at the public API boundary instead of letting NumPy raise an
        opaque axis/reshape error deep inside a kernel.
        """
        if not isinstance(weight, np.ndarray):
            weight = np.asarray(weight)
        if weight.ndim != 2:
            raise ValueError(
                f"quantize_weight expects a 2-D weight matrix "
                f"[out_features, in_features], got shape {weight.shape} "
                f"({weight.ndim}-D)."
            )
        if self.config.scale_type == ScaleType.PER_GROUP:
            in_features = weight.shape[1]
            group_size = self.config.group_size
            if in_features < group_size:
                raise ValueError(
                    f"PER_GROUP quantization needs in_features "
                    f"({in_features}) >= group_size ({group_size}). "
                    f"Reduce group_size or use PER_CHANNEL/PER_TENSOR."
                )
        return weight

    def set_calibration_stats(self, stats: Dict[str, Any]) -> None:
        """Set calibration statistics for quantization.

        Args:
            stats: Dictionary containing calibration statistics like
                   min/max values, activation scales, hessians, etc.
        """
        self._calibration_stats = stats

    def get_calibration_stats(self) -> Dict[str, Any]:
        """Get current calibration statistics."""
        return self._calibration_stats

    def quantize_model(self, model: Any) -> Any:
        """Quantize entire model."""
        # Generic model quantization
        if hasattr(model, 'named_parameters'):
            for name, param in model.named_parameters():
                if 'weight' in name and param.ndim >= 2:
                    qtensor = self.quantize_weight(param.data, name)
                    # Replace with quantized
                    param.data = qtensor
        return model


class INT8Quantizer(Quantizer):
    """INT8 quantization."""

    def __init__(self, config: QuantConfig = None):
        config = config or QuantConfig(bits=8, quant_type=QuantType.INT8)
        super().__init__(config)

    def quantize_weight(
        self,
        weight: np.ndarray,
        name: str = ""
    ) -> QuantizedTensor:
        """Quantize weight to INT8."""
        weight = self._validate_weight(weight)
        qdata, scales, zeros = quantize_int8(
            weight,
            self.config.scale_type,
            self.config.group_size,
            self.config.symmetric
        )

        return QuantizedTensor(
            data=qdata,
            scales=scales,
            zeros=zeros,
            bits=8,
            scale_type=self.config.scale_type,
            group_size=self.config.group_size,
            original_shape=weight.shape,
            config=self.config
        )


class INT4Quantizer(Quantizer):
    """INT4 quantization with packing."""

    def __init__(self, config: QuantConfig = None):
        config = config or QuantConfig(bits=4, quant_type=QuantType.INT4)
        super().__init__(config)

    def quantize_weight(
        self,
        weight: np.ndarray,
        name: str = ""
    ) -> QuantizedTensor:
        """Quantize weight to INT4."""
        weight = self._validate_weight(weight)
        qdata, scales, zeros = quantize_int4(
            weight,
            self.config.scale_type,
            self.config.group_size,
            self.config.symmetric
        )

        return QuantizedTensor(
            data=qdata,
            scales=scales,
            zeros=zeros,
            bits=4,
            scale_type=self.config.scale_type,
            group_size=self.config.group_size,
            original_shape=weight.shape,
            config=self.config
        )


class FP8Quantizer(Quantizer):
    """FP8 quantization supporting E4M3 and E5M2 formats.

    FP8 E4M3: 4-bit exponent, 3-bit mantissa - higher precision, smaller range
              Best for forward pass weights and activations
              Range: [-448, 448]

    FP8 E5M2: 5-bit exponent, 2-bit mantissa - lower precision, larger range
              Best for gradients in backward pass
              Range: [-57344, 57344]
    """

    def __init__(
        self,
        config: QuantConfig = None,
        format: str = "e4m3"
    ):
        """Initialize FP8 quantizer.

        Args:
            config: Quantization configuration
            format: FP8 format - "e4m3" or "e5m2"
        """
        if format not in ("e4m3", "e5m2"):
            raise ValueError(f"Unsupported FP8 format: {format}. Use 'e4m3' or 'e5m2'")

        quant_type = QuantType.FP8_E4M3 if format == "e4m3" else QuantType.FP8_E5M2
        config = config or QuantConfig(bits=8, quant_type=quant_type)
        super().__init__(config)
        self.format = format

    def quantize_weight(
        self,
        weight: np.ndarray,
        name: str = ""
    ) -> FP8QuantizedTensor:
        """Quantize weight to FP8 format.

        Args:
            weight: Weight tensor to quantize
            name: Layer name (for logging)

        Returns:
            FP8QuantizedTensor with quantized data and scales
        """
        weight = self._validate_weight(weight)
        if self.format == "e4m3":
            qdata, scales = quantize_fp8_e4m3(weight, self.config.scale_type)
        else:
            qdata, scales = quantize_fp8_e5m2(weight, self.config.scale_type)

        logger.debug(f"Quantized {name} to FP8 {self.format}: shape={weight.shape}")

        return FP8QuantizedTensor(
            data=qdata,
            scales=scales,
            format=self.format,
            scale_type=self.config.scale_type,
            original_shape=weight.shape
        )

    def quantize_model(self, model: Any) -> Any:
        """Quantize entire model to FP8.

        This is particularly useful for mixed-precision training where
        different layers can use different FP8 formats.
        """
        if hasattr(model, 'named_parameters'):
            for name, param in model.named_parameters():
                if 'weight' in name and param.ndim >= 2:
                    qtensor = self.quantize_weight(param.data, name)
                    param.data = qtensor
        return model


class GPTQQuantizer(Quantizer):
    """
    GPTQ quantization.

    Uses Hessian-based optimization for better accuracy.
    """

    def __init__(
        self,
        config: QuantConfig = None,
        actorder: bool = True,
        percdamp: float = 0.01
    ):
        config = config or QuantConfig(
            bits=4,
            quant_type=QuantType.INT4,
            scale_type=ScaleType.PER_GROUP
        )
        super().__init__(config)
        self.actorder = actorder
        self.percdamp = percdamp

    def quantize_weight(
        self,
        weight: np.ndarray,
        name: str = "",
        hessian: np.ndarray = None
    ) -> QuantizedTensor:
        """Quantize using GPTQ algorithm."""
        weight = self._validate_weight(weight)
        W = weight.copy()
        out_features, in_features = W.shape

        # Initialize Hessian if not provided
        if hessian is None:
            H = np.eye(in_features)
        else:
            H = hessian

        # Add damping
        damp = self.percdamp * np.mean(np.diag(H))
        H = H + damp * np.eye(in_features)

        # Activation order
        if self.actorder:
            perm = np.argsort(np.diag(H))[::-1]
            W = W[:, perm]
            H = H[perm][:, perm]
        else:
            perm = np.arange(in_features)

        # Quantize column by column
        Q = np.zeros_like(W)
        group_size = self.config.group_size
        num_groups = in_features // group_size

        scales = np.zeros((out_features, num_groups))
        zeros = np.zeros((out_features, num_groups))

        for g in range(num_groups):
            start = g * group_size
            end = start + group_size

            # Get group weight
            group_w = W[:, start:end]

            # Compute scales
            if self.config.symmetric:
                max_val = np.abs(group_w).max(axis=1)
                scales[:, g] = max_val / 7.0
                scales[:, g] = np.where(scales[:, g] == 0, 1, scales[:, g])
            else:
                min_val = group_w.min(axis=1)
                max_val = group_w.max(axis=1)
                scales[:, g] = (max_val - min_val) / 15.0
                scales[:, g] = np.where(scales[:, g] == 0, 1, scales[:, g])
                zeros[:, g] = np.round(-min_val / scales[:, g])

            # Quantize
            for col in range(start, end):
                w = W[:, col]

                # Quantize
                if self.config.symmetric:
                    q = np.clip(np.round(w / scales[:, g]), -8, 7)
                else:
                    q = np.clip(np.round(w / scales[:, g]) + zeros[:, g], 0, 15)

                Q[:, col] = q

                # Error feedback (simplified)
                err = w - (q - (zeros[:, g] if not self.config.symmetric else 0)) * scales[:, g]
                if col < in_features - 1:
                    W[:, col + 1:] -= np.outer(err, H[col, col + 1:]) / H[col, col]

        # Restore order
        inv_perm = np.argsort(perm)
        Q = Q[:, inv_perm]

        # Pack INT4
        from ..core.types import pack_int4
        packed = pack_int4(Q.astype(np.int8))

        return QuantizedTensor(
            data=packed,
            scales=scales,
            zeros=zeros if not self.config.symmetric else None,
            bits=4,
            scale_type=ScaleType.PER_GROUP,
            group_size=group_size,
            original_shape=weight.shape,
            config=self.config
        )

    def _compute_optimal_order(self, hessian: np.ndarray) -> np.ndarray:
        """Compute optimal ordering based on Hessian diagonal.

        Args:
            hessian: Hessian matrix

        Returns:
            Optimal column ordering (descending by diagonal values)
        """
        diag = np.diag(hessian)
        return np.argsort(diag)[::-1]


class AWQQuantizer(Quantizer):
    """
    Activation-aware Weight Quantization (AWQ).

    Protects salient weights based on activation distribution.
    """

    def __init__(
        self,
        config: QuantConfig = None,
        w_bit: int = 4,
        auto_scale: bool = True
    ):
        config = config or QuantConfig(
            bits=4,
            quant_type=QuantType.INT4,
            scale_type=ScaleType.PER_GROUP
        )
        super().__init__(config)
        self.w_bit = w_bit
        self.auto_scale = auto_scale

    def quantize_weight(
        self,
        weight: np.ndarray,
        name: str = "",
        act_scales: np.ndarray = None,
        activation_scale: np.ndarray = None  # Alias for act_scales
    ) -> QuantizedTensor:
        """Quantize using AWQ with activation awareness.

        Args:
            weight: Weight tensor to quantize
            name: Optional layer name
            act_scales: Activation scales (per-channel)
            activation_scale: Alias for act_scales (for compatibility)
        """
        weight = self._validate_weight(weight)
        out_features, in_features = weight.shape

        # Support both parameter names
        if activation_scale is not None and act_scales is None:
            act_scales = activation_scale

        # Compute activation scales if not provided
        if act_scales is None:
            act_scales = np.ones(in_features)

        # Find optimal scales
        if self.auto_scale:
            s = self._search_optimal_scales(weight, act_scales)
        else:
            s = np.ones(in_features)

        # Apply scales to weight
        scaled_weight = weight * s.reshape(1, -1)

        # Quantize
        from ..core.types import quantize_int4
        qdata, scales, zeros = quantize_int4(
            scaled_weight,
            self.config.scale_type,
            self.config.group_size,
            self.config.symmetric
        )

        return QuantizedTensor(
            data=qdata,
            scales=scales,
            zeros=zeros,
            bits=4,
            scale_type=self.config.scale_type,
            group_size=self.config.group_size,
            original_shape=weight.shape,
            activation_scale=act_scales,  # Store for inspection
            config=self.config
        )

    def _search_optimal_scales(
        self,
        weight: np.ndarray,
        act_scales: np.ndarray
    ) -> np.ndarray:
        """Search for optimal weight scales."""
        in_features = weight.shape[1]
        best_scales = np.ones(in_features)

        # Grid search for scale factors
        for alpha in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            # Scales proportional to activation magnitude
            s = np.power(act_scales, alpha)
            s = s / s.mean()  # Normalize

            # Compute quantization error
            scaled = weight * s.reshape(1, -1)
            quant, scales, _ = quantize_int4(
                scaled,
                ScaleType.PER_GROUP,
                self.config.group_size,
                True
            )

            # Dequantize
            from ..core.types import unpack_int4
            dequant = unpack_int4(quant)
            num_groups = dequant.shape[1] // self.config.group_size
            dequant = dequant.reshape(weight.shape[0], num_groups, -1)
            dequant = dequant * scales[:, :, np.newaxis]
            dequant = dequant.reshape(weight.shape)

            # Error
            err = np.mean((scaled - dequant) ** 2)

            # Update best
            if err < np.mean((weight * best_scales.reshape(1, -1) - dequant) ** 2):
                best_scales = s

        return best_scales

    def _search_optimal_scale(
        self,
        weight: np.ndarray,
        act_scales: np.ndarray
    ) -> np.ndarray:
        """Alias for _search_optimal_scales for API compatibility."""
        return self._search_optimal_scales(weight, act_scales)

    def _compute_activation_scale(
        self,
        calibration_data: List[np.ndarray]
    ) -> np.ndarray:
        """Compute activation scales from calibration data.

        Args:
            calibration_data: List of activation samples

        Returns:
            Per-channel activation scales
        """
        if not calibration_data:
            raise ValueError("Calibration data is empty")

        # Concatenate all samples
        all_acts = np.concatenate(calibration_data, axis=0)

        # Compute per-channel max absolute value
        scales = np.abs(all_acts).max(axis=0)

        # Avoid zeros
        scales = np.where(scales == 0, 1.0, scales)

        return scales


class SmoothQuantQuantizer(Quantizer):
    """
    SmoothQuant for activation/weight co-quantization.

    Smooths difficulty between activations and weights.
    """

    def __init__(self, config: QuantConfig = None, alpha: float = 0.5):
        config = config or QuantConfig(bits=8, quant_type=QuantType.INT8)
        super().__init__(config)
        self.alpha = alpha

    def quantize_weight(
        self,
        weight: np.ndarray,
        name: str = "",
        act_scales: np.ndarray = None
    ) -> QuantizedTensor:
        """Quantize with smoothing."""
        weight = self._validate_weight(weight)
        if act_scales is None:
            # Default to per-channel max
            act_scales = np.ones(weight.shape[1])

        # Compute smoothing scales
        w_scales = np.abs(weight).max(axis=0)
        s = np.power(act_scales, self.alpha) / np.power(w_scales, 1 - self.alpha)
        s = np.where(np.isnan(s) | np.isinf(s), 1.0, s)

        # Apply smoothing
        smoothed = weight / s.reshape(1, -1)

        # Quantize
        qdata, scales, zeros = quantize_int8(
            smoothed,
            self.config.scale_type,
            self.config.group_size,
            self.config.symmetric
        )

        return QuantizedTensor(
            data=qdata,
            scales=scales,
            zeros=zeros,
            bits=8,
            scale_type=self.config.scale_type,
            group_size=self.config.group_size,
            original_shape=weight.shape,
            config=self.config
        )
