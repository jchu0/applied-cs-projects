"""Mixed precision training support.

Enables FP16/FP32 mixed precision training for faster
computation and reduced memory usage.
"""

from enum import Enum
from typing import Dict, Optional
import numpy as np


class PrecisionMode(Enum):
    """Precision mode for training."""
    FP32 = "fp32"          # Full precision
    FP16 = "fp16"          # Half precision
    MIXED = "mixed"        # FP16 forward, FP32 backward
    DYNAMIC = "dynamic"    # Auto-select based on overflow


class MixedPrecisionManager:
    """Manages mixed precision training.

    Handles conversion between FP16 and FP32, loss scaling
    to prevent underflow, and gradient overflow detection.

    Attributes:
        mode: Precision mode (FP32, FP16, MIXED, DYNAMIC).
        loss_scale: Current loss scale factor.
        dynamic_scaling: Whether to dynamically adjust loss scale.
    """

    def __init__(
        self,
        mode: PrecisionMode = PrecisionMode.MIXED,
        initial_loss_scale: float = 65536.0,
        dynamic_scaling: bool = True,
        scale_window: int = 1000,
        scale_factor: float = 2.0,
    ):
        """Initialize mixed precision manager.

        Args:
            mode: Precision mode to use.
            initial_loss_scale: Initial loss scale factor.
            dynamic_scaling: Whether to adjust loss scale dynamically.
            scale_window: Steps between scale increases.
            scale_factor: Factor to multiply/divide scale by.
        """
        self.mode = mode
        self.loss_scale = initial_loss_scale
        self.dynamic_scaling = dynamic_scaling
        self.scale_window = scale_window
        self.scale_factor = scale_factor

        self._steps_since_overflow = 0
        self._overflow_count = 0
        self._total_steps = 0

    def to_fp16(self, tensor: np.ndarray) -> np.ndarray:
        """Convert tensor to FP16.

        Args:
            tensor: Input tensor.

        Returns:
            FP16 tensor.
        """
        return tensor.astype(np.float16)

    def to_fp32(self, tensor: np.ndarray) -> np.ndarray:
        """Convert tensor to FP32.

        Args:
            tensor: Input tensor.

        Returns:
            FP32 tensor.
        """
        return tensor.astype(np.float32)

    def scale_loss(self, loss: float) -> float:
        """Scale loss for FP16 training.

        Args:
            loss: Original loss value.

        Returns:
            Scaled loss.
        """
        if self.mode == PrecisionMode.FP32:
            return loss
        return loss * self.loss_scale

    def unscale_gradients(
        self,
        gradients: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Unscale gradients after backward pass.

        Args:
            gradients: Scaled gradients.

        Returns:
            Unscaled gradients.
        """
        if self.mode == PrecisionMode.FP32:
            return gradients

        unscaled = {}
        for name, grad in gradients.items():
            unscaled[name] = grad.astype(np.float32) / self.loss_scale

        return unscaled

    def check_overflow(
        self,
        gradients: Dict[str, np.ndarray],
    ) -> bool:
        """Check for gradient overflow/underflow.

        Args:
            gradients: Gradients to check.

        Returns:
            True if overflow detected.
        """
        for grad in gradients.values():
            if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
                return True
        return False

    def update_scale(self, overflow: bool) -> None:
        """Update loss scale based on overflow status.

        Args:
            overflow: Whether overflow occurred this step.
        """
        if not self.dynamic_scaling:
            return

        self._total_steps += 1

        if overflow:
            # Reduce scale on overflow
            self.loss_scale = max(1.0, self.loss_scale / self.scale_factor)
            self._steps_since_overflow = 0
            self._overflow_count += 1
        else:
            self._steps_since_overflow += 1

            # Increase scale if no overflow for scale_window steps
            if self._steps_since_overflow >= self.scale_window:
                self.loss_scale = min(65536.0, self.loss_scale * self.scale_factor)
                self._steps_since_overflow = 0

    def step(
        self,
        gradients: Dict[str, np.ndarray],
    ) -> Optional[Dict[str, np.ndarray]]:
        """Process gradients for a training step.

        Args:
            gradients: Gradients from backward pass.

        Returns:
            Processed gradients, or None if skipped due to overflow.
        """
        # Unscale gradients
        unscaled = self.unscale_gradients(gradients)

        # Check for overflow
        overflow = self.check_overflow(unscaled)
        self.update_scale(overflow)

        if overflow:
            return None  # Skip this step

        return unscaled

    def get_precision_for_param(self, param_name: str) -> np.dtype:
        """Get precision to use for a parameter.

        Args:
            param_name: Name of the parameter.

        Returns:
            numpy dtype to use.
        """
        if self.mode == PrecisionMode.FP16:
            return np.float16
        elif self.mode == PrecisionMode.FP32:
            return np.float32
        else:
            # Mixed mode: use FP32 for parameters
            return np.float32

    def get_stats(self) -> Dict[str, float]:
        """Get mixed precision statistics.

        Returns:
            Dictionary of statistics.
        """
        return {
            "mode": self.mode.value,
            "loss_scale": self.loss_scale,
            "total_steps": self._total_steps,
            "overflow_count": self._overflow_count,
            "overflow_rate": (
                self._overflow_count / self._total_steps
                if self._total_steps > 0 else 0
            ),
            "steps_since_overflow": self._steps_since_overflow,
        }
