"""LARS optimizer implementation."""

from typing import Any, Dict, Optional
import numpy as np

from paramserver.optimizer.base import UpdateEngine


class LARSEngine(UpdateEngine):
    """Layer-wise Adaptive Rate Scaling (LARS) optimizer.

    LARS adapts the learning rate for each layer independently based on
    the ratio of parameter norms to gradient norms. This helps training
    with very large batch sizes by preventing divergence.

    The local learning rate for each layer is computed as:
        local_lr = trust_coeff * ||params|| / (||grads|| + weight_decay * ||params|| + eps)

    Then the update is:
        v_t = momentum * v_{t-1} + lr * local_lr * (grads + weight_decay * params)
        params = params - v_t

    Attributes:
        lr: Base learning rate.
        momentum: Momentum factor.
        weight_decay: Weight decay coefficient.
        trust_coefficient: Trust ratio for local learning rate.
        eps: Small constant for numerical stability.
    """

    def __init__(
        self,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 0.0001,
        trust_coefficient: float = 0.001,
        eps: float = 1e-8,
    ):
        """Initialize LARS optimizer.

        Args:
            lr: Base learning rate.
            momentum: Momentum factor.
            weight_decay: Weight decay (L2 penalty).
            trust_coefficient: Trust coefficient for scaling.
            eps: Term added for numerical stability.

        Raises:
            ValueError: If any hyperparameter is invalid.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if trust_coefficient < 0.0:
            raise ValueError(f"Invalid trust_coefficient: {trust_coefficient}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.trust_coefficient = trust_coefficient
        self.eps = eps

        # Velocity buffer for momentum
        self._velocity: Dict[str, np.ndarray] = {}

    def apply(
        self,
        params: np.ndarray,
        gradients: np.ndarray,
        param_id: Optional[str] = None,
    ) -> np.ndarray:
        """Apply LARS update to parameters.

        Args:
            params: Current parameter values.
            gradients: Gradient values.
            param_id: Identifier for the parameter.

        Returns:
            Updated parameter values.
        """
        if param_id is None:
            param_id = "_default"

        # Compute norms
        param_norm = np.linalg.norm(params)
        grad_norm = np.linalg.norm(gradients)

        # Compute local learning rate (trust ratio)
        if param_norm > 0 and grad_norm > 0:
            # Include weight decay in gradient norm calculation
            if self.weight_decay > 0:
                grad_with_decay_norm = np.linalg.norm(
                    gradients + self.weight_decay * params
                )
            else:
                grad_with_decay_norm = grad_norm

            local_lr = self.trust_coefficient * param_norm / (
                grad_with_decay_norm + self.eps
            )
        else:
            local_lr = 1.0

        # Add weight decay to gradient
        if self.weight_decay > 0:
            update = gradients + self.weight_decay * params
        else:
            update = gradients

        # Scale by local learning rate
        update = local_lr * update

        # Apply momentum
        if self.momentum > 0:
            if param_id not in self._velocity:
                self._velocity[param_id] = np.zeros_like(params)

            self._velocity[param_id] = (
                self.momentum * self._velocity[param_id] + self.lr * update
            )
            update = self._velocity[param_id]
        else:
            update = self.lr * update

        return params - update

    def get_state(self) -> Dict[str, Any]:
        """Get optimizer state for checkpointing."""
        return {
            "velocity": {k: v.copy() for k, v in self._velocity.items()},
            "lr": self.lr,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "trust_coefficient": self.trust_coefficient,
            "eps": self.eps,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load optimizer state from checkpoint."""
        if "velocity" in state:
            self._velocity = {k: v.copy() for k, v in state["velocity"].items()}
        if "lr" in state:
            self.lr = state["lr"]
        if "momentum" in state:
            self.momentum = state["momentum"]
        if "weight_decay" in state:
            self.weight_decay = state["weight_decay"]
        if "trust_coefficient" in state:
            self.trust_coefficient = state["trust_coefficient"]
        if "eps" in state:
            self.eps = state["eps"]

    def reset(self) -> None:
        """Reset optimizer state."""
        self._velocity.clear()

    def set_lr(self, lr: float) -> None:
        """Update learning rate.

        Args:
            lr: New learning rate.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        self.lr = lr

    def compute_local_lr(
        self,
        params: np.ndarray,
        gradients: np.ndarray,
    ) -> float:
        """Compute the local learning rate for a layer.

        Useful for debugging and monitoring.

        Args:
            params: Parameter values.
            gradients: Gradient values.

        Returns:
            Local learning rate multiplier.
        """
        param_norm = np.linalg.norm(params)
        grad_norm = np.linalg.norm(gradients)

        if param_norm > 0 and grad_norm > 0:
            if self.weight_decay > 0:
                grad_with_decay_norm = np.linalg.norm(
                    gradients + self.weight_decay * params
                )
            else:
                grad_with_decay_norm = grad_norm

            return self.trust_coefficient * param_norm / (
                grad_with_decay_norm + self.eps
            )
        return 1.0
