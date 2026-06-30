"""Stochastic Gradient Descent optimizer."""

from typing import Any, Dict, Optional
import numpy as np

from paramserver.optimizer.base import UpdateEngine


class SGDEngine(UpdateEngine):
    """Stochastic Gradient Descent with optional momentum.

    Implements SGD with optional momentum and weight decay.
    The update rule with momentum is:
        v_t = momentum * v_{t-1} + gradient
        params = params - lr * v_t

    With weight decay:
        params = params - lr * (gradient + weight_decay * params)

    Attributes:
        lr: Learning rate.
        momentum: Momentum factor (default: 0.0).
        weight_decay: Weight decay coefficient (default: 0.0).
        dampening: Dampening for momentum (default: 0.0).
        nesterov: Enable Nesterov momentum (default: False).
    """

    def __init__(
        self,
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
    ):
        """Initialize SGD optimizer.

        Args:
            lr: Learning rate.
            momentum: Momentum factor.
            weight_decay: Weight decay (L2 penalty).
            dampening: Dampening for momentum.
            nesterov: Whether to use Nesterov momentum.

        Raises:
            ValueError: If learning rate is negative.
            ValueError: If momentum is negative.
            ValueError: If nesterov is True but momentum is 0.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = dampening
        self.nesterov = nesterov

        # Velocity buffer for momentum
        self._velocity: Dict[str, np.ndarray] = {}

    def apply(
        self,
        params: np.ndarray,
        gradients: np.ndarray,
        param_id: Optional[str] = None,
    ) -> np.ndarray:
        """Apply SGD update to parameters.

        Args:
            params: Current parameter values.
            gradients: Gradient values.
            param_id: Identifier for the parameter (required for momentum).

        Returns:
            Updated parameter values.
        """
        # Apply weight decay
        if self.weight_decay != 0:
            gradients = gradients + self.weight_decay * params

        # Apply momentum if specified
        if self.momentum > 0:
            if param_id is None:
                # Without param_id, can't track momentum state
                param_id = "_default"

            if param_id not in self._velocity:
                self._velocity[param_id] = np.zeros_like(params)

            v = self._velocity[param_id]

            # Update velocity
            if self.dampening > 0:
                v = self.momentum * v + (1 - self.dampening) * gradients
            else:
                v = self.momentum * v + gradients

            self._velocity[param_id] = v

            # Apply Nesterov correction if enabled
            if self.nesterov:
                update = gradients + self.momentum * v
            else:
                update = v
        else:
            update = gradients

        # Apply update
        return params - self.lr * update

    def get_state(self) -> Dict[str, Any]:
        """Get optimizer state for checkpointing."""
        return {
            "velocity": {k: v.copy() for k, v in self._velocity.items()},
            "lr": self.lr,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "dampening": self.dampening,
            "nesterov": self.nesterov,
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
        if "dampening" in state:
            self.dampening = state["dampening"]
        if "nesterov" in state:
            self.nesterov = state["nesterov"]

    def reset(self) -> None:
        """Reset momentum state."""
        self._velocity.clear()

    def set_lr(self, lr: float) -> None:
        """Update learning rate.

        Args:
            lr: New learning rate.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        self.lr = lr
