"""Adam optimizer implementation."""

from typing import Any, Dict, Optional
import numpy as np

from paramserver.optimizer.base import UpdateEngine


class AdamEngine(UpdateEngine):
    """Adam optimizer with adaptive learning rates.

    Implements the Adam (Adaptive Moment Estimation) optimizer which
    maintains per-parameter adaptive learning rates using first and
    second moment estimates of gradients.

    The update rules are:
        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)
        params = params - lr * m_hat / (sqrt(v_hat) + eps)

    Attributes:
        lr: Learning rate.
        beta1: Exponential decay rate for first moment.
        beta2: Exponential decay rate for second moment.
        eps: Small constant for numerical stability.
        weight_decay: Weight decay (L2 penalty).
        amsgrad: Whether to use AMSGrad variant.
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ):
        """Initialize Adam optimizer.

        Args:
            lr: Learning rate.
            beta1: Coefficient for first moment estimate.
            beta2: Coefficient for second moment estimate.
            eps: Term added to denominator for numerical stability.
            weight_decay: Weight decay (L2 penalty).
            amsgrad: Whether to use AMSGrad variant.

        Raises:
            ValueError: If any hyperparameter is invalid.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2: {beta2}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad

        # State
        self._m: Dict[str, np.ndarray] = {}  # First moment
        self._v: Dict[str, np.ndarray] = {}  # Second moment
        self._v_max: Dict[str, np.ndarray] = {}  # Max second moment (AMSGrad)
        self._t: Dict[str, int] = {}  # Timestep per parameter

    def apply(
        self,
        params: np.ndarray,
        gradients: np.ndarray,
        param_id: Optional[str] = None,
    ) -> np.ndarray:
        """Apply Adam update to parameters.

        Args:
            params: Current parameter values.
            gradients: Gradient values.
            param_id: Identifier for the parameter (required for state tracking).

        Returns:
            Updated parameter values.
        """
        if param_id is None:
            param_id = "_default"

        # Initialize state if needed
        if param_id not in self._m:
            self._m[param_id] = np.zeros_like(params)
            self._v[param_id] = np.zeros_like(params)
            if self.amsgrad:
                self._v_max[param_id] = np.zeros_like(params)
            self._t[param_id] = 0

        # Increment timestep
        self._t[param_id] += 1
        t = self._t[param_id]

        # Apply weight decay (decoupled, AdamW style)
        if self.weight_decay != 0:
            params = params - self.lr * self.weight_decay * params

        # Update biased first moment estimate
        self._m[param_id] = self.beta1 * self._m[param_id] + (1 - self.beta1) * gradients

        # Update biased second raw moment estimate
        self._v[param_id] = self.beta2 * self._v[param_id] + (1 - self.beta2) * (gradients ** 2)

        # Compute bias-corrected first moment estimate
        m_hat = self._m[param_id] / (1 - self.beta1 ** t)

        # Compute bias-corrected second moment estimate
        v_hat = self._v[param_id] / (1 - self.beta2 ** t)

        # AMSGrad: use maximum of past squared gradients
        if self.amsgrad:
            self._v_max[param_id] = np.maximum(self._v_max[param_id], v_hat)
            denom = np.sqrt(self._v_max[param_id]) + self.eps
        else:
            denom = np.sqrt(v_hat) + self.eps

        # Update parameters
        return params - self.lr * m_hat / denom

    def get_state(self) -> Dict[str, Any]:
        """Get optimizer state for checkpointing."""
        return {
            "m": {k: v.copy() for k, v in self._m.items()},
            "v": {k: v.copy() for k, v in self._v.items()},
            "v_max": {k: v.copy() for k, v in self._v_max.items()},
            "t": dict(self._t),
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "amsgrad": self.amsgrad,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load optimizer state from checkpoint."""
        if "m" in state:
            self._m = {k: v.copy() for k, v in state["m"].items()}
        if "v" in state:
            self._v = {k: v.copy() for k, v in state["v"].items()}
        if "v_max" in state:
            self._v_max = {k: v.copy() for k, v in state["v_max"].items()}
        if "t" in state:
            self._t = dict(state["t"])
        if "lr" in state:
            self.lr = state["lr"]
        if "beta1" in state:
            self.beta1 = state["beta1"]
        if "beta2" in state:
            self.beta2 = state["beta2"]
        if "eps" in state:
            self.eps = state["eps"]
        if "weight_decay" in state:
            self.weight_decay = state["weight_decay"]
        if "amsgrad" in state:
            self.amsgrad = state["amsgrad"]

    def reset(self) -> None:
        """Reset optimizer state."""
        self._m.clear()
        self._v.clear()
        self._v_max.clear()
        self._t.clear()

    def set_lr(self, lr: float) -> None:
        """Update learning rate.

        Args:
            lr: New learning rate.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        self.lr = lr
