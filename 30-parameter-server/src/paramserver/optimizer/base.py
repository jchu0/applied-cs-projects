"""Base class for parameter update engines."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class UpdateEngine(ABC):
    """Abstract base class for parameter update engines.

    An UpdateEngine applies gradient updates to parameters. Different
    implementations provide various optimization algorithms (SGD, Adam, etc.).
    """

    @abstractmethod
    def apply(
        self,
        params: np.ndarray,
        gradients: np.ndarray,
        param_id: Optional[str] = None,
    ) -> np.ndarray:
        """Apply gradient update to parameters.

        Args:
            params: Current parameter values.
            gradients: Gradient values to apply.
            param_id: Optional identifier for the parameter (used for
                maintaining per-parameter state like momentum).

        Returns:
            Updated parameter values.
        """
        pass

    def get_state(self) -> Dict[str, Any]:
        """Get optimizer state for checkpointing.

        Returns:
            Dictionary containing optimizer state.
        """
        return {}

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load optimizer state from checkpoint.

        Args:
            state: Dictionary containing optimizer state.
        """
        pass

    def reset(self) -> None:
        """Reset optimizer state."""
        pass

    @property
    def name(self) -> str:
        """Return the name of this optimizer."""
        return self.__class__.__name__
