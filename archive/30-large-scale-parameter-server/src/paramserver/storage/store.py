"""Parameter storage implementation."""

import numpy as np
import asyncio
from typing import Any
import logging

from ..schemas import Parameter, Gradient

logger = logging.getLogger(__name__)


class ParameterStore:
    """In-memory parameter storage with versioning."""

    def __init__(self):
        """Initialize parameter store."""
        self._parameters: dict[str, Parameter] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._history: dict[str, list[np.ndarray]] = {}
        self._max_history = 5

    async def get_lock(self, name: str) -> asyncio.Lock:
        """Get lock for parameter.

        Args:
            name: Parameter name

        Returns:
            Lock for parameter
        """
        if name not in self._locks:
            self._locks[name] = asyncio.Lock()
        return self._locks[name]

    async def set(self, parameter: Parameter):
        """Set parameter value.

        Args:
            parameter: Parameter to store
        """
        lock = await self.get_lock(parameter.name)
        async with lock:
            # Store old version in history
            if parameter.name in self._parameters:
                old_param = self._parameters[parameter.name]
                if old_param.data is not None:
                    if parameter.name not in self._history:
                        self._history[parameter.name] = []
                    self._history[parameter.name].append(old_param.data.copy())

                    # Limit history size
                    if len(self._history[parameter.name]) > self._max_history:
                        self._history[parameter.name].pop(0)

            self._parameters[parameter.name] = parameter

    async def get(self, name: str) -> Parameter | None:
        """Get parameter by name.

        Args:
            name: Parameter name

        Returns:
            Parameter or None
        """
        return self._parameters.get(name)

    async def get_many(self, names: list[str]) -> dict[str, Parameter]:
        """Get multiple parameters.

        Args:
            names: Parameter names

        Returns:
            Dictionary of parameters
        """
        return {
            name: self._parameters[name]
            for name in names
            if name in self._parameters
        }

    async def update(self, name: str, data: np.ndarray) -> Parameter | None:
        """Update parameter data.

        Args:
            name: Parameter name
            data: New data

        Returns:
            Updated parameter or None
        """
        lock = await self.get_lock(name)
        async with lock:
            if name not in self._parameters:
                return None

            param = self._parameters[name]

            # Store history
            if param.data is not None:
                if name not in self._history:
                    self._history[name] = []
                self._history[name].append(param.data.copy())
                if len(self._history[name]) > self._max_history:
                    self._history[name].pop(0)

            param.data = data
            param.version += 1
            return param

    async def apply_gradient(
        self,
        name: str,
        gradient: np.ndarray,
        learning_rate: float = 0.01
    ) -> Parameter | None:
        """Apply gradient to parameter.

        Args:
            name: Parameter name
            gradient: Gradient data
            learning_rate: Learning rate

        Returns:
            Updated parameter
        """
        lock = await self.get_lock(name)
        async with lock:
            if name not in self._parameters:
                return None

            param = self._parameters[name]
            if param.data is None:
                return None

            # Store history
            if name not in self._history:
                self._history[name] = []
            self._history[name].append(param.data.copy())
            if len(self._history[name]) > self._max_history:
                self._history[name].pop(0)

            # Apply gradient
            param.data = param.data - learning_rate * gradient
            param.version += 1
            return param

    async def get_version(self, name: str) -> int:
        """Get parameter version.

        Args:
            name: Parameter name

        Returns:
            Version number
        """
        if name not in self._parameters:
            return -1
        return self._parameters[name].version

    async def rollback(self, name: str, steps: int = 1) -> bool:
        """Rollback parameter to previous version.

        Args:
            name: Parameter name
            steps: Number of steps to rollback

        Returns:
            True if successful
        """
        lock = await self.get_lock(name)
        async with lock:
            if name not in self._history:
                return False

            history = self._history[name]
            if len(history) < steps:
                return False

            # Get historical data
            for _ in range(steps):
                if history:
                    old_data = history.pop()
                    if name in self._parameters:
                        self._parameters[name].data = old_data
                        self._parameters[name].version -= 1

            return True

    def list_parameters(self) -> list[str]:
        """List all parameter names.

        Returns:
            List of parameter names
        """
        return list(self._parameters.keys())

    def get_stats(self) -> dict[str, Any]:
        """Get storage statistics.

        Returns:
            Statistics dictionary
        """
        total_size = sum(
            p.data.nbytes if p.data is not None else 0
            for p in self._parameters.values()
        )
        return {
            "num_parameters": len(self._parameters),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024)
        }
