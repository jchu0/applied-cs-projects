"""Single parameter server shard implementation."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from paramserver.schemas import (
    GradientUpdate,
    ParameterMetadata,
    ServerStatus,
)
from paramserver.optimizer.base import UpdateEngine
from paramserver.consistency.base import ConsistencyModel

logger = logging.getLogger(__name__)


class ParameterServer:
    """A single parameter server shard.

    Manages a subset of model parameters and handles push/pull operations
    from workers. Applies gradient updates using the configured optimizer
    and consistency model.

    Attributes:
        shard_id: Unique identifier for this shard.
        update_engine: Optimizer for applying gradient updates.
        consistency: Consistency model for synchronization.
        status: Current server status.
    """

    def __init__(
        self,
        shard_id: int,
        update_engine: UpdateEngine,
        consistency: ConsistencyModel,
    ):
        """Initialize parameter server.

        Args:
            shard_id: Unique identifier for this shard.
            update_engine: Optimizer for gradient updates.
            consistency: Consistency model to use.
        """
        self.shard_id = shard_id
        self.update_engine = update_engine
        self.consistency = consistency
        self.status = ServerStatus.READY

        # Parameter storage
        self._params: Dict[str, np.ndarray] = {}
        self._metadata: Dict[str, ParameterMetadata] = {}

        # Gradient buffer for deferred updates
        self._gradient_buffer: Dict[str, List[GradientUpdate]] = {}

        # Locks for thread safety
        self._locks: Dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

        # Statistics
        self._total_pulls = 0
        self._total_pushes = 0

        logger.info(
            "ParameterServer shard %d ready (optimizer=%s, consistency=%s)",
            shard_id,
            type(update_engine).__name__,
            type(consistency).__name__,
        )

    async def initialize(self, params: Dict[str, np.ndarray]) -> None:
        """Initialize parameters for this shard.

        Args:
            params: Dictionary mapping parameter names to initial values.
        """
        async with self._global_lock:
            for name, value in params.items():
                self._params[name] = value.copy()
                self._metadata[name] = ParameterMetadata(
                    name=name,
                    shape=value.shape,
                    dtype=value.dtype,
                    version=0,
                )
                self._locks[name] = asyncio.Lock()
                self._gradient_buffer[name] = []

            logger.info(
                "Shard %d initialized %d parameters", self.shard_id, len(self._params)
            )

    async def pull(
        self,
        param_names: List[str],
        worker_id: int,
        include_versions: bool = True,
    ) -> Dict[str, Tuple[np.ndarray, int]]:
        """Pull parameters for a worker.

        Args:
            param_names: List of parameter names to retrieve.
            worker_id: ID of the requesting worker.
            include_versions: Whether to include version numbers.

        Returns:
            Dictionary mapping parameter names to (value, version) tuples.
        """
        result: Dict[str, Tuple[np.ndarray, int]] = {}

        for name in param_names:
            if name not in self._params:
                continue

            async with self._locks[name]:
                value = self._params[name].copy()
                version = self._metadata[name].version if include_versions else 0
                result[name] = (value, version)

        self._total_pulls += 1
        return result

    async def push(
        self,
        gradients: Dict[str, np.ndarray],
        worker_id: int,
        clock: int,
    ) -> int:
        """Push gradient updates from a worker.

        Args:
            gradients: Dictionary mapping parameter names to gradients.
            worker_id: ID of the worker sending gradients.
            clock: Worker's logical clock.

        Returns:
            Number of updates applied.
        """
        applied = 0

        for name, grad in gradients.items():
            if name not in self._params:
                continue

            async with self._locks[name]:
                # Check consistency model
                param_version = self._metadata[name].version
                if not self.consistency.can_apply(param_version, clock):
                    # Buffer the gradient for later
                    update = GradientUpdate(
                        worker_id=worker_id,
                        param_name=name,
                        gradient=grad.copy(),
                        clock=clock,
                    )
                    self._gradient_buffer[name].append(update)
                    continue

                # Apply update
                self._params[name] = self.update_engine.apply(
                    self._params[name],
                    grad,
                    param_id=f"{self.shard_id}:{name}",
                )

                # Update metadata
                meta = self._metadata[name]
                meta.version += 1
                meta.last_update_worker = worker_id
                meta.total_updates += 1
                applied += 1

                # Process any buffered gradients
                await self._process_buffer(name)

        self._total_pushes += 1
        logger.debug(
            "Shard %d push from worker %d: %d/%d updates applied (clock=%d)",
            self.shard_id, worker_id, applied, len(gradients), clock,
        )
        return applied

    async def _process_buffer(self, param_name: str) -> int:
        """Process buffered gradients for a parameter.

        Args:
            param_name: Name of the parameter.

        Returns:
            Number of buffered gradients applied.
        """
        buffer = self._gradient_buffer[param_name]
        if not buffer:
            return 0

        applied = 0
        remaining = []

        for update in buffer:
            param_version = self._metadata[param_name].version
            if self.consistency.can_apply(param_version, update.clock):
                # Apply buffered gradient
                self._params[param_name] = self.update_engine.apply(
                    self._params[param_name],
                    update.gradient,
                    param_id=f"{self.shard_id}:{param_name}",
                )

                meta = self._metadata[param_name]
                meta.version += 1
                meta.last_update_worker = update.worker_id
                meta.total_updates += 1
                applied += 1
            else:
                remaining.append(update)

        self._gradient_buffer[param_name] = remaining
        return applied

    async def get_param(self, name: str) -> Optional[np.ndarray]:
        """Get a single parameter value.

        Args:
            name: Parameter name.

        Returns:
            Parameter value or None if not found.
        """
        if name not in self._params:
            return None
        async with self._locks[name]:
            return self._params[name].copy()

    async def set_param(
        self,
        name: str,
        value: np.ndarray,
        update_version: bool = True,
    ) -> None:
        """Set a parameter value directly.

        Args:
            name: Parameter name.
            value: New parameter value.
            update_version: Whether to increment the version.
        """
        if name not in self._params:
            # Initialize new parameter
            self._params[name] = value.copy()
            self._metadata[name] = ParameterMetadata(
                name=name,
                shape=value.shape,
                dtype=value.dtype,
            )
            self._locks[name] = asyncio.Lock()
            self._gradient_buffer[name] = []
        else:
            async with self._locks[name]:
                self._params[name] = value.copy()
                if update_version:
                    self._metadata[name].version += 1

    def get_metadata(self, name: str) -> Optional[ParameterMetadata]:
        """Get metadata for a parameter.

        Args:
            name: Parameter name.

        Returns:
            Parameter metadata or None.
        """
        return self._metadata.get(name)

    def get_all_params(self) -> Dict[str, np.ndarray]:
        """Get all parameters (not thread-safe, use for checkpointing)."""
        return {name: value.copy() for name, value in self._params.items()}

    def get_all_metadata(self) -> Dict[str, ParameterMetadata]:
        """Get all parameter metadata."""
        return self._metadata.copy()

    @property
    def param_names(self) -> List[str]:
        """Get list of parameter names in this shard."""
        return list(self._params.keys())

    @property
    def total_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.size for p in self._params.values())

    @property
    def stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            "shard_id": self.shard_id,
            "num_params": len(self._params),
            "total_params": self.total_params,
            "total_pulls": self._total_pulls,
            "total_pushes": self._total_pushes,
            "buffered_updates": sum(
                len(buf) for buf in self._gradient_buffer.values()
            ),
            "status": self.status.value,
        }

    async def health_check(self) -> bool:
        """Check if the server is healthy.

        Returns:
            True if healthy, False otherwise.
        """
        return self.status in (ServerStatus.READY, ServerStatus.BUSY)
