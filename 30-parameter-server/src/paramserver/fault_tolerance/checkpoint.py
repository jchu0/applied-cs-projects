"""Checkpointing system for fault tolerance.

.. warning::
   **SECURITY: checkpoints are serialized with** :mod:`pickle`. Loading a
   checkpoint calls :func:`pickle.load`, which executes arbitrary code embedded
   in the file. Only ever load checkpoint files you produced yourself or that
   come from a fully trusted source. Never load a ``.pkl`` file received over an
   untrusted channel. As a guard, :meth:`CheckpointManager.load_checkpoint`
   refuses paths outside its ``storage_path`` directory unless the caller
   explicitly opts in with ``allow_external=True``.
"""

import asyncio
import json
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    """A checkpoint of training state.

    Attributes:
        checkpoint_id: Unique identifier for this checkpoint.
        epoch: Training epoch when checkpoint was created.
        global_step: Global training step.
        params: Dictionary mapping parameter names to values.
        optimizer_state: Optimizer state for each server shard.
        worker_clocks: Clock values for each worker.
        timestamp: Time when checkpoint was created.
        metadata: Additional metadata.
    """
    checkpoint_id: str
    epoch: int
    global_step: int
    params: Dict[str, np.ndarray]
    optimizer_state: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    worker_clocks: Dict[int, int] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CheckpointManager:
    """Manages distributed checkpointing with rotation.

    Handles saving, loading, and rotating checkpoints. Supports both
    synchronous and asynchronous save operations.

    .. warning::
       **SECURITY:** checkpoints are pickled, so loading one executes arbitrary
       code from the file. Only load trusted files. Loads are restricted to
       ``storage_path`` by default (see :meth:`load_checkpoint`).

    Attributes:
        storage_path: Directory for storing checkpoints.
        checkpoint_interval: Steps between checkpoints.
        max_checkpoints: Maximum checkpoints to keep (older ones deleted).
    """

    def __init__(
        self,
        storage_path: str,
        checkpoint_interval: int = 1000,
        max_checkpoints: int = 5,
    ):
        """Initialize checkpoint manager.

        Args:
            storage_path: Directory to store checkpoints.
            checkpoint_interval: Steps between automatic checkpoints.
            max_checkpoints: Maximum number of checkpoints to retain.
        """
        self.storage_path = Path(storage_path)
        self.checkpoint_interval = checkpoint_interval
        self.max_checkpoints = max_checkpoints

        # Track saved checkpoints
        self._checkpoints: List[str] = []

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Load existing checkpoint list
        self._scan_existing_checkpoints()

    def _scan_existing_checkpoints(self) -> None:
        """Scan storage directory for existing checkpoints."""
        self._checkpoints = []
        if not self.storage_path.exists():
            return

        for path in sorted(self.storage_path.glob("checkpoint_*.pkl")):
            self._checkpoints.append(str(path))

    async def save_checkpoint(
        self,
        params: Dict[str, np.ndarray],
        epoch: int,
        global_step: int,
        optimizer_state: Optional[Dict[int, Dict[str, Any]]] = None,
        worker_clocks: Optional[Dict[int, int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save a checkpoint asynchronously.

        Args:
            params: Model parameters.
            epoch: Current epoch.
            global_step: Current global step.
            optimizer_state: Optional optimizer state per shard.
            worker_clocks: Optional worker clock values.
            metadata: Optional additional metadata.

        Returns:
            Path to saved checkpoint.
        """
        async with self._lock:
            checkpoint_id = f"checkpoint_{global_step}_{int(time.time())}"
            checkpoint = Checkpoint(
                checkpoint_id=checkpoint_id,
                epoch=epoch,
                global_step=global_step,
                params=params,
                optimizer_state=optimizer_state or {},
                worker_clocks=worker_clocks or {},
                metadata=metadata or {},
            )

            checkpoint_path = self.storage_path / f"{checkpoint_id}.pkl"

            # Save asynchronously
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._save_to_file,
                checkpoint,
                str(checkpoint_path),
            )

            self._checkpoints.append(str(checkpoint_path))

            # Rotate old checkpoints
            await self._rotate_checkpoints()

            logger.info(
                "Saved checkpoint %s (epoch=%d, step=%d, %d params)",
                checkpoint_id, epoch, global_step, len(params),
            )
            return str(checkpoint_path)

    def _save_to_file(self, checkpoint: Checkpoint, path: str) -> None:
        """Save checkpoint to file (blocking)."""
        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)

        # Also save metadata as JSON for easy inspection
        metadata_path = path.replace(".pkl", "_meta.json")
        meta = {
            "checkpoint_id": checkpoint.checkpoint_id,
            "epoch": checkpoint.epoch,
            "global_step": checkpoint.global_step,
            "timestamp": checkpoint.timestamp,
            "num_params": len(checkpoint.params),
            "param_names": list(checkpoint.params.keys()),
            "worker_clocks": checkpoint.worker_clocks,
            "metadata": checkpoint.metadata,
        }
        with open(metadata_path, "w") as f:
            json.dump(meta, f, indent=2)

    async def load_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        allow_external: bool = False,
    ) -> Optional[Checkpoint]:
        """Load a checkpoint.

        .. warning::
           **SECURITY:** checkpoints are unpickled, which executes arbitrary code
           embedded in the file. Only load files you trust. To reduce this
           footgun, loads are restricted to this manager's ``storage_path`` (the
           documented trusted directory). Loading a path outside it raises
           :class:`ValueError` unless you explicitly pass ``allow_external=True``
           to confirm you trust the file.

        Args:
            checkpoint_path: Path to checkpoint. If None, loads latest.
            allow_external: Permit loading a path outside ``storage_path``. Only
                set this when you fully trust the source of the file.

        Returns:
            Loaded Checkpoint or None if not found.

        Raises:
            ValueError: If ``checkpoint_path`` is outside ``storage_path`` and
                ``allow_external`` is False.
        """
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()

        if checkpoint_path is None or not Path(checkpoint_path).exists():
            logger.warning("No checkpoint found to load (path=%s)", checkpoint_path)
            return None

        if not allow_external and not self._is_within_storage(checkpoint_path):
            raise ValueError(
                f"Refusing to load checkpoint {checkpoint_path!r}: it is outside the "
                f"trusted storage directory {str(self.storage_path)!r}. Checkpoints "
                "are unpickled (arbitrary code execution); pass allow_external=True "
                "only if you fully trust this file."
            )

        checkpoint = await asyncio.get_event_loop().run_in_executor(
            None,
            self._load_from_file,
            checkpoint_path,
        )
        logger.info(
            "Loaded checkpoint from %s (step=%d)",
            checkpoint_path, checkpoint.global_step,
        )
        return checkpoint

    def _is_within_storage(self, path: str) -> bool:
        """Return True if ``path`` resolves to a file inside ``storage_path``.

        Used to guard :meth:`load_checkpoint` against loading (and thus
        unpickling) files from arbitrary locations. Resolves both paths so
        ``..`` traversal and symlinks cannot escape the trusted directory.
        """
        try:
            base = self.storage_path.resolve()
            target = Path(path).resolve()
        except OSError:  # pragma: no cover - resolution failure is treated as outside
            return False
        return target == base or base in target.parents

    def _load_from_file(self, path: str) -> Checkpoint:
        """Load checkpoint from file (blocking).

        .. warning::
           **SECURITY:** uses :func:`pickle.load`, which executes arbitrary code
           from the file. Only call on trusted files. Public loads go through
           :meth:`load_checkpoint`, which restricts paths to ``storage_path``.
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    async def _rotate_checkpoints(self) -> None:
        """Remove old checkpoints to stay within max_checkpoints."""
        while len(self._checkpoints) > self.max_checkpoints:
            oldest = self._checkpoints.pop(0)
            await self._delete_checkpoint(oldest)

    async def _delete_checkpoint(self, path: str) -> None:
        """Delete a checkpoint file."""
        await asyncio.get_event_loop().run_in_executor(
            None,
            self._delete_file,
            path,
        )

    def _delete_file(self, path: str) -> None:
        """Delete file (blocking)."""
        try:
            os.remove(path)
            # Also remove metadata file
            meta_path = path.replace(".pkl", "_meta.json")
            if os.path.exists(meta_path):
                os.remove(meta_path)
        except OSError:
            pass

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to the latest checkpoint.

        Returns:
            Path to latest checkpoint or None.
        """
        if not self._checkpoints:
            return None
        return self._checkpoints[-1]

    def get_all_checkpoints(self) -> List[str]:
        """Get list of all checkpoint paths.

        Returns:
            List of checkpoint paths, oldest first.
        """
        return self._checkpoints.copy()

    def should_checkpoint(self, global_step: int) -> bool:
        """Check if a checkpoint should be created.

        Args:
            global_step: Current global step.

        Returns:
            True if checkpoint should be created.
        """
        return global_step > 0 and global_step % self.checkpoint_interval == 0

    async def cleanup(self) -> None:
        """Remove all checkpoints."""
        async with self._lock:
            for path in self._checkpoints:
                await self._delete_checkpoint(path)
            self._checkpoints.clear()

    def get_checkpoint_info(self, path: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a checkpoint without loading full data.

        Args:
            path: Checkpoint path.

        Returns:
            Checkpoint metadata dict or None.
        """
        meta_path = path.replace(".pkl", "_meta.json")
        if not os.path.exists(meta_path):
            return None

        with open(meta_path, "r") as f:
            return json.load(f)
