"""Training worker implementation."""

import asyncio
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
import numpy as np

from paramserver.schemas import WorkerInfo, WorkerStatus
from paramserver.server.cluster import ParameterServerCluster


# Type aliases
GradientFn = Callable[[Dict[str, np.ndarray], Any], Tuple[Dict[str, np.ndarray], float]]
DataIterator = Iterator[Any]


class Worker:
    """Training worker that computes gradients and communicates with PS cluster.

    A Worker pulls parameters from the parameter server cluster, computes
    gradients using a provided gradient function, and pushes gradients back
    to update the model.

    Attributes:
        worker_id: Unique identifier for this worker.
        ps_cluster: Parameter server cluster to communicate with.
        info: WorkerInfo tracking status and statistics.
    """

    def __init__(
        self,
        worker_id: int,
        ps_cluster: ParameterServerCluster,
        param_names: Optional[List[str]] = None,
        compute_gradients: Optional[GradientFn] = None,
    ):
        """Initialize worker.

        Args:
            worker_id: Unique identifier for this worker.
            ps_cluster: Parameter server cluster for push/pull operations.
            param_names: List of parameter names to train. If None, will be
                determined from the first pull.
            compute_gradients: Function to compute gradients given parameters
                and a data batch. Signature: (params, batch) -> (gradients, loss).
        """
        self.worker_id = worker_id
        self.ps_cluster = ps_cluster
        self._param_names = param_names or []
        self._compute_gradients = compute_gradients

        # Worker state
        self.info = WorkerInfo(worker_id=worker_id)
        self._clock = 0
        self._local_params: Dict[str, np.ndarray] = {}

        # Training state
        self._running = False
        self._total_loss = 0.0
        self._loss_count = 0

    @property
    def param_names(self) -> List[str]:
        """Get parameter names."""
        return self._param_names

    @param_names.setter
    def param_names(self, names: List[str]) -> None:
        """Set parameter names."""
        self._param_names = names

    @property
    def clock(self) -> int:
        """Get current logical clock."""
        return self._clock

    def set_gradient_fn(self, fn: GradientFn) -> None:
        """Set the gradient computation function.

        Args:
            fn: Function with signature (params, batch) -> (gradients, loss).
        """
        self._compute_gradients = fn

    async def pull_params(self) -> Dict[str, np.ndarray]:
        """Pull latest parameters from parameter servers.

        Returns:
            Dictionary mapping parameter names to values.
        """
        self.info.status = WorkerStatus.PULLING

        params = await self.ps_cluster.pull(
            self._param_names,
            self.worker_id,
        )

        self._local_params = params
        self.info.status = WorkerStatus.IDLE
        self.info.update_heartbeat()

        return params

    async def push_gradients(
        self,
        gradients: Dict[str, np.ndarray],
    ) -> int:
        """Push gradients to parameter servers.

        Args:
            gradients: Dictionary mapping parameter names to gradients.

        Returns:
            Number of updates applied.
        """
        self.info.status = WorkerStatus.PUSHING

        applied = await self.ps_cluster.push(
            gradients,
            self.worker_id,
            self._clock,
        )

        self._clock += 1
        self.info.clock = self._clock
        self.info.total_updates += 1
        self.info.status = WorkerStatus.IDLE
        self.info.update_heartbeat()

        return applied

    async def train_step(self, batch: Any) -> float:
        """Execute one training step.

        Args:
            batch: Training data batch.

        Returns:
            Loss value for this step.

        Raises:
            ValueError: If gradient function is not set.
        """
        if self._compute_gradients is None:
            raise ValueError("Gradient function not set. Call set_gradient_fn first.")

        self.info.status = WorkerStatus.TRAINING

        # 1. Pull latest parameters
        params = await self.pull_params()

        # 2. Compute gradients
        gradients, loss = self._compute_gradients(params, batch)

        # 3. Push gradients
        await self.push_gradients(gradients)

        # Update stats
        self.info.total_steps += 1
        self._total_loss += loss
        self._loss_count += 1

        return loss

    async def train_loop(
        self,
        data_iterator: DataIterator,
        num_steps: Optional[int] = None,
        log_interval: int = 100,
        callback: Optional[Callable[[int, float], None]] = None,
    ) -> Dict[str, Any]:
        """Run training loop for specified steps.

        Args:
            data_iterator: Iterator yielding training batches.
            num_steps: Number of steps to train. If None, trains until
                data_iterator is exhausted.
            log_interval: Steps between logging.
            callback: Optional callback(step, loss) after each step.

        Returns:
            Dictionary with training statistics.
        """
        self._running = True
        step = 0
        total_loss = 0.0

        try:
            while self._running:
                # Check step limit
                if num_steps is not None and step >= num_steps:
                    break

                # Get batch
                try:
                    batch = next(data_iterator)
                except StopIteration:
                    break

                # Train step
                loss = await self.train_step(batch)
                total_loss += loss
                step += 1

                # Callback
                if callback:
                    callback(step, loss)

        finally:
            self._running = False

        return {
            "steps": step,
            "avg_loss": total_loss / step if step > 0 else 0.0,
            "final_clock": self._clock,
        }

    async def async_train_step(self, batch: Any) -> float:
        """Execute async training step (don't wait for push completion).

        Args:
            batch: Training data batch.

        Returns:
            Loss value.
        """
        if self._compute_gradients is None:
            raise ValueError("Gradient function not set")

        self.info.status = WorkerStatus.TRAINING

        # Pull in background while preparing batch
        pull_task = asyncio.create_task(self.pull_params())

        # Wait for params
        params = await pull_task

        # Compute gradients
        gradients, loss = self._compute_gradients(params, batch)

        # Push without waiting (fire and forget)
        asyncio.create_task(self._async_push(gradients))

        self.info.total_steps += 1
        return loss

    async def _async_push(self, gradients: Dict[str, np.ndarray]) -> None:
        """Push gradients asynchronously."""
        try:
            await self.push_gradients(gradients)
        except Exception:
            # Log error but don't fail
            pass

    def stop(self) -> None:
        """Stop the training loop."""
        self._running = False

    @property
    def avg_loss(self) -> float:
        """Get average loss over all steps."""
        if self._loss_count == 0:
            return 0.0
        return self._total_loss / self._loss_count

    @property
    def stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            "worker_id": self.worker_id,
            "clock": self._clock,
            "total_steps": self.info.total_steps,
            "total_updates": self.info.total_updates,
            "avg_loss": self.avg_loss,
            "status": self.info.status.value,
        }


class MockGradientComputer:
    """Mock gradient computer for testing.

    Simulates gradient computation with random gradients.
    """

    def __init__(
        self,
        param_shapes: Dict[str, Tuple[int, ...]],
        gradient_scale: float = 0.01,
    ):
        """Initialize mock gradient computer.

        Args:
            param_shapes: Dictionary mapping param names to shapes.
            gradient_scale: Scale of random gradients.
        """
        self.param_shapes = param_shapes
        self.gradient_scale = gradient_scale
        self._step = 0

    def __call__(
        self,
        params: Dict[str, np.ndarray],
        batch: Any,
    ) -> Tuple[Dict[str, np.ndarray], float]:
        """Compute mock gradients.

        Args:
            params: Current parameters.
            batch: Data batch (ignored).

        Returns:
            Tuple of (gradients, loss).
        """
        gradients = {}
        for name, shape in self.param_shapes.items():
            if name in params:
                gradients[name] = np.random.randn(*shape) * self.gradient_scale

        # Simulate decreasing loss
        loss = 1.0 / (1.0 + 0.01 * self._step)
        self._step += 1

        return gradients, loss


class DataBatchGenerator:
    """Simple batch generator for testing."""

    def __init__(
        self,
        num_batches: int = 1000,
        batch_size: int = 32,
        feature_dim: int = 64,
    ):
        """Initialize batch generator.

        Args:
            num_batches: Total number of batches to generate.
            batch_size: Size of each batch.
            feature_dim: Feature dimension.
        """
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.feature_dim = feature_dim
        self._current = 0

    def __iter__(self):
        self._current = 0
        return self

    def __next__(self):
        if self._current >= self.num_batches:
            raise StopIteration

        self._current += 1
        return {
            "features": np.random.randn(self.batch_size, self.feature_dim),
            "labels": np.random.randint(0, 10, size=self.batch_size),
        }
