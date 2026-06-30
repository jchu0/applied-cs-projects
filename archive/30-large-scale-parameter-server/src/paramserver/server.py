"""Main parameter server implementation."""

import asyncio
import time
import logging
from typing import Any
import numpy as np

from .schemas import (
    Parameter,
    Gradient,
    ServerNode,
    WorkerNode,
    ConsistencyModel,
    AggregationConfig,
    PartitionStrategy,
    NodeStatus,
    generate_id,
)
from .storage import ParameterStore, ParameterPartitioner, ShardManager
from .coordination import SyncManager, StalenessTracker
from .aggregation import GradientAggregator, AsyncAggregator
from .communication import MessageHandler

logger = logging.getLogger(__name__)


class ParameterServer:
    """Large-scale parameter server with sharding and aggregation."""

    def __init__(
        self,
        node_id: str = None,
        host: str = "localhost",
        port: int = 5000,
        num_shards: int = 4,
        consistency_model: ConsistencyModel = ConsistencyModel.BSP,
        aggregation_config: AggregationConfig = None,
        learning_rate: float = 0.01
    ):
        """Initialize parameter server.

        Args:
            node_id: Server node ID
            host: Server host
            port: Server port
            num_shards: Number of parameter shards
            consistency_model: Consistency model
            aggregation_config: Aggregation configuration
            learning_rate: Learning rate for updates
        """
        self.node_id = node_id or generate_id()
        self.host = host
        self.port = port
        self.learning_rate = learning_rate

        # Storage
        self.store = ParameterStore()
        self.shard_manager = ShardManager(num_shards)
        self.partitioner = ParameterPartitioner(
            PartitionStrategy(num_shards=num_shards)
        )

        # Coordination
        self.sync_manager = SyncManager(consistency_model)
        self.staleness_tracker = StalenessTracker()
        self.consistency_model = consistency_model

        # Aggregation
        if consistency_model == ConsistencyModel.ASP:
            self.aggregator = AsyncAggregator(aggregation_config)
        else:
            self.aggregator = GradientAggregator(aggregation_config)

        # Communication
        self.message_handler = MessageHandler(self)

        # Workers
        self._workers: dict[str, WorkerNode] = {}
        self._expected_workers = 0

        # Metrics
        self._pull_count = 0
        self._push_count = 0
        self._total_bytes_transferred = 0

    async def initialize_parameter(
        self,
        name: str,
        shape: tuple,
        initializer: str = "zeros"
    ) -> Parameter:
        """Initialize a parameter.

        Args:
            name: Parameter name
            shape: Parameter shape
            initializer: Initialization method

        Returns:
            Initialized parameter
        """
        if initializer == "zeros":
            data = np.zeros(shape, dtype=np.float32)
        elif initializer == "ones":
            data = np.ones(shape, dtype=np.float32)
        elif initializer == "random":
            data = np.random.randn(*shape).astype(np.float32) * 0.01
        elif initializer == "xavier":
            fan_in = shape[0] if len(shape) > 0 else 1
            fan_out = shape[1] if len(shape) > 1 else 1
            std = np.sqrt(2.0 / (fan_in + fan_out))
            data = np.random.randn(*shape).astype(np.float32) * std
        else:
            data = np.zeros(shape, dtype=np.float32)

        parameter = Parameter(
            name=name,
            shape=shape,
            dtype="float32",
            data=data,
            version=0
        )

        # Store in appropriate shard
        shard_id = self.partitioner.get_shard(name)
        parameter.shard_id = shard_id
        self.shard_manager.store(shard_id, parameter)

        # Also store in main store
        await self.store.set(parameter)

        logger.info(f"Initialized parameter {name} with shape {shape}")
        return parameter

    async def pull(
        self,
        worker_id: str,
        parameter_names: list[str]
    ) -> dict[str, Parameter]:
        """Pull parameters for worker.

        Args:
            worker_id: Worker ID
            parameter_names: Parameters to pull

        Returns:
            Dictionary of parameters
        """
        self._pull_count += 1

        # Check staleness for SSP/bounded
        if self.consistency_model in [ConsistencyModel.SSP, ConsistencyModel.BOUNDED]:
            for name in parameter_names:
                if self.staleness_tracker.should_block_push(worker_id, name):
                    logger.warning(f"Worker {worker_id} too stale for {name}")
                    # Wait for sync
                    await asyncio.sleep(0.1)

        # Get parameters
        parameters = await self.store.get_many(parameter_names)

        # Record versions for staleness tracking
        for name, param in parameters.items():
            self.staleness_tracker.record_worker_pull(
                worker_id, name, param.version
            )

            # Track bytes
            if param.data is not None:
                self._total_bytes_transferred += param.data.nbytes

        return parameters

    async def push(
        self,
        worker_id: str,
        gradients: list[Gradient],
        iteration: int
    ) -> dict[str, Any]:
        """Push gradients from worker.

        Args:
            worker_id: Worker ID
            gradients: Gradients to push
            iteration: Current iteration

        Returns:
            Push result
        """
        self._push_count += 1

        results = {}

        for gradient in gradients:
            # Track bytes
            self._total_bytes_transferred += gradient.data.nbytes

            if self.consistency_model == ConsistencyModel.ASP:
                # Async mode: apply immediately
                aggregated = await self.aggregator.aggregate_immediate(gradient)
                param = await self.store.apply_gradient(
                    gradient.name,
                    aggregated,
                    self.learning_rate
                )
                results[gradient.name] = {
                    "applied": True,
                    "version": param.version if param else -1
                }
            else:
                # Sync mode: add to pending
                await self.aggregator.add_gradient(gradient)
                pending = await self.aggregator.get_pending_count(gradient.name)
                results[gradient.name] = {
                    "pending": pending,
                    "expected": self._expected_workers
                }

                # Check if should aggregate
                if pending >= self._expected_workers:
                    aggregated = await self.aggregator.aggregate(
                        gradient.name,
                        self._expected_workers
                    )
                    if aggregated is not None:
                        param = await self.store.apply_gradient(
                            gradient.name,
                            aggregated,
                            self.learning_rate
                        )
                        # Update staleness tracker
                        self.staleness_tracker.update_parameter_version(
                            gradient.name,
                            param.version
                        )
                        results[gradient.name]["applied"] = True
                        results[gradient.name]["version"] = param.version

        return results

    async def barrier(
        self,
        worker_id: str,
        iteration: int
    ) -> bool:
        """Worker arrives at barrier.

        Args:
            worker_id: Worker ID
            iteration: Iteration number

        Returns:
            True when barrier complete
        """
        # Create barrier if needed
        await self.sync_manager.create_barrier(
            iteration,
            self._expected_workers
        )

        # Arrive at barrier
        complete = await self.sync_manager.arrive_at_barrier(
            worker_id,
            iteration
        )

        if not complete:
            # Wait for barrier
            complete = await self.sync_manager.wait_for_barrier(
                iteration,
                timeout=60.0
            )

        return complete

    def register_worker(
        self,
        worker_id: str,
        host: str = "localhost",
        port: int = 6000
    ):
        """Register a worker.

        Args:
            worker_id: Worker ID
            host: Worker host
            port: Worker port
        """
        worker = WorkerNode(
            worker_id=worker_id,
            host=host,
            port=port
        )
        self._workers[worker_id] = worker
        self._expected_workers = len(self._workers)
        logger.info(f"Registered worker {worker_id}")

    def deregister_worker(self, worker_id: str):
        """Deregister a worker.

        Args:
            worker_id: Worker ID
        """
        self._workers.pop(worker_id, None)
        self._expected_workers = len(self._workers)
        logger.info(f"Deregistered worker {worker_id}")

    async def worker_heartbeat(self, worker_id: str) -> bool:
        """Handle worker heartbeat.

        Args:
            worker_id: Worker ID

        Returns:
            True if successful
        """
        if worker_id in self._workers:
            self._workers[worker_id].last_heartbeat = time.time()
            return True
        return False

    def get_stats(self) -> dict[str, Any]:
        """Get server statistics.

        Returns:
            Statistics dictionary
        """
        store_stats = self.store.get_stats()

        return {
            "node_id": self.node_id,
            "consistency_model": self.consistency_model.value,
            "num_workers": len(self._workers),
            "global_iteration": self.sync_manager.get_global_iteration(),
            "pull_count": self._pull_count,
            "push_count": self._push_count,
            "bytes_transferred_mb": self._total_bytes_transferred / (1024 * 1024),
            **store_stats
        }


def create_server(
    num_shards: int = 4,
    consistency_model: str = "bsp",
    learning_rate: float = 0.01
) -> ParameterServer:
    """Create a parameter server.

    Args:
        num_shards: Number of shards
        consistency_model: Consistency model name
        learning_rate: Learning rate

    Returns:
        Configured server
    """
    model_map = {
        "bsp": ConsistencyModel.BSP,
        "asp": ConsistencyModel.ASP,
        "ssp": ConsistencyModel.SSP,
        "bounded": ConsistencyModel.BOUNDED
    }

    return ParameterServer(
        num_shards=num_shards,
        consistency_model=model_map.get(consistency_model, ConsistencyModel.BSP),
        learning_rate=learning_rate
    )
