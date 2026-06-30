"""Parameter server cluster management."""

import asyncio
from typing import Any, Callable, Dict, List, Optional
import numpy as np

from paramserver.schemas import ShardInfo
from paramserver.server.parameter_server import ParameterServer
from paramserver.server.sharding import ShardingStrategy, UniformSharding
from paramserver.optimizer.base import UpdateEngine
from paramserver.optimizer.sgd import SGDEngine
from paramserver.consistency.base import ConsistencyModel
from paramserver.consistency.hogwild import HogwildConsistency


class ParameterServerCluster:
    """Manages a cluster of parameter servers.

    Coordinates multiple parameter server shards and routes requests
    to the appropriate server based on the sharding strategy.

    Attributes:
        servers: Dictionary mapping shard IDs to ParameterServer instances.
        sharding: Strategy for distributing parameters across servers.
    """

    def __init__(
        self,
        servers: Optional[List[ParameterServer]] = None,
        sharding_strategy: Optional[ShardingStrategy] = None,
    ):
        """Initialize the cluster.

        Args:
            servers: List of ParameterServer instances. If None, servers
                will be created during initialize().
            sharding_strategy: Strategy for distributing parameters.
                Defaults to UniformSharding.
        """
        self.servers: Dict[int, ParameterServer] = {}
        if servers:
            self.servers = {s.shard_id: s for s in servers}

        self.sharding = sharding_strategy or UniformSharding()
        self._initialized = False
        self._shard_info: List[ShardInfo] = []

    @classmethod
    def create(
        cls,
        num_servers: int,
        update_engine_factory: Optional[Callable[[], UpdateEngine]] = None,
        consistency_factory: Optional[Callable[[], ConsistencyModel]] = None,
        sharding_strategy: Optional[ShardingStrategy] = None,
    ) -> "ParameterServerCluster":
        """Create a new cluster with the specified number of servers.

        Args:
            num_servers: Number of parameter servers to create.
            update_engine_factory: Factory function to create update engines.
                Defaults to SGDEngine with lr=0.01.
            consistency_factory: Factory function to create consistency models.
                Defaults to HogwildConsistency.
            sharding_strategy: Sharding strategy to use.

        Returns:
            Initialized ParameterServerCluster.
        """
        if update_engine_factory is None:
            update_engine_factory = lambda: SGDEngine(lr=0.01)
        if consistency_factory is None:
            consistency_factory = HogwildConsistency

        servers = []
        for i in range(num_servers):
            server = ParameterServer(
                shard_id=i,
                update_engine=update_engine_factory(),
                consistency=consistency_factory(),
            )
            servers.append(server)

        return cls(servers, sharding_strategy)

    async def initialize(self, model_params: Dict[str, np.ndarray]) -> None:
        """Initialize all servers with model parameters.

        Args:
            model_params: Dictionary mapping parameter names to values.
        """
        if not self.servers:
            raise ValueError("No servers in cluster")

        # Compute sharding
        self._shard_info = self.sharding.compute_shards(
            model_params,
            len(self.servers),
        )

        # Initialize each server with its shard of parameters
        init_tasks = []
        for shard in self._shard_info:
            server = self.servers[shard.shard_id]
            shard_params = {
                name: model_params[name]
                for name in shard.param_names
                if name in model_params
            }
            init_tasks.append(server.initialize(shard_params))

        await asyncio.gather(*init_tasks)
        self._initialized = True

    async def pull(
        self,
        param_names: List[str],
        worker_id: int,
    ) -> Dict[str, np.ndarray]:
        """Pull parameters from appropriate servers.

        Args:
            param_names: List of parameter names to retrieve.
            worker_id: ID of the requesting worker.

        Returns:
            Dictionary mapping parameter names to values.
        """
        # Group parameters by server
        server_params: Dict[int, List[str]] = {}
        for name in param_names:
            shard_id = self.sharding.get_shard_for_param(name)
            if shard_id not in server_params:
                server_params[shard_id] = []
            server_params[shard_id].append(name)

        # Pull from each server in parallel
        tasks = []
        for shard_id, names in server_params.items():
            if shard_id in self.servers:
                task = self.servers[shard_id].pull(names, worker_id)
                tasks.append(task)

        results = await asyncio.gather(*tasks)

        # Merge results
        merged: Dict[str, np.ndarray] = {}
        for result in results:
            for name, (value, _) in result.items():
                merged[name] = value

        return merged

    async def pull_with_versions(
        self,
        param_names: List[str],
        worker_id: int,
    ) -> Dict[str, tuple]:
        """Pull parameters with version info.

        Args:
            param_names: List of parameter names.
            worker_id: Requesting worker ID.

        Returns:
            Dictionary mapping names to (value, version) tuples.
        """
        # Group by server
        server_params: Dict[int, List[str]] = {}
        for name in param_names:
            shard_id = self.sharding.get_shard_for_param(name)
            if shard_id not in server_params:
                server_params[shard_id] = []
            server_params[shard_id].append(name)

        # Pull from each server
        tasks = []
        for shard_id, names in server_params.items():
            if shard_id in self.servers:
                task = self.servers[shard_id].pull(
                    names, worker_id, include_versions=True
                )
                tasks.append(task)

        results = await asyncio.gather(*tasks)

        # Merge
        merged = {}
        for result in results:
            merged.update(result)

        return merged

    async def push(
        self,
        gradients: Dict[str, np.ndarray],
        worker_id: int,
        clock: int,
    ) -> int:
        """Push gradients to appropriate servers.

        Args:
            gradients: Dictionary mapping parameter names to gradients.
            worker_id: ID of the worker sending gradients.
            clock: Worker's logical clock.

        Returns:
            Total number of updates applied.
        """
        # Group gradients by server
        server_grads: Dict[int, Dict[str, np.ndarray]] = {}
        for name, grad in gradients.items():
            shard_id = self.sharding.get_shard_for_param(name)
            if shard_id not in server_grads:
                server_grads[shard_id] = {}
            server_grads[shard_id][name] = grad

        # Push to each server in parallel
        tasks = []
        for shard_id, grads in server_grads.items():
            if shard_id in self.servers:
                task = self.servers[shard_id].push(grads, worker_id, clock)
                tasks.append(task)

        results = await asyncio.gather(*tasks)
        return sum(results)

    async def get_all_params(self) -> Dict[str, np.ndarray]:
        """Get all parameters from all servers.

        Returns:
            Dictionary mapping parameter names to values.
        """
        all_params = {}
        for server in self.servers.values():
            all_params.update(server.get_all_params())
        return all_params

    def get_shard_info(self) -> List[ShardInfo]:
        """Get information about all shards.

        Returns:
            List of ShardInfo objects.
        """
        return self._shard_info

    def get_server(self, shard_id: int) -> Optional[ParameterServer]:
        """Get a specific server by shard ID.

        Args:
            shard_id: The shard ID.

        Returns:
            ParameterServer instance or None.
        """
        return self.servers.get(shard_id)

    @property
    def num_servers(self) -> int:
        """Get number of servers in the cluster."""
        return len(self.servers)

    @property
    def total_params(self) -> int:
        """Get total number of parameters across all servers."""
        return sum(s.total_params for s in self.servers.values())

    @property
    def is_initialized(self) -> bool:
        """Check if cluster is initialized."""
        return self._initialized

    async def health_check(self) -> Dict[int, bool]:
        """Check health of all servers.

        Returns:
            Dictionary mapping shard IDs to health status.
        """
        results = {}
        for shard_id, server in self.servers.items():
            results[shard_id] = await server.health_check()
        return results

    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get statistics for the entire cluster.

        Returns:
            Dictionary of cluster statistics.
        """
        server_stats = [s.stats for s in self.servers.values()]
        return {
            "num_servers": self.num_servers,
            "total_params": self.total_params,
            "initialized": self._initialized,
            "total_pulls": sum(s["total_pulls"] for s in server_stats),
            "total_pushes": sum(s["total_pushes"] for s in server_stats),
            "servers": server_stats,
        }
