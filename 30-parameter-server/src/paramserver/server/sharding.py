"""Sharding strategies for distributing parameters across servers."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import numpy as np

from paramserver.schemas import ShardInfo


class ShardingStrategy(ABC):
    """Abstract base class for parameter sharding strategies.

    A sharding strategy determines how model parameters are distributed
    across multiple parameter servers.
    """

    @abstractmethod
    def compute_shards(
        self,
        model_params: Dict[str, np.ndarray],
        num_servers: int,
    ) -> List[ShardInfo]:
        """Compute how to distribute parameters across servers.

        Args:
            model_params: Dictionary mapping parameter names to their values.
            num_servers: Number of parameter servers available.

        Returns:
            List of ShardInfo objects, one per server.
        """
        pass

    @abstractmethod
    def get_shard_for_param(self, param_name: str) -> int:
        """Get the shard ID that contains a given parameter.

        Args:
            param_name: Name of the parameter.

        Returns:
            Shard ID (server index) containing this parameter.
        """
        pass

    def get_params_for_shard(self, shard_id: int) -> List[str]:
        """Get list of parameter names assigned to a shard.

        Args:
            shard_id: The shard ID to query.

        Returns:
            List of parameter names in this shard.
        """
        raise NotImplementedError


class UniformSharding(ShardingStrategy):
    """Distribute parameters uniformly across servers by name.

    Parameters are assigned to servers using a hash of their names,
    ensuring a roughly even distribution across all servers.

    Attributes:
        num_servers: Number of servers to distribute across.
        _param_to_shard: Cache mapping parameter names to shard IDs.
        _shard_to_params: Cache mapping shard IDs to parameter names.
    """

    def __init__(self, num_servers: Optional[int] = None):
        """Initialize uniform sharding.

        Args:
            num_servers: Number of servers. If None, will be set during compute_shards.
        """
        self.num_servers = num_servers
        self._param_to_shard: Dict[str, int] = {}
        self._shard_to_params: Dict[int, List[str]] = {}

    def compute_shards(
        self,
        model_params: Dict[str, np.ndarray],
        num_servers: int,
    ) -> List[ShardInfo]:
        """Compute uniform distribution of parameters.

        Args:
            model_params: Dictionary mapping parameter names to values.
            num_servers: Number of servers to distribute across.

        Returns:
            List of ShardInfo objects describing the distribution.
        """
        self.num_servers = num_servers

        # Reset caches
        self._param_to_shard.clear()
        self._shard_to_params.clear()

        # Initialize shard info
        shards: Dict[int, ShardInfo] = {}
        for i in range(num_servers):
            shards[i] = ShardInfo(
                shard_id=i,
                server_address=f"server-{i}",
                param_names=[],
                total_params=0,
                memory_bytes=0,
            )
            self._shard_to_params[i] = []

        # Assign parameters to shards using hash
        for name, value in model_params.items():
            shard_id = self._hash_to_shard(name)
            self._param_to_shard[name] = shard_id
            self._shard_to_params[shard_id].append(name)

            shard = shards[shard_id]
            shard.param_names.append(name)
            shard.total_params += value.size
            shard.memory_bytes += value.nbytes

        return list(shards.values())

    def get_shard_for_param(self, param_name: str) -> int:
        """Get shard ID for a parameter.

        Args:
            param_name: Name of the parameter.

        Returns:
            Shard ID containing this parameter.

        Raises:
            ValueError: If num_servers is not set.
        """
        if param_name in self._param_to_shard:
            return self._param_to_shard[param_name]

        if self.num_servers is None:
            raise ValueError("num_servers must be set before querying shards")

        # Compute hash for new parameter
        shard_id = self._hash_to_shard(param_name)
        self._param_to_shard[param_name] = shard_id
        return shard_id

    def get_params_for_shard(self, shard_id: int) -> List[str]:
        """Get parameter names in a shard.

        Args:
            shard_id: The shard ID to query.

        Returns:
            List of parameter names in this shard.
        """
        return self._shard_to_params.get(shard_id, [])

    def _hash_to_shard(self, param_name: str) -> int:
        """Hash parameter name to shard ID.

        Args:
            param_name: Name of the parameter.

        Returns:
            Shard ID (0 to num_servers-1).
        """
        if self.num_servers is None:
            raise ValueError("num_servers must be set")
        return hash(param_name) % self.num_servers


class RoundRobinSharding(ShardingStrategy):
    """Assign parameters to servers in round-robin order.

    Parameters are assigned in the order they are provided, cycling
    through servers sequentially.
    """

    def __init__(self, num_servers: Optional[int] = None):
        """Initialize round-robin sharding."""
        self.num_servers = num_servers
        self._param_to_shard: Dict[str, int] = {}
        self._shard_to_params: Dict[int, List[str]] = {}
        self._param_order: List[str] = []

    def compute_shards(
        self,
        model_params: Dict[str, np.ndarray],
        num_servers: int,
    ) -> List[ShardInfo]:
        """Compute round-robin distribution.

        Args:
            model_params: Dictionary mapping parameter names to values.
            num_servers: Number of servers.

        Returns:
            List of ShardInfo objects.
        """
        self.num_servers = num_servers
        self._param_to_shard.clear()
        self._shard_to_params.clear()
        self._param_order = list(model_params.keys())

        # Initialize shards
        shards: Dict[int, ShardInfo] = {}
        for i in range(num_servers):
            shards[i] = ShardInfo(
                shard_id=i,
                server_address=f"server-{i}",
                param_names=[],
                total_params=0,
                memory_bytes=0,
            )
            self._shard_to_params[i] = []

        # Assign in round-robin order
        for idx, (name, value) in enumerate(model_params.items()):
            shard_id = idx % num_servers
            self._param_to_shard[name] = shard_id
            self._shard_to_params[shard_id].append(name)

            shard = shards[shard_id]
            shard.param_names.append(name)
            shard.total_params += value.size
            shard.memory_bytes += value.nbytes

        return list(shards.values())

    def get_shard_for_param(self, param_name: str) -> int:
        """Get shard ID for a parameter."""
        if param_name not in self._param_to_shard:
            raise ValueError(f"Unknown parameter: {param_name}")
        return self._param_to_shard[param_name]

    def get_params_for_shard(self, shard_id: int) -> List[str]:
        """Get parameters in a shard."""
        return self._shard_to_params.get(shard_id, [])


class SizeBalancedSharding(ShardingStrategy):
    """Balance parameters across servers by total size.

    Uses a greedy algorithm to assign parameters to the server with
    the least total parameter count, resulting in roughly equal memory
    usage across servers.
    """

    def __init__(self, num_servers: Optional[int] = None):
        """Initialize size-balanced sharding."""
        self.num_servers = num_servers
        self._param_to_shard: Dict[str, int] = {}
        self._shard_to_params: Dict[int, List[str]] = {}

    def compute_shards(
        self,
        model_params: Dict[str, np.ndarray],
        num_servers: int,
    ) -> List[ShardInfo]:
        """Compute size-balanced distribution.

        Args:
            model_params: Dictionary mapping parameter names to values.
            num_servers: Number of servers.

        Returns:
            List of ShardInfo objects.
        """
        self.num_servers = num_servers
        self._param_to_shard.clear()
        self._shard_to_params.clear()

        # Initialize shards
        shards: Dict[int, ShardInfo] = {}
        for i in range(num_servers):
            shards[i] = ShardInfo(
                shard_id=i,
                server_address=f"server-{i}",
                param_names=[],
                total_params=0,
                memory_bytes=0,
            )
            self._shard_to_params[i] = []

        # Sort parameters by size (largest first for better packing)
        sorted_params = sorted(
            model_params.items(),
            key=lambda x: x[1].size,
            reverse=True,
        )

        # Greedily assign to least-loaded server
        for name, value in sorted_params:
            # Find server with least parameters
            min_shard_id = min(
                range(num_servers),
                key=lambda i: shards[i].total_params,
            )

            self._param_to_shard[name] = min_shard_id
            self._shard_to_params[min_shard_id].append(name)

            shard = shards[min_shard_id]
            shard.param_names.append(name)
            shard.total_params += value.size
            shard.memory_bytes += value.nbytes

        return list(shards.values())

    def get_shard_for_param(self, param_name: str) -> int:
        """Get shard ID for a parameter."""
        if param_name not in self._param_to_shard:
            raise ValueError(f"Unknown parameter: {param_name}")
        return self._param_to_shard[param_name]

    def get_params_for_shard(self, shard_id: int) -> List[str]:
        """Get parameters in a shard."""
        return self._shard_to_params.get(shard_id, [])
