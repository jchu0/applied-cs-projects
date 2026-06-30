"""Parameter sharding and partitioning."""

import hashlib
import numpy as np
from typing import Any

from ..schemas import Parameter, ShardInfo, PartitionStrategy


class ParameterPartitioner:
    """Partitions parameters across shards."""

    def __init__(self, strategy: PartitionStrategy):
        """Initialize partitioner.

        Args:
            strategy: Partitioning strategy
        """
        self.strategy = strategy
        self.num_shards = strategy.num_shards

    def get_shard(self, parameter_name: str) -> int:
        """Get shard ID for parameter.

        Args:
            parameter_name: Parameter name

        Returns:
            Shard ID
        """
        if self.strategy.strategy_type == "hash":
            return self._hash_partition(parameter_name)
        elif self.strategy.strategy_type == "range":
            return self._range_partition(parameter_name)
        elif self.strategy.strategy_type == "round_robin":
            return self._round_robin_partition(parameter_name)
        else:
            return self._hash_partition(parameter_name)

    def _hash_partition(self, name: str) -> int:
        """Hash-based partitioning."""
        hash_val = int(hashlib.md5(name.encode()).hexdigest(), 16)
        return hash_val % self.num_shards

    def _range_partition(self, name: str) -> int:
        """Range-based partitioning (alphabetical)."""
        first_char = name[0].lower() if name else 'a'
        char_ord = ord(first_char) - ord('a')
        chars_per_shard = 26 // self.num_shards
        return min(char_ord // max(chars_per_shard, 1), self.num_shards - 1)

    def _round_robin_partition(self, name: str) -> int:
        """Round-robin partitioning by name hash."""
        return hash(name) % self.num_shards


class ShardManager:
    """Manages parameter shards."""

    def __init__(self, num_shards: int):
        """Initialize shard manager.

        Args:
            num_shards: Number of shards
        """
        self.num_shards = num_shards
        self._shards: dict[int, dict[str, Parameter]] = {
            i: {} for i in range(num_shards)
        }
        self._shard_sizes: dict[int, int] = {i: 0 for i in range(num_shards)}

    def store(self, shard_id: int, parameter: Parameter):
        """Store parameter in shard.

        Args:
            shard_id: Shard ID
            parameter: Parameter to store
        """
        if shard_id not in self._shards:
            raise ValueError(f"Invalid shard ID: {shard_id}")

        old_size = 0
        if parameter.name in self._shards[shard_id]:
            old_param = self._shards[shard_id][parameter.name]
            if old_param.data is not None:
                old_size = old_param.data.nbytes

        self._shards[shard_id][parameter.name] = parameter

        if parameter.data is not None:
            new_size = parameter.data.nbytes
            self._shard_sizes[shard_id] += new_size - old_size

    def get(self, shard_id: int, name: str) -> Parameter | None:
        """Get parameter from shard.

        Args:
            shard_id: Shard ID
            name: Parameter name

        Returns:
            Parameter or None
        """
        if shard_id not in self._shards:
            return None
        return self._shards[shard_id].get(name)

    def get_all(self, shard_id: int) -> list[Parameter]:
        """Get all parameters in shard.

        Args:
            shard_id: Shard ID

        Returns:
            List of parameters
        """
        if shard_id not in self._shards:
            return []
        return list(self._shards[shard_id].values())

    def delete(self, shard_id: int, name: str) -> bool:
        """Delete parameter from shard.

        Args:
            shard_id: Shard ID
            name: Parameter name

        Returns:
            True if deleted
        """
        if shard_id not in self._shards:
            return False

        if name in self._shards[shard_id]:
            param = self._shards[shard_id].pop(name)
            if param.data is not None:
                self._shard_sizes[shard_id] -= param.data.nbytes
            return True
        return False

    def get_shard_info(self, shard_id: int) -> ShardInfo:
        """Get shard information.

        Args:
            shard_id: Shard ID

        Returns:
            Shard information
        """
        params = list(self._shards.get(shard_id, {}).keys())
        return ShardInfo(
            shard_id=shard_id,
            parameter_names=params,
            node_id="",  # Set by server
            size_bytes=self._shard_sizes.get(shard_id, 0),
            num_parameters=len(params)
        )

    def get_total_size(self) -> int:
        """Get total size across all shards.

        Returns:
            Total size in bytes
        """
        return sum(self._shard_sizes.values())

    def rebalance_shards(self) -> dict[int, list[str]]:
        """Identify parameters to move for rebalancing.

        Returns:
            Mapping of shard_id to parameters to move
        """
        avg_size = self.get_total_size() / self.num_shards
        threshold = avg_size * 0.2  # 20% threshold

        moves = {}
        for shard_id, size in self._shard_sizes.items():
            if size > avg_size + threshold:
                # Find parameters to move
                excess = size - avg_size
                to_move = []
                moved_size = 0

                for name, param in self._shards[shard_id].items():
                    if param.data is not None:
                        param_size = param.data.nbytes
                        if moved_size + param_size <= excess:
                            to_move.append(name)
                            moved_size += param_size

                if to_move:
                    moves[shard_id] = to_move

        return moves
