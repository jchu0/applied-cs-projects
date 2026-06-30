"""Replica management for fault tolerance."""

import asyncio
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import numpy as np


class ReplicationStrategy(Enum):
    """Replication strategy for parameter updates."""
    SYNC = "sync"       # Wait for all replicas
    ASYNC = "async"     # Fire and forget
    QUORUM = "quorum"   # Wait for majority


class ReplicaManager:
    """Manages parameter server replicas for fault tolerance.

    Handles replication of parameter updates to backup servers and
    provides failover when primary servers fail.

    Attributes:
        num_replicas: Number of replicas per shard.
        strategy: Replication strategy (sync, async, quorum).
    """

    def __init__(
        self,
        num_replicas: int = 2,
        strategy: ReplicationStrategy = ReplicationStrategy.SYNC,
    ):
        """Initialize replica manager.

        Args:
            num_replicas: Number of replica servers per primary.
            strategy: How to handle replication (sync, async, quorum).
        """
        self.num_replicas = num_replicas
        self.strategy = strategy

        # Map primary shard ID to list of replica servers
        # Each replica is an object with update_param method
        self._replicas: Dict[int, List[Any]] = {}

        # Track replica health
        self._replica_healthy: Dict[int, List[bool]] = {}

        # Failover callbacks
        self._failover_callbacks: List[Callable[[int, Any], None]] = []

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

    def register_replica(
        self,
        primary_shard_id: int,
        replica_server: Any,
    ) -> None:
        """Register a replica for a primary shard.

        Args:
            primary_shard_id: ID of the primary shard.
            replica_server: Replica server instance.
        """
        if primary_shard_id not in self._replicas:
            self._replicas[primary_shard_id] = []
            self._replica_healthy[primary_shard_id] = []

        self._replicas[primary_shard_id].append(replica_server)
        self._replica_healthy[primary_shard_id].append(True)

    def unregister_replica(
        self,
        primary_shard_id: int,
        replica_index: int,
    ) -> Optional[Any]:
        """Unregister a replica.

        Args:
            primary_shard_id: Primary shard ID.
            replica_index: Index of replica to remove.

        Returns:
            Removed replica or None.
        """
        if primary_shard_id not in self._replicas:
            return None

        replicas = self._replicas[primary_shard_id]
        if replica_index >= len(replicas):
            return None

        self._replica_healthy[primary_shard_id].pop(replica_index)
        return replicas.pop(replica_index)

    async def replicate_update(
        self,
        primary_shard_id: int,
        param_name: str,
        new_value: np.ndarray,
    ) -> int:
        """Replicate a parameter update to replicas.

        Args:
            primary_shard_id: ID of the primary shard.
            param_name: Name of the parameter.
            new_value: New parameter value.

        Returns:
            Number of successful replications.
        """
        if primary_shard_id not in self._replicas:
            return 0

        replicas = self._replicas[primary_shard_id]
        if not replicas:
            return 0

        if self.strategy == ReplicationStrategy.SYNC:
            return await self._sync_replicate(
                primary_shard_id, param_name, new_value
            )
        elif self.strategy == ReplicationStrategy.ASYNC:
            return await self._async_replicate(
                primary_shard_id, param_name, new_value
            )
        elif self.strategy == ReplicationStrategy.QUORUM:
            return await self._quorum_replicate(
                primary_shard_id, param_name, new_value
            )

        return 0

    async def _sync_replicate(
        self,
        shard_id: int,
        param_name: str,
        value: np.ndarray,
    ) -> int:
        """Synchronous replication - wait for all replicas."""
        replicas = self._replicas[shard_id]
        health = self._replica_healthy[shard_id]

        tasks = []
        for i, replica in enumerate(replicas):
            if health[i]:
                task = self._update_replica(shard_id, i, replica, param_name, value)
                tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = sum(1 for r in results if r is True)
        return success_count

    async def _async_replicate(
        self,
        shard_id: int,
        param_name: str,
        value: np.ndarray,
    ) -> int:
        """Asynchronous replication - fire and forget."""
        replicas = self._replicas[shard_id]
        health = self._replica_healthy[shard_id]

        count = 0
        for i, replica in enumerate(replicas):
            if health[i]:
                # Fire and forget
                asyncio.create_task(
                    self._update_replica(shard_id, i, replica, param_name, value)
                )
                count += 1

        return count

    async def _quorum_replicate(
        self,
        shard_id: int,
        param_name: str,
        value: np.ndarray,
    ) -> int:
        """Quorum replication - wait for majority."""
        replicas = self._replicas[shard_id]
        health = self._replica_healthy[shard_id]

        tasks = []
        for i, replica in enumerate(replicas):
            if health[i]:
                # Must wrap coroutines in tasks for asyncio.wait()
                task = asyncio.create_task(
                    self._update_replica(shard_id, i, replica, param_name, value)
                )
                tasks.append(task)

        if not tasks:
            return 0

        # Wait for quorum (majority)
        quorum = (len(tasks) // 2) + 1
        pending = set(tasks)
        success_count = 0

        while pending and success_count < quorum:
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                if task.result() is True:
                    success_count += 1

        # Cancel remaining tasks
        for task in pending:
            task.cancel()

        return success_count

    async def _update_replica(
        self,
        shard_id: int,
        replica_index: int,
        replica: Any,
        param_name: str,
        value: np.ndarray,
    ) -> bool:
        """Update a single replica."""
        try:
            if hasattr(replica, "set_param"):
                await replica.set_param(param_name, value)
            elif hasattr(replica, "update_param"):
                await replica.update_param(param_name, value)
            return True
        except Exception:
            # Mark replica as unhealthy
            self._replica_healthy[shard_id][replica_index] = False
            return False

    async def failover(self, failed_shard_id: int) -> Optional[Any]:
        """Promote a replica to primary on failure.

        Args:
            failed_shard_id: ID of the failed shard.

        Returns:
            New primary server or None if no replicas.
        """
        async with self._lock:
            if failed_shard_id not in self._replicas:
                return None

            replicas = self._replicas[failed_shard_id]
            health = self._replica_healthy[failed_shard_id]

            # Find first healthy replica
            for i, is_healthy in enumerate(health):
                if is_healthy:
                    new_primary = replicas.pop(i)
                    health.pop(i)

                    # Notify callbacks
                    for callback in self._failover_callbacks:
                        try:
                            callback(failed_shard_id, new_primary)
                        except Exception:
                            pass

                    return new_primary

            return None

    def add_failover_callback(
        self,
        callback: Callable[[int, Any], None],
    ) -> None:
        """Add a callback for failover events.

        Args:
            callback: Function called with (shard_id, new_primary).
        """
        self._failover_callbacks.append(callback)

    def get_replica_count(self, shard_id: int) -> int:
        """Get number of replicas for a shard.

        Args:
            shard_id: Shard ID to query.

        Returns:
            Number of replicas.
        """
        return len(self._replicas.get(shard_id, []))

    def get_healthy_replica_count(self, shard_id: int) -> int:
        """Get number of healthy replicas for a shard.

        Args:
            shard_id: Shard ID to query.

        Returns:
            Number of healthy replicas.
        """
        if shard_id not in self._replica_healthy:
            return 0
        return sum(self._replica_healthy[shard_id])

    def mark_replica_healthy(
        self,
        shard_id: int,
        replica_index: int,
        healthy: bool = True,
    ) -> None:
        """Mark a replica as healthy or unhealthy.

        Args:
            shard_id: Shard ID.
            replica_index: Replica index.
            healthy: Health status.
        """
        if shard_id in self._replica_healthy:
            if replica_index < len(self._replica_healthy[shard_id]):
                self._replica_healthy[shard_id][replica_index] = healthy

    def get_stats(self) -> Dict[str, Any]:
        """Get replication statistics.

        Returns:
            Dictionary of statistics.
        """
        total_replicas = sum(len(r) for r in self._replicas.values())
        healthy_replicas = sum(
            sum(h) for h in self._replica_healthy.values()
        )

        return {
            "num_shards": len(self._replicas),
            "total_replicas": total_replicas,
            "healthy_replicas": healthy_replicas,
            "strategy": self.strategy.value,
        }
