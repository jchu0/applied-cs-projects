"""Tests for replica manager."""

import pytest
import asyncio
import numpy as np

from paramserver.fault_tolerance.replica import (
    ReplicaManager,
    ReplicationStrategy,
)


class MockReplica:
    """Mock replica server for testing."""

    def __init__(self, should_fail: bool = False):
        self.params = {}
        self.should_fail = should_fail
        self.update_count = 0

    async def set_param(self, name: str, value: np.ndarray) -> None:
        if self.should_fail:
            raise Exception("Replica failed")
        self.params[name] = value.copy()
        self.update_count += 1


class TestReplicaManagerInit:
    """Tests for ReplicaManager initialization."""

    def test_create_default(self):
        """Test default creation."""
        manager = ReplicaManager()
        assert manager.num_replicas == 2
        assert manager.strategy == ReplicationStrategy.SYNC

    def test_create_custom(self):
        """Test custom creation."""
        manager = ReplicaManager(
            num_replicas=3,
            strategy=ReplicationStrategy.ASYNC,
        )
        assert manager.num_replicas == 3
        assert manager.strategy == ReplicationStrategy.ASYNC


class TestReplicaRegistration:
    """Tests for replica registration."""

    def test_register_replica(self):
        """Test registering a replica."""
        manager = ReplicaManager()
        replica = MockReplica()

        manager.register_replica(0, replica)

        assert manager.get_replica_count(0) == 1

    def test_register_multiple_replicas(self):
        """Test registering multiple replicas."""
        manager = ReplicaManager()

        for i in range(3):
            manager.register_replica(0, MockReplica())

        assert manager.get_replica_count(0) == 3

    def test_unregister_replica(self):
        """Test unregistering a replica."""
        manager = ReplicaManager()
        replica = MockReplica()

        manager.register_replica(0, replica)
        removed = manager.unregister_replica(0, 0)

        assert removed is replica
        assert manager.get_replica_count(0) == 0

    def test_unregister_nonexistent(self):
        """Test unregistering nonexistent replica."""
        manager = ReplicaManager()
        removed = manager.unregister_replica(99, 0)
        assert removed is None


class TestSyncReplication:
    """Tests for synchronous replication."""

    @pytest.mark.asyncio
    async def test_sync_replicate(self):
        """Test synchronous replication."""
        manager = ReplicaManager(strategy=ReplicationStrategy.SYNC)

        replicas = [MockReplica() for _ in range(2)]
        for replica in replicas:
            manager.register_replica(0, replica)

        value = np.array([1.0, 2.0, 3.0])
        success = await manager.replicate_update(0, "w1", value)

        assert success == 2
        for replica in replicas:
            np.testing.assert_array_equal(replica.params["w1"], value)

    @pytest.mark.asyncio
    async def test_sync_partial_failure(self):
        """Test sync replication with partial failure."""
        manager = ReplicaManager(strategy=ReplicationStrategy.SYNC)

        healthy = MockReplica()
        failing = MockReplica(should_fail=True)

        manager.register_replica(0, healthy)
        manager.register_replica(0, failing)

        value = np.array([1.0])
        success = await manager.replicate_update(0, "w1", value)

        # One succeeded, one failed
        assert success == 1
        np.testing.assert_array_equal(healthy.params["w1"], value)


class TestAsyncReplication:
    """Tests for asynchronous replication."""

    @pytest.mark.asyncio
    async def test_async_replicate(self):
        """Test asynchronous replication."""
        manager = ReplicaManager(strategy=ReplicationStrategy.ASYNC)

        replicas = [MockReplica() for _ in range(2)]
        for replica in replicas:
            manager.register_replica(0, replica)

        value = np.array([1.0, 2.0])
        count = await manager.replicate_update(0, "w1", value)

        # Returns immediately with count of tasks started
        assert count == 2

        # Wait for async tasks to complete
        await asyncio.sleep(0.1)

        for replica in replicas:
            assert "w1" in replica.params


class TestQuorumReplication:
    """Tests for quorum replication."""

    @pytest.mark.asyncio
    async def test_quorum_replicate(self):
        """Test quorum replication."""
        manager = ReplicaManager(strategy=ReplicationStrategy.QUORUM)

        replicas = [MockReplica() for _ in range(3)]
        for replica in replicas:
            manager.register_replica(0, replica)

        value = np.array([1.0])
        success = await manager.replicate_update(0, "w1", value)

        # Quorum is 2 out of 3
        assert success >= 2


class TestFailover:
    """Tests for failover functionality."""

    @pytest.mark.asyncio
    async def test_failover(self):
        """Test promoting replica to primary."""
        manager = ReplicaManager()

        replica1 = MockReplica()
        replica2 = MockReplica()

        manager.register_replica(0, replica1)
        manager.register_replica(0, replica2)

        new_primary = await manager.failover(0)

        assert new_primary is replica1
        assert manager.get_replica_count(0) == 1

    @pytest.mark.asyncio
    async def test_failover_no_replicas(self):
        """Test failover with no replicas."""
        manager = ReplicaManager()
        new_primary = await manager.failover(0)
        assert new_primary is None

    @pytest.mark.asyncio
    async def test_failover_callback(self):
        """Test failover callback is called."""
        manager = ReplicaManager()
        replica = MockReplica()
        manager.register_replica(0, replica)

        callback_called = []

        def on_failover(shard_id, new_primary):
            callback_called.append((shard_id, new_primary))

        manager.add_failover_callback(on_failover)
        await manager.failover(0)

        assert len(callback_called) == 1
        assert callback_called[0][0] == 0


class TestReplicaHealth:
    """Tests for replica health tracking."""

    def test_mark_replica_healthy(self):
        """Test marking replica health."""
        manager = ReplicaManager()
        replica = MockReplica()
        manager.register_replica(0, replica)

        assert manager.get_healthy_replica_count(0) == 1

        manager.mark_replica_healthy(0, 0, healthy=False)
        assert manager.get_healthy_replica_count(0) == 0

        manager.mark_replica_healthy(0, 0, healthy=True)
        assert manager.get_healthy_replica_count(0) == 1

    @pytest.mark.asyncio
    async def test_unhealthy_excluded(self):
        """Test unhealthy replicas excluded from replication."""
        manager = ReplicaManager(strategy=ReplicationStrategy.SYNC)

        healthy = MockReplica()
        manager.register_replica(0, healthy)
        manager.register_replica(0, MockReplica())

        # Mark second replica as unhealthy
        manager.mark_replica_healthy(0, 1, healthy=False)

        value = np.array([1.0])
        success = await manager.replicate_update(0, "w1", value)

        # Only healthy replica should be updated
        assert success == 1


class TestReplicaStats:
    """Tests for replica statistics."""

    def test_get_stats(self):
        """Test getting replica stats."""
        manager = ReplicaManager(strategy=ReplicationStrategy.QUORUM)

        for shard in range(2):
            for _ in range(3):
                manager.register_replica(shard, MockReplica())

        stats = manager.get_stats()

        assert stats["num_shards"] == 2
        assert stats["total_replicas"] == 6
        assert stats["healthy_replicas"] == 6
        assert stats["strategy"] == "quorum"
