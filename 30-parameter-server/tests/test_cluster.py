"""Tests for ParameterServerCluster."""

import pytest
import numpy as np
import asyncio

from paramserver.server.cluster import ParameterServerCluster
from paramserver.server.parameter_server import ParameterServer
from paramserver.server.sharding import UniformSharding, SizeBalancedSharding
from paramserver.optimizer.sgd import SGDEngine
from paramserver.consistency.hogwild import HogwildConsistency


class TestClusterCreation:
    """Tests for cluster creation."""

    def test_create_default(self):
        """Test default cluster creation."""
        cluster = ParameterServerCluster.create(num_servers=3)
        assert cluster.num_servers == 3
        assert not cluster.is_initialized

    def test_create_with_custom_engine(self):
        """Test cluster with custom update engine."""
        def engine_factory():
            return SGDEngine(lr=0.05, momentum=0.9)

        cluster = ParameterServerCluster.create(
            num_servers=2,
            update_engine_factory=engine_factory,
        )

        assert cluster.num_servers == 2
        # Check that custom engine was used
        server = cluster.get_server(0)
        assert server.update_engine.lr == 0.05
        assert server.update_engine.momentum == 0.9

    def test_create_with_custom_sharding(self):
        """Test cluster with custom sharding strategy."""
        sharding = SizeBalancedSharding()
        cluster = ParameterServerCluster.create(
            num_servers=3,
            sharding_strategy=sharding,
        )

        assert cluster.sharding is sharding

    def test_create_from_servers(self, sgd_engine, hogwild_consistency):
        """Test cluster from existing servers."""
        servers = [
            ParameterServer(i, SGDEngine(lr=0.01), HogwildConsistency())
            for i in range(3)
        ]
        cluster = ParameterServerCluster(servers=servers)

        assert cluster.num_servers == 3


class TestClusterInitialization:
    """Tests for cluster initialization."""

    @pytest.mark.asyncio
    async def test_initialize(self, ps_cluster, sample_params):
        """Test cluster initialization."""
        await ps_cluster.initialize(sample_params)

        assert ps_cluster.is_initialized
        assert ps_cluster.total_params > 0

    @pytest.mark.asyncio
    async def test_initialize_distributes_params(self, ps_cluster, sample_params):
        """Test that params are distributed across servers."""
        await ps_cluster.initialize(sample_params)

        # Collect all params from all servers
        all_params = {}
        for server in ps_cluster.servers.values():
            all_params.update(server.get_all_params())

        assert set(all_params.keys()) == set(sample_params.keys())

    @pytest.mark.asyncio
    async def test_initialize_empty_cluster(self, sample_params):
        """Test initializing empty cluster raises error."""
        cluster = ParameterServerCluster()

        with pytest.raises(ValueError, match="No servers"):
            await cluster.initialize(sample_params)

    @pytest.mark.asyncio
    async def test_shard_info(self, ps_cluster, sample_params):
        """Test getting shard info after init."""
        await ps_cluster.initialize(sample_params)

        info = ps_cluster.get_shard_info()
        assert len(info) == ps_cluster.num_servers

        # All params should be in shard info
        all_params = set()
        for shard in info:
            all_params.update(shard.param_names)
        assert all_params == set(sample_params.keys())


class TestClusterPull:
    """Tests for pull operations."""

    @pytest.mark.asyncio
    async def test_pull_single_param(self, initialized_cluster, sample_params):
        """Test pulling a single parameter."""
        result = await initialized_cluster.pull(["layer1.weight"], worker_id=0)

        assert "layer1.weight" in result
        np.testing.assert_array_equal(
            result["layer1.weight"],
            sample_params["layer1.weight"],
        )

    @pytest.mark.asyncio
    async def test_pull_all_params(self, initialized_cluster, sample_params):
        """Test pulling all parameters."""
        result = await initialized_cluster.pull(
            list(sample_params.keys()),
            worker_id=0,
        )

        assert len(result) == len(sample_params)
        for name, value in sample_params.items():
            np.testing.assert_array_equal(result[name], value)

    @pytest.mark.asyncio
    async def test_pull_with_versions(self, initialized_cluster, sample_params):
        """Test pulling with version info."""
        result = await initialized_cluster.pull_with_versions(
            list(sample_params.keys()),
            worker_id=0,
        )

        for name in sample_params.keys():
            assert name in result
            value, version = result[name]
            assert version == 0  # Initial version

    @pytest.mark.asyncio
    async def test_pull_distributes_to_servers(self, initialized_cluster, sample_params):
        """Test that pulls go to correct servers."""
        # Pull from multiple workers
        for worker_id in range(5):
            result = await initialized_cluster.pull(
                list(sample_params.keys()),
                worker_id=worker_id,
            )
            assert len(result) == len(sample_params)


class TestClusterPush:
    """Tests for push operations."""

    @pytest.mark.asyncio
    async def test_push_single_gradient(self, initialized_cluster, sample_params):
        """Test pushing a single gradient."""
        grad = np.random.randn(*sample_params["layer1.weight"].shape).astype(np.float32) * 0.01
        grads = {"layer1.weight": grad}

        applied = await initialized_cluster.push(grads, worker_id=0, clock=0)
        assert applied == 1

        # Verify update was applied
        result = await initialized_cluster.pull(["layer1.weight"], worker_id=0)
        expected = sample_params["layer1.weight"] - 0.01 * grad
        np.testing.assert_array_almost_equal(result["layer1.weight"], expected)

    @pytest.mark.asyncio
    async def test_push_all_gradients(self, initialized_cluster, sample_params):
        """Test pushing gradients for all parameters."""
        grads = {
            name: np.random.randn(*value.shape).astype(np.float32) * 0.01
            for name, value in sample_params.items()
        }

        applied = await initialized_cluster.push(grads, worker_id=0, clock=0)
        assert applied == len(sample_params)

    @pytest.mark.asyncio
    async def test_push_updates_version(self, initialized_cluster, sample_params):
        """Test that push updates version."""
        grads = {"layer1.weight": np.zeros_like(sample_params["layer1.weight"])}

        await initialized_cluster.push(grads, worker_id=0, clock=0)

        result = await initialized_cluster.pull_with_versions(
            ["layer1.weight"],
            worker_id=0,
        )
        _, version = result["layer1.weight"]
        assert version == 1


class TestClusterProperties:
    """Tests for cluster properties and methods."""

    @pytest.mark.asyncio
    async def test_get_all_params(self, initialized_cluster, sample_params):
        """Test getting all parameters."""
        all_params = await initialized_cluster.get_all_params()

        assert len(all_params) == len(sample_params)
        for name, value in sample_params.items():
            np.testing.assert_array_equal(all_params[name], value)

    @pytest.mark.asyncio
    async def test_total_params(self, initialized_cluster, sample_params):
        """Test total params property."""
        expected = sum(p.size for p in sample_params.values())
        assert initialized_cluster.total_params == expected

    @pytest.mark.asyncio
    async def test_health_check(self, initialized_cluster):
        """Test cluster health check."""
        health = await initialized_cluster.health_check()

        assert len(health) == initialized_cluster.num_servers
        assert all(healthy for healthy in health.values())

    @pytest.mark.asyncio
    async def test_cluster_stats(self, initialized_cluster, sample_params):
        """Test cluster statistics."""
        # Do some operations
        await initialized_cluster.pull(["layer1.weight"], worker_id=0)
        grads = {"layer1.weight": np.zeros_like(sample_params["layer1.weight"])}
        await initialized_cluster.push(grads, worker_id=0, clock=0)

        stats = initialized_cluster.get_cluster_stats()

        assert stats["num_servers"] == initialized_cluster.num_servers
        assert stats["initialized"] is True
        assert stats["total_pulls"] >= 1
        assert stats["total_pushes"] >= 1

    def test_get_server(self, ps_cluster):
        """Test getting server by shard ID."""
        server = ps_cluster.get_server(0)
        assert server is not None
        assert server.shard_id == 0

        server_none = ps_cluster.get_server(999)
        assert server_none is None


class TestClusterConcurrency:
    """Tests for concurrent cluster operations."""

    @pytest.mark.asyncio
    async def test_concurrent_pulls(self, initialized_cluster, sample_params):
        """Test concurrent pulls from multiple workers."""
        tasks = [
            initialized_cluster.pull(list(sample_params.keys()), worker_id=i)
            for i in range(10)
        ]
        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        for result in results:
            assert len(result) == len(sample_params)

    @pytest.mark.asyncio
    async def test_concurrent_push_pull(self, initialized_cluster, sample_params):
        """Test concurrent push and pull operations."""
        async def push_worker(worker_id):
            grads = {
                name: np.random.randn(*value.shape).astype(np.float32) * 0.001
                for name, value in sample_params.items()
            }
            return await initialized_cluster.push(grads, worker_id, worker_id)

        async def pull_worker(worker_id):
            return await initialized_cluster.pull(
                list(sample_params.keys()),
                worker_id,
            )

        # Run pushes and pulls concurrently
        tasks = []
        for i in range(5):
            tasks.append(push_worker(i))
            tasks.append(pull_worker(i + 100))

        results = await asyncio.gather(*tasks)
        assert len(results) == 10

    @pytest.mark.asyncio
    async def test_stress_test(self, initialized_cluster, sample_params):
        """Stress test with many concurrent operations."""
        async def worker_task(worker_id):
            for _ in range(10):
                # Pull
                params = await initialized_cluster.pull(
                    list(sample_params.keys()),
                    worker_id,
                )
                # Compute gradient (simulated)
                grads = {
                    name: np.random.randn(*value.shape).astype(np.float32) * 0.001
                    for name, value in params.items()
                }
                # Push
                await initialized_cluster.push(grads, worker_id, _ * worker_id)

        tasks = [worker_task(i) for i in range(5)]
        await asyncio.gather(*tasks)

        # Cluster should still be healthy
        health = await initialized_cluster.health_check()
        assert all(healthy for healthy in health.values())
