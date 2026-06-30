"""Tests for ParameterServer operations."""

import pytest
import numpy as np
import asyncio

from paramserver import (
    ParameterServer,
    ConsistencyModel,
    AggregationConfig,
    AggregationType,
    Parameter,
    Gradient,
    WorkerClient,
    create_server,
)


class TestParameterServerInitialization:
    """Tests for ParameterServer initialization."""

    def test_default_initialization(self):
        """Test server initializes with default settings."""
        server = ParameterServer()

        assert server.node_id is not None
        assert server.host == "localhost"
        assert server.port == 5000
        assert server.learning_rate == 0.01
        assert server.consistency_model == ConsistencyModel.BSP

    def test_custom_initialization(self):
        """Test server with custom settings."""
        server = ParameterServer(
            node_id="custom_node",
            host="0.0.0.0",
            port=8000,
            num_shards=8,
            consistency_model=ConsistencyModel.ASP,
            learning_rate=0.001,
        )

        assert server.node_id == "custom_node"
        assert server.host == "0.0.0.0"
        assert server.port == 8000
        assert server.learning_rate == 0.001
        assert server.consistency_model == ConsistencyModel.ASP

    def test_create_server_factory(self):
        """Test create_server factory function."""
        server = create_server(
            num_shards=8,
            consistency_model="asp",
            learning_rate=0.001,
        )

        assert server.consistency_model == ConsistencyModel.ASP
        assert server.learning_rate == 0.001

    def test_create_server_invalid_model(self):
        """Test create_server with invalid consistency model defaults to BSP."""
        server = create_server(consistency_model="invalid")
        assert server.consistency_model == ConsistencyModel.BSP


class TestParameterInitialization:
    """Tests for parameter initialization."""

    @pytest.mark.asyncio
    async def test_zeros_initialization(self, bsp_server):
        """Test parameter initialization with zeros."""
        param = await bsp_server.initialize_parameter(
            name="test.weight",
            shape=(10, 10),
            initializer="zeros",
        )

        assert param.name == "test.weight"
        assert param.shape == (10, 10)
        assert param.version == 0
        assert np.allclose(param.data, np.zeros((10, 10)))

    @pytest.mark.asyncio
    async def test_ones_initialization(self, bsp_server):
        """Test parameter initialization with ones."""
        param = await bsp_server.initialize_parameter(
            name="test.bias",
            shape=(10,),
            initializer="ones",
        )

        assert param.name == "test.bias"
        assert np.allclose(param.data, np.ones(10))

    @pytest.mark.asyncio
    async def test_random_initialization(self, bsp_server):
        """Test parameter initialization with random values."""
        param = await bsp_server.initialize_parameter(
            name="test.random",
            shape=(100, 100),
            initializer="random",
        )

        assert param.data is not None
        assert param.data.shape == (100, 100)
        # Random values should be small (scaled by 0.01)
        assert np.abs(param.data).max() < 0.1

    @pytest.mark.asyncio
    async def test_xavier_initialization(self, bsp_server):
        """Test parameter initialization with Xavier."""
        param = await bsp_server.initialize_parameter(
            name="test.xavier",
            shape=(256, 128),
            initializer="xavier",
        )

        assert param.data is not None
        assert param.data.shape == (256, 128)
        # Xavier init should have reasonable variance
        std = np.std(param.data)
        expected_std = np.sqrt(2.0 / (256 + 128))
        assert 0.5 * expected_std < std < 2.0 * expected_std

    @pytest.mark.asyncio
    async def test_invalid_initializer_defaults_to_zeros(self, bsp_server):
        """Test that unknown initializer defaults to zeros."""
        param = await bsp_server.initialize_parameter(
            name="test.unknown",
            shape=(5, 5),
            initializer="unknown_init",
        )

        assert np.allclose(param.data, np.zeros((5, 5)))


class TestPullOperations:
    """Tests for parameter pull operations."""

    @pytest.mark.asyncio
    async def test_basic_pull(self, bsp_server):
        """Test basic parameter pull."""
        bsp_server.register_worker("worker_1")

        await bsp_server.initialize_parameter("param1", (10, 10), "zeros")
        await bsp_server.initialize_parameter("param2", (20, 20), "zeros")

        result = await bsp_server.pull("worker_1", ["param1", "param2"])

        assert "param1" in result
        assert "param2" in result
        assert result["param1"].shape == (10, 10)
        assert result["param2"].shape == (20, 20)

    @pytest.mark.asyncio
    async def test_pull_nonexistent_parameter(self, bsp_server):
        """Test pull with non-existent parameter."""
        bsp_server.register_worker("worker_1")

        result = await bsp_server.pull("worker_1", ["nonexistent"])

        assert "nonexistent" not in result

    @pytest.mark.asyncio
    async def test_pull_increments_counter(self, bsp_server):
        """Test that pull increments pull counter."""
        bsp_server.register_worker("worker_1")
        await bsp_server.initialize_parameter("param1", (10, 10), "zeros")

        initial_count = bsp_server._pull_count

        await bsp_server.pull("worker_1", ["param1"])
        await bsp_server.pull("worker_1", ["param1"])

        assert bsp_server._pull_count == initial_count + 2

    @pytest.mark.asyncio
    async def test_pull_tracks_bytes(self, bsp_server):
        """Test that pull tracks bytes transferred."""
        bsp_server.register_worker("worker_1")
        await bsp_server.initialize_parameter("param1", (100, 100), "zeros")

        initial_bytes = bsp_server._total_bytes_transferred

        await bsp_server.pull("worker_1", ["param1"])

        # 100*100*4 bytes for float32
        expected_bytes = 100 * 100 * 4
        assert bsp_server._total_bytes_transferred >= initial_bytes + expected_bytes


class TestPushOperations:
    """Tests for gradient push operations."""

    @pytest.mark.asyncio
    async def test_basic_push_bsp(self, bsp_server):
        """Test basic push in BSP mode."""
        bsp_server.register_worker("worker_1")
        await bsp_server.initialize_parameter("param1", (10, 10), "zeros")

        gradient = Gradient(
            name="param1",
            data=np.ones((10, 10), dtype=np.float32),
            worker_id="worker_1",
            iteration=0,
        )

        result = await bsp_server.push("worker_1", [gradient], iteration=0)

        assert "param1" in result
        assert "pending" in result["param1"]

    @pytest.mark.asyncio
    async def test_push_asp_immediate_apply(self, asp_server):
        """Test that ASP mode applies gradients immediately."""
        asp_server.register_worker("worker_1")
        await asp_server.initialize_parameter("param1", (10, 10), "zeros")

        gradient = Gradient(
            name="param1",
            data=np.ones((10, 10), dtype=np.float32),
            worker_id="worker_1",
            iteration=0,
        )

        result = await asp_server.push("worker_1", [gradient], iteration=0)

        assert result["param1"]["applied"] is True
        assert result["param1"]["version"] == 1

    @pytest.mark.asyncio
    async def test_push_bsp_aggregation(self, server_with_workers):
        """Test BSP push aggregates all worker gradients."""
        server = server_with_workers
        await server.initialize_parameter("param1", (10, 10), "ones")

        # All 4 workers push gradients
        for i in range(4):
            gradient = Gradient(
                name="param1",
                data=np.full((10, 10), 0.1, dtype=np.float32),
                worker_id=f"worker_{i}",
                iteration=0,
            )
            result = await server.push(f"worker_{i}", [gradient], iteration=0)

        # After all workers push, gradient should be applied
        assert "applied" in result["param1"]
        assert result["param1"]["applied"] is True

    @pytest.mark.asyncio
    async def test_push_increments_counter(self, bsp_server):
        """Test that push increments push counter."""
        bsp_server.register_worker("worker_1")
        await bsp_server.initialize_parameter("param1", (10, 10), "zeros")

        initial_count = bsp_server._push_count

        gradient = Gradient(
            name="param1",
            data=np.ones((10, 10), dtype=np.float32),
            worker_id="worker_1",
            iteration=0,
        )

        await bsp_server.push("worker_1", [gradient], iteration=0)

        assert bsp_server._push_count == initial_count + 1


class TestWorkerManagement:
    """Tests for worker registration and management."""

    def test_register_worker(self, bsp_server):
        """Test worker registration."""
        bsp_server.register_worker("worker_1", "localhost", 6000)

        assert "worker_1" in bsp_server._workers
        assert bsp_server._expected_workers == 1

    def test_register_multiple_workers(self, bsp_server):
        """Test multiple worker registration."""
        for i in range(4):
            bsp_server.register_worker(f"worker_{i}")

        assert len(bsp_server._workers) == 4
        assert bsp_server._expected_workers == 4

    def test_deregister_worker(self, bsp_server):
        """Test worker deregistration."""
        bsp_server.register_worker("worker_1")
        bsp_server.register_worker("worker_2")

        bsp_server.deregister_worker("worker_1")

        assert "worker_1" not in bsp_server._workers
        assert "worker_2" in bsp_server._workers
        assert bsp_server._expected_workers == 1

    def test_deregister_nonexistent_worker(self, bsp_server):
        """Test deregistering non-existent worker doesn't raise."""
        bsp_server.register_worker("worker_1")

        # Should not raise
        bsp_server.deregister_worker("nonexistent")

        assert bsp_server._expected_workers == 1

    @pytest.mark.asyncio
    async def test_worker_heartbeat(self, bsp_server):
        """Test worker heartbeat."""
        bsp_server.register_worker("worker_1")

        result = await bsp_server.worker_heartbeat("worker_1")

        assert result is True

    @pytest.mark.asyncio
    async def test_heartbeat_nonexistent_worker(self, bsp_server):
        """Test heartbeat for non-existent worker."""
        result = await bsp_server.worker_heartbeat("nonexistent")

        assert result is False


class TestBarrierSynchronization:
    """Tests for barrier synchronization."""

    @pytest.mark.asyncio
    async def test_single_worker_barrier(self, bsp_server):
        """Test barrier with single worker."""
        bsp_server.register_worker("worker_1")

        result = await bsp_server.barrier("worker_1", iteration=0)

        assert result is True

    @pytest.mark.asyncio
    async def test_multiple_workers_barrier(self):
        """Test barrier with multiple workers."""
        server = ParameterServer(consistency_model=ConsistencyModel.BSP)
        for i in range(3):
            server.register_worker(f"worker_{i}")

        # Create async tasks for all workers
        async def worker_barrier(worker_id: str, iteration: int):
            return await server.barrier(worker_id, iteration)

        # All workers arrive at barrier
        results = await asyncio.gather(
            worker_barrier("worker_0", 0),
            worker_barrier("worker_1", 0),
            worker_barrier("worker_2", 0),
        )

        assert all(results)

    @pytest.mark.asyncio
    async def test_barrier_updates_global_iteration(self, bsp_server):
        """Test that barrier updates global iteration."""
        bsp_server.register_worker("worker_1")

        await bsp_server.barrier("worker_1", iteration=5)

        assert bsp_server.sync_manager.get_global_iteration() == 5


class TestServerStatistics:
    """Tests for server statistics."""

    def test_get_stats_empty_server(self, bsp_server):
        """Test stats on empty server."""
        stats = bsp_server.get_stats()

        assert "node_id" in stats
        assert stats["consistency_model"] == "bsp"
        assert stats["num_workers"] == 0
        assert stats["pull_count"] == 0
        assert stats["push_count"] == 0

    @pytest.mark.asyncio
    async def test_get_stats_with_data(self, bsp_server):
        """Test stats with data."""
        bsp_server.register_worker("worker_1")
        bsp_server.register_worker("worker_2")

        await bsp_server.initialize_parameter("param1", (100, 100), "zeros")
        await bsp_server.pull("worker_1", ["param1"])

        stats = bsp_server.get_stats()

        assert stats["num_workers"] == 2
        assert stats["pull_count"] == 1
        assert stats["num_parameters"] == 1


class TestWorkerClient:
    """Tests for WorkerClient."""

    @pytest.mark.asyncio
    async def test_client_pull(self, connected_worker_client, bsp_server):
        """Test client pull operation."""
        await bsp_server.initialize_parameter("param1", (10, 10), "zeros")

        result = await connected_worker_client.pull(["param1"])

        assert "param1" in result

    @pytest.mark.asyncio
    async def test_client_push(self, connected_worker_client, bsp_server):
        """Test client push operation."""
        await bsp_server.initialize_parameter("param1", (10, 10), "zeros")

        gradient = Gradient(
            name="param1",
            data=np.ones((10, 10), dtype=np.float32),
            worker_id="worker_1",
            iteration=0,
        )

        result = await connected_worker_client.push([gradient], iteration=0)

        assert "param1" in result

    @pytest.mark.asyncio
    async def test_client_barrier(self, connected_worker_client, bsp_server):
        """Test client barrier operation."""
        result = await connected_worker_client.barrier(iteration=0)

        assert result is True

    @pytest.mark.asyncio
    async def test_client_heartbeat(self, connected_worker_client, bsp_server):
        """Test client heartbeat operation."""
        result = await connected_worker_client.heartbeat()

        assert result is True


class TestConcurrentOperations:
    """Tests for concurrent server operations."""

    @pytest.mark.asyncio
    async def test_concurrent_pulls(self, bsp_server):
        """Test concurrent pull operations."""
        for i in range(4):
            bsp_server.register_worker(f"worker_{i}")

        await bsp_server.initialize_parameter("param1", (100, 100), "random")

        async def pull_worker(worker_id: str):
            return await bsp_server.pull(worker_id, ["param1"])

        results = await asyncio.gather(*[
            pull_worker(f"worker_{i}") for i in range(4)
        ])

        assert len(results) == 4
        for result in results:
            assert "param1" in result

    @pytest.mark.asyncio
    async def test_concurrent_pushes(self, server_with_workers):
        """Test concurrent push operations."""
        server = server_with_workers
        await server.initialize_parameter("param1", (50, 50), "zeros")

        async def push_worker(worker_id: str, value: float):
            gradient = Gradient(
                name="param1",
                data=np.full((50, 50), value, dtype=np.float32),
                worker_id=worker_id,
                iteration=0,
            )
            return await server.push(worker_id, [gradient], iteration=0)

        results = await asyncio.gather(*[
            push_worker(f"worker_{i}", float(i)) for i in range(4)
        ])

        assert len(results) == 4

    @pytest.mark.asyncio
    async def test_concurrent_pull_push(self, bsp_server):
        """Test concurrent pull and push operations."""
        bsp_server.register_worker("reader")
        bsp_server.register_worker("writer")

        await bsp_server.initialize_parameter("shared_param", (20, 20), "zeros")

        async def reader():
            for _ in range(10):
                await bsp_server.pull("reader", ["shared_param"])
                await asyncio.sleep(0.01)

        async def writer():
            for i in range(10):
                gradient = Gradient(
                    name="shared_param",
                    data=np.ones((20, 20), dtype=np.float32) * i,
                    worker_id="writer",
                    iteration=i,
                )
                await bsp_server.push("writer", [gradient], iteration=i)
                await asyncio.sleep(0.01)

        await asyncio.gather(reader(), writer())

        # Should complete without errors
        assert bsp_server._pull_count >= 10
        assert bsp_server._push_count >= 10


class TestGradientApplication:
    """Tests for gradient application correctness."""

    @pytest.mark.asyncio
    async def test_gradient_descent_step(self, asp_server):
        """Test that gradient descent updates parameter correctly."""
        asp_server.register_worker("worker_1")

        # Initialize parameter with ones
        await asp_server.initialize_parameter("param1", (5, 5), "ones")

        # Push gradient of all ones
        gradient = Gradient(
            name="param1",
            data=np.ones((5, 5), dtype=np.float32),
            worker_id="worker_1",
            iteration=0,
        )

        await asp_server.push("worker_1", [gradient], iteration=0)

        # Pull updated parameter
        result = await asp_server.pull("worker_1", ["param1"])

        # param = param - lr * gradient = 1 - 0.01 * 1 = 0.99
        expected = np.full((5, 5), 0.99, dtype=np.float32)
        assert np.allclose(result["param1"].data, expected, atol=1e-5)

    @pytest.mark.asyncio
    async def test_multiple_gradient_steps(self, asp_server):
        """Test multiple gradient steps."""
        asp_server.register_worker("worker_1")

        await asp_server.initialize_parameter("param1", (3, 3), "zeros")

        for i in range(10):
            gradient = Gradient(
                name="param1",
                data=np.ones((3, 3), dtype=np.float32),
                worker_id="worker_1",
                iteration=i,
            )
            await asp_server.push("worker_1", [gradient], iteration=i)

        result = await asp_server.pull("worker_1", ["param1"])

        # After 10 steps: 0 - 10 * 0.01 * 1 = -0.1
        expected = np.full((3, 3), -0.1, dtype=np.float32)
        assert np.allclose(result["param1"].data, expected, atol=1e-5)
