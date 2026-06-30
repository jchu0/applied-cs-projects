"""Tests for ParameterServer."""

import pytest
import numpy as np

from paramserver.server.parameter_server import ParameterServer
from paramserver.optimizer.sgd import SGDEngine
from paramserver.consistency.hogwild import HogwildConsistency
from paramserver.schemas import ServerStatus


class TestParameterServerInit:
    """Tests for ParameterServer initialization."""

    def test_create_server(self, sgd_engine, hogwild_consistency):
        """Test server creation."""
        server = ParameterServer(
            shard_id=0,
            update_engine=sgd_engine,
            consistency=hogwild_consistency,
        )
        assert server.shard_id == 0
        assert server.status == ServerStatus.READY

    @pytest.mark.asyncio
    async def test_initialize(self, parameter_server, small_params):
        """Test parameter initialization."""
        await parameter_server.initialize(small_params)

        for name, value in small_params.items():
            stored = await parameter_server.get_param(name)
            np.testing.assert_array_equal(stored, value)

    @pytest.mark.asyncio
    async def test_initialize_creates_metadata(self, parameter_server, small_params):
        """Test that metadata is created on init."""
        await parameter_server.initialize(small_params)

        for name in small_params.keys():
            meta = parameter_server.get_metadata(name)
            assert meta is not None
            assert meta.name == name
            assert meta.version == 0


class TestParameterServerPull:
    """Tests for pull operations."""

    @pytest.mark.asyncio
    async def test_pull_single_param(self, parameter_server, small_params):
        """Test pulling a single parameter."""
        await parameter_server.initialize(small_params)

        result = await parameter_server.pull(["w1"], worker_id=0)
        assert "w1" in result
        np.testing.assert_array_equal(result["w1"][0], small_params["w1"])

    @pytest.mark.asyncio
    async def test_pull_multiple_params(self, parameter_server, small_params):
        """Test pulling multiple parameters."""
        await parameter_server.initialize(small_params)

        result = await parameter_server.pull(list(small_params.keys()), worker_id=0)
        assert len(result) == len(small_params)

    @pytest.mark.asyncio
    async def test_pull_includes_versions(self, parameter_server, small_params):
        """Test that pull includes version numbers."""
        await parameter_server.initialize(small_params)

        result = await parameter_server.pull(["w1"], worker_id=0, include_versions=True)
        value, version = result["w1"]
        assert version == 0

    @pytest.mark.asyncio
    async def test_pull_nonexistent_param(self, parameter_server, small_params):
        """Test pulling a parameter that doesn't exist."""
        await parameter_server.initialize(small_params)

        result = await parameter_server.pull(["nonexistent"], worker_id=0)
        assert "nonexistent" not in result

    @pytest.mark.asyncio
    async def test_pull_increments_counter(self, parameter_server, small_params):
        """Test that pull increments statistics."""
        await parameter_server.initialize(small_params)
        assert parameter_server._total_pulls == 0

        await parameter_server.pull(["w1"], worker_id=0)
        assert parameter_server._total_pulls == 1


class TestParameterServerPush:
    """Tests for push operations."""

    @pytest.mark.asyncio
    async def test_push_single_gradient(self, parameter_server, small_params):
        """Test pushing a single gradient."""
        await parameter_server.initialize(small_params)

        grads = {"w1": np.array([0.1, 0.2, 0.3], dtype=np.float32)}
        applied = await parameter_server.push(grads, worker_id=0, clock=0)

        assert applied == 1

        # Check param was updated
        new_value = await parameter_server.get_param("w1")
        # With lr=0.01: new = old - 0.01 * grad
        expected = small_params["w1"] - 0.01 * grads["w1"]
        np.testing.assert_array_almost_equal(new_value, expected)

    @pytest.mark.asyncio
    async def test_push_multiple_gradients(self, parameter_server, small_params):
        """Test pushing multiple gradients."""
        await parameter_server.initialize(small_params)

        grads = {
            "w1": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "w2": np.array([0.4, 0.5], dtype=np.float32),
        }
        applied = await parameter_server.push(grads, worker_id=0, clock=0)

        assert applied == 2

    @pytest.mark.asyncio
    async def test_push_increments_version(self, parameter_server, small_params):
        """Test that push increments parameter version."""
        await parameter_server.initialize(small_params)

        grads = {"w1": np.array([0.1, 0.2, 0.3], dtype=np.float32)}
        await parameter_server.push(grads, worker_id=0, clock=0)

        meta = parameter_server.get_metadata("w1")
        assert meta.version == 1
        assert meta.total_updates == 1
        assert meta.last_update_worker == 0

    @pytest.mark.asyncio
    async def test_push_nonexistent_param(self, parameter_server, small_params):
        """Test pushing gradient for nonexistent param."""
        await parameter_server.initialize(small_params)

        grads = {"nonexistent": np.array([0.1])}
        applied = await parameter_server.push(grads, worker_id=0, clock=0)

        assert applied == 0

    @pytest.mark.asyncio
    async def test_push_increments_counter(self, parameter_server, small_params):
        """Test that push increments statistics."""
        await parameter_server.initialize(small_params)
        assert parameter_server._total_pushes == 0

        grads = {"w1": np.array([0.1, 0.2, 0.3], dtype=np.float32)}
        await parameter_server.push(grads, worker_id=0, clock=0)

        assert parameter_server._total_pushes == 1


class TestParameterServerDirectAccess:
    """Tests for direct parameter access."""

    @pytest.mark.asyncio
    async def test_get_param(self, parameter_server, small_params):
        """Test getting a single parameter."""
        await parameter_server.initialize(small_params)

        value = await parameter_server.get_param("w1")
        np.testing.assert_array_equal(value, small_params["w1"])

    @pytest.mark.asyncio
    async def test_get_nonexistent_param(self, parameter_server, small_params):
        """Test getting a nonexistent parameter."""
        await parameter_server.initialize(small_params)

        value = await parameter_server.get_param("nonexistent")
        assert value is None

    @pytest.mark.asyncio
    async def test_set_param(self, parameter_server, small_params):
        """Test setting a parameter directly."""
        await parameter_server.initialize(small_params)

        new_value = np.array([9.0, 9.0, 9.0], dtype=np.float32)
        await parameter_server.set_param("w1", new_value)

        stored = await parameter_server.get_param("w1")
        np.testing.assert_array_equal(stored, new_value)

    @pytest.mark.asyncio
    async def test_set_new_param(self, parameter_server, small_params):
        """Test adding a new parameter."""
        await parameter_server.initialize(small_params)

        new_value = np.array([1.0, 2.0], dtype=np.float32)
        await parameter_server.set_param("new_param", new_value)

        stored = await parameter_server.get_param("new_param")
        np.testing.assert_array_equal(stored, new_value)

    @pytest.mark.asyncio
    async def test_get_all_params(self, parameter_server, small_params):
        """Test getting all parameters."""
        await parameter_server.initialize(small_params)

        all_params = parameter_server.get_all_params()
        assert len(all_params) == len(small_params)

        for name, value in small_params.items():
            np.testing.assert_array_equal(all_params[name], value)


class TestParameterServerProperties:
    """Tests for server properties and statistics."""

    @pytest.mark.asyncio
    async def test_param_names(self, parameter_server, small_params):
        """Test param_names property."""
        await parameter_server.initialize(small_params)

        names = parameter_server.param_names
        assert set(names) == set(small_params.keys())

    @pytest.mark.asyncio
    async def test_total_params(self, parameter_server, small_params):
        """Test total_params property."""
        await parameter_server.initialize(small_params)

        expected = sum(p.size for p in small_params.values())
        assert parameter_server.total_params == expected

    @pytest.mark.asyncio
    async def test_stats(self, parameter_server, small_params):
        """Test stats property."""
        await parameter_server.initialize(small_params)

        # Do some operations
        await parameter_server.pull(["w1"], worker_id=0)
        grads = {"w1": np.array([0.1, 0.2, 0.3], dtype=np.float32)}
        await parameter_server.push(grads, worker_id=0, clock=0)

        stats = parameter_server.stats
        assert stats["shard_id"] == 0
        assert stats["num_params"] == len(small_params)
        assert stats["total_pulls"] == 1
        assert stats["total_pushes"] == 1
        assert stats["status"] == "ready"

    @pytest.mark.asyncio
    async def test_health_check(self, parameter_server, small_params):
        """Test health check."""
        await parameter_server.initialize(small_params)

        health = await parameter_server.health_check()
        assert health is True

        parameter_server.status = ServerStatus.FAILED
        health = await parameter_server.health_check()
        assert health is False


class TestParameterServerConcurrency:
    """Tests for concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_pulls(self, parameter_server, small_params):
        """Test concurrent pull operations."""
        import asyncio

        await parameter_server.initialize(small_params)

        # Multiple concurrent pulls
        tasks = [
            parameter_server.pull(["w1"], worker_id=i)
            for i in range(10)
        ]
        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        for result in results:
            assert "w1" in result

    @pytest.mark.asyncio
    async def test_concurrent_pushes(self, parameter_server, small_params):
        """Test concurrent push operations."""
        import asyncio

        await parameter_server.initialize(small_params)

        grads = {"w1": np.array([0.01, 0.01, 0.01], dtype=np.float32)}

        # Multiple concurrent pushes
        tasks = [
            parameter_server.push(grads, worker_id=i, clock=i)
            for i in range(10)
        ]
        results = await asyncio.gather(*tasks)

        assert sum(results) == 10

        # Version should be incremented for each push
        meta = parameter_server.get_metadata("w1")
        assert meta.version == 10
