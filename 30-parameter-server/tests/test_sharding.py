"""Tests for sharding strategies."""

import pytest
import numpy as np

from paramserver.server.sharding import (
    ShardingStrategy,
    UniformSharding,
    RoundRobinSharding,
    SizeBalancedSharding,
)


class TestUniformSharding:
    """Tests for UniformSharding strategy."""

    def test_compute_shards_basic(self, sample_params):
        """Test basic shard computation."""
        sharding = UniformSharding()
        shards = sharding.compute_shards(sample_params, num_servers=3)

        assert len(shards) == 3
        assert all(s.shard_id in [0, 1, 2] for s in shards)

        # All params should be assigned
        all_params = set()
        for shard in shards:
            all_params.update(shard.param_names)
        assert all_params == set(sample_params.keys())

    def test_shard_for_param(self, sample_params):
        """Test parameter to shard mapping."""
        sharding = UniformSharding()
        sharding.compute_shards(sample_params, num_servers=3)

        for name in sample_params.keys():
            shard_id = sharding.get_shard_for_param(name)
            assert shard_id in [0, 1, 2]

    def test_consistent_mapping(self, sample_params):
        """Test that param mapping is consistent."""
        sharding = UniformSharding()
        sharding.compute_shards(sample_params, num_servers=3)

        # Same param should always map to same shard
        for name in sample_params.keys():
            shard1 = sharding.get_shard_for_param(name)
            shard2 = sharding.get_shard_for_param(name)
            assert shard1 == shard2

    def test_params_for_shard(self, sample_params):
        """Test getting params for a shard."""
        sharding = UniformSharding()
        shards = sharding.compute_shards(sample_params, num_servers=3)

        for shard in shards:
            params = sharding.get_params_for_shard(shard.shard_id)
            assert params == shard.param_names

    def test_empty_shard(self):
        """Test with fewer params than servers."""
        sharding = UniformSharding()
        params = {"w1": np.array([1.0])}
        shards = sharding.compute_shards(params, num_servers=5)

        # All shards should exist, but most will be empty
        assert len(shards) == 5
        non_empty = [s for s in shards if s.param_names]
        assert len(non_empty) == 1

    def test_memory_calculation(self):
        """Test memory bytes calculation."""
        sharding = UniformSharding()
        params = {
            "w1": np.zeros(100, dtype=np.float32),
            "w2": np.zeros(200, dtype=np.float32),
        }
        shards = sharding.compute_shards(params, num_servers=2)

        total_memory = sum(s.memory_bytes for s in shards)
        # 300 floats * 4 bytes = 1200 bytes
        assert total_memory == 1200

    def test_num_servers_not_set(self):
        """Test error when num_servers not set."""
        sharding = UniformSharding()

        with pytest.raises(ValueError, match="num_servers must be set"):
            sharding.get_shard_for_param("unknown_param")


class TestRoundRobinSharding:
    """Tests for RoundRobinSharding strategy."""

    def test_compute_shards(self, sample_params):
        """Test round-robin distribution."""
        sharding = RoundRobinSharding()
        shards = sharding.compute_shards(sample_params, num_servers=3)

        assert len(shards) == 3

        # Params should be distributed round-robin
        all_params = set()
        for shard in shards:
            all_params.update(shard.param_names)
        assert all_params == set(sample_params.keys())

    def test_sequential_assignment(self):
        """Test that params are assigned sequentially."""
        sharding = RoundRobinSharding()
        params = {f"p{i}": np.array([i]) for i in range(6)}
        shards = sharding.compute_shards(params, num_servers=3)

        # With 6 params and 3 servers, each gets 2
        for shard in shards:
            assert len(shard.param_names) == 2

    def test_unknown_param(self):
        """Test error for unknown parameter."""
        sharding = RoundRobinSharding()
        params = {"w1": np.array([1.0])}
        sharding.compute_shards(params, num_servers=2)

        with pytest.raises(ValueError, match="Unknown parameter"):
            sharding.get_shard_for_param("unknown")


class TestSizeBalancedSharding:
    """Tests for SizeBalancedSharding strategy."""

    def test_compute_shards(self, sample_params):
        """Test size-balanced distribution."""
        sharding = SizeBalancedSharding()
        shards = sharding.compute_shards(sample_params, num_servers=3)

        assert len(shards) == 3

        # All params assigned
        all_params = set()
        for shard in shards:
            all_params.update(shard.param_names)
        assert all_params == set(sample_params.keys())

    def test_balances_by_size(self):
        """Test that large params are spread out."""
        sharding = SizeBalancedSharding()
        params = {
            "large1": np.zeros(1000),
            "large2": np.zeros(1000),
            "small1": np.zeros(10),
            "small2": np.zeros(10),
        }
        shards = sharding.compute_shards(params, num_servers=2)

        # Each shard should get one large and one small
        sizes = [s.total_params for s in shards]
        # The difference should be small
        assert abs(sizes[0] - sizes[1]) <= 990  # Rough balance

    def test_single_server(self):
        """Test with single server."""
        sharding = SizeBalancedSharding()
        params = {"w1": np.array([1.0]), "w2": np.array([2.0])}
        shards = sharding.compute_shards(params, num_servers=1)

        assert len(shards) == 1
        assert len(shards[0].param_names) == 2

    def test_get_params_for_shard(self):
        """Test getting params for a shard."""
        sharding = SizeBalancedSharding()
        params = {"w1": np.array([1.0]), "w2": np.array([2.0])}
        sharding.compute_shards(params, num_servers=2)

        # Each shard should have at least one param
        all_retrieved = []
        for i in range(2):
            retrieved = sharding.get_params_for_shard(i)
            all_retrieved.extend(retrieved)

        assert set(all_retrieved) == set(params.keys())


class TestShardingEdgeCases:
    """Edge case tests for sharding strategies."""

    @pytest.mark.parametrize("strategy_cls", [
        UniformSharding,
        RoundRobinSharding,
        SizeBalancedSharding,
    ])
    def test_single_param(self, strategy_cls):
        """Test with single parameter."""
        sharding = strategy_cls()
        params = {"w": np.array([1.0, 2.0])}
        shards = sharding.compute_shards(params, num_servers=3)

        # All shards exist, but only one has the param
        assert len(shards) == 3
        param_count = sum(len(s.param_names) for s in shards)
        assert param_count == 1

    @pytest.mark.parametrize("strategy_cls", [
        UniformSharding,
        RoundRobinSharding,
        SizeBalancedSharding,
    ])
    def test_many_params(self, strategy_cls):
        """Test with many parameters."""
        sharding = strategy_cls()
        params = {f"w{i}": np.random.randn(10) for i in range(100)}
        shards = sharding.compute_shards(params, num_servers=10)

        # All params should be assigned
        all_params = set()
        for shard in shards:
            all_params.update(shard.param_names)
        assert len(all_params) == 100

    @pytest.mark.parametrize("strategy_cls", [
        UniformSharding,
        RoundRobinSharding,
        SizeBalancedSharding,
    ])
    def test_varying_param_sizes(self, strategy_cls):
        """Test with varying parameter sizes."""
        sharding = strategy_cls()
        params = {
            "tiny": np.zeros(1),
            "small": np.zeros(10),
            "medium": np.zeros(100),
            "large": np.zeros(1000),
            "huge": np.zeros(10000),
        }
        shards = sharding.compute_shards(params, num_servers=3)

        total_params = sum(s.total_params for s in shards)
        expected_total = sum(p.size for p in params.values())
        assert total_params == expected_total
