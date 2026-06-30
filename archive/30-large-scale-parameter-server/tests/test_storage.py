"""Tests for storage and sharding components."""

import pytest
import numpy as np
import asyncio

from paramserver import (
    ParameterStore,
    ShardManager,
    ParameterPartitioner,
    PartitionStrategy,
    Parameter,
    Gradient,
)


class TestParameterStore:
    """Tests for ParameterStore."""

    @pytest.mark.asyncio
    async def test_set_and_get(self, parameter_store, sample_parameter):
        """Test basic set and get operations."""
        await parameter_store.set(sample_parameter)

        result = await parameter_store.get(sample_parameter.name)

        assert result is not None
        assert result.name == sample_parameter.name
        assert np.array_equal(result.data, sample_parameter.data)

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, parameter_store):
        """Test get returns None for nonexistent parameter."""
        result = await parameter_store.get("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_many(self, parameter_store, sample_parameters):
        """Test get_many for multiple parameters."""
        for param in sample_parameters:
            await parameter_store.set(param)

        names = [p.name for p in sample_parameters[:3]]
        result = await parameter_store.get_many(names)

        assert len(result) == 3
        for name in names:
            assert name in result

    @pytest.mark.asyncio
    async def test_get_many_partial(self, parameter_store, sample_parameters):
        """Test get_many with some nonexistent parameters."""
        await parameter_store.set(sample_parameters[0])

        result = await parameter_store.get_many(["layer0.weight", "nonexistent"])

        assert "layer0.weight" in result
        assert "nonexistent" not in result

    @pytest.mark.asyncio
    async def test_update(self, parameter_store, sample_parameter):
        """Test update operation."""
        await parameter_store.set(sample_parameter)

        new_data = np.ones_like(sample_parameter.data) * 2.0
        result = await parameter_store.update(sample_parameter.name, new_data)

        assert result is not None
        assert result.version == 1  # Version should increment
        assert np.array_equal(result.data, new_data)

    @pytest.mark.asyncio
    async def test_update_nonexistent(self, parameter_store):
        """Test update returns None for nonexistent parameter."""
        result = await parameter_store.update("nonexistent", np.zeros(10))

        assert result is None

    @pytest.mark.asyncio
    async def test_apply_gradient(self, parameter_store):
        """Test applying gradient to parameter."""
        param = Parameter(
            name="test",
            shape=(10,),
            data=np.ones(10, dtype=np.float32),
        )
        await parameter_store.set(param)

        gradient = np.ones(10, dtype=np.float32)
        result = await parameter_store.apply_gradient("test", gradient, learning_rate=0.1)

        assert result is not None
        assert result.version == 1
        # new_value = old_value - lr * gradient = 1 - 0.1 * 1 = 0.9
        expected = np.full(10, 0.9, dtype=np.float32)
        assert np.allclose(result.data, expected)

    @pytest.mark.asyncio
    async def test_apply_gradient_nonexistent(self, parameter_store):
        """Test apply_gradient returns None for nonexistent parameter."""
        gradient = np.ones(10, dtype=np.float32)
        result = await parameter_store.apply_gradient("nonexistent", gradient)

        assert result is None

    @pytest.mark.asyncio
    async def test_get_version(self, parameter_store, sample_parameter):
        """Test getting parameter version."""
        sample_parameter.version = 5
        await parameter_store.set(sample_parameter)

        version = await parameter_store.get_version(sample_parameter.name)

        assert version == 5

    @pytest.mark.asyncio
    async def test_get_version_nonexistent(self, parameter_store):
        """Test get_version returns -1 for nonexistent parameter."""
        version = await parameter_store.get_version("nonexistent")

        assert version == -1

    @pytest.mark.asyncio
    async def test_version_history(self, parameter_store):
        """Test that history is maintained across updates."""
        param = Parameter(
            name="test",
            shape=(5,),
            data=np.zeros(5, dtype=np.float32),
        )
        await parameter_store.set(param)

        # Perform multiple updates
        for i in range(3):
            new_data = np.full(5, float(i + 1), dtype=np.float32)
            await parameter_store.update("test", new_data)

        # Check history exists
        assert "test" in parameter_store._history
        assert len(parameter_store._history["test"]) == 3

    @pytest.mark.asyncio
    async def test_rollback(self, parameter_store):
        """Test rollback functionality."""
        param = Parameter(
            name="test",
            shape=(5,),
            data=np.zeros(5, dtype=np.float32),
        )
        await parameter_store.set(param)

        # Update with new values
        await parameter_store.update("test", np.ones(5, dtype=np.float32))
        await parameter_store.update("test", np.full(5, 2.0, dtype=np.float32))

        # Rollback one step
        success = await parameter_store.rollback("test", steps=1)

        assert success is True
        result = await parameter_store.get("test")
        assert np.allclose(result.data, np.ones(5))

    @pytest.mark.asyncio
    async def test_rollback_multiple_steps(self, parameter_store):
        """Test multi-step rollback."""
        param = Parameter(
            name="test",
            shape=(5,),
            data=np.zeros(5, dtype=np.float32),
        )
        await parameter_store.set(param)

        # Multiple updates
        for i in range(5):
            await parameter_store.update("test", np.full(5, float(i + 1), dtype=np.float32))

        # Rollback 3 steps
        success = await parameter_store.rollback("test", steps=3)

        assert success is True
        result = await parameter_store.get("test")
        # Should be at value 2 (rolled back from 5 -> 4 -> 3 -> 2)
        assert np.allclose(result.data, np.full(5, 2.0))

    @pytest.mark.asyncio
    async def test_rollback_insufficient_history(self, parameter_store):
        """Test rollback fails with insufficient history."""
        param = Parameter(
            name="test",
            shape=(5,),
            data=np.zeros(5, dtype=np.float32),
        )
        await parameter_store.set(param)
        await parameter_store.update("test", np.ones(5, dtype=np.float32))

        # Try to rollback more than history
        success = await parameter_store.rollback("test", steps=10)

        assert success is False

    @pytest.mark.asyncio
    async def test_rollback_no_history(self, parameter_store):
        """Test rollback fails with no history."""
        success = await parameter_store.rollback("nonexistent", steps=1)

        assert success is False

    def test_list_parameters(self, parameter_store):
        """Test listing parameter names."""
        # Manually add parameters
        for i in range(3):
            param = Parameter(
                name=f"param_{i}",
                shape=(10,),
                data=np.zeros(10, dtype=np.float32),
            )
            parameter_store._parameters[param.name] = param

        names = parameter_store.list_parameters()

        assert len(names) == 3
        assert set(names) == {"param_0", "param_1", "param_2"}

    def test_get_stats(self, parameter_store):
        """Test getting storage statistics."""
        # Add some parameters
        for i in range(3):
            param = Parameter(
                name=f"param_{i}",
                shape=(100, 100),
                data=np.zeros((100, 100), dtype=np.float32),
            )
            parameter_store._parameters[param.name] = param

        stats = parameter_store.get_stats()

        assert stats["num_parameters"] == 3
        # 3 params * 100 * 100 * 4 bytes
        expected_bytes = 3 * 100 * 100 * 4
        assert stats["total_size_bytes"] == expected_bytes

    @pytest.mark.asyncio
    async def test_concurrent_updates(self, parameter_store):
        """Test concurrent updates to same parameter."""
        param = Parameter(
            name="shared",
            shape=(10,),
            data=np.zeros(10, dtype=np.float32),
        )
        await parameter_store.set(param)

        async def update_task(value: float):
            new_data = np.full(10, value, dtype=np.float32)
            await parameter_store.update("shared", new_data)

        # Run concurrent updates
        await asyncio.gather(*[update_task(float(i)) for i in range(10)])

        # Should complete without errors
        result = await parameter_store.get("shared")
        assert result is not None


class TestParameterPartitioner:
    """Tests for ParameterPartitioner."""

    def test_hash_partition(self, partitioner):
        """Test hash partitioning."""
        shard_id = partitioner.get_shard("layer1.weight")

        assert 0 <= shard_id < 4

    def test_hash_partition_consistency(self, partitioner):
        """Test that hash partitioning is consistent."""
        name = "encoder.attention.weight"

        shard_ids = [partitioner.get_shard(name) for _ in range(100)]

        assert all(s == shard_ids[0] for s in shard_ids)

    def test_hash_partition_distribution(self, partitioner):
        """Test that hash partitioning distributes evenly."""
        names = [f"param_{i}" for i in range(1000)]
        shard_counts = {i: 0 for i in range(4)}

        for name in names:
            shard_id = partitioner.get_shard(name)
            shard_counts[shard_id] += 1

        # Each shard should have roughly 250 params (1000/4)
        for count in shard_counts.values():
            assert 150 < count < 350  # Allow reasonable variance

    def test_range_partition(self, range_partitioner):
        """Test range partitioning."""
        # Parameters starting with 'a' should be in shard 0
        shard_a = range_partitioner.get_shard("attention.weight")
        # Parameters starting with 'z' should be in higher shard
        shard_z = range_partitioner.get_shard("zero_layer.weight")

        assert shard_a <= shard_z

    def test_range_partition_alphabetical(self, range_partitioner):
        """Test range partitioning is alphabetically based."""
        names_by_shard = {}

        for letter in "abcdefghijklmnopqrstuvwxyz":
            name = f"{letter}_param"
            shard = range_partitioner.get_shard(name)
            if shard not in names_by_shard:
                names_by_shard[shard] = []
            names_by_shard[shard].append(letter)

        # Check that within each shard, letters are consecutive
        for letters in names_by_shard.values():
            if len(letters) > 1:
                ords = [ord(l) for l in letters]
                assert max(ords) - min(ords) == len(letters) - 1

    def test_round_robin_partition(self, round_robin_partitioner):
        """Test round-robin partitioning."""
        shard_id = round_robin_partitioner.get_shard("some_param")

        assert 0 <= shard_id < 4

    def test_round_robin_consistency(self, round_robin_partitioner):
        """Test round-robin is consistent for same name."""
        name = "consistent_param"

        shard_ids = [round_robin_partitioner.get_shard(name) for _ in range(10)]

        assert all(s == shard_ids[0] for s in shard_ids)


class TestShardManager:
    """Tests for ShardManager."""

    def test_initialization(self, shard_manager):
        """Test shard manager initialization."""
        assert shard_manager.num_shards == 4
        assert len(shard_manager._shards) == 4

    def test_store_and_get(self, shard_manager, sample_parameter):
        """Test storing and retrieving parameter from shard."""
        shard_manager.store(0, sample_parameter)

        result = shard_manager.get(0, sample_parameter.name)

        assert result is not None
        assert result.name == sample_parameter.name

    def test_store_invalid_shard(self, shard_manager, sample_parameter):
        """Test storing to invalid shard raises error."""
        with pytest.raises(ValueError, match="Invalid shard ID"):
            shard_manager.store(999, sample_parameter)

    def test_get_from_wrong_shard(self, shard_manager, sample_parameter):
        """Test get returns None if parameter not in shard."""
        shard_manager.store(0, sample_parameter)

        result = shard_manager.get(1, sample_parameter.name)

        assert result is None

    def test_get_invalid_shard(self, shard_manager):
        """Test get from invalid shard returns None."""
        result = shard_manager.get(999, "some_param")

        assert result is None

    def test_get_all(self, shard_manager, sample_parameters):
        """Test getting all parameters from a shard."""
        # Store multiple parameters in shard 0
        for param in sample_parameters[:3]:
            shard_manager.store(0, param)

        result = shard_manager.get_all(0)

        assert len(result) == 3

    def test_get_all_empty_shard(self, shard_manager):
        """Test get_all on empty shard returns empty list."""
        result = shard_manager.get_all(0)

        assert result == []

    def test_get_all_invalid_shard(self, shard_manager):
        """Test get_all on invalid shard returns empty list."""
        result = shard_manager.get_all(999)

        assert result == []

    def test_delete(self, shard_manager, sample_parameter):
        """Test deleting parameter from shard."""
        shard_manager.store(0, sample_parameter)

        success = shard_manager.delete(0, sample_parameter.name)

        assert success is True
        assert shard_manager.get(0, sample_parameter.name) is None

    def test_delete_nonexistent(self, shard_manager):
        """Test deleting nonexistent parameter returns False."""
        success = shard_manager.delete(0, "nonexistent")

        assert success is False

    def test_delete_invalid_shard(self, shard_manager):
        """Test delete from invalid shard returns False."""
        success = shard_manager.delete(999, "some_param")

        assert success is False

    def test_get_shard_info(self, shard_manager, sample_parameters):
        """Test getting shard information."""
        for param in sample_parameters[:2]:
            shard_manager.store(0, param)

        info = shard_manager.get_shard_info(0)

        assert info.shard_id == 0
        assert info.num_parameters == 2
        assert len(info.parameter_names) == 2

    def test_get_total_size(self, shard_manager, sample_parameters):
        """Test getting total size across all shards."""
        for i, param in enumerate(sample_parameters):
            shard_manager.store(i % 4, param)

        total_size = shard_manager.get_total_size()

        expected_size = sum(
            p.data.nbytes for p in sample_parameters if p.data is not None
        )
        assert total_size == expected_size

    def test_shard_size_tracking(self, shard_manager):
        """Test that shard sizes are tracked correctly."""
        param1 = Parameter(
            name="param1",
            shape=(100, 100),
            data=np.zeros((100, 100), dtype=np.float32),
        )
        param2 = Parameter(
            name="param2",
            shape=(50, 50),
            data=np.zeros((50, 50), dtype=np.float32),
        )

        shard_manager.store(0, param1)
        shard_manager.store(0, param2)

        info = shard_manager.get_shard_info(0)
        expected_size = 100 * 100 * 4 + 50 * 50 * 4

        assert info.size_bytes == expected_size

    def test_shard_size_updates_on_replacement(self, shard_manager):
        """Test that shard size updates when parameter is replaced."""
        param_small = Parameter(
            name="param",
            shape=(10, 10),
            data=np.zeros((10, 10), dtype=np.float32),
        )
        param_large = Parameter(
            name="param",
            shape=(100, 100),
            data=np.zeros((100, 100), dtype=np.float32),
        )

        shard_manager.store(0, param_small)
        initial_size = shard_manager._shard_sizes[0]

        shard_manager.store(0, param_large)
        new_size = shard_manager._shard_sizes[0]

        assert new_size > initial_size
        assert new_size == 100 * 100 * 4

    def test_shard_size_decreases_on_delete(self, shard_manager):
        """Test that shard size decreases on delete."""
        param = Parameter(
            name="param",
            shape=(100, 100),
            data=np.zeros((100, 100), dtype=np.float32),
        )

        shard_manager.store(0, param)
        assert shard_manager._shard_sizes[0] == 100 * 100 * 4

        shard_manager.delete(0, "param")
        assert shard_manager._shard_sizes[0] == 0

    def test_rebalance_shards_no_imbalance(self, shard_manager, sample_parameters):
        """Test rebalancing with balanced shards."""
        # Distribute evenly
        for i, param in enumerate(sample_parameters):
            shard_manager.store(i % 4, param)

        moves = shard_manager.rebalance_shards()

        # With even distribution, no moves needed
        assert len(moves) == 0 or sum(len(v) for v in moves.values()) < 2

    def test_rebalance_shards_with_imbalance(self, shard_manager):
        """Test rebalancing with imbalanced shards."""
        # Create imbalance: all in shard 0
        for i in range(10):
            param = Parameter(
                name=f"param_{i}",
                shape=(100, 100),
                data=np.zeros((100, 100), dtype=np.float32),
            )
            shard_manager.store(0, param)

        moves = shard_manager.rebalance_shards()

        # Should suggest moving some parameters from shard 0
        assert 0 in moves
        assert len(moves[0]) > 0


class TestShardingIntegration:
    """Integration tests for sharding components."""

    def test_partitioner_and_manager_integration(self, partitioner, shard_manager):
        """Test that partitioner and manager work together."""
        param_names = ["encoder.layer1", "decoder.layer1", "attention.heads", "output.final"]

        for name in param_names:
            param = Parameter(
                name=name,
                shape=(64, 64),
                data=np.random.randn(64, 64).astype(np.float32),
            )
            shard_id = partitioner.get_shard(name)
            shard_manager.store(shard_id, param)

        # Verify all parameters are stored correctly
        for name in param_names:
            shard_id = partitioner.get_shard(name)
            result = shard_manager.get(shard_id, name)
            assert result is not None
            assert result.name == name

    def test_different_strategies_different_distributions(self, shard_manager):
        """Test that different strategies produce different distributions."""
        param_names = [f"param_{i}" for i in range(100)]

        strategies = ["hash", "range", "round_robin"]
        distributions = {}

        for strategy_type in strategies:
            strategy = PartitionStrategy(num_shards=4, strategy_type=strategy_type)
            partitioner = ParameterPartitioner(strategy)

            distribution = {i: 0 for i in range(4)}
            for name in param_names:
                shard_id = partitioner.get_shard(name)
                distribution[shard_id] += 1

            distributions[strategy_type] = distribution

        # At least two strategies should produce different distributions
        unique_distributions = set(tuple(d.values()) for d in distributions.values())
        assert len(unique_distributions) >= 2

    @pytest.mark.asyncio
    async def test_store_and_parameterstore_consistency(
        self, parameter_store, shard_manager, partitioner
    ):
        """Test that ParameterStore and ShardManager stay consistent."""
        param = Parameter(
            name="test_param",
            shape=(32, 32),
            data=np.random.randn(32, 32).astype(np.float32),
        )

        # Store in both
        shard_id = partitioner.get_shard(param.name)
        shard_manager.store(shard_id, param)
        await parameter_store.set(param)

        # Verify both have the parameter
        shard_result = shard_manager.get(shard_id, param.name)
        store_result = await parameter_store.get(param.name)

        assert shard_result is not None
        assert store_result is not None
        assert np.array_equal(shard_result.data, store_result.data)
