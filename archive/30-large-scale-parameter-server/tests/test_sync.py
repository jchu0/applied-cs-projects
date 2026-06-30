"""Tests for synchronization protocols and aggregation."""

import pytest
import numpy as np
import asyncio

from paramserver import (
    SyncManager,
    StalenessTracker,
    GradientAggregator,
    AsyncAggregator,
    SparsifiedAggregator,
    ConsistencyModel,
    AggregationConfig,
    AggregationType,
    Gradient,
    SyncBarrier,
)


class TestSyncManager:
    """Tests for SyncManager."""

    @pytest.mark.asyncio
    async def test_create_barrier(self, sync_manager_bsp):
        """Test creating a synchronization barrier."""
        barrier = await sync_manager_bsp.create_barrier(iteration=0, expected_workers=4)

        assert barrier.iteration == 0
        assert barrier.expected_workers == 4
        assert len(barrier.arrived_workers) == 0

    @pytest.mark.asyncio
    async def test_create_barrier_idempotent(self, sync_manager_bsp):
        """Test that creating same barrier twice returns existing one."""
        barrier1 = await sync_manager_bsp.create_barrier(iteration=0, expected_workers=4)
        barrier2 = await sync_manager_bsp.create_barrier(iteration=0, expected_workers=4)

        assert barrier1.barrier_id == barrier2.barrier_id

    @pytest.mark.asyncio
    async def test_arrive_at_barrier(self, sync_manager_bsp):
        """Test worker arriving at barrier."""
        await sync_manager_bsp.create_barrier(iteration=0, expected_workers=2)

        complete = await sync_manager_bsp.arrive_at_barrier("worker_1", iteration=0)

        assert complete is False  # Not complete yet

    @pytest.mark.asyncio
    async def test_barrier_completes(self, sync_manager_bsp):
        """Test barrier completes when all workers arrive."""
        await sync_manager_bsp.create_barrier(iteration=0, expected_workers=2)

        await sync_manager_bsp.arrive_at_barrier("worker_1", iteration=0)
        complete = await sync_manager_bsp.arrive_at_barrier("worker_2", iteration=0)

        assert complete is True

    @pytest.mark.asyncio
    async def test_arrive_nonexistent_barrier(self, sync_manager_bsp):
        """Test arriving at nonexistent barrier returns False."""
        complete = await sync_manager_bsp.arrive_at_barrier("worker_1", iteration=999)

        assert complete is False

    @pytest.mark.asyncio
    async def test_wait_for_barrier(self, sync_manager_bsp):
        """Test waiting for barrier completion."""
        await sync_manager_bsp.create_barrier(iteration=0, expected_workers=2)

        async def complete_barrier():
            await asyncio.sleep(0.1)
            await sync_manager_bsp.arrive_at_barrier("worker_1", iteration=0)
            await sync_manager_bsp.arrive_at_barrier("worker_2", iteration=0)

        # Start completing barrier in background
        asyncio.create_task(complete_barrier())

        # Wait for barrier
        result = await sync_manager_bsp.wait_for_barrier(iteration=0, timeout=5.0)

        assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_barrier_timeout(self, sync_manager_bsp):
        """Test barrier wait times out."""
        await sync_manager_bsp.create_barrier(iteration=0, expected_workers=10)

        result = await sync_manager_bsp.wait_for_barrier(iteration=0, timeout=0.2)

        assert result is False

    @pytest.mark.asyncio
    async def test_get_barrier_status_exists(self, sync_manager_bsp):
        """Test getting status of existing barrier."""
        await sync_manager_bsp.create_barrier(iteration=0, expected_workers=4)
        await sync_manager_bsp.arrive_at_barrier("worker_1", iteration=0)

        status = await sync_manager_bsp.get_barrier_status(iteration=0)

        assert status["exists"] is True
        assert status["expected"] == 4
        assert status["arrived"] == 1
        assert status["complete"] is False

    @pytest.mark.asyncio
    async def test_get_barrier_status_nonexistent(self, sync_manager_bsp):
        """Test getting status of nonexistent barrier."""
        status = await sync_manager_bsp.get_barrier_status(iteration=999)

        assert status["exists"] is False

    @pytest.mark.asyncio
    async def test_cleanup_old_barriers(self, sync_manager_bsp):
        """Test cleaning up old barriers."""
        # Create many barriers
        for i in range(20):
            await sync_manager_bsp.create_barrier(iteration=i, expected_workers=1)
            await sync_manager_bsp.arrive_at_barrier("worker_1", iteration=i)

        await sync_manager_bsp.cleanup_old_barriers(keep_recent=5)

        # Should only have last 5 barriers
        assert len(sync_manager_bsp._barriers) == 5
        # Recent iterations should exist
        for i in range(15, 20):
            assert i in sync_manager_bsp._barriers
        # Old iterations should be gone
        for i in range(15):
            assert i not in sync_manager_bsp._barriers

    def test_get_global_iteration(self, sync_manager_bsp):
        """Test getting global iteration."""
        assert sync_manager_bsp.get_global_iteration() == 0

    @pytest.mark.asyncio
    async def test_global_iteration_updates(self, sync_manager_bsp):
        """Test global iteration updates on barrier completion."""
        await sync_manager_bsp.create_barrier(iteration=5, expected_workers=1)
        await sync_manager_bsp.arrive_at_barrier("worker_1", iteration=5)

        assert sync_manager_bsp.get_global_iteration() == 5

    def test_get_worker_iteration(self, sync_manager_bsp):
        """Test getting worker iteration."""
        assert sync_manager_bsp.get_worker_iteration("worker_1") == 0

    @pytest.mark.asyncio
    async def test_worker_iteration_updates(self, sync_manager_bsp):
        """Test worker iteration updates on arrival."""
        await sync_manager_bsp.create_barrier(iteration=3, expected_workers=1)
        await sync_manager_bsp.arrive_at_barrier("worker_1", iteration=3)

        assert sync_manager_bsp.get_worker_iteration("worker_1") == 3


class TestStalenessTracker:
    """Tests for StalenessTracker."""

    def test_initialization(self, staleness_tracker):
        """Test staleness tracker initialization."""
        assert staleness_tracker.max_staleness == 3

    def test_update_parameter_version(self, staleness_tracker):
        """Test updating parameter version."""
        staleness_tracker.update_parameter_version("param1", 5)

        assert staleness_tracker._parameter_versions["param1"] == 5

    def test_record_worker_pull(self, staleness_tracker):
        """Test recording worker pull."""
        staleness_tracker.record_worker_pull("worker_1", "param1", 3)

        assert staleness_tracker._worker_versions["worker_1"]["param1"] == 3

    def test_check_staleness_not_stale(self, staleness_tracker):
        """Test checking staleness when not stale."""
        staleness_tracker.update_parameter_version("param1", 5)
        staleness_tracker.record_worker_pull("worker_1", "param1", 4)

        is_stale, staleness = staleness_tracker.check_staleness("worker_1", "param1")

        assert is_stale is False
        assert staleness == 1

    def test_check_staleness_stale(self, staleness_tracker):
        """Test checking staleness when stale."""
        staleness_tracker.update_parameter_version("param1", 10)
        staleness_tracker.record_worker_pull("worker_1", "param1", 3)

        is_stale, staleness = staleness_tracker.check_staleness("worker_1", "param1")

        assert is_stale is True
        assert staleness == 7  # 10 - 3 = 7, exceeds max_staleness of 3

    def test_check_staleness_boundary(self, staleness_tracker):
        """Test staleness at boundary."""
        staleness_tracker.update_parameter_version("param1", 6)
        staleness_tracker.record_worker_pull("worker_1", "param1", 3)

        is_stale, staleness = staleness_tracker.check_staleness("worker_1", "param1")

        assert is_stale is False  # staleness = 3, max = 3
        assert staleness == 3

    def test_should_block_push_false(self, staleness_tracker):
        """Test should_block_push when not stale."""
        staleness_tracker.update_parameter_version("param1", 5)
        staleness_tracker.record_worker_pull("worker_1", "param1", 4)

        should_block = staleness_tracker.should_block_push("worker_1", "param1")

        assert should_block is False

    def test_should_block_push_true(self, staleness_tracker):
        """Test should_block_push when stale."""
        staleness_tracker.update_parameter_version("param1", 10)
        staleness_tracker.record_worker_pull("worker_1", "param1", 0)

        should_block = staleness_tracker.should_block_push("worker_1", "param1")

        assert should_block is True

    def test_get_staleness_stats(self, staleness_tracker):
        """Test getting staleness statistics."""
        staleness_tracker.update_parameter_version("param1", 5)
        staleness_tracker.update_parameter_version("param2", 10)
        staleness_tracker.record_worker_pull("worker_1", "param1", 3)
        staleness_tracker.record_worker_pull("worker_1", "param2", 8)

        stats = staleness_tracker.get_staleness_stats()

        assert stats["max_allowed"] == 3
        assert stats["num_tracked"] == 2
        # Average staleness: (5-3 + 10-8) / 2 = (2 + 2) / 2 = 2
        assert stats["avg_staleness"] == 2.0

    def test_get_staleness_stats_empty(self, staleness_tracker):
        """Test staleness stats when empty."""
        stats = staleness_tracker.get_staleness_stats()

        assert stats["avg_staleness"] == 0.0
        assert stats["num_tracked"] == 0


class TestGradientAggregator:
    """Tests for GradientAggregator."""

    @pytest.mark.asyncio
    async def test_add_gradient(self, gradient_aggregator, sample_gradient):
        """Test adding gradient."""
        await gradient_aggregator.add_gradient(sample_gradient)

        count = await gradient_aggregator.get_pending_count(sample_gradient.name)
        assert count == 1

    @pytest.mark.asyncio
    async def test_aggregate_mean(self, mean_aggregator, gradients_for_aggregation):
        """Test mean aggregation."""
        for grad in gradients_for_aggregation:
            await mean_aggregator.add_gradient(grad)

        result = await mean_aggregator.aggregate("test_param")

        # Gradients have values 1, 2, 3, 4 - mean is 2.5
        expected = np.full((10, 10), 2.5, dtype=np.float32)
        assert np.allclose(result, expected)

    @pytest.mark.asyncio
    async def test_aggregate_sum(self, sum_aggregator, gradients_for_aggregation):
        """Test sum aggregation."""
        for grad in gradients_for_aggregation:
            await sum_aggregator.add_gradient(grad)

        result = await sum_aggregator.aggregate("test_param")

        # Sum of 1+2+3+4 = 10
        expected = np.full((10, 10), 10.0, dtype=np.float32)
        assert np.allclose(result, expected)

    @pytest.mark.asyncio
    async def test_aggregate_with_expected_count_insufficient(
        self, gradient_aggregator, gradients_for_aggregation
    ):
        """Test aggregation fails when not enough gradients."""
        await gradient_aggregator.add_gradient(gradients_for_aggregation[0])

        result = await gradient_aggregator.aggregate("test_param", expected_count=4)

        assert result is None

    @pytest.mark.asyncio
    async def test_aggregate_clears_pending(
        self, gradient_aggregator, gradients_for_aggregation
    ):
        """Test that aggregation clears pending gradients."""
        for grad in gradients_for_aggregation:
            await gradient_aggregator.add_gradient(grad)

        await gradient_aggregator.aggregate("test_param")

        count = await gradient_aggregator.get_pending_count("test_param")
        assert count == 0

    @pytest.mark.asyncio
    async def test_aggregate_nonexistent(self, gradient_aggregator):
        """Test aggregating nonexistent parameter returns None."""
        result = await gradient_aggregator.aggregate("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_aggregate_with_clipping(self, aggregator_with_clipping):
        """Test gradient clipping."""
        # Create gradient with large values
        large_gradient = Gradient(
            name="test",
            data=np.full((10, 10), 100.0, dtype=np.float32),
            worker_id="worker_1",
            iteration=0,
        )

        await aggregator_with_clipping.add_gradient(large_gradient)
        result = await aggregator_with_clipping.aggregate("test")

        # Norm should be clipped to 1.0
        norm = np.linalg.norm(result)
        assert np.isclose(norm, 1.0, atol=1e-5)

    @pytest.mark.asyncio
    async def test_aggregate_with_momentum(self, aggregator_with_momentum):
        """Test momentum application."""
        # First gradient
        grad1 = Gradient(
            name="test",
            data=np.ones((5, 5), dtype=np.float32),
            worker_id="worker_1",
            iteration=0,
        )
        await aggregator_with_momentum.add_gradient(grad1)
        result1 = await aggregator_with_momentum.aggregate("test")

        # Second gradient
        grad2 = Gradient(
            name="test",
            data=np.ones((5, 5), dtype=np.float32),
            worker_id="worker_1",
            iteration=1,
        )
        await aggregator_with_momentum.add_gradient(grad2)
        result2 = await aggregator_with_momentum.aggregate("test")

        # Second result should include momentum from first
        # momentum_buffer = 0.9 * 1.0 + 1.0 = 1.9
        expected = np.full((5, 5), 1.9, dtype=np.float32)
        assert np.allclose(result2, expected)

    @pytest.mark.asyncio
    async def test_clear_pending_specific(self, gradient_aggregator, multiple_gradients):
        """Test clearing pending for specific parameter."""
        for grad in multiple_gradients:
            await gradient_aggregator.add_gradient(grad)

        await gradient_aggregator.clear_pending("layer1.weight")

        count = await gradient_aggregator.get_pending_count("layer1.weight")
        assert count == 0

    @pytest.mark.asyncio
    async def test_clear_pending_all(self, gradient_aggregator, multiple_gradients):
        """Test clearing all pending gradients."""
        for grad in multiple_gradients:
            await gradient_aggregator.add_gradient(grad)

        await gradient_aggregator.clear_pending()

        assert len(gradient_aggregator._pending) == 0


class TestAsyncAggregator:
    """Tests for AsyncAggregator."""

    @pytest.mark.asyncio
    async def test_aggregate_immediate(self, async_aggregator, sample_gradient):
        """Test immediate aggregation."""
        result = await async_aggregator.aggregate_immediate(sample_gradient)

        assert result is not None
        assert result.shape == sample_gradient.data.shape

    @pytest.mark.asyncio
    async def test_aggregate_immediate_with_clipping(self):
        """Test immediate aggregation with clipping."""
        config = AggregationConfig(clip_norm=1.0)
        aggregator = AsyncAggregator(config)

        large_gradient = Gradient(
            name="test",
            data=np.full((10, 10), 100.0, dtype=np.float32),
            worker_id="worker_1",
            iteration=0,
        )

        result = await aggregator.aggregate_immediate(large_gradient)

        norm = np.linalg.norm(result)
        assert np.isclose(norm, 1.0, atol=1e-5)

    @pytest.mark.asyncio
    async def test_aggregate_immediate_with_momentum(self):
        """Test immediate aggregation with momentum."""
        config = AggregationConfig(momentum=0.9)
        aggregator = AsyncAggregator(config)

        grad1 = Gradient(
            name="test",
            data=np.ones((5, 5), dtype=np.float32),
            worker_id="worker_1",
            iteration=0,
        )
        await aggregator.aggregate_immediate(grad1)

        grad2 = Gradient(
            name="test",
            data=np.ones((5, 5), dtype=np.float32),
            worker_id="worker_1",
            iteration=1,
        )
        result = await aggregator.aggregate_immediate(grad2)

        # With momentum: 0.9 * 1.0 + 1.0 = 1.9
        expected = np.full((5, 5), 1.9, dtype=np.float32)
        assert np.allclose(result, expected)

    @pytest.mark.asyncio
    async def test_get_update_count(self, async_aggregator, sample_gradient):
        """Test update count tracking."""
        await async_aggregator.aggregate_immediate(sample_gradient)
        await async_aggregator.aggregate_immediate(sample_gradient)
        await async_aggregator.aggregate_immediate(sample_gradient)

        count = async_aggregator.get_update_count(sample_gradient.name)

        assert count == 3


class TestSparsifiedAggregator:
    """Tests for SparsifiedAggregator."""

    @pytest.mark.asyncio
    async def test_aggregate_sparse(self, sparsified_aggregator):
        """Test sparse aggregation."""
        gradient = Gradient(
            name="test",
            data=np.random.randn(100, 100).astype(np.float32),
            worker_id="worker_1",
            iteration=0,
        )

        await sparsified_aggregator.add_gradient(gradient)
        sparse, indices = await sparsified_aggregator.aggregate_sparse("test")

        assert sparse is not None
        assert indices is not None
        # With top_k=0.1, should keep 10% of elements
        non_zero = np.count_nonzero(sparse)
        total = sparse.size
        assert non_zero <= int(total * 0.1) + 1  # +1 for rounding

    @pytest.mark.asyncio
    async def test_aggregate_sparse_residual(self, sparsified_aggregator):
        """Test that residual is stored for next iteration."""
        gradient = Gradient(
            name="test",
            data=np.ones((10, 10), dtype=np.float32),
            worker_id="worker_1",
            iteration=0,
        )

        await sparsified_aggregator.add_gradient(gradient)
        await sparsified_aggregator.aggregate_sparse("test")

        # Residual should be stored
        assert "test" in sparsified_aggregator._residuals

    @pytest.mark.asyncio
    async def test_aggregate_sparse_nonexistent(self, sparsified_aggregator):
        """Test sparse aggregation of nonexistent parameter."""
        sparse, indices = await sparsified_aggregator.aggregate_sparse("nonexistent")

        assert sparse is None
        assert indices is None

    @pytest.mark.asyncio
    async def test_aggregate_sparse_residual_accumulation(self):
        """Test that residuals accumulate across iterations."""
        aggregator = SparsifiedAggregator(top_k=0.1)

        # Multiple iterations with same gradient
        for i in range(5):
            gradient = Gradient(
                name="test",
                data=np.ones((100,), dtype=np.float32),
                worker_id="worker_1",
                iteration=i,
            )
            await aggregator.add_gradient(gradient)
            sparse, _ = await aggregator.aggregate_sparse("test")

            # After first iteration, residuals should influence output
            if i > 0:
                # More elements should be non-zero as residuals accumulate
                assert sparse is not None


class TestSyncBarrier:
    """Tests for SyncBarrier dataclass."""

    def test_is_complete_false(self):
        """Test barrier not complete."""
        barrier = SyncBarrier(
            barrier_id="test",
            iteration=0,
            expected_workers=3,
        )

        barrier.arrived_workers.add("worker_1")

        assert barrier.is_complete is False

    def test_is_complete_true(self):
        """Test barrier complete."""
        barrier = SyncBarrier(
            barrier_id="test",
            iteration=0,
            expected_workers=2,
        )

        barrier.arrived_workers.add("worker_1")
        barrier.arrived_workers.add("worker_2")

        assert barrier.is_complete is True

    def test_is_complete_over_expected(self):
        """Test barrier complete with more than expected."""
        barrier = SyncBarrier(
            barrier_id="test",
            iteration=0,
            expected_workers=2,
        )

        barrier.arrived_workers.add("worker_1")
        barrier.arrived_workers.add("worker_2")
        barrier.arrived_workers.add("worker_3")

        assert barrier.is_complete is True


class TestConsistencyModelIntegration:
    """Integration tests for different consistency models."""

    @pytest.mark.asyncio
    async def test_bsp_requires_all_workers(self):
        """Test BSP waits for all workers."""
        sync_manager = SyncManager(ConsistencyModel.BSP)

        await sync_manager.create_barrier(iteration=0, expected_workers=3)

        # Not complete after 2 workers
        await sync_manager.arrive_at_barrier("worker_1", iteration=0)
        complete = await sync_manager.arrive_at_barrier("worker_2", iteration=0)
        assert complete is False

        # Complete after 3rd worker
        complete = await sync_manager.arrive_at_barrier("worker_3", iteration=0)
        assert complete is True

    @pytest.mark.asyncio
    async def test_concurrent_barrier_arrivals(self):
        """Test concurrent arrivals at barrier."""
        sync_manager = SyncManager(ConsistencyModel.BSP)

        await sync_manager.create_barrier(iteration=0, expected_workers=4)

        async def arrive(worker_id: str):
            return await sync_manager.arrive_at_barrier(worker_id, iteration=0)

        # All workers arrive concurrently
        results = await asyncio.gather(*[
            arrive(f"worker_{i}") for i in range(4)
        ])

        # At least one should report complete
        assert any(results)

        # Check final status
        status = await sync_manager.get_barrier_status(iteration=0)
        assert status["complete"] is True

    @pytest.mark.asyncio
    async def test_staleness_with_sync_manager(self):
        """Test staleness tracking works with sync manager."""
        sync_manager = SyncManager(ConsistencyModel.SSP)
        staleness_tracker = StalenessTracker(max_staleness=2)

        # Worker 1 is ahead
        for i in range(5):
            await sync_manager.create_barrier(iteration=i, expected_workers=1)
            await sync_manager.arrive_at_barrier("worker_1", iteration=i)
            staleness_tracker.update_parameter_version("param1", i)

        # Worker 2 is behind
        staleness_tracker.record_worker_pull("worker_2", "param1", 1)

        # Worker 2 should be blocked
        should_block = staleness_tracker.should_block_push("worker_2", "param1")
        assert should_block is True  # Staleness = 4 - 1 = 3, exceeds max of 2
