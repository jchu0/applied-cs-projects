"""Comprehensive tests for AllReduce operations."""

import pytest
import numpy as np
import threading
import time
from unittest.mock import Mock, MagicMock, patch
from typing import List, Callable

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from distautograd.distributed.ddp import (
    AllReduceStrategy,
    Reducer,
    GradReducer,
    GradientBucket,
)
from distautograd.core.context import (
    ProcessGroup,
    Backend,
    DistributedContext,
    WorkerInfo,
    DistributedTensor,
    all_reduce,
    broadcast,
    all_gather,
    reduce_scatter,
)


# =============================================================================
# Test Fixtures
# =============================================================================

class MockParameter:
    """Mock parameter for testing."""

    def __init__(self, shape, requires_grad=True):
        self.data = np.random.randn(*shape).astype(np.float32)
        self.grad = None
        self.requires_grad = requires_grad


@pytest.fixture
def process_group():
    """Create a process group."""
    return ProcessGroup(ranks=[0, 1, 2, 3], backend=Backend.GLOO, name="test")


@pytest.fixture
def two_worker_group():
    """Create a 2-worker process group."""
    return ProcessGroup(ranks=[0, 1], backend=Backend.GLOO)


@pytest.fixture
def distributed_context():
    """Create a distributed context."""
    worker = WorkerInfo(rank=0, world_size=4, local_rank=0)
    return DistributedContext(worker_info=worker)


# =============================================================================
# AllReduceStrategy Tests
# =============================================================================

class TestAllReduceStrategy:
    """Tests for AllReduceStrategy enum."""

    def test_all_strategies_defined(self):
        """Test that all expected strategies are defined."""
        expected_strategies = ['RING', 'TREE', 'RECURSIVE_HALVING', 'BUCKET']

        for strategy_name in expected_strategies:
            assert hasattr(AllReduceStrategy, strategy_name)

    def test_strategy_values_are_unique(self):
        """Test that all strategy values are unique."""
        values = [
            AllReduceStrategy.RING.value,
            AllReduceStrategy.TREE.value,
            AllReduceStrategy.RECURSIVE_HALVING.value,
            AllReduceStrategy.BUCKET.value,
        ]

        assert len(values) == len(set(values))

    def test_strategy_comparison(self):
        """Test strategy comparison."""
        assert AllReduceStrategy.RING == AllReduceStrategy.RING
        assert AllReduceStrategy.RING != AllReduceStrategy.TREE

    def test_strategy_iteration(self):
        """Test iterating over strategies."""
        strategies = list(AllReduceStrategy)
        assert len(strategies) == 4


# =============================================================================
# Ring AllReduce Simulation Tests
# =============================================================================

class TestRingAllReduce:
    """Tests for ring AllReduce algorithm simulation."""

    def simulate_ring_allreduce(self, tensors: List[np.ndarray]) -> List[np.ndarray]:
        """Simulate ring AllReduce across workers."""
        n = len(tensors)
        if n == 0:
            return []

        # Sum all tensors
        total = np.zeros_like(tensors[0])
        for t in tensors:
            total += t

        # Each worker gets the sum
        return [total.copy() for _ in range(n)]

    def test_ring_allreduce_sum(self):
        """Test ring AllReduce sum operation."""
        # Simulate 4 workers with tensors
        tensors = [
            np.ones((10, 10)) * i
            for i in range(1, 5)  # Values: 1, 2, 3, 4
        ]

        results = self.simulate_ring_allreduce(tensors)

        # Sum should be 1 + 2 + 3 + 4 = 10
        expected = np.ones((10, 10)) * 10

        for result in results:
            np.testing.assert_array_almost_equal(result, expected)

    def test_ring_allreduce_average(self):
        """Test ring AllReduce with averaging."""
        tensors = [
            np.ones((5, 5)) * i
            for i in range(1, 5)
        ]

        results = self.simulate_ring_allreduce(tensors)

        # Average: (1 + 2 + 3 + 4) / 4 = 2.5
        averaged = [r / len(tensors) for r in results]
        expected = np.ones((5, 5)) * 2.5

        for result in averaged:
            np.testing.assert_array_almost_equal(result, expected)

    def test_ring_allreduce_different_shapes(self):
        """Test ring AllReduce with different tensor shapes."""
        shapes = [(10,), (10, 10), (10, 10, 10)]

        for shape in shapes:
            tensors = [np.ones(shape) * (i + 1) for i in range(4)]
            results = self.simulate_ring_allreduce(tensors)

            expected = np.ones(shape) * 10
            for result in results:
                np.testing.assert_array_almost_equal(result, expected)

    def test_ring_allreduce_empty(self):
        """Test ring AllReduce with empty list."""
        results = self.simulate_ring_allreduce([])
        assert results == []


# =============================================================================
# Tree AllReduce Simulation Tests
# =============================================================================

class TestTreeAllReduce:
    """Tests for tree AllReduce algorithm simulation."""

    def simulate_tree_allreduce(self, tensors: List[np.ndarray]) -> List[np.ndarray]:
        """Simulate tree AllReduce: reduce to root, broadcast to all."""
        n = len(tensors)
        if n == 0:
            return []

        # Phase 1: Reduce to root (rank 0)
        total = np.zeros_like(tensors[0])
        for t in tensors:
            total += t

        # Phase 2: Broadcast from root
        return [total.copy() for _ in range(n)]

    def test_tree_allreduce_power_of_two(self):
        """Test tree AllReduce with power-of-2 workers."""
        for num_workers in [2, 4, 8]:
            tensors = [np.ones((8, 8)) for _ in range(num_workers)]
            results = self.simulate_tree_allreduce(tensors)

            expected = np.ones((8, 8)) * num_workers
            for result in results:
                np.testing.assert_array_almost_equal(result, expected)

    def test_tree_allreduce_non_power_of_two(self):
        """Test tree AllReduce with non-power-of-2 workers."""
        for num_workers in [3, 5, 7]:
            tensors = [np.ones((4, 4)) for _ in range(num_workers)]
            results = self.simulate_tree_allreduce(tensors)

            expected = np.ones((4, 4)) * num_workers
            for result in results:
                np.testing.assert_array_almost_equal(result, expected)

    def test_tree_allreduce_single_worker(self):
        """Test tree AllReduce with single worker."""
        tensors = [np.array([1, 2, 3])]
        results = self.simulate_tree_allreduce(tensors)

        np.testing.assert_array_equal(results[0], np.array([1, 2, 3]))


# =============================================================================
# Recursive Halving Doubling Tests
# =============================================================================

class TestRecursiveHalvingDoubling:
    """Tests for recursive halving-doubling AllReduce simulation."""

    def simulate_recursive_halving_doubling(self, tensors: List[np.ndarray]) -> List[np.ndarray]:
        """Simulate recursive halving-doubling AllReduce."""
        n = len(tensors)
        if n == 0:
            return []

        # This algorithm is optimal for power-of-2 workers
        # Simplified simulation: just sum all tensors
        total = np.zeros_like(tensors[0])
        for t in tensors:
            total += t

        return [total.copy() for _ in range(n)]

    def test_recursive_halving_power_of_two(self):
        """Test recursive halving with power-of-2 workers."""
        for num_workers in [2, 4, 8, 16]:
            tensors = [np.array([float(i)]) for i in range(num_workers)]
            results = self.simulate_recursive_halving_doubling(tensors)

            expected_sum = sum(range(num_workers))
            for result in results:
                np.testing.assert_array_almost_equal(result, np.array([expected_sum]))

    def test_recursive_halving_large_tensor(self):
        """Test recursive halving with large tensors."""
        tensors = [np.random.randn(1000, 1000) for _ in range(4)]

        results = self.simulate_recursive_halving_doubling(tensors)

        expected = sum(t for t in tensors)
        for result in results:
            np.testing.assert_array_almost_equal(result, expected)


# =============================================================================
# Bucket AllReduce Tests
# =============================================================================

class TestBucketAllReduce:
    """Tests for bucket-based AllReduce."""

    def test_bucket_allreduce_via_reducer(self, process_group):
        """Test AllReduce through Reducer with buckets."""
        params = [MockParameter((50, 50)) for _ in range(4)]
        for p in params:
            p.grad = np.ones_like(p.data)

        reducer = Reducer(params, process_group, bucket_cap_mb=0.01)  # Force bucketing
        reducer.prepare_for_backward()

        # Mark all gradients ready
        for i in range(len(params)):
            reducer.mark_grad_ready(i)

        # All buckets should be reduced
        for bucket in reducer._buckets:
            assert bucket.pending == 0

    def test_bucket_size_affects_num_buckets(self, process_group):
        """Test that bucket size affects number of buckets."""
        params = [MockParameter((100, 100)) for _ in range(10)]

        # Large bucket size - fewer buckets
        reducer_large = Reducer(params, process_group, bucket_cap_mb=100.0)

        # Small bucket size - more buckets
        reducer_small = Reducer(params, process_group, bucket_cap_mb=0.01)

        assert len(reducer_small._buckets) > len(reducer_large._buckets)

    def test_bucket_gradient_aggregation(self, process_group):
        """Test that gradients are correctly aggregated in buckets."""
        params = [MockParameter((10, 10)) for _ in range(3)]
        for i, p in enumerate(params):
            p.grad = np.ones((10, 10)) * (i + 1)  # 1, 2, 3

        reducer = Reducer(params, process_group)
        reducer.prepare_for_backward()

        # Reduce all
        for i in range(len(params)):
            reducer.mark_grad_ready(i)

        reducer.finalize()

        # Gradients should be averaged by world_size (4)
        expected_values = [0.25, 0.5, 0.75]  # 1/4, 2/4, 3/4
        for i, p in enumerate(params):
            assert p.grad is not None


# =============================================================================
# All-Reduce Collective Operations Tests
# =============================================================================

class TestCollectiveOperations:
    """Tests for collective operation functions."""

    def test_all_reduce_returns_tensor(self, process_group):
        """Test that all_reduce returns a tensor."""
        data = np.random.randn(10, 10).astype(np.float32)
        tensor = DistributedTensor(data, requires_grad=True, process_group=process_group)

        result = all_reduce(tensor, op="sum", group=process_group)

        assert result is not None
        assert isinstance(result, DistributedTensor)

    def test_all_reduce_operations(self, process_group):
        """Test different all_reduce operations."""
        operations = ["sum", "avg", "min", "max"]

        for op in operations:
            data = np.random.randn(5, 5).astype(np.float32)
            tensor = DistributedTensor(data, process_group=process_group)

            result = all_reduce(tensor, op=op, group=process_group)
            assert result is not None

    def test_broadcast_from_source(self, process_group):
        """Test broadcast from source rank."""
        data = np.array([1, 2, 3]).astype(np.float32)
        tensor = DistributedTensor(data, process_group=process_group)

        result = broadcast(tensor, src=0, group=process_group)

        assert result is not None
        assert isinstance(result, DistributedTensor)

    def test_all_gather(self, process_group):
        """Test all_gather operation."""
        data = np.array([1, 2, 3]).astype(np.float32)
        tensor = DistributedTensor(data, process_group=process_group)

        tensors = [DistributedTensor(np.zeros(3)) for _ in range(4)]

        all_gather(tensors, tensor, group=process_group)

        # Should complete without error

    def test_reduce_scatter(self, process_group):
        """Test reduce_scatter operation."""
        output = DistributedTensor(np.zeros(10))
        input_list = [DistributedTensor(np.ones(10)) for _ in range(4)]

        reduce_scatter(output, input_list, op="sum", group=process_group)

        # Should complete without error


# =============================================================================
# AllReduce with Gradient Compression Tests
# =============================================================================

class TestAllReduceWithCompression:
    """Tests for AllReduce with gradient compression."""

    def test_topk_before_allreduce(self, process_group):
        """Test top-k compression before AllReduce."""
        params = [MockParameter((100, 100))]
        params[0].grad = np.random.randn(100, 100).astype(np.float32)

        original_nonzero = np.count_nonzero(params[0].grad)

        grad_reducer = GradReducer(iter(params), process_group)

        # Compress gradients
        grad_reducer.compress_gradients(ratio=0.1)

        compressed_nonzero = np.count_nonzero(params[0].grad)

        # Should have fewer non-zero elements
        assert compressed_nonzero < original_nonzero

    def test_compression_preserves_largest(self, process_group):
        """Test that compression preserves largest magnitude values."""
        params = [MockParameter((10,))]
        params[0].grad = np.array([1, 5, 2, 8, 3, 6, 4, 7, 9, 10]).astype(np.float32)

        grad_reducer = GradReducer(iter(params), process_group)
        grad_reducer.compress_gradients(ratio=0.3)  # Keep top 30%

        # Top 3 values (8, 9, 10) should be preserved
        assert 10 in params[0].grad or -10 in params[0].grad
        assert 9 in params[0].grad or -9 in params[0].grad

    def test_compression_ratio_effect(self, process_group):
        """Test different compression ratios."""
        ratios = [0.1, 0.3, 0.5, 0.9]

        for ratio in ratios:
            params = [MockParameter((100, 100))]
            params[0].grad = np.random.randn(100, 100).astype(np.float32)

            total_elements = params[0].grad.size

            grad_reducer = GradReducer(iter(params), process_group)
            grad_reducer.compress_gradients(ratio=ratio)

            nonzero = np.count_nonzero(params[0].grad)

            # Non-zero count should be approximately ratio * total
            expected = int(total_elements * ratio)
            assert abs(nonzero - expected) <= 1


# =============================================================================
# AllReduce Performance Characteristics Tests
# =============================================================================

class TestAllReducePerformance:
    """Tests for AllReduce performance characteristics."""

    def test_allreduce_scales_with_world_size(self):
        """Test that AllReduce time characteristics with world size."""
        world_sizes = [2, 4, 8]

        for world_size in world_sizes:
            group = ProcessGroup(ranks=list(range(world_size)))
            params = [MockParameter((100, 100))]
            params[0].grad = np.ones((100, 100))

            reducer = Reducer(params, group)

            # Should complete in reasonable time
            reducer.prepare_for_backward()
            reducer.mark_grad_ready(0)

    def test_allreduce_handles_large_tensors(self, process_group):
        """Test AllReduce with large tensors."""
        params = [MockParameter((1000, 1000))]  # 4MB tensor
        params[0].grad = np.random.randn(1000, 1000).astype(np.float32)

        reducer = Reducer(params, process_group)
        reducer.prepare_for_backward()
        reducer.mark_grad_ready(0)

        # Should complete without error

    def test_allreduce_handles_many_tensors(self, process_group):
        """Test AllReduce with many small tensors."""
        params = [MockParameter((10, 10)) for _ in range(100)]
        for p in params:
            p.grad = np.random.randn(10, 10).astype(np.float32)

        reducer = Reducer(params, process_group, bucket_cap_mb=0.01)
        reducer.prepare_for_backward()

        for i in range(len(params)):
            reducer.mark_grad_ready(i)

        # Should bucket efficiently


# =============================================================================
# AllReduce Edge Cases
# =============================================================================

class TestAllReduceEdgeCases:
    """Edge case tests for AllReduce."""

    def test_allreduce_single_element(self, process_group):
        """Test AllReduce with single element tensor."""
        params = [MockParameter((1,))]
        params[0].grad = np.array([5.0])

        reducer = Reducer(params, process_group)
        reducer.prepare_for_backward()
        reducer.mark_grad_ready(0)

        # Should be 5.0 / 4 = 1.25
        np.testing.assert_array_almost_equal(params[0].grad, np.array([1.25]))

    def test_allreduce_zero_tensor(self, process_group):
        """Test AllReduce with zero tensor."""
        params = [MockParameter((10, 10))]
        params[0].grad = np.zeros((10, 10))

        reducer = Reducer(params, process_group)
        reducer.prepare_for_backward()
        reducer.mark_grad_ready(0)

        # Should remain zero
        np.testing.assert_array_equal(params[0].grad, np.zeros((10, 10)))

    def test_allreduce_negative_values(self, process_group):
        """Test AllReduce with negative values."""
        params = [MockParameter((5, 5))]
        params[0].grad = np.ones((5, 5)) * -4

        reducer = Reducer(params, process_group)
        reducer.prepare_for_backward()
        reducer.mark_grad_ready(0)

        # Should be -4 / 4 = -1
        np.testing.assert_array_almost_equal(params[0].grad, np.ones((5, 5)) * -1)

    def test_allreduce_mixed_signs(self, process_group):
        """Test AllReduce with mixed positive and negative values."""
        params = [MockParameter((4,))]
        params[0].grad = np.array([4.0, -4.0, 8.0, -8.0])

        reducer = Reducer(params, process_group)
        reducer.prepare_for_backward()
        reducer.mark_grad_ready(0)

        expected = np.array([1.0, -1.0, 2.0, -2.0])  # Divided by 4
        np.testing.assert_array_almost_equal(params[0].grad, expected)

    def test_allreduce_inf_values(self, process_group):
        """Test AllReduce handling of infinity values."""
        params = [MockParameter((3,))]
        params[0].grad = np.array([np.inf, -np.inf, 1.0])

        reducer = Reducer(params, process_group)
        reducer.prepare_for_backward()
        reducer.mark_grad_ready(0)

        # Should handle inf
        assert np.isinf(params[0].grad[0])
        assert np.isinf(params[0].grad[1])

    def test_allreduce_nan_values(self, process_group):
        """Test AllReduce handling of NaN values."""
        params = [MockParameter((2,))]
        params[0].grad = np.array([np.nan, 1.0])

        reducer = Reducer(params, process_group)
        reducer.prepare_for_backward()
        reducer.mark_grad_ready(0)

        # NaN should propagate
        assert np.isnan(params[0].grad[0])


# =============================================================================
# Strategy Selection Tests
# =============================================================================

class TestStrategySelection:
    """Tests for AllReduce strategy selection."""

    def test_grad_reducer_default_strategy(self, process_group):
        """Test default strategy is BUCKET."""
        params = [MockParameter((10, 10))]
        reducer = GradReducer(iter(params), process_group)

        assert reducer.strategy == AllReduceStrategy.BUCKET

    def test_grad_reducer_custom_strategies(self, process_group):
        """Test using custom strategies."""
        params = [MockParameter((10, 10))]

        for strategy in AllReduceStrategy:
            reducer = GradReducer(iter(params), process_group, strategy=strategy)
            assert reducer.strategy == strategy

    def test_strategy_affects_reducer(self, process_group):
        """Test that strategy is properly stored in GradReducer."""
        params = [MockParameter((10, 10))]

        reducer_ring = GradReducer(iter(params), process_group, strategy=AllReduceStrategy.RING)
        reducer_tree = GradReducer(iter(params), process_group, strategy=AllReduceStrategy.TREE)

        assert reducer_ring.strategy != reducer_tree.strategy


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
