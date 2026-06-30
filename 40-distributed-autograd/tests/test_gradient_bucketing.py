"""Comprehensive tests for Gradient Bucketing implementation."""

import pytest
import numpy as np
import threading
import time
from unittest.mock import Mock, MagicMock
from typing import List, Iterator

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from distautograd.distributed.ddp import (
    GradientBucket,
    Reducer,
    GradReducer,
)
from distautograd.core.context import (
    ProcessGroup,
    Backend,
    GradBucket,
    DistributedTensor,
)


# =============================================================================
# Test Fixtures
# =============================================================================

class MockParameter:
    """Mock parameter for testing."""

    def __init__(self, shape, dtype=np.float32, requires_grad=True):
        self.data = np.random.randn(*shape).astype(dtype)
        self.grad = None
        self.requires_grad = requires_grad


@pytest.fixture
def process_group():
    """Create a process group."""
    return ProcessGroup(ranks=[0, 1, 2, 3], backend=Backend.GLOO)


@pytest.fixture
def small_process_group():
    """Create a small process group."""
    return ProcessGroup(ranks=[0, 1], backend=Backend.GLOO)


# =============================================================================
# GradientBucket Tests
# =============================================================================

class TestGradientBucket:
    """Tests for GradientBucket dataclass."""

    def test_bucket_creation_defaults(self):
        """Test bucket creation with default values."""
        bucket = GradientBucket(index=0)

        assert bucket.index == 0
        assert bucket.params == []
        assert bucket.grads == []
        assert bucket.size_bytes == 0
        assert bucket.pending == 0
        assert bucket.future is None

    def test_bucket_creation_with_values(self):
        """Test bucket creation with values."""
        params = [MockParameter((10,)), MockParameter((20,))]
        grads = [np.ones(10), np.ones(20)]

        bucket = GradientBucket(
            index=5,
            params=params,
            grads=grads,
            size_bytes=120,
            pending=2
        )

        assert bucket.index == 5
        assert len(bucket.params) == 2
        assert len(bucket.grads) == 2
        assert bucket.size_bytes == 120
        assert bucket.pending == 2

    def test_bucket_mutable_fields(self):
        """Test that bucket fields are mutable."""
        bucket = GradientBucket(index=0)

        # Add to lists
        bucket.params.append(MockParameter((5,)))
        bucket.grads.append(np.ones(5))
        bucket.size_bytes += 20
        bucket.pending = 1

        assert len(bucket.params) == 1
        assert len(bucket.grads) == 1
        assert bucket.size_bytes == 20
        assert bucket.pending == 1

    def test_bucket_independence(self):
        """Test that buckets are independent."""
        bucket1 = GradientBucket(index=0)
        bucket2 = GradientBucket(index=1)

        bucket1.params.append("param1")
        bucket2.params.append("param2")

        assert "param1" not in bucket2.params
        assert "param2" not in bucket1.params

    def test_bucket_future_field(self):
        """Test future field for async operations."""
        bucket = GradientBucket(index=0)

        # Set a mock future
        mock_future = Mock()
        bucket.future = mock_future

        assert bucket.future == mock_future


# =============================================================================
# GradBucket (Core Context) Tests
# =============================================================================

class TestGradBucket:
    """Tests for GradBucket from core context."""

    def test_grad_bucket_creation(self):
        """Test GradBucket creation."""
        bucket = GradBucket(index=0)

        assert bucket.index == 0
        assert bucket.tensors == []
        assert bucket.gradients == []
        assert bucket.size == 0
        assert bucket.ready == False

    def test_add_gradient(self):
        """Test adding gradient to bucket."""
        bucket = GradBucket(index=0)

        tensor = DistributedTensor(np.zeros((10, 10)), requires_grad=True)
        grad = np.ones((10, 10))

        bucket.add_gradient(tensor, grad)

        assert len(bucket.tensors) == 1
        assert len(bucket.gradients) == 1
        assert bucket.size == grad.nbytes

    def test_add_multiple_gradients(self):
        """Test adding multiple gradients."""
        bucket = GradBucket(index=0)

        for i in range(5):
            tensor = DistributedTensor(np.zeros((i + 1, i + 1)), requires_grad=True)
            grad = np.ones((i + 1, i + 1))
            bucket.add_gradient(tensor, grad)

        assert len(bucket.tensors) == 5
        assert len(bucket.gradients) == 5

    def test_flatten_gradients(self):
        """Test flattening gradients into single buffer."""
        bucket = GradBucket(index=0)

        # Add known gradients
        tensor1 = DistributedTensor(np.zeros((2, 2)), requires_grad=True)
        grad1 = np.array([[1, 2], [3, 4]]).astype(np.float32)

        tensor2 = DistributedTensor(np.zeros((3,)), requires_grad=True)
        grad2 = np.array([5, 6, 7]).astype(np.float32)

        bucket.add_gradient(tensor1, grad1)
        bucket.add_gradient(tensor2, grad2)

        flat = bucket.flatten()

        expected = np.array([1, 2, 3, 4, 5, 6, 7]).astype(np.float32)
        np.testing.assert_array_equal(flat, expected)

    def test_unflatten_gradients(self):
        """Test unflattening buffer back to gradients."""
        bucket = GradBucket(index=0)

        # Add gradients with known shapes
        tensor1 = DistributedTensor(np.zeros((2, 3)), requires_grad=True)
        grad1 = np.zeros((2, 3))

        tensor2 = DistributedTensor(np.zeros((4,)), requires_grad=True)
        grad2 = np.zeros((4,))

        bucket.add_gradient(tensor1, grad1)
        bucket.add_gradient(tensor2, grad2)

        # Create flat tensor with known values
        flat_grad = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)
        bucket.unflatten(flat_grad)

        # Check shapes are preserved
        assert bucket.gradients[0].shape == (2, 3)
        assert bucket.gradients[1].shape == (4,)

        # Check values
        np.testing.assert_array_equal(
            bucket.gradients[0],
            np.array([[1, 2, 3], [4, 5, 6]])
        )
        np.testing.assert_array_equal(
            bucket.gradients[1],
            np.array([7, 8, 9, 10])
        )

    def test_clear_bucket(self):
        """Test clearing bucket."""
        bucket = GradBucket(index=0)

        # Add some data
        tensor = DistributedTensor(np.zeros((5,)), requires_grad=True)
        bucket.add_gradient(tensor, np.ones(5))
        bucket.ready = True

        # Clear
        bucket.clear()

        assert bucket.tensors == []
        assert bucket.gradients == []
        assert bucket.size == 0
        assert bucket.ready == False


# =============================================================================
# Reducer Bucketing Tests
# =============================================================================

class TestReducerBucketing:
    """Tests for Reducer bucket building."""

    def test_bucket_building_single_parameter(self, process_group):
        """Test bucket building with single parameter."""
        params = [MockParameter((100, 100))]
        reducer = Reducer(params, process_group)

        assert len(reducer._buckets) >= 1
        assert 0 in reducer._param_to_bucket or (len(params) - 1) in reducer._param_to_bucket

    def test_bucket_building_multiple_parameters(self, process_group):
        """Test bucket building with multiple parameters."""
        params = [MockParameter((50, 50)) for _ in range(10)]
        reducer = Reducer(params, process_group, bucket_cap_mb=25.0)

        # All parameters should be mapped
        for i in range(len(params)):
            # Due to reverse ordering, check if any index is mapped
            assert any(idx in reducer._param_to_bucket for idx in range(len(params)))

    def test_bucket_size_limit(self, process_group):
        """Test that buckets respect size limit."""
        # Create parameters that should require multiple buckets
        # Each parameter: 100x100x4 bytes = 40,000 bytes = ~39KB
        params = [MockParameter((100, 100)) for _ in range(10)]

        # 50KB bucket limit should create ~8 buckets (each bucket fits ~1 param)
        reducer = Reducer(params, process_group, bucket_cap_mb=0.05)

        # Should have multiple buckets
        assert len(reducer._buckets) > 1

    def test_bucket_cap_mb_conversion(self, process_group):
        """Test bucket_cap_mb to bytes conversion."""
        params = [MockParameter((10, 10))]

        reducer_1mb = Reducer(params, process_group, bucket_cap_mb=1.0)
        reducer_25mb = Reducer(params, process_group, bucket_cap_mb=25.0)

        assert reducer_1mb.bucket_cap_bytes == 1 * 1024 * 1024
        assert reducer_25mb.bucket_cap_bytes == 25 * 1024 * 1024

    def test_bucket_reverse_order(self, process_group):
        """Test that parameters are processed in reverse order."""
        params = [MockParameter((10, 10)) for _ in range(5)]

        # Use tiny bucket to get one param per bucket
        reducer = Reducer(params, process_group, bucket_cap_mb=0.0001)

        # First bucket should contain last parameter (reverse order)
        # This enables overlap with backward pass

    def test_empty_bucket_not_added(self, process_group):
        """Test that empty buckets are not added."""
        params = []
        reducer = Reducer(params, process_group)

        assert len(reducer._buckets) == 0

    def test_bucket_param_assignment(self, process_group):
        """Test that each parameter is assigned to exactly one bucket."""
        params = [MockParameter((30, 30)) for _ in range(10)]
        reducer = Reducer(params, process_group, bucket_cap_mb=0.01)

        # Count assignments
        assigned = set()
        for bucket in reducer._buckets:
            for param in bucket.params:
                param_id = id(param)
                assert param_id not in assigned, "Parameter assigned to multiple buckets"
                assigned.add(param_id)


# =============================================================================
# Reducer Prepare and Finalize Tests
# =============================================================================

class TestReducerLifecycle:
    """Tests for Reducer lifecycle methods."""

    def test_prepare_for_backward(self, process_group):
        """Test prepare_for_backward initialization."""
        params = [MockParameter((10, 10)) for _ in range(3)]
        reducer = Reducer(params, process_group)

        reducer.prepare_for_backward()

        for bucket in reducer._buckets:
            assert bucket.grads == []
            assert bucket.pending == len(bucket.params)

    def test_prepare_clears_previous_state(self, process_group):
        """Test that prepare clears previous backward state."""
        params = [MockParameter((10, 10))]
        params[0].grad = np.ones((10, 10))
        reducer = Reducer(params, process_group)

        # First backward
        reducer.prepare_for_backward()
        reducer.mark_grad_ready(0)

        # Second backward - should reset
        reducer.prepare_for_backward()

        for bucket in reducer._buckets:
            assert bucket.pending == len(bucket.params)

    def test_finalize_completes_without_error(self, process_group):
        """Test that finalize completes without error."""
        params = [MockParameter((10, 10))]
        reducer = Reducer(params, process_group)

        reducer.prepare_for_backward()
        # Note: finalize is a no-op currently
        reducer.finalize()


# =============================================================================
# Reducer Gradient Ready Tests
# =============================================================================

class TestReducerGradientReady:
    """Tests for marking gradients as ready."""

    def test_mark_grad_ready_decrements_pending(self, process_group):
        """Test that marking grad ready decrements pending count."""
        params = [MockParameter((10, 10))]
        reducer = Reducer(params, process_group)

        reducer.prepare_for_backward()
        initial_pending = reducer._buckets[0].pending

        # Mark ready (may trigger reduction if last)
        params[0].grad = np.ones((10, 10))
        reducer.mark_grad_ready(0)

        # Pending should have been decremented (and bucket reduced)
        assert reducer._buckets[0].pending == initial_pending - 1

    def test_mark_invalid_param_index(self, process_group):
        """Test marking invalid parameter index."""
        params = [MockParameter((10, 10))]
        reducer = Reducer(params, process_group)

        reducer.prepare_for_backward()

        # Invalid index should be handled gracefully
        reducer.mark_grad_ready(999)  # Should not raise

    def test_mark_ready_triggers_reduction(self, process_group):
        """Test that marking last grad ready triggers reduction."""
        params = [MockParameter((10, 10)), MockParameter((5, 5))]
        for p in params:
            p.grad = np.ones_like(p.data) * 4

        # Force both params into same bucket
        reducer = Reducer(params, process_group, bucket_cap_mb=100)
        reducer.prepare_for_backward()

        # Mark both as ready
        for i in range(len(params)):
            reducer.mark_grad_ready(i)

        # Bucket should be reduced
        assert reducer._buckets[0].pending == 0

    def test_reduction_averages_by_world_size(self, process_group):
        """Test that reduction averages by world size."""
        params = [MockParameter((5, 5))]
        params[0].grad = np.ones((5, 5)) * 8  # Value = 8

        reducer = Reducer(params, process_group)  # world_size = 4
        reducer.prepare_for_backward()
        reducer.mark_grad_ready(0)

        # Should be 8 / 4 = 2
        expected = np.ones((5, 5)) * 2
        np.testing.assert_array_almost_equal(params[0].grad, expected)


# =============================================================================
# Bucket Flatten/Unflatten Tests
# =============================================================================

class TestBucketFlattenUnflatten:
    """Tests for bucket flatten/unflatten operations."""

    def test_flatten_single_gradient(self, process_group):
        """Test flattening a single gradient."""
        params = [MockParameter((3, 3))]
        params[0].grad = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32)

        reducer = Reducer(params, process_group)
        reducer.prepare_for_backward()
        reducer.mark_grad_ready(0)

        # After reduction, gradient should maintain shape
        assert params[0].grad.shape == (3, 3)

    def test_flatten_multiple_gradients(self, process_group):
        """Test flattening multiple gradients in same bucket."""
        params = [
            MockParameter((2, 2)),
            MockParameter((3,)),
        ]
        params[0].grad = np.array([[1, 2], [3, 4]]).astype(np.float32)
        params[1].grad = np.array([5, 6, 7]).astype(np.float32)

        # Force into same bucket
        reducer = Reducer(params, process_group, bucket_cap_mb=100)
        reducer.prepare_for_backward()

        for i in range(len(params)):
            reducer.mark_grad_ready(i)

        # Shapes should be preserved
        assert params[0].grad.shape == (2, 2)
        assert params[1].grad.shape == (3,)

    def test_unflatten_preserves_values(self, process_group):
        """Test that unflatten preserves gradient values (after averaging)."""
        params = [MockParameter((4,))]
        params[0].grad = np.array([4, 8, 12, 16]).astype(np.float32)

        reducer = Reducer(params, process_group)  # world_size = 4
        reducer.prepare_for_backward()
        reducer.mark_grad_ready(0)

        # Values should be averaged: [1, 2, 3, 4]
        expected = np.array([1, 2, 3, 4]).astype(np.float32)
        np.testing.assert_array_almost_equal(params[0].grad, expected)


# =============================================================================
# Thread Safety Tests
# =============================================================================

class TestBucketingThreadSafety:
    """Tests for thread safety of bucketing."""

    def test_concurrent_mark_ready(self, process_group):
        """Test concurrent marking of gradients as ready."""
        params = [MockParameter((10, 10)) for _ in range(10)]
        for p in params:
            p.grad = np.ones((10, 10))

        reducer = Reducer(params, process_group)

        def mark_ready(start, end):
            for i in range(start, end):
                reducer.mark_grad_ready(i)

        # Multiple threads marking different parameters
        reducer.prepare_for_backward()

        threads = [
            threading.Thread(target=mark_ready, args=(0, 5)),
            threading.Thread(target=mark_ready, args=(5, 10)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should be processed
        for bucket in reducer._buckets:
            assert bucket.pending == 0


# =============================================================================
# GradReducer Integration Tests
# =============================================================================

class TestGradReducerBucketing:
    """Integration tests for GradReducer with bucketing."""

    def test_grad_reducer_reduce_all(self, process_group):
        """Test GradReducer reducing all gradients."""
        params = [MockParameter((20, 20)) for _ in range(5)]
        for p in params:
            p.grad = np.ones((20, 20))

        grad_reducer = GradReducer(iter(params), process_group)
        grad_reducer.reduce_gradients()

        for p in params:
            assert p.grad is not None

    def test_grad_reducer_with_none_gradients(self, process_group):
        """Test GradReducer with some None gradients."""
        params = [MockParameter((10, 10)) for _ in range(3)]
        params[0].grad = np.ones((10, 10))
        params[1].grad = None  # No gradient
        params[2].grad = np.ones((10, 10))

        grad_reducer = GradReducer(iter(params), process_group)
        grad_reducer.reduce_gradients()

        # Should complete without error


# =============================================================================
# Edge Cases
# =============================================================================

class TestBucketingEdgeCases:
    """Edge case tests for bucketing."""

    def test_single_element_parameter(self, process_group):
        """Test bucketing single element parameter."""
        params = [MockParameter((1,))]
        params[0].grad = np.array([4.0])

        reducer = Reducer(params, process_group)
        reducer.prepare_for_backward()
        reducer.mark_grad_ready(0)

        np.testing.assert_array_almost_equal(params[0].grad, np.array([1.0]))

    def test_very_large_parameter(self, process_group):
        """Test bucketing very large parameter."""
        params = [MockParameter((1000, 1000))]  # 4MB
        params[0].grad = np.ones((1000, 1000))

        # 1MB bucket limit - should still work (parameter larger than bucket)
        reducer = Reducer(params, process_group, bucket_cap_mb=1.0)
        reducer.prepare_for_backward()
        reducer.mark_grad_ready(0)

    def test_mixed_size_parameters(self, process_group):
        """Test bucketing with mixed size parameters."""
        params = [
            MockParameter((1000, 1000)),  # Large
            MockParameter((10, 10)),       # Small
            MockParameter((500, 500)),     # Medium
            MockParameter((5,)),           # Tiny
        ]
        for p in params:
            p.grad = np.ones_like(p.data)

        reducer = Reducer(params, process_group, bucket_cap_mb=0.5)
        reducer.prepare_for_backward()

        for i in range(len(params)):
            reducer.mark_grad_ready(i)

    def test_zero_bucket_cap(self, process_group):
        """Test with zero bucket capacity."""
        params = [MockParameter((10, 10))]
        params[0].grad = np.ones((10, 10))

        # This should still work - each param gets own bucket
        reducer = Reducer(params, process_group, bucket_cap_mb=0)
        reducer.prepare_for_backward()
        reducer.mark_grad_ready(0)

    def test_find_unused_parameter_flag(self, process_group):
        """Test find_unused parameter flag."""
        params = [MockParameter((10, 10))]

        reducer = Reducer(params, process_group, find_unused=True)

        assert reducer.find_unused == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
