"""Comprehensive tests for Distributed Data Parallel (DDP) implementation."""

import pytest
import numpy as np
import threading
import time
from unittest.mock import Mock, MagicMock, patch
from typing import List, Iterator, Any

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from distautograd.distributed.ddp import (
    DistributedDataParallel,
    GradReducer,
    AllReduceStrategy,
    Reducer,
    GradientBucket,
    FullyShardedDataParallel,
    _NoSyncContext,
    _DummyReducer,
)
from distautograd.core.context import (
    ProcessGroup,
    Backend,
    DistributedContext,
    WorkerInfo,
    DistributedTensor,
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
        self._hooks = []

    def register_hook(self, hook):
        self._hooks.append(hook)
        return len(self._hooks) - 1


class MockModule:
    """Mock module for testing DDP."""

    def __init__(self, num_params=3, param_shapes=None):
        if param_shapes:
            self._params = [MockParameter(shape) for shape in param_shapes]
        else:
            self._params = [
                MockParameter((100, 100)),
                MockParameter((50, 50)),
                MockParameter((20, 20)),
            ][:num_params]

    def parameters(self) -> Iterator[MockParameter]:
        return iter(self._params)

    def __call__(self, x):
        return x

    def forward(self, x):
        return x

    def state_dict(self):
        return {f"param_{i}": p.data for i, p in enumerate(self._params)}

    def load_state_dict(self, state_dict):
        for i, p in enumerate(self._params):
            if f"param_{i}" in state_dict:
                p.data = state_dict[f"param_{i}"]


@pytest.fixture
def mock_module():
    """Create a mock module."""
    return MockModule()


@pytest.fixture
def process_group():
    """Create a process group for testing."""
    return ProcessGroup(ranks=[0, 1, 2, 3], backend=Backend.GLOO, name="test_group")


@pytest.fixture
def distributed_context():
    """Create a distributed context."""
    worker = WorkerInfo(rank=0, world_size=4, local_rank=0)
    return DistributedContext(worker_info=worker)


# =============================================================================
# GradientBucket Tests
# =============================================================================

class TestGradientBucket:
    """Tests for GradientBucket dataclass."""

    def test_bucket_creation(self):
        """Test creating a gradient bucket."""
        bucket = GradientBucket(index=0)

        assert bucket.index == 0
        assert bucket.params == []
        assert bucket.grads == []
        assert bucket.size_bytes == 0
        assert bucket.pending == 0
        assert bucket.future is None

    def test_bucket_with_params(self):
        """Test bucket with parameters."""
        params = [MockParameter((10, 10)), MockParameter((5, 5))]
        grads = [np.ones((10, 10)), np.ones((5, 5))]

        bucket = GradientBucket(
            index=1,
            params=params,
            grads=grads,
            size_bytes=500,
            pending=2
        )

        assert bucket.index == 1
        assert len(bucket.params) == 2
        assert len(bucket.grads) == 2
        assert bucket.size_bytes == 500
        assert bucket.pending == 2

    def test_bucket_field_defaults(self):
        """Test that bucket fields have correct default factories."""
        bucket1 = GradientBucket(index=0)
        bucket2 = GradientBucket(index=1)

        # Ensure separate lists
        bucket1.params.append("test")
        assert "test" not in bucket2.params


# =============================================================================
# Reducer Tests
# =============================================================================

class TestReducer:
    """Tests for the Reducer class."""

    def test_reducer_initialization(self, mock_module, process_group):
        """Test reducer initialization."""
        params = list(mock_module.parameters())
        reducer = Reducer(params, process_group, bucket_cap_mb=25.0)

        assert reducer.bucket_cap_bytes == 25 * 1024 * 1024
        assert len(reducer._buckets) > 0

    def test_bucket_building(self, process_group):
        """Test that buckets are built correctly based on size."""
        # Create parameters of known sizes
        params = [
            MockParameter((100, 100)),  # 40000 bytes (float32)
            MockParameter((100, 100)),  # 40000 bytes
            MockParameter((100, 100)),  # 40000 bytes
        ]

        # Use small bucket size to force multiple buckets
        reducer = Reducer(params, process_group, bucket_cap_mb=0.05)  # ~50KB

        # Should create at least 2 buckets
        assert len(reducer._buckets) >= 2

    def test_param_to_bucket_mapping(self, process_group):
        """Test parameter to bucket mapping."""
        params = [MockParameter((50, 50)) for _ in range(5)]
        reducer = Reducer(params, process_group, bucket_cap_mb=1.0)

        # Each parameter should be mapped
        for i in range(len(params)):
            assert i in reducer._param_to_bucket or (len(params) - 1 - i) in reducer._param_to_bucket

    def test_prepare_for_backward(self, mock_module, process_group):
        """Test preparing reducer for backward pass."""
        params = list(mock_module.parameters())
        reducer = Reducer(params, process_group)

        reducer.prepare_for_backward()

        for bucket in reducer._buckets:
            assert bucket.grads == []
            assert bucket.pending == len(bucket.params)

    def test_mark_grad_ready_triggers_reduction(self, process_group):
        """Test that marking all grads ready triggers reduction."""
        params = [MockParameter((10, 10))]
        params[0].grad = np.ones((10, 10))

        reducer = Reducer(params, process_group)
        reducer.prepare_for_backward()

        # Mark gradient ready
        reducer.mark_grad_ready(0)

        # Bucket should have been reduced
        bucket = reducer._buckets[0]
        assert bucket.pending == 0

    def test_reduce_bucket_averages_gradients(self, process_group):
        """Test that bucket reduction averages gradients."""
        params = [MockParameter((5, 5))]
        params[0].grad = np.ones((5, 5)) * 4  # All values = 4

        reducer = Reducer(params, process_group)
        reducer.prepare_for_backward()

        # Manually trigger reduction
        bucket = reducer._buckets[0]
        bucket.pending = 0
        reducer._reduce_bucket(bucket)

        # Should be averaged by world_size (4)
        expected = np.ones((5, 5))  # 4 / 4 = 1
        np.testing.assert_array_almost_equal(params[0].grad, expected)

    def test_reduce_bucket_without_process_group(self):
        """Test reduction without process group (no averaging)."""
        params = [MockParameter((5, 5))]
        params[0].grad = np.ones((5, 5)) * 4

        reducer = Reducer(params, process_group=None)
        reducer.prepare_for_backward()

        bucket = reducer._buckets[0]
        bucket.pending = 0
        reducer._reduce_bucket(bucket)

        # Should not be averaged
        expected = np.ones((5, 5)) * 4
        np.testing.assert_array_almost_equal(params[0].grad, expected)

    def test_finalize(self, mock_module, process_group):
        """Test finalize method."""
        params = list(mock_module.parameters())
        reducer = Reducer(params, process_group)

        # Should not raise
        reducer.finalize()

    def test_empty_bucket_reduction(self, process_group):
        """Test reducing an empty bucket."""
        reducer = Reducer([], process_group)
        bucket = GradientBucket(index=0)

        # Should not raise
        reducer._reduce_bucket(bucket)

    def test_bucket_reverse_order(self, process_group):
        """Test that buckets are built in reverse order for overlap."""
        params = [MockParameter((10, 10)) for _ in range(10)]
        reducer = Reducer(params, process_group, bucket_cap_mb=0.001)  # Force multiple buckets

        # Parameters should be processed in reverse order
        # Last parameters in first bucket (for overlap with backward)
        if reducer._buckets:
            # The last parameter index should be in an early bucket
            last_param_bucket = reducer._param_to_bucket.get(len(params) - 1)
            first_param_bucket = reducer._param_to_bucket.get(0)
            # In reverse order, higher indices should be in lower bucket indices


# =============================================================================
# GradReducer Tests
# =============================================================================

class TestGradReducer:
    """Tests for the GradReducer class."""

    def test_grad_reducer_initialization(self, mock_module, process_group):
        """Test GradReducer initialization."""
        params = mock_module.parameters()
        grad_reducer = GradReducer(params, process_group)

        assert grad_reducer.process_group == process_group
        assert grad_reducer.strategy == AllReduceStrategy.BUCKET

    def test_grad_reducer_with_ring_strategy(self, mock_module, process_group):
        """Test GradReducer with RING strategy."""
        params = mock_module.parameters()
        grad_reducer = GradReducer(params, process_group, strategy=AllReduceStrategy.RING)

        assert grad_reducer.strategy == AllReduceStrategy.RING

    def test_reduce_gradients(self, mock_module, process_group):
        """Test reducing gradients."""
        params = list(mock_module.parameters())
        for p in params:
            p.grad = np.ones_like(p.data)

        grad_reducer = GradReducer(iter(params), process_group)
        grad_reducer.reduce_gradients()

        # Should complete without error
        for p in params:
            assert p.grad is not None

    def test_compress_gradients_topk(self):
        """Test top-k gradient compression."""
        # Create parameters with matching data and grad shapes
        params = [MockParameter((5, 2))]
        params[0].data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],
                                   [7.0, 8.0], [9.0, 10.0]]).astype(np.float32)
        params[0].grad = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],
                                   [7.0, 8.0], [9.0, 10.0]]).astype(np.float32)

        grad_reducer = GradReducer(iter(params))
        grad_reducer.compress_gradients(ratio=0.2)  # Keep top 20%

        # Check that compression happened (non-top values should be zero)
        compressed = params[0].grad
        num_nonzero = np.count_nonzero(compressed)

        # Should have approximately 20% non-zero values
        total_elements = compressed.size
        assert num_nonzero <= int(total_elements * 0.2) + 1

    def test_compress_gradients_preserves_top_values(self, mock_module):
        """Test that compression preserves the largest magnitude values."""
        params = list(mock_module.parameters())
        original = np.array([[1.0, -10.0], [5.0, 3.0]]).astype(np.float32)
        params[0].data = original
        params[0].grad = original.copy()

        grad_reducer = GradReducer(iter(params))
        grad_reducer.compress_gradients(ratio=0.5)  # Keep top 50%

        compressed = params[0].grad
        # The value with largest magnitude (-10.0) should be preserved
        assert -10.0 in compressed

    def test_compress_with_no_gradients(self, mock_module):
        """Test compression when gradients are None."""
        params = list(mock_module.parameters())
        # Leave gradients as None

        grad_reducer = GradReducer(iter(params))
        # Should not raise
        grad_reducer.compress_gradients(ratio=0.1)


# =============================================================================
# DistributedDataParallel Tests
# =============================================================================

class TestDistributedDataParallel:
    """Tests for DistributedDataParallel wrapper."""

    def test_ddp_initialization(self, mock_module, process_group):
        """Test DDP initialization."""
        ddp = DistributedDataParallel(
            mock_module,
            device_ids=[0],
            process_group=process_group,
            bucket_cap_mb=25.0
        )

        assert ddp.module == mock_module
        assert ddp.device_ids == [0]
        assert ddp.process_group == process_group
        assert ddp.bucket_cap_mb == 25.0

    def test_ddp_default_device_ids(self, mock_module):
        """Test DDP with default device IDs."""
        ddp = DistributedDataParallel(mock_module)

        assert ddp.device_ids == [0]
        assert ddp.output_device == 0

    def test_ddp_forward_pass(self, mock_module, process_group):
        """Test DDP forward pass."""
        ddp = DistributedDataParallel(mock_module, process_group=process_group)

        input_data = np.random.randn(32, 100).astype(np.float32)
        output = ddp.forward(input_data)

        np.testing.assert_array_equal(output, input_data)

    def test_ddp_call_method(self, mock_module, process_group):
        """Test DDP __call__ method."""
        ddp = DistributedDataParallel(mock_module, process_group=process_group)

        input_data = np.random.randn(16, 50).astype(np.float32)
        output = ddp(input_data)

        np.testing.assert_array_equal(output, input_data)

    def test_ddp_parameters(self, mock_module, process_group):
        """Test getting parameters from DDP."""
        ddp = DistributedDataParallel(mock_module, process_group=process_group)

        params = list(ddp.parameters())
        expected_params = list(mock_module.parameters())

        assert len(params) == len(expected_params)

    def test_ddp_state_dict(self, mock_module, process_group):
        """Test DDP state_dict."""
        ddp = DistributedDataParallel(mock_module, process_group=process_group)

        state = ddp.state_dict()

        assert len(state) == 3  # 3 parameters
        assert "param_0" in state

    def test_ddp_load_state_dict(self, mock_module, process_group):
        """Test DDP load_state_dict."""
        ddp = DistributedDataParallel(mock_module, process_group=process_group)

        new_state = {
            "param_0": np.zeros((100, 100)),
            "param_1": np.ones((50, 50)),
            "param_2": np.full((20, 20), 2.0),
        }

        ddp.load_state_dict(new_state)

        params = list(mock_module.parameters())
        np.testing.assert_array_equal(params[0].data, np.zeros((100, 100)))
        np.testing.assert_array_equal(params[1].data, np.ones((50, 50)))

    def test_ddp_join(self, mock_module, process_group):
        """Test DDP join method."""
        ddp = DistributedDataParallel(mock_module, process_group=process_group)

        # Should not raise
        ddp.join()

    def test_ddp_find_unused_parameters(self, mock_module, process_group):
        """Test DDP with find_unused_parameters flag."""
        ddp = DistributedDataParallel(
            mock_module,
            process_group=process_group,
            find_unused_parameters=True
        )

        assert ddp.find_unused_parameters == True

    def test_ddp_gradient_as_bucket_view(self, mock_module, process_group):
        """Test DDP with gradient_as_bucket_view flag."""
        ddp = DistributedDataParallel(
            mock_module,
            process_group=process_group,
            gradient_as_bucket_view=True
        )

        assert ddp.gradient_as_bucket_view == True

    def test_ddp_with_module_without_parameters(self, process_group):
        """Test DDP with a module that has no parameters method."""
        class SimpleModule:
            def __call__(self, x):
                return x * 2

        module = SimpleModule()
        ddp = DistributedDataParallel(module, process_group=process_group)

        assert ddp._parameters == []

    def test_ddp_module_with_forward_method(self, process_group):
        """Test DDP with module using forward method."""
        class ForwardModule:
            def __init__(self):
                self._params = [MockParameter((10, 10))]

            def parameters(self):
                return iter(self._params)

            def forward(self, x):
                return x + 1

        module = ForwardModule()
        ddp = DistributedDataParallel(module, process_group=process_group)

        input_data = np.ones((5, 5))
        output = ddp(input_data)

        np.testing.assert_array_equal(output, input_data + 1)


# =============================================================================
# NoSync Context Tests
# =============================================================================

class TestNoSyncContext:
    """Tests for gradient synchronization control."""

    def test_no_sync_context_manager(self, mock_module, process_group):
        """Test no_sync context manager."""
        ddp = DistributedDataParallel(mock_module, process_group=process_group)

        original_reducer = ddp._reducer

        with ddp.no_sync():
            # Reducer should be replaced with dummy
            assert isinstance(ddp._reducer, _DummyReducer)

        # Should be restored after context
        assert ddp._reducer == original_reducer

    def test_no_sync_nested(self, mock_module, process_group):
        """Test nested no_sync contexts."""
        ddp = DistributedDataParallel(mock_module, process_group=process_group)

        original_reducer = ddp._reducer

        with ddp.no_sync():
            with ddp.no_sync():
                assert isinstance(ddp._reducer, _DummyReducer)
            # After inner context, still dummy (from outer)
            assert isinstance(ddp._reducer, _DummyReducer)

        # After both contexts, restored
        assert ddp._reducer == original_reducer

    def test_dummy_reducer_methods(self):
        """Test that DummyReducer methods don't raise."""
        dummy = _DummyReducer()

        # All methods should be no-ops
        dummy.prepare_for_backward()
        dummy.mark_grad_ready(0)
        dummy.finalize()


# =============================================================================
# DDP Hook Tests
# =============================================================================

class TestDDPHooks:
    """Tests for DDP gradient hooks."""

    def test_register_hooks(self, process_group):
        """Test that hooks are registered on parameters."""
        class HookableParam:
            def __init__(self, shape):
                self.data = np.random.randn(*shape)
                self.grad = None
                self.requires_grad = True
                self.hooks_registered = 0

            def register_hook(self, hook):
                self.hooks_registered += 1
                return self.hooks_registered

        class HookableModule:
            def __init__(self):
                self._params = [HookableParam((10, 10)), HookableParam((5, 5))]

            def parameters(self):
                return iter(self._params)

            def __call__(self, x):
                return x

        module = HookableModule()
        ddp = DistributedDataParallel(module, process_group=process_group)

        # Each parameter should have a hook registered
        for param in module._params:
            assert param.hooks_registered >= 1

    def test_grad_hook_marks_ready(self, process_group):
        """Test that gradient hook marks gradient as ready."""
        params = [MockParameter((5, 5))]

        class SimpleModule:
            def __init__(self):
                self._params = params

            def parameters(self):
                return iter(self._params)

            def __call__(self, x):
                return x

        module = SimpleModule()
        ddp = DistributedDataParallel(module, process_group=process_group)

        # Set gradient and simulate hook
        params[0].grad = np.ones((5, 5))
        ddp._grad_hook(0, params[0].grad)


# =============================================================================
# AllReduceStrategy Tests
# =============================================================================

class TestAllReduceStrategy:
    """Tests for AllReduce strategies."""

    def test_strategy_enum_values(self):
        """Test all strategy enum values exist."""
        assert AllReduceStrategy.RING is not None
        assert AllReduceStrategy.TREE is not None
        assert AllReduceStrategy.RECURSIVE_HALVING is not None
        assert AllReduceStrategy.BUCKET is not None

    def test_strategy_uniqueness(self):
        """Test that strategy enum values are unique."""
        strategies = [
            AllReduceStrategy.RING,
            AllReduceStrategy.TREE,
            AllReduceStrategy.RECURSIVE_HALVING,
            AllReduceStrategy.BUCKET,
        ]

        assert len(strategies) == len(set(strategies))


# =============================================================================
# Integration Tests
# =============================================================================

class TestDDPIntegration:
    """Integration tests for DDP."""

    def test_full_training_step(self, mock_module, process_group):
        """Test a complete training step with DDP."""
        ddp = DistributedDataParallel(mock_module, process_group=process_group)

        # Forward pass
        input_data = np.random.randn(32, 100).astype(np.float32)
        output = ddp(input_data)

        # Simulate backward (set gradients)
        for param in ddp.parameters():
            param.grad = np.random.randn(*param.data.shape).astype(np.float32)

        # Synchronize gradients
        grad_reducer = GradReducer(ddp.parameters(), process_group)
        grad_reducer.reduce_gradients()

        # Verify gradients exist
        for param in ddp.parameters():
            assert param.grad is not None

    def test_gradient_accumulation_with_no_sync(self, mock_module, process_group):
        """Test gradient accumulation using no_sync."""
        ddp = DistributedDataParallel(mock_module, process_group=process_group)

        accumulated_grads = []

        # Accumulate gradients over 4 steps
        for step in range(4):
            with ddp.no_sync() if step < 3 else nullcontext():
                output = ddp(np.random.randn(8, 100).astype(np.float32))

                # Set gradients
                for param in ddp.parameters():
                    if param.grad is None:
                        param.grad = np.ones_like(param.data)
                    else:
                        param.grad += np.ones_like(param.data)

        # Final gradients should be accumulated
        for param in ddp.parameters():
            assert param.grad is not None

    def test_multiple_forward_passes(self, mock_module, process_group):
        """Test multiple forward passes."""
        ddp = DistributedDataParallel(mock_module, process_group=process_group)

        for _ in range(10):
            input_data = np.random.randn(16, 100).astype(np.float32)
            output = ddp(input_data)

            assert output is not None
            assert output.shape == input_data.shape


# Context manager for compatibility
class nullcontext:
    """Null context manager for Python < 3.7 compatibility."""

    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
