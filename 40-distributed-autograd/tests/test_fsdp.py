"""Comprehensive tests for Fully Sharded Data Parallel (FSDP) implementation."""

import pytest
import numpy as np
import threading
from unittest.mock import Mock, MagicMock, patch
from typing import List, Iterator, Any

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from distautograd.distributed.ddp import (
    FullyShardedDataParallel,
)
from distautograd.core.context import (
    ProcessGroup,
    Backend,
    DistributedContext,
    WorkerInfo,
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
        self._full_data = None


class MockModule:
    """Mock module for testing FSDP."""

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


@pytest.fixture
def mock_module():
    """Create a mock module."""
    return MockModule()


@pytest.fixture
def process_group():
    """Create a process group for testing."""
    return ProcessGroup(ranks=[0, 1, 2, 3], backend=Backend.GLOO, name="test_group")


@pytest.fixture
def two_worker_group():
    """Create a 2-worker process group."""
    return ProcessGroup(ranks=[0, 1], backend=Backend.GLOO, name="two_worker")


# =============================================================================
# FSDP Initialization Tests
# =============================================================================

class TestFSDPInitialization:
    """Tests for FSDP initialization."""

    def test_fsdp_basic_initialization(self, mock_module, process_group):
        """Test basic FSDP initialization."""
        fsdp = FullyShardedDataParallel(
            mock_module,
            process_group=process_group
        )

        assert fsdp.module == mock_module
        assert fsdp.process_group == process_group
        assert fsdp.sharding_strategy == "FULL_SHARD"

    def test_fsdp_sharding_strategies(self, mock_module, process_group):
        """Test different sharding strategies."""
        strategies = ["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"]

        for strategy in strategies:
            fsdp = FullyShardedDataParallel(
                mock_module,
                process_group=process_group,
                sharding_strategy=strategy
            )
            assert fsdp.sharding_strategy == strategy

    def test_fsdp_cpu_offload(self, mock_module, process_group):
        """Test FSDP with CPU offload enabled."""
        fsdp = FullyShardedDataParallel(
            mock_module,
            process_group=process_group,
            cpu_offload=True
        )

        assert fsdp.cpu_offload == True

    def test_fsdp_backward_prefetch(self, mock_module, process_group):
        """Test FSDP backward prefetch options."""
        prefetch_options = ["BACKWARD_PRE", "BACKWARD_POST"]

        for prefetch in prefetch_options:
            fsdp = FullyShardedDataParallel(
                mock_module,
                process_group=process_group,
                backward_prefetch=prefetch
            )
            assert fsdp.backward_prefetch == prefetch

    def test_fsdp_without_process_group(self, mock_module):
        """Test FSDP without process group (no sharding)."""
        fsdp = FullyShardedDataParallel(mock_module)

        assert fsdp.process_group is None
        # Parameters should remain unsharded
        for param in fsdp.parameters():
            assert not hasattr(param, '_full_data') or param._full_data is None

    def test_fsdp_module_without_parameters(self, process_group):
        """Test FSDP with module that has no parameters."""
        class NoParamModule:
            def __call__(self, x):
                return x * 2

        module = NoParamModule()
        fsdp = FullyShardedDataParallel(module, process_group=process_group)

        assert fsdp._parameters == []


# =============================================================================
# Parameter Sharding Tests
# =============================================================================

class TestParameterSharding:
    """Tests for parameter sharding functionality."""

    def test_parameter_sharding_basic(self, two_worker_group):
        """Test basic parameter sharding."""
        # Create module with known parameter sizes
        module = MockModule(param_shapes=[(100,)])
        param = list(module.parameters())[0]
        original_size = param.data.size

        fsdp = FullyShardedDataParallel(
            module,
            process_group=two_worker_group
        )

        # After sharding, parameter should be smaller
        sharded_param = list(fsdp.parameters())[0]
        expected_shard_size = original_size // 2
        assert sharded_param.data.size == expected_shard_size

    def test_full_data_preserved(self, two_worker_group):
        """Test that full data is preserved in _full_data."""
        module = MockModule(param_shapes=[(100,)])
        param = list(module.parameters())[0]
        original_data = param.data.copy()

        fsdp = FullyShardedDataParallel(
            module,
            process_group=two_worker_group
        )

        # Full data should be stored
        sharded_param = list(fsdp.parameters())[0]
        np.testing.assert_array_equal(sharded_param._full_data, original_data)

    def test_sharding_multiple_parameters(self, two_worker_group):
        """Test sharding with multiple parameters."""
        module = MockModule(param_shapes=[(100,), (50,), (20,)])

        fsdp = FullyShardedDataParallel(
            module,
            process_group=two_worker_group
        )

        params = list(fsdp.parameters())
        assert len(params) == 3

        # Each should be sharded
        assert params[0].data.size == 50  # 100 / 2
        assert params[1].data.size == 25  # 50 / 2
        assert params[2].data.size == 10  # 20 / 2

    def test_sharding_with_four_workers(self, process_group):
        """Test sharding across 4 workers."""
        module = MockModule(param_shapes=[(100,)])

        fsdp = FullyShardedDataParallel(
            module,
            process_group=process_group  # 4 workers
        )

        param = list(fsdp.parameters())[0]
        # Should be 100 / 4 = 25
        assert param.data.size == 25

    def test_sharding_preserves_dtype(self, two_worker_group):
        """Test that sharding preserves data type."""
        module = MockModule(param_shapes=[(100,)])
        param = list(module.parameters())[0]
        param.data = param.data.astype(np.float64)

        fsdp = FullyShardedDataParallel(
            module,
            process_group=two_worker_group
        )

        sharded_param = list(fsdp.parameters())[0]
        assert sharded_param.data.dtype == np.float64


# =============================================================================
# All-Gather Tests
# =============================================================================

class TestAllGather:
    """Tests for all-gather operations."""

    def test_all_gather_params(self, two_worker_group):
        """Test all-gather restores full parameters."""
        module = MockModule(param_shapes=[(100,)])
        original_data = list(module.parameters())[0].data.copy()

        fsdp = FullyShardedDataParallel(
            module,
            process_group=two_worker_group
        )

        # After sharding, call all-gather
        fsdp._all_gather_params()

        # Parameters should be restored to full size
        param = list(fsdp.parameters())[0]
        np.testing.assert_array_equal(param.data, original_data)

    def test_all_gather_multiple_calls(self, two_worker_group):
        """Test multiple all-gather calls."""
        module = MockModule(param_shapes=[(100,)])

        fsdp = FullyShardedDataParallel(
            module,
            process_group=two_worker_group
        )

        # Multiple all-gather calls should be idempotent
        for _ in range(5):
            fsdp._all_gather_params()

        param = list(fsdp.parameters())[0]
        assert param.data is not None


# =============================================================================
# Forward Pass Tests
# =============================================================================

class TestFSDPForward:
    """Tests for FSDP forward pass."""

    def test_forward_with_all_gather(self, two_worker_group):
        """Test that forward triggers all-gather."""
        module = MockModule(param_shapes=[(100,)])

        fsdp = FullyShardedDataParallel(
            module,
            process_group=two_worker_group
        )

        # Before forward, params are sharded
        input_data = np.random.randn(10, 10).astype(np.float32)

        # Forward should call all-gather and run module
        output = fsdp.forward(input_data)

        # Output should match input (mock module returns input)
        np.testing.assert_array_equal(output, input_data)

    def test_forward_via_call(self, two_worker_group):
        """Test forward via __call__."""
        module = MockModule(param_shapes=[(100,)])

        fsdp = FullyShardedDataParallel(
            module,
            process_group=two_worker_group
        )

        input_data = np.random.randn(5, 5).astype(np.float32)
        output = fsdp(input_data)

        np.testing.assert_array_equal(output, input_data)

    def test_forward_with_custom_module(self, two_worker_group):
        """Test forward with custom module logic."""
        class CustomModule:
            def __init__(self):
                self._params = [MockParameter((100,))]

            def parameters(self):
                return iter(self._params)

            def __call__(self, x):
                return x * 2 + 1

        module = CustomModule()
        fsdp = FullyShardedDataParallel(
            module,
            process_group=two_worker_group
        )

        input_data = np.array([1, 2, 3, 4, 5]).astype(np.float32)
        output = fsdp(input_data)

        expected = input_data * 2 + 1
        np.testing.assert_array_equal(output, expected)

    def test_forward_with_kwargs(self, two_worker_group):
        """Test forward with keyword arguments."""
        class KwargsModule:
            def __init__(self):
                self._params = [MockParameter((10,))]

            def parameters(self):
                return iter(self._params)

            def __call__(self, x, scale=1.0):
                return x * scale

        module = KwargsModule()
        fsdp = FullyShardedDataParallel(
            module,
            process_group=two_worker_group
        )

        input_data = np.array([1, 2, 3]).astype(np.float32)
        output = fsdp(input_data, scale=2.0)

        expected = input_data * 2.0
        np.testing.assert_array_equal(output, expected)


# =============================================================================
# Parameters Iterator Tests
# =============================================================================

class TestFSDPParameters:
    """Tests for FSDP parameters method."""

    def test_parameters_returns_iterator(self, mock_module, process_group):
        """Test that parameters() returns an iterator."""
        fsdp = FullyShardedDataParallel(
            mock_module,
            process_group=process_group
        )

        params_iter = fsdp.parameters()

        # Should be iterable
        params_list = list(params_iter)
        assert len(params_list) == 3

    def test_parameters_match_module(self, mock_module, process_group):
        """Test that parameters match module parameters."""
        fsdp = FullyShardedDataParallel(
            mock_module,
            process_group=process_group
        )

        fsdp_params = list(fsdp.parameters())
        module_params = list(mock_module.parameters())

        assert len(fsdp_params) == len(module_params)

    def test_empty_parameters(self, process_group):
        """Test module with no parameters."""
        class EmptyModule:
            def __call__(self, x):
                return x

        module = EmptyModule()
        fsdp = FullyShardedDataParallel(module, process_group=process_group)

        params = list(fsdp.parameters())
        assert params == []


# =============================================================================
# FSDP Integration Tests
# =============================================================================

class TestFSDPIntegration:
    """Integration tests for FSDP."""

    def test_full_training_step(self, two_worker_group):
        """Test a complete training step with FSDP."""
        module = MockModule(param_shapes=[(100,), (50,)])

        fsdp = FullyShardedDataParallel(
            module,
            process_group=two_worker_group
        )

        # Forward pass
        input_data = np.random.randn(32, 10).astype(np.float32)
        output = fsdp(input_data)

        # Set gradients (simulate backward)
        for param in fsdp.parameters():
            if hasattr(param, '_full_data') and param._full_data is not None:
                param.grad = np.random.randn(*param._full_data.shape).astype(np.float32)
            else:
                param.grad = np.random.randn(*param.data.shape).astype(np.float32)

        # Verify forward completed
        assert output is not None

    def test_multiple_forward_backward(self, two_worker_group):
        """Test multiple forward/backward passes."""
        module = MockModule(param_shapes=[(100,)])

        fsdp = FullyShardedDataParallel(
            module,
            process_group=two_worker_group
        )

        for iteration in range(10):
            input_data = np.random.randn(16, 10).astype(np.float32)
            output = fsdp(input_data)

            for param in fsdp.parameters():
                param.grad = np.random.randn(*param.data.shape)

            assert output is not None

    def test_fsdp_memory_efficiency(self, process_group):
        """Test that FSDP reduces parameter memory footprint."""
        # Large parameter
        module = MockModule(param_shapes=[(10000,)])
        original_param = list(module.parameters())[0]
        original_size = original_param.data.nbytes

        fsdp = FullyShardedDataParallel(
            module,
            process_group=process_group  # 4 workers
        )

        sharded_param = list(fsdp.parameters())[0]
        sharded_size = sharded_param.data.nbytes

        # Sharded size should be 1/4 of original
        assert sharded_size == original_size // 4

    def test_fsdp_with_different_dtypes(self, two_worker_group):
        """Test FSDP with different data types."""
        dtypes = [np.float32, np.float64, np.float16]

        for dtype in dtypes:
            module = MockModule(param_shapes=[(100,)])
            param = list(module.parameters())[0]
            param.data = param.data.astype(dtype)

            fsdp = FullyShardedDataParallel(
                module,
                process_group=two_worker_group
            )

            input_data = np.random.randn(10, 10).astype(dtype)
            output = fsdp(input_data)

            assert output.dtype == dtype


# =============================================================================
# Edge Cases
# =============================================================================

class TestFSDPEdgeCases:
    """Edge case tests for FSDP."""

    def test_single_element_parameter(self, two_worker_group):
        """Test sharding a single-element parameter."""
        module = MockModule(param_shapes=[(2,)])  # Minimum for 2 workers

        fsdp = FullyShardedDataParallel(
            module,
            process_group=two_worker_group
        )

        param = list(fsdp.parameters())[0]
        assert param.data.size == 1  # 2 / 2 = 1

    def test_odd_sized_parameter(self, two_worker_group):
        """Test sharding parameter with odd size."""
        module = MockModule(param_shapes=[(101,)])

        fsdp = FullyShardedDataParallel(
            module,
            process_group=two_worker_group
        )

        # Should handle odd sizes gracefully
        param = list(fsdp.parameters())[0]
        assert param.data.size == 50  # 101 // 2 = 50 (integer division)

    def test_very_large_world_size(self):
        """Test with large world size."""
        large_group = ProcessGroup(ranks=list(range(8)), backend=Backend.GLOO)

        module = MockModule(param_shapes=[(1000,)])

        fsdp = FullyShardedDataParallel(
            module,
            process_group=large_group
        )

        param = list(fsdp.parameters())[0]
        assert param.data.size == 125  # 1000 / 8

    def test_module_with_forward_method(self, two_worker_group):
        """Test module that only has forward method."""
        class ForwardOnlyModule:
            def __init__(self):
                self._params = [MockParameter((100,))]

            def parameters(self):
                return iter(self._params)

            def forward(self, x):
                return x + 1

        module = ForwardOnlyModule()
        fsdp = FullyShardedDataParallel(
            module,
            process_group=two_worker_group
        )

        input_data = np.ones((5, 5)).astype(np.float32)
        output = fsdp(input_data)

        expected = input_data + 1
        np.testing.assert_array_equal(output, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
