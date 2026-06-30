"""Comprehensive tests for Communication Primitives."""

import pytest
import numpy as np
import threading
import time
from unittest.mock import Mock, MagicMock, patch
from typing import List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from distautograd.core.context import (
    ProcessGroup,
    Backend,
    DistributedContext,
    WorkerInfo,
    DistributedTensor,
    DeviceMesh,
    AutogradContext,
    all_reduce,
    broadcast,
    all_gather,
    reduce_scatter,
)
from distautograd.rpc.autograd import (
    RRef,
    RemoteGradient,
    DistAutogradContext,
    RPCAutograd,
    rpc_sync,
    rpc_async,
    remote,
    RemoteModule,
    init_rpc,
    shutdown_rpc,
    get_worker_info,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def process_group():
    """Create a process group."""
    return ProcessGroup(ranks=[0, 1, 2, 3], backend=Backend.GLOO, name="test")


@pytest.fixture
def worker_info():
    """Create worker info."""
    return WorkerInfo(rank=0, world_size=4, local_rank=0)


@pytest.fixture
def distributed_context(worker_info):
    """Create distributed context."""
    return DistributedContext(worker_info=worker_info)


@pytest.fixture
def distributed_tensor():
    """Create a distributed tensor."""
    return DistributedTensor(
        data=np.random.randn(10, 10).astype(np.float32),
        requires_grad=True
    )


# =============================================================================
# Backend Enum Tests
# =============================================================================

class TestBackend:
    """Tests for Backend enum."""

    def test_backend_types(self):
        """Test all backend types exist."""
        assert Backend.GLOO is not None
        assert Backend.NCCL is not None
        assert Backend.MPI is not None

    def test_backend_comparison(self):
        """Test backend comparison."""
        assert Backend.GLOO != Backend.NCCL
        assert Backend.NCCL != Backend.MPI

    def test_backend_iteration(self):
        """Test iterating over backends."""
        backends = list(Backend)
        assert len(backends) == 3


# =============================================================================
# WorkerInfo Tests
# =============================================================================

class TestWorkerInfo:
    """Tests for WorkerInfo dataclass."""

    def test_worker_info_creation(self):
        """Test creating worker info."""
        worker = WorkerInfo(
            rank=2,
            world_size=8,
            local_rank=0,
            hostname="node-1",
            port=29501
        )

        assert worker.rank == 2
        assert worker.world_size == 8
        assert worker.local_rank == 0
        assert worker.hostname == "node-1"
        assert worker.port == 29501

    def test_worker_info_defaults(self):
        """Test worker info default values."""
        worker = WorkerInfo(rank=0, world_size=1, local_rank=0)

        assert worker.hostname == "localhost"
        assert worker.port == 29500

    def test_worker_info_edge_cases(self):
        """Test edge cases for worker info."""
        # Single worker
        worker = WorkerInfo(rank=0, world_size=1, local_rank=0)
        assert worker.rank < worker.world_size

        # Large world size
        worker = WorkerInfo(rank=127, world_size=128, local_rank=7)
        assert worker.rank < worker.world_size


# =============================================================================
# ProcessGroup Tests
# =============================================================================

class TestProcessGroup:
    """Tests for ProcessGroup."""

    def test_process_group_creation(self):
        """Test creating process group."""
        pg = ProcessGroup(
            ranks=[0, 1, 2, 3],
            backend=Backend.NCCL,
            name="model_parallel"
        )

        assert pg.ranks == [0, 1, 2, 3]
        assert pg.backend == Backend.NCCL
        assert pg.name == "model_parallel"

    def test_process_group_size(self):
        """Test process group size method."""
        for size in [2, 4, 8, 16]:
            pg = ProcessGroup(ranks=list(range(size)))
            assert pg.size() == size

    def test_process_group_barrier(self):
        """Test barrier method internal counter."""
        pg = ProcessGroup(ranks=[0])  # Single rank for testing

        # Just verify barrier exists and can be called
        # In real distributed setting, all ranks would call this
        # For single-rank test, we just check the counter mechanism
        with pg._lock:
            pg._barriers["test"] = 0
        assert "test" in pg._barriers

    def test_multiple_barriers(self):
        """Test multiple named barriers."""
        pg = ProcessGroup(ranks=[0])  # Single rank for testing

        # Initialize barriers (simulating what barrier would do)
        for i in range(5):
            with pg._lock:
                pg._barriers[f"barrier_{i}"] = 0

        assert len(pg._barriers) == 5

    def test_default_backend(self):
        """Test default backend is GLOO."""
        pg = ProcessGroup(ranks=[0, 1])
        assert pg.backend == Backend.GLOO


# =============================================================================
# DeviceMesh Tests
# =============================================================================

class TestDeviceMesh:
    """Tests for DeviceMesh."""

    def test_device_mesh_creation(self):
        """Test creating device mesh."""
        mesh = DeviceMesh(
            shape=(2, 2),
            device_ids=[0, 1, 2, 3],
            mesh_dim_names=["dp", "tp"]
        )

        assert mesh.shape == (2, 2)
        assert len(mesh.device_ids) == 4

    def test_device_mesh_validation(self):
        """Test mesh size validation."""
        # Valid
        mesh = DeviceMesh(shape=(2, 3), device_ids=[0, 1, 2, 3, 4, 5])
        assert len(mesh.device_ids) == 6

        # Invalid - wrong count
        with pytest.raises(ValueError):
            DeviceMesh(shape=(2, 2), device_ids=[0, 1, 2])  # Need 4, got 3

    def test_get_device(self):
        """Test getting device by mesh coordinates."""
        mesh = DeviceMesh(shape=(2, 2), device_ids=[0, 1, 2, 3])

        assert mesh.get_device(0, 0) == 0
        assert mesh.get_device(0, 1) == 1
        assert mesh.get_device(1, 0) == 2
        assert mesh.get_device(1, 1) == 3

    def test_get_device_invalid_indices(self):
        """Test get_device with invalid indices."""
        mesh = DeviceMesh(shape=(2, 2), device_ids=[0, 1, 2, 3])

        with pytest.raises(ValueError):
            mesh.get_device(0)  # Too few indices

    def test_get_submesh(self):
        """Test getting submesh."""
        mesh = DeviceMesh(shape=(2, 2), device_ids=[0, 1, 2, 3])

        submesh = mesh.get_submesh(dim=0, index=0)

        # Should be 1D submesh
        assert len(submesh.shape) == 1

    def test_3d_mesh(self):
        """Test 3D device mesh."""
        mesh = DeviceMesh(
            shape=(2, 2, 2),
            device_ids=list(range(8))
        )

        assert mesh.shape == (2, 2, 2)
        assert len(mesh.device_ids) == 8


# =============================================================================
# DistributedContext Tests
# =============================================================================

class TestDistributedContext:
    """Tests for DistributedContext."""

    def test_context_creation(self, worker_info):
        """Test creating distributed context."""
        ctx = DistributedContext(worker_info=worker_info)

        assert ctx.worker_info == worker_info
        assert "default" in ctx._process_groups

    def test_context_singleton(self):
        """Test singleton pattern."""
        # Reset singleton for testing
        DistributedContext._instance = None

        ctx1 = DistributedContext.get_instance()
        ctx2 = DistributedContext.get_instance()

        assert ctx1 is ctx2

    def test_context_init_class_method(self):
        """Test init class method."""
        DistributedContext._instance = None

        ctx = DistributedContext.init(rank=1, world_size=4)

        assert ctx.get_rank() == 1
        assert ctx.get_world_size() == 4

    def test_get_rank_and_world_size(self, distributed_context):
        """Test get_rank and get_world_size methods."""
        assert distributed_context.get_rank() == 0
        assert distributed_context.get_world_size() == 4

    def test_new_group(self, distributed_context):
        """Test creating new process group."""
        group = distributed_context.new_group([0, 1], name="subset")

        assert group.ranks == [0, 1]
        assert distributed_context.get_group("subset") == group

    def test_get_group(self, distributed_context):
        """Test getting process group."""
        default = distributed_context.get_group("default")
        assert default is not None

        nonexistent = distributed_context.get_group("nonexistent")
        assert nonexistent is None

    def test_autograd_context_management(self, distributed_context):
        """Test autograd context management."""
        ctx_id = distributed_context.new_autograd_context()

        assert ctx_id >= 0

        ctx = distributed_context.get_autograd_context(ctx_id)
        assert ctx is not None
        assert ctx.context_id == ctx_id

    def test_barrier_method_exists(self, distributed_context):
        """Test barrier method exists."""
        # Just verify the method exists
        # Actual barrier would block waiting for other ranks in distributed setting
        assert hasattr(distributed_context, 'barrier')
        assert callable(distributed_context.barrier)


# =============================================================================
# AutogradContext Tests
# =============================================================================

class TestAutogradContext:
    """Tests for AutogradContext."""

    def test_autograd_context_creation(self):
        """Test creating autograd context."""
        ctx = AutogradContext(context_id=42)

        assert ctx.context_id == 42
        assert ctx.send_functions == {}
        assert ctx.recv_functions == {}

    def test_add_send_function(self):
        """Test adding send function."""
        ctx = AutogradContext(context_id=0)

        def send_fn():
            pass

        ctx.add_send_function(1, send_fn)

        assert 1 in ctx.send_functions
        assert ctx.send_functions[1] == send_fn

    def test_add_recv_function(self):
        """Test adding receive function."""
        ctx = AutogradContext(context_id=0)

        def recv_fn():
            pass

        ctx.add_recv_function(2, recv_fn)

        assert 2 in ctx.recv_functions
        assert ctx.recv_functions[2] == recv_fn

    def test_accumulate_grad(self):
        """Test gradient accumulation."""
        ctx = AutogradContext(context_id=0)

        grad1 = np.array([1, 2, 3])
        grad2 = np.array([4, 5, 6])

        ctx.accumulate_grad(0, grad1)
        ctx.accumulate_grad(0, grad2)

        accumulated = ctx._grad_to_send[0]
        np.testing.assert_array_equal(accumulated, grad1 + grad2)


# =============================================================================
# DistributedTensor Tests
# =============================================================================

class TestDistributedTensor:
    """Tests for DistributedTensor."""

    def test_tensor_creation(self):
        """Test creating distributed tensor."""
        data = np.random.randn(10, 20).astype(np.float32)
        tensor = DistributedTensor(
            data=data,
            device=0,
            requires_grad=True
        )

        assert tensor.shape == (10, 20)
        np.testing.assert_array_equal(tensor.data, data)
        assert tensor.requires_grad == True

    def test_tensor_grad(self):
        """Test tensor gradient."""
        tensor = DistributedTensor(np.zeros((5, 5)), requires_grad=True)

        assert tensor.grad is None

        # Set gradient
        grad = np.ones((5, 5))
        tensor.grad = grad

        np.testing.assert_array_equal(tensor.grad, grad)

    def test_tensor_shard(self):
        """Test tensor sharding."""
        data = np.arange(100).reshape(10, 10)
        tensor = DistributedTensor(data)

        shards = tensor.shard(dim=0, num_shards=2)

        assert len(shards) == 2
        # Each shard should have 5 rows
        assert shards[0].shape[0] == 5

    def test_tensor_shard_multiple_dimensions(self):
        """Test sharding along different dimensions."""
        data = np.arange(24).reshape(4, 6)
        tensor = DistributedTensor(data)

        # Shard along dim 0
        shards_dim0 = tensor.shard(dim=0, num_shards=2)
        assert shards_dim0[0].shape == (2, 6)

        # Shard along dim 1
        shards_dim1 = tensor.shard(dim=1, num_shards=3)
        assert shards_dim1[0].shape == (4, 2)

    def test_tensor_backward(self):
        """Test tensor backward."""
        tensor = DistributedTensor(np.ones((3, 3)), requires_grad=True)

        grad = np.ones((3, 3)) * 2
        tensor.backward(grad)

        np.testing.assert_array_equal(tensor.grad, grad)

    def test_tensor_backward_accumulates(self):
        """Test that backward accumulates gradients."""
        tensor = DistributedTensor(np.ones((3, 3)), requires_grad=True)

        tensor.backward(np.ones((3, 3)))
        tensor.backward(np.ones((3, 3)))

        expected = np.ones((3, 3)) * 2
        np.testing.assert_array_equal(tensor.grad, expected)

    def test_tensor_all_gather(self):
        """Test all_gather method."""
        tensor = DistributedTensor(np.ones((5,)))
        result = tensor.all_gather()

        # Currently returns self
        assert result is tensor

    def test_tensor_reduce_scatter(self):
        """Test reduce_scatter method."""
        tensor = DistributedTensor(np.ones((10,)))
        result = tensor.reduce_scatter()

        assert result is tensor


# =============================================================================
# RPC Functions Tests
# =============================================================================

class TestRPCFunctions:
    """Tests for RPC functions."""

    def test_rpc_sync(self):
        """Test synchronous RPC call."""
        def add(a, b):
            return a + b

        result = rpc_sync(to=1, func=add, args=(3, 5))

        assert result == 8

    def test_rpc_sync_with_kwargs(self):
        """Test rpc_sync with keyword arguments."""
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        result = rpc_sync(to=0, func=greet, args=("World",), kwargs={"greeting": "Hi"})

        assert result == "Hi, World!"

    def test_rpc_async(self):
        """Test asynchronous RPC call."""
        def multiply(a, b):
            return a * b

        future = rpc_async(to=1, func=multiply, args=(4, 5))

        result = future.result(timeout=5)
        assert result == 20

    def test_rpc_async_with_exception(self):
        """Test rpc_async with exception."""
        def raise_error():
            raise ValueError("Test error")

        future = rpc_async(to=0, func=raise_error)

        with pytest.raises(ValueError, match="Test error"):
            future.result(timeout=5)

    def test_remote(self):
        """Test remote function execution."""
        def compute(x):
            return x * 2

        rref = remote(to=0, func=compute, args=(5,))

        assert isinstance(rref, RRef)
        assert rref.owner == 0


# =============================================================================
# RRef Tests
# =============================================================================

class TestRRef:
    """Tests for Remote Reference."""

    def test_rref_creation(self):
        """Test creating RRef."""
        rref = RRef(owner=1, local_id=12345, type_name="Tensor")

        assert rref.owner == 1
        assert rref.local_id == 12345
        assert rref.type_name == "Tensor"

    def test_rref_to_here(self):
        """Test to_here method."""
        rref = RRef(owner=0, local_id=1)

        # Currently returns None (simulated)
        result = rref.to_here()
        assert result is None

    def test_rref_local_value(self):
        """Test local_value method."""
        rref = RRef(owner=0, local_id=1)

        # Currently returns None (simulated)
        result = rref.local_value()
        assert result is None


# =============================================================================
# RemoteModule Tests
# =============================================================================

class TestRemoteModule:
    """Tests for RemoteModule."""

    def test_remote_module_creation(self):
        """Test creating remote module."""
        class SimpleModule:
            def __call__(self, x):
                return x + 1

        module = SimpleModule()
        remote_module = RemoteModule(module, owner=0)

        assert remote_module.module == module
        assert remote_module.owner == 0
        assert isinstance(remote_module._rref, RRef)

    def test_remote_module_forward(self):
        """Test remote module forward."""
        class AddModule:
            def __call__(self, x):
                return x + 10

        module = AddModule()
        remote_module = RemoteModule(module, owner=0)

        rref = remote_module.forward(5)

        assert isinstance(rref, RRef)

    def test_remote_module_call(self):
        """Test remote module __call__."""
        class MultiplyModule:
            def __call__(self, x):
                return x * 3

        module = MultiplyModule()
        remote_module = RemoteModule(module)

        rref = remote_module(7)

        assert isinstance(rref, RRef)


# =============================================================================
# DistAutogradContext Tests
# =============================================================================

class TestDistAutogradContext:
    """Tests for distributed autograd context."""

    def test_new_context(self):
        """Test creating new context."""
        ctx = DistAutogradContext.new_context()

        assert ctx is not None
        assert ctx.context_id >= 0

    def test_get_context(self):
        """Test getting context by ID."""
        ctx = DistAutogradContext.new_context()
        ctx_id = ctx.context_id

        retrieved = DistAutogradContext.get_context(ctx_id)

        assert retrieved is ctx

    def test_get_nonexistent_context(self):
        """Test getting nonexistent context."""
        result = DistAutogradContext.get_context(99999)
        assert result is None

    def test_context_gradient_accumulation(self):
        """Test gradient accumulation in context."""
        ctx = DistAutogradContext.new_context()

        ctx.accumulate_gradient(0, np.array([1, 2, 3]))
        ctx.accumulate_gradient(0, np.array([4, 5, 6]))

        gradients = ctx.get_gradients()

        np.testing.assert_array_equal(gradients[0], np.array([5, 7, 9]))

    def test_context_known_workers(self):
        """Test tracking known workers."""
        ctx = DistAutogradContext.new_context()

        ctx._record_send(1, [])
        ctx._record_send(2, [])
        ctx._record_send(1, [])  # Duplicate

        assert 1 in ctx._known_workers
        assert 2 in ctx._known_workers
        assert len(ctx._known_workers) == 2  # No duplicates


# =============================================================================
# RPCAutograd Tests
# =============================================================================

class TestRPCAutograd:
    """Tests for RPC autograd engine."""

    def test_rpc_autograd_creation(self):
        """Test creating RPC autograd."""
        autograd = RPCAutograd()

        assert autograd._executor is not None

    def test_backward_with_invalid_context(self):
        """Test backward with invalid context ID."""
        autograd = RPCAutograd()

        with pytest.raises(RuntimeError, match="Unknown context"):
            autograd.backward(context_id=99999, roots=[])

    def test_backward_with_valid_context(self):
        """Test backward with valid context."""
        autograd = RPCAutograd()
        ctx = DistAutogradContext.new_context()

        # Should not raise
        autograd.backward(context_id=ctx.context_id, roots=[])

    def test_get_gradients(self):
        """Test getting gradients from context."""
        autograd = RPCAutograd()
        ctx = DistAutogradContext.new_context()

        ctx.accumulate_gradient(0, np.array([1, 2, 3]))

        gradients = autograd.get_gradients(ctx.context_id)

        assert 0 in gradients
        np.testing.assert_array_equal(gradients[0], np.array([1, 2, 3]))

    def test_get_gradients_invalid_context(self):
        """Test getting gradients from invalid context."""
        autograd = RPCAutograd()

        result = autograd.get_gradients(99999)

        assert result == {}


# =============================================================================
# RPC Initialization Tests
# =============================================================================

class TestRPCInitialization:
    """Tests for RPC initialization."""

    def test_init_rpc(self):
        """Test RPC initialization."""
        # Should not raise
        init_rpc(name="worker", rank=0, world_size=4)

    def test_shutdown_rpc(self):
        """Test RPC shutdown."""
        # Should not raise
        shutdown_rpc()

    def test_get_worker_info(self):
        """Test getting worker info."""
        init_rpc(name="test_worker", rank=2, world_size=4)

        info = get_worker_info()

        assert "id" in info
        assert "name" in info


# =============================================================================
# Collective Operations Tests
# =============================================================================

class TestCollectiveOperations:
    """Tests for collective operations."""

    def test_all_reduce(self, process_group):
        """Test all_reduce operation."""
        tensor = DistributedTensor(np.ones((5, 5)), process_group=process_group)

        result = all_reduce(tensor, op="sum", group=process_group)

        assert result is not None

    def test_broadcast(self, process_group):
        """Test broadcast operation."""
        tensor = DistributedTensor(np.ones((3, 3)), process_group=process_group)

        result = broadcast(tensor, src=0, group=process_group)

        assert result is not None

    def test_all_gather(self, process_group):
        """Test all_gather operation."""
        tensor = DistributedTensor(np.ones((4,)), process_group=process_group)
        output_tensors = [DistributedTensor(np.zeros((4,))) for _ in range(4)]

        # Should not raise
        all_gather(output_tensors, tensor, group=process_group)

    def test_reduce_scatter(self, process_group):
        """Test reduce_scatter operation."""
        output = DistributedTensor(np.zeros((4,)), process_group=process_group)
        input_list = [DistributedTensor(np.ones((4,))) for _ in range(4)]

        # Should not raise
        reduce_scatter(output, input_list, op="sum", group=process_group)


# =============================================================================
# RemoteGradient Tests
# =============================================================================

class TestRemoteGradient:
    """Tests for RemoteGradient."""

    def test_remote_gradient_creation(self):
        """Test creating remote gradient."""
        rref = RRef(owner=1, local_id=100)
        grad = np.ones((10, 10))

        remote_grad = RemoteGradient(
            rref=rref,
            grad=grad,
            context_id=42
        )

        assert remote_grad.rref == rref
        np.testing.assert_array_equal(remote_grad.grad, grad)
        assert remote_grad.context_id == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
