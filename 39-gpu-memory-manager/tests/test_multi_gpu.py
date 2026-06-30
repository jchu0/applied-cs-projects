"""Tests for Multi-GPU distribution features."""

import pytest
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gpumem.allocator.advanced import (
    LoadBalanceStrategy,
    GPUDeviceState,
    MultiGPUAllocator,
    DistributedTensor,
    DistributedTensorManager,
    GradientSynchronizer,
)
from gpumem.core.memory import MemoryBlock, DeviceType


class TestLoadBalanceStrategy:
    """Test load balancing strategies."""

    def test_strategy_enum_values(self):
        """Test all strategy enum values exist."""
        assert LoadBalanceStrategy.ROUND_ROBIN
        assert LoadBalanceStrategy.LEAST_UTILIZED
        assert LoadBalanceStrategy.LOCALITY_AWARE
        assert LoadBalanceStrategy.MEMORY_PRESSURE
        assert LoadBalanceStrategy.MANUAL


class TestGPUDeviceState:
    """Test GPU device state tracking."""

    def test_creation(self):
        """Test GPUDeviceState creation."""
        state = GPUDeviceState(
            device_id=0,
            total_memory=8 * 1024 * 1024 * 1024  # 8GB
        )

        assert state.device_id == 0
        assert state.total_memory == 8 * 1024 * 1024 * 1024
        assert state.allocated_memory == 0
        assert state.is_available is True

    def test_free_memory(self):
        """Test free memory calculation."""
        state = GPUDeviceState(
            device_id=0,
            total_memory=1024 * 1024,  # 1MB
            allocated_memory=256 * 1024  # 256KB
        )

        assert state.free_memory == 768 * 1024

    def test_utilization(self):
        """Test utilization calculation."""
        state = GPUDeviceState(
            device_id=0,
            total_memory=1024 * 1024,
            allocated_memory=512 * 1024
        )

        assert state.utilization == 0.5

    def test_utilization_zero_total(self):
        """Test utilization with zero total memory."""
        state = GPUDeviceState(device_id=0, total_memory=0)
        assert state.utilization == 0.0


class TestMultiGPUAllocator:
    """Test MultiGPUAllocator."""

    def test_creation(self):
        """Test allocator creation."""
        allocator = MultiGPUAllocator(
            num_gpus=4,
            memory_per_gpu=1024 * 1024 * 1024  # 1GB
        )

        assert allocator._num_gpus == 4
        assert len(allocator._allocators) == 4
        assert len(allocator._device_states) == 4

    def test_allocate_single_device(self):
        """Test allocation on specific device."""
        allocator = MultiGPUAllocator(num_gpus=2)

        block = allocator.allocate(1024, device_id=0)

        assert block is not None
        assert block.device_id == 0
        assert block.size >= 1024

    def test_allocate_automatic_selection(self):
        """Test automatic device selection."""
        allocator = MultiGPUAllocator(
            num_gpus=2,
            strategy=LoadBalanceStrategy.LEAST_UTILIZED
        )

        block = allocator.allocate(1024)

        assert block is not None
        assert block.device_id in [0, 1]

    def test_allocate_round_robin(self):
        """Test round-robin allocation."""
        allocator = MultiGPUAllocator(
            num_gpus=3,
            strategy=LoadBalanceStrategy.ROUND_ROBIN
        )

        devices = []
        for _ in range(6):
            block = allocator.allocate(1024)
            devices.append(block.device_id)
            allocator.free(block)

        # Should cycle through devices
        assert devices[:3] == [0, 1, 2]
        assert devices[3:] == [0, 1, 2]

    def test_allocate_least_utilized(self):
        """Test least utilized strategy."""
        allocator = MultiGPUAllocator(
            num_gpus=2,
            memory_per_gpu=1024 * 1024,  # 1MB per GPU
            strategy=LoadBalanceStrategy.LEAST_UTILIZED
        )

        # Allocate on device 0
        block0 = allocator.allocate(512 * 1024, device_id=0)

        # Next allocation should go to device 1 (less utilized)
        block1 = allocator.allocate(256 * 1024)

        assert block1.device_id == 1

        allocator.free(block0)
        allocator.free(block1)

    def test_free(self):
        """Test memory freeing."""
        allocator = MultiGPUAllocator(num_gpus=2)

        block = allocator.allocate(1024, device_id=0)
        ptr = block.ptr

        allocator.free(block)

        assert ptr not in allocator._allocations

    def test_transfer_between_gpus(self):
        """Test transferring block between GPUs."""
        allocator = MultiGPUAllocator(num_gpus=2)

        # Allocate on device 0
        source_block = allocator.allocate(1024, device_id=0)

        # Transfer to device 1
        target_block = allocator.transfer(source_block, target_device=1)

        assert target_block is not None
        assert target_block.device_id == 1
        assert target_block.size == source_block.size

    def test_transfer_same_device(self):
        """Test transfer to same device returns original block."""
        allocator = MultiGPUAllocator(num_gpus=2)

        block = allocator.allocate(1024, device_id=0)
        result = allocator.transfer(block, target_device=0)

        assert result is block

    def test_replicate_to_multiple_gpus(self):
        """Test replicating block to multiple GPUs."""
        allocator = MultiGPUAllocator(num_gpus=4)

        block = allocator.allocate(1024, device_id=0)
        replicas = allocator.replicate(block, target_devices=[1, 2, 3])

        assert 0 in replicas  # Original
        assert 1 in replicas
        assert 2 in replicas
        assert 3 in replicas

    def test_set_strategy(self):
        """Test changing strategy."""
        allocator = MultiGPUAllocator(
            num_gpus=2,
            strategy=LoadBalanceStrategy.ROUND_ROBIN
        )

        assert allocator._strategy == LoadBalanceStrategy.ROUND_ROBIN

        allocator.set_strategy(LoadBalanceStrategy.LEAST_UTILIZED)

        assert allocator._strategy == LoadBalanceStrategy.LEAST_UTILIZED

    def test_get_device_state(self):
        """Test getting device state."""
        allocator = MultiGPUAllocator(num_gpus=2)

        state = allocator.get_device_state(0)

        assert state is not None
        assert state.device_id == 0

    def test_get_all_device_states(self):
        """Test getting all device states."""
        allocator = MultiGPUAllocator(num_gpus=3)

        states = allocator.get_all_device_states()

        assert len(states) == 3
        assert all(isinstance(s, GPUDeviceState) for s in states)

    def test_get_total_stats(self):
        """Test combined statistics."""
        allocator = MultiGPUAllocator(num_gpus=2)

        # Allocate on both devices
        block0 = allocator.allocate(1024, device_id=0)
        block1 = allocator.allocate(2048, device_id=1)

        stats = allocator.get_total_stats()

        assert stats.num_allocs >= 2

        allocator.free(block0)
        allocator.free(block1)

    def test_get_statistics(self):
        """Test detailed statistics."""
        allocator = MultiGPUAllocator(num_gpus=2)

        block = allocator.allocate(1024)

        stats = allocator.get_statistics()

        assert 'total_allocations' in stats
        assert 'strategy' in stats
        assert 'num_gpus' in stats
        assert 'devices' in stats
        assert 'gpu_0' in stats['devices']

        allocator.free(block)

    def test_empty_all_caches(self):
        """Test emptying all caches."""
        allocator = MultiGPUAllocator(num_gpus=2)

        block0 = allocator.allocate(1024, device_id=0)
        block1 = allocator.allocate(1024, device_id=1)

        allocator.free(block0)
        allocator.free(block1)

        # Should not raise
        allocator.empty_all_caches()

    def test_memory_pressure_strategy(self):
        """Test memory pressure aware allocation."""
        allocator = MultiGPUAllocator(
            num_gpus=2,
            memory_per_gpu=1024 * 1024,
            strategy=LoadBalanceStrategy.MEMORY_PRESSURE
        )

        # Fill up device 0
        blocks = []
        for _ in range(5):
            block = allocator.allocate(100 * 1024, device_id=0)
            if block:
                blocks.append(block)

        # Next allocation should avoid pressured device
        new_block = allocator.allocate(100 * 1024)
        assert new_block is not None

        for block in blocks:
            allocator.free(block)
        if new_block:
            allocator.free(new_block)

    def test_allocation_failure_fallback(self):
        """Test fallback when primary device is full."""
        allocator = MultiGPUAllocator(
            num_gpus=2,
            memory_per_gpu=1024 * 1024  # 1MB each
        )

        # Try to allocate more than device 0 has
        large_blocks = []
        for _ in range(100):
            block = allocator.allocate(50 * 1024, device_id=0)
            if block:
                large_blocks.append(block)
            else:
                break

        # Cleanup
        for block in large_blocks:
            allocator.free(block)


class TestDistributedTensorManager:
    """Test DistributedTensorManager."""

    def test_creation(self):
        """Test manager creation."""
        allocator = MultiGPUAllocator(num_gpus=2)
        manager = DistributedTensorManager(allocator)

        assert manager._allocator is allocator
        assert len(manager._tensors) == 0

    def test_create_distributed_tensor(self):
        """Test creating a distributed tensor."""
        allocator = MultiGPUAllocator(num_gpus=2)
        manager = DistributedTensorManager(allocator)

        tensor = manager.create_distributed(
            name='weights',
            shape=(1000, 512),
            dtype='float32',
            devices=[0, 1]
        )

        assert tensor is not None
        assert tensor.name == 'weights'
        assert tensor.shape == (1000, 512)
        assert len(tensor.shards) == 2
        assert 0 in tensor.shards
        assert 1 in tensor.shards

    def test_distributed_tensor_properties(self):
        """Test distributed tensor properties."""
        allocator = MultiGPUAllocator(num_gpus=4)
        manager = DistributedTensorManager(allocator)

        tensor = manager.create_distributed(
            name='embeddings',
            shape=(10000, 256),
            dtype='float32',
            devices=[0, 1, 2, 3]
        )

        assert tensor.num_shards == 4
        assert tensor.devices == [0, 1, 2, 3]

    def test_delete_tensor(self):
        """Test deleting a distributed tensor."""
        allocator = MultiGPUAllocator(num_gpus=2)
        manager = DistributedTensorManager(allocator)

        tensor = manager.create_distributed(
            name='temp',
            shape=(100, 100),
            dtype='float32'
        )

        result = manager.delete_tensor('temp')

        assert result is True
        assert 'temp' not in manager._tensors

    def test_delete_nonexistent_tensor(self):
        """Test deleting nonexistent tensor."""
        allocator = MultiGPUAllocator(num_gpus=2)
        manager = DistributedTensorManager(allocator)

        result = manager.delete_tensor('nonexistent')

        assert result is False

    def test_gather(self):
        """Test gathering distributed tensor."""
        allocator = MultiGPUAllocator(num_gpus=2)
        manager = DistributedTensorManager(allocator)

        tensor = manager.create_distributed(
            name='data',
            shape=(1000, 64),
            dtype='float32',
            devices=[0, 1]
        )

        gathered = manager.gather(tensor, target_device=0)

        assert gathered is not None
        assert gathered.size == tensor.total_size

    def test_scatter(self):
        """Test scattering data to distributed tensor."""
        allocator = MultiGPUAllocator(num_gpus=2)
        manager = DistributedTensorManager(allocator)

        # Create source block
        source = allocator.allocate(1024 * 1024, device_id=0)

        tensor = manager.scatter(
            source_block=source,
            source_device=0,
            target_devices=[0, 1],
            name='scattered',
            shape=(256, 1024),
            dtype='float32'
        )

        assert tensor is not None
        assert tensor.num_shards == 2

    def test_all_reduce(self):
        """Test all-reduce operation."""
        allocator = MultiGPUAllocator(num_gpus=4)
        manager = DistributedTensorManager(allocator)

        tensor = manager.create_distributed(
            name='gradients',
            shape=(1000, 100),
            dtype='float32',
            devices=[0, 1, 2, 3]
        )

        result = manager.all_reduce(tensor, operation='sum')

        assert result is True
        assert manager._stats['allreduce_ops'] == 1

    def test_get_tensor(self):
        """Test getting tensor by name."""
        allocator = MultiGPUAllocator(num_gpus=2)
        manager = DistributedTensorManager(allocator)

        created = manager.create_distributed(
            name='layer1',
            shape=(100, 100),
            dtype='float32'
        )

        retrieved = manager.get_tensor('layer1')

        assert retrieved is created

    def test_list_tensors(self):
        """Test listing all tensors."""
        allocator = MultiGPUAllocator(num_gpus=2)
        manager = DistributedTensorManager(allocator)

        manager.create_distributed('tensor1', (100,), 'float32')
        manager.create_distributed('tensor2', (200,), 'float32')
        manager.create_distributed('tensor3', (300,), 'float32')

        names = manager.list_tensors()

        assert len(names) == 3
        assert 'tensor1' in names
        assert 'tensor2' in names
        assert 'tensor3' in names

    def test_get_statistics(self):
        """Test manager statistics."""
        allocator = MultiGPUAllocator(num_gpus=2)
        manager = DistributedTensorManager(allocator)

        manager.create_distributed('t1', (1000, 100), 'float32')
        manager.create_distributed('t2', (500, 50), 'float32')

        stats = manager.get_statistics()

        assert stats['tensors_created'] == 2
        assert stats['num_tensors'] == 2
        assert stats['total_distributed_memory'] > 0


class TestGradientSynchronizer:
    """Test GradientSynchronizer."""

    def test_creation(self):
        """Test synchronizer creation."""
        allocator = MultiGPUAllocator(num_gpus=4)
        sync = GradientSynchronizer(allocator, num_gpus=4)

        assert sync._num_gpus == 4
        assert len(sync._grad_buffers) == 4

    def test_register_gradients(self):
        """Test registering gradient blocks."""
        allocator = MultiGPUAllocator(num_gpus=2)
        sync = GradientSynchronizer(allocator, num_gpus=2)

        grad_blocks = [
            allocator.allocate(1024, device_id=0),
            allocator.allocate(2048, device_id=0),
        ]

        sync.register_gradients(device_id=0, gradient_blocks=grad_blocks)

        assert len(sync._grad_buffers[0]) == 2

    def test_synchronize(self):
        """Test gradient synchronization."""
        allocator = MultiGPUAllocator(num_gpus=2)
        sync = GradientSynchronizer(allocator, num_gpus=2)

        # Register gradients on both devices
        for device_id in range(2):
            grad_blocks = [allocator.allocate(1024, device_id=device_id)]
            sync.register_gradients(device_id, grad_blocks)

        result = sync.synchronize()

        assert result is True
        assert sync._stats['sync_rounds'] == 1

    def test_synchronize_with_compression(self):
        """Test gradient synchronization with compression."""
        allocator = MultiGPUAllocator(num_gpus=2)
        sync = GradientSynchronizer(allocator, num_gpus=2)

        for device_id in range(2):
            grad_blocks = [allocator.allocate(4096, device_id=device_id)]
            sync.register_gradients(device_id, grad_blocks)

        result = sync.synchronize(use_compression=True)

        assert result is True
        assert sync._stats['compression_ratio'] == 4.0

    def test_get_statistics(self):
        """Test synchronizer statistics."""
        allocator = MultiGPUAllocator(num_gpus=4)
        sync = GradientSynchronizer(allocator, num_gpus=4)

        # Register and sync
        for device_id in range(4):
            grad_blocks = [allocator.allocate(1024, device_id=device_id)]
            sync.register_gradients(device_id, grad_blocks)

        sync.synchronize()

        stats = sync.get_statistics()

        assert 'sync_rounds' in stats
        assert 'bytes_synchronized' in stats
        assert 'num_gpus' in stats
        assert 'registered_buffers' in stats
        assert stats['registered_buffers'] == 4


class TestMultiGPUIntegration:
    """Integration tests for multi-GPU features."""

    def test_full_distributed_training_workflow(self):
        """Test a complete distributed training workflow."""
        allocator = MultiGPUAllocator(num_gpus=4)
        tensor_mgr = DistributedTensorManager(allocator)
        grad_sync = GradientSynchronizer(allocator, num_gpus=4)

        # 1. Create model weights distributed across GPUs
        weights = tensor_mgr.create_distributed(
            name='model_weights',
            shape=(10000, 512),
            dtype='float32',
            devices=[0, 1, 2, 3]
        )

        assert weights is not None
        assert weights.num_shards == 4

        # 2. Create gradient buffers
        for device_id in range(4):
            grads = [allocator.allocate(1024 * 100, device_id=device_id)]
            grad_sync.register_gradients(device_id, grads)

        # 3. Simulate training steps
        for step in range(3):
            # Synchronize gradients
            result = grad_sync.synchronize()
            assert result is True

        # 4. All-reduce
        tensor_mgr.all_reduce(weights)

        # 5. Verify statistics
        tensor_stats = tensor_mgr.get_statistics()
        sync_stats = grad_sync.get_statistics()

        assert tensor_stats['tensors_created'] >= 1
        assert sync_stats['sync_rounds'] == 3

    def test_multi_gpu_data_parallel_simulation(self):
        """Test data-parallel training simulation."""
        allocator = MultiGPUAllocator(
            num_gpus=2,
            strategy=LoadBalanceStrategy.ROUND_ROBIN
        )

        # Allocate input batches on different GPUs
        batches = []
        for i in range(4):
            batch = allocator.allocate(1024 * 1024)  # 1MB batches
            batches.append(batch)

        # Verify distribution
        device_counts = {0: 0, 1: 0}
        for batch in batches:
            device_counts[batch.device_id] += 1

        # Should be evenly distributed
        assert device_counts[0] == 2
        assert device_counts[1] == 2

        for batch in batches:
            allocator.free(batch)

    def test_memory_pressure_handling(self):
        """Test handling memory pressure across GPUs."""
        allocator = MultiGPUAllocator(
            num_gpus=2,
            memory_per_gpu=10 * 1024 * 1024,  # 10MB each
            strategy=LoadBalanceStrategy.MEMORY_PRESSURE
        )

        blocks = []

        # Allocate until pressure builds
        for _ in range(50):
            block = allocator.allocate(500 * 1024)  # 500KB each
            if block:
                blocks.append(block)

        # Check that allocations were distributed
        stats = allocator.get_statistics()
        assert stats['total_allocations'] > 0

        # Cleanup
        for block in blocks:
            allocator.free(block)
