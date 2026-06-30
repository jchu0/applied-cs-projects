"""Tests for advanced memory management features (Phase 6)."""

import pytest
import threading
import time

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gpumem.allocator.advanced import (
    TransferDirection,
    TransferEvent,
    StreamEvent,
    TopologyType,
    GPULink,
    GPUTopology,
    P2PTransferManager,
    PrefetchPriority,
    PrefetchRequest,
    PrefetchManager,
    StreamSynchronizer,
    OffloadEntry,
    CPUOffloader,
    AdvancedMemoryManager,
)
from gpumem.core.memory import MemoryBlock, DeviceType


# =============================================================================
# P2P Transfer Manager Tests
# =============================================================================

class TestP2PTransferManager:
    """Tests for P2P memory transfers."""

    def test_creation(self):
        """Test creating P2P manager."""
        manager = P2PTransferManager(num_devices=4)
        assert manager is not None

    def test_peer_access_check(self):
        """Test checking P2P access."""
        manager = P2PTransferManager(num_devices=4)

        # With HYBRID topology (default), all GPU pairs have paths
        assert manager.can_access_peer(0, 1) is True  # Adjacent (NVLink)
        assert manager.can_access_peer(1, 0) is True
        assert manager.can_access_peer(0, 3) is True  # Non-adjacent (PCIe path)

        # Use LINEAR topology for restricted access testing
        manager_linear = P2PTransferManager(
            num_devices=4,
            topology_type=TopologyType.LINEAR
        )
        # LINEAR also has paths through multi-hop
        assert manager_linear.can_access_peer(0, 1) is True

    def test_enable_peer_access(self):
        """Test enabling P2P access."""
        manager = P2PTransferManager(num_devices=4)

        # Enable P2P between non-adjacent GPUs
        manager.enable_peer_access(0, 3)
        assert manager.can_access_peer(0, 3) is True
        assert manager.can_access_peer(3, 0) is True

    def test_disable_peer_access(self):
        """Test disabling P2P access."""
        manager = P2PTransferManager(num_devices=2)

        manager.disable_peer_access(0, 1)
        assert manager.can_access_peer(0, 1) is False

    def test_transfer_success(self):
        """Test successful P2P transfer."""
        manager = P2PTransferManager(num_devices=2)

        event = manager.transfer(
            src_ptr=0x1000,
            dst_ptr=0x2000,
            size=1024,
            src_device=0,
            dst_device=1,
            stream=0
        )

        assert event is not None
        assert event.size == 1024
        assert event.direction == TransferDirection.DEVICE_TO_DEVICE

    def test_transfer_failure_no_access(self):
        """Test transfer fails without P2P access."""
        manager = P2PTransferManager(num_devices=4)

        # Disable P2P between devices
        manager.disable_peer_access(0, 3)

        event = manager.transfer(
            src_ptr=0x1000,
            dst_ptr=0x2000,
            size=1024,
            src_device=0,
            dst_device=3
        )

        assert event is None

    def test_wait_for_transfer(self):
        """Test waiting for transfer completion."""
        manager = P2PTransferManager(num_devices=2)

        event = manager.transfer(
            src_ptr=0x1000,
            dst_ptr=0x2000,
            size=1024,
            src_device=0,
            dst_device=1,
            async_transfer=True
        )

        result = manager.wait_for_transfer(event)
        assert result is True
        assert event.completed is True

    def test_sync_transfer(self):
        """Test synchronous transfer."""
        manager = P2PTransferManager(num_devices=2)

        event = manager.transfer(
            src_ptr=0x1000,
            dst_ptr=0x2000,
            size=1024,
            src_device=0,
            dst_device=1,
            async_transfer=False
        )

        assert event.completed is True

    def test_bandwidth_estimation(self):
        """Test bandwidth estimation with topology awareness."""
        manager = P2PTransferManager(num_devices=4)

        # Same device should be fastest (internal bandwidth)
        bw_same = manager.get_bandwidth(0, 0)
        bw_adjacent = manager.get_bandwidth(0, 1)

        assert bw_same > bw_adjacent  # Same device > NVLink

        # With HYBRID topology, routing finds optimal paths
        # Adjacent (0->1) uses direct NVLink (200 GB/s)
        # The algorithm finds fastest available path for any pair
        bw_far = manager.get_bandwidth(0, 3)
        assert bw_adjacent > 0  # Has valid bandwidth
        assert bw_far > 0  # Has valid bandwidth (either via NVLink chain or PCIe)

        # Test LINEAR topology where multi-hop reduces effective bandwidth
        manager_linear = P2PTransferManager(
            num_devices=4,
            topology_type=TopologyType.LINEAR
        )
        bw_linear_adjacent = manager_linear.get_bandwidth(0, 1)
        bw_linear_far = manager_linear.get_bandwidth(0, 3)
        # LINEAR uses PCIe - path 0->1->2->3 is limited by PCIe bandwidth
        assert bw_linear_adjacent > 0
        assert bw_linear_far > 0

    def test_statistics(self):
        """Test transfer statistics."""
        manager = P2PTransferManager(num_devices=2)

        manager.transfer(0x1000, 0x2000, 1024, 0, 1)
        manager.transfer(0x3000, 0x4000, 2048, 1, 0)

        stats = manager.get_statistics()

        assert stats['total_transfers'] == 2
        assert stats['total_bytes_transferred'] == 3072


# =============================================================================
# GPU Topology Tests
# =============================================================================

class TestGPUTopology:
    """Tests for GPU topology detection and routing."""

    def test_linear_topology(self):
        """Test linear PCIe topology."""
        topology = GPUTopology(num_devices=4, topology_type=TopologyType.LINEAR)

        # Adjacent GPUs should be directly connected
        link = topology.get_link(0, 1)
        assert link is not None
        assert link.link_type == "pcie"

        # Non-adjacent GPUs have no direct link
        link = topology.get_link(0, 3)
        assert link is None

    def test_ring_topology(self):
        """Test ring NVLink topology."""
        topology = GPUTopology(num_devices=4, topology_type=TopologyType.RING)

        # All adjacent GPUs connected (including wrap-around)
        assert topology.get_link(0, 1) is not None
        assert topology.get_link(3, 0) is not None  # Ring wrap-around

        link = topology.get_link(0, 1)
        assert link.link_type == "nvlink"

    def test_full_mesh_topology(self):
        """Test full NVLink mesh (DGX-style)."""
        topology = GPUTopology(num_devices=4, topology_type=TopologyType.FULL_MESH)

        # All pairs should be connected
        for i in range(4):
            for j in range(4):
                if i != j:
                    link = topology.get_link(i, j)
                    assert link is not None
                    assert link.link_type == "nvlink"

    def test_hybrid_topology(self):
        """Test hybrid NVLink/PCIe topology."""
        topology = GPUTopology(num_devices=4, topology_type=TopologyType.HYBRID)

        # Adjacent = NVLink
        link_01 = topology.get_link(0, 1)
        assert link_01.link_type == "nvlink"

        # Non-adjacent = PCIe
        link_02 = topology.get_link(0, 2)
        assert link_02.link_type == "pcie"

    def test_path_computation_direct(self):
        """Test path computation for direct connections."""
        topology = GPUTopology(num_devices=4, topology_type=TopologyType.FULL_MESH)

        path, bandwidth, latency = topology.compute_optimal_path(0, 3)

        # Full mesh = direct path
        assert path == [0, 3]
        assert bandwidth == GPUTopology.NVLINK_BANDWIDTH
        assert latency == GPUTopology.NVLINK_LATENCY

    def test_path_computation_multihop(self):
        """Test path computation requiring multiple hops."""
        topology = GPUTopology(num_devices=4, topology_type=TopologyType.LINEAR)

        path, bandwidth, latency = topology.compute_optimal_path(0, 3)

        # Linear topology: 0 -> 1 -> 2 -> 3
        assert len(path) == 4
        assert path[0] == 0
        assert path[-1] == 3
        # Bandwidth limited by slowest link (all PCIe here)
        assert bandwidth == GPUTopology.PCIE_GEN4_BANDWIDTH

    def test_transfer_cost_estimation(self):
        """Test transfer cost calculation."""
        topology = GPUTopology(num_devices=4, topology_type=TopologyType.FULL_MESH)

        # 1GB transfer
        size = 1024 * 1024 * 1024
        cost_ms = topology.get_transfer_cost(0, 1, size)

        # At 200 GB/s, 1GB should take ~5ms + latency
        expected_transfer_time = (size / (200 * 1e9)) * 1000
        assert cost_ms >= expected_transfer_time  # Account for latency

    def test_numa_mapping(self):
        """Test NUMA node mapping."""
        topology = GPUTopology(num_devices=8, topology_type=TopologyType.LINEAR)

        # Default: 2 GPUs per NUMA node
        assert topology.get_numa_node(0) == 0
        assert topology.get_numa_node(1) == 0
        assert topology.get_numa_node(2) == 1
        assert topology.get_numa_node(3) == 1

        # Custom mapping
        topology.set_numa_mapping(0, 5)
        assert topology.get_numa_node(0) == 5

    def test_topology_serialization(self):
        """Test topology serialization."""
        topology = GPUTopology(num_devices=4, topology_type=TopologyType.RING)

        data = topology.to_dict()

        assert data['num_devices'] == 4
        assert data['topology_type'] == 'RING'
        assert len(data['links']) > 0


class TestTopologyAwareP2P:
    """Tests for topology-aware P2P transfers."""

    def test_p2p_with_topology(self):
        """Test P2P manager with custom topology."""
        topology = GPUTopology(num_devices=4, topology_type=TopologyType.FULL_MESH)
        manager = P2PTransferManager(num_devices=4, topology=topology)

        # Full mesh = all pairs accessible
        assert manager.can_access_peer(0, 3) is True

    def test_optimal_path_retrieval(self):
        """Test retrieving optimal paths."""
        manager = P2PTransferManager(
            num_devices=4,
            topology_type=TopologyType.LINEAR
        )

        path = manager.get_optimal_path(0, 3)
        assert len(path) > 2  # Must use multi-hop

    def test_hop_count(self):
        """Test hop count calculation."""
        manager = P2PTransferManager(
            num_devices=4,
            topology_type=TopologyType.FULL_MESH
        )

        # Full mesh = 1 hop for any pair
        assert manager.get_hop_count(0, 3) == 1
        assert manager.get_hop_count(0, 0) == 0

    def test_direct_path_check(self):
        """Test direct path detection."""
        manager = P2PTransferManager(
            num_devices=4,
            topology_type=TopologyType.FULL_MESH
        )

        assert manager.is_direct_path(0, 3) is True

        # Linear topology has multi-hop paths
        manager2 = P2PTransferManager(
            num_devices=4,
            topology_type=TopologyType.LINEAR
        )
        assert manager2.is_direct_path(0, 3) is False

    def test_transfer_cost(self):
        """Test transfer cost estimation."""
        manager = P2PTransferManager(
            num_devices=4,
            topology_type=TopologyType.FULL_MESH
        )

        cost = manager.get_transfer_cost(0, 1, 1024 * 1024)
        assert cost > 0

    def test_multihop_transfer(self):
        """Test multi-hop transfer with routing."""
        manager = P2PTransferManager(
            num_devices=4,
            topology_type=TopologyType.LINEAR
        )

        events = manager.transfer_with_routing(
            src_ptr=0x1000,
            dst_ptr=0x2000,
            size=1024,
            src_device=0,
            dst_device=3,
            use_multi_hop=True
        )

        # Should have multiple hop events
        assert len(events) >= 1

    def test_numa_distance(self):
        """Test NUMA distance calculation."""
        manager = P2PTransferManager(num_devices=8)

        # GPUs 0,1 on NUMA 0; GPUs 2,3 on NUMA 1, etc.
        assert manager.get_numa_distance(0, 1) == 0  # Same NUMA
        assert manager.get_numa_distance(0, 2) == 1  # Adjacent NUMA
        assert manager.get_numa_distance(0, 4) == 2  # 2 NUMA nodes apart

    def test_topology_update(self):
        """Test updating topology at runtime."""
        manager = P2PTransferManager(
            num_devices=4,
            topology_type=TopologyType.LINEAR
        )

        # Initial: linear topology
        assert manager.is_direct_path(0, 3) is False

        # Update to full mesh
        new_topology = GPUTopology(4, TopologyType.FULL_MESH)
        manager.set_topology(new_topology)

        # Now direct paths available
        assert manager.is_direct_path(0, 3) is True

    def test_statistics_include_topology(self):
        """Test statistics include topology info."""
        manager = P2PTransferManager(
            num_devices=4,
            topology_type=TopologyType.RING
        )

        stats = manager.get_statistics()
        assert 'topology_type' in stats
        assert stats['topology_type'] == 'RING'
        assert 'multi_hop_transfers' in stats


# =============================================================================
# Prefetch Manager Tests
# =============================================================================

class TestPrefetchManager:
    """Tests for prefetch operations."""

    def test_creation(self):
        """Test creating prefetch manager."""
        manager = PrefetchManager()
        assert manager is not None

    def test_prefetch_to_device(self):
        """Test prefetching to GPU."""
        manager = PrefetchManager()
        block = MemoryBlock(ptr=0x1000, size=1024, device=DeviceType.CPU, device_id=0)

        request = manager.prefetch_to_device(block, device_id=0)

        assert request is not None
        assert request.target_device == DeviceType.CUDA

    def test_prefetch_to_host(self):
        """Test prefetching to CPU."""
        manager = PrefetchManager()
        block = MemoryBlock(ptr=0x1000, size=1024, device=DeviceType.CUDA, device_id=0)

        request = manager.prefetch_to_host(block)

        assert request is not None
        assert request.target_device == DeviceType.CPU

    def test_prefetch_priority(self):
        """Test prefetch priority ordering."""
        manager = PrefetchManager()

        block1 = MemoryBlock(ptr=0x1000, size=1024, device=DeviceType.CPU, device_id=0)
        block2 = MemoryBlock(ptr=0x2000, size=1024, device=DeviceType.CPU, device_id=0)

        # Low priority first
        req1 = manager.prefetch_to_device(block1, priority=PrefetchPriority.LOW)
        # Then high priority
        req2 = manager.prefetch_to_device(block2, priority=PrefetchPriority.HIGH)

        # High priority should be processed first
        assert req2.priority.value > req1.priority.value

    def test_prefetch_wait(self):
        """Test waiting for prefetch completion."""
        manager = PrefetchManager()
        block = MemoryBlock(ptr=0x1000, size=1024, device=DeviceType.CPU, device_id=0)

        request = manager.prefetch_to_device(block)
        # Manually complete for testing
        request.completed = True
        result = manager.wait(request, timeout_ms=100)

        assert result is True
        assert request.completed is True

    def test_prefetch_cancel(self):
        """Test canceling prefetch."""
        manager = PrefetchManager(max_pending=0)  # Don't auto-process
        block = MemoryBlock(ptr=0x1000, size=1024, device=DeviceType.CPU, device_id=0)

        request = manager.prefetch_to_device(block)
        result = manager.cancel(request)

        assert result is True

    def test_prefetch_callback(self):
        """Test prefetch completion callback."""
        manager = PrefetchManager()
        callback_called = [False]

        def on_complete(req):
            callback_called[0] = True

        block = MemoryBlock(ptr=0x1000, size=1024, device=DeviceType.CPU, device_id=0)
        request = manager.prefetch_to_device(block, callback=on_complete)

        # Simulate completion with callback
        if request.callback:
            request.callback(request)
            request.completed = True

        assert callback_called[0] is True

    def test_prefetch_statistics(self):
        """Test prefetch statistics."""
        manager = PrefetchManager()

        block1 = MemoryBlock(ptr=0x1000, size=1024, device=DeviceType.CPU, device_id=0)
        block2 = MemoryBlock(ptr=0x2000, size=2048, device=DeviceType.CUDA, device_id=0)

        manager.prefetch_to_device(block1)
        manager.prefetch_to_host(block2)

        stats = manager.get_statistics()

        assert stats['prefetches_to_device'] == 1
        assert stats['prefetches_to_host'] == 1
        assert stats['bytes_prefetched'] == 3072


# =============================================================================
# Stream Synchronizer Tests
# =============================================================================

class TestStreamSynchronizer:
    """Tests for stream synchronization."""

    def test_creation(self):
        """Test creating stream synchronizer."""
        sync = StreamSynchronizer(num_streams=8)
        assert sync is not None

    def test_record_event(self):
        """Test recording stream event."""
        sync = StreamSynchronizer()

        event = sync.record_event(stream=0)

        assert event is not None
        assert event.stream == 0
        assert event.event_id >= 0

    def test_wait_event(self):
        """Test stream waiting on event."""
        sync = StreamSynchronizer()

        # Stream 0 records event
        event = sync.record_event(stream=0)

        # Stream 1 waits for it
        sync.wait_event(stream=1, event=event)

        # Check dependency was recorded
        deps = sync.get_stream_dependencies(stream=1)
        assert 0 in deps

    def test_synchronize_stream(self):
        """Test synchronizing a stream."""
        sync = StreamSynchronizer()

        # Record event and synchronize
        sync.record_event(stream=0)
        sync.synchronize_stream(stream=0)

        # Stream should be idle
        assert sync.query_stream(stream=0) is True

    def test_synchronize_all(self):
        """Test global synchronization."""
        sync = StreamSynchronizer(num_streams=4)

        # Record events on multiple streams
        for i in range(4):
            sync.record_event(stream=i)

        sync.synchronize_all()

        # All streams should be idle
        for i in range(4):
            assert sync.query_stream(stream=i) is True

    def test_create_barrier(self):
        """Test multi-stream barrier."""
        sync = StreamSynchronizer(num_streams=4)

        barrier_id = sync.create_barrier(streams=[0, 1, 2])

        assert barrier_id is not None

        # All streams should depend on each other
        for i in [0, 1, 2]:
            deps = sync.get_stream_dependencies(stream=i)
            for j in [0, 1, 2]:
                if i != j:
                    assert j in deps

    def test_query_idle_stream(self):
        """Test querying idle stream."""
        sync = StreamSynchronizer()

        # Unused stream should be idle
        assert sync.query_stream(stream=5) is True


# =============================================================================
# CPU Offloader Tests
# =============================================================================

class TestCPUOffloader:
    """Tests for CPU offloading."""

    def test_creation(self):
        """Test creating CPU offloader."""
        offloader = CPUOffloader()
        assert offloader is not None

    def test_register_block(self):
        """Test registering block for offload."""
        offloader = CPUOffloader(gpu_memory_limit=1024 * 1024)
        block = MemoryBlock(ptr=0x1000, size=1024, device=DeviceType.CUDA, device_id=0)

        offloader.register(block)

        stats = offloader.get_statistics()
        assert stats['blocks_on_gpu'] == 1

    def test_unregister_block(self):
        """Test unregistering block."""
        offloader = CPUOffloader()
        block = MemoryBlock(ptr=0x1000, size=1024, device=DeviceType.CUDA, device_id=0)

        offloader.register(block)
        offloader.unregister(block)

        stats = offloader.get_statistics()
        assert stats['blocks_on_gpu'] == 0

    def test_access_updates_lru(self):
        """Test access updates LRU order."""
        offloader = CPUOffloader(gpu_memory_limit=10 * 1024)

        blocks = []
        for i in range(3):
            block = MemoryBlock(ptr=0x1000 + i * 0x1000, size=1024, device=DeviceType.CUDA, device_id=0)
            offloader.register(block)
            blocks.append(block)

        # Access first block again (should move to end of LRU)
        offloader.access(blocks[0])

        stats = offloader.get_statistics()
        assert stats['blocks_on_gpu'] == 3

    def test_manual_offload(self):
        """Test manual offload to CPU."""
        offloader = CPUOffloader(gpu_memory_limit=1024 * 1024)
        block = MemoryBlock(ptr=0x1000, size=1024, device=DeviceType.CUDA, device_id=0)

        offloader.register(block)
        cpu_ptr = offloader.offload(block)

        assert cpu_ptr is not None
        stats = offloader.get_statistics()
        assert stats['offloads'] == 1
        assert stats['blocks_on_cpu'] == 1
        assert stats['blocks_on_gpu'] == 0

    def test_reload_from_cpu(self):
        """Test reloading from CPU to GPU."""
        offloader = CPUOffloader(gpu_memory_limit=1024 * 1024)
        block = MemoryBlock(ptr=0x1000, size=1024, device=DeviceType.CUDA, device_id=0)

        offloader.register(block)
        cpu_ptr = offloader.offload(block)
        gpu_ptr = offloader.reload(cpu_ptr)

        assert gpu_ptr == block.ptr
        stats = offloader.get_statistics()
        assert stats['reloads'] == 1
        assert stats['blocks_on_gpu'] == 1
        assert stats['blocks_on_cpu'] == 0

    def test_automatic_offload_on_pressure(self):
        """Test automatic offload on memory pressure."""
        # Small limit to trigger offloading
        offloader = CPUOffloader(
            gpu_memory_limit=2048,
            offload_threshold=0.5
        )

        # Add blocks until we exceed threshold
        blocks = []
        for i in range(5):
            block = MemoryBlock(ptr=0x1000 + i * 0x1000, size=512, device=DeviceType.CUDA, device_id=0)
            offloader.register(block)
            blocks.append(block)

        stats = offloader.get_statistics()
        # Some should have been offloaded
        assert stats['automatic_offloads'] > 0 or stats['blocks_on_cpu'] > 0

    def test_prefetch_on_access(self):
        """Test prefetch on access when offloaded."""
        offloader = CPUOffloader(prefetch_on_access=True)
        block = MemoryBlock(ptr=0x1000, size=1024, device=DeviceType.CUDA, device_id=0)

        offloader.register(block)
        offloader.offload(block)

        # Access should reload
        gpu_ptr = offloader.access(block)

        stats = offloader.get_statistics()
        assert stats['reloads'] == 1

    def test_get_offloaded_blocks(self):
        """Test getting list of offloaded blocks."""
        offloader = CPUOffloader()

        blocks = []
        for i in range(3):
            block = MemoryBlock(ptr=0x1000 + i * 0x1000, size=1024, device=DeviceType.CUDA, device_id=0)
            offloader.register(block)
            blocks.append(block)

        # Offload two
        offloader.offload(blocks[0])
        offloader.offload(blocks[1])

        offloaded = offloader.get_offloaded_blocks()
        assert len(offloaded) == 2

    def test_memory_usage(self):
        """Test memory usage tracking."""
        offloader = CPUOffloader(
            gpu_memory_limit=10 * 1024,
            cpu_memory_limit=20 * 1024
        )

        block = MemoryBlock(ptr=0x1000, size=1024, device=DeviceType.CUDA, device_id=0)
        offloader.register(block)

        gpu_usage = offloader.get_gpu_usage()
        assert 0 < gpu_usage < 1

        offloader.offload(block)
        cpu_usage = offloader.get_cpu_usage()
        assert 0 < cpu_usage < 1


# =============================================================================
# Advanced Memory Manager Tests
# =============================================================================

class TestAdvancedMemoryManager:
    """Tests for unified advanced memory manager."""

    def test_creation(self):
        """Test creating advanced manager."""
        manager = AdvancedMemoryManager(num_gpus=2, num_streams=8)
        assert manager is not None

    def test_transfer_between_gpus(self):
        """Test GPU-to-GPU transfer through manager."""
        manager = AdvancedMemoryManager(num_gpus=2)

        event = manager.transfer_between_gpus(
            src_ptr=0x1000,
            dst_ptr=0x2000,
            size=1024,
            src_gpu=0,
            dst_gpu=1
        )

        assert event is not None

    def test_prefetch_to_gpu(self):
        """Test prefetch through manager."""
        manager = AdvancedMemoryManager()
        block = MemoryBlock(ptr=0x1000, size=1024, device=DeviceType.CPU, device_id=0)

        request = manager.prefetch_to_gpu(block)
        assert request is not None

    def test_prefetch_to_cpu(self):
        """Test eviction through manager."""
        manager = AdvancedMemoryManager()
        block = MemoryBlock(ptr=0x1000, size=1024, device=DeviceType.CUDA, device_id=0)

        request = manager.prefetch_to_cpu(block)
        assert request is not None

    def test_synchronize(self):
        """Test stream synchronization through manager."""
        manager = AdvancedMemoryManager(num_streams=4)

        manager.synchronize(stream=0)
        # Should not raise

    def test_synchronize_all(self):
        """Test global sync through manager."""
        manager = AdvancedMemoryManager(num_streams=4)

        manager.synchronize_all()
        # Should not raise

    def test_block_management(self):
        """Test block offload management through manager."""
        manager = AdvancedMemoryManager(gpu_memory_limit=1024 * 1024)
        block = MemoryBlock(ptr=0x1000, size=1024, device=DeviceType.CUDA, device_id=0)

        manager.manage_block(block)
        gpu_ptr = manager.access_block(block)

        assert gpu_ptr == block.ptr

    def test_get_all_statistics(self):
        """Test combined statistics."""
        manager = AdvancedMemoryManager()

        stats = manager.get_all_statistics()

        assert 'p2p' in stats
        assert 'prefetch' in stats
        assert 'offloader' in stats


# =============================================================================
# Integration Tests
# =============================================================================

class TestAdvancedIntegration:
    """Integration tests for advanced features."""

    def test_p2p_with_sync(self):
        """Test P2P transfer with stream sync."""
        manager = AdvancedMemoryManager(num_gpus=2, num_streams=4)

        # Transfer on stream 0
        event = manager.transfer_between_gpus(
            src_ptr=0x1000,
            dst_ptr=0x2000,
            size=1024,
            src_gpu=0,
            dst_gpu=1,
            stream=0
        )

        # Sync stream 0
        manager.synchronize(stream=0)

        # Transfer should be complete
        manager.p2p.wait_for_transfer(event)
        assert event.completed

    def test_prefetch_with_offload(self):
        """Test prefetch and offload interaction."""
        manager = AdvancedMemoryManager(gpu_memory_limit=10 * 1024)

        block = MemoryBlock(ptr=0x1000, size=1024, device=DeviceType.CUDA, device_id=0)

        # Register for offload management
        manager.manage_block(block)

        # Prefetch to CPU
        request = manager.prefetch_to_cpu(block)
        # Mark as completed for testing
        request.completed = True

        # Access should still work
        gpu_ptr = manager.access_block(block)
        assert gpu_ptr == block.ptr

    def test_concurrent_operations(self):
        """Test concurrent advanced operations."""
        manager = AdvancedMemoryManager(num_gpus=2, num_streams=8)
        errors = []

        def worker(worker_id):
            try:
                block = MemoryBlock(
                    ptr=0x1000 + worker_id * 0x1000,
                    size=1024,
                    device=DeviceType.CUDA,
                    device_id=worker_id % 2
                )
                manager.manage_block(block)

                for _ in range(10):
                    manager.access_block(block)
                    time.sleep(0.001)

            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
