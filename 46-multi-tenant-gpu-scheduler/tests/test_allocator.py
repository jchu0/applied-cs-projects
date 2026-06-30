"""Unit tests for GPU allocator module."""

import pytest
import time
from unittest.mock import MagicMock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gpusched.core.resources import (
    GPUType, GPUResources, GPU, Node, Container, Pod,
    create_gpu, create_node
)
from gpusched.allocator.allocator import (
    AllocationMode, GPUAllocation, MIGProfile, MIG_PROFILES,
    GPUAllocator, MIGAllocator, SharedGPUAllocator, AllocationManager
)


class TestGPUAllocation:
    """Tests for GPUAllocation class."""

    def test_initialization(self):
        """Test allocation record initialization."""
        with patch('time.time', return_value=1000.0):
            allocation = GPUAllocation(
                allocation_id="alloc-001",
                pod_id="pod-001",
                node_id="node-001",
                gpu_id="gpu-001",
                mode=AllocationMode.EXCLUSIVE,
                memory_allocated_gb=80.0,
                compute_fraction=1.0
            )

        assert allocation.allocation_id == "alloc-001"
        assert allocation.mode == AllocationMode.EXCLUSIVE
        assert allocation.start_time == 1000.0
        assert allocation.end_time is None


class TestMIGProfile:
    """Tests for MIG profiles."""

    def test_standard_profiles(self):
        """Test standard MIG profiles."""
        assert "1g.5gb" in MIG_PROFILES
        assert "2g.10gb" in MIG_PROFILES
        assert "7g.40gb" in MIG_PROFILES

        profile = MIG_PROFILES["1g.5gb"]
        assert profile.memory_gb == 5.0
        assert profile.max_instances == 7


class TestGPUAllocator:
    """Tests for base GPUAllocator."""

    def test_allocate_exclusive(self):
        """Test exclusive GPU allocation."""
        allocator = GPUAllocator()

        node = create_node("host-001", num_gpus=2, gpu_memory_gb=80.0)
        gpu = node.gpus[0]

        pod = Pod(
            pod_id="pod-001",
            name="test-pod",
            namespace="default",
            containers=[Container(
                name="main",
                image="app",
                gpu_resources=GPUResources(count=1, memory_gb=80.0)
            )]
        )

        allocation = allocator.allocate(pod, node, gpu)

        assert allocation is not None
        assert allocation.mode == AllocationMode.EXCLUSIVE
        assert allocation.memory_allocated_gb == 80.0
        assert allocation.compute_fraction == 1.0
        assert pod.pod_id in gpu.allocated_jobs

    def test_allocate_insufficient_memory(self):
        """Test allocation with insufficient memory."""
        allocator = GPUAllocator()

        node = create_node("host-001", gpu_memory_gb=40.0)
        gpu = node.gpus[0]

        pod = Pod(
            pod_id="pod-001",
            name="test-pod",
            namespace="default",
            containers=[Container(
                name="main",
                image="app",
                gpu_resources=GPUResources(count=1, memory_gb=80.0)
            )]
        )

        allocation = allocator.allocate(pod, node, gpu)
        assert allocation is None

    def test_release(self):
        """Test releasing allocation."""
        allocator = GPUAllocator()

        node = create_node("host-001", gpu_memory_gb=80.0)
        gpu = node.gpus[0]

        # Create allocation
        allocation = GPUAllocation(
            allocation_id="alloc-001",
            pod_id="pod-001",
            node_id=node.node_id,
            gpu_id=gpu.gpu_id,
            mode=AllocationMode.EXCLUSIVE,
            memory_allocated_gb=80.0,
            compute_fraction=1.0
        )

        gpu.allocated_jobs.append("pod-001")
        gpu.available_memory_gb = 0.0

        # Release
        allocator.release(allocation, node, gpu)

        assert "pod-001" not in gpu.allocated_jobs
        assert gpu.available_memory_gb == 80.0
        assert allocation.end_time is not None


class TestMIGAllocator:
    """Tests for MIG allocator."""

    def test_can_use_mig(self):
        """Test MIG capability check."""
        allocator = MIGAllocator()

        # A100 supports MIG
        gpu_a100 = create_gpu("node-001", GPUType.A100)
        assert allocator.can_use_mig(gpu_a100) is True

        # V100 doesn't support MIG
        gpu_v100 = create_gpu("node-001", GPUType.V100)
        assert allocator.can_use_mig(gpu_v100) is False

    def test_find_mig_profile(self):
        """Test finding suitable MIG profile."""
        allocator = MIGAllocator()

        # Small request
        small_req = GPUResources(count=1, memory_gb=5.0, compute_units=0.15)
        profile = allocator.find_mig_profile(small_req)
        assert profile.profile_name == "1g.5gb"

        # Medium request
        med_req = GPUResources(count=1, memory_gb=15.0, compute_units=0.3)
        profile = allocator.find_mig_profile(med_req)
        assert profile.profile_name == "2g.10gb"

        # Large request
        large_req = GPUResources(count=1, memory_gb=35.0, compute_units=0.5)
        profile = allocator.find_mig_profile(large_req)
        assert profile.profile_name == "7g.40gb"

    def test_allocate_mig_instance(self):
        """Test MIG instance allocation."""
        allocator = MIGAllocator()

        node = create_node("host-001", gpu_type=GPUType.A100, gpu_memory_gb=80.0)
        gpu = node.gpus[0]
        gpu.mig_enabled = True

        pod = Pod(
            pod_id="pod-001",
            name="test-pod",
            namespace="default",
            containers=[Container(
                name="main",
                image="app",
                gpu_resources=GPUResources(count=1, memory_gb=10.0)
            )]
        )

        allocation = allocator.allocate(pod, node, gpu)

        assert allocation is not None
        assert allocation.mode == AllocationMode.MIG
        assert allocation.mig_instance_id is not None
        assert allocation.mig_instance_id in gpu.mig_instances
        assert allocation.memory_allocated_gb == 10.0

    def test_mig_instance_limit(self):
        """Test MIG instance count limits."""
        allocator = MIGAllocator()

        node = create_node("host-001", gpu_type=GPUType.A100)
        gpu = node.gpus[0]
        gpu.mig_enabled = True

        # Fill up with 1g.5gb instances (max 7)
        for i in range(7):
            gpu.mig_instances.append(f"mig-{i}")

        pod = Pod(
            pod_id="pod-001",
            name="test-pod",
            namespace="default",
            containers=[Container(
                name="main",
                image="app",
                gpu_resources=GPUResources(count=1, memory_gb=5.0)
            )]
        )

        allocation = allocator.allocate(pod, node, gpu)
        assert allocation is None  # No more instances available


class TestSharedGPUAllocator:
    """Tests for shared GPU allocator."""

    def test_allocate_shared(self):
        """Test shared GPU allocation."""
        allocator = SharedGPUAllocator(max_sharing_factor=4)

        node = create_node("host-001", gpu_memory_gb=80.0)
        gpu = node.gpus[0]

        # First allocation
        pod1 = Pod(
            pod_id="pod-001",
            name="pod-1",
            namespace="default",
            containers=[Container(
                name="main",
                image="app",
                gpu_resources=GPUResources(count=1, memory_gb=20.0, compute_units=0.25)
            )]
        )

        alloc1 = allocator.allocate(pod1, node, gpu)
        assert alloc1 is not None
        assert alloc1.mode == AllocationMode.SHARED
        assert alloc1.compute_fraction == 0.25

        # Second allocation on same GPU
        pod2 = Pod(
            pod_id="pod-002",
            name="pod-2",
            namespace="default",
            containers=[Container(
                name="main",
                image="app",
                gpu_resources=GPUResources(count=1, memory_gb=20.0, compute_units=0.25)
            )]
        )

        alloc2 = allocator.allocate(pod2, node, gpu)
        assert alloc2 is not None
        assert len(gpu.allocated_jobs) == 2

    def test_max_sharing_limit(self):
        """Test maximum sharing factor limit."""
        allocator = SharedGPUAllocator(max_sharing_factor=2)

        node = create_node("host-001", gpu_memory_gb=80.0)
        gpu = node.gpus[0]

        # Fill up to max sharing
        gpu.allocated_jobs = ["pod-1", "pod-2"]

        pod = Pod(
            pod_id="pod-003",
            name="pod-3",
            namespace="default",
            containers=[Container(
                name="main",
                image="app",
                gpu_resources=GPUResources(count=1, memory_gb=10.0)
            )]
        )

        allocation = allocator.allocate(pod, node, gpu)
        assert allocation is None  # Max sharing reached

    def test_memory_limit_shared(self):
        """Test memory limits in shared allocation."""
        allocator = SharedGPUAllocator()

        node = create_node("host-001", gpu_memory_gb=80.0)
        gpu = node.gpus[0]
        gpu.available_memory_gb = 20.0  # Most memory already used

        pod = Pod(
            pod_id="pod-001",
            name="pod",
            namespace="default",
            containers=[Container(
                name="main",
                image="app",
                gpu_resources=GPUResources(count=1, memory_gb=30.0)
            )]
        )

        allocation = allocator.allocate(pod, node, gpu)
        assert allocation is None  # Not enough memory


class TestAllocationManager:
    """Tests for AllocationManager."""

    def test_initialization(self):
        """Test manager initialization."""
        from gpusched.core.resources import Cluster
        cluster = Cluster("cluster-001")
        manager = AllocationManager(cluster)

        assert manager.cluster == cluster
        assert len(manager.allocations) == 0
        assert manager.allocator is not None

    def test_allocate_pod(self):
        """Test pod allocation."""
        from gpusched.core.resources import Cluster
        cluster = Cluster("cluster-001")

        node = create_node("host-001", num_gpus=2)
        cluster.add_node(node)

        manager = AllocationManager(cluster)

        pod = Pod(
            pod_id="pod-001",
            name="test-pod",
            namespace="default",
            containers=[Container(
                name="main",
                image="app",
                gpu_resources=GPUResources(count=2, memory_gb=40.0)
            )]
        )
        pod.assigned_node = node.node_id

        allocations = manager.allocate_pod(pod, node.node_id, ["gpu-0", "gpu-1"])

        assert len(allocations) == 2
        assert pod.pod_id in manager.allocations
        assert len(manager.allocations[pod.pod_id]) == 2

    def test_release_pod(self):
        """Test releasing pod allocations."""
        from gpusched.core.resources import Cluster
        cluster = Cluster("cluster-001")

        node = create_node("host-001", num_gpus=1)
        cluster.add_node(node)

        manager = AllocationManager(cluster)

        # Create allocation
        allocation = GPUAllocation(
            allocation_id="alloc-001",
            pod_id="pod-001",
            node_id=node.node_id,
            gpu_id=node.gpus[0].gpu_id,
            mode=AllocationMode.EXCLUSIVE,
            memory_allocated_gb=80.0,
            compute_fraction=1.0
        )

        manager.allocations["pod-001"] = [allocation]
        node.gpus[0].allocated_jobs.append("pod-001")
        node.gpus[0].available_memory_gb = 0

        # Release
        manager.release_pod("pod-001")

        assert "pod-001" not in manager.allocations
        assert "pod-001" not in node.gpus[0].allocated_jobs
        assert node.gpus[0].available_memory_gb == 80.0

    def test_get_pod_allocations(self):
        """Test getting pod allocations."""
        from gpusched.core.resources import Cluster
        cluster = Cluster("cluster-001")
        manager = AllocationManager(cluster)

        allocation = GPUAllocation(
            allocation_id="alloc-001",
            pod_id="pod-001",
            node_id="node-001",
            gpu_id="gpu-001",
            mode=AllocationMode.EXCLUSIVE,
            memory_allocated_gb=80.0,
            compute_fraction=1.0
        )

        manager.allocations["pod-001"] = [allocation]

        allocations = manager.get_pod_allocations("pod-001")
        assert len(allocations) == 1
        assert allocations[0].allocation_id == "alloc-001"

        # Non-existent pod
        assert manager.get_pod_allocations("pod-999") == []

    def test_update_allocation_stats(self):
        """Test updating allocation statistics."""
        from gpusched.core.resources import Cluster
        cluster = Cluster("cluster-001")

        node = create_node("host-001", num_gpus=2)
        cluster.add_node(node)

        manager = AllocationManager(cluster)

        # Create some allocations
        for i in range(2):
            allocation = GPUAllocation(
                allocation_id=f"alloc-{i}",
                pod_id=f"pod-{i}",
                node_id=node.node_id,
                gpu_id=node.gpus[i].gpu_id,
                mode=AllocationMode.EXCLUSIVE,
                memory_allocated_gb=80.0,
                compute_fraction=1.0
            )
            manager.allocations[f"pod-{i}"] = [allocation]
            node.gpus[i].allocated_jobs.append(f"pod-{i}")

        stats = manager.update_allocation_stats()

        assert stats["total_allocations"] == 2
        assert stats["total_gpus_allocated"] == 2
        assert stats["allocation_by_mode"][AllocationMode.EXCLUSIVE] == 2

    def test_cleanup_expired(self):
        """Test cleaning up expired allocations."""
        from gpusched.core.resources import Cluster
        cluster = Cluster("cluster-001")
        manager = AllocationManager(cluster)

        # Create expired allocation
        with patch('time.time', return_value=1000.0):
            allocation = GPUAllocation(
                allocation_id="alloc-001",
                pod_id="pod-001",
                node_id="node-001",
                gpu_id="gpu-001",
                mode=AllocationMode.EXCLUSIVE,
                memory_allocated_gb=80.0,
                compute_fraction=1.0
            )
            allocation.end_time = 900.0  # Ended before current time

        manager.allocations["pod-001"] = [allocation]

        with patch('time.time', return_value=2000.0):
            manager.cleanup_expired(max_age_seconds=1000)

        assert "pod-001" not in manager.allocations