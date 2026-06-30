"""Unit tests for core GPU resources module."""

import pytest
import time
import uuid
from unittest.mock import MagicMock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gpusched.core.resources import (
    GPUType, JobState, PriorityClass, GPUResources,
    GPU, Node, Container, Pod, Job, Queue, Tenant, Cluster,
    create_gpu, create_node, create_training_job
)


class TestGPUType:
    """Tests for GPUType enum."""

    def test_gpu_types(self):
        """Test available GPU types."""
        assert GPUType.A100.value == "a100"
        assert GPUType.H100.value == "h100"
        assert GPUType.V100.value == "v100"
        assert GPUType.T4.value == "t4"


class TestGPUResources:
    """Tests for GPUResources class."""

    def test_initialization(self):
        """Test resource initialization."""
        resources = GPUResources(count=4, memory_gb=32.0, compute_units=2.0)
        assert resources.count == 4
        assert resources.memory_gb == 32.0
        assert resources.compute_units == 2.0
        assert resources.gpu_type is None

    def test_fits(self):
        """Test resource fitting check."""
        required = GPUResources(count=2, memory_gb=16.0, compute_units=0.5)
        available = GPUResources(count=4, memory_gb=32.0, compute_units=1.0)

        assert required.fits(available) is True

        # Test insufficient count
        required.count = 5
        assert required.fits(available) is False

        # Test insufficient memory
        required.count = 2
        required.memory_gb = 40.0
        assert required.fits(available) is False

        # Test GPU type mismatch
        required.memory_gb = 16.0
        required.gpu_type = GPUType.A100
        available.gpu_type = GPUType.V100
        assert required.fits(available) is False

    def test_subtract(self):
        """Test resource subtraction."""
        r1 = GPUResources(count=4, memory_gb=32.0, compute_units=2.0)
        r2 = GPUResources(count=2, memory_gb=16.0, compute_units=0.5)

        result = r1.subtract(r2)
        assert result.count == 2
        assert result.memory_gb == 16.0
        assert result.compute_units == 1.5

    def test_add(self):
        """Test resource addition."""
        r1 = GPUResources(count=2, memory_gb=16.0, compute_units=1.0)
        r2 = GPUResources(count=2, memory_gb=16.0, compute_units=0.5)

        result = r1.add(r2)
        assert result.count == 4
        assert result.memory_gb == 32.0
        assert result.compute_units == 1.5


class TestGPU:
    """Tests for GPU class."""

    def test_initialization(self):
        """Test GPU initialization."""
        gpu = GPU(
            gpu_id="gpu-001",
            node_id="node-001",
            gpu_type=GPUType.A100,
            total_memory_gb=80.0,
            available_memory_gb=80.0,
            compute_capability=(8, 0)
        )

        assert gpu.gpu_id == "gpu-001"
        assert gpu.gpu_type == GPUType.A100
        assert gpu.total_memory_gb == 80.0
        assert gpu.mig_enabled is False
        assert len(gpu.allocated_jobs) == 0

    def test_available_compute(self):
        """Test available compute calculation."""
        gpu = create_gpu("node-001", GPUType.A100, 80.0)

        # Initially fully available
        assert gpu.available_compute == 1.0

        # Allocate some jobs
        gpu.allocated_jobs = ["job1", "job2"]
        assert gpu.available_compute == 0.5

        # Max allocation
        gpu.allocated_jobs = ["job1", "job2", "job3", "job4"]
        assert gpu.available_compute == 0.0

    def test_get_available_resources(self):
        """Test getting available resources."""
        gpu = create_gpu("node-001", GPUType.A100, 80.0)
        gpu.available_memory_gb = 60.0

        resources = gpu.get_available_resources()
        assert resources.count == 1
        assert resources.memory_gb == 60.0
        assert resources.compute_units == 1.0
        assert resources.gpu_type == GPUType.A100

    def test_can_allocate(self):
        """Test allocation check."""
        gpu = create_gpu("node-001", GPUType.A100, 80.0)

        # Can allocate within limits
        requirements = GPUResources(count=1, memory_gb=40.0, compute_units=0.5)
        assert gpu.can_allocate(requirements) is True

        # Cannot allocate beyond memory
        requirements = GPUResources(count=1, memory_gb=100.0, compute_units=0.5)
        assert gpu.can_allocate(requirements) is False


class TestNode:
    """Tests for Node class."""

    def test_initialization(self):
        """Test node initialization."""
        node = create_node("host-001", num_gpus=4, gpu_type=GPUType.A100)

        assert node.hostname == "host-001"
        assert len(node.gpus) == 4
        assert node.total_gpu_count == 4
        assert node.available_gpu_count == 4

    def test_get_gpu_by_type(self):
        """Test getting GPUs by type."""
        node = create_node("host-001", num_gpus=2, gpu_type=GPUType.A100)

        # Add different GPU types
        node.gpus.append(create_gpu(node.node_id, GPUType.V100, 32.0))

        a100_gpus = node.get_gpu_by_type(GPUType.A100)
        assert len(a100_gpus) == 2

        v100_gpus = node.get_gpu_by_type(GPUType.V100)
        assert len(v100_gpus) == 1

    def test_is_schedulable(self):
        """Test schedulability check."""
        node = create_node("host-001", num_gpus=4)

        # Initially schedulable
        assert node.is_schedulable() is True

        # Not ready
        node.conditions["Ready"] = False
        assert node.is_schedulable() is False

        # Has NoSchedule taint
        node.conditions["Ready"] = True
        node.taints.append("NoSchedule")
        assert node.is_schedulable() is False


class TestContainer:
    """Tests for Container class."""

    def test_initialization(self):
        """Test container initialization."""
        container = Container(
            name="trainer",
            image="pytorch:latest",
            gpu_resources=GPUResources(count=2, memory_gb=32.0)
        )

        assert container.name == "trainer"
        assert container.image == "pytorch:latest"
        assert container.gpu_resources.count == 2
        assert container.cpu_request == 1.0


class TestPod:
    """Tests for Pod class."""

    def test_initialization(self):
        """Test pod initialization."""
        container = Container(name="main", image="app:latest")
        pod = Pod(
            pod_id="pod-001",
            name="training-pod",
            namespace="default",
            containers=[container]
        )

        assert pod.pod_id == "pod-001"
        assert pod.state == JobState.PENDING
        assert pod.priority == PriorityClass.NORMAL
        assert pod.assigned_node is None

    def test_total_gpu_request(self):
        """Test total GPU request calculation."""
        containers = [
            Container(
                name="c1",
                image="img1",
                gpu_resources=GPUResources(count=2, memory_gb=16.0)
            ),
            Container(
                name="c2",
                image="img2",
                gpu_resources=GPUResources(count=1, memory_gb=8.0)
            )
        ]

        pod = Pod(
            pod_id="pod-001",
            name="multi-container",
            namespace="default",
            containers=containers
        )

        total = pod.total_gpu_request
        assert total.count == 3
        assert total.memory_gb == 24.0

    def test_wait_time(self):
        """Test wait time calculation."""
        with patch('time.time', return_value=1000.0):
            pod = Pod(
                pod_id="pod-001",
                name="test-pod",
                namespace="default",
                containers=[]
            )

        with patch('time.time', return_value=1010.0):
            assert pod.wait_time == 10.0

        # After starting
        pod.started_at = 1005.0
        assert pod.wait_time == 5.0


class TestJob:
    """Tests for Job class."""

    def test_initialization(self):
        """Test job initialization."""
        job = create_training_job(
            name="training-job",
            num_gpus=2,
            parallelism=3
        )

        assert job.name == "training-job"
        assert len(job.pods) == 3
        assert job.parallelism == 3
        assert job.tenant_id == "default"

    def test_state(self):
        """Test job state calculation."""
        job = create_training_job("job", num_gpus=1, parallelism=3)

        # All pending
        assert job.state == JobState.PENDING

        # Some running
        job.pods[0].state = JobState.RUNNING
        assert job.state == JobState.RUNNING

        # One failed
        job.pods[1].state = JobState.FAILED
        assert job.state == JobState.FAILED

        # All completed
        job.pods[0].state = JobState.COMPLETED
        job.pods[1].state = JobState.COMPLETED
        job.pods[2].state = JobState.COMPLETED
        assert job.state == JobState.COMPLETED

    def test_total_gpu_request(self):
        """Test total job GPU request."""
        job = create_training_job(
            name="job",
            num_gpus=2,
            gpu_memory_gb=16.0,
            parallelism=3
        )

        total = job.total_gpu_request
        assert total.count == 6  # 2 GPUs * 3 pods
        assert total.memory_gb == 48.0  # 16 GB * 3 pods


class TestQueue:
    """Tests for Queue class."""

    def test_initialization(self):
        """Test queue initialization."""
        queue = Queue(
            name="high-priority",
            tenant_id="tenant-001",
            gpu_quota=50
        )

        assert queue.name == "high-priority"
        assert queue.gpu_quota == 50
        assert queue.gpu_used == 0
        assert queue.available_gpu_quota == 50

    def test_can_admit(self):
        """Test job admission check."""
        queue = Queue(
            name="default",
            tenant_id="tenant-001",
            gpu_quota=10,
            max_jobs=5
        )

        job = create_training_job("job", num_gpus=4)

        # Can admit initially
        assert queue.can_admit(job) is True

        # Cannot admit - exceeds GPU quota
        queue.gpu_used = 7
        assert queue.can_admit(job) is False

        # Cannot admit - max jobs reached
        queue.gpu_used = 0
        queue.pending_jobs = ["j1", "j2", "j3"]
        queue.running_jobs = ["j4", "j5"]
        assert queue.can_admit(job) is False


class TestTenant:
    """Tests for Tenant class."""

    def test_initialization(self):
        """Test tenant initialization."""
        tenant = Tenant(
            tenant_id="tenant-001",
            name="Research Team",
            total_gpu_quota=100
        )

        assert tenant.tenant_id == "tenant-001"
        assert tenant.total_gpu_quota == 100
        assert tenant.gpu_used == 0
        assert tenant.priority_class == PriorityClass.NORMAL

    def test_utilization(self):
        """Test utilization calculation."""
        tenant = Tenant(
            tenant_id="tenant-001",
            name="Team",
            total_gpu_quota=100
        )

        assert tenant.utilization == 0.0

        tenant.gpu_used = 25
        assert tenant.utilization == 0.25

        tenant.gpu_used = 100
        assert tenant.utilization == 1.0


class TestCluster:
    """Tests for Cluster class."""

    def test_initialization(self):
        """Test cluster initialization."""
        cluster = Cluster(cluster_id="cluster-001")

        assert cluster.cluster_id == "cluster-001"
        assert len(cluster.nodes) == 0
        assert len(cluster.jobs) == 0
        assert cluster.total_gpus == 0

    def test_add_remove_node(self):
        """Test adding and removing nodes."""
        cluster = Cluster(cluster_id="cluster-001")
        node = create_node("host-001", num_gpus=4)

        cluster.add_node(node)
        assert len(cluster.nodes) == 1
        assert cluster.total_gpus == 4

        cluster.remove_node(node.node_id)
        assert len(cluster.nodes) == 0

    def test_submit_job(self):
        """Test job submission."""
        cluster = Cluster(cluster_id="cluster-001")

        # Add queue
        queue = Queue("default", "tenant-001", gpu_quota=10)
        cluster.queues["default"] = queue

        job = create_training_job("job", num_gpus=2)
        job.queue_name = "default"

        # Submit job
        assert cluster.submit_job(job) is True
        assert job.job_id in cluster.jobs
        assert len(cluster.pods) == len(job.pods)
        assert job.job_id in queue.pending_jobs

    def test_get_schedulable_nodes(self):
        """Test getting schedulable nodes."""
        cluster = Cluster(cluster_id="cluster-001")

        # Add schedulable node
        node1 = create_node("host-001", num_gpus=4)
        cluster.add_node(node1)

        # Add non-schedulable node
        node2 = create_node("host-002", num_gpus=4)
        node2.conditions["Ready"] = False
        cluster.add_node(node2)

        schedulable = cluster.get_schedulable_nodes()
        assert len(schedulable) == 1
        assert schedulable[0].node_id == node1.node_id

    def test_pending_running_pods(self):
        """Test getting pending and running pods."""
        cluster = Cluster(cluster_id="cluster-001")

        job = create_training_job("job", parallelism=3)
        cluster.submit_job(job)

        # All pending initially
        assert len(cluster.pending_pods) == 3
        assert len(cluster.running_pods) == 0

        # Move some to running
        job.pods[0].state = JobState.RUNNING
        job.pods[1].state = JobState.RUNNING

        assert len(cluster.pending_pods) == 1
        assert len(cluster.running_pods) == 2