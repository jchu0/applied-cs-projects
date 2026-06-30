"""Pytest configuration and fixtures for GPU scheduler tests."""

import pytest
import sys
import os
from unittest.mock import MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gpusched.core.resources import (
    GPUType, PriorityClass, GPUResources,
    Cluster, Node, GPU, Pod, Container, Job, Queue, Tenant,
    create_node, create_gpu, create_training_job
)
from gpusched.scheduler.scheduler import GPUScheduler
from gpusched.allocator.allocator import AllocationManager
from gpusched.monitor.monitor import GPUMonitor


@pytest.fixture
def sample_cluster():
    """Create a sample cluster with nodes and GPUs."""
    cluster = Cluster("test-cluster")

    # Add diverse nodes
    node1 = create_node("host-001", num_gpus=4, gpu_type=GPUType.A100, gpu_memory_gb=80.0)
    node1.labels = {"zone": "us-west-1", "type": "compute"}

    node2 = create_node("host-002", num_gpus=8, gpu_type=GPUType.V100, gpu_memory_gb=32.0)
    node2.labels = {"zone": "us-west-2", "type": "compute"}

    node3 = create_node("host-003", num_gpus=2, gpu_type=GPUType.T4, gpu_memory_gb=16.0)
    node3.labels = {"zone": "us-east-1", "type": "inference"}

    cluster.add_node(node1)
    cluster.add_node(node2)
    cluster.add_node(node3)

    return cluster


@pytest.fixture
def multi_tenant_cluster():
    """Create a cluster with multiple tenants and queues."""
    cluster = Cluster("multi-tenant-cluster")

    # Add nodes
    for i in range(4):
        node = create_node(f"host-{i:03d}", num_gpus=8, gpu_type=GPUType.A100)
        cluster.add_node(node)

    # Create tenants
    tenants = [
        Tenant("tenant-ml", "Machine Learning Team", total_gpu_quota=16),
        Tenant("tenant-data", "Data Team", total_gpu_quota=8),
        Tenant("tenant-research", "Research Team", total_gpu_quota=8)
    ]

    for tenant in tenants:
        cluster.tenants[tenant.tenant_id] = tenant

    # Create queues
    queues = [
        Queue("ml-training", "tenant-ml", gpu_quota=12, priority_weight=2.0),
        Queue("ml-inference", "tenant-ml", gpu_quota=4, priority_weight=1.0),
        Queue("data-processing", "tenant-data", gpu_quota=8, priority_weight=1.5),
        Queue("research-exp", "tenant-research", gpu_quota=8, priority_weight=1.0)
    ]

    for queue in queues:
        cluster.queues[queue.name] = queue

    return cluster


@pytest.fixture
def sample_jobs():
    """Create a variety of sample jobs."""
    jobs = []

    # Training job
    training_job = create_training_job(
        name="model-training",
        namespace="ml",
        num_gpus=4,
        gpu_memory_gb=32.0,
        parallelism=2,
        priority=PriorityClass.HIGH,
        tenant_id="tenant-ml"
    )
    training_job.queue_name = "ml-training"
    jobs.append(training_job)

    # Inference job
    inference_job = create_training_job(
        name="model-inference",
        namespace="ml",
        num_gpus=1,
        gpu_memory_gb=8.0,
        parallelism=1,
        priority=PriorityClass.NORMAL,
        tenant_id="tenant-ml"
    )
    inference_job.queue_name = "ml-inference"
    jobs.append(inference_job)

    # Data processing job
    data_job = create_training_job(
        name="data-pipeline",
        namespace="data",
        num_gpus=2,
        gpu_memory_gb=16.0,
        parallelism=4,
        priority=PriorityClass.LOW,
        tenant_id="tenant-data"
    )
    data_job.queue_name = "data-processing"
    data_job.preemptible = True
    jobs.append(data_job)

    # Research experiment (gang scheduled)
    research_job = create_training_job(
        name="distributed-exp",
        namespace="research",
        num_gpus=2,
        gpu_memory_gb=32.0,
        parallelism=4,
        priority=PriorityClass.NORMAL,
        tenant_id="tenant-research"
    )
    research_job.queue_name = "research-exp"
    research_job.gang_schedule = True
    jobs.append(research_job)

    return jobs


@pytest.fixture
def scheduler_with_cluster(sample_cluster):
    """Create a scheduler with a sample cluster."""
    return GPUScheduler(sample_cluster)


@pytest.fixture
def allocator_with_cluster(sample_cluster):
    """Create an allocator with a sample cluster."""
    return AllocationManager(sample_cluster)


@pytest.fixture
def monitor_with_cluster(sample_cluster):
    """Create a monitor with a sample cluster."""
    return GPUMonitor(sample_cluster)


@pytest.fixture
def sample_pod():
    """Create a sample pod."""
    container = Container(
        name="trainer",
        image="pytorch:latest",
        gpu_resources=GPUResources(
            count=2,
            memory_gb=32.0,
            compute_units=1.0,
            gpu_type=GPUType.A100
        ),
        cpu_request=8.0,
        memory_request_gb=64.0
    )

    pod = Pod(
        pod_id="pod-001",
        name="training-pod",
        namespace="default",
        containers=[container],
        priority=PriorityClass.NORMAL
    )

    return pod


@pytest.fixture
def gpu_resources_fixtures():
    """Various GPU resource requirements."""
    return {
        "small": GPUResources(count=1, memory_gb=8.0, compute_units=0.25),
        "medium": GPUResources(count=2, memory_gb=16.0, compute_units=0.5),
        "large": GPUResources(count=4, memory_gb=32.0, compute_units=1.0),
        "xlarge": GPUResources(count=8, memory_gb=80.0, compute_units=1.0),
        "a100_specific": GPUResources(
            count=2,
            memory_gb=80.0,
            compute_units=1.0,
            gpu_type=GPUType.A100
        ),
        "v100_specific": GPUResources(
            count=4,
            memory_gb=32.0,
            compute_units=1.0,
            gpu_type=GPUType.V100
        )
    }


@pytest.fixture
def mock_gpu_metrics():
    """Mock GPU metrics for testing."""
    from gpusched.monitor.monitor import GPUMetrics

    return [
        GPUMetrics(
            gpu_id="gpu-001",
            utilization=75.0,
            memory_used_gb=60.0,
            memory_total_gb=80.0,
            temperature=70.0,
            power_usage=250.0,
            pcie_throughput_mb=5000.0,
            error_count=0
        ),
        GPUMetrics(
            gpu_id="gpu-002",
            utilization=50.0,
            memory_used_gb=20.0,
            memory_total_gb=80.0,
            temperature=65.0,
            power_usage=200.0,
            pcie_throughput_mb=3000.0,
            error_count=0
        ),
        GPUMetrics(
            gpu_id="gpu-003",
            utilization=95.0,
            memory_used_gb=75.0,
            memory_total_gb=80.0,
            temperature=85.0,
            power_usage=300.0,
            pcie_throughput_mb=8000.0,
            error_count=2
        )
    ]


@pytest.fixture
def failed_node():
    """Create a node in failed state."""
    node = create_node("failed-host", num_gpus=4)
    node.conditions["Ready"] = False
    node.conditions["MemoryPressure"] = True
    node.conditions["DiskPressure"] = True
    node.taints.append("node.kubernetes.io/unreachable:NoSchedule")

    return node


@pytest.fixture
def busy_node():
    """Create a fully allocated node."""
    node = create_node("busy-host", num_gpus=4, gpu_type=GPUType.A100)

    # Mark all GPUs as allocated
    for gpu in node.gpus:
        gpu.allocated_jobs = ["job-1", "job-2"]
        gpu.available_memory_gb = 0
        gpu.utilization = 95.0

    node.available_cpu_cores = 2
    node.available_memory_gb = 10.0

    return node


@pytest.fixture(autouse=True)
def reset_singleton_state():
    """Reset any singleton state between tests."""
    # Add any cleanup needed for singleton objects
    yield
    # Cleanup after test


@pytest.fixture
def time_mock():
    """Mock time.time() for consistent testing."""
    with pytest.mock.patch('time.time') as mock_time:
        mock_time.return_value = 1000.0
        yield mock_time