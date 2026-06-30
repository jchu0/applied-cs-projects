"""Shared test fixtures."""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import asyncio
from datetime import datetime
from typing import AsyncGenerator
from uuid import uuid4

import pytest
import pytest_asyncio

from ml_orchestrator.core.models import (
    Checkpoint,
    CheckpointConfig,
    CheckpointPolicy,
    DistributedConfig,
    Experiment,
    JobConfig,
    JobPriority,
    JobStatus,
    MetricValue,
    ResourceRequest,
    StorageBackend,
    TrainingJob,
    WorkerInfo,
    WorkerStatus,
)
from ml_orchestrator.core.job_manager import JobManager, JobStore
from ml_orchestrator.scheduling.scheduler import Scheduler
from ml_orchestrator.scheduling.priority_queue import PriorityQueue
from ml_orchestrator.resources.allocator import ResourceAllocator
from ml_orchestrator.resources.gpu_manager import GPUManager, GPUPool, GPUInfo
from ml_orchestrator.checkpoint.manager import CheckpointManager
from ml_orchestrator.checkpoint.storage import LocalStorage
from ml_orchestrator.experiment.tracker import ExperimentTracker


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_job_config() -> JobConfig:
    """Create a sample job configuration."""
    return JobConfig(
        script_path="/path/to/train.py",
        script_args=["--epochs", "10"],
        epochs=10,
        batch_size=32,
        learning_rate=0.001,
        timeout_hours=2.0,
    )


@pytest.fixture
def sample_distributed_config() -> DistributedConfig:
    """Create a sample distributed config."""
    return DistributedConfig(
        enabled=True,
        world_size=4,
        num_nodes=2,
        gpus_per_node=2,
        backend="nccl",
    )


@pytest.fixture
def sample_resource_request() -> ResourceRequest:
    """Create a sample resource request."""
    return ResourceRequest(
        cpus=4,
        memory_gb=16.0,
        gpus=1,
        gpu_type="A100",
        gpu_memory_gb=40.0,
        storage_gb=50.0,
    )


@pytest.fixture
def sample_job(sample_job_config: JobConfig, sample_resource_request: ResourceRequest) -> TrainingJob:
    """Create a sample training job."""
    return TrainingJob(
        name="test-job",
        user_id="user-123",
        team_id="team-456",
        config=sample_job_config,
        resources=sample_resource_request,
        priority=JobPriority.NORMAL,
    )


@pytest.fixture
def sample_worker() -> WorkerInfo:
    """Create a sample worker."""
    return WorkerInfo(
        hostname="worker-1",
        ip_address="192.168.1.100",
        port=8000,
        resources=ResourceRequest(
            cpus=32,
            memory_gb=128.0,
            gpus=4,
            gpu_type="A100",
            gpu_memory_gb=80.0,
            storage_gb=1000.0,
        ),
        status=WorkerStatus.READY,
        labels={"zone": "us-east-1a"},
    )


@pytest.fixture
def sample_gpu() -> GPUInfo:
    """Create a sample GPU info."""
    return GPUInfo(
        node_id="node-1",
        device_index=0,
        name="NVIDIA A100-SXM4-80GB",
        gpu_type="A100",
        memory_total_gb=80.0,
        memory_free_gb=80.0,
    )


@pytest_asyncio.fixture
async def job_store() -> AsyncGenerator[JobStore, None]:
    """Create a job store."""
    store = JobStore()
    yield store


@pytest_asyncio.fixture
async def job_manager(job_store: JobStore) -> AsyncGenerator[JobManager, None]:
    """Create a job manager."""
    manager = JobManager(store=job_store)
    yield manager


@pytest_asyncio.fixture
async def scheduler(job_manager: JobManager) -> AsyncGenerator[Scheduler, None]:
    """Create a scheduler."""
    sched = Scheduler(job_manager)
    yield sched
    await sched.stop()


@pytest_asyncio.fixture
async def priority_queue() -> AsyncGenerator[PriorityQueue, None]:
    """Create a priority queue."""
    queue = PriorityQueue()
    yield queue
    await queue.clear()


@pytest_asyncio.fixture
async def resource_allocator() -> AsyncGenerator[ResourceAllocator, None]:
    """Create a resource allocator."""
    allocator = ResourceAllocator()
    yield allocator


@pytest_asyncio.fixture
async def gpu_manager() -> AsyncGenerator[GPUManager, None]:
    """Create a GPU manager."""
    manager = GPUManager()
    yield manager


@pytest_asyncio.fixture
async def gpu_pool() -> AsyncGenerator[GPUPool, None]:
    """Create a GPU pool."""
    pool = GPUPool("test-pool")
    yield pool


@pytest_asyncio.fixture
async def checkpoint_manager(tmp_path) -> AsyncGenerator[CheckpointManager, None]:
    """Create a checkpoint manager with temp storage."""
    storage = LocalStorage(str(tmp_path / "checkpoints"))
    manager = CheckpointManager(default_storage=storage, local_path=str(tmp_path / "checkpoints"))
    yield manager


@pytest_asyncio.fixture
async def experiment_tracker() -> AsyncGenerator[ExperimentTracker, None]:
    """Create an experiment tracker."""
    tracker = ExperimentTracker()
    yield tracker


@pytest.fixture
def make_job(sample_job_config: JobConfig):
    """Factory to create test jobs."""
    def _make_job(
        name: str = "test-job",
        user_id: str = "user-123",
        priority: JobPriority = JobPriority.NORMAL,
        status: JobStatus = JobStatus.PENDING,
        gpus: int = 0,
    ) -> TrainingJob:
        return TrainingJob(
            name=name,
            user_id=user_id,
            config=sample_job_config,
            resources=ResourceRequest(cpus=2, memory_gb=8.0, gpus=gpus),
            priority=priority,
            status=status,
        )
    return _make_job


@pytest.fixture
def make_worker():
    """Factory to create test workers."""
    def _make_worker(
        hostname: str = "worker",
        cpus: int = 16,
        memory_gb: float = 64.0,
        gpus: int = 2,
        status: WorkerStatus = WorkerStatus.READY,
    ) -> WorkerInfo:
        return WorkerInfo(
            hostname=hostname,
            ip_address=f"192.168.1.{hash(hostname) % 255}",
            resources=ResourceRequest(
                cpus=cpus,
                memory_gb=memory_gb,
                gpus=gpus,
            ),
            status=status,
        )
    return _make_worker
