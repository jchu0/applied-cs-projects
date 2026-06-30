"""
ML Training Orchestrator

A production-grade distributed ML training orchestration platform.
"""

__version__ = "0.1.0"

from ml_orchestrator.core.models import (
    Checkpoint,
    JobConfig,
    JobPriority,
    JobStatus,
    MetricType,
    MetricValue,
    ResourceRequest,
    TrainingJob,
)
from ml_orchestrator.core.job_manager import JobManager
from ml_orchestrator.scheduling.scheduler import Scheduler
from ml_orchestrator.resources.allocator import ResourceAllocator
from ml_orchestrator.resources.gpu_manager import GPUManager
from ml_orchestrator.checkpoint.manager import CheckpointManager
from ml_orchestrator.experiment.tracker import ExperimentTracker

__all__ = [
    # Core models
    "TrainingJob",
    "JobConfig",
    "JobStatus",
    "JobPriority",
    "ResourceRequest",
    "Checkpoint",
    "MetricValue",
    "MetricType",
    # Managers
    "JobManager",
    "Scheduler",
    "ResourceAllocator",
    "GPUManager",
    "CheckpointManager",
    "ExperimentTracker",
]
