"""Core data models for ML Training Orchestrator."""

from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


class JobStatus(str, Enum):
    """Status of a training job."""

    PENDING = "pending"  # Job submitted, awaiting validation
    QUEUED = "queued"  # Job validated, waiting for resources
    SCHEDULED = "scheduled"  # Resources allocated, waiting to start
    STARTING = "starting"  # Job is being initialized
    RUNNING = "running"  # Job is actively training
    PAUSED = "paused"  # Job paused by user
    CHECKPOINTING = "checkpointing"  # Job is saving checkpoint
    PREEMPTED = "preempted"  # Job preempted for higher priority
    COMPLETED = "completed"  # Job finished successfully
    FAILED = "failed"  # Job failed with error
    CANCELLED = "cancelled"  # Job cancelled by user
    TIMEOUT = "timeout"  # Job exceeded time limit


class JobPriority(int, Enum):
    """Priority levels for training jobs."""

    LOWEST = 0
    LOW = 25
    NORMAL = 50
    HIGH = 75
    HIGHEST = 100
    CRITICAL = 150  # System-level priority, bypasses normal scheduling


class MetricType(str, Enum):
    """Type of metric being tracked."""

    SCALAR = "scalar"
    HISTOGRAM = "histogram"
    IMAGE = "image"
    TEXT = "text"
    ARTIFACT = "artifact"


class CheckpointStatus(str, Enum):
    """Status of a checkpoint."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    DELETED = "deleted"


class CheckpointPolicy(str, Enum):
    """Policy for when to create checkpoints."""

    PERIODIC = "periodic"  # Every N epochs/steps
    BEST_METRIC = "best_metric"  # When metric improves
    ON_PREEMPTION = "on_preemption"  # When job is preempted
    ON_FAILURE = "on_failure"  # When job fails
    MANUAL = "manual"  # User-triggered only


class WorkerStatus(str, Enum):
    """Status of a worker node."""

    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    DRAINING = "draining"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


class StorageBackend(str, Enum):
    """Storage backend for checkpoints and artifacts."""

    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"


class SchedulingPolicy(str, Enum):
    """Scheduling policy for jobs."""

    FIFO = "fifo"
    FAIR_SHARE = "fair_share"
    PRIORITY = "priority"
    GANG = "gang"
    BACKFILL = "backfill"


class AllocationStrategy(str, Enum):
    """Strategy for resource allocation."""

    FIRST_FIT = "first_fit"
    BEST_FIT = "best_fit"
    WORST_FIT = "worst_fit"
    BIN_PACKING = "bin_packing"


class ResourceRequest(BaseModel):
    """Resource requirements for a training job."""

    cpus: int = Field(default=1, ge=0, le=256, description="Number of CPU cores")
    memory_gb: float = Field(default=4.0, ge=0, le=2048, description="Memory in GB")
    gpus: int = Field(default=0, ge=0, le=16, description="Number of GPUs")
    gpu_type: Optional[str] = Field(default=None, description="Specific GPU type (e.g., 'A100')")
    gpu_memory_gb: Optional[float] = Field(default=None, ge=0, description="GPU memory per GPU")
    network_bandwidth_gbps: Optional[float] = Field(
        default=None, ge=0, description="Network bandwidth requirement"
    )
    storage_gb: float = Field(default=10.0, ge=0, description="Storage space in GB")
    shared_memory_gb: Optional[float] = Field(
        default=None, ge=0, description="Shared memory requirement"
    )

    @field_validator("gpu_type")
    @classmethod
    def validate_gpu_type(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            valid_types = {
                "A100",
                "A100-80GB",
                "H100",
                "H100-80GB",
                "V100",
                "V100-32GB",
                "T4",
                "L4",
                "A10G",
                "RTX3090",
                "RTX4090",
            }
            if v.upper() not in {t.upper() for t in valid_types}:
                # Allow unknown types for flexibility
                pass
        return v

    def fits_in(self, available: "ResourceRequest") -> bool:
        """Check if this request fits in available resources."""
        if self.cpus > available.cpus:
            return False
        if self.memory_gb > available.memory_gb:
            return False
        if self.gpus > available.gpus:
            return False
        if self.storage_gb > available.storage_gb:
            return False
        if self.gpu_type and available.gpu_type and self.gpu_type != available.gpu_type:
            return False
        if self.gpu_memory_gb and available.gpu_memory_gb:
            if self.gpu_memory_gb > available.gpu_memory_gb:
                return False
        return True

    def subtract(self, other: "ResourceRequest") -> "ResourceRequest":
        """Subtract resources (for tracking remaining capacity)."""
        return ResourceRequest(
            cpus=self.cpus - other.cpus,
            memory_gb=self.memory_gb - other.memory_gb,
            gpus=self.gpus - other.gpus,
            gpu_type=self.gpu_type,
            gpu_memory_gb=self.gpu_memory_gb,
            storage_gb=self.storage_gb - other.storage_gb,
        )

    def add(self, other: "ResourceRequest") -> "ResourceRequest":
        """Add resources (for tracking total capacity)."""
        return ResourceRequest(
            cpus=self.cpus + other.cpus,
            memory_gb=self.memory_gb + other.memory_gb,
            gpus=self.gpus + other.gpus,
            gpu_type=self.gpu_type or other.gpu_type,
            gpu_memory_gb=max(self.gpu_memory_gb or 0, other.gpu_memory_gb or 0) or None,
            storage_gb=self.storage_gb + other.storage_gb,
        )


class DistributedConfig(BaseModel):
    """Configuration for distributed training."""

    enabled: bool = Field(default=False, description="Enable distributed training")
    world_size: int = Field(default=1, ge=1, description="Total number of workers")
    num_nodes: int = Field(default=1, ge=1, description="Number of nodes")
    gpus_per_node: int = Field(default=1, ge=1, description="GPUs per node")
    backend: str = Field(default="nccl", description="Communication backend")
    master_addr: Optional[str] = Field(default=None, description="Master node address")
    master_port: int = Field(default=29500, ge=1024, le=65535, description="Master port")
    rdzv_backend: str = Field(default="c10d", description="Rendezvous backend")
    elastic: bool = Field(default=False, description="Enable elastic training")
    min_nodes: Optional[int] = Field(default=None, ge=1, description="Minimum nodes for elastic")
    max_nodes: Optional[int] = Field(default=None, ge=1, description="Maximum nodes for elastic")

    @model_validator(mode="after")
    def validate_elastic_config(self) -> "DistributedConfig":
        if self.elastic:
            if self.min_nodes is None:
                self.min_nodes = 1
            if self.max_nodes is None:
                # Use world_size / gpus_per_node as max_nodes, fallback to num_nodes
                self.max_nodes = max(self.num_nodes, self.world_size // self.gpus_per_node)
            if self.min_nodes > self.max_nodes:
                raise ValueError("min_nodes cannot be greater than max_nodes")
        return self


class RetryConfig(BaseModel):
    """Configuration for job retry behavior."""

    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    retry_delay_seconds: int = Field(default=60, ge=0, description="Delay between retries")
    exponential_backoff: bool = Field(default=True, description="Use exponential backoff")
    max_delay_seconds: int = Field(default=3600, ge=60, description="Maximum delay")
    retry_on_preemption: bool = Field(default=True, description="Retry when preempted")


class CheckpointConfig(BaseModel):
    """Configuration for checkpointing."""

    enabled: bool = Field(default=True, description="Enable checkpointing")
    policy: CheckpointPolicy = Field(default=CheckpointPolicy.PERIODIC)
    interval_epochs: Optional[int] = Field(default=1, ge=1, description="Checkpoint every N epochs")
    interval_steps: Optional[int] = Field(default=None, ge=1, description="Checkpoint every N steps")
    keep_last_n: int = Field(default=3, ge=1, description="Keep last N checkpoints")
    storage_backend: StorageBackend = Field(default=StorageBackend.LOCAL)
    storage_path: str = Field(default="/checkpoints", description="Storage path/bucket")
    save_optimizer: bool = Field(default=True, description="Save optimizer state")
    save_scheduler: bool = Field(default=True, description="Save LR scheduler state")
    metric_to_track: Optional[str] = Field(default=None, description="Metric for best model")
    metric_mode: str = Field(default="min", description="'min' or 'max' for metric")


class JobConfig(BaseModel):
    """Configuration for a training job."""

    # Training script
    script_path: str = Field(..., min_length=1, description="Path to training script")
    script_args: list[str] = Field(default_factory=list, description="Script arguments")
    environment: dict[str, str] = Field(default_factory=dict, description="Environment variables")

    # Training parameters
    epochs: Optional[int] = Field(default=None, ge=1, description="Number of epochs")
    max_steps: Optional[int] = Field(default=None, ge=1, description="Maximum training steps")
    batch_size: Optional[int] = Field(default=None, ge=1, description="Batch size")
    learning_rate: Optional[float] = Field(default=None, gt=0, description="Learning rate")

    # Timeouts
    timeout_hours: Optional[float] = Field(default=None, ge=0.1, description="Job timeout")
    startup_timeout_minutes: int = Field(default=10, ge=1, description="Startup timeout")

    # Dependencies
    pip_packages: list[str] = Field(default_factory=list, description="Pip packages to install")
    conda_env: Optional[str] = Field(default=None, description="Conda environment name")
    docker_image: Optional[str] = Field(default=None, description="Docker image to use")

    # Distributed training
    distributed: DistributedConfig = Field(default_factory=DistributedConfig)

    # Checkpointing
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)

    # Retry
    retry: RetryConfig = Field(default_factory=RetryConfig)

    # Tags and metadata
    tags: dict[str, str] = Field(default_factory=dict, description="Job tags")
    description: Optional[str] = Field(default=None, description="Job description")

    @model_validator(mode="after")
    def validate_training_limit(self) -> "JobConfig":
        # Check mutual exclusivity of epochs and max_steps
        if self.epochs is not None and self.max_steps is not None:
            raise ValueError("Cannot specify both epochs and max_steps")
        if self.epochs is None and self.max_steps is None and self.timeout_hours is None:
            # Default to 24 hour timeout if no limit specified
            self.timeout_hours = 24.0
        return self


class MetricValue(BaseModel):
    """A metric value with metadata."""

    name: str = Field(..., min_length=1, description="Metric name")
    value: float = Field(..., description="Metric value")
    step: Optional[int] = Field(default=None, ge=0, description="Training step")
    epoch: Optional[int] = Field(default=None, ge=0, description="Training epoch")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metric_type: MetricType = Field(default=MetricType.SCALAR)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Checkpoint(BaseModel):
    """A training checkpoint."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Checkpoint ID")
    job_id: str = Field(..., description="Associated job ID")
    epoch: int = Field(..., ge=0, description="Epoch number")
    step: int = Field(..., ge=0, description="Global step")
    path: str = Field(..., description="Storage path")
    size_bytes: int = Field(default=0, ge=0, description="Checkpoint size")
    status: CheckpointStatus = Field(default=CheckpointStatus.PENDING)
    metrics: dict[str, float] = Field(default_factory=dict, description="Metrics at checkpoint")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def is_valid(self) -> bool:
        """Check if checkpoint is valid and usable."""
        return self.status == CheckpointStatus.COMPLETED and self.size_bytes > 0


class WorkerInfo(BaseModel):
    """Information about a worker node."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Worker ID")
    hostname: str = Field(..., description="Worker hostname")
    ip_address: str = Field(..., description="Worker IP address")
    port: int = Field(default=8000, ge=1024, le=65535, description="Worker port")
    status: WorkerStatus = Field(default=WorkerStatus.INITIALIZING)
    resources: ResourceRequest = Field(default_factory=ResourceRequest)
    allocated_resources: ResourceRequest = Field(
        default_factory=lambda: ResourceRequest(cpus=0, memory_gb=0, gpus=0, storage_gb=0)
    )
    current_jobs: list[str] = Field(default_factory=list, description="Running job IDs")
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)
    registered_at: datetime = Field(default_factory=datetime.utcnow)
    labels: dict[str, str] = Field(default_factory=dict, description="Node labels")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def available_resources(self) -> ResourceRequest:
        """Calculate available resources."""
        return self.resources.subtract(self.allocated_resources)

    def is_healthy(self, timeout_seconds: int = 60) -> bool:
        """Check if worker is healthy based on heartbeat."""
        if self.status in (WorkerStatus.UNHEALTHY, WorkerStatus.OFFLINE):
            return False
        age = datetime.utcnow() - self.last_heartbeat
        return age < timedelta(seconds=timeout_seconds)

    def can_run_job(self, resources: ResourceRequest) -> bool:
        """Check if worker can run a job with given resources."""
        if self.status not in (WorkerStatus.READY, WorkerStatus.BUSY):
            return False
        return resources.fits_in(self.available_resources)


class TrainingJob(BaseModel):
    """A training job."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Job ID")
    name: str = Field(..., min_length=1, max_length=256, description="Job name")
    user_id: str = Field(..., description="User who submitted the job")
    team_id: Optional[str] = Field(default=None, description="Team the job belongs to")
    experiment_id: Optional[str] = Field(default=None, description="Associated experiment")

    # Configuration
    config: JobConfig = Field(..., description="Job configuration")
    resources: ResourceRequest = Field(default_factory=ResourceRequest)

    # Status and scheduling
    status: JobStatus = Field(default=JobStatus.PENDING)
    priority: JobPriority = Field(default=JobPriority.NORMAL)
    preemptible: bool = Field(default=True, description="Can be preempted")

    # Assigned resources
    assigned_workers: list[str] = Field(default_factory=list, description="Assigned worker IDs")
    assigned_gpus: list[str] = Field(default_factory=list, description="Assigned GPU IDs")

    # Progress
    current_epoch: int = Field(default=0, ge=0)
    current_step: int = Field(default=0, ge=0)
    progress_percent: float = Field(default=0.0, ge=0, le=100)

    # Checkpoints
    checkpoints: list[Checkpoint] = Field(default_factory=list)
    latest_checkpoint_id: Optional[str] = Field(default=None)
    resume_from_checkpoint: Optional[str] = Field(default=None)

    # Metrics
    metrics: list[MetricValue] = Field(default_factory=list)
    best_metrics: dict[str, float] = Field(default_factory=dict)

    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    queued_at: Optional[datetime] = Field(default=None)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    paused_at: Optional[datetime] = Field(default=None)

    # Error handling
    retry_count: int = Field(default=0, ge=0)
    error_message: Optional[str] = Field(default=None)
    error_traceback: Optional[str] = Field(default=None)

    # Metadata
    tags: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        """Check if job is in a terminal state."""
        return self.status in (
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.TIMEOUT,
        )

    @property
    def is_active(self) -> bool:
        """Check if job is actively running."""
        return self.status in (JobStatus.RUNNING, JobStatus.CHECKPOINTING)

    @property
    def is_schedulable(self) -> bool:
        """Check if job can be scheduled."""
        return self.status in (JobStatus.QUEUED, JobStatus.PREEMPTED)

    @property
    def queue_time(self) -> Optional[timedelta]:
        """Time spent in queue."""
        if self.queued_at:
            end = self.started_at or datetime.utcnow()
            return end - self.queued_at
        return None

    @property
    def run_time(self) -> Optional[timedelta]:
        """Time spent running."""
        if self.started_at:
            end = self.completed_at or datetime.utcnow()
            return end - self.started_at
        return None

    @property
    def total_time(self) -> Optional[timedelta]:
        """Total time since creation."""
        end = self.completed_at or datetime.utcnow()
        return end - self.created_at

    def get_latest_checkpoint(self) -> Optional[Checkpoint]:
        """Get the latest valid checkpoint."""
        valid_checkpoints = [c for c in self.checkpoints if c.is_valid()]
        if not valid_checkpoints:
            return None
        return max(valid_checkpoints, key=lambda c: (c.epoch, c.step))

    def get_best_checkpoint(self, metric: str, mode: str = "min") -> Optional[Checkpoint]:
        """Get the checkpoint with best metric value."""
        valid_checkpoints = [
            c for c in self.checkpoints if c.is_valid() and metric in c.metrics
        ]
        if not valid_checkpoints:
            return None
        if mode == "min":
            return min(valid_checkpoints, key=lambda c: c.metrics[metric])
        return max(valid_checkpoints, key=lambda c: c.metrics[metric])

    def add_metric(self, metric: MetricValue) -> None:
        """Add a metric value and update best metrics."""
        self.metrics.append(metric)
        if metric.name not in self.best_metrics:
            self.best_metrics[metric.name] = metric.value
        else:
            # Assume lower is better for loss, higher for accuracy
            if "loss" in metric.name.lower() or "error" in metric.name.lower():
                self.best_metrics[metric.name] = min(
                    self.best_metrics[metric.name], metric.value
                )
            else:
                self.best_metrics[metric.name] = max(
                    self.best_metrics[metric.name], metric.value
                )


class ResourceQuota(BaseModel):
    """Resource quota for a user or team."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    entity_id: str = Field(..., description="User or team ID")
    entity_type: str = Field(..., description="'user' or 'team'")

    # Limits
    max_cpus: int = Field(default=100, ge=1)
    max_memory_gb: float = Field(default=500.0, ge=1)
    max_gpus: int = Field(default=8, ge=0)
    max_concurrent_jobs: int = Field(default=10, ge=1)
    max_job_duration_hours: float = Field(default=168, ge=1)  # 1 week default

    # Current usage
    used_cpus: int = Field(default=0, ge=0)
    used_memory_gb: float = Field(default=0.0, ge=0)
    used_gpus: int = Field(default=0, ge=0)
    active_jobs: int = Field(default=0, ge=0)

    # Priority
    priority_boost: int = Field(default=0, description="Priority boost for this entity")

    def can_allocate(self, resources: ResourceRequest) -> bool:
        """Check if resources can be allocated within quota."""
        if self.used_cpus + resources.cpus > self.max_cpus:
            return False
        if self.used_memory_gb + resources.memory_gb > self.max_memory_gb:
            return False
        if self.used_gpus + resources.gpus > self.max_gpus:
            return False
        if self.active_jobs >= self.max_concurrent_jobs:
            return False
        return True

    def allocate(self, resources: ResourceRequest) -> None:
        """Allocate resources."""
        self.used_cpus += resources.cpus
        self.used_memory_gb += resources.memory_gb
        self.used_gpus += resources.gpus
        self.active_jobs += 1

    def release(self, resources: ResourceRequest) -> None:
        """Release resources."""
        self.used_cpus = max(0, self.used_cpus - resources.cpus)
        self.used_memory_gb = max(0, self.used_memory_gb - resources.memory_gb)
        self.used_gpus = max(0, self.used_gpus - resources.gpus)
        self.active_jobs = max(0, self.active_jobs - 1)


class Experiment(BaseModel):
    """An experiment tracking multiple runs."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., min_length=1, max_length=256)
    user_id: str = Field(...)
    team_id: Optional[str] = Field(default=None)
    description: Optional[str] = Field(default=None)

    # Configuration
    base_config: dict[str, Any] = Field(default_factory=dict)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)

    # Runs
    job_ids: list[str] = Field(default_factory=list)

    # Status
    status: str = Field(default="active")  # active, completed, archived
    best_job_id: Optional[str] = Field(default=None)
    best_metric_value: Optional[float] = Field(default=None)
    best_metric_name: Optional[str] = Field(default=None)

    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Tags and metadata
    tags: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
