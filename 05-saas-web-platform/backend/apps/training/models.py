"""
Models for distributed ML training jobs.
"""
import uuid
from django.db import models
from django.conf import settings
from django.utils import timezone
from apps.tenants.models import Tenant
from apps.resources.models import ResourceAllocation


class TrainingFramework(models.TextChoices):
    """Supported ML training frameworks."""
    PYTORCH = 'pytorch', 'PyTorch'
    TENSORFLOW = 'tensorflow', 'TensorFlow'
    JAX = 'jax', 'JAX'
    DEEPSPEED = 'deepspeed', 'DeepSpeed'
    MEGATRON = 'megatron', 'Megatron-LM'


class DistributedStrategy(models.TextChoices):
    """Distributed training strategies."""
    DATA_PARALLEL = 'data_parallel', 'Data Parallel (DDP)'
    MODEL_PARALLEL = 'model_parallel', 'Model Parallel'
    PIPELINE_PARALLEL = 'pipeline_parallel', 'Pipeline Parallel'
    TENSOR_PARALLEL = 'tensor_parallel', 'Tensor Parallel'
    FSDP = 'fsdp', 'Fully Sharded Data Parallel'
    ZERO = 'zero', 'ZeRO Optimization'


class JobStatus(models.TextChoices):
    """Training job status."""
    PENDING = 'pending', 'Pending'
    QUEUED = 'queued', 'Queued'
    INITIALIZING = 'initializing', 'Initializing'
    RUNNING = 'running', 'Running'
    CHECKPOINTING = 'checkpointing', 'Checkpointing'
    COMPLETED = 'completed', 'Completed'
    FAILED = 'failed', 'Failed'
    CANCELLED = 'cancelled', 'Cancelled'
    PREEMPTED = 'preempted', 'Preempted'


class TrainingJob(models.Model):
    """Distributed training job configuration."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(
        Tenant,
        on_delete=models.CASCADE,
        related_name='training_jobs'
    )
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)

    # Framework and strategy
    framework = models.CharField(
        max_length=20,
        choices=TrainingFramework.choices,
        default=TrainingFramework.PYTORCH
    )
    strategy = models.CharField(
        max_length=20,
        choices=DistributedStrategy.choices,
        default=DistributedStrategy.DATA_PARALLEL
    )

    # Code and environment
    code_repository = models.URLField(blank=True)
    code_branch = models.CharField(max_length=100, default='main')
    code_commit = models.CharField(max_length=40, blank=True)
    docker_image = models.CharField(max_length=255, blank=True)
    entry_point = models.CharField(max_length=500)
    arguments = models.JSONField(default=dict)
    environment = models.JSONField(default=dict)

    # Resource requirements
    gpu_count = models.IntegerField(default=1)
    gpu_type = models.CharField(max_length=50, blank=True)
    nodes_count = models.IntegerField(default=1)
    memory_gb = models.IntegerField(default=64)
    storage_gb = models.IntegerField(default=100)

    # Distributed configuration
    world_size = models.IntegerField(default=1)
    gradient_accumulation_steps = models.IntegerField(default=1)
    mixed_precision = models.CharField(
        max_length=10,
        choices=[('fp32', 'FP32'), ('fp16', 'FP16'), ('bf16', 'BF16')],
        default='fp16'
    )

    # DeepSpeed / FSDP config
    distributed_config = models.JSONField(default=dict)

    # Status
    status = models.CharField(
        max_length=20,
        choices=JobStatus.choices,
        default=JobStatus.PENDING
    )
    priority = models.IntegerField(default=5)
    preemptible = models.BooleanField(default=False)

    # Execution
    allocation = models.ForeignKey(
        ResourceAllocation,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='training_jobs'
    )
    worker_hosts = models.JSONField(default=list)
    master_addr = models.CharField(max_length=255, blank=True)
    master_port = models.IntegerField(default=29500)

    # Progress
    current_epoch = models.IntegerField(default=0)
    total_epochs = models.IntegerField(default=1)
    current_step = models.IntegerField(default=0)
    total_steps = models.IntegerField(null=True, blank=True)
    progress_percent = models.FloatField(default=0)

    # Metrics
    metrics = models.JSONField(default=dict)
    last_loss = models.FloatField(null=True, blank=True)
    best_loss = models.FloatField(null=True, blank=True)
    learning_rate = models.FloatField(null=True, blank=True)
    throughput_samples_sec = models.FloatField(null=True, blank=True)

    # Checkpointing
    checkpoint_enabled = models.BooleanField(default=True)
    checkpoint_interval_steps = models.IntegerField(default=1000)
    checkpoint_path = models.CharField(max_length=500, blank=True)
    latest_checkpoint = models.CharField(max_length=500, blank=True)
    resume_from_checkpoint = models.CharField(max_length=500, blank=True)

    # Timing
    submitted_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    estimated_completion = models.DateTimeField(null=True, blank=True)
    max_runtime_hours = models.IntegerField(default=24)

    # Output
    output_path = models.CharField(max_length=500, blank=True)
    logs_path = models.CharField(max_length=500, blank=True)
    tensorboard_path = models.CharField(max_length=500, blank=True)

    # Error handling
    error_message = models.TextField(blank=True)
    retry_count = models.IntegerField(default=0)
    max_retries = models.IntegerField(default=3)

    # Ownership
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name='created_training_jobs'
    )
    tags = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'training_jobs'
        ordering = ['-priority', '-submitted_at']
        indexes = [
            models.Index(fields=['tenant', 'status']),
            models.Index(fields=['status', 'priority']),
            models.Index(fields=['framework', 'status']),
        ]

    def __str__(self):
        return f"{self.name} ({self.framework})"

    @property
    def duration_hours(self):
        if not self.started_at:
            return 0
        end = self.completed_at or timezone.now()
        return (end - self.started_at).total_seconds() / 3600

    @property
    def is_distributed(self):
        return self.world_size > 1 or self.nodes_count > 1


class TrainingRun(models.Model):
    """Individual training run/attempt within a job."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    job = models.ForeignKey(
        TrainingJob,
        on_delete=models.CASCADE,
        related_name='runs'
    )
    run_number = models.IntegerField()

    # Status
    status = models.CharField(
        max_length=20,
        choices=JobStatus.choices,
        default=JobStatus.PENDING
    )

    # Timing
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    # Results
    final_loss = models.FloatField(null=True, blank=True)
    final_metrics = models.JSONField(default=dict)
    exit_code = models.IntegerField(null=True, blank=True)
    error_message = models.TextField(blank=True)
    logs = models.TextField(blank=True)

    # Checkpoint
    final_checkpoint = models.CharField(max_length=500, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'training_runs'
        ordering = ['job', 'run_number']
        unique_together = ['job', 'run_number']

    def __str__(self):
        return f"{self.job.name} - Run {self.run_number}"


class Experiment(models.Model):
    """Group of related training jobs for experiment tracking."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(
        Tenant,
        on_delete=models.CASCADE,
        related_name='experiments'
    )
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)

    # Configuration
    base_config = models.JSONField(default=dict)
    hyperparameters = models.JSONField(default=dict)

    # Status
    is_active = models.BooleanField(default=True)

    # Best result tracking
    best_job = models.ForeignKey(
        TrainingJob,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='+'
    )
    best_metric_name = models.CharField(max_length=100, default='loss')
    best_metric_value = models.FloatField(null=True, blank=True)
    best_metric_minimize = models.BooleanField(default=True)

    # Ownership
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True
    )
    tags = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'experiments'
        ordering = ['-created_at']

    def __str__(self):
        return self.name


class HyperparameterSweep(models.Model):
    """Hyperparameter sweep configuration."""

    class SearchStrategy(models.TextChoices):
        GRID = 'grid', 'Grid Search'
        RANDOM = 'random', 'Random Search'
        BAYESIAN = 'bayesian', 'Bayesian Optimization'
        HYPERBAND = 'hyperband', 'Hyperband'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    experiment = models.ForeignKey(
        Experiment,
        on_delete=models.CASCADE,
        related_name='sweeps'
    )
    name = models.CharField(max_length=255)

    # Search configuration
    strategy = models.CharField(
        max_length=20,
        choices=SearchStrategy.choices,
        default=SearchStrategy.RANDOM
    )
    parameter_space = models.JSONField()
    objective_metric = models.CharField(max_length=100, default='loss')
    objective_minimize = models.BooleanField(default=True)

    # Limits
    max_trials = models.IntegerField(default=10)
    max_parallel_trials = models.IntegerField(default=2)
    early_stopping_patience = models.IntegerField(default=5)

    # Progress
    completed_trials = models.IntegerField(default=0)
    best_trial_params = models.JSONField(null=True, blank=True)
    best_trial_value = models.FloatField(null=True, blank=True)

    # Status
    is_active = models.BooleanField(default=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'hyperparameter_sweeps'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} ({self.strategy})"


class ModelArtifact(models.Model):
    """Trained model artifact."""

    class ArtifactType(models.TextChoices):
        CHECKPOINT = 'checkpoint', 'Checkpoint'
        FINAL_MODEL = 'final', 'Final Model'
        ONNX = 'onnx', 'ONNX Export'
        TENSORRT = 'tensorrt', 'TensorRT'
        TORCHSCRIPT = 'torchscript', 'TorchScript'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    job = models.ForeignKey(
        TrainingJob,
        on_delete=models.CASCADE,
        related_name='artifacts'
    )
    name = models.CharField(max_length=255)

    # Type and path
    artifact_type = models.CharField(
        max_length=20,
        choices=ArtifactType.choices,
        default=ArtifactType.CHECKPOINT
    )
    file_path = models.CharField(max_length=500)
    file_size_bytes = models.BigIntegerField(null=True, blank=True)
    checksum = models.CharField(max_length=64, blank=True)

    # Metrics at time of save
    step = models.IntegerField(null=True, blank=True)
    epoch = models.IntegerField(null=True, blank=True)
    metrics = models.JSONField(default=dict)

    # Model info
    model_architecture = models.CharField(max_length=255, blank=True)
    parameter_count = models.BigIntegerField(null=True, blank=True)
    framework_version = models.CharField(max_length=50, blank=True)

    # Status
    is_best = models.BooleanField(default=False)
    is_final = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'model_artifacts'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} ({self.artifact_type})"
