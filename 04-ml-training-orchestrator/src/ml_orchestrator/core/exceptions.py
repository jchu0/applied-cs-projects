"""Custom exceptions for ML Training Orchestrator."""

from typing import Any, Optional


class OrchestratorError(Exception):
    """Base exception for all orchestrator errors."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


class JobNotFoundError(OrchestratorError):
    """Raised when a job cannot be found."""

    def __init__(self, job_id: str):
        super().__init__(f"Job not found: {job_id}", {"job_id": job_id})
        self.job_id = job_id


class JobStateError(OrchestratorError):
    """Raised when a job state transition is invalid."""

    def __init__(
        self,
        job_id: str,
        current_state: str,
        target_state: str,
        message: Optional[str] = None,
    ):
        msg = message or f"Invalid state transition from {current_state} to {target_state}"
        super().__init__(
            msg,
            {
                "job_id": job_id,
                "current_state": current_state,
                "target_state": target_state,
            },
        )
        self.job_id = job_id
        self.current_state = current_state
        self.target_state = target_state


class ResourceError(OrchestratorError):
    """Raised when there's a resource allocation issue."""

    def __init__(self, message: str, resource_type: Optional[str] = None):
        super().__init__(message, {"resource_type": resource_type} if resource_type else None)
        self.resource_type = resource_type


class ResourceExhaustedError(ResourceError):
    """Raised when resources are exhausted."""

    def __init__(self, resource_type: str, requested: float, available: float):
        super().__init__(
            f"Insufficient {resource_type}: requested {requested}, available {available}",
            resource_type,
        )
        self.requested = requested
        self.available = available


class QuotaExceededError(ResourceError):
    """Raised when user/team quota is exceeded."""

    def __init__(self, entity: str, resource_type: str, quota: float, used: float):
        super().__init__(
            f"Quota exceeded for {entity}: {resource_type} quota {quota}, used {used}",
            resource_type,
        )
        self.entity = entity
        self.quota = quota
        self.used = used


class GPUNotAvailableError(ResourceError):
    """Raised when requested GPU type is not available."""

    def __init__(self, gpu_type: str, count: int):
        super().__init__(
            f"GPU not available: {count}x {gpu_type}",
            "gpu",
        )
        self.gpu_type = gpu_type
        self.count = count


class CheckpointError(OrchestratorError):
    """Raised when there's a checkpoint operation issue."""

    def __init__(self, message: str, checkpoint_id: Optional[str] = None):
        super().__init__(
            message, {"checkpoint_id": checkpoint_id} if checkpoint_id else None
        )
        self.checkpoint_id = checkpoint_id


class CheckpointNotFoundError(CheckpointError):
    """Raised when a checkpoint cannot be found."""

    def __init__(self, checkpoint_id: str):
        super().__init__(f"Checkpoint not found: {checkpoint_id}", checkpoint_id)


class CheckpointCorruptedError(CheckpointError):
    """Raised when a checkpoint is corrupted."""

    def __init__(self, checkpoint_id: str, reason: str):
        super().__init__(
            f"Checkpoint corrupted: {checkpoint_id} - {reason}",
            checkpoint_id,
        )
        self.reason = reason


class SchedulingError(OrchestratorError):
    """Raised when there's a scheduling issue."""

    pass


class PreemptionError(SchedulingError):
    """Raised when job preemption fails."""

    def __init__(self, job_id: str, reason: str):
        super().__init__(
            f"Failed to preempt job {job_id}: {reason}",
            {"job_id": job_id, "reason": reason},
        )
        self.job_id = job_id
        self.reason = reason


class WorkerError(OrchestratorError):
    """Raised when there's a worker-related issue."""

    def __init__(self, message: str, worker_id: Optional[str] = None):
        super().__init__(message, {"worker_id": worker_id} if worker_id else None)
        self.worker_id = worker_id


class WorkerNotFoundError(WorkerError):
    """Raised when a worker cannot be found."""

    def __init__(self, worker_id: str):
        super().__init__(f"Worker not found: {worker_id}", worker_id)


class WorkerUnhealthyError(WorkerError):
    """Raised when a worker is unhealthy."""

    def __init__(self, worker_id: str, last_heartbeat: str):
        super().__init__(
            f"Worker unhealthy: {worker_id}, last heartbeat: {last_heartbeat}",
            worker_id,
        )
        self.last_heartbeat = last_heartbeat


class ExperimentError(OrchestratorError):
    """Raised when there's an experiment tracking issue."""

    def __init__(self, message: str, experiment_id: Optional[str] = None):
        super().__init__(
            message, {"experiment_id": experiment_id} if experiment_id else None
        )
        self.experiment_id = experiment_id


class ExperimentNotFoundError(ExperimentError):
    """Raised when an experiment cannot be found."""

    def __init__(self, experiment_id: str):
        super().__init__(f"Experiment not found: {experiment_id}", experiment_id)


class ConfigurationError(OrchestratorError):
    """Raised when there's a configuration issue."""

    pass


class ValidationError(OrchestratorError):
    """Raised when validation fails."""

    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(message, {"field": field} if field else None)
        self.field = field
