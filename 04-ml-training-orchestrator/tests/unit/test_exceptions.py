"""Tests for the custom exception hierarchy."""

from ml_orchestrator.core.exceptions import (
    OrchestratorError,
    JobNotFoundError,
    JobStateError,
    ResourceError,
    ResourceExhaustedError,
    QuotaExceededError,
    GPUNotAvailableError,
    CheckpointError,
    CheckpointNotFoundError,
    CheckpointCorruptedError,
    SchedulingError,
    PreemptionError,
    WorkerError,
    WorkerNotFoundError,
    WorkerUnhealthyError,
    ExperimentError,
    ExperimentNotFoundError,
    ConfigurationError,
    ValidationError,
)


def test_base_error_message_and_details():
    err = OrchestratorError("boom", {"k": "v"})
    assert err.message == "boom"
    assert err.details == {"k": "v"}
    assert "boom" in str(err)
    assert "Details" in str(err)  # details rendered when present


def test_base_error_without_details():
    err = OrchestratorError("boom")
    assert err.details == {}
    assert str(err) == "boom"


def test_job_not_found():
    err = JobNotFoundError("job-1")
    assert isinstance(err, OrchestratorError)
    assert err.job_id == "job-1"
    assert err.details["job_id"] == "job-1"


def test_job_state_error():
    err = JobStateError("job-1", "running", "pending")
    assert err.current_state == "running"
    assert err.target_state == "pending"
    assert err.details["job_id"] == "job-1"
    custom = JobStateError("j", "a", "b", message="nope")
    assert "nope" in str(custom)


def test_resource_errors():
    base = ResourceError("bad", resource_type="cpu")
    assert base.resource_type == "cpu"
    assert base.details["resource_type"] == "cpu"

    exhausted = ResourceExhaustedError("gpu", requested=4, available=1)
    assert isinstance(exhausted, ResourceError)
    assert exhausted.requested == 4 and exhausted.available == 1

    quota = QuotaExceededError("team-a", "gpu", quota=10, used=12)
    assert quota.entity == "team-a" and quota.used == 12

    gpu = GPUNotAvailableError("A100", 8)
    assert gpu.gpu_type == "A100" and gpu.count == 8
    assert gpu.resource_type == "gpu"


def test_checkpoint_errors():
    base = CheckpointError("oops", checkpoint_id="ckpt-1")
    assert base.checkpoint_id == "ckpt-1"
    nf = CheckpointNotFoundError("ckpt-2")
    assert isinstance(nf, CheckpointError)
    assert "ckpt-2" in str(nf)
    corrupt = CheckpointCorruptedError("ckpt-3", "bad hash")
    assert corrupt.reason == "bad hash"
    assert "bad hash" in str(corrupt)


def test_scheduling_errors():
    assert issubclass(SchedulingError, OrchestratorError)
    pre = PreemptionError("job-9", "no candidates")
    assert pre.job_id == "job-9" and pre.reason == "no candidates"


def test_worker_errors():
    base = WorkerError("down", worker_id="w-1")
    assert base.worker_id == "w-1"
    nf = WorkerNotFoundError("w-2")
    assert isinstance(nf, WorkerError) and "w-2" in str(nf)
    unhealthy = WorkerUnhealthyError("w-3", "2026-01-01T00:00:00")
    assert unhealthy.last_heartbeat == "2026-01-01T00:00:00"


def test_experiment_errors():
    base = ExperimentError("x", experiment_id="exp-1")
    assert base.experiment_id == "exp-1"
    nf = ExperimentNotFoundError("exp-2")
    assert isinstance(nf, ExperimentError) and "exp-2" in str(nf)


def test_config_and_validation_errors():
    assert issubclass(ConfigurationError, OrchestratorError)
    v = ValidationError("bad field", field="epochs")
    assert v.field == "epochs"
    assert v.details["field"] == "epochs"
