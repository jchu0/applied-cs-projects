"""Job management API endpoints."""

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel, Field

from ml_orchestrator.core.models import (
    JobConfig,
    JobPriority,
    JobStatus,
    ResourceRequest,
    TrainingJob,
)
from ml_orchestrator.core.exceptions import (
    JobNotFoundError,
    JobStateError,
    ValidationError,
)


router = APIRouter()


class SubmitJobRequest(BaseModel):
    """Request to submit a new training job."""

    name: str = Field(..., min_length=1, max_length=256)
    user_id: str = Field(...)
    config: JobConfig
    resources: Optional[ResourceRequest] = None
    priority: JobPriority = JobPriority.NORMAL
    team_id: Optional[str] = None
    experiment_id: Optional[str] = None
    preemptible: bool = True
    tags: dict[str, str] = Field(default_factory=dict)
    resume_from_checkpoint: Optional[str] = None


class UpdateJobRequest(BaseModel):
    """Request to update a job."""

    priority: Optional[JobPriority] = None
    tags: Optional[dict[str, str]] = None


class JobResponse(BaseModel):
    """Job response model."""

    id: str
    name: str
    user_id: str
    status: str
    priority: int
    progress_percent: float
    current_epoch: int
    current_step: int
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None

    @classmethod
    def from_job(cls, job: TrainingJob) -> "JobResponse":
        return cls(
            id=job.id,
            name=job.name,
            user_id=job.user_id,
            status=job.status.value,
            priority=job.priority.value,
            progress_percent=job.progress_percent,
            current_epoch=job.current_epoch,
            current_step=job.current_step,
            created_at=job.created_at.isoformat(),
            started_at=job.started_at.isoformat() if job.started_at else None,
            completed_at=job.completed_at.isoformat() if job.completed_at else None,
            error_message=job.error_message,
        )


@router.post("", response_model=JobResponse)
async def submit_job(
    request: Request,
    body: SubmitJobRequest,
) -> JobResponse:
    """Submit a new training job."""
    state = request.app.state.orchestrator

    try:
        job = await state.job_manager.submit_job(
            name=body.name,
            user_id=body.user_id,
            config=body.config,
            resources=body.resources,
            priority=body.priority,
            team_id=body.team_id,
            experiment_id=body.experiment_id,
            preemptible=body.preemptible,
            tags=body.tags,
            resume_from_checkpoint=body.resume_from_checkpoint,
        )
        return JobResponse.from_job(job)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("", response_model=list[JobResponse])
async def list_jobs(
    request: Request,
    status: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> list[JobResponse]:
    """List training jobs."""
    state = request.app.state.orchestrator

    job_status = None
    if status:
        try:
            job_status = JobStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

    jobs = await state.job_manager.list_jobs(
        status=job_status,
        user_id=user_id,
        limit=limit,
        offset=offset,
    )

    return [JobResponse.from_job(job) for job in jobs]


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    request: Request,
    job_id: str,
) -> JobResponse:
    """Get a job by ID."""
    state = request.app.state.orchestrator

    try:
        job = await state.job_manager.get_job(job_id)
        return JobResponse.from_job(job)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")


@router.get("/{job_id}/details")
async def get_job_details(
    request: Request,
    job_id: str,
) -> dict[str, Any]:
    """Get full job details including config and metrics."""
    state = request.app.state.orchestrator

    try:
        job = await state.job_manager.get_job(job_id)
        return job.model_dump()
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")


@router.patch("/{job_id}", response_model=JobResponse)
async def update_job(
    request: Request,
    job_id: str,
    body: UpdateJobRequest,
) -> JobResponse:
    """Update job priority or tags."""
    state = request.app.state.orchestrator

    try:
        job = await state.job_manager.get_job(job_id)

        if body.priority is not None:
            job.priority = body.priority
            await state.scheduler.update_job_priority(job_id, body.priority)

        if body.tags is not None:
            job.tags.update(body.tags)

        return JobResponse.from_job(job)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")


@router.post("/{job_id}/pause", response_model=JobResponse)
async def pause_job(
    request: Request,
    job_id: str,
) -> JobResponse:
    """Pause a running job."""
    state = request.app.state.orchestrator

    try:
        job = await state.job_manager.pause_job(job_id)
        return JobResponse.from_job(job)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    except JobStateError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{job_id}/resume", response_model=JobResponse)
async def resume_job(
    request: Request,
    job_id: str,
) -> JobResponse:
    """Resume a paused job."""
    state = request.app.state.orchestrator

    try:
        job = await state.job_manager.resume_job(job_id)
        return JobResponse.from_job(job)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    except JobStateError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{job_id}/cancel", response_model=JobResponse)
async def cancel_job(
    request: Request,
    job_id: str,
    reason: str = Query("User cancelled"),
) -> JobResponse:
    """Cancel a job."""
    state = request.app.state.orchestrator

    try:
        job = await state.job_manager.cancel_job(job_id, reason)
        return JobResponse.from_job(job)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    except JobStateError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{job_id}/metrics")
async def get_job_metrics(
    request: Request,
    job_id: str,
) -> dict[str, Any]:
    """Get metrics for a job."""
    state = request.app.state.orchestrator

    try:
        job = await state.job_manager.get_job(job_id)
        return {
            "job_id": job_id,
            "metrics": [m.model_dump() for m in job.metrics],
            "best_metrics": job.best_metrics,
        }
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")


@router.get("/{job_id}/checkpoints")
async def get_job_checkpoints(
    request: Request,
    job_id: str,
) -> dict[str, Any]:
    """Get checkpoints for a job."""
    state = request.app.state.orchestrator

    try:
        job = await state.job_manager.get_job(job_id)
        return {
            "job_id": job_id,
            "checkpoints": [c.model_dump() for c in job.checkpoints],
            "latest_checkpoint_id": job.latest_checkpoint_id,
        }
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")


@router.get("/{job_id}/queue-position")
async def get_queue_position(
    request: Request,
    job_id: str,
) -> dict[str, Any]:
    """Get job's position in the queue."""
    state = request.app.state.orchestrator

    try:
        await state.job_manager.get_job(job_id)  # Verify job exists
        position = await state.scheduler.get_queue_position(job_id)
        return {
            "job_id": job_id,
            "queue_position": position,
            "in_queue": position is not None,
        }
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
