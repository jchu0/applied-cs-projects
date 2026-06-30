"""Experiment tracking API endpoints."""

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel, Field

from ml_orchestrator.core.exceptions import ExperimentNotFoundError


router = APIRouter()


class CreateExperimentRequest(BaseModel):
    """Request to create an experiment."""

    name: str = Field(..., min_length=1, max_length=256)
    user_id: str
    team_id: Optional[str] = None
    description: Optional[str] = None
    base_config: dict[str, Any] = Field(default_factory=dict)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    tags: dict[str, str] = Field(default_factory=dict)


class ExperimentResponse(BaseModel):
    """Experiment response model."""

    id: str
    name: str
    user_id: str
    team_id: Optional[str]
    description: Optional[str]
    status: str
    job_ids: list[str]
    best_job_id: Optional[str]
    best_metric_name: Optional[str]
    best_metric_value: Optional[float]
    created_at: str
    updated_at: str
    tags: dict[str, str]


class LogMetricsRequest(BaseModel):
    """Request to log metrics."""

    metrics: dict[str, float]
    step: Optional[int] = None
    epoch: Optional[int] = None


@router.post("", response_model=ExperimentResponse)
async def create_experiment(
    request: Request,
    body: CreateExperimentRequest,
) -> ExperimentResponse:
    """Create a new experiment."""
    state = request.app.state.orchestrator

    experiment = await state.experiment_tracker.create_experiment(
        name=body.name,
        user_id=body.user_id,
        team_id=body.team_id,
        description=body.description,
        base_config=body.base_config,
        hyperparameters=body.hyperparameters,
        tags=body.tags,
    )

    return ExperimentResponse(
        id=experiment.id,
        name=experiment.name,
        user_id=experiment.user_id,
        team_id=experiment.team_id,
        description=experiment.description,
        status=experiment.status,
        job_ids=experiment.job_ids,
        best_job_id=experiment.best_job_id,
        best_metric_name=experiment.best_metric_name,
        best_metric_value=experiment.best_metric_value,
        created_at=experiment.created_at.isoformat(),
        updated_at=experiment.updated_at.isoformat(),
        tags=experiment.tags,
    )


@router.get("", response_model=list[ExperimentResponse])
async def list_experiments(
    request: Request,
    user_id: Optional[str] = Query(None),
    team_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
) -> list[ExperimentResponse]:
    """List experiments."""
    state = request.app.state.orchestrator

    experiments = await state.experiment_tracker.list_experiments(
        user_id=user_id,
        team_id=team_id,
        status=status,
        limit=limit,
    )

    return [
        ExperimentResponse(
            id=e.id,
            name=e.name,
            user_id=e.user_id,
            team_id=e.team_id,
            description=e.description,
            status=e.status,
            job_ids=e.job_ids,
            best_job_id=e.best_job_id,
            best_metric_name=e.best_metric_name,
            best_metric_value=e.best_metric_value,
            created_at=e.created_at.isoformat(),
            updated_at=e.updated_at.isoformat(),
            tags=e.tags,
        )
        for e in experiments
    ]


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    request: Request,
    experiment_id: str,
) -> ExperimentResponse:
    """Get an experiment by ID."""
    state = request.app.state.orchestrator

    try:
        experiment = await state.experiment_tracker.get_experiment(experiment_id)
        return ExperimentResponse(
            id=experiment.id,
            name=experiment.name,
            user_id=experiment.user_id,
            team_id=experiment.team_id,
            description=experiment.description,
            status=experiment.status,
            job_ids=experiment.job_ids,
            best_job_id=experiment.best_job_id,
            best_metric_name=experiment.best_metric_name,
            best_metric_value=experiment.best_metric_value,
            created_at=experiment.created_at.isoformat(),
            updated_at=experiment.updated_at.isoformat(),
            tags=experiment.tags,
        )
    except ExperimentNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Experiment not found: {experiment_id}"
        )


@router.delete("/{experiment_id}")
async def delete_experiment(
    request: Request,
    experiment_id: str,
) -> dict[str, bool]:
    """Delete an experiment."""
    state = request.app.state.orchestrator

    success = await state.experiment_tracker.delete_experiment(experiment_id)
    if not success:
        raise HTTPException(
            status_code=404, detail=f"Experiment not found: {experiment_id}"
        )

    return {"success": True}


@router.get("/{experiment_id}/runs")
async def list_runs(
    request: Request,
    experiment_id: str,
    status: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
) -> list[dict[str, Any]]:
    """List runs for an experiment."""
    state = request.app.state.orchestrator

    runs = await state.experiment_tracker.list_runs(
        experiment_id=experiment_id,
        status=status,
        limit=limit,
    )

    return [
        {
            "id": r.id,
            "name": r.name,
            "job_id": r.job_id,
            "status": r.status,
            "params": r.params,
            "latest_metrics": r.get_latest_metrics(),
            "created_at": r.created_at.isoformat(),
            "finished_at": r.finished_at.isoformat() if r.finished_at else None,
        }
        for r in runs
    ]


@router.get("/{experiment_id}/runs/{run_id}")
async def get_run(
    request: Request,
    experiment_id: str,
    run_id: str,
) -> dict[str, Any]:
    """Get a run by ID."""
    state = request.app.state.orchestrator

    run = await state.experiment_tracker.get_run(run_id)
    if not run or run.experiment_id != experiment_id:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    return {
        "id": run.id,
        "experiment_id": run.experiment_id,
        "job_id": run.job_id,
        "name": run.name,
        "status": run.status,
        "config": run.config,
        "params": run.params,
        "metrics": {
            name: [m.model_dump() for m in values]
            for name, values in run.metrics.items()
        },
        "artifacts": run.artifacts,
        "tags": run.tags,
        "created_at": run.created_at.isoformat(),
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
    }


@router.post("/{experiment_id}/runs/{run_id}/metrics")
async def log_metrics(
    request: Request,
    experiment_id: str,
    run_id: str,
    body: LogMetricsRequest,
) -> dict[str, bool]:
    """Log metrics for a run."""
    state = request.app.state.orchestrator

    run = await state.experiment_tracker.get_run(run_id)
    if not run or run.experiment_id != experiment_id:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    await state.experiment_tracker.log_metrics(
        run_id=run_id,
        metrics=body.metrics,
        step=body.step,
        epoch=body.epoch,
    )

    return {"success": True}


@router.get("/{experiment_id}/runs/{run_id}/metrics/{metric_name}")
async def get_metric_history(
    request: Request,
    experiment_id: str,
    run_id: str,
    metric_name: str,
) -> dict[str, Any]:
    """Get metric history for a run."""
    state = request.app.state.orchestrator

    run = await state.experiment_tracker.get_run(run_id)
    if not run or run.experiment_id != experiment_id:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    history = await state.experiment_tracker.get_metric_history(run_id, metric_name)

    return {
        "run_id": run_id,
        "metric_name": metric_name,
        "values": [m.model_dump() for m in history],
    }


@router.post("/{experiment_id}/compare")
async def compare_runs(
    request: Request,
    experiment_id: str,
    run_ids: list[str],
    metrics: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Compare multiple runs."""
    state = request.app.state.orchestrator

    from ml_orchestrator.experiment.comparison import ExperimentComparison

    comparison = ExperimentComparison(state.experiment_tracker)
    result = await comparison.compare_runs(
        run_ids=run_ids,
        metrics=metrics,
    )

    return {
        "run_ids": result.run_ids,
        "metrics": {
            name: {
                "runs": mc.runs,
                "best_run_id": mc.best_run_id,
                "best_value": mc.best_value,
                "mean": mc.mean,
                "std": mc.std,
            }
            for name, mc in result.metrics.items()
        },
        "params": {
            name: {
                "values": pc.values,
                "unique_values": pc.unique_values,
                "is_constant": pc.is_constant,
            }
            for name, pc in result.params.items()
        },
        "ranking": result.ranking,
        "best_overall_run": result.best_overall_run,
    }


@router.get("/{experiment_id}/best-run")
async def get_best_run(
    request: Request,
    experiment_id: str,
    metric: str = Query(...),
    mode: str = Query("min"),
) -> dict[str, Any]:
    """Get the best run for an experiment."""
    state = request.app.state.orchestrator

    run = await state.experiment_tracker.get_best_run(
        experiment_id=experiment_id,
        metric=metric,
        mode=mode,
    )

    if not run:
        return {"best_run": None}

    return {
        "best_run": {
            "id": run.id,
            "name": run.name,
            "job_id": run.job_id,
            "metrics": run.get_latest_metrics(),
            "params": run.params,
        },
    }
