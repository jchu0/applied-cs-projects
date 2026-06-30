"""Experiment tracking for ML training jobs."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4
import structlog

from ml_orchestrator.core.models import (
    Experiment,
    MetricType,
    MetricValue,
    TrainingJob,
)
from ml_orchestrator.core.exceptions import ExperimentNotFoundError


logger = structlog.get_logger(__name__)


@dataclass
class Run:
    """A single run within an experiment."""

    id: str = field(default_factory=lambda: str(uuid4()))
    experiment_id: str = ""
    job_id: str = ""
    name: str = ""
    status: str = "running"  # running, completed, failed
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, list[MetricValue]] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    finished_at: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_latest_metrics(self) -> dict[str, float]:
        """Get the latest value for each metric."""
        result = {}
        for name, values in self.metrics.items():
            if values:
                result[name] = values[-1].value
        return result

    def get_metric_history(self, name: str) -> list[MetricValue]:
        """Get history of a specific metric."""
        return self.metrics.get(name, [])


class ExperimentTracker:
    """
    Tracks experiments, runs, metrics, and artifacts.

    Provides:
    - Experiment and run management
    - Metric logging with history
    - Parameter tracking
    - Artifact management
    - Comparison utilities
    """

    def __init__(self):
        self._experiments: dict[str, Experiment] = {}
        self._runs: dict[str, Run] = {}
        self._experiment_runs: dict[str, list[str]] = {}  # exp_id -> run_ids
        self._job_runs: dict[str, str] = {}  # job_id -> run_id
        self._lock = asyncio.Lock()

    async def create_experiment(
        self,
        name: str,
        user_id: str,
        team_id: Optional[str] = None,
        description: Optional[str] = None,
        base_config: Optional[dict[str, Any]] = None,
        hyperparameters: Optional[dict[str, Any]] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> Experiment:
        """
        Create a new experiment.

        Args:
            name: Experiment name
            user_id: User ID
            team_id: Optional team ID
            description: Optional description
            base_config: Base configuration shared by runs
            hyperparameters: Hyperparameter search space
            tags: Tags for organization

        Returns:
            Created Experiment
        """
        async with self._lock:
            experiment = Experiment(
                name=name,
                user_id=user_id,
                team_id=team_id,
                description=description,
                base_config=base_config or {},
                hyperparameters=hyperparameters or {},
                tags=tags or {},
            )

            self._experiments[experiment.id] = experiment
            self._experiment_runs[experiment.id] = []

            logger.info(
                "experiment_created",
                experiment_id=experiment.id,
                name=name,
                user_id=user_id,
            )

            return experiment

    async def get_experiment(self, experiment_id: str) -> Experiment:
        """
        Get experiment by ID.

        Raises:
            ExperimentNotFoundError: If experiment not found
        """
        async with self._lock:
            experiment = self._experiments.get(experiment_id)
            if not experiment:
                raise ExperimentNotFoundError(experiment_id)
            return experiment

    async def list_experiments(
        self,
        user_id: Optional[str] = None,
        team_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> list[Experiment]:
        """List experiments with optional filtering."""
        async with self._lock:
            experiments = list(self._experiments.values())

            if user_id:
                experiments = [e for e in experiments if e.user_id == user_id]
            if team_id:
                experiments = [e for e in experiments if e.team_id == team_id]
            if status:
                experiments = [e for e in experiments if e.status == status]

            # Sort by created_at descending
            experiments.sort(key=lambda e: e.created_at, reverse=True)

            return experiments[:limit]

    async def update_experiment(
        self,
        experiment_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        status: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> Experiment:
        """Update experiment fields."""
        async with self._lock:
            experiment = self._experiments.get(experiment_id)
            if not experiment:
                raise ExperimentNotFoundError(experiment_id)

            if name:
                experiment.name = name
            if description:
                experiment.description = description
            if status:
                experiment.status = status
            if tags:
                experiment.tags.update(tags)

            experiment.updated_at = datetime.utcnow()

            return experiment

    async def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment and its runs."""
        async with self._lock:
            experiment = self._experiments.pop(experiment_id, None)
            if not experiment:
                return False

            # Delete associated runs
            run_ids = self._experiment_runs.pop(experiment_id, [])
            for run_id in run_ids:
                self._runs.pop(run_id, None)

            logger.info(
                "experiment_deleted",
                experiment_id=experiment_id,
                runs_deleted=len(run_ids),
            )

            return True

    async def create_run(
        self,
        experiment_id: str,
        job: TrainingJob,
        name: Optional[str] = None,
        params: Optional[dict[str, Any]] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> Run:
        """
        Create a new run within an experiment.

        Args:
            experiment_id: Parent experiment ID
            job: Associated training job
            name: Optional run name
            params: Run parameters
            tags: Run tags

        Returns:
            Created Run
        """
        async with self._lock:
            experiment = self._experiments.get(experiment_id)
            if not experiment:
                raise ExperimentNotFoundError(experiment_id)

            run = Run(
                experiment_id=experiment_id,
                job_id=job.id,
                name=name or f"run-{len(self._experiment_runs.get(experiment_id, [])) + 1}",
                config=job.config.model_dump(),
                params=params or {},
                tags=tags or {},
            )

            self._runs[run.id] = run
            self._experiment_runs[experiment_id].append(run.id)
            self._job_runs[job.id] = run.id
            experiment.job_ids.append(job.id)

            logger.info(
                "run_created",
                run_id=run.id,
                experiment_id=experiment_id,
                job_id=job.id,
            )

            return run

    async def get_run(self, run_id: str) -> Optional[Run]:
        """Get run by ID."""
        async with self._lock:
            return self._runs.get(run_id)

    async def get_run_by_job(self, job_id: str) -> Optional[Run]:
        """Get run by job ID."""
        async with self._lock:
            run_id = self._job_runs.get(job_id)
            if run_id:
                return self._runs.get(run_id)
            return None

    async def list_runs(
        self,
        experiment_id: str,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> list[Run]:
        """List runs for an experiment."""
        async with self._lock:
            run_ids = self._experiment_runs.get(experiment_id, [])
            runs = [self._runs[rid] for rid in run_ids if rid in self._runs]

            if status:
                runs = [r for r in runs if r.status == status]

            # Sort by created_at descending
            runs.sort(key=lambda r: r.created_at, reverse=True)

            return runs[:limit]

    async def log_metric(
        self,
        run_id: str,
        name: str,
        value: float,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        metric_type: MetricType = MetricType.SCALAR,
    ) -> None:
        """
        Log a metric value for a run.

        Args:
            run_id: Run ID
            name: Metric name
            value: Metric value
            step: Training step
            epoch: Training epoch
            metric_type: Type of metric
        """
        async with self._lock:
            run = self._runs.get(run_id)
            if not run:
                return

            metric = MetricValue(
                name=name,
                value=value,
                step=step,
                epoch=epoch,
                metric_type=metric_type,
            )

            if name not in run.metrics:
                run.metrics[name] = []
            run.metrics[name].append(metric)

            # Update experiment best if applicable
            await self._update_experiment_best(run, name, value)

    async def _update_experiment_best(
        self,
        run: Run,
        metric_name: str,
        value: float,
    ) -> None:
        """Update experiment's best metric tracking."""
        experiment = self._experiments.get(run.experiment_id)
        if not experiment:
            return

        # Determine if this is better
        # Convention: "loss" metrics are minimized, others maximized
        is_loss = "loss" in metric_name.lower() or "error" in metric_name.lower()

        if experiment.best_metric_name != metric_name:
            # First time tracking this metric
            if experiment.best_metric_value is None:
                experiment.best_metric_name = metric_name
                experiment.best_metric_value = value
                experiment.best_job_id = run.job_id
        else:
            # Compare with existing best
            current_best = experiment.best_metric_value
            if current_best is not None:
                if is_loss:
                    is_better = value < current_best
                else:
                    is_better = value > current_best

                if is_better:
                    experiment.best_metric_value = value
                    experiment.best_job_id = run.job_id

    async def log_metrics(
        self,
        run_id: str,
        metrics: dict[str, float],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ) -> None:
        """Log multiple metrics at once."""
        for name, value in metrics.items():
            await self.log_metric(run_id, name, value, step, epoch)

    async def log_param(
        self,
        run_id: str,
        key: str,
        value: Any,
    ) -> None:
        """Log a parameter for a run."""
        async with self._lock:
            run = self._runs.get(run_id)
            if run:
                run.params[key] = value

    async def log_params(
        self,
        run_id: str,
        params: dict[str, Any],
    ) -> None:
        """Log multiple parameters at once."""
        async with self._lock:
            run = self._runs.get(run_id)
            if run:
                run.params.update(params)

    async def log_artifact(
        self,
        run_id: str,
        artifact_path: str,
    ) -> None:
        """Log an artifact path for a run."""
        async with self._lock:
            run = self._runs.get(run_id)
            if run:
                run.artifacts.append(artifact_path)

    async def finish_run(
        self,
        run_id: str,
        status: str = "completed",
    ) -> Optional[Run]:
        """Mark a run as finished."""
        async with self._lock:
            run = self._runs.get(run_id)
            if not run:
                return None

            run.status = status
            run.finished_at = datetime.utcnow()

            logger.info(
                "run_finished",
                run_id=run_id,
                status=status,
            )

            return run

    async def get_metric_history(
        self,
        run_id: str,
        metric_name: str,
    ) -> list[MetricValue]:
        """Get full history of a metric for a run."""
        async with self._lock:
            run = self._runs.get(run_id)
            if not run:
                return []
            return run.get_metric_history(metric_name)

    async def get_metrics_summary(
        self,
        run_id: str,
    ) -> dict[str, dict[str, float]]:
        """
        Get summary statistics for all metrics in a run.

        Returns:
            Dict of metric name -> {min, max, last, avg}
        """
        async with self._lock:
            run = self._runs.get(run_id)
            if not run:
                return {}

            summary = {}
            for name, values in run.metrics.items():
                if not values:
                    continue
                vals = [v.value for v in values]
                summary[name] = {
                    "min": min(vals),
                    "max": max(vals),
                    "last": vals[-1],
                    "avg": sum(vals) / len(vals),
                    "count": len(vals),
                }

            return summary

    async def compare_runs(
        self,
        run_ids: list[str],
        metrics: Optional[list[str]] = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Compare multiple runs.

        Args:
            run_ids: List of run IDs to compare
            metrics: Specific metrics to compare (None = all)

        Returns:
            Dict of run_id -> comparison data
        """
        async with self._lock:
            comparison = {}

            for run_id in run_ids:
                run = self._runs.get(run_id)
                if not run:
                    continue

                latest_metrics = run.get_latest_metrics()
                if metrics:
                    latest_metrics = {
                        k: v for k, v in latest_metrics.items() if k in metrics
                    }

                comparison[run_id] = {
                    "name": run.name,
                    "status": run.status,
                    "params": run.params,
                    "metrics": latest_metrics,
                    "created_at": run.created_at.isoformat(),
                    "finished_at": (
                        run.finished_at.isoformat() if run.finished_at else None
                    ),
                }

            return comparison

    async def get_best_run(
        self,
        experiment_id: str,
        metric: str,
        mode: str = "min",
    ) -> Optional[Run]:
        """
        Get the best run for an experiment based on a metric.

        Args:
            experiment_id: Experiment ID
            metric: Metric name to compare
            mode: "min" or "max"

        Returns:
            Best run or None
        """
        runs = await self.list_runs(experiment_id, status="completed")

        if not runs:
            return None

        best_run = None
        best_value = None

        for run in runs:
            latest = run.get_latest_metrics()
            if metric not in latest:
                continue

            value = latest[metric]
            if best_value is None:
                best_value = value
                best_run = run
            elif mode == "min" and value < best_value:
                best_value = value
                best_run = run
            elif mode == "max" and value > best_value:
                best_value = value
                best_run = run

        return best_run

    async def get_stats(self) -> dict[str, Any]:
        """Get tracker statistics."""
        async with self._lock:
            return {
                "total_experiments": len(self._experiments),
                "total_runs": len(self._runs),
                "experiments_by_status": {
                    status: sum(
                        1 for e in self._experiments.values() if e.status == status
                    )
                    for status in ["active", "completed", "archived"]
                },
                "runs_by_status": {
                    status: sum(
                        1 for r in self._runs.values() if r.status == status
                    )
                    for status in ["running", "completed", "failed"]
                },
            }
