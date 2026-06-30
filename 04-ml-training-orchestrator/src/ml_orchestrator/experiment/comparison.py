"""Experiment comparison utilities."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import structlog

from ml_orchestrator.experiment.tracker import ExperimentTracker, Run


logger = structlog.get_logger(__name__)


@dataclass
class MetricComparison:
    """Comparison of a metric across runs."""

    metric_name: str
    runs: dict[str, float]  # run_id -> latest value
    best_run_id: Optional[str] = None
    best_value: Optional[float] = None
    worst_run_id: Optional[str] = None
    worst_value: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    mode: str = "min"  # whether lower or higher is better


@dataclass
class ParamComparison:
    """Comparison of parameters across runs."""

    param_name: str
    values: dict[str, Any]  # run_id -> value
    unique_values: list[Any] = field(default_factory=list)
    is_constant: bool = False


@dataclass
class ComparisonResult:
    """Result of comparing multiple runs."""

    run_ids: list[str]
    metrics: dict[str, MetricComparison]
    params: dict[str, ParamComparison]
    best_overall_run: Optional[str] = None
    ranking: list[str] = field(default_factory=list)  # run_ids sorted by primary metric
    created_at: datetime = field(default_factory=datetime.utcnow)


class ExperimentComparison:
    """
    Utilities for comparing experiments and runs.

    Supports:
    - Multi-run metric comparison
    - Parameter analysis
    - Ranking by primary metric
    - Statistical summaries
    """

    def __init__(self, tracker: ExperimentTracker):
        self._tracker = tracker

    async def compare_runs(
        self,
        run_ids: list[str],
        primary_metric: Optional[str] = None,
        primary_metric_mode: str = "min",
        metrics: Optional[list[str]] = None,
        params: Optional[list[str]] = None,
    ) -> ComparisonResult:
        """
        Compare multiple runs.

        Args:
            run_ids: List of run IDs to compare
            primary_metric: Metric for ranking runs
            primary_metric_mode: "min" or "max" for ranking
            metrics: Specific metrics to compare (None = all)
            params: Specific params to compare (None = all)

        Returns:
            ComparisonResult with detailed comparisons
        """
        # Fetch runs
        runs: list[Run] = []
        for run_id in run_ids:
            run = await self._tracker.get_run(run_id)
            if run:
                runs.append(run)

        if not runs:
            return ComparisonResult(run_ids=run_ids, metrics={}, params={})

        # Compare metrics
        metric_comparisons = await self._compare_metrics(
            runs, metrics, primary_metric_mode
        )

        # Compare params
        param_comparisons = await self._compare_params(runs, params)

        # Determine ranking
        ranking = []
        best_overall = None
        if primary_metric and primary_metric in metric_comparisons:
            mc = metric_comparisons[primary_metric]
            # Sort runs by metric
            run_values = [
                (rid, mc.runs.get(rid))
                for rid in run_ids
                if mc.runs.get(rid) is not None
            ]
            if run_values:
                if primary_metric_mode == "min":
                    run_values.sort(key=lambda x: x[1])
                else:
                    run_values.sort(key=lambda x: x[1], reverse=True)
                ranking = [rv[0] for rv in run_values]
                best_overall = ranking[0] if ranking else None

        return ComparisonResult(
            run_ids=run_ids,
            metrics=metric_comparisons,
            params=param_comparisons,
            best_overall_run=best_overall,
            ranking=ranking,
        )

    async def _compare_metrics(
        self,
        runs: list[Run],
        metric_names: Optional[list[str]],
        default_mode: str,
    ) -> dict[str, MetricComparison]:
        """Compare metrics across runs."""
        # Collect all metric names if not specified
        if metric_names is None:
            all_metrics: set[str] = set()
            for run in runs:
                all_metrics.update(run.metrics.keys())
            metric_names = list(all_metrics)

        comparisons = {}

        for metric_name in metric_names:
            values = {}
            for run in runs:
                latest = run.get_latest_metrics()
                if metric_name in latest:
                    values[run.id] = latest[metric_name]

            if not values:
                continue

            # Determine mode based on metric name
            mode = default_mode
            if "loss" in metric_name.lower() or "error" in metric_name.lower():
                mode = "min"
            elif "accuracy" in metric_name.lower() or "score" in metric_name.lower():
                mode = "max"

            # Find best and worst
            if mode == "min":
                best_id = min(values, key=values.get)
                worst_id = max(values, key=values.get)
            else:
                best_id = max(values, key=values.get)
                worst_id = min(values, key=values.get)

            # Calculate statistics
            vals = list(values.values())
            mean = sum(vals) / len(vals)
            variance = sum((v - mean) ** 2 for v in vals) / len(vals)
            std = variance ** 0.5

            comparisons[metric_name] = MetricComparison(
                metric_name=metric_name,
                runs=values,
                best_run_id=best_id,
                best_value=values[best_id],
                worst_run_id=worst_id,
                worst_value=values[worst_id],
                mean=mean,
                std=std,
                mode=mode,
            )

        return comparisons

    async def _compare_params(
        self,
        runs: list[Run],
        param_names: Optional[list[str]],
    ) -> dict[str, ParamComparison]:
        """Compare parameters across runs."""
        # Collect all param names if not specified
        if param_names is None:
            all_params: set[str] = set()
            for run in runs:
                all_params.update(run.params.keys())
            param_names = list(all_params)

        comparisons = {}

        for param_name in param_names:
            values = {}
            for run in runs:
                if param_name in run.params:
                    values[run.id] = run.params[param_name]

            if not values:
                continue

            unique = list(set(values.values()))
            is_constant = len(unique) == 1

            comparisons[param_name] = ParamComparison(
                param_name=param_name,
                values=values,
                unique_values=unique,
                is_constant=is_constant,
            )

        return comparisons

    async def find_correlations(
        self,
        run_ids: list[str],
        param: str,
        metric: str,
    ) -> dict[str, Any]:
        """
        Find correlation between a parameter and a metric.

        Args:
            run_ids: Runs to analyze
            param: Parameter name
            metric: Metric name

        Returns:
            Correlation analysis
        """
        runs = []
        for run_id in run_ids:
            run = await self._tracker.get_run(run_id)
            if run:
                runs.append(run)

        # Collect data points
        data_points = []
        for run in runs:
            if param in run.params:
                latest = run.get_latest_metrics()
                if metric in latest:
                    try:
                        param_val = float(run.params[param])
                        metric_val = latest[metric]
                        data_points.append((param_val, metric_val))
                    except (TypeError, ValueError):
                        pass

        if len(data_points) < 2:
            return {
                "param": param,
                "metric": metric,
                "correlation": None,
                "data_points": len(data_points),
                "message": "Not enough numeric data points",
            }

        # Calculate Pearson correlation
        n = len(data_points)
        sum_x = sum(p[0] for p in data_points)
        sum_y = sum(p[1] for p in data_points)
        sum_xy = sum(p[0] * p[1] for p in data_points)
        sum_x2 = sum(p[0] ** 2 for p in data_points)
        sum_y2 = sum(p[1] ** 2 for p in data_points)

        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5

        if denominator == 0:
            correlation = 0.0
        else:
            correlation = numerator / denominator

        return {
            "param": param,
            "metric": metric,
            "correlation": correlation,
            "data_points": n,
            "interpretation": self._interpret_correlation(correlation),
        }

    def _interpret_correlation(self, r: float) -> str:
        """Interpret correlation coefficient."""
        abs_r = abs(r)
        if abs_r < 0.1:
            strength = "negligible"
        elif abs_r < 0.3:
            strength = "weak"
        elif abs_r < 0.5:
            strength = "moderate"
        elif abs_r < 0.7:
            strength = "strong"
        else:
            strength = "very strong"

        direction = "positive" if r > 0 else "negative"
        return f"{strength} {direction} correlation"

    async def generate_leaderboard(
        self,
        experiment_id: str,
        metric: str,
        mode: str = "min",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Generate a leaderboard for an experiment.

        Args:
            experiment_id: Experiment ID
            metric: Metric to rank by
            mode: "min" or "max"
            limit: Number of top runs

        Returns:
            List of leaderboard entries
        """
        runs = await self._tracker.list_runs(experiment_id, status="completed")

        # Get metric values
        entries = []
        for run in runs:
            latest = run.get_latest_metrics()
            if metric not in latest:
                continue

            entries.append({
                "rank": 0,  # Will be set after sorting
                "run_id": run.id,
                "run_name": run.name,
                "value": latest[metric],
                "params": run.params,
                "finished_at": run.finished_at.isoformat() if run.finished_at else None,
            })

        # Sort
        entries.sort(key=lambda x: x["value"], reverse=(mode == "max"))

        # Assign ranks
        for i, entry in enumerate(entries[:limit]):
            entry["rank"] = i + 1

        return entries[:limit]

    async def summarize_experiment(
        self,
        experiment_id: str,
    ) -> dict[str, Any]:
        """
        Generate a summary of an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            Summary dictionary
        """
        experiment = await self._tracker.get_experiment(experiment_id)
        runs = await self._tracker.list_runs(experiment_id)

        completed = [r for r in runs if r.status == "completed"]
        failed = [r for r in runs if r.status == "failed"]
        running = [r for r in runs if r.status == "running"]

        # Collect all metrics
        all_metrics: dict[str, list[float]] = {}
        for run in completed:
            for name, values in run.metrics.items():
                if name not in all_metrics:
                    all_metrics[name] = []
                if values:
                    all_metrics[name].append(values[-1].value)

        # Calculate metric statistics
        metric_stats = {}
        for name, values in all_metrics.items():
            if values:
                metric_stats[name] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                    "count": len(values),
                }

        # Find varied params
        varied_params = set()
        if len(completed) > 1:
            for param in completed[0].params:
                values = set()
                for run in completed:
                    if param in run.params:
                        values.add(str(run.params[param]))
                if len(values) > 1:
                    varied_params.add(param)

        return {
            "experiment_id": experiment_id,
            "name": experiment.name,
            "status": experiment.status,
            "total_runs": len(runs),
            "completed_runs": len(completed),
            "failed_runs": len(failed),
            "running_runs": len(running),
            "best_job_id": experiment.best_job_id,
            "best_metric_name": experiment.best_metric_name,
            "best_metric_value": experiment.best_metric_value,
            "metric_statistics": metric_stats,
            "varied_parameters": list(varied_params),
            "created_at": experiment.created_at.isoformat(),
            "updated_at": experiment.updated_at.isoformat(),
        }
