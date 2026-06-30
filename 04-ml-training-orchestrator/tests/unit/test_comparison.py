"""Tests for experiment run comparison."""

import pytest

from ml_orchestrator.experiment.tracker import ExperimentTracker
from ml_orchestrator.experiment.comparison import ExperimentComparison, ComparisonResult


async def _setup_runs(make_job_factory):
    """Build an experiment with two runs that have params + metrics."""
    tracker = ExperimentTracker()
    exp = await tracker.create_experiment(name="exp", user_id="u")
    runs = []
    for i, loss in enumerate([0.8, 0.4]):
        job = make_job_factory(name=f"j{i}")
        run = await tracker.create_run(exp.id, job, params={"lr": 0.01 * (i + 1)})
        await tracker.log_metric(run.id, "loss", loss, step=1)
        await tracker.log_metric(run.id, "acc", 1 - loss, step=1)
        await tracker.finish_run(run.id)
        runs.append(run)
    return tracker, exp, runs


@pytest.mark.asyncio
async def test_compare_runs(make_job):
    tracker, exp, runs = await _setup_runs(make_job)
    comp = ExperimentComparison(tracker)
    result = await comp.compare_runs(
        [r.id for r in runs], primary_metric="loss", primary_metric_mode="min"
    )
    assert isinstance(result, ComparisonResult)


@pytest.mark.asyncio
async def test_find_correlations(make_job):
    tracker, exp, runs = await _setup_runs(make_job)
    comp = ExperimentComparison(tracker)
    corr = await comp.find_correlations([r.id for r in runs], "lr", "loss")
    assert isinstance(corr, dict)


@pytest.mark.asyncio
async def test_generate_leaderboard(make_job):
    tracker, exp, runs = await _setup_runs(make_job)
    comp = ExperimentComparison(tracker)
    board = await comp.generate_leaderboard(exp.id, "loss")
    assert isinstance(board, list)


@pytest.mark.asyncio
async def test_summarize_experiment(make_job):
    tracker, exp, runs = await _setup_runs(make_job)
    comp = ExperimentComparison(tracker)
    summary = await comp.summarize_experiment(exp.id)
    assert isinstance(summary, dict)
