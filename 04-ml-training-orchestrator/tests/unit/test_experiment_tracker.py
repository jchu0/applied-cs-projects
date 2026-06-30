"""Tests for the experiment tracker."""

import pytest

from ml_orchestrator.core.exceptions import ExperimentNotFoundError
from ml_orchestrator.experiment.tracker import ExperimentTracker


async def _make_experiment(tracker, name="exp"):
    return await tracker.create_experiment(name=name, user_id="user-1", team_id="team-1")


class TestExperiments:
    @pytest.mark.asyncio
    async def test_create_get_list(self):
        tracker = ExperimentTracker()
        exp = await _make_experiment(tracker)
        assert (await tracker.get_experiment(exp.id)).id == exp.id
        assert len(await tracker.list_experiments()) == 1

    @pytest.mark.asyncio
    async def test_get_missing_raises(self):
        tracker = ExperimentTracker()
        with pytest.raises(ExperimentNotFoundError):
            await tracker.get_experiment("missing")

    @pytest.mark.asyncio
    async def test_update_and_delete(self):
        tracker = ExperimentTracker()
        exp = await _make_experiment(tracker)
        updated = await tracker.update_experiment(exp.id, description="new desc")
        assert updated.description == "new desc"
        assert await tracker.delete_experiment(exp.id) is True


class TestRunsAndMetrics:
    @pytest.mark.asyncio
    async def test_create_run_and_log_metrics(self, sample_job):
        tracker = ExperimentTracker()
        exp = await _make_experiment(tracker)
        run = await tracker.create_run(exp.id, sample_job, name="run-1")
        assert run.experiment_id == exp.id

        await tracker.log_metric(run.id, "loss", 1.0, step=0)
        await tracker.log_metric(run.id, "loss", 0.5, step=1)
        await tracker.log_metrics(run.id, {"acc": 0.9}, step=1)

        history = await tracker.get_metric_history(run.id, "loss")
        assert len(history) == 2
        summary = await tracker.get_metrics_summary(run.id)
        assert "loss" in summary

    @pytest.mark.asyncio
    async def test_log_params_and_artifacts(self, sample_job):
        tracker = ExperimentTracker()
        exp = await _make_experiment(tracker)
        run = await tracker.create_run(exp.id, sample_job)
        await tracker.log_param(run.id, "lr", 0.001)
        await tracker.log_params(run.id, {"batch_size": 32, "epochs": 10})
        await tracker.log_artifact(run.id, "/tmp/model.pt")
        fetched = await tracker.get_run(run.id)
        assert fetched is not None
        assert fetched.params.get("lr") == 0.001
        assert fetched.params.get("batch_size") == 32
        assert "/tmp/model.pt" in fetched.artifacts

    @pytest.mark.asyncio
    async def test_finish_run_and_lookup_by_job(self, sample_job):
        tracker = ExperimentTracker()
        exp = await _make_experiment(tracker)
        run = await tracker.create_run(exp.id, sample_job)
        await tracker.log_metric(run.id, "loss", 0.3)
        await tracker.finish_run(run.id)
        by_job = await tracker.get_run_by_job(sample_job.id)
        assert by_job is not None and by_job.id == run.id
        # Run helper methods
        assert isinstance(run.get_latest_metrics(), dict)
        assert isinstance(run.get_metric_history("loss"), list)

    @pytest.mark.asyncio
    async def test_list_runs(self, sample_job):
        tracker = ExperimentTracker()
        exp = await _make_experiment(tracker)
        await tracker.create_run(exp.id, sample_job)
        runs = await tracker.list_runs(experiment_id=exp.id)
        assert len(runs) == 1
