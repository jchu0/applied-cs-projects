"""Tests for the Scheduler's worker/queue/quota/stats APIs (no run loop)."""

import pytest

from ml_orchestrator.core.models import ResourceQuota, JobPriority
from ml_orchestrator.core.job_manager import JobManager
from ml_orchestrator.scheduling.scheduler import Scheduler


def _sched():
    # Construct directly without start() so no background loop runs.
    return Scheduler(JobManager())


class TestWorkers:
    @pytest.mark.asyncio
    async def test_register_list_heartbeat_unregister(self, make_worker):
        s = _sched()
        w = make_worker("w1")
        await s.register_worker(w)
        assert len(await s.list_workers()) == 1
        assert (await s.get_worker(w.id)).id == w.id
        assert await s.worker_heartbeat(w.id) is True
        assert isinstance(await s.get_healthy_workers(), list)
        await s.unregister_worker(w.id)
        assert len(await s.list_workers()) == 0


class TestQueue:
    @pytest.mark.asyncio
    async def test_queue_position_update_dequeue(self, make_job):
        s = _sched()
        a = make_job(name="a", priority=JobPriority.LOW)
        b = make_job(name="b", priority=JobPriority.HIGH)
        await s.queue_job(a)
        await s.queue_job(b)
        assert await s.get_queue_size() == 2
        assert await s.get_queue_position(b.id) in (0, 1)
        assert await s.update_job_priority(a.id, JobPriority.CRITICAL) is True
        assert await s.dequeue_job(a.id) is True
        assert await s.get_queue_size() == 1
        assert await s.get_queue_position("missing") is None


class TestQuota:
    @pytest.mark.asyncio
    async def test_quota_crud(self):
        s = _sched()
        q = ResourceQuota(entity_id="team-1", entity_type="team", max_gpus=4)
        await s.set_quota(q)
        assert (await s.get_quota("team-1")).max_gpus == 4
        assert await s.remove_quota("team-1") is True
        assert await s.get_quota("team-1") is None
        assert await s.remove_quota("missing") is False


class TestStats:
    @pytest.mark.asyncio
    async def test_stats_and_utilization(self, make_worker):
        s = _sched()
        await s.register_worker(make_worker("w1"))
        assert isinstance(await s.get_stats(), dict)
        assert isinstance(await s.get_resource_utilization(), dict)
