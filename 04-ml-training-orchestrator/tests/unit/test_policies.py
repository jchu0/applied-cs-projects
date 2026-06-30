"""Tests for scheduling policies."""

import pytest

from ml_orchestrator.core.models import ResourceRequest, JobPriority
from ml_orchestrator.scheduling.policies import (
    FIFOPolicy,
    PriorityPolicy,
    FairSharePolicy,
    GangSchedulingPolicy,
    BackfillPolicy,
    PreemptivePolicy,
    SchedulingDecision,
)


def _avail(gpus=8):
    return {"w1": ResourceRequest(cpus=64, memory_gb=256.0, gpus=gpus)}


class TestFIFO:
    @pytest.mark.asyncio
    async def test_selects_when_resources_available(self, make_job):
        policy = FIFOPolicy()
        jobs = [make_job(name="a", gpus=0), make_job(name="b", gpus=0)]
        decision = await policy.select_job(jobs, _avail(), {})
        assert isinstance(decision, SchedulingDecision)
        assert decision.job_id in {j.id for j in jobs}
        assert decision.assigned_workers == ["w1"]

    @pytest.mark.asyncio
    async def test_none_when_resources_insufficient(self, make_job):
        policy = FIFOPolicy()
        jobs = [make_job(gpus=4)]  # needs GPUs
        decision = await policy.select_job(jobs, _avail(gpus=0), {})
        assert decision is None


class TestPriority:
    @pytest.mark.asyncio
    async def test_prefers_higher_priority(self, make_job):
        policy = PriorityPolicy()
        low = make_job(name="low", priority=JobPriority.LOW, gpus=0)
        high = make_job(name="high", priority=JobPriority.HIGH, gpus=0)
        decision = await policy.select_job([low, high], _avail(), {})
        assert decision is not None
        assert decision.job_id == high.id


class TestFairShare:
    @pytest.mark.asyncio
    async def test_select_and_record_completion(self, make_job):
        policy = FairSharePolicy()
        decision = await policy.select_job([make_job(gpus=0)], _avail(), {})
        assert decision is not None
        # record_completion updates usage tracking without error
        policy.record_completion(make_job(gpus=0))


@pytest.mark.parametrize("PolicyCls", [GangSchedulingPolicy, BackfillPolicy, PreemptivePolicy])
@pytest.mark.asyncio
async def test_other_policies_run(PolicyCls, make_job):
    policy = PolicyCls()
    result = await policy.select_job([make_job(gpus=0)], _avail(), {})
    assert result is None or isinstance(result, SchedulingDecision)
