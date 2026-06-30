"""Tests for resource allocation policies and the ResourceAllocator."""

import pytest

from ml_orchestrator.resources.allocator import (
    FirstFitPolicy,
    BestFitPolicy,
    WorstFitPolicy,
    BinPackingPolicy,
    AffinityAwarePolicy,
    ResourceAllocator,
    AllocationResult,
)

POLICIES = [FirstFitPolicy, BestFitPolicy, WorstFitPolicy, BinPackingPolicy, AffinityAwarePolicy]


@pytest.mark.parametrize("PolicyCls", POLICIES)
@pytest.mark.asyncio
async def test_policy_allocates_when_a_worker_fits(PolicyCls, make_job, make_worker):
    policy = PolicyCls()
    job = make_job(gpus=0)
    workers = [make_worker("w1"), make_worker("w2")]
    result = await policy.allocate(job, workers)
    assert isinstance(result, AllocationResult)
    assert result.success
    assert result.worker_id in {w.id for w in workers}


@pytest.mark.parametrize("PolicyCls", POLICIES)
@pytest.mark.asyncio
async def test_policy_returns_none_when_no_worker_fits(PolicyCls, make_job, make_worker):
    policy = PolicyCls()
    job = make_job(gpus=0)  # needs cpus=2
    workers = [make_worker("tiny", cpus=1, memory_gb=1.0, gpus=0)]
    result = await policy.allocate(job, workers)
    assert result is None or not result.success


class TestResourceAllocator:
    @pytest.mark.asyncio
    async def test_register_get_list_unregister(self, make_worker):
        alloc = ResourceAllocator()
        w = make_worker("w1")
        await alloc.register_worker(w)
        assert (await alloc.get_worker(w.id)).id == w.id
        assert len(await alloc.list_workers()) == 1
        await alloc.unregister_worker(w.id)
        assert len(await alloc.list_workers()) == 0

    @pytest.mark.asyncio
    async def test_allocate_via_strategy(self, make_worker, make_job):
        alloc = ResourceAllocator()
        await alloc.register_worker(make_worker("w1"))
        result = await alloc.allocate(make_job(gpus=0))
        assert result.success
        assert isinstance(await alloc.get_utilization(), dict)
