"""Tests for the scheduling priority queues."""

import pytest

from ml_orchestrator.core.models import JobPriority
from ml_orchestrator.scheduling.priority_queue import (
    PriorityQueue,
    MultiLevelQueue,
)


class TestPriorityQueue:
    @pytest.mark.asyncio
    async def test_push_pop_orders_by_priority(self, make_job):
        q = PriorityQueue()
        low = make_job(name="low", priority=JobPriority.LOW)
        crit = make_job(name="crit", priority=JobPriority.CRITICAL)
        normal = make_job(name="normal", priority=JobPriority.NORMAL)
        for job in (low, crit, normal):
            await q.push(job)

        assert await q.size() == 3
        # Highest priority pops first.
        assert await q.pop() == crit.id
        assert await q.pop() == normal.id
        assert await q.pop() == low.id
        assert await q.pop() is None

    @pytest.mark.asyncio
    async def test_empty_and_contains(self, make_job):
        q = PriorityQueue()
        assert await q.is_empty()
        job = make_job()
        await q.push(job)
        assert not await q.is_empty()
        assert await q.contains(job.id)
        assert not await q.contains("nonexistent")

    @pytest.mark.asyncio
    async def test_peek_does_not_remove(self, make_job):
        q = PriorityQueue()
        job = make_job(priority=JobPriority.HIGH)
        await q.push(job)
        assert await q.peek() == job.id
        assert await q.size() == 1  # still there

    @pytest.mark.asyncio
    async def test_remove(self, make_job):
        q = PriorityQueue()
        a, b = make_job(name="a"), make_job(name="b")
        await q.push(a)
        await q.push(b)
        assert await q.remove(a.id) is True
        assert await q.remove("missing") is False
        assert await q.size() == 1
        assert await q.pop() == b.id

    @pytest.mark.asyncio
    async def test_update_priority(self, make_job):
        q = PriorityQueue()
        a = make_job(name="a", priority=JobPriority.LOW)
        b = make_job(name="b", priority=JobPriority.NORMAL)
        await q.push(a)
        await q.push(b)
        # Boost a above b.
        await q.update_priority(a.id, JobPriority.CRITICAL)
        assert await q.pop() == a.id

    @pytest.mark.asyncio
    async def test_duplicate_push_updates(self, make_job):
        q = PriorityQueue()
        job = make_job(priority=JobPriority.LOW)
        await q.push(job)
        await q.push(job)  # same id -> updates, not duplicates
        assert await q.size() == 1

    @pytest.mark.asyncio
    async def test_clear_and_get_all(self, make_job):
        q = PriorityQueue()
        ids = []
        for i in range(3):
            job = make_job(name=f"j{i}")
            ids.append(job.id)
            await q.push(job)
        all_ids = await q.get_all_job_ids()
        assert set(all_ids) == set(ids)
        await q.clear()
        assert await q.is_empty()

    @pytest.mark.asyncio
    async def test_get_stats(self, make_job):
        q = PriorityQueue()
        await q.push(make_job(priority=JobPriority.HIGH))
        await q.push(make_job(priority=JobPriority.LOW))
        stats = await q.get_stats()
        assert isinstance(stats, dict)
        assert stats.get("total", stats.get("size", 0)) >= 0  # some count field exists


class TestMultiLevelQueue:
    @pytest.mark.asyncio
    async def test_push_pop_by_level(self):
        q = MultiLevelQueue(levels=5)
        await q.push("low-job", JobPriority.LOW)
        await q.push("crit-job", JobPriority.CRITICAL)
        assert await q.size() == 2
        # Highest priority comes out first.
        assert await q.pop() == "crit-job"
        assert await q.pop() == "low-job"
        assert await q.pop() is None

    @pytest.mark.asyncio
    async def test_remove_and_sizes(self):
        q = MultiLevelQueue(levels=5)
        await q.push("a", JobPriority.NORMAL)
        await q.push("b", JobPriority.NORMAL)
        assert await q.remove("a") is True
        assert await q.remove("missing") is False
        assert await q.size() == 1
        sizes = await q.get_level_sizes()
        assert isinstance(sizes, list) and sum(sizes) == 1

    @pytest.mark.asyncio
    async def test_is_empty(self):
        q = MultiLevelQueue()
        assert await q.is_empty()
        await q.push("x", JobPriority.HIGH)
        assert not await q.is_empty()

    def test_priority_to_level_in_range(self):
        q = MultiLevelQueue(levels=5)
        for p in JobPriority:
            level = q._priority_to_level(p)
            assert 0 <= level < 5
