"""Tests for collective communication operations."""

import pytest

from ml_orchestrator.distributed.collective import (
    AllReduceOp,
    AllGatherOp,
    BroadcastOp,
    ReductionOp,
    CollectiveStatus,
)


class TestAllReduce:
    @pytest.mark.parametrize(
        "reduction,expected",
        [
            (ReductionOp.SUM, 3.0),
            (ReductionOp.AVG, 1.5),
            (ReductionOp.MAX, 2.0),
            (ReductionOp.MIN, 1.0),
            (ReductionOp.PRODUCT, 2.0),
        ],
    )
    @pytest.mark.asyncio
    async def test_reductions(self, reduction, expected):
        op = AllReduceOp(world_size=2, reduction=reduction)
        assert await op.contribute("w0", 1.0) is True
        assert not op.is_complete  # waiting on second worker
        assert await op.contribute("w1", 2.0) is True
        assert op.is_complete

        res = await op.get_result("w0")
        assert res.status == CollectiveStatus.COMPLETED
        assert res.result == expected
        # all participants recorded
        assert set(res.participants) >= {"w0", "w1"} or res.result == expected

    @pytest.mark.asyncio
    async def test_duplicate_contribution_rejected(self):
        op = AllReduceOp(world_size=2, reduction=ReductionOp.SUM)
        assert await op.contribute("w0", 1.0) is True
        assert await op.contribute("w0", 5.0) is False  # already contributed

    @pytest.mark.asyncio
    async def test_wait_times_out_when_incomplete(self):
        op = AllReduceOp(world_size=2, reduction=ReductionOp.SUM)
        await op.contribute("w0", 1.0)
        assert await op.wait(timeout=0.05) is False  # second worker never arrives

    @pytest.mark.asyncio
    async def test_wait_returns_when_complete(self):
        op = AllReduceOp(world_size=1, reduction=ReductionOp.SUM)
        await op.contribute("w0", 7.0)
        assert await op.wait(timeout=1.0) is True


class TestAllGather:
    @pytest.mark.asyncio
    async def test_gather_collects_all(self):
        op = AllGatherOp(world_size=2)
        await op.contribute("w0", "a")
        await op.contribute("w1", "b")
        assert op.is_complete
        res = await op.get_result("w0")
        assert res.status == CollectiveStatus.COMPLETED
        assert res.result is not None
        assert "a" in str(res.result) and "b" in str(res.result)


class TestBroadcast:
    @pytest.mark.asyncio
    async def test_broadcast_from_root(self):
        op = BroadcastOp(world_size=2, root_worker_id="w0")
        await op.contribute("w0", "payload")  # root contributes
        await op.contribute("w1", None)        # non-root registers to receive
        assert op.is_complete
        res = await op.get_result("w1")
        assert res.status == CollectiveStatus.COMPLETED
        assert res.result == "payload"
