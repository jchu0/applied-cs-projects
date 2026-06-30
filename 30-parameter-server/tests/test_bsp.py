"""Tests for BSP consistency model."""

import pytest
import asyncio

from paramserver.consistency.bsp import BSPConsistency


class TestBSPInit:
    """Tests for BSP initialization."""

    def test_create_bsp(self):
        """Test BSP creation."""
        bsp = BSPConsistency(num_workers=4)
        assert bsp.num_workers == 4

    def test_invalid_num_workers(self):
        """Test that invalid num_workers raises error."""
        with pytest.raises(ValueError, match="num_workers must be >= 1"):
            BSPConsistency(num_workers=0)

        with pytest.raises(ValueError, match="num_workers must be >= 1"):
            BSPConsistency(num_workers=-1)

    def test_name(self):
        """Test BSP name property."""
        bsp = BSPConsistency(num_workers=2)
        assert bsp.name == "BSPConsistency"


class TestBSPCanApply:
    """Tests for can_apply method."""

    def test_cannot_apply_no_workers_arrived(self):
        """Test that updates cannot be applied before workers arrive."""
        bsp = BSPConsistency(num_workers=3)

        # No workers have arrived at clock 0
        assert bsp.can_apply(param_version=0, worker_clock=0) is False

    @pytest.mark.asyncio
    async def test_cannot_apply_partial_arrival(self):
        """Test that partial arrival doesn't allow updates."""
        bsp = BSPConsistency(num_workers=3)

        # Only 2 of 3 workers arrive
        await bsp.worker_arrived(0, clock=0)
        await bsp.worker_arrived(1, clock=0)

        assert bsp.can_apply(param_version=0, worker_clock=0) is False

    @pytest.mark.asyncio
    async def test_can_apply_all_arrived(self):
        """Test that updates can be applied when all workers arrive."""
        bsp = BSPConsistency(num_workers=3)

        # All workers arrive
        await bsp.worker_arrived(0, clock=0)
        await bsp.worker_arrived(1, clock=0)
        await bsp.worker_arrived(2, clock=0)

        assert bsp.can_apply(param_version=0, worker_clock=0) is True


class TestBSPWorkerArrived:
    """Tests for worker_arrived method."""

    @pytest.mark.asyncio
    async def test_single_worker_arrival(self):
        """Test single worker arrival."""
        bsp = BSPConsistency(num_workers=3)

        result = await bsp.worker_arrived(0, clock=0)
        assert result is False  # Not all arrived yet

    @pytest.mark.asyncio
    async def test_all_workers_arrival_returns_true(self):
        """Test that last worker arrival returns True."""
        bsp = BSPConsistency(num_workers=2)

        result1 = await bsp.worker_arrived(0, clock=0)
        assert result1 is False

        result2 = await bsp.worker_arrived(1, clock=0)
        assert result2 is True

    @pytest.mark.asyncio
    async def test_duplicate_arrival_ignored(self):
        """Test that duplicate arrivals don't double-count."""
        bsp = BSPConsistency(num_workers=2)

        await bsp.worker_arrived(0, clock=0)
        await bsp.worker_arrived(0, clock=0)  # Duplicate

        assert bsp.can_apply(0, 0) is False  # Still waiting for worker 1

    @pytest.mark.asyncio
    async def test_multiple_barriers(self):
        """Test multiple clock barriers."""
        bsp = BSPConsistency(num_workers=2)

        # Clock 0 barrier
        await bsp.worker_arrived(0, clock=0)
        await bsp.worker_arrived(1, clock=0)
        assert bsp.can_apply(0, 0) is True

        # Clock 1 barrier - partial
        await bsp.worker_arrived(0, clock=1)
        assert bsp.can_apply(0, 1) is False

        # Clock 1 barrier - complete
        await bsp.worker_arrived(1, clock=1)
        assert bsp.can_apply(0, 1) is True


class TestBSPWaitForBarrier:
    """Tests for wait_for_barrier method."""

    @pytest.mark.asyncio
    async def test_wait_returns_immediately_if_complete(self):
        """Test that wait returns immediately for complete barriers."""
        bsp = BSPConsistency(num_workers=2)

        await bsp.worker_arrived(0, clock=0)
        await bsp.worker_arrived(1, clock=0)

        result = await bsp.wait_for_barrier(0, timeout=0.1)
        assert result is True

    @pytest.mark.asyncio
    async def test_wait_times_out(self):
        """Test that wait times out for incomplete barriers."""
        bsp = BSPConsistency(num_workers=2)

        await bsp.worker_arrived(0, clock=0)
        # Worker 1 hasn't arrived

        result = await bsp.wait_for_barrier(0, timeout=0.05)
        assert result is False

    @pytest.mark.asyncio
    async def test_wait_succeeds_when_workers_arrive(self):
        """Test that wait succeeds when workers arrive."""
        bsp = BSPConsistency(num_workers=2)

        async def complete_barrier():
            await asyncio.sleep(0.01)
            await bsp.worker_arrived(0, clock=0)
            await bsp.worker_arrived(1, clock=0)

        wait_task = asyncio.create_task(bsp.wait_for_barrier(0, timeout=1.0))
        complete_task = asyncio.create_task(complete_barrier())

        await complete_task
        result = await wait_task
        assert result is True


class TestBSPBarrierStatus:
    """Tests for barrier status methods."""

    @pytest.mark.asyncio
    async def test_get_barrier_status(self):
        """Test getting barrier status."""
        bsp = BSPConsistency(num_workers=3)

        await bsp.worker_arrived(0, clock=5)
        await bsp.worker_arrived(1, clock=5)

        status = bsp.get_barrier_status(5)
        assert status["clock"] == 5
        assert status["arrived"] == 2
        assert status["total"] == 3
        assert status["complete"] is False

    @pytest.mark.asyncio
    async def test_get_barrier_status_complete(self):
        """Test barrier status when complete."""
        bsp = BSPConsistency(num_workers=2)

        await bsp.worker_arrived(0, clock=0)
        await bsp.worker_arrived(1, clock=0)

        status = bsp.get_barrier_status(0)
        assert status["complete"] is True

    def test_get_barrier_status_nonexistent(self):
        """Test status for non-existent barrier."""
        bsp = BSPConsistency(num_workers=2)

        status = bsp.get_barrier_status(999)
        assert status["arrived"] == 0


class TestBSPSlowestWorker:
    """Tests for get_slowest_worker method."""

    def test_no_workers_returns_none(self):
        """Test that no workers returns None."""
        bsp = BSPConsistency(num_workers=3)
        assert bsp.get_slowest_worker() is None

    @pytest.mark.asyncio
    async def test_find_slowest_worker(self):
        """Test finding the slowest worker."""
        bsp = BSPConsistency(num_workers=3)

        # Workers 0 and 2 have arrived at clock 0
        await bsp.worker_arrived(0, clock=0)
        await bsp.worker_arrived(2, clock=0)

        # Worker 1 is missing
        slowest = bsp.get_slowest_worker()
        assert slowest == 1


class TestBSPClearBarriers:
    """Tests for clearing barriers."""

    @pytest.mark.asyncio
    async def test_clear_barrier(self):
        """Test clearing a single barrier."""
        bsp = BSPConsistency(num_workers=2)

        await bsp.worker_arrived(0, clock=0)
        await bsp.worker_arrived(1, clock=0)

        bsp.clear_barrier(0)

        # Barrier should be gone
        assert bsp.can_apply(0, 0) is False

    @pytest.mark.asyncio
    async def test_clear_old_barriers(self):
        """Test clearing old barriers."""
        bsp = BSPConsistency(num_workers=2)

        # Create barriers at clocks 0, 1, 2
        for clock in range(3):
            await bsp.worker_arrived(0, clock)
            await bsp.worker_arrived(1, clock)

        # Clear barriers older than clock 2
        cleared = bsp.clear_old_barriers(current_clock=2)
        assert cleared == 2  # Clocks 0 and 1

        # Clock 0 and 1 should be gone
        assert bsp.can_apply(0, 0) is False
        assert bsp.can_apply(0, 1) is False
        # Clock 2 should still exist
        assert bsp.can_apply(0, 2) is True

    def test_reset(self):
        """Test resetting all barriers."""
        bsp = BSPConsistency(num_workers=2)
        bsp.reset()
        assert bsp.pending_barriers == 0


class TestBSPPendingBarriers:
    """Tests for pending_barriers property."""

    @pytest.mark.asyncio
    async def test_pending_barriers_count(self):
        """Test counting pending barriers."""
        bsp = BSPConsistency(num_workers=2)

        # Create incomplete barrier at clock 0
        await bsp.worker_arrived(0, clock=0)

        assert bsp.pending_barriers == 1

        # Complete barrier at clock 0
        await bsp.worker_arrived(1, clock=0)

        assert bsp.pending_barriers == 0

    @pytest.mark.asyncio
    async def test_multiple_pending(self):
        """Test multiple pending barriers."""
        bsp = BSPConsistency(num_workers=2)

        # Incomplete barriers at clocks 0, 1, 2
        await bsp.worker_arrived(0, clock=0)
        await bsp.worker_arrived(0, clock=1)
        await bsp.worker_arrived(0, clock=2)

        assert bsp.pending_barriers == 3


class TestBSPConcurrency:
    """Tests for concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_arrivals(self):
        """Test concurrent worker arrivals."""
        bsp = BSPConsistency(num_workers=10)

        async def worker_arrive(worker_id):
            return await bsp.worker_arrived(worker_id, clock=0)

        # All workers arrive concurrently
        tasks = [worker_arrive(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        # Exactly one should return True (the last one)
        assert sum(results) == 1
        assert bsp.can_apply(0, 0) is True

    @pytest.mark.asyncio
    async def test_concurrent_waits(self):
        """Test concurrent barrier waits."""
        bsp = BSPConsistency(num_workers=3)

        async def wait_and_arrive(worker_id):
            # Start waiting, then arrive
            wait_task = asyncio.create_task(
                bsp.wait_for_barrier(0, timeout=1.0)
            )
            await asyncio.sleep(0.01 * worker_id)
            await bsp.worker_arrived(worker_id, clock=0)
            return await wait_task

        tasks = [wait_and_arrive(i) for i in range(3)]
        results = await asyncio.gather(*tasks)

        assert all(results)  # All waits should succeed
