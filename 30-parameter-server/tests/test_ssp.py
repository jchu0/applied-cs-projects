"""Tests for SSP consistency model."""

import pytest
import asyncio

from paramserver.consistency.ssp import SSPConsistency


class TestSSPInit:
    """Tests for SSP initialization."""

    def test_create_ssp(self):
        """Test SSP creation with default threshold."""
        ssp = SSPConsistency()
        assert ssp.staleness_threshold == 3

    def test_create_with_custom_threshold(self):
        """Test SSP creation with custom threshold."""
        ssp = SSPConsistency(staleness_threshold=5)
        assert ssp.staleness_threshold == 5

    def test_invalid_threshold(self):
        """Test that negative threshold raises error."""
        with pytest.raises(ValueError, match="staleness_threshold must be >= 0"):
            SSPConsistency(staleness_threshold=-1)

    def test_zero_threshold(self):
        """Test zero threshold (equivalent to BSP)."""
        ssp = SSPConsistency(staleness_threshold=0)
        assert ssp.staleness_threshold == 0

    def test_name(self):
        """Test SSP name property."""
        ssp = SSPConsistency()
        assert ssp.name == "SSPConsistency"


class TestSSPCanApply:
    """Tests for can_apply method."""

    def test_can_apply_no_workers(self):
        """Test that updates can apply when no workers registered."""
        ssp = SSPConsistency(staleness_threshold=3)
        assert ssp.can_apply(param_version=0, worker_clock=10) is True

    @pytest.mark.asyncio
    async def test_can_apply_within_threshold(self):
        """Test updates within threshold."""
        ssp = SSPConsistency(staleness_threshold=3)

        await ssp.update_worker_clock(0, clock=10)
        await ssp.update_worker_clock(1, clock=8)

        # Worker at clock 10 is 2 ahead of min (8), threshold is 3
        assert ssp.can_apply(0, 10) is True
        # Clock 11 would be 3 ahead (equal to threshold)
        assert ssp.can_apply(0, 11) is True

    @pytest.mark.asyncio
    async def test_cannot_apply_beyond_threshold(self):
        """Test updates beyond threshold."""
        ssp = SSPConsistency(staleness_threshold=3)

        await ssp.update_worker_clock(0, clock=10)
        await ssp.update_worker_clock(1, clock=5)

        # Clock 10 is 5 ahead of min (5), exceeds threshold of 3
        assert ssp.can_apply(0, 10) is False

    @pytest.mark.asyncio
    async def test_threshold_boundary(self):
        """Test exactly at threshold boundary."""
        ssp = SSPConsistency(staleness_threshold=3)

        await ssp.update_worker_clock(0, clock=5)
        await ssp.update_worker_clock(1, clock=2)

        # Clock 5 is exactly 3 ahead of min (2)
        assert ssp.can_apply(0, 5) is True
        # Clock 6 is 4 ahead, exceeds threshold
        assert ssp.can_apply(0, 6) is False


class TestSSPUpdateWorkerClock:
    """Tests for update_worker_clock method."""

    @pytest.mark.asyncio
    async def test_update_clock(self):
        """Test basic clock update."""
        ssp = SSPConsistency()

        await ssp.update_worker_clock(0, clock=5)
        assert ssp.get_worker_clock(0) == 5

    @pytest.mark.asyncio
    async def test_update_clock_monotonic(self):
        """Test that clock updates are monotonic (max)."""
        ssp = SSPConsistency()

        await ssp.update_worker_clock(0, clock=5)
        await ssp.update_worker_clock(0, clock=3)  # Lower value

        # Should keep the higher value
        assert ssp.get_worker_clock(0) == 5

    @pytest.mark.asyncio
    async def test_multiple_workers(self):
        """Test updating multiple workers."""
        ssp = SSPConsistency()

        await ssp.update_worker_clock(0, clock=10)
        await ssp.update_worker_clock(1, clock=8)
        await ssp.update_worker_clock(2, clock=12)

        assert ssp.get_worker_clock(0) == 10
        assert ssp.get_worker_clock(1) == 8
        assert ssp.get_worker_clock(2) == 12


class TestSSPWaitForStaleness:
    """Tests for wait_for_staleness method."""

    @pytest.mark.asyncio
    async def test_wait_returns_immediately_if_ok(self):
        """Test wait returns immediately when staleness is acceptable."""
        ssp = SSPConsistency(staleness_threshold=5)

        await ssp.update_worker_clock(0, clock=10)
        await ssp.update_worker_clock(1, clock=8)

        # Worker 0 at clock 10 is within threshold
        result = await ssp.wait_for_staleness(0, 10, timeout=0.1)
        assert result is True

    @pytest.mark.asyncio
    async def test_wait_times_out(self):
        """Test wait times out when staleness persists."""
        ssp = SSPConsistency(staleness_threshold=2)

        await ssp.update_worker_clock(0, clock=10)
        await ssp.update_worker_clock(1, clock=3)  # Very behind

        # Trying to go to clock 10 with min at 3, staleness 7 > threshold 2
        result = await ssp.wait_for_staleness(0, 10, timeout=0.05)
        assert result is False

    @pytest.mark.asyncio
    async def test_wait_succeeds_when_slow_worker_catches_up(self):
        """Test wait succeeds when slow worker advances."""
        ssp = SSPConsistency(staleness_threshold=2)

        await ssp.update_worker_clock(0, clock=10)
        await ssp.update_worker_clock(1, clock=3)

        async def slow_worker_catches_up():
            await asyncio.sleep(0.01)
            await ssp.update_worker_clock(1, clock=8)  # Now within threshold

        wait_task = asyncio.create_task(
            ssp.wait_for_staleness(0, 10, timeout=1.0)
        )
        catchup_task = asyncio.create_task(slow_worker_catches_up())

        await catchup_task
        result = await wait_task
        assert result is True


class TestSSPClockStats:
    """Tests for clock statistics methods."""

    @pytest.mark.asyncio
    async def test_get_min_clock(self):
        """Test getting minimum clock."""
        ssp = SSPConsistency()

        assert ssp.get_min_clock() == 0  # No workers

        await ssp.update_worker_clock(0, clock=10)
        await ssp.update_worker_clock(1, clock=5)
        await ssp.update_worker_clock(2, clock=8)

        assert ssp.get_min_clock() == 5

    @pytest.mark.asyncio
    async def test_get_max_clock(self):
        """Test getting maximum clock."""
        ssp = SSPConsistency()

        assert ssp.get_max_clock() == 0  # No workers

        await ssp.update_worker_clock(0, clock=10)
        await ssp.update_worker_clock(1, clock=5)
        await ssp.update_worker_clock(2, clock=8)

        assert ssp.get_max_clock() == 10

    @pytest.mark.asyncio
    async def test_get_current_staleness(self):
        """Test getting current staleness."""
        ssp = SSPConsistency()

        assert ssp.get_current_staleness() == 0  # No workers

        await ssp.update_worker_clock(0, clock=10)
        await ssp.update_worker_clock(1, clock=3)

        assert ssp.get_current_staleness() == 7

    @pytest.mark.asyncio
    async def test_get_staleness_for_worker(self):
        """Test getting staleness for specific worker."""
        ssp = SSPConsistency()

        await ssp.update_worker_clock(0, clock=10)
        await ssp.update_worker_clock(1, clock=5)

        # Worker 0 is 0 behind max (10)
        assert ssp.get_staleness_for_worker(0) == 0
        # Worker 1 is 5 behind max (10)
        assert ssp.get_staleness_for_worker(1) == 5


class TestSSPWorkerManagement:
    """Tests for worker registration methods."""

    def test_register_worker(self):
        """Test registering a worker."""
        ssp = SSPConsistency()

        ssp.register_worker(5, initial_clock=10)
        assert ssp.get_worker_clock(5) == 10

    def test_unregister_worker(self):
        """Test unregistering a worker."""
        ssp = SSPConsistency()

        ssp.register_worker(0, initial_clock=5)
        ssp.register_worker(1, initial_clock=10)

        ssp.unregister_worker(0)

        assert ssp.get_worker_clock(0) == 0  # Returns default
        assert ssp.get_worker_clock(1) == 10

    def test_reset(self):
        """Test resetting SSP state."""
        ssp = SSPConsistency()

        ssp.register_worker(0, initial_clock=5)
        ssp.register_worker(1, initial_clock=10)

        ssp.reset()

        assert ssp.get_min_clock() == 0
        assert ssp.get_max_clock() == 0


class TestSSPIsWorkerBlocked:
    """Tests for is_worker_blocked method."""

    @pytest.mark.asyncio
    async def test_worker_not_blocked(self):
        """Test worker not blocked within threshold."""
        ssp = SSPConsistency(staleness_threshold=5)

        await ssp.update_worker_clock(0, clock=10)
        await ssp.update_worker_clock(1, clock=8)

        # Worker trying to reach clock 12 (4 ahead of min)
        assert ssp.is_worker_blocked(0, target_clock=12) is False

    @pytest.mark.asyncio
    async def test_worker_blocked(self):
        """Test worker blocked beyond threshold."""
        ssp = SSPConsistency(staleness_threshold=3)

        await ssp.update_worker_clock(0, clock=10)
        await ssp.update_worker_clock(1, clock=5)

        # Worker trying to reach clock 10 (5 ahead of min)
        assert ssp.is_worker_blocked(0, target_clock=10) is True


class TestSSPGetStats:
    """Tests for get_stats method."""

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test getting SSP statistics."""
        ssp = SSPConsistency(staleness_threshold=4)

        await ssp.update_worker_clock(0, clock=10)
        await ssp.update_worker_clock(1, clock=6)
        await ssp.update_worker_clock(2, clock=8)

        stats = ssp.get_stats()

        assert stats["threshold"] == 4
        assert stats["num_workers"] == 3
        assert stats["min_clock"] == 6
        assert stats["max_clock"] == 10
        assert stats["current_staleness"] == 4
        assert stats["worker_clocks"] == {0: 10, 1: 6, 2: 8}


class TestSSPConcurrency:
    """Tests for concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_clock_updates(self):
        """Test concurrent clock updates."""
        ssp = SSPConsistency()

        async def update_clock(worker_id, clock):
            await ssp.update_worker_clock(worker_id, clock)

        # Multiple concurrent updates
        tasks = [
            update_clock(i, i * 10) for i in range(10)
        ]
        await asyncio.gather(*tasks)

        assert ssp.get_min_clock() == 0
        assert ssp.get_max_clock() == 90

    @pytest.mark.asyncio
    async def test_concurrent_waits(self):
        """Test concurrent staleness waits."""
        ssp = SSPConsistency(staleness_threshold=5)

        # Register workers
        for i in range(5):
            ssp.register_worker(i, initial_clock=10)

        async def worker_advance(worker_id):
            await asyncio.sleep(0.01 * worker_id)
            await ssp.update_worker_clock(worker_id, 15)
            return True

        # All workers advance concurrently
        tasks = [worker_advance(i) for i in range(5)]
        results = await asyncio.gather(*tasks)

        assert all(results)
        assert ssp.get_min_clock() == 15
