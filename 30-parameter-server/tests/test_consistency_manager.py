"""Tests for ConsistencyManager."""

import pytest
import asyncio

from paramserver.consistency.manager import (
    ConsistencyManager,
    ConsistencyType,
    create_consistency_model,
)
from paramserver.consistency.hogwild import HogwildConsistency
from paramserver.consistency.bsp import BSPConsistency
from paramserver.consistency.ssp import SSPConsistency


class TestConsistencyManagerInit:
    """Tests for ConsistencyManager initialization."""

    def test_create_hogwild_from_string(self):
        """Test creating Hogwild from string type."""
        manager = ConsistencyManager("hogwild")
        assert manager.consistency_type == ConsistencyType.HOGWILD
        assert isinstance(manager.model, HogwildConsistency)

    def test_create_hogwild_from_enum(self):
        """Test creating Hogwild from enum type."""
        manager = ConsistencyManager(ConsistencyType.HOGWILD)
        assert isinstance(manager.model, HogwildConsistency)

    def test_create_bsp(self):
        """Test creating BSP manager."""
        manager = ConsistencyManager("bsp", num_workers=4)
        assert manager.consistency_type == ConsistencyType.BSP
        assert isinstance(manager.model, BSPConsistency)
        assert manager.model.num_workers == 4

    def test_create_ssp(self):
        """Test creating SSP manager."""
        manager = ConsistencyManager("ssp", staleness_threshold=5)
        assert manager.consistency_type == ConsistencyType.SSP
        assert isinstance(manager.model, SSPConsistency)
        assert manager.model.staleness_threshold == 5

    def test_case_insensitive(self):
        """Test that type string is case insensitive."""
        manager1 = ConsistencyManager("HOGWILD")
        manager2 = ConsistencyManager("HogWild")
        manager3 = ConsistencyManager("hogwild")

        assert all(
            m.consistency_type == ConsistencyType.HOGWILD
            for m in [manager1, manager2, manager3]
        )

    def test_invalid_type(self):
        """Test that invalid type raises error."""
        with pytest.raises(ValueError, match="Unknown consistency type"):
            ConsistencyManager("invalid_type")


class TestConsistencyManagerFactoryMethods:
    """Tests for factory methods."""

    def test_create_hogwild(self):
        """Test create_hogwild factory."""
        manager = ConsistencyManager.create_hogwild()
        assert manager.consistency_type == ConsistencyType.HOGWILD

    def test_create_bsp(self):
        """Test create_bsp factory."""
        manager = ConsistencyManager.create_bsp(num_workers=5)
        assert manager.consistency_type == ConsistencyType.BSP
        assert manager.num_workers == 5

    def test_create_ssp(self):
        """Test create_ssp factory."""
        manager = ConsistencyManager.create_ssp(
            num_workers=3,
            staleness_threshold=4,
        )
        assert manager.consistency_type == ConsistencyType.SSP


class TestConsistencyManagerCanApply:
    """Tests for can_apply_gradient method."""

    def test_hogwild_always_allows(self):
        """Test Hogwild always allows gradients."""
        manager = ConsistencyManager.create_hogwild()

        assert manager.can_apply_gradient(param_version=0, worker_clock=100) is True
        assert manager.can_apply_gradient(param_version=100, worker_clock=0) is True

    @pytest.mark.asyncio
    async def test_bsp_requires_all_workers(self):
        """Test BSP requires all workers."""
        manager = ConsistencyManager.create_bsp(num_workers=2)

        # No workers arrived
        assert manager.can_apply_gradient(0, 0) is False

        # One worker
        await manager.on_worker_update(0, clock=0)
        assert manager.can_apply_gradient(0, 0) is False

        # Both workers
        await manager.on_worker_update(1, clock=0)
        assert manager.can_apply_gradient(0, 0) is True

    @pytest.mark.asyncio
    async def test_ssp_staleness_check(self):
        """Test SSP staleness checking."""
        manager = ConsistencyManager.create_ssp(staleness_threshold=3)

        await manager.on_worker_update(0, clock=10)
        await manager.on_worker_update(1, clock=6)

        # Clock 10 is 4 ahead of min (6), exceeds threshold 3
        assert manager.can_apply_gradient(0, 10) is False

        # Clock 8 is 2 ahead of min (6), within threshold
        assert manager.can_apply_gradient(0, 8) is True


class TestConsistencyManagerOnWorkerUpdate:
    """Tests for on_worker_update method."""

    @pytest.mark.asyncio
    async def test_hogwild_no_tracking(self):
        """Test Hogwild doesn't track updates (no-op)."""
        manager = ConsistencyManager.create_hogwild()

        # Should not raise
        await manager.on_worker_update(0, clock=5)
        await manager.on_worker_update(1, clock=10)

    @pytest.mark.asyncio
    async def test_bsp_barrier_tracking(self):
        """Test BSP tracks barriers."""
        manager = ConsistencyManager.create_bsp(num_workers=2)

        await manager.on_worker_update(0, clock=0)
        status = manager.model.get_barrier_status(0)
        assert status["arrived"] == 1

        await manager.on_worker_update(1, clock=0)
        status = manager.model.get_barrier_status(0)
        assert status["complete"] is True

    @pytest.mark.asyncio
    async def test_ssp_clock_tracking(self):
        """Test SSP tracks worker clocks."""
        manager = ConsistencyManager.create_ssp()

        await manager.on_worker_update(0, clock=5)
        await manager.on_worker_update(1, clock=10)

        assert manager.model.get_worker_clock(0) == 5
        assert manager.model.get_worker_clock(1) == 10


class TestConsistencyManagerWaitIfNeeded:
    """Tests for wait_if_needed method."""

    @pytest.mark.asyncio
    async def test_hogwild_never_waits(self):
        """Test Hogwild never waits."""
        manager = ConsistencyManager.create_hogwild()

        result = await manager.wait_if_needed(0, target_clock=100, timeout=0.01)
        assert result is True

    @pytest.mark.asyncio
    async def test_bsp_waits_for_barrier(self):
        """Test BSP waits for barrier."""
        manager = ConsistencyManager.create_bsp(num_workers=2)

        # Start waiting, then complete barrier
        async def complete_barrier():
            await asyncio.sleep(0.01)
            await manager.on_worker_update(0, clock=0)
            await manager.on_worker_update(1, clock=0)

        wait_task = asyncio.create_task(
            manager.wait_if_needed(0, target_clock=0, timeout=1.0)
        )
        complete_task = asyncio.create_task(complete_barrier())

        await complete_task
        result = await wait_task
        assert result is True

    @pytest.mark.asyncio
    async def test_ssp_waits_for_staleness(self):
        """Test SSP waits for staleness."""
        manager = ConsistencyManager.create_ssp(staleness_threshold=2)

        await manager.on_worker_update(0, clock=10)
        await manager.on_worker_update(1, clock=3)  # Very behind

        # Worker 0 at clock 10 is too far ahead
        # Should timeout waiting
        result = await manager.wait_if_needed(0, target_clock=10, timeout=0.05)
        assert result is False


class TestConsistencyManagerGetStats:
    """Tests for get_stats method."""

    def test_hogwild_stats(self):
        """Test Hogwild stats."""
        manager = ConsistencyManager.create_hogwild()
        stats = manager.get_stats()

        assert stats["type"] == "hogwild"
        assert stats["num_workers"] == 1

    @pytest.mark.asyncio
    async def test_bsp_stats(self):
        """Test BSP stats."""
        manager = ConsistencyManager.create_bsp(num_workers=3)

        await manager.on_worker_update(0, clock=0)

        stats = manager.get_stats()
        assert stats["type"] == "bsp"
        assert stats["num_workers"] == 3
        assert stats["pending_barriers"] == 1

    @pytest.mark.asyncio
    async def test_ssp_stats(self):
        """Test SSP stats."""
        manager = ConsistencyManager.create_ssp(staleness_threshold=5)

        await manager.on_worker_update(0, clock=10)
        await manager.on_worker_update(1, clock=5)

        stats = manager.get_stats()
        assert stats["type"] == "ssp"
        assert stats["threshold"] == 5
        assert stats["min_clock"] == 5
        assert stats["max_clock"] == 10
        assert stats["current_staleness"] == 5


class TestConsistencyManagerReset:
    """Tests for reset method."""

    @pytest.mark.asyncio
    async def test_reset_bsp(self):
        """Test resetting BSP manager."""
        manager = ConsistencyManager.create_bsp(num_workers=2)

        await manager.on_worker_update(0, clock=0)
        manager.reset()

        stats = manager.get_stats()
        assert stats["pending_barriers"] == 0

    @pytest.mark.asyncio
    async def test_reset_ssp(self):
        """Test resetting SSP manager."""
        manager = ConsistencyManager.create_ssp()

        await manager.on_worker_update(0, clock=10)
        manager.reset()

        assert manager.model.get_min_clock() == 0


class TestCreateConsistencyModelFunction:
    """Tests for create_consistency_model factory function."""

    def test_create_hogwild(self):
        """Test creating Hogwild model."""
        model = create_consistency_model("hogwild")
        assert isinstance(model, HogwildConsistency)

    def test_create_bsp(self):
        """Test creating BSP model."""
        model = create_consistency_model("bsp", num_workers=4)
        assert isinstance(model, BSPConsistency)
        assert model.num_workers == 4

    def test_create_ssp(self):
        """Test creating SSP model."""
        model = create_consistency_model("ssp", staleness_threshold=5)
        assert isinstance(model, SSPConsistency)
        assert model.staleness_threshold == 5

    def test_case_insensitive(self):
        """Test case insensitivity."""
        model1 = create_consistency_model("HOGWILD")
        model2 = create_consistency_model("hogwild")
        assert type(model1) == type(model2)

    def test_invalid_type(self):
        """Test invalid type raises error."""
        with pytest.raises(ValueError, match="Unknown model type"):
            create_consistency_model("invalid")


class TestConsistencyType:
    """Tests for ConsistencyType enum."""

    def test_enum_values(self):
        """Test enum values."""
        assert ConsistencyType.HOGWILD.value == "hogwild"
        assert ConsistencyType.BSP.value == "bsp"
        assert ConsistencyType.SSP.value == "ssp"

    def test_from_string(self):
        """Test creating from string."""
        assert ConsistencyType("hogwild") == ConsistencyType.HOGWILD
        assert ConsistencyType("bsp") == ConsistencyType.BSP
        assert ConsistencyType("ssp") == ConsistencyType.SSP
