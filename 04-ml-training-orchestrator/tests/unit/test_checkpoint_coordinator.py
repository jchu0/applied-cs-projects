"""Tests for the distributed checkpoint coordinator."""

import pytest

from ml_orchestrator.checkpoint.coordinator import (
    CheckpointCoordinator,
    CoordinatedCheckpoint,
    WorkerCheckpointState,
    CoordinatedCheckpointState,
)


class TestCoordinatedCheckpointProperties:
    def test_states_and_readiness(self):
        ck = CoordinatedCheckpoint(job_id="j", world_size=2)
        assert ck.state == CoordinatedCheckpointState.PENDING
        assert not ck.is_complete
        assert not ck.all_workers_ready  # no workers yet

        ck.workers["w0"] = WorkerCheckpointState(worker_id="w0", rank=0,
                                                 state=CoordinatedCheckpointState.BARRIER_WAIT)
        ck.workers["w1"] = WorkerCheckpointState(worker_id="w1", rank=1,
                                                 state=CoordinatedCheckpointState.BARRIER_WAIT)
        assert ck.all_workers_ready
        assert not ck.all_workers_saved

        for w in ck.workers.values():
            w.state = CoordinatedCheckpointState.COMPLETED
        assert ck.all_workers_saved

    def test_is_expired(self):
        live = CoordinatedCheckpoint(job_id="j", timeout_seconds=300)
        assert not live.is_expired
        expired = CoordinatedCheckpoint(job_id="j", timeout_seconds=0)
        assert expired.is_expired


class TestCheckpointCoordinator:
    @pytest.mark.asyncio
    async def test_single_worker_happy_path(self):
        coord = CheckpointCoordinator()
        ck = await coord.initiate_checkpoint("job-1", epoch=1, step=10, world_size=1)
        assert (await coord.get_active_checkpoint("job-1")).id == ck.id

        await coord.worker_acknowledge(ck.id, "w0", rank=0)
        assert await coord.wait_for_barrier(ck.id, "w0") is True
        await coord.worker_saved(ck.id, "w0", "/path/w0.ckpt", size_bytes=100)
        assert await coord.wait_for_completion(ck.id) is True

        fetched = await coord.get_checkpoint(ck.id)
        assert fetched.all_workers_saved
        paths = await coord.get_worker_paths(ck.id)
        assert "/path/w0.ckpt" in paths.values() or "/path/w0.ckpt" in paths

    @pytest.mark.asyncio
    async def test_get_missing_and_stats(self):
        coord = CheckpointCoordinator()
        assert await coord.get_checkpoint("missing") is None
        assert await coord.get_active_checkpoint("missing") is None
        assert isinstance(await coord.get_stats(), dict)
        assert isinstance(await coord.check_timeouts(), list)

    @pytest.mark.asyncio
    async def test_cleanup(self):
        coord = CheckpointCoordinator()
        ck = await coord.initiate_checkpoint("job-2", epoch=1, step=1, world_size=1)
        await coord.cleanup_checkpoint(ck.id)
        assert await coord.get_checkpoint(ck.id) is None
