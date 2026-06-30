"""Tests for GPU info, pools, and the GPU manager."""

import pytest

from ml_orchestrator.resources.gpu_manager import (
    GPUInfo,
    GPUPool,
    GPUManager,
    GPUStatus,
)


def _gpu(node="node-1", idx=0, gpu_type="A100", mem=80.0):
    return GPUInfo(
        node_id=node, device_index=idx, name="A100",
        gpu_type=gpu_type, memory_total_gb=mem, memory_free_gb=mem,
    )


class TestGPUInfo:
    def test_type_and_memory_predicates(self):
        g = _gpu()
        assert g.is_available
        assert g.matches_type("a100") and g.matches_type(None)
        assert not g.matches_type("V100")
        assert g.has_memory(40.0) and g.has_memory(None)
        assert not g.has_memory(100.0)

    def test_memory_utilization_and_to_dict(self):
        g = _gpu()
        g.memory_used_gb = 40.0
        assert g.memory_utilization == 50.0
        assert GPUInfo(memory_total_gb=0).memory_utilization == 0.0
        d = g.to_dict()
        assert d["id"] == g.id and d["gpu_type"] == "A100"


class TestGPUPool:
    @pytest.mark.asyncio
    async def test_add_get_list_remove(self):
        pool = GPUPool("p")
        g = _gpu()
        await pool.add_gpu(g)
        assert await pool.get_gpu(g.id) is g
        assert len(await pool.list_gpus()) == 1
        assert await pool.remove_gpu(g.id) is g
        assert await pool.get_gpu(g.id) is None

    @pytest.mark.asyncio
    async def test_available_count_by_type(self):
        pool = GPUPool()
        for i in range(3):
            await pool.add_gpu(_gpu(idx=i))
        assert await pool.get_available_count() == 3
        assert await pool.get_available_count(gpu_type="A100") == 3
        assert await pool.get_available_count(gpu_type="V100") == 0

    @pytest.mark.asyncio
    async def test_allocate_and_release(self):
        pool = GPUPool()
        for i in range(4):
            await pool.add_gpu(_gpu(idx=i))
        alloc = await pool.allocate(job_id="job-1", count=2, gpu_type="A100")
        assert len(alloc.gpu_ids) == 2
        assert await pool.get_available_count() == 2
        assert (await pool.get_allocation("job-1")).job_id == "job-1"
        assert await pool.release("job-1") == 2
        assert await pool.get_available_count() == 4

    @pytest.mark.asyncio
    async def test_allocate_insufficient_raises(self):
        pool = GPUPool()
        await pool.add_gpu(_gpu())
        with pytest.raises(Exception):
            await pool.allocate(job_id="j", count=5)

    @pytest.mark.asyncio
    async def test_update_stats_and_get_stats(self):
        pool = GPUPool()
        g = _gpu()
        await pool.add_gpu(g)
        assert await pool.update_gpu_stats(g.id, utilization=50.0, memory_used_gb=10.0)
        assert not await pool.update_gpu_stats("missing", utilization=1.0)
        updated = await pool.get_gpu(g.id)
        assert updated.utilization_percent == 50.0
        assert isinstance(await pool.get_stats(), dict)


class TestGPUManager:
    @pytest.mark.asyncio
    async def test_pool_lifecycle(self):
        m = GPUManager()
        p = await m.create_pool("custom")
        assert await m.get_pool("custom") is p
        assert await m.get_pool() is not None  # default always present
        assert await m.delete_pool("custom") is True
        assert await m.delete_pool("default") is False  # default protected

    @pytest.mark.asyncio
    async def test_register_gpu_into_default_pool(self):
        m = GPUManager()
        g = await m.register_gpu(
            node_id="n1", device_index=0, name="A100", gpu_type="A100", memory_gb=80.0
        )
        assert g.gpu_type == "A100"
        pool = await m.get_pool()
        assert len(await pool.list_gpus()) == 1
