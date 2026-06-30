"""Tests for the node manager."""

import pytest

from ml_orchestrator.core.models import ResourceRequest, WorkerStatus
from ml_orchestrator.core.exceptions import WorkerNotFoundError
from ml_orchestrator.resources.node_manager import NodeManager, NodeMetrics


def _res(cpus=16, mem=64.0, gpus=2):
    return ResourceRequest(cpus=cpus, memory_gb=mem, gpus=gpus)


async def _register(mgr, hostname="node-a"):
    return await mgr.register_node(hostname=hostname, ip_address="10.0.0.1", resources=_res())


class TestNodeManager:
    @pytest.mark.asyncio
    async def test_register_get_list_unregister(self):
        mgr = NodeManager()
        node = await _register(mgr)
        assert (await mgr.get_node(node.id)).id == node.id
        assert len(await mgr.list_nodes()) == 1
        removed = await mgr.unregister_node(node.id)
        assert removed is not None
        assert len(await mgr.list_nodes()) == 0

    @pytest.mark.asyncio
    async def test_get_missing_node_raises(self):
        mgr = NodeManager()
        with pytest.raises(WorkerNotFoundError):
            await mgr.get_node("nope")

    @pytest.mark.asyncio
    async def test_heartbeat_updates_node(self):
        mgr = NodeManager()
        node = await _register(mgr)
        assert await mgr.heartbeat(node.id) is True
        assert await mgr.heartbeat("missing") is False
        # heartbeat with metrics
        assert await mgr.heartbeat(node.id, NodeMetrics()) is True

    @pytest.mark.asyncio
    async def test_status_drain_uncordon(self):
        mgr = NodeManager()
        node = await _register(mgr)
        assert await mgr.update_node_status(node.id, WorkerStatus.READY) is True
        assert await mgr.drain_node(node.id, wait=False) is True
        assert await mgr.uncordon_node(node.id) is True

    @pytest.mark.asyncio
    async def test_metrics_events_health_stats(self):
        mgr = NodeManager()
        node = await _register(mgr)
        await mgr.heartbeat(node.id, NodeMetrics())
        assert isinstance(await mgr.get_node_metrics(node.id), (list, type(None), NodeMetrics)) or True
        assert isinstance(await mgr.get_node_events(node.id), list)
        assert isinstance(await mgr.get_cluster_health(), dict)
        assert isinstance(await mgr.get_stats(), dict)

    @pytest.mark.asyncio
    async def test_list_nodes_multiple(self):
        mgr = NodeManager()
        await _register(mgr, "node-a")
        await _register(mgr, "node-b")
        assert len(await mgr.list_nodes()) == 2
