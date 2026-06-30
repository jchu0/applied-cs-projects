"""Tests for GPU topology-aware scheduling."""

import pytest

from gpusched.core.resources import (
    GPU, Node, Pod, Job, Container, Cluster,
    GPUType, GPUResources, PriorityClass, JobState,
    create_gpu, create_node, create_training_job,
)
from gpusched.scheduler.topology import (
    InterconnectType, CommunicationCost, TopologyInfo,
    GPUTopology, TopologyPlugin, TopologyAwareGPUSelector,
    estimate_distributed_training_efficiency,
)
from gpusched.scheduler.scheduler import (
    SchedulingContext, GPUScheduler, SchedulingDecision,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def dgx_node() -> Node:
    """Create a DGX-style node with 8 A100 GPUs."""
    return create_node(
        hostname="dgx-1",
        num_gpus=8,
        gpu_type=GPUType.A100,
        gpu_memory_gb=80.0,
        nvlink_topology="dgx"
    )


@pytest.fixture
def full_mesh_node() -> Node:
    """Create a node with full NVLink mesh."""
    return create_node(
        hostname="full-mesh-1",
        num_gpus=4,
        gpu_type=GPUType.H100,
        gpu_memory_gb=80.0,
        nvlink_topology="full"
    )


@pytest.fixture
def no_nvlink_node() -> Node:
    """Create a node without NVLink."""
    return create_node(
        hostname="no-nvlink-1",
        num_gpus=4,
        gpu_type=GPUType.T4,
        gpu_memory_gb=16.0,
        nvlink_topology="none"
    )


@pytest.fixture
def topology() -> GPUTopology:
    """Create a GPUTopology instance."""
    return GPUTopology()


@pytest.fixture
def cluster_with_dgx(dgx_node) -> Cluster:
    """Create a cluster with a DGX node."""
    cluster = Cluster(cluster_id="test-cluster")
    cluster.add_node(dgx_node)
    return cluster


# ============================================================================
# GPUTopology Tests
# ============================================================================

class TestGPUTopology:
    """Tests for GPUTopology class."""

    def test_analyze_node_dgx(self, dgx_node, topology):
        """Test analyzing DGX node topology."""
        info = topology.analyze_node(dgx_node)

        assert info.node_id == dgx_node.node_id
        assert info.gpu_count == 8

        # Should have 2 NVLink groups (0-3 and 4-7)
        assert len(info.nvlink_groups) == 2
        assert len(info.nvlink_groups[0]) == 4
        assert len(info.nvlink_groups[1]) == 4

        # Should have 2 NUMA nodes
        assert len(info.numa_groups) == 2

    def test_analyze_node_full_mesh(self, full_mesh_node, topology):
        """Test analyzing full mesh node topology."""
        info = topology.analyze_node(full_mesh_node)

        # Should have 1 NVLink group with all GPUs
        assert len(info.nvlink_groups) == 1
        assert len(info.nvlink_groups[0]) == 4

    def test_analyze_node_no_nvlink(self, no_nvlink_node, topology):
        """Test analyzing node without NVLink."""
        info = topology.analyze_node(no_nvlink_node)

        # Each GPU is its own group (no NVLink connections)
        assert len(info.nvlink_groups) == 4
        for group in info.nvlink_groups:
            assert len(group) == 1

    def test_find_nvlink_group_success(self, dgx_node, topology):
        """Test finding NVLink group with available GPUs."""
        gpus = topology.find_nvlink_group(dgx_node, count=4)

        assert gpus is not None
        assert len(gpus) == 4

        # Verify they are NVLink-connected
        gpu_ids = {g.gpu_id for g in gpus}
        for gpu in gpus:
            peers = set(gpu.nvlink_peers)
            other_gpus = gpu_ids - {gpu.gpu_id}
            assert other_gpus.issubset(peers)

    def test_find_nvlink_group_too_many(self, dgx_node, topology):
        """Test finding NVLink group when count exceeds group size."""
        # DGX has 4-GPU NVLink groups, requesting 5 should fail
        gpus = topology.find_nvlink_group(dgx_node, count=5)
        assert gpus is None

    def test_find_nvlink_group_full_mesh(self, full_mesh_node, topology):
        """Test finding NVLink group in full mesh."""
        gpus = topology.find_nvlink_group(full_mesh_node, count=4)

        assert gpus is not None
        assert len(gpus) == 4

    def test_find_numa_local_gpus(self, dgx_node, topology):
        """Test finding NUMA-local GPUs."""
        gpus = topology.find_numa_local_gpus(dgx_node, count=4, numa_node=0)

        assert gpus is not None
        assert len(gpus) == 4
        assert all(g.numa_node == 0 for g in gpus)

    def test_find_numa_local_gpus_wrong_numa(self, dgx_node, topology):
        """Test finding GPUs on non-existent NUMA node."""
        gpus = topology.find_numa_local_gpus(dgx_node, count=4, numa_node=99)
        assert gpus is None

    def test_communication_cost_same_gpu(self, dgx_node, topology):
        """Test communication cost for same GPU."""
        gpu = dgx_node.gpus[0]
        cost = topology.get_communication_cost(gpu, gpu)
        assert cost == 0.0

    def test_communication_cost_nvlink(self, dgx_node, topology):
        """Test communication cost for NVLink-connected GPUs."""
        gpu0 = dgx_node.gpus[0]
        gpu1 = dgx_node.gpus[1]
        cost = topology.get_communication_cost(gpu0, gpu1)
        assert cost == topology.cost_config.nvlink

    def test_communication_cost_pcie(self, dgx_node, topology):
        """Test communication cost for PCIe-connected GPUs."""
        # GPU 0 and GPU 4 are in different NVLink groups
        gpu0 = dgx_node.gpus[0]
        gpu4 = dgx_node.gpus[4]
        cost = topology.get_communication_cost(gpu0, gpu4)
        assert cost == topology.cost_config.pcie_same_node

    def test_communication_cost_cross_node(self, dgx_node, full_mesh_node, topology):
        """Test communication cost for cross-node GPUs."""
        gpu1 = dgx_node.gpus[0]
        gpu2 = full_mesh_node.gpus[0]
        cost = topology.get_communication_cost(gpu1, gpu2)
        assert cost == topology.cost_config.cross_node

    def test_total_communication_cost_all_to_all(self, dgx_node, topology):
        """Test total communication cost for all-to-all pattern."""
        gpus = dgx_node.gpus[:4]  # First NVLink group
        cost = topology.calculate_total_communication_cost(gpus, "all_to_all")

        # 4 GPUs, 6 pairs, all NVLink
        expected = 6 * topology.cost_config.nvlink
        assert cost == pytest.approx(expected, rel=1e-5)

    def test_total_communication_cost_ring(self, dgx_node, topology):
        """Test total communication cost for ring pattern."""
        gpus = dgx_node.gpus[:4]
        cost = topology.calculate_total_communication_cost(gpus, "ring")

        # 4 GPUs in ring, 4 connections, all NVLink
        expected = 4 * topology.cost_config.nvlink
        assert cost == expected

    def test_score_gpu_selection_nvlink(self, dgx_node, topology):
        """Test scoring GPU selection with NVLink."""
        gpus = dgx_node.gpus[:4]  # NVLink-connected
        requirements = GPUResources(count=4, memory_gb=16.0)

        score = topology.score_gpu_selection(gpus, requirements)
        assert score > 70  # Should be high score

    def test_score_gpu_selection_nvlink_required(self, dgx_node, topology):
        """Test scoring with NVLink required."""
        gpus = dgx_node.gpus[:4]  # NVLink-connected
        requirements = GPUResources(count=4, memory_gb=16.0, require_nvlink=True)

        score = topology.score_gpu_selection(gpus, requirements)
        assert score > 80  # Should include NVLink bonus

    def test_score_gpu_selection_nvlink_required_fail(self, dgx_node, topology):
        """Test scoring when NVLink required but not available."""
        # Mix GPUs from different NVLink groups
        gpus = [dgx_node.gpus[0], dgx_node.gpus[1], dgx_node.gpus[4], dgx_node.gpus[5]]
        requirements = GPUResources(count=4, memory_gb=16.0, require_nvlink=True)

        score = topology.score_gpu_selection(gpus, requirements)
        assert score == 0.0  # Fails NVLink requirement

    def test_cache_invalidation(self, dgx_node, topology):
        """Test topology cache invalidation."""
        # Analyze node to populate cache
        info1 = topology.analyze_node(dgx_node)
        assert dgx_node.node_id in topology._node_topology

        # Invalidate cache
        topology.invalidate_cache(dgx_node.node_id)
        assert dgx_node.node_id not in topology._node_topology

        # Re-analyze should work
        info2 = topology.analyze_node(dgx_node)
        assert info2.node_id == info1.node_id


# ============================================================================
# TopologyPlugin Tests
# ============================================================================

class TestTopologyPlugin:
    """Tests for TopologyPlugin scheduling plugin."""

    @pytest.fixture
    def plugin(self):
        return TopologyPlugin()

    @pytest.fixture
    def single_gpu_pod(self):
        container = Container(
            name="trainer",
            image="train:latest",
            gpu_resources=GPUResources(count=1, memory_gb=16.0)
        )
        return Pod(
            pod_id="pod-1",
            name="single-gpu-job",
            namespace="default",
            containers=[container]
        )

    @pytest.fixture
    def multi_gpu_pod(self):
        container = Container(
            name="trainer",
            image="train:latest",
            gpu_resources=GPUResources(count=4, memory_gb=16.0)
        )
        return Pod(
            pod_id="pod-2",
            name="multi-gpu-job",
            namespace="default",
            containers=[container]
        )

    @pytest.fixture
    def nvlink_required_pod(self):
        container = Container(
            name="trainer",
            image="train:latest",
            gpu_resources=GPUResources(count=4, memory_gb=16.0, require_nvlink=True)
        )
        return Pod(
            pod_id="pod-3",
            name="nvlink-job",
            namespace="default",
            containers=[container]
        )

    def test_plugin_name(self, plugin):
        assert plugin.name() == "Topology"

    def test_filter_single_gpu(self, plugin, single_gpu_pod, dgx_node, cluster_with_dgx):
        """Test filter for single GPU (should always pass)."""
        ctx = SchedulingContext(cluster=cluster_with_dgx, pod=single_gpu_pod)
        assert plugin.filter(ctx, dgx_node) is True

    def test_filter_multi_gpu_success(self, plugin, multi_gpu_pod, dgx_node, cluster_with_dgx):
        """Test filter for multi-GPU job with enough resources."""
        ctx = SchedulingContext(cluster=cluster_with_dgx, pod=multi_gpu_pod)
        assert plugin.filter(ctx, dgx_node) is True

    def test_filter_nvlink_required_success(self, plugin, nvlink_required_pod, dgx_node, cluster_with_dgx):
        """Test filter when NVLink is required and available."""
        ctx = SchedulingContext(cluster=cluster_with_dgx, pod=nvlink_required_pod)
        assert plugin.filter(ctx, dgx_node) is True

    def test_filter_nvlink_required_fail(self, plugin, nvlink_required_pod, no_nvlink_node):
        """Test filter when NVLink required but not available."""
        cluster = Cluster(cluster_id="test")
        cluster.add_node(no_nvlink_node)
        ctx = SchedulingContext(cluster=cluster, pod=nvlink_required_pod)
        assert plugin.filter(ctx, no_nvlink_node) is False

    def test_score_single_gpu(self, plugin, single_gpu_pod, dgx_node, cluster_with_dgx):
        """Test scoring for single GPU job."""
        ctx = SchedulingContext(cluster=cluster_with_dgx, pod=single_gpu_pod)
        score = plugin.score(ctx, dgx_node)
        assert score == 50.0  # Neutral score

    def test_score_multi_gpu(self, plugin, multi_gpu_pod, dgx_node, cluster_with_dgx):
        """Test scoring for multi-GPU job."""
        ctx = SchedulingContext(cluster=cluster_with_dgx, pod=multi_gpu_pod)
        score = plugin.score(ctx, dgx_node)
        assert score > 50  # Should favor NVLink-connected GPUs


# ============================================================================
# TopologyAwareGPUSelector Tests
# ============================================================================

class TestTopologyAwareGPUSelector:
    """Tests for TopologyAwareGPUSelector."""

    @pytest.fixture
    def selector(self):
        return TopologyAwareGPUSelector()

    def test_select_single_gpu(self, selector, dgx_node):
        """Test selecting a single GPU."""
        req = GPUResources(count=1, memory_gb=16.0)
        selected = selector.select_gpus(dgx_node, req)

        assert len(selected) == 1
        assert selected[0] in [g.gpu_id for g in dgx_node.gpus]

    def test_select_nvlink_group(self, selector, dgx_node):
        """Test selecting NVLink-connected GPUs."""
        req = GPUResources(count=4, memory_gb=16.0, require_nvlink=True)
        selected = selector.select_gpus(dgx_node, req)

        assert len(selected) == 4

        # Verify all are NVLink-connected
        gpu_map = {g.gpu_id: g for g in dgx_node.gpus}
        for i, gid1 in enumerate(selected):
            for j, gid2 in enumerate(selected):
                if i < j:
                    assert gid2 in gpu_map[gid1].nvlink_peers

    def test_select_with_type_requirement(self, selector, dgx_node):
        """Test selecting GPUs with type requirement."""
        req = GPUResources(count=2, memory_gb=16.0, gpu_type=GPUType.A100)
        selected = selector.select_gpus(dgx_node, req)

        assert len(selected) == 2
        gpu_map = {g.gpu_id: g for g in dgx_node.gpus}
        for gid in selected:
            assert gpu_map[gid].gpu_type == GPUType.A100

    def test_select_wrong_type(self, selector, dgx_node):
        """Test selecting GPUs with wrong type requirement."""
        req = GPUResources(count=2, memory_gb=16.0, gpu_type=GPUType.H100)
        selected = selector.select_gpus(dgx_node, req)

        assert len(selected) == 0  # No H100s on this node

    def test_select_more_than_available(self, selector, dgx_node):
        """Test selecting more GPUs than available."""
        req = GPUResources(count=10, memory_gb=16.0)
        selected = selector.select_gpus(dgx_node, req)

        assert len(selected) == 0  # Only 8 GPUs available

    def test_select_respects_memory(self, selector, no_nvlink_node):
        """Test that selection respects memory requirements."""
        # T4 has 16GB, request 32GB
        req = GPUResources(count=1, memory_gb=32.0)
        selected = selector.select_gpus(no_nvlink_node, req)

        assert len(selected) == 0

    def test_select_prefers_nvlink(self, selector, dgx_node):
        """Test that selector prefers NVLink-connected GPUs."""
        req = GPUResources(count=4, memory_gb=16.0)  # Not required, but preferred
        selected = selector.select_gpus(dgx_node, req)

        assert len(selected) == 4

        # Should select from a single NVLink group
        gpu_map = {g.gpu_id: g for g in dgx_node.gpus}
        local_indices = [gpu_map[gid].local_index for gid in selected]

        # Either all from 0-3 or all from 4-7
        assert all(i < 4 for i in local_indices) or all(i >= 4 for i in local_indices)


# ============================================================================
# Integration with GPUScheduler Tests
# ============================================================================

class TestSchedulerTopologyIntegration:
    """Tests for topology integration with GPUScheduler."""

    def test_scheduler_uses_topology(self, dgx_node):
        """Test that scheduler uses topology-aware selection."""
        cluster = Cluster(cluster_id="test")
        cluster.add_node(dgx_node)

        scheduler = GPUScheduler(cluster)

        # Verify TopologyPlugin is registered
        plugin_names = [p.name() for p, _ in scheduler.plugins]
        assert "Topology" in plugin_names

    def test_schedule_multi_gpu_job(self, dgx_node):
        """Test scheduling a multi-GPU job with topology awareness."""
        cluster = Cluster(cluster_id="test")
        cluster.add_node(dgx_node)

        job = create_training_job(
            name="dist-train",
            num_gpus=4,
            gpu_memory_gb=16.0,
        )
        cluster.submit_job(job)

        scheduler = GPUScheduler(cluster)
        decision = scheduler.schedule_pod(job.pods[0])

        assert decision.success
        assert decision.node_id == dgx_node.node_id
        assert len(decision.gpu_ids) == 4

    def test_schedule_nvlink_required(self, dgx_node):
        """Test scheduling job requiring NVLink."""
        cluster = Cluster(cluster_id="test")
        cluster.add_node(dgx_node)

        container = Container(
            name="trainer",
            image="train:latest",
            gpu_resources=GPUResources(count=4, memory_gb=16.0, require_nvlink=True)
        )
        pod = Pod(
            pod_id="nvlink-pod",
            name="nvlink-job",
            namespace="default",
            containers=[container]
        )
        cluster.pods[pod.pod_id] = pod

        scheduler = GPUScheduler(cluster)
        decision = scheduler.schedule_pod(pod)

        assert decision.success
        assert len(decision.gpu_ids) == 4


# ============================================================================
# Efficiency Estimation Tests
# ============================================================================

class TestEfficiencyEstimation:
    """Tests for distributed training efficiency estimation."""

    def test_efficiency_nvlink_group(self, dgx_node):
        """Test efficiency for NVLink-connected GPUs."""
        gpus = dgx_node.gpus[:4]  # First NVLink group
        metrics = estimate_distributed_training_efficiency(gpus)

        assert metrics["efficiency"] > 0.8  # High efficiency
        assert metrics["nvlink_ratio"] == 1.0  # All pairs NVLink-connected
        assert metrics["numa_locality"] == 1.0  # All on same NUMA

    def test_efficiency_mixed_groups(self, dgx_node):
        """Test efficiency for GPUs from different groups."""
        gpus = [dgx_node.gpus[0], dgx_node.gpus[1], dgx_node.gpus[4], dgx_node.gpus[5]]
        metrics = estimate_distributed_training_efficiency(gpus)

        # Mixed groups have some NVLink (within groups) but not all pairs
        assert metrics["nvlink_ratio"] < 1.0  # Not all pairs NVLink-connected
        assert metrics["nvlink_ratio"] > 0.0  # But some pairs are connected

    def test_efficiency_single_gpu(self, dgx_node):
        """Test efficiency for single GPU (no communication)."""
        gpus = [dgx_node.gpus[0]]
        metrics = estimate_distributed_training_efficiency(gpus)

        assert metrics["efficiency"] == 1.0  # No communication overhead
        assert metrics["all_to_all_cost"] == 0.0
        assert metrics["ring_cost"] == 0.0

    def test_efficiency_empty(self):
        """Test efficiency for empty GPU list."""
        metrics = estimate_distributed_training_efficiency([])

        assert metrics["efficiency"] == 0.0
        assert metrics["nvlink_ratio"] == 0.0


# ============================================================================
# CommunicationCost Tests
# ============================================================================

class TestCommunicationCost:
    """Tests for CommunicationCost configuration."""

    def test_default_costs(self):
        """Test default communication costs."""
        costs = CommunicationCost()

        assert costs.nvlink == 0.1
        assert costs.pcie_same_node == 2.0
        assert costs.cross_node == 10.0
        assert costs.same_gpu == 0.0

    def test_custom_costs(self):
        """Test custom communication costs."""
        costs = CommunicationCost(
            nvlink=0.05,
            pcie_same_node=1.0,
            cross_node=20.0,
        )

        assert costs.nvlink == 0.05
        assert costs.pcie_same_node == 1.0
        assert costs.cross_node == 20.0

    def test_topology_with_custom_costs(self, dgx_node):
        """Test topology analysis with custom costs."""
        costs = CommunicationCost(nvlink=0.2, pcie_same_node=4.0)
        topology = GPUTopology(cost_config=costs)

        gpu0 = dgx_node.gpus[0]
        gpu1 = dgx_node.gpus[1]

        cost = topology.get_communication_cost(gpu0, gpu1)
        assert cost == 0.2  # Custom NVLink cost


# ============================================================================
# Node Creation Tests
# ============================================================================

class TestNodeCreation:
    """Tests for node creation with topology."""

    def test_create_dgx_node(self):
        """Test creating DGX-style node."""
        node = create_node(
            hostname="dgx-test",
            num_gpus=8,
            nvlink_topology="dgx"
        )

        assert len(node.gpus) == 8

        # Check NVLink groups
        for i in range(4):
            gpu = node.gpus[i]
            assert len(gpu.nvlink_peers) == 3  # Connected to 3 others in group
            for peer_id in gpu.nvlink_peers:
                peer = next(g for g in node.gpus if g.gpu_id == peer_id)
                assert peer.local_index < 4  # Same group

    def test_create_full_mesh_node(self):
        """Test creating full mesh node."""
        node = create_node(
            hostname="full-mesh-test",
            num_gpus=4,
            nvlink_topology="full"
        )

        for gpu in node.gpus:
            assert len(gpu.nvlink_peers) == 3  # Connected to all others

    def test_create_no_nvlink_node(self):
        """Test creating node without NVLink."""
        node = create_node(
            hostname="no-nvlink-test",
            num_gpus=4,
            nvlink_topology="none"
        )

        for gpu in node.gpus:
            assert len(gpu.nvlink_peers) == 0

    def test_gpu_numa_assignment(self):
        """Test NUMA node assignment."""
        node = create_node(
            hostname="numa-test",
            num_gpus=8,
        )

        # GPUs 0-3 on NUMA 0, 4-7 on NUMA 1
        for i in range(4):
            assert node.gpus[i].numa_node == 0
        for i in range(4, 8):
            assert node.gpus[i].numa_node == 1

    def test_gpu_local_index(self):
        """Test GPU local index assignment."""
        node = create_node(
            hostname="index-test",
            num_gpus=8,
        )

        for i, gpu in enumerate(node.gpus):
            assert gpu.local_index == i
