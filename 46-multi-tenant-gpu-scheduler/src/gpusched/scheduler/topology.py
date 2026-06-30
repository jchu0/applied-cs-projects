"""GPU Topology-Aware Scheduling.

Provides topology analysis and scheduling plugins for optimal GPU placement
based on NVLink connectivity, NUMA affinity, and PCIe bus topology.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TYPE_CHECKING
import math

from ..core.resources import (
    Cluster, Node, Pod, Job, GPU, GPUResources,
    JobState, GPUType
)

# Note: We define the plugin interface locally to avoid circular imports.
# The main scheduler module will use duck typing for plugin compatibility.

@dataclass
class TopologySchedulingContext:
    """Context for topology-aware scheduling decisions."""
    cluster: Cluster
    pod: Pod
    job: Optional[Job] = None


class InterconnectType(Enum):
    """GPU interconnect types."""
    NVLINK = "nvlink"          # High-bandwidth NVLink (300+ GB/s)
    PCIE_SAME_NODE = "pcie"    # Same-node PCIe (32 GB/s)
    NETWORK = "network"        # Cross-node network (10-100 Gbps)


@dataclass
class CommunicationCost:
    """Communication cost constants for different interconnects."""
    nvlink: float = 0.1           # NVLink: ~300 GB/s
    pcie_same_node: float = 2.0   # PCIe Gen3/4: ~32 GB/s
    cross_node: float = 10.0      # Network: much slower
    same_gpu: float = 0.0         # No communication needed


@dataclass
class TopologyInfo:
    """GPU topology information for a node."""
    node_id: str
    gpu_count: int
    nvlink_groups: list[list[str]]  # Groups of NVLink-connected GPUs
    numa_groups: dict[int, list[str]]  # GPUs by NUMA node
    pcie_groups: dict[str, list[str]]  # GPUs by PCIe root complex


class GPUTopology:
    """Analyze and manage GPU topology within a cluster.

    Builds connectivity graphs from GPU NVLink peer information
    and provides methods for finding optimal GPU placements.
    """

    def __init__(self, cost_config: Optional[CommunicationCost] = None):
        self.cost_config = cost_config or CommunicationCost()
        self._node_topology: dict[str, TopologyInfo] = {}

    def analyze_node(self, node: Node) -> TopologyInfo:
        """Analyze GPU topology for a node."""
        if node.node_id in self._node_topology:
            return self._node_topology[node.node_id]

        # Build NVLink graph
        nvlink_graph = self._build_nvlink_graph(node)

        # Find connected components (NVLink groups)
        nvlink_groups = self._find_nvlink_groups(node.gpus, nvlink_graph)

        # Group by NUMA node
        numa_groups: dict[int, list[str]] = {}
        for gpu in node.gpus:
            if gpu.numa_node not in numa_groups:
                numa_groups[gpu.numa_node] = []
            numa_groups[gpu.numa_node].append(gpu.gpu_id)

        # Group by PCIe root (first part of PCIe address)
        pcie_groups: dict[str, list[str]] = {}
        for gpu in node.gpus:
            if gpu.pcie_bus:
                root = gpu.pcie_bus.split(":")[0] if ":" in gpu.pcie_bus else "default"
            else:
                root = "default"
            if root not in pcie_groups:
                pcie_groups[root] = []
            pcie_groups[root].append(gpu.gpu_id)

        info = TopologyInfo(
            node_id=node.node_id,
            gpu_count=len(node.gpus),
            nvlink_groups=nvlink_groups,
            numa_groups=numa_groups,
            pcie_groups=pcie_groups,
        )

        self._node_topology[node.node_id] = info
        return info

    def _build_nvlink_graph(self, node: Node) -> dict[str, set[str]]:
        """Build NVLink connectivity graph for a node."""
        graph: dict[str, set[str]] = {}

        for gpu in node.gpus:
            if gpu.gpu_id not in graph:
                graph[gpu.gpu_id] = set()
            for peer_id in gpu.nvlink_peers:
                graph[gpu.gpu_id].add(peer_id)
                if peer_id not in graph:
                    graph[peer_id] = set()
                graph[peer_id].add(gpu.gpu_id)

        return graph

    def _find_nvlink_groups(
        self,
        gpus: list[GPU],
        nvlink_graph: dict[str, set[str]]
    ) -> list[list[str]]:
        """Find connected components in NVLink graph."""
        visited: set[str] = set()
        groups: list[list[str]] = []

        gpu_ids = {g.gpu_id for g in gpus}

        for gpu in gpus:
            if gpu.gpu_id in visited:
                continue

            # BFS to find connected component
            group = []
            queue = [gpu.gpu_id]

            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                if current not in gpu_ids:
                    continue

                visited.add(current)
                group.append(current)

                for peer in nvlink_graph.get(current, set()):
                    if peer not in visited and peer in gpu_ids:
                        queue.append(peer)

            if group:
                groups.append(sorted(group))

        return groups

    def find_nvlink_group(
        self,
        node: Node,
        count: int
    ) -> Optional[list[GPU]]:
        """Find a group of NVLink-connected GPUs on a node.

        Args:
            node: Node to search
            count: Number of GPUs needed

        Returns:
            List of GPUs if a suitable group is found, None otherwise
        """
        topo = self.analyze_node(node)

        # Find a group with enough available GPUs
        gpu_map = {g.gpu_id: g for g in node.gpus}

        for group in topo.nvlink_groups:
            available = [
                gpu_map[gid] for gid in group
                if gid in gpu_map and len(gpu_map[gid].allocated_jobs) == 0
            ]
            if len(available) >= count:
                return available[:count]

        return None

    def find_numa_local_gpus(
        self,
        node: Node,
        count: int,
        numa_node: Optional[int] = None
    ) -> Optional[list[GPU]]:
        """Find GPUs on the same NUMA node.

        Args:
            node: Node to search
            count: Number of GPUs needed
            numa_node: Specific NUMA node preference (None for any)

        Returns:
            List of GPUs if found, None otherwise
        """
        topo = self.analyze_node(node)
        gpu_map = {g.gpu_id: g for g in node.gpus}

        numa_nodes_to_check = [numa_node] if numa_node is not None else sorted(topo.numa_groups.keys())

        for numa in numa_nodes_to_check:
            if numa not in topo.numa_groups:
                continue

            available = [
                gpu_map[gid] for gid in topo.numa_groups[numa]
                if gid in gpu_map and len(gpu_map[gid].allocated_jobs) == 0
            ]
            if len(available) >= count:
                return available[:count]

        return None

    def get_communication_cost(
        self,
        gpu1: GPU,
        gpu2: GPU,
        node1: Optional[Node] = None,
        node2: Optional[Node] = None
    ) -> float:
        """Calculate communication cost between two GPUs.

        Args:
            gpu1: First GPU
            gpu2: Second GPU
            node1: Node containing gpu1 (optional, for cross-node detection)
            node2: Node containing gpu2 (optional)

        Returns:
            Communication cost (lower is better)
        """
        if gpu1.gpu_id == gpu2.gpu_id:
            return self.cost_config.same_gpu

        # Check if on different nodes
        if gpu1.node_id != gpu2.node_id:
            return self.cost_config.cross_node

        # Same node - check NVLink
        if gpu2.gpu_id in gpu1.nvlink_peers:
            return self.cost_config.nvlink

        # Same node, no NVLink - use PCIe
        return self.cost_config.pcie_same_node

    def calculate_total_communication_cost(
        self,
        gpus: list[GPU],
        pattern: str = "all_to_all"
    ) -> float:
        """Calculate total communication cost for a GPU group.

        Args:
            gpus: List of GPUs in the group
            pattern: Communication pattern ("all_to_all", "ring", "tree")

        Returns:
            Total communication cost
        """
        if len(gpus) <= 1:
            return 0.0

        total_cost = 0.0

        if pattern == "all_to_all":
            # All-to-all communication (e.g., all-reduce)
            for i, gpu1 in enumerate(gpus):
                for j, gpu2 in enumerate(gpus):
                    if i < j:
                        total_cost += self.get_communication_cost(gpu1, gpu2)

        elif pattern == "ring":
            # Ring communication
            for i in range(len(gpus)):
                next_i = (i + 1) % len(gpus)
                total_cost += self.get_communication_cost(gpus[i], gpus[next_i])

        elif pattern == "tree":
            # Binary tree communication
            for i in range(1, len(gpus)):
                parent_i = (i - 1) // 2
                total_cost += self.get_communication_cost(gpus[i], gpus[parent_i])

        return total_cost

    def score_gpu_selection(
        self,
        gpus: list[GPU],
        requirements: GPUResources
    ) -> float:
        """Score a GPU selection based on topology.

        Higher score is better.

        Args:
            gpus: Selected GPUs
            requirements: Resource requirements

        Returns:
            Score (0-100)
        """
        if not gpus:
            return 0.0

        # Base score
        score = 50.0

        # Check NVLink requirement
        if requirements.require_nvlink:
            # All GPUs must be NVLink-connected
            for i, gpu1 in enumerate(gpus):
                for j, gpu2 in enumerate(gpus):
                    if i < j:
                        if gpu2.gpu_id not in gpu1.nvlink_peers:
                            return 0.0  # Fails NVLink requirement
            score += 30.0  # Bonus for meeting NVLink requirement

        # Calculate communication cost
        comm_cost = self.calculate_total_communication_cost(gpus, "all_to_all")

        # Max possible cost if all cross-node
        max_cost = len(gpus) * (len(gpus) - 1) / 2 * self.cost_config.cross_node
        if max_cost > 0:
            # Lower cost = higher score
            cost_score = (1 - comm_cost / max_cost) * 30
            score += cost_score

        # NUMA locality bonus
        numa_nodes = {g.numa_node for g in gpus}
        if len(numa_nodes) == 1:
            score += 10.0  # All on same NUMA node

        # Same node bonus
        node_ids = {g.node_id for g in gpus}
        if len(node_ids) == 1:
            score += 10.0  # All on same node

        return min(100.0, score)

    def invalidate_cache(self, node_id: Optional[str] = None) -> None:
        """Invalidate topology cache.

        Args:
            node_id: Specific node to invalidate, or None for all
        """
        if node_id:
            self._node_topology.pop(node_id, None)
        else:
            self._node_topology.clear()


class TopologyPlugin:
    """Topology-aware scheduling plugin.

    Scores nodes based on GPU interconnect topology to optimize
    placement for distributed training workloads.

    This class implements the SchedulingPlugin interface via duck typing
    to avoid circular imports.
    """

    def __init__(self, topology: Optional[GPUTopology] = None):
        self.topology = topology or GPUTopology()

    def name(self) -> str:
        return "Topology"

    def filter(self, ctx, node: Node) -> bool:
        """Filter nodes based on topology requirements.

        Args:
            ctx: SchedulingContext with pod and cluster info
            node: Node to evaluate
        """
        requirements = ctx.pod.total_gpu_request

        if requirements.count <= 1:
            return True  # Single GPU doesn't need topology

        # Check NVLink requirement
        if requirements.require_nvlink:
            nvlink_group = self.topology.find_nvlink_group(node, requirements.count)
            if nvlink_group is None:
                return False

        # Check same-node requirement (usually True by default)
        if requirements.require_same_node:
            available_count = sum(
                1 for g in node.gpus if len(g.allocated_jobs) == 0
            )
            if available_count < requirements.count:
                return False

        # Check NUMA preference
        if requirements.numa_preference is not None:
            numa_gpus = self.topology.find_numa_local_gpus(
                node, requirements.count, requirements.numa_preference
            )
            if numa_gpus is None:
                return False

        return True

    def score(self, ctx, node: Node) -> float:
        """Score node based on topology optimization.

        Args:
            ctx: SchedulingContext with pod and cluster info
            node: Node to evaluate
        """
        requirements = ctx.pod.total_gpu_request

        if requirements.count <= 1:
            return 50.0  # Neutral score for single GPU

        # Analyze node topology
        topo = self.topology.analyze_node(node)

        # Get available GPUs
        available_gpus = [g for g in node.gpus if len(g.allocated_jobs) == 0]
        if len(available_gpus) < requirements.count:
            return 0.0

        # Try to find best GPU selection
        best_score = 0.0
        best_gpus: Optional[list[GPU]] = None

        # Strategy 1: Try NVLink groups first
        if requirements.require_nvlink or len(topo.nvlink_groups) > 0:
            nvlink_gpus = self.topology.find_nvlink_group(node, requirements.count)
            if nvlink_gpus:
                score = self.topology.score_gpu_selection(nvlink_gpus, requirements)
                if score > best_score:
                    best_score = score
                    best_gpus = nvlink_gpus

        # Strategy 2: Try NUMA-local placement
        if requirements.numa_preference is not None:
            numa_gpus = self.topology.find_numa_local_gpus(
                node, requirements.count, requirements.numa_preference
            )
            if numa_gpus:
                score = self.topology.score_gpu_selection(numa_gpus, requirements)
                if score > best_score:
                    best_score = score
                    best_gpus = numa_gpus
        else:
            # Try each NUMA node
            for numa in sorted(topo.numa_groups.keys()):
                numa_gpus = self.topology.find_numa_local_gpus(node, requirements.count, numa)
                if numa_gpus:
                    score = self.topology.score_gpu_selection(numa_gpus, requirements)
                    if score > best_score:
                        best_score = score
                        best_gpus = numa_gpus

        # Strategy 3: Any available GPUs (fallback)
        if best_gpus is None:
            selected = available_gpus[:requirements.count]
            best_score = self.topology.score_gpu_selection(selected, requirements)

        return best_score


class TopologyAwareGPUSelector:
    """Select GPUs based on topology for optimal placement."""

    def __init__(self, topology: Optional[GPUTopology] = None):
        self.topology = topology or GPUTopology()

    def select_gpus(
        self,
        node: Node,
        requirements: GPUResources
    ) -> list[str]:
        """Select optimal GPUs based on topology.

        Args:
            node: Node to select GPUs from
            requirements: GPU resource requirements

        Returns:
            List of selected GPU IDs
        """
        if requirements.count <= 0:
            return []

        available_gpus = [g for g in node.gpus if len(g.allocated_jobs) == 0]
        if len(available_gpus) < requirements.count:
            return []

        # Single GPU - just pick one with matching type
        if requirements.count == 1:
            for gpu in available_gpus:
                if requirements.gpu_type is None or gpu.gpu_type == requirements.gpu_type:
                    if requirements.memory_gb <= gpu.available_memory_gb:
                        return [gpu.gpu_id]
            return []

        # Multiple GPUs - use topology-aware selection
        best_gpus: Optional[list[GPU]] = None
        best_score = -1.0

        # Strategy 1: Find NVLink-connected group
        if requirements.require_nvlink:
            nvlink_gpus = self.topology.find_nvlink_group(node, requirements.count)
            if nvlink_gpus:
                # Filter by type and memory
                valid = self._filter_by_requirements(nvlink_gpus, requirements)
                if len(valid) >= requirements.count:
                    score = self.topology.score_gpu_selection(valid[:requirements.count], requirements)
                    if score > best_score:
                        best_score = score
                        best_gpus = valid[:requirements.count]
            # If NVLink required but not found, return empty
            if best_gpus is None:
                return []
        else:
            # Try NVLink groups first for better performance
            nvlink_gpus = self.topology.find_nvlink_group(node, requirements.count)
            if nvlink_gpus:
                valid = self._filter_by_requirements(nvlink_gpus, requirements)
                if len(valid) >= requirements.count:
                    score = self.topology.score_gpu_selection(valid[:requirements.count], requirements)
                    if score > best_score:
                        best_score = score
                        best_gpus = valid[:requirements.count]

        # Strategy 2: Try NUMA-local placement
        topo = self.topology.analyze_node(node)
        for numa in sorted(topo.numa_groups.keys()):
            numa_gpus = self.topology.find_numa_local_gpus(node, requirements.count, numa)
            if numa_gpus:
                valid = self._filter_by_requirements(numa_gpus, requirements)
                if len(valid) >= requirements.count:
                    score = self.topology.score_gpu_selection(valid[:requirements.count], requirements)
                    if score > best_score:
                        best_score = score
                        best_gpus = valid[:requirements.count]

        # Strategy 3: Any available GPUs (fallback)
        if best_gpus is None:
            valid = self._filter_by_requirements(available_gpus, requirements)
            if len(valid) >= requirements.count:
                best_gpus = valid[:requirements.count]

        if best_gpus:
            return [g.gpu_id for g in best_gpus]
        return []

    def _filter_by_requirements(
        self,
        gpus: list[GPU],
        requirements: GPUResources
    ) -> list[GPU]:
        """Filter GPUs by resource requirements."""
        result = []
        for gpu in gpus:
            if requirements.gpu_type and gpu.gpu_type != requirements.gpu_type:
                continue
            if gpu.available_memory_gb < requirements.memory_gb:
                continue
            result.append(gpu)
        return result


def estimate_distributed_training_efficiency(
    gpus: list[GPU],
    topology: Optional[GPUTopology] = None
) -> dict[str, float]:
    """Estimate distributed training efficiency for a GPU selection.

    Returns metrics about expected communication efficiency.

    Args:
        gpus: Selected GPUs for training
        topology: GPUTopology instance (optional)

    Returns:
        Dictionary with efficiency metrics
    """
    if not gpus:
        return {"efficiency": 0.0, "nvlink_ratio": 0.0, "numa_locality": 0.0}

    # Single GPU has perfect efficiency (no distributed communication needed)
    if len(gpus) == 1:
        return {
            "efficiency": 1.0,
            "all_to_all_efficiency": 1.0,
            "ring_efficiency": 1.0,
            "nvlink_ratio": 1.0,
            "numa_locality": 1.0,
            "all_to_all_cost": 0.0,
            "ring_cost": 0.0,
        }

    topo = topology or GPUTopology()

    # Calculate communication cost
    all_to_all_cost = topo.calculate_total_communication_cost(gpus, "all_to_all")
    ring_cost = topo.calculate_total_communication_cost(gpus, "ring")

    # Best possible cost (all NVLink)
    n = len(gpus)
    best_all_to_all = n * (n - 1) / 2 * topo.cost_config.nvlink
    best_ring = n * topo.cost_config.nvlink

    # Worst possible cost (all cross-node)
    worst_all_to_all = n * (n - 1) / 2 * topo.cost_config.cross_node
    worst_ring = n * topo.cost_config.cross_node

    # Calculate efficiency
    if worst_all_to_all > best_all_to_all:
        all_to_all_efficiency = 1 - (all_to_all_cost - best_all_to_all) / (worst_all_to_all - best_all_to_all)
    else:
        all_to_all_efficiency = 1.0

    if worst_ring > best_ring:
        ring_efficiency = 1 - (ring_cost - best_ring) / (worst_ring - best_ring)
    else:
        ring_efficiency = 1.0

    # NVLink ratio
    nvlink_pairs = 0
    total_pairs = 0
    for i, gpu1 in enumerate(gpus):
        for j, gpu2 in enumerate(gpus):
            if i < j:
                total_pairs += 1
                if gpu2.gpu_id in gpu1.nvlink_peers:
                    nvlink_pairs += 1

    nvlink_ratio = nvlink_pairs / total_pairs if total_pairs > 0 else 0.0

    # NUMA locality
    numa_nodes = {g.numa_node for g in gpus}
    numa_locality = 1.0 / len(numa_nodes) if numa_nodes else 0.0

    return {
        "efficiency": (all_to_all_efficiency + ring_efficiency) / 2,
        "all_to_all_efficiency": all_to_all_efficiency,
        "ring_efficiency": ring_efficiency,
        "nvlink_ratio": nvlink_ratio,
        "numa_locality": numa_locality,
        "all_to_all_cost": all_to_all_cost,
        "ring_cost": ring_cost,
    }
