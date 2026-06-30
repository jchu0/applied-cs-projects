"""GPU scheduling algorithms."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import heapq

from ..core.resources import (
    Cluster, Node, Pod, Job, GPU, Queue, Tenant,
    JobState, PriorityClass, GPUResources
)
from .topology import TopologyPlugin, TopologyAwareGPUSelector, GPUTopology


@dataclass
class SchedulingDecision:
    """Result of scheduling decision."""
    pod_id: str
    node_id: str | None
    gpu_ids: list[str]
    success: bool
    reason: str = ""
    score: float = 0.0


@dataclass
class SchedulingContext:
    """Context for scheduling decisions."""
    cluster: Cluster
    pod: Pod
    job: Job | None = None
    preemption_candidates: list[Pod] | None = None


class SchedulingPlugin(ABC):
    """Base class for scheduling plugins."""

    @abstractmethod
    def name(self) -> str:
        """Get plugin name."""
        pass

    @abstractmethod
    def filter(self, ctx: SchedulingContext, node: Node) -> bool:
        """Filter nodes that can run the pod."""
        pass

    @abstractmethod
    def score(self, ctx: SchedulingContext, node: Node) -> float:
        """Score nodes (higher is better)."""
        pass


class NodeAffinityPlugin(SchedulingPlugin):
    """Filter and score based on node affinity."""

    def name(self) -> str:
        return "NodeAffinity"

    def filter(self, ctx: SchedulingContext, node: Node) -> bool:
        # Check node selector
        for key, value in ctx.pod.node_selector.items():
            if node.labels.get(key) != value:
                return False

        # Check tolerations for taints
        for taint in node.taints:
            if taint not in ctx.pod.tolerations:
                if "NoSchedule" in taint:
                    return False
        return True

    def score(self, ctx: SchedulingContext, node: Node) -> float:
        score = 0.0
        # Prefer nodes matching preferred affinity
        affinity = ctx.pod.affinity.get("nodeAffinity", {})
        preferred = affinity.get("preferredDuringScheduling", [])
        for pref in preferred:
            weight = pref.get("weight", 1)
            match_expr = pref.get("matchExpressions", [])
            for expr in match_expr:
                key = expr.get("key")
                if key in node.labels:
                    score += weight
        return score


class GPUResourcePlugin(SchedulingPlugin):
    """Filter and score based on GPU resources."""

    def name(self) -> str:
        return "GPUResource"

    def filter(self, ctx: SchedulingContext, node: Node) -> bool:
        required = ctx.pod.total_gpu_request

        # Per-GPU requirements (each GPU needs to satisfy memory/compute, count is for total GPUs needed)
        per_gpu_req = GPUResources(
            count=1,
            memory_gb=required.memory_gb,
            compute_units=required.compute_units / max(1, required.count),
            gpu_type=required.gpu_type
        )

        # Check GPU count - each GPU must be able to satisfy per-GPU requirements
        available_gpus = [g for g in node.gpus if g.can_allocate(per_gpu_req)]
        if len(available_gpus) < required.count:
            return False

        # Check GPU type if specified
        if required.gpu_type:
            typed_gpus = [g for g in available_gpus
                         if g.gpu_type == required.gpu_type]
            if len(typed_gpus) < required.count:
                return False

        return True

    def score(self, ctx: SchedulingContext, node: Node) -> float:
        required = ctx.pod.total_gpu_request

        # Score based on remaining resources (bin packing)
        total_available = sum(g.available_memory_gb for g in node.gpus)
        total_capacity = sum(g.total_memory_gb for g in node.gpus)

        if total_capacity == 0:
            return 0.0

        # Higher score for nodes that will be more utilized
        utilization_after = (total_capacity - total_available +
                            required.memory_gb * required.count) / total_capacity
        return utilization_after * 100


class BinPackingPlugin(SchedulingPlugin):
    """Bin-packing scoring to minimize fragmentation."""

    def name(self) -> str:
        return "BinPacking"

    def filter(self, ctx: SchedulingContext, node: Node) -> bool:
        return True  # No filtering, only scoring

    def score(self, ctx: SchedulingContext, node: Node) -> float:
        required = ctx.pod.total_gpu_request

        # Score based on how well pod fits (minimize waste)
        available_gpus = sorted(
            node.gpus,
            key=lambda g: g.available_memory_gb
        )

        waste = 0.0
        allocated = 0
        for gpu in available_gpus:
            if allocated >= required.count:
                break
            if gpu.can_allocate(GPUResources(
                count=1,
                memory_gb=required.memory_gb,
                compute_units=required.compute_units / required.count
            )):
                waste += gpu.available_memory_gb - required.memory_gb
                allocated += 1

        # Lower waste = higher score
        max_waste = sum(g.total_memory_gb for g in node.gpus)
        if max_waste == 0:
            return 0.0
        return (1 - waste / max_waste) * 100


class SpreadingPlugin(SchedulingPlugin):
    """Spread workloads across nodes."""

    def name(self) -> str:
        return "Spreading"

    def filter(self, ctx: SchedulingContext, node: Node) -> bool:
        return True

    def score(self, ctx: SchedulingContext, node: Node) -> float:
        # Lower utilization = higher score
        if len(node.gpus) == 0:
            return 0.0
        utilization = 1 - (node.available_gpu_count / len(node.gpus))
        return (1 - utilization) * 100


class FairSharePlugin(SchedulingPlugin):
    """Fair share scoring across tenants."""

    def __init__(self, cluster: Cluster):
        self.cluster = cluster

    def name(self) -> str:
        return "FairShare"

    def filter(self, ctx: SchedulingContext, node: Node) -> bool:
        return True

    def score(self, ctx: SchedulingContext, node: Node) -> float:
        if not ctx.job:
            return 50.0

        tenant_id = ctx.job.tenant_id
        if tenant_id not in self.cluster.tenants:
            return 50.0

        tenant = self.cluster.tenants[tenant_id]
        # Lower current utilization = higher score (fair share)
        return (1 - tenant.utilization) * 100


class GPUScheduler:
    """Main GPU scheduler."""

    def __init__(self, cluster: Cluster):
        self.cluster = cluster
        self.plugins: list[tuple[SchedulingPlugin, float]] = []
        self._setup_default_plugins()

    def _setup_default_plugins(self) -> None:
        """Set up default scheduling plugins."""
        self.topology = GPUTopology()
        self.gpu_selector = TopologyAwareGPUSelector(self.topology)
        self.plugins = [
            (NodeAffinityPlugin(), 1.0),
            (GPUResourcePlugin(), 2.0),
            (BinPackingPlugin(), 1.5),
            (TopologyPlugin(self.topology), 2.0),  # Topology-aware scoring
            (FairSharePlugin(self.cluster), 1.0),
        ]

    def add_plugin(self, plugin: SchedulingPlugin, weight: float = 1.0) -> None:
        """Add scheduling plugin."""
        self.plugins.append((plugin, weight))

    def schedule_pod(self, pod: Pod) -> SchedulingDecision:
        """Schedule a single pod."""
        job = None
        for j in self.cluster.jobs.values():
            if any(p.pod_id == pod.pod_id for p in j.pods):
                job = j
                break

        ctx = SchedulingContext(
            cluster=self.cluster,
            pod=pod,
            job=job
        )

        # Filter nodes
        feasible_nodes = []
        for node in self.cluster.get_schedulable_nodes():
            feasible = True
            for plugin, _ in self.plugins:
                if not plugin.filter(ctx, node):
                    feasible = False
                    break
            if feasible:
                feasible_nodes.append(node)

        if not feasible_nodes:
            return SchedulingDecision(
                pod_id=pod.pod_id,
                node_id=None,
                gpu_ids=[],
                success=False,
                reason="No feasible nodes"
            )

        # Score nodes
        node_scores: list[tuple[float, Node]] = []
        for node in feasible_nodes:
            total_score = 0.0
            for plugin, weight in self.plugins:
                score = plugin.score(ctx, node)
                total_score += score * weight
            node_scores.append((total_score, node))

        # Select best node
        node_scores.sort(key=lambda x: -x[0])
        best_score, best_node = node_scores[0]

        # Select GPUs on node
        gpu_ids = self._select_gpus(pod, best_node)

        return SchedulingDecision(
            pod_id=pod.pod_id,
            node_id=best_node.node_id,
            gpu_ids=gpu_ids,
            success=True,
            score=best_score
        )

    def _select_gpus(self, pod: Pod, node: Node) -> list[str]:
        """Select specific GPUs on a node using topology-aware selection."""
        required = pod.total_gpu_request

        # Use topology-aware GPU selection
        selected = self.gpu_selector.select_gpus(node, required)
        if selected:
            return selected

        # Fallback to simple selection if topology-aware fails
        fallback_selected = []
        available_gpus = sorted(
            [g for g in node.gpus if len(g.allocated_jobs) == 0],
            key=lambda g: g.available_memory_gb
        )

        for gpu in available_gpus:
            if len(fallback_selected) >= required.count:
                break
            if gpu.available_memory_gb >= required.memory_gb:
                fallback_selected.append(gpu.gpu_id)

        return fallback_selected

    def schedule_gang(self, job: Job) -> list[SchedulingDecision]:
        """Schedule all pods of a gang job together."""
        decisions = []

        # Try to schedule all pods
        temp_decisions = []
        for pod in job.pods:
            decision = self.schedule_pod(pod)
            temp_decisions.append(decision)

        # Check if all pods can be scheduled
        if all(d.success for d in temp_decisions):
            decisions = temp_decisions
        else:
            # Gang scheduling failed
            for pod in job.pods:
                decisions.append(SchedulingDecision(
                    pod_id=pod.pod_id,
                    node_id=None,
                    gpu_ids=[],
                    success=False,
                    reason="Gang scheduling failed - insufficient resources"
                ))

        return decisions

    def run_scheduling_cycle(self) -> list[SchedulingDecision]:
        """Run one scheduling cycle for all pending pods."""
        decisions = []

        # Sort pending pods by priority (high first) and wait time (longest first)
        pending = sorted(
            self.cluster.pending_pods,
            key=lambda p: (-p.priority.value, -p.wait_time)
        )

        # Group by job for gang scheduling
        job_pods: dict[str, list[Pod]] = {}
        standalone_pods = []

        for pod in pending:
            job = None
            for j in self.cluster.jobs.values():
                if any(p.pod_id == pod.pod_id for p in j.pods):
                    job = j
                    break

            if job and job.gang_schedule:
                if job.job_id not in job_pods:
                    job_pods[job.job_id] = []
                job_pods[job.job_id].append(pod)
            else:
                standalone_pods.append(pod)

        # Schedule gang jobs first
        for job_id, pods in job_pods.items():
            job = self.cluster.jobs[job_id]
            gang_decisions = self.schedule_gang(job)
            decisions.extend(gang_decisions)

        # Schedule standalone pods
        for pod in standalone_pods:
            decision = self.schedule_pod(pod)
            decisions.append(decision)

        return decisions


class PriorityQueue:
    """Priority queue for scheduling."""

    def __init__(self):
        self._heap: list[tuple[int, float, str]] = []
        self._entry_finder: dict[str, tuple[int, float, str]] = {}
        self._counter = 0

    def push(self, pod_id: str, priority: int, timestamp: float) -> None:
        """Add or update pod in queue."""
        if pod_id in self._entry_finder:
            self.remove(pod_id)
        entry = (-priority, timestamp, pod_id)
        self._entry_finder[pod_id] = entry
        heapq.heappush(self._heap, entry)

    def pop(self) -> str | None:
        """Pop highest priority pod."""
        while self._heap:
            priority, timestamp, pod_id = heapq.heappop(self._heap)
            if pod_id in self._entry_finder:
                del self._entry_finder[pod_id]
                return pod_id
        return None

    def remove(self, pod_id: str) -> None:
        """Remove pod from queue."""
        if pod_id in self._entry_finder:
            del self._entry_finder[pod_id]

    def __len__(self) -> int:
        return len(self._entry_finder)


class QueueScheduler:
    """Multi-queue scheduler with fair sharing."""

    def __init__(self, cluster: Cluster):
        self.cluster = cluster
        self.gpu_scheduler = GPUScheduler(cluster)
        self.queue_priorities: dict[str, PriorityQueue] = {}

    def enqueue(self, pod: Pod, queue_name: str = "default") -> bool:
        """Add pod to queue."""
        if queue_name not in self.queue_priorities:
            self.queue_priorities[queue_name] = PriorityQueue()

        queue = self.queue_priorities[queue_name]
        queue.push(pod.pod_id, pod.priority.value, pod.created_at)
        return True

    def schedule(self) -> list[SchedulingDecision]:
        """Schedule pods from all queues fairly."""
        decisions = []

        # Calculate fair share for each queue
        total_weight = sum(
            self.cluster.queues[q].priority_weight
            for q in self.queue_priorities
            if q in self.cluster.queues
        )

        if total_weight == 0:
            total_weight = len(self.queue_priorities)

        # Schedule from each queue proportionally
        for queue_name, pq in self.queue_priorities.items():
            if len(pq) == 0:
                continue

            weight = 1.0
            if queue_name in self.cluster.queues:
                weight = self.cluster.queues[queue_name].priority_weight

            # Number of pods to schedule from this queue
            share = int((weight / total_weight) * 10) + 1

            for _ in range(min(share, len(pq))):
                pod_id = pq.pop()
                if pod_id and pod_id in self.cluster.pods:
                    pod = self.cluster.pods[pod_id]
                    decision = self.gpu_scheduler.schedule_pod(pod)
                    decisions.append(decision)

                    if not decision.success:
                        # Re-queue failed pod
                        pq.push(pod_id, pod.priority.value, pod.created_at)

        return decisions


class PreemptionScheduler:
    """Scheduler with preemption support."""

    def __init__(self, cluster: Cluster):
        self.cluster = cluster
        self.gpu_scheduler = GPUScheduler(cluster)

    def find_preemption_candidates(
        self,
        pod: Pod,
        node: Node
    ) -> list[Pod]:
        """Find pods that can be preempted."""
        candidates = {}

        # Find lower priority running pods
        for gpu in node.gpus:
            for job_id in gpu.allocated_jobs:
                if job_id in self.cluster.jobs:
                    job = self.cluster.jobs[job_id]
                    for running_pod in job.pods:
                        if running_pod.state == JobState.RUNNING:
                            if running_pod.priority.value < pod.priority.value:
                                if job.preemptible:
                                    candidates[running_pod.pod_id] = running_pod

        return list(candidates.values())

    def schedule_with_preemption(self, pod: Pod) -> SchedulingDecision:
        """Try to schedule with preemption if needed."""
        # First try normal scheduling
        decision = self.gpu_scheduler.schedule_pod(pod)
        if decision.success:
            return decision

        # Try preemption
        for node in self.cluster.get_schedulable_nodes():
            candidates = self.find_preemption_candidates(pod, node)

            # Calculate resources freed by preemption
            freed = GPUResources(count=0, memory_gb=0, compute_units=0)
            for candidate in candidates:
                freed = freed.add(candidate.total_gpu_request)

            # Check if preemption would help
            required = pod.total_gpu_request
            if freed.count >= required.count:
                # Preempt and schedule
                return SchedulingDecision(
                    pod_id=pod.pod_id,
                    node_id=node.node_id,
                    gpu_ids=[],  # Will be assigned after preemption
                    success=True,
                    reason=f"Preempting {len(candidates)} pods"
                )

        return SchedulingDecision(
            pod_id=pod.pod_id,
            node_id=None,
            gpu_ids=[],
            success=False,
            reason="No preemption candidates available"
        )
