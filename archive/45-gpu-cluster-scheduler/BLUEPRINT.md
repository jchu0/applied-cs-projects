# Project 45: Multi-Tenant GPU Cluster Scheduler

## Executive Summary

A sophisticated GPU cluster scheduler combining features from Kubernetes, Ray, and Slurm for ML workloads. This system implements advanced scheduling algorithms including gang scheduling, fair sharing, preemption, and GPU topology awareness to maximize cluster utilization while providing multi-tenant isolation and QoS guarantees.

## Architecture Overview

### System Design

```
+------------------------------------------------------------------+
|                  GPU Cluster Scheduler                            |
+------------------------------------------------------------------+
|                                                                    |
|  +-------------------+     +-------------------+     +-----------+ |
|  | Job Submission    |     | Scheduler Core    |     | Resource  | |
|  | API               |---->| (Algorithms)      |---->| Manager   | |
|  +-------------------+     +-------------------+     +-----------+ |
|         |                          |                       |       |
|         v                          v                       v       |
|  +-------------------+     +-------------------+     +-----------+ |
|  | Queue Manager     |     | Topology          |     | Preemption| |
|  | (Priority/Fair)   |     | Analyzer          |     | Controller| |
|  +-------------------+     +-------------------+     +-----------+ |
|                                    |                               |
|  +----------------------------------------------------------+     |
|  |                    Node Manager                           |     |
|  |  +--------+  +--------+  +--------+  +--------+           |     |
|  |  | Node 0 |  | Node 1 |  | Node 2 |  | Node N |           |     |
|  |  | 8xA100 |  | 8xA100 |  | 8xA100 |  | 8xA100 |           |     |
|  |  +--------+  +--------+  +--------+  +--------+           |     |
|  +----------------------------------------------------------+     |
+------------------------------------------------------------------+
```

### Core Components

#### 1. Job and Resource Definitions

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum
import time
import uuid

class JobState(Enum):
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PREEMPTED = "preempted"
    CANCELLED = "cancelled"

class JobPriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

class SchedulingPolicy(Enum):
    FIFO = "fifo"
    FAIR_SHARE = "fair_share"
    DRF = "dominant_resource_fairness"
    PRIORITY = "priority"

@dataclass
class ResourceRequirements:
    """Resource requirements for a job."""
    gpus: int = 1
    gpu_memory_gb: float = 0  # 0 means any
    cpus: int = 1
    memory_gb: float = 4

    # GPU topology requirements
    require_nvlink: bool = False
    require_same_node: bool = True
    gpu_type: Optional[str] = None  # e.g., "A100", "H100"

    # Elasticity
    min_gpus: Optional[int] = None  # For elastic jobs
    max_gpus: Optional[int] = None

@dataclass
class Job:
    """A schedulable job/task."""
    job_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    user_id: str = ""
    tenant_id: str = ""

    # Resource requirements
    resources: ResourceRequirements = field(default_factory=ResourceRequirements)

    # Scheduling parameters
    priority: JobPriority = JobPriority.NORMAL
    preemptible: bool = True
    gang_size: int = 1  # For gang scheduling (all-or-nothing)

    # Runtime state
    state: JobState = JobState.PENDING
    allocated_gpus: List['GPU'] = field(default_factory=list)
    allocated_node: Optional[str] = None

    # Timing
    submit_time: float = field(default_factory=time.time)
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    # Checkpointing
    checkpoint_path: Optional[str] = None
    supports_checkpoint: bool = False

    def __lt__(self, other):
        return (self.priority.value, -self.submit_time) > \
               (other.priority.value, -other.submit_time)


@dataclass
class GPU:
    """Represents a single GPU."""
    gpu_id: str
    node_id: str
    gpu_type: str  # e.g., "A100-80GB"
    memory_gb: float

    # Topology
    numa_node: int = 0
    pcie_bus: str = ""
    nvlink_peers: List[str] = field(default_factory=list)

    # State
    allocated: bool = False
    allocated_job_id: Optional[str] = None


@dataclass
class Node:
    """Represents a compute node."""
    node_id: str
    hostname: str

    # Resources
    gpus: List[GPU] = field(default_factory=list)
    total_cpus: int = 64
    total_memory_gb: float = 512

    # Current allocation
    allocated_cpus: int = 0
    allocated_memory_gb: float = 0

    # State
    is_healthy: bool = True
    last_heartbeat: float = field(default_factory=time.time)

    @property
    def free_gpus(self) -> List[GPU]:
        return [g for g in self.gpus if not g.allocated]

    @property
    def num_free_gpus(self) -> int:
        return len(self.free_gpus)
```

#### 2. Scheduling Algorithms

```python
from abc import ABC, abstractmethod
import heapq
from collections import defaultdict

class SchedulingAlgorithm(ABC):
    """Base class for scheduling algorithms."""

    @abstractmethod
    def select_job(self,
                   pending_jobs: List[Job],
                   cluster_state: 'ClusterState') -> Optional[Job]:
        """Select next job to schedule."""
        pass

    @abstractmethod
    def select_resources(self,
                         job: Job,
                         cluster_state: 'ClusterState') -> Optional[List[GPU]]:
        """Select GPUs for a job."""
        pass


class FIFOScheduler(SchedulingAlgorithm):
    """Simple FIFO scheduling."""

    def select_job(self,
                   pending_jobs: List[Job],
                   cluster_state: 'ClusterState') -> Optional[Job]:
        if not pending_jobs:
            return None

        # Sort by submit time
        sorted_jobs = sorted(pending_jobs, key=lambda j: j.submit_time)

        for job in sorted_jobs:
            if cluster_state.can_allocate(job):
                return job
        return None

    def select_resources(self,
                         job: Job,
                         cluster_state: 'ClusterState') -> Optional[List[GPU]]:
        return cluster_state.find_gpus(job.resources)


class DRFScheduler(SchedulingAlgorithm):
    """
    Dominant Resource Fairness scheduler.

    Ensures fair allocation based on each user's dominant resource.
    """

    def __init__(self):
        self.user_allocations: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {'gpus': 0, 'cpus': 0, 'memory': 0}
        )

    def select_job(self,
                   pending_jobs: List[Job],
                   cluster_state: 'ClusterState') -> Optional[Job]:
        if not pending_jobs:
            return None

        total_gpus = cluster_state.total_gpus
        total_cpus = cluster_state.total_cpus
        total_memory = cluster_state.total_memory

        # Calculate dominant share for each user with pending jobs
        user_dominant_shares = {}
        for job in pending_jobs:
            user = job.user_id
            alloc = self.user_allocations[user]

            gpu_share = alloc['gpus'] / total_gpus if total_gpus > 0 else 0
            cpu_share = alloc['cpus'] / total_cpus if total_cpus > 0 else 0
            mem_share = alloc['memory'] / total_memory if total_memory > 0 else 0

            dominant_share = max(gpu_share, cpu_share, mem_share)
            user_dominant_shares[user] = dominant_share

        # Select job from user with lowest dominant share
        sorted_jobs = sorted(
            pending_jobs,
            key=lambda j: (user_dominant_shares.get(j.user_id, 0), j.submit_time)
        )

        for job in sorted_jobs:
            if cluster_state.can_allocate(job):
                return job
        return None

    def select_resources(self,
                         job: Job,
                         cluster_state: 'ClusterState') -> Optional[List[GPU]]:
        return cluster_state.find_gpus(job.resources)

    def update_allocation(self, job: Job, allocated: bool):
        """Update user allocation when job starts/ends."""
        user = job.user_id
        sign = 1 if allocated else -1

        self.user_allocations[user]['gpus'] += sign * job.resources.gpus
        self.user_allocations[user]['cpus'] += sign * job.resources.cpus
        self.user_allocations[user]['memory'] += sign * job.resources.memory_gb


class GangScheduler(SchedulingAlgorithm):
    """
    Gang scheduling for distributed training.

    Ensures all workers of a job are scheduled together.
    """

    def select_job(self,
                   pending_jobs: List[Job],
                   cluster_state: 'ClusterState') -> Optional[Job]:
        # Filter to gang jobs
        gang_jobs = [j for j in pending_jobs if j.gang_size > 1]

        for job in gang_jobs:
            # Check if we can allocate all workers at once
            required_gpus = job.resources.gpus * job.gang_size

            if cluster_state.can_allocate_gang(job, required_gpus):
                return job

        return None

    def select_resources(self,
                         job: Job,
                         cluster_state: 'ClusterState') -> Optional[List[GPU]]:
        total_gpus = job.resources.gpus * job.gang_size
        return cluster_state.find_gpus_for_gang(job.resources, total_gpus)


class PriorityScheduler(SchedulingAlgorithm):
    """Priority-based scheduling with preemption support."""

    def __init__(self, preemption_enabled: bool = True):
        self.preemption_enabled = preemption_enabled

    def select_job(self,
                   pending_jobs: List[Job],
                   cluster_state: 'ClusterState') -> Optional[Job]:
        if not pending_jobs:
            return None

        # Sort by priority (high first), then submit time
        sorted_jobs = sorted(
            pending_jobs,
            key=lambda j: (-j.priority.value, j.submit_time)
        )

        for job in sorted_jobs:
            if cluster_state.can_allocate(job):
                return job

            # Check if preemption can make room
            if self.preemption_enabled:
                preemptable = self._find_preemptable_jobs(
                    job, cluster_state
                )
                if preemptable:
                    return job  # Scheduler will handle preemption

        return None

    def _find_preemptable_jobs(self,
                                high_priority_job: Job,
                                cluster_state: 'ClusterState') -> List[Job]:
        """Find lower-priority jobs that can be preempted."""
        running_jobs = cluster_state.get_running_jobs()
        preemptable = []
        freed_gpus = 0

        # Sort running jobs by priority (low first)
        sorted_running = sorted(
            running_jobs,
            key=lambda j: (j.priority.value, -j.start_time)
        )

        for job in sorted_running:
            if job.priority.value < high_priority_job.priority.value:
                if job.preemptible:
                    preemptable.append(job)
                    freed_gpus += job.resources.gpus

                    if freed_gpus >= high_priority_job.resources.gpus:
                        return preemptable

        return []

    def select_resources(self,
                         job: Job,
                         cluster_state: 'ClusterState') -> Optional[List[GPU]]:
        return cluster_state.find_gpus(job.resources)
```

#### 3. Cluster State and Resource Management

```python
class ClusterState:
    """Maintains cluster state and resource availability."""

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.jobs: Dict[str, Job] = {}
        self.gpu_topology: 'GPUTopology' = GPUTopology()

    def add_node(self, node: Node):
        """Add a node to the cluster."""
        self.nodes[node.node_id] = node
        self.gpu_topology.add_node(node)

    def can_allocate(self, job: Job) -> bool:
        """Check if job can be allocated."""
        gpus = self.find_gpus(job.resources)
        return gpus is not None and len(gpus) >= job.resources.gpus

    def can_allocate_gang(self, job: Job, total_gpus: int) -> bool:
        """Check if gang job can be allocated."""
        gpus = self.find_gpus_for_gang(job.resources, total_gpus)
        return gpus is not None and len(gpus) >= total_gpus

    def find_gpus(self, requirements: ResourceRequirements) -> Optional[List[GPU]]:
        """Find GPUs matching requirements."""
        candidates = []

        for node in self.nodes.values():
            if not node.is_healthy:
                continue

            # Check CPU/memory availability
            if node.total_cpus - node.allocated_cpus < requirements.cpus:
                continue
            if node.total_memory_gb - node.allocated_memory_gb < requirements.memory_gb:
                continue

            # Find matching free GPUs
            free_gpus = [
                g for g in node.free_gpus
                if self._gpu_matches(g, requirements)
            ]

            if len(free_gpus) >= requirements.gpus:
                if requirements.require_same_node:
                    # Take GPUs from this node
                    selected = self._select_best_gpus(
                        free_gpus, requirements.gpus, requirements
                    )
                    if selected:
                        return selected

                candidates.extend(free_gpus)

        if not requirements.require_same_node and len(candidates) >= requirements.gpus:
            return candidates[:requirements.gpus]

        return None

    def find_gpus_for_gang(self,
                           requirements: ResourceRequirements,
                           total_gpus: int) -> Optional[List[GPU]]:
        """Find GPUs for gang scheduling across multiple nodes."""
        all_candidates = []

        # Prefer packing within nodes for communication efficiency
        for node in self.nodes.values():
            if not node.is_healthy:
                continue

            free_gpus = [
                g for g in node.free_gpus
                if self._gpu_matches(g, requirements)
            ]

            all_candidates.extend(free_gpus)

        if len(all_candidates) >= total_gpus:
            # Sort by node to encourage packing
            all_candidates.sort(key=lambda g: g.node_id)
            return all_candidates[:total_gpus]

        return None

    def _gpu_matches(self, gpu: GPU, requirements: ResourceRequirements) -> bool:
        """Check if GPU matches requirements."""
        if requirements.gpu_type and gpu.gpu_type != requirements.gpu_type:
            return False
        if requirements.gpu_memory_gb > 0 and gpu.memory_gb < requirements.gpu_memory_gb:
            return False
        return True

    def _select_best_gpus(self,
                          gpus: List[GPU],
                          num_gpus: int,
                          requirements: ResourceRequirements) -> List[GPU]:
        """Select best GPUs considering topology."""
        if requirements.require_nvlink:
            # Find NVLink-connected subset
            return self.gpu_topology.find_nvlink_group(gpus, num_gpus)

        # Otherwise just take first available
        return gpus[:num_gpus]

    def allocate(self, job: Job, gpus: List[GPU]) -> None:
        """Allocate resources to job."""
        for gpu in gpus:
            gpu.allocated = True
            gpu.allocated_job_id = job.job_id

        job.allocated_gpus = gpus
        job.state = JobState.SCHEDULED

        if gpus:
            node = self.nodes[gpus[0].node_id]
            node.allocated_cpus += job.resources.cpus
            node.allocated_memory_gb += job.resources.memory_gb
            job.allocated_node = node.node_id

        self.jobs[job.job_id] = job

    def deallocate(self, job: Job) -> None:
        """Release resources from job."""
        for gpu in job.allocated_gpus:
            gpu.allocated = False
            gpu.allocated_job_id = None

        if job.allocated_node:
            node = self.nodes[job.allocated_node]
            node.allocated_cpus -= job.resources.cpus
            node.allocated_memory_gb -= job.resources.memory_gb

        job.allocated_gpus = []
        job.allocated_node = None

    def get_running_jobs(self) -> List[Job]:
        """Get all running jobs."""
        return [j for j in self.jobs.values() if j.state == JobState.RUNNING]

    @property
    def total_gpus(self) -> int:
        return sum(len(n.gpus) for n in self.nodes.values())

    @property
    def total_cpus(self) -> int:
        return sum(n.total_cpus for n in self.nodes.values())

    @property
    def total_memory(self) -> float:
        return sum(n.total_memory_gb for n in self.nodes.values())


class GPUTopology:
    """Analyze and manage GPU topology."""

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.nvlink_graph: Dict[str, Set[str]] = defaultdict(set)

    def add_node(self, node: Node):
        """Add node and build NVLink graph."""
        self.nodes[node.node_id] = node

        for gpu in node.gpus:
            for peer_id in gpu.nvlink_peers:
                self.nvlink_graph[gpu.gpu_id].add(peer_id)
                self.nvlink_graph[peer_id].add(gpu.gpu_id)

    def find_nvlink_group(self, gpus: List[GPU], size: int) -> Optional[List[GPU]]:
        """Find a group of NVLink-connected GPUs."""
        if len(gpus) < size:
            return None

        # Build subgraph
        gpu_ids = {g.gpu_id for g in gpus}
        gpu_map = {g.gpu_id: g for g in gpus}

        # Find connected components using DFS
        visited = set()
        components = []

        for gpu in gpus:
            if gpu.gpu_id in visited:
                continue

            component = []
            stack = [gpu.gpu_id]

            while stack:
                gid = stack.pop()
                if gid in visited:
                    continue
                visited.add(gid)

                if gid in gpu_ids:
                    component.append(gpu_map[gid])

                    for peer in self.nvlink_graph[gid]:
                        if peer not in visited and peer in gpu_ids:
                            stack.append(peer)

            if len(component) >= size:
                return component[:size]
            components.append(component)

        return None

    def get_communication_cost(self, gpus: List[GPU]) -> float:
        """Estimate communication cost for a set of GPUs."""
        cost = 0.0

        for i, g1 in enumerate(gpus):
            for g2 in gpus[i+1:]:
                if g1.node_id != g2.node_id:
                    cost += 10.0  # Cross-node
                elif g2.gpu_id not in self.nvlink_graph[g1.gpu_id]:
                    cost += 2.0   # Same node, no NVLink
                else:
                    cost += 0.1   # NVLink connected

        return cost
```

#### 4. Preemption Controller

```python
class PreemptionController:
    """Handles job preemption with checkpointing."""

    def __init__(self, cluster_state: ClusterState):
        self.cluster_state = cluster_state
        self.preemption_queue: List[Job] = []

    def preempt_jobs(self,
                     jobs_to_preempt: List[Job],
                     high_priority_job: Job) -> bool:
        """
        Preempt jobs to make room for high-priority job.

        Returns True if preemption successful.
        """
        for job in jobs_to_preempt:
            # Checkpoint if supported
            if job.supports_checkpoint:
                success = self._checkpoint_job(job)
                if not success:
                    return False

            # Stop job
            self._stop_job(job)

            # Release resources
            self.cluster_state.deallocate(job)

            # Mark as preempted
            job.state = JobState.PREEMPTED
            self.preemption_queue.append(job)

        return True

    def _checkpoint_job(self, job: Job) -> bool:
        """Save job checkpoint."""
        # Implementation would interact with job runtime
        checkpoint_path = f"/checkpoints/{job.job_id}/{time.time()}"
        job.checkpoint_path = checkpoint_path

        # Signal job to checkpoint
        # ...

        return True

    def _stop_job(self, job: Job) -> None:
        """Stop a running job."""
        # Implementation would send SIGTERM/SIGKILL
        pass

    def resume_preempted_jobs(self) -> List[Job]:
        """Get preempted jobs ready to resume."""
        ready = []

        for job in self.preemption_queue:
            if self.cluster_state.can_allocate(job):
                ready.append(job)
                self.preemption_queue.remove(job)

        return ready
```

### Enterprise Features

#### Multi-Tenant Quota Management

```python
from dataclasses import dataclass
from typing import Dict

@dataclass
class TenantQuota:
    """Resource quotas for a tenant."""
    tenant_id: str
    max_gpus: int
    max_cpus: int
    max_memory_gb: float

    # Burst allowance
    burst_gpus: int = 0  # Extra GPUs when cluster is idle

    # Priorities
    min_priority: JobPriority = JobPriority.LOW
    max_priority: JobPriority = JobPriority.HIGH


class QuotaManager:
    """Manage and enforce tenant quotas."""

    def __init__(self):
        self.quotas: Dict[str, TenantQuota] = {}
        self.usage: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {'gpus': 0, 'cpus': 0, 'memory': 0}
        )

    def add_quota(self, quota: TenantQuota):
        """Add or update tenant quota."""
        self.quotas[quota.tenant_id] = quota

    def check_quota(self, job: Job) -> bool:
        """Check if job fits within tenant quota."""
        tenant_id = job.tenant_id

        if tenant_id not in self.quotas:
            return True  # No quota defined

        quota = self.quotas[tenant_id]
        usage = self.usage[tenant_id]

        # Check GPU quota
        if usage['gpus'] + job.resources.gpus > quota.max_gpus:
            # Check burst allowance
            if usage['gpus'] + job.resources.gpus > quota.max_gpus + quota.burst_gpus:
                return False

        # Check CPU quota
        if usage['cpus'] + job.resources.cpus > quota.max_cpus:
            return False

        # Check memory quota
        if usage['memory'] + job.resources.memory_gb > quota.max_memory_gb:
            return False

        # Check priority
        if job.priority.value < quota.min_priority.value:
            return False
        if job.priority.value > quota.max_priority.value:
            return False

        return True

    def update_usage(self, job: Job, allocated: bool):
        """Update tenant usage when job starts/ends."""
        tenant_id = job.tenant_id
        sign = 1 if allocated else -1

        self.usage[tenant_id]['gpus'] += sign * job.resources.gpus
        self.usage[tenant_id]['cpus'] += sign * job.resources.cpus
        self.usage[tenant_id]['memory'] += sign * job.resources.memory_gb

    def get_usage_report(self, tenant_id: str) -> Dict:
        """Get usage report for tenant."""
        if tenant_id not in self.quotas:
            return {}

        quota = self.quotas[tenant_id]
        usage = self.usage[tenant_id]

        return {
            'gpu_usage': usage['gpus'],
            'gpu_quota': quota.max_gpus,
            'gpu_utilization': usage['gpus'] / quota.max_gpus if quota.max_gpus > 0 else 0,
            'cpu_usage': usage['cpus'],
            'cpu_quota': quota.max_cpus,
            'memory_usage': usage['memory'],
            'memory_quota': quota.max_memory_gb
        }


class UsageReporter:
    """Generate usage reports and analytics."""

    def __init__(self, quota_manager: QuotaManager):
        self.quota_manager = quota_manager
        self.job_history: List[Job] = []

    def record_job(self, job: Job):
        """Record completed job for reporting."""
        self.job_history.append(job)

    def generate_report(self,
                        tenant_id: Optional[str] = None,
                        start_time: Optional[float] = None,
                        end_time: Optional[float] = None) -> Dict:
        """Generate usage report."""
        jobs = self.job_history

        if tenant_id:
            jobs = [j for j in jobs if j.tenant_id == tenant_id]
        if start_time:
            jobs = [j for j in jobs if j.submit_time >= start_time]
        if end_time:
            jobs = [j for j in jobs if j.end_time and j.end_time <= end_time]

        total_gpu_hours = sum(
            j.resources.gpus * (j.end_time - j.start_time) / 3600
            for j in jobs if j.end_time and j.start_time
        )

        return {
            'total_jobs': len(jobs),
            'total_gpu_hours': total_gpu_hours,
            'jobs_by_state': self._count_by_state(jobs),
            'avg_wait_time': self._avg_wait_time(jobs),
            'avg_run_time': self._avg_run_time(jobs)
        }

    def _count_by_state(self, jobs: List[Job]) -> Dict[str, int]:
        counts = defaultdict(int)
        for job in jobs:
            counts[job.state.value] += 1
        return dict(counts)

    def _avg_wait_time(self, jobs: List[Job]) -> float:
        wait_times = [
            j.start_time - j.submit_time
            for j in jobs if j.start_time
        ]
        return sum(wait_times) / len(wait_times) if wait_times else 0

    def _avg_run_time(self, jobs: List[Job]) -> float:
        run_times = [
            j.end_time - j.start_time
            for j in jobs if j.end_time and j.start_time
        ]
        return sum(run_times) / len(run_times) if run_times else 0
```

#### Backfilling

```python
class BackfillScheduler:
    """
    Backfill scheduler to improve utilization.

    Schedules smaller jobs in gaps while waiting for large jobs.
    """

    def __init__(self, base_scheduler: SchedulingAlgorithm):
        self.base_scheduler = base_scheduler
        self.reservations: Dict[str, 'Reservation'] = {}

    def schedule(self,
                 pending_jobs: List[Job],
                 cluster_state: ClusterState) -> List[Tuple[Job, List[GPU]]]:
        """Schedule jobs with backfilling."""
        scheduled = []

        # Sort by priority then size (large first)
        sorted_jobs = sorted(
            pending_jobs,
            key=lambda j: (-j.priority.value, -j.resources.gpus, j.submit_time)
        )

        for job in sorted_jobs:
            gpus = cluster_state.find_gpus(job.resources)

            if gpus:
                # Can schedule immediately
                scheduled.append((job, gpus))
                cluster_state.allocate(job, gpus)
            else:
                # Make reservation for large job
                reservation = self._make_reservation(job, cluster_state)
                if reservation:
                    self.reservations[job.job_id] = reservation

        # Try to backfill smaller jobs
        backfill_candidates = [
            j for j in sorted_jobs
            if j.job_id not in [s[0].job_id for s in scheduled]
            and j.job_id not in self.reservations
        ]

        for job in backfill_candidates:
            if self._can_backfill(job, cluster_state):
                gpus = cluster_state.find_gpus(job.resources)
                if gpus:
                    scheduled.append((job, gpus))
                    cluster_state.allocate(job, gpus)

        return scheduled

    def _make_reservation(self,
                          job: Job,
                          cluster_state: ClusterState) -> Optional['Reservation']:
        """Make resource reservation for a job."""
        # Estimate when resources will be available
        # based on expected completion of running jobs
        return Reservation(
            job_id=job.job_id,
            gpus_needed=job.resources.gpus,
            estimated_start=time.time() + 3600  # Placeholder
        )

    def _can_backfill(self, job: Job, cluster_state: ClusterState) -> bool:
        """Check if job can be backfilled without delaying reservations."""
        # Check if job can complete before any reservation starts
        for reservation in self.reservations.values():
            # Estimate job completion time
            # ...
            pass
        return True


@dataclass
class Reservation:
    """Resource reservation for a pending job."""
    job_id: str
    gpus_needed: int
    estimated_start: float
```

## API Reference

### Job Submission

```python
from scheduler import ClusterScheduler, Job, ResourceRequirements

# Create scheduler
scheduler = ClusterScheduler(policy=SchedulingPolicy.DRF)

# Submit job
job = Job(
    name="training-job",
    user_id="user123",
    tenant_id="ml-team",
    resources=ResourceRequirements(
        gpus=4,
        gpu_type="A100",
        require_nvlink=True
    ),
    priority=JobPriority.HIGH
)

job_id = scheduler.submit(job)
```

### Monitoring

```python
# Get job status
status = scheduler.get_job_status(job_id)

# Get cluster utilization
util = scheduler.get_cluster_utilization()
print(f"GPU utilization: {util['gpu_utilization']:.1%}")

# Get tenant report
report = scheduler.get_tenant_report("ml-team")
```

## Implementation Phases

### Phase 1: Core Data Structures (Week 1)
- Job and resource definitions
- Node and GPU models
- Basic cluster state

### Phase 2: Basic Schedulers (Weeks 2-3)
- FIFO scheduler
- Priority scheduler
- Resource allocation logic

### Phase 3: Advanced Scheduling (Weeks 4-5)
- DRF scheduler
- Gang scheduling
- GPU topology awareness

### Phase 4: Preemption (Weeks 6-7)
- Preemption controller
- Checkpointing integration
- Job resumption

### Phase 5: Multi-Tenancy (Weeks 8-9)
- Quota management
- Usage tracking
- Reporting

### Phase 6: Enterprise Features (Weeks 10-14)
- Backfilling
- Fault tolerance
- RL-based optimization

## Testing Strategy

### Unit Tests

```python
class TestSchedulers:
    def test_fifo_ordering(self):
        scheduler = FIFOScheduler()
        # ...

    def test_drf_fairness(self):
        scheduler = DRFScheduler()
        # Verify fair allocation

class TestClusterState:
    def test_gpu_allocation(self):
        state = ClusterState()
        # Add nodes, allocate, verify

    def test_nvlink_group_finding(self):
        topology = GPUTopology()
        # Test NVLink group selection
```

## Performance Targets

| Metric | Target |
|--------|--------|
| Scheduling latency | <100ms |
| Cluster utilization | >85% |
| Fair share deviation | <5% |
| Preemption time | <30s |

## Dependencies

- Python 3.9+
- gRPC for API
- etcd/Redis for state storage
- Prometheus for metrics

## References

- Kubernetes scheduler
- YARN capacity scheduler
- Slurm documentation
- Gandiva: Introspective Cluster Scheduling for Deep Learning
