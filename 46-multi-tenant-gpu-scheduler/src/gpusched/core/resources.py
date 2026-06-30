"""Core resource abstractions for GPU scheduling."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import time
import uuid


class GPUType(Enum):
    """GPU hardware types."""
    A100 = "a100"
    H100 = "h100"
    V100 = "v100"
    T4 = "t4"
    A10G = "a10g"
    L4 = "l4"


class JobState(Enum):
    """Job lifecycle states."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PREEMPTED = "preempted"


class PriorityClass(Enum):
    """Job priority classes."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class GPUResources:
    """GPU resource requirements or availability."""
    count: int = 1
    memory_gb: float = 16.0
    compute_units: float = 1.0  # Fraction of GPU compute
    gpu_type: GPUType | None = None
    # Topology requirements
    require_nvlink: bool = False  # Require NVLink-connected GPUs
    require_same_node: bool = True  # All GPUs must be on same node
    numa_preference: int | None = None  # Preferred NUMA node

    def fits(self, available: "GPUResources") -> bool:
        """Check if requirements fit in available resources."""
        if self.count > available.count:
            return False
        if self.memory_gb > available.memory_gb:
            return False
        if self.compute_units > available.compute_units:
            return False
        if self.gpu_type and available.gpu_type:
            if self.gpu_type != available.gpu_type:
                return False
        return True

    def subtract(self, other: "GPUResources") -> "GPUResources":
        """Subtract resources."""
        return GPUResources(
            count=self.count - other.count,
            memory_gb=self.memory_gb - other.memory_gb,
            compute_units=self.compute_units - other.compute_units,
            gpu_type=self.gpu_type
        )

    def add(self, other: "GPUResources") -> "GPUResources":
        """Add resources."""
        # Prefer the non-None gpu_type, or use self if both are set
        result_gpu_type = self.gpu_type or other.gpu_type
        return GPUResources(
            count=self.count + other.count,
            memory_gb=self.memory_gb + other.memory_gb,
            compute_units=self.compute_units + other.compute_units,
            gpu_type=result_gpu_type,
            # Preserve topology requirements (True if either requires it)
            require_nvlink=self.require_nvlink or other.require_nvlink,
            require_same_node=self.require_same_node and other.require_same_node,
            numa_preference=self.numa_preference or other.numa_preference,
        )


@dataclass
class GPU:
    """Physical GPU device."""
    gpu_id: str
    node_id: str
    gpu_type: GPUType
    total_memory_gb: float
    available_memory_gb: float
    compute_capability: tuple[int, int]
    mig_enabled: bool = False
    mig_instances: list[str] = field(default_factory=list)
    allocated_jobs: list[str] = field(default_factory=list)
    utilization: float = 0.0
    temperature: float = 0.0
    power_usage: float = 0.0
    # Topology information
    nvlink_peers: list[str] = field(default_factory=list)  # GPU IDs connected via NVLink
    numa_node: int = 0  # NUMA node affinity
    pcie_bus: str = ""  # PCIe bus address (e.g., "0000:3b:00.0")
    local_index: int = 0  # Local GPU index on node (0-7 for 8-GPU node)

    @property
    def available_compute(self) -> float:
        """Get available compute fraction."""
        return max(0, 1.0 - len(self.allocated_jobs) * 0.25)

    def get_available_resources(self) -> GPUResources:
        """Get available resources on this GPU."""
        return GPUResources(
            count=1,
            memory_gb=self.available_memory_gb,
            compute_units=self.available_compute,
            gpu_type=self.gpu_type
        )

    def can_allocate(self, requirements: GPUResources) -> bool:
        """Check if GPU can satisfy requirements."""
        return requirements.fits(self.get_available_resources())


@dataclass
class Node:
    """Compute node with GPUs."""
    node_id: str
    hostname: str
    gpus: list[GPU]
    total_cpu_cores: int = 64
    available_cpu_cores: int = 64
    total_memory_gb: float = 256.0
    available_memory_gb: float = 256.0
    labels: dict[str, str] = field(default_factory=dict)
    taints: list[str] = field(default_factory=list)
    conditions: dict[str, bool] = field(default_factory=dict)

    @property
    def total_gpu_count(self) -> int:
        return len(self.gpus)

    @property
    def available_gpu_count(self) -> int:
        return sum(1 for g in self.gpus if len(g.allocated_jobs) == 0)

    def get_gpu_by_type(self, gpu_type: GPUType) -> list[GPU]:
        """Get GPUs of specific type."""
        return [g for g in self.gpus if g.gpu_type == gpu_type]

    def is_schedulable(self) -> bool:
        """Check if node is schedulable."""
        return self.conditions.get("Ready", True) and "NoSchedule" not in self.taints


@dataclass
class Container:
    """Container within a pod."""
    name: str
    image: str
    command: list[str] = field(default_factory=list)
    args: list[str] = field(default_factory=list)
    gpu_resources: GPUResources = field(default_factory=GPUResources)
    cpu_request: float = 1.0
    memory_request_gb: float = 4.0
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class Pod:
    """Pod containing containers."""
    pod_id: str
    name: str
    namespace: str
    containers: list[Container]
    priority: PriorityClass = PriorityClass.NORMAL
    node_selector: dict[str, str] = field(default_factory=dict)
    tolerations: list[str] = field(default_factory=list)
    affinity: dict[str, Any] = field(default_factory=dict)
    state: JobState = JobState.PENDING
    assigned_node: str | None = None
    assigned_gpus: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=lambda: time.time())
    started_at: float | None = None
    completed_at: float | None = None

    @property
    def total_gpu_request(self) -> GPUResources:
        """Get total GPU requirements across all containers."""
        total = GPUResources(count=0, memory_gb=0, compute_units=0)
        for container in self.containers:
            total = total.add(container.gpu_resources)
        return total

    @property
    def wait_time(self) -> float:
        """Get time waiting in queue."""
        if self.started_at:
            return self.started_at - self.created_at
        return time.time() - self.created_at


@dataclass
class Job:
    """Training or inference job."""
    job_id: str
    name: str
    namespace: str
    pods: list[Pod]
    parallelism: int = 1
    completions: int = 1
    priority: PriorityClass = PriorityClass.NORMAL
    preemptible: bool = True
    gang_schedule: bool = False  # All pods must schedule together
    max_runtime_seconds: int | None = None
    tenant_id: str = "default"
    queue_name: str = "default"
    created_at: float = field(default_factory=lambda: time.time())

    @property
    def state(self) -> JobState:
        """Get overall job state."""
        states = [p.state for p in self.pods]
        if all(s == JobState.COMPLETED for s in states):
            return JobState.COMPLETED
        if any(s == JobState.FAILED for s in states):
            return JobState.FAILED
        if any(s == JobState.RUNNING for s in states):
            return JobState.RUNNING
        if any(s == JobState.SCHEDULED for s in states):
            return JobState.SCHEDULED
        return JobState.PENDING

    @property
    def total_gpu_request(self) -> GPUResources:
        """Get total GPU requirements for entire job."""
        total = GPUResources(count=0, memory_gb=0, compute_units=0)
        for pod in self.pods:
            total = total.add(pod.total_gpu_request)
        return total


@dataclass
class Queue:
    """Job queue with quotas."""
    name: str
    tenant_id: str
    priority_weight: float = 1.0
    gpu_quota: int = 100
    gpu_used: int = 0
    memory_quota_gb: float = 1000.0
    memory_used_gb: float = 0.0
    max_jobs: int = 100
    pending_jobs: list[str] = field(default_factory=list)
    running_jobs: list[str] = field(default_factory=list)
    preemptible: bool = True

    @property
    def available_gpu_quota(self) -> int:
        return self.gpu_quota - self.gpu_used

    def can_admit(self, job: Job) -> bool:
        """Check if queue can admit job."""
        if len(self.pending_jobs) + len(self.running_jobs) >= self.max_jobs:
            return False
        gpu_req = job.total_gpu_request.count
        if self.gpu_used + gpu_req > self.gpu_quota:
            return False
        return True


@dataclass
class Tenant:
    """Multi-tenant resource management."""
    tenant_id: str
    name: str
    queues: list[str] = field(default_factory=list)
    total_gpu_quota: int = 100
    gpu_used: int = 0
    priority_class: PriorityClass = PriorityClass.NORMAL
    fairshare_weight: float = 1.0

    @property
    def utilization(self) -> float:
        """Get tenant GPU utilization."""
        if self.total_gpu_quota == 0:
            return 0.0
        return self.gpu_used / self.total_gpu_quota


@dataclass
class Cluster:
    """GPU cluster state."""
    cluster_id: str
    nodes: dict[str, Node] = field(default_factory=dict)
    jobs: dict[str, Job] = field(default_factory=dict)
    pods: dict[str, Pod] = field(default_factory=dict)
    queues: dict[str, Queue] = field(default_factory=dict)
    tenants: dict[str, Tenant] = field(default_factory=dict)

    @property
    def total_gpus(self) -> int:
        """Get total GPUs in cluster."""
        return sum(len(n.gpus) for n in self.nodes.values())

    @property
    def available_gpus(self) -> int:
        """Get available GPUs in cluster."""
        return sum(n.available_gpu_count for n in self.nodes.values())

    @property
    def pending_pods(self) -> list[Pod]:
        """Get all pending pods."""
        return [p for p in self.pods.values() if p.state == JobState.PENDING]

    @property
    def running_pods(self) -> list[Pod]:
        """Get all running pods."""
        return [p for p in self.pods.values() if p.state == JobState.RUNNING]

    def add_node(self, node: Node) -> None:
        """Add node to cluster."""
        self.nodes[node.node_id] = node

    def remove_node(self, node_id: str) -> None:
        """Remove node from cluster."""
        if node_id in self.nodes:
            del self.nodes[node_id]

    def submit_job(self, job: Job) -> bool:
        """Submit job to cluster."""
        # Check queue quota
        if job.queue_name in self.queues:
            queue = self.queues[job.queue_name]
            if not queue.can_admit(job):
                return False
            queue.pending_jobs.append(job.job_id)

        self.jobs[job.job_id] = job
        for pod in job.pods:
            self.pods[pod.pod_id] = pod
        return True

    def get_schedulable_nodes(self) -> list[Node]:
        """Get nodes that can accept pods."""
        return [n for n in self.nodes.values() if n.is_schedulable()]


def create_gpu(
    node_id: str,
    gpu_type: GPUType = GPUType.A100,
    memory_gb: float = 80.0,
    local_index: int = 0,
    numa_node: int = 0,
    pcie_bus: str = "",
    nvlink_peers: list[str] | None = None,
) -> GPU:
    """Create a GPU instance."""
    compute_cap = {
        GPUType.A100: (8, 0),
        GPUType.H100: (9, 0),
        GPUType.V100: (7, 0),
        GPUType.T4: (7, 5),
        GPUType.A10G: (8, 6),
        GPUType.L4: (8, 9),
    }

    return GPU(
        gpu_id=str(uuid.uuid4())[:8],
        node_id=node_id,
        gpu_type=gpu_type,
        total_memory_gb=memory_gb,
        available_memory_gb=memory_gb,
        compute_capability=compute_cap.get(gpu_type, (8, 0)),
        local_index=local_index,
        numa_node=numa_node,
        pcie_bus=pcie_bus or f"0000:{local_index:02x}:00.0",
        nvlink_peers=nvlink_peers or [],
    )


def create_node(
    hostname: str,
    num_gpus: int = 8,
    gpu_type: GPUType = GPUType.A100,
    gpu_memory_gb: float = 80.0,
    nvlink_topology: str = "dgx"  # "dgx", "none", or "full"
) -> Node:
    """Create a compute node with GPU topology.

    Args:
        hostname: Node hostname
        num_gpus: Number of GPUs
        gpu_type: Type of GPUs
        gpu_memory_gb: GPU memory in GB
        nvlink_topology: NVLink topology type:
            - "dgx": DGX-style (GPUs 0-3 connected, 4-7 connected)
            - "full": All GPUs connected via NVLink
            - "none": No NVLink connections
    """
    node_id = str(uuid.uuid4())[:8]

    # Create GPUs with local indices
    gpus = []
    for i in range(num_gpus):
        # NUMA node: GPUs 0-3 on NUMA 0, 4-7 on NUMA 1 (for 8-GPU systems)
        numa = 0 if i < num_gpus // 2 else 1

        gpu = create_gpu(
            node_id=node_id,
            gpu_type=gpu_type,
            memory_gb=gpu_memory_gb,
            local_index=i,
            numa_node=numa,
        )
        gpus.append(gpu)

    # Set up NVLink topology
    if nvlink_topology == "dgx" and num_gpus >= 4:
        # DGX-style: GPUs 0-3 fully connected, GPUs 4-7 fully connected
        for i in range(min(4, num_gpus)):
            for j in range(min(4, num_gpus)):
                if i != j:
                    gpus[i].nvlink_peers.append(gpus[j].gpu_id)

        for i in range(4, num_gpus):
            for j in range(4, num_gpus):
                if i != j:
                    gpus[i].nvlink_peers.append(gpus[j].gpu_id)

    elif nvlink_topology == "full":
        # Full mesh: all GPUs connected
        for i in range(num_gpus):
            for j in range(num_gpus):
                if i != j:
                    gpus[i].nvlink_peers.append(gpus[j].gpu_id)

    return Node(
        node_id=node_id,
        hostname=hostname,
        gpus=gpus,
        conditions={"Ready": True}
    )


def create_training_job(
    name: str,
    namespace: str = "default",
    num_gpus: int = 1,
    gpu_memory_gb: float = 16.0,
    parallelism: int = 1,
    priority: PriorityClass = PriorityClass.NORMAL,
    tenant_id: str = "default"
) -> Job:
    """Create a training job."""
    job_id = str(uuid.uuid4())[:8]

    pods = []
    for i in range(parallelism):
        container = Container(
            name="trainer",
            image="training:latest",
            gpu_resources=GPUResources(
                count=num_gpus,
                memory_gb=gpu_memory_gb
            )
        )

        pod = Pod(
            pod_id=f"{job_id}-pod-{i}",
            name=f"{name}-{i}",
            namespace=namespace,
            containers=[container],
            priority=priority
        )
        pods.append(pod)

    return Job(
        job_id=job_id,
        name=name,
        namespace=namespace,
        pods=pods,
        parallelism=parallelism,
        completions=parallelism,
        priority=priority,
        tenant_id=tenant_id
    )
