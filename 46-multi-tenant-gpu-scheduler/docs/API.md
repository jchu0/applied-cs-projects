# Multi-Tenant GPU Scheduler - API Documentation

## Table of Contents

1. [Job Management API](#job-management-api)
2. [Resource Management API](#resource-management-api)
3. [Scheduling API](#scheduling-api)
4. [Monitoring API](#monitoring-api)
5. [Allocation API](#allocation-api)

---

## Job Management API

### Create Training Job

Create a new training job with GPU requirements.

```python
from gpusched.core.resources import create_training_job, PriorityClass

job = create_training_job(
    name="model-training",
    namespace="ml-team",
    num_gpus=4,
    gpu_memory_gb=80.0,
    parallelism=2,  # Number of parallel pods
    priority=PriorityClass.HIGH,
    tenant_id="tenant-ml"
)
```

**Parameters:**
- `name` (str): Job name
- `namespace` (str): Kubernetes-like namespace
- `num_gpus` (int): GPUs per pod
- `gpu_memory_gb` (float): GPU memory requirement
- `parallelism` (int): Number of parallel pods
- `priority` (PriorityClass): Job priority level
- `tenant_id` (str): Tenant identifier

**Returns:** `Job` object

### Submit Job to Cluster

```python
from gpusched.core.resources import Cluster

cluster = Cluster("production")
success = cluster.submit_job(job)
```

**Returns:** `bool` - Success status

### Create Custom Pod

```python
from gpusched.core.resources import Pod, Container, GPUResources, GPUType

container = Container(
    name="trainer",
    image="pytorch:2.0",
    gpu_resources=GPUResources(
        count=2,
        memory_gb=40.0,
        compute_units=1.0,
        gpu_type=GPUType.A100
    ),
    env={"BATCH_SIZE": "64", "EPOCHS": "100"}
)

pod = Pod(
    pod_id="custom-pod-001",
    name="distributed-training",
    namespace="research",
    containers=[container],
    node_selector={"zone": "us-west"},
    tolerations=["gpu=NoSchedule"]
)
```

---

## Resource Management API

### Create and Manage Nodes

```python
from gpusched.core.resources import create_node, GPUType

# Create node with GPUs
node = create_node(
    hostname="gpu-server-01",
    num_gpus=8,
    gpu_type=GPUType.A100,
    gpu_memory_gb=80.0
)

# Add labels
node.labels = {
    "zone": "us-west-1",
    "type": "training",
    "gpu": "a100"
}

# Add taints
node.taints = ["dedicated=ml-team:NoSchedule"]

# Add to cluster
cluster.add_node(node)
```

### Manage Tenants

```python
from gpusched.core.resources import Tenant, PriorityClass

tenant = Tenant(
    tenant_id="tenant-research",
    name="Research Team",
    total_gpu_quota=32,
    priority_class=PriorityClass.HIGH,
    fairshare_weight=2.0
)

cluster.tenants[tenant.tenant_id] = tenant
```

### Create Queues

```python
from gpusched.core.resources import Queue

queue = Queue(
    name="high-priority-training",
    tenant_id="tenant-research",
    priority_weight=2.0,
    gpu_quota=16,
    memory_quota_gb=1280.0,
    max_jobs=10,
    preemptible=False
)

cluster.queues[queue.name] = queue
```

---

## Scheduling API

### Basic GPU Scheduling

```python
from gpusched.scheduler.scheduler import GPUScheduler

scheduler = GPUScheduler(cluster)

# Schedule single pod
decision = scheduler.schedule_pod(pod)

if decision.success:
    print(f"Scheduled on node: {decision.node_id}")
    print(f"Assigned GPUs: {decision.gpu_ids}")
    print(f"Score: {decision.score}")
else:
    print(f"Failed: {decision.reason}")
```

### Gang Scheduling

```python
# For distributed training requiring all pods to start together
job.gang_schedule = True
decisions = scheduler.schedule_gang(job)

# Check if all pods scheduled
all_success = all(d.success for d in decisions)
```

### Queue-based Scheduling

```python
from gpusched.scheduler.scheduler import QueueScheduler

queue_scheduler = QueueScheduler(cluster)

# Enqueue pod
queue_scheduler.enqueue(pod, queue_name="high-priority-training")

# Run scheduling cycle
decisions = queue_scheduler.schedule()
```

### Preemption Scheduling

```python
from gpusched.scheduler.scheduler import PreemptionScheduler

preemption_scheduler = PreemptionScheduler(cluster)

# Try scheduling with preemption
decision = preemption_scheduler.schedule_with_preemption(high_priority_pod)

if decision.success and "Preempting" in decision.reason:
    print(f"Preempting lower priority pods to schedule")
```

### Custom Scheduling Plugin

```python
from gpusched.scheduler.scheduler import SchedulingPlugin, SchedulingContext

class CustomPlugin(SchedulingPlugin):
    def name(self) -> str:
        return "CustomScoring"

    def filter(self, ctx: SchedulingContext, node: Node) -> bool:
        # Custom filtering logic
        return node.available_gpu_count >= ctx.pod.total_gpu_request.count

    def score(self, ctx: SchedulingContext, node: Node) -> float:
        # Custom scoring logic
        return node.available_gpu_count * 10

# Add to scheduler
scheduler.add_plugin(CustomPlugin(), weight=1.5)
```

---

## Monitoring API

### Collect Metrics

```python
from gpusched.monitor.monitor import MetricsCollector

collector = MetricsCollector(cluster, collection_interval=60)

# Collect current metrics
collector.collect_metrics()

# Get summary
summary = collector.get_metrics_summary()
print(f"Cluster utilization: {summary['avg_gpu_utilization']}%")
print(f"Available GPUs: {summary['available_gpus']}")
```

### Monitor GPUs

```python
from gpusched.monitor.monitor import GPUMonitor

monitor = GPUMonitor(cluster)

# Start monitoring
monitor.start_monitoring(interval=30)

# Get current status
status = monitor.get_status()
print(f"Cluster health: {status['cluster_health']['overall_health']}")

# Get GPU statistics
gpu_stats = monitor.get_gpu_stats()
for stat in gpu_stats:
    print(f"GPU {stat['gpu_id']}: {stat['utilization']}% utilized")
```

### Export Metrics

```python
# Prometheus format
prometheus_metrics = monitor.export_metrics(format="prometheus")
print(prometheus_metrics)

# JSON format
json_metrics = monitor.export_metrics(format="json")
```

### Alert Management

```python
from gpusched.monitor.monitor import AlertManager, AlertLevel

alert_manager = AlertManager()

# Create custom alert
alert_id = alert_manager.create_alert(
    level=AlertLevel.WARNING,
    message="GPU utilization above 90%",
    source="gpu-001",
    details={"utilization": 92.5, "temperature": 78}
)

# Get active alerts
critical_alerts = alert_manager.get_active_alerts(min_level=AlertLevel.CRITICAL)

# Resolve alert
alert_manager.resolve_alert(alert_id)
```

### Health Checking

```python
from gpusched.monitor.monitor import HealthChecker

health_checker = HealthChecker()

# Check cluster health
health = health_checker.check_cluster_health(cluster)
print(f"Healthy nodes: {health['healthy_nodes']}/{health['total_nodes']}")
print(f"Healthy GPUs: {health['healthy_gpus']}/{health['total_gpus']}")

# Check specific GPU
gpu_health = health_checker.check_gpu_health(gpu)
if not gpu_health["healthy"]:
    print(f"Issues: {gpu_health['issues']}")
```

---

## Allocation API

### Allocate GPU Resources

```python
from gpusched.allocator.allocator import AllocationManager

allocator = AllocationManager(cluster)

# Allocate resources for pod
allocations = allocator.allocate_pod(
    pod=pod,
    node_id="node-001",
    gpu_ids=["gpu-001", "gpu-002"]
)

for allocation in allocations:
    print(f"Allocated {allocation.memory_allocated_gb}GB on {allocation.gpu_id}")
```

### MIG Allocation

```python
from gpusched.allocator.allocator import MIGAllocator

mig_allocator = MIGAllocator()

# Check MIG support
if mig_allocator.can_use_mig(gpu):
    # Find suitable MIG profile
    profile = mig_allocator.find_mig_profile(pod.total_gpu_request)
    print(f"Using MIG profile: {profile.profile_name}")

    # Allocate MIG instance
    allocation = mig_allocator.allocate(pod, node, gpu)
    print(f"MIG instance: {allocation.mig_instance_id}")
```

### Shared GPU Allocation

```python
from gpusched.allocator.allocator import SharedGPUAllocator

shared_allocator = SharedGPUAllocator(max_sharing_factor=4)

# Allocate shared GPU
allocation = shared_allocator.allocate(pod, node, gpu)
print(f"Compute fraction: {allocation.compute_fraction}")
```

### Release Allocations

```python
# Release when pod completes
allocator.release_pod(pod.pod_id)

# Get allocation statistics
stats = allocator.update_allocation_stats()
print(f"Total allocations: {stats['total_allocations']}")
print(f"GPUs allocated: {stats['total_gpus_allocated']}")
```

---

## Complete Example

### End-to-End Job Scheduling

```python
from gpusched.core.resources import (
    Cluster, create_node, create_training_job,
    Queue, Tenant, PriorityClass, GPUType
)
from gpusched.scheduler.scheduler import QueueScheduler
from gpusched.allocator.allocator import AllocationManager
from gpusched.monitor.monitor import GPUMonitor

# Initialize cluster
cluster = Cluster("ml-cluster")

# Add nodes
for i in range(4):
    node = create_node(
        hostname=f"gpu-node-{i:02d}",
        num_gpus=8,
        gpu_type=GPUType.A100,
        gpu_memory_gb=80.0
    )
    cluster.add_node(node)

# Create tenant
tenant = Tenant(
    tenant_id="ml-team",
    name="Machine Learning Team",
    total_gpu_quota=16
)
cluster.tenants[tenant.tenant_id] = tenant

# Create queue
queue = Queue(
    name="training-queue",
    tenant_id="ml-team",
    gpu_quota=16,
    priority_weight=1.0
)
cluster.queues[queue.name] = queue

# Initialize components
scheduler = QueueScheduler(cluster)
allocator = AllocationManager(cluster)
monitor = GPUMonitor(cluster)

# Create and submit job
job = create_training_job(
    name="bert-training",
    num_gpus=4,
    gpu_memory_gb=60.0,
    parallelism=2,
    priority=PriorityClass.HIGH,
    tenant_id="ml-team"
)
job.queue_name = "training-queue"

# Submit job
cluster.submit_job(job)

# Enqueue pods
for pod in job.pods:
    scheduler.enqueue(pod, queue.name)

# Schedule pods
decisions = scheduler.schedule()

# Process scheduling decisions
for decision in decisions:
    if decision.success:
        pod = cluster.pods[decision.pod_id]

        # Update pod state
        pod.state = JobState.SCHEDULED
        pod.assigned_node = decision.node_id
        pod.assigned_gpus = decision.gpu_ids

        # Allocate resources
        allocations = allocator.allocate_pod(
            pod,
            decision.node_id,
            decision.gpu_ids
        )

        # Start pod
        pod.state = JobState.RUNNING
        pod.started_at = time.time()

        print(f"Pod {pod.name} running on {decision.node_id}")

# Monitor execution
monitor.start_monitoring(interval=60)
status = monitor.get_status()
print(f"Cluster health: {status['cluster_health']['overall_health']}")

# Check for alerts
alerts = monitor.alert_manager.get_active_alerts()
for alert in alerts:
    print(f"Alert: {alert.message}")

# On completion
for pod in job.pods:
    pod.state = JobState.COMPLETED
    pod.completed_at = time.time()
    allocator.release_pod(pod.pod_id)

print(f"Job {job.name} completed successfully")
```

---

## Error Handling

### Common Error Scenarios

```python
# Handle scheduling failures
decision = scheduler.schedule_pod(pod)
if not decision.success:
    if "No feasible nodes" in decision.reason:
        # No nodes meet requirements
        print("Insufficient resources, pod pending")
    elif "quota exceeded" in decision.reason:
        # Quota limit reached
        print("Tenant quota exceeded")
    else:
        print(f"Scheduling failed: {decision.reason}")

# Handle allocation failures
try:
    allocations = allocator.allocate_pod(pod, node_id, gpu_ids)
except ValueError as e:
    print(f"Allocation error: {e}")
    # Rollback scheduling decision
    pod.state = JobState.PENDING
    pod.assigned_node = None

# Handle node failures
if not node.is_schedulable():
    # Reschedule pods from failed node
    affected_pods = [p for p in cluster.pods.values()
                    if p.assigned_node == node.node_id]
    for pod in affected_pods:
        pod.state = JobState.PENDING
        pod.assigned_node = None
        # Re-enqueue for scheduling
```

---

## Best Practices

1. **Resource Requests**: Always specify accurate resource requirements
2. **Priority Classes**: Use appropriate priority levels for workloads
3. **Preemption**: Mark batch jobs as preemptible
4. **Monitoring**: Set up alerts for critical thresholds
5. **Quotas**: Configure reasonable quotas per tenant
6. **Affinity Rules**: Use node selectors for specific hardware requirements
7. **Gang Scheduling**: Use only for truly distributed workloads
8. **Health Checks**: Regularly check cluster and GPU health