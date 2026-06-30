# GPU Cluster Scheduler

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A sophisticated GPU cluster scheduler combining features from Kubernetes, Ray, and Slurm for ML workloads. This system implements advanced scheduling algorithms including gang scheduling, fair sharing, preemption, and GPU topology awareness to maximize cluster utilization while providing multi-tenant isolation and QoS guarantees.

## Features

- **Multiple Scheduling Algorithms**
  - FIFO (First-In-First-Out) scheduling
  - Priority-based scheduling with preemption
  - DRF (Dominant Resource Fairness) for multi-tenant clusters
  - Gang scheduling for distributed training jobs

- **GPU Topology Awareness**
  - NVLink-aware GPU placement
  - NUMA-aware scheduling
  - Communication cost optimization
  - Same-node placement constraints

- **Preemption and Checkpointing**
  - Priority-based preemption
  - Job checkpointing support
  - Graceful job termination
  - Automatic job resumption

- **Multi-Tenant Quota Management**
  - Per-tenant resource quotas
  - Burst capacity allowance
  - Usage tracking and reporting
  - Priority limits per tenant

- **Enterprise Features**
  - Backfill scheduling for improved utilization
  - Resource reservations
  - Node health monitoring
  - Comprehensive metrics and reporting

## Quick Start

### Installation

```bash
cd projects/45-gpu-cluster-scheduler
pip install -e ".[full]"
```

### Basic Usage

```python
from gpu_scheduler import ClusterScheduler, Job, ResourceRequirements, JobPriority

# Create scheduler with DRF policy
scheduler = ClusterScheduler(policy='drf')

# Define resource requirements
resources = ResourceRequirements(
    gpus=4,
    gpu_type="A100",
    memory_gb=64,
    cpus=16,
    require_nvlink=True,
    require_same_node=True
)

# Submit a job
job = Job(
    name="training-job",
    user_id="user123",
    tenant_id="ml-team",
    resources=resources,
    priority=JobPriority.HIGH
)

job_id = scheduler.submit(job)
print(f"Submitted job: {job_id}")
```

### Setting Up a Cluster

```python
from gpu_scheduler import ClusterScheduler, Node, GPU

# Create scheduler
scheduler = ClusterScheduler(policy='priority')

# Add nodes to cluster
node = Node(
    node_id="node-0",
    hostname="gpu-server-0",
    total_cpus=64,
    total_memory_gb=512,
    gpus=[
        GPU(gpu_id=f"gpu-0-{i}", node_id="node-0", gpu_type="A100-80GB", memory_gb=80)
        for i in range(8)
    ]
)
scheduler.add_node(node)

# Configure NVLink topology
for i in range(8):
    node.gpus[i].nvlink_peers = [f"gpu-0-{j}" for j in range(8) if j != i]
```

### Multi-Tenant Configuration

```python
from gpu_scheduler import QuotaManager, TenantQuota, JobPriority

# Create quota manager
quota_manager = QuotaManager()

# Define tenant quotas
ml_team_quota = TenantQuota(
    tenant_id="ml-team",
    max_gpus=32,
    max_cpus=256,
    max_memory_gb=2048,
    burst_gpus=8,
    max_priority=JobPriority.HIGH
)
quota_manager.add_quota(ml_team_quota)

# Check quota before submission
if quota_manager.check_quota(job):
    scheduler.submit(job)
```

### Monitoring

```python
# Get job status
status = scheduler.get_job_status(job_id)
print(f"Job state: {status.state}")

# Get cluster utilization
util = scheduler.get_cluster_utilization()
print(f"GPU utilization: {util['gpu_utilization']:.1%}")
print(f"CPU utilization: {util['cpu_utilization']:.1%}")

# Get tenant usage report
report = scheduler.get_tenant_report("ml-team")
print(f"GPU hours used: {report['total_gpu_hours']:.1f}")
```

## Architecture

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

## Scheduling Algorithms

### FIFO Scheduler
Simple first-come-first-served scheduling. Jobs are processed in submission order.

```python
scheduler = ClusterScheduler(policy='fifo')
```

### Priority Scheduler
Higher priority jobs are scheduled first, with optional preemption of lower priority jobs.

```python
scheduler = ClusterScheduler(policy='priority', preemption_enabled=True)
```

### DRF (Dominant Resource Fairness) Scheduler
Ensures fair resource allocation across multiple users based on their dominant resource.

```python
scheduler = ClusterScheduler(policy='drf')
```

### Gang Scheduler
Ensures all workers of a distributed job are scheduled together (all-or-nothing).

```python
job = Job(
    name="distributed-training",
    resources=ResourceRequirements(gpus=8),
    gang_size=4  # 4 workers, each with 8 GPUs = 32 total GPUs
)
```

### Backfill Scheduler
Schedules smaller jobs in gaps while waiting for large jobs to improve utilization.

```python
scheduler = ClusterScheduler(policy='backfill')
```

## Performance Targets

| Metric | Target |
|--------|--------|
| Scheduling latency | <100ms |
| Cluster utilization | >85% |
| Fair share deviation | <5% |
| Preemption time | <30s |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_schedulers.py -v

# Run with coverage
pytest tests/ --cov=gpu_scheduler --cov-report=html
```

## Documentation

- [Architecture Overview](docs/ARCHITECTURE.md) - System design and components
- [API Reference](docs/API.md) - Complete API documentation
- [Deployment Guide](docs/DEPLOYMENT.md) - Production deployment instructions

## Dependencies

- Python >= 3.9
- gRPC for API
- etcd/Redis for state storage (optional)
- Prometheus for metrics (optional)

## Configuration

### Environment Variables

```bash
# Scheduler configuration
export GPU_SCHEDULER_POLICY=drf
export GPU_SCHEDULER_PREEMPTION=true

# State storage
export GPU_SCHEDULER_STATE_BACKEND=redis
export GPU_SCHEDULER_REDIS_URL=redis://localhost:6379

# Metrics
export GPU_SCHEDULER_METRICS_PORT=9090
```

### YAML Configuration

```yaml
scheduler:
  policy: drf
  preemption:
    enabled: true
    grace_period_seconds: 30

quotas:
  default:
    max_gpus: 8
    max_cpus: 64
    max_memory_gb: 256

nodes:
  health_check_interval_seconds: 30
  unhealthy_threshold: 3
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- Kubernetes Scheduler
- YARN Capacity Scheduler
- Slurm Documentation
- Gandiva: Introspective Cluster Scheduling for Deep Learning
- Tiresias: A GPU Cluster Manager for Distributed Deep Learning
