# ML Training Orchestrator - Blueprint

## Overview

A production-grade distributed ML training orchestration platform that manages training jobs, GPU resources, distributed training coordination, checkpointing, and experiment tracking. Designed for teams running ML workloads at scale.

> **Concepts covered:** [§03 Distributed training — data parallelism](../../03-machine-learning-engineering/05-distributed-training/data-parallelism/data-parallelism.md) · [§03 Model parallelism](../../03-machine-learning-engineering/05-distributed-training/model-parallelism/model-parallelism.md) · [§03 Ray-based distributed training](../../03-machine-learning-engineering/05-distributed-training/ray/ray-distributed.md) · [§03 Production ML / MLOps](../../03-machine-learning-engineering/04-production-ml/). Pairs with [Project 30 (parameter server)](../30-parameter-server/) and [Project 40 (distributed autograd)](../40-distributed-autograd/) for the actual gradient-distribution mechanics. Map: [`CONCEPT_TO_PROJECT_MAP.md`](../CONCEPT_TO_PROJECT_MAP.md).

## Architecture

```
+------------------+     +------------------+     +------------------+
|   CLI/Web UI     | --> |   FastAPI        | --> |   Scheduler      |
|                  |     |   Gateway        |     |   (Priority Q)   |
+------------------+     +------------------+     +------------------+
                                  |                        |
                    +-------------+-------------+          |
                    |                           |          v
         +------------------+        +------------------+------------------+
         |   Job Manager    | <----> |   Resource       |   GPU Pool       |
         |   (State Machine)|        |   Allocator      |   Manager        |
         +------------------+        +------------------+------------------+
                    |                        |
         +----------+-----------+            |
         |          |           |            |
    +--------+  +--------+  +--------+  +--------+
    |Distrib.|  |Checkpoint|  |Experi.|  |Worker |
    |Training|  |Manager  |  |Tracker|  |Pool   |
    +--------+  +--------+  +--------+  +--------+
```

## Core Components

### Phase 1: Core Job Management (Days 1-3)
- **Job Model**: Define training job structure with config, resources, status
- **Job State Machine**: States (PENDING, QUEUED, RUNNING, PAUSED, COMPLETED, FAILED, CANCELLED)
- **Job Store**: Async storage with SQLAlchemy + PostgreSQL/SQLite
- **Job Lifecycle**: Submit, queue, start, pause, resume, cancel, retry

### Phase 2: Resource Allocation (Days 4-6)
- **Resource Model**: CPU, Memory, GPU, Network bandwidth
- **GPU Manager**: Pool management, utilization tracking, multi-GPU allocation
- **Resource Allocator**: First-fit, Best-fit, Bin-packing strategies
- **Quota System**: Per-user, per-team quotas with enforcement

### Phase 3: Scheduler (Days 7-9)
- **Priority Queue**: Multi-level priority with aging
- **Scheduling Policies**: FIFO, Fair-share, Preemptive, Gang scheduling
- **Backfill Scheduling**: Utilize gaps in resource allocation
- **Affinity/Anti-affinity**: GPU topology awareness

### Phase 4: Distributed Training (Days 10-12)
- **Worker Management**: Worker registration, heartbeats, health checks
- **Communication Backend**: gRPC-based parameter server protocol
- **Collective Operations**: AllReduce, AllGather, Broadcast coordination
- **Elastic Training**: Dynamic worker scaling, fault recovery

### Phase 5: Checkpoint Management (Days 13-15)
- **Checkpoint Store**: Local, S3, GCS storage backends
- **Checkpoint Policies**: Periodic, best-model, on-failure
- **Checkpoint Coordination**: Synchronized checkpointing for distributed jobs
- **Recovery**: Auto-resume from latest checkpoint

### Phase 6: Experiment Tracking (Days 16-18)
- **Experiment Registry**: Track experiments, runs, metrics, artifacts
- **Metric Logging**: Scalars, histograms, images, custom metrics
- **Artifact Store**: Model files, configs, logs
- **Comparison**: Side-by-side run comparison

### Phase 7: API & Integration (Days 19-21)
- **REST API**: FastAPI with async endpoints
- **CLI**: Rich command-line interface
- **Webhooks**: Job state change notifications
- **Observability**: Prometheus metrics, structured logging

## Data Models

### TrainingJob
```python
class TrainingJob:
    id: str
    name: str
    config: JobConfig
    resources: ResourceRequest
    status: JobStatus
    priority: int
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    checkpoints: List[Checkpoint]
    metrics: Dict[str, MetricValue]
```

### ResourceRequest
```python
class ResourceRequest:
    cpus: int
    memory_gb: float
    gpus: int
    gpu_type: Optional[str]
    gpu_memory_gb: Optional[float]
```

### Checkpoint
```python
class Checkpoint:
    id: str
    job_id: str
    epoch: int
    step: int
    path: str
    metrics: Dict[str, float]
    created_at: datetime
```

## Key Algorithms

### Priority Scheduling with Aging
```
effective_priority = base_priority + (now - queue_time) * aging_factor
```

### Resource Bin-Packing
```
1. Sort jobs by resource request (descending)
2. For each job:
   a. Find node with best fit (least waste)
   b. If no fit, wait or preempt lower priority
3. Pack job to selected node
```

### Distributed Checkpoint Coordination
```
1. Leader initiates checkpoint barrier
2. All workers stop training, flush gradients
3. Each worker saves local shard
4. Workers report completion to leader
5. Leader records global checkpoint metadata
6. Resume training on barrier release
```

## Performance Targets

- Job submission: <100ms latency
- Scheduling decision: <50ms for 1000 pending jobs
- Checkpoint coordination: <5s for 100 workers
- Resource utilization: >85% GPU efficiency
- API throughput: >1000 req/s

## Dependencies

- Python 3.10+
- FastAPI + Uvicorn
- SQLAlchemy 2.0 (async)
- Redis (for distributed coordination)
- PostgreSQL (production) / SQLite (dev)
- Prometheus (metrics)
- Structlog (logging)

## Testing Strategy

- Unit tests: Core logic, state machines, algorithms
- Integration tests: API, database, coordination
- Performance tests: Scheduler throughput, checkpoint latency
- Chaos tests: Worker failures, network partitions
