# ML Training Orchestrator

A distributed ML training orchestration platform that manages training jobs, GPU resources,
scheduling, distributed training coordination, checkpointing, and experiment tracking.

> **Status:** reference implementation / teaching scaffold built to a strong blueprint — not production-grade. See [../PROJECTS_STATUS.md](../PROJECTS_STATUS.md) and the [2026-06 audit](../../docs/AUDIT_2026-06_public-readiness.md).

> **Concepts covered:** §03 distributed training (data parallelism, model parallelism, Ray),
> §03 production-ML / MLOps. Pairs with [Project 30 (parameter server)](../30-parameter-server/)
> and [Project 40 (distributed autograd)](../40-distributed-autograd/).
> See [../CONCEPT_TO_PROJECT_MAP.md](../CONCEPT_TO_PROJECT_MAP.md).

## What's real vs simulated

- **Real:** job state machine, priority-queue scheduler (FIFO / fair-share / preemptive),
  GPU pool tracking, resource allocator (first-fit / best-fit / bin-packing), quota enforcement,
  checkpoint manager, experiment registry, metric logging, and the FastAPI REST gateway.
  Three real concurrency bugs (priority-queue tombstone leak, non-reentrant-lock deadlocks in
  `NodeManager` and `ResourceAllocator`) were discovered and fixed during testing.
- **Simulated / incomplete:** the distributed training layer (`src/ml_orchestrator/distributed/`)
  coordinates collective operations (AllReduce, AllGather, Broadcast) through in-process logic —
  there is no real PyTorch `dist` backend or inter-process communication. The CLI is a thin HTTP
  client with minimal coverage. No GPU hardware is required or queried at runtime.

## Layout

```
src/ml_orchestrator/
  core/           # TrainingJob models, JobManager state machine, exceptions
  resources/      # GPUManager, NodeManager, ResourceAllocator, quota
  scheduling/     # PriorityQueue, scheduling policies, Scheduler
  distributed/    # Collective ops coordinator, elastic training
  checkpoint/     # CheckpointManager, coordinator, storage backends
  experiment/     # ExperimentTracker, artifact store, run comparison
  api/            # FastAPI REST gateway (jobs, resources, experiments, health)
  cli.py          # Rich CLI (HTTP client wrapper)
tests/
  unit/           # ~60 unit-test files, ~167 test functions
BLUEPRINT.md      # Full architecture, data models, implementation phases
```

## Build & Run

```bash
conda activate dev
cd 06-real-world-projects/04-ml-training-orchestrator
pip install -e ".[dev]"
pytest tests/ -v
```

Coverage is ~60 % (Tier 2). A PostgreSQL instance is not required for the unit tests — the job
store falls back to SQLite in-memory. For the full API, start a backing DB and run:

```bash
uvicorn ml_orchestrator.api.app:app --reload
```
