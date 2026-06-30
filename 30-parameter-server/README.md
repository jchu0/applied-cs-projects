# 30 — Large-Scale Parameter Server with Model Sharding

A distributed parameter server for large-scale ML training, implementing async push/pull pipelines, multiple consistency models (Hogwild!, SSP, BSP), gradient compression, and fault-tolerant checkpointing across sharded parameter stores.

> **Status:** reference implementation / teaching scaffold built to a strong blueprint — not production-grade. See [../PROJECTS_STATUS.md](../PROJECTS_STATUS.md) and the [2026-06 audit](../../docs/AUDIT_2026-06_public-readiness.md).

> **Concepts covered:** §03 Data parallelism · §03 Model parallelism (sharding). Compare [Project 40](../40-distributed-autograd/) for AllReduce/FSDP-style training; see [Project 04](../04-ml-training-orchestrator/) for the orchestration layer above both.

---

## What's real vs simulated

The consistency models (BSP, SSP, Hogwild!), sharding logic, gradient compression, optimizers (Adam, LARS), mixed-precision support, and checkpoint/replica managers are fully implemented in Python.

A **real network/RPC transport** now ships in [`src/paramserver/transport.py`](src/paramserver/transport.py): an async length-prefixed RPC over TCP (stdlib `asyncio` only — no gRPC/ZMQ dependency). `serve_parameter_server()` exposes a shard on a socket and `RemoteParameterServer` is a drop-in client proxy, so workers can `pull`/`push` across processes or machines (tested end-to-end over loopback). The high-level `ShardCluster`/consistency code still wires shards together with in-process asyncio primitives — those have not yet been moved onto the transport, so the default multi-shard cluster remains single-process. (Pickle is used on the wire for NumPy payloads; for trusted internal networks only.) Structured logging (stdlib `logging`) is wired into the core server, checkpoint, and health modules; coverage of the remaining modules is still partial.

---

## Layout

```
src/paramserver/
  server/          # ParameterServer, ShardCluster, sharding strategies
  consistency/     # BSP, SSP, Hogwild! consistency managers
  optimizer/       # Adam, LARS, LR schedulers
  fault_tolerance/ # CheckpointManager, ReplicaManager, HealthMonitor
  enterprise/      # Metrics, mixed-precision
  schemas.py       # Pydantic data models

tests/             # 414 test functions across 20 test files (~5 750 lines)
BLUEPRINT.md       # Full architecture, design decisions, implementation phases
PROGRESS.md        # Feature checklist (may be stale — verify against source)
```

---

## Build & Run

```bash
conda activate dev
cd 06-real-world-projects/30-parameter-server
pip install -e ".[dev]"
pytest tests/ -v
```

Run a focused subset:

```bash
pytest tests/test_parameter_server.py tests/test_consistency.py -v
pytest tests/ --cov=src --cov-report=term-missing
```
