# Streaming Platform

> **Concepts covered:** §02 data-engineering — `04-streaming`; §01 software-engineering — `python/03-concurrency`

Real-time streaming platform exploring Kafka- and Flink-style stream processing: ingestion, windowed aggregation, and exactly-once-style delivery semantics, implemented from scratch in Python.

> **Status:** reference implementation / teaching scaffold built to a strong blueprint — not a production-grade system. See the production-readiness note in [`../PROJECTS_STATUS.md`](../PROJECTS_STATUS.md) and the [2026-06 audit](../../docs/AUDIT_2026-06_public-readiness.md).

## Design

The architecture and design decisions live in [`docs/BLUEPRINT.md`](docs/BLUEPRINT.md). Implementation status is tracked in [`docs/PROGRESS.md`](docs/PROGRESS.md).

## Layout

```
src/streaming/   # core stream-processing engine
tests/           # 9 test modules
docs/            # BLUEPRINT.md (design), PROGRESS.md (status)
docker-compose.yml
```

## Running

```bash
conda activate dev
cd 06-real-world-projects/08-streaming-platform
pip install -e ".[dev]"        # check pyproject.toml for available extras
pytest tests/ -v
docker-compose up -d           # optional: backing services (Kafka, etc.)
```
