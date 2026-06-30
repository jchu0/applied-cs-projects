# Data Observability Platform

> **Concepts covered:** §02 data-engineering — `05-data-quality`; §05 cross-cutting-concerns — `observability`

A data observability platform with ML-based anomaly detection: freshness/volume/schema-drift monitors, metric collection, and alerting over data pipelines — implemented from scratch in Python.

> **Status:** reference implementation / teaching scaffold built to a strong blueprint — not a production-grade system. See the production-readiness note in [`../PROJECTS_STATUS.md`](../PROJECTS_STATUS.md) and the [2026-06 audit](../../docs/AUDIT_2026-06_public-readiness.md).

## Design

The architecture and design decisions live in [`docs/BLUEPRINT.md`](docs/BLUEPRINT.md). Implementation status is tracked in [`docs/PROGRESS.md`](docs/PROGRESS.md).

## Layout

```
src/observability/   # monitors, detectors, alerting
tests/               # 15 test modules
docs/                # BLUEPRINT.md (design), PROGRESS.md (status)
```

## Running

```bash
conda activate dev
cd 06-real-world-projects/09-data-observability
pip install -e ".[dev]"        # check pyproject.toml for available extras
pytest tests/ -v
```
