# Feature Engineering Platform

A production-grade ML feature engineering platform providing feature transformations, an online/offline feature store, DAG-based pipelines, feature validation, drift detection, and a REST API for serving.

> **Status:** reference implementation / teaching scaffold built to a strong blueprint — not production-grade. See [../PROJECTS_STATUS.md](../PROJECTS_STATUS.md) and the [2026-06 audit](../../docs/AUDIT_2026-06_public-readiness.md).

> **Concepts covered:** §03 Feature stores · §03 MLOps · §03 ML monitoring (drift detection) · §02 Data pipelines. Pairs with [P04 ML Training Orchestrator](../04-ml-training-orchestrator/) and [P09 Data Observability](../09-data-observability/). See [CONCEPT_TO_PROJECT_MAP.md](../CONCEPT_TO_PROJECT_MAP.md).

---

## What's real vs simulated

The core feature transformations (numeric, categorical, temporal, text), the offline store (DuckDB/Parquet), the in-memory online store, the feature registry, DAG pipeline execution, and drift detection are all fully implemented. The **Phase 5 ML framework wrappers** (scikit-learn Pipeline adapters, PyTorch Dataset integration, TensorFlow `tf.data` connectors) are not yet written — this is a known gap noted in [PROJECTS_STATUS.md](../PROJECTS_STATUS.md). Everything else in the blueprint is present.

---

## Layout

```
src/feature_platform/
  api/            FastAPI REST endpoints for feature serving
  core/           Feature definitions and base types
  discovery/      Feature search and lineage
  monitoring/     Drift detection, alerting, statistics
  pipeline/       DAG execution engine
  store/          Online (in-memory/Redis) and offline (DuckDB) stores
  transformers/   Numeric, categorical, temporal, and text transformers
  validation/     Schema and statistical validation, advanced drift

tests/            ~274 tests across 7 files
BLUEPRINT.md      Full technical design
docs/             API and architecture documentation
```

---

## Build and run

```bash
conda activate dev
cd 06-real-world-projects/50-feature-engineering-platform
pip install -e ".[dev]"
pytest tests/ -v
```

To run with the REST API:

```bash
pip install -e ".[full]"
uvicorn feature_platform.api.main:app --reload
```
