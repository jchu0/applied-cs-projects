# Project 28: AI Workflow Engine - Setup Guide

## Overview

A from-scratch workflow orchestration engine for AI/ML pipelines. Flows are
declared in YAML, JSON, a dict, or a small indentation-based DSL, compiled into
a DAG, and executed level-by-level with bounded async parallelism. The package
is `aiworkflow` and lives under `src/aiworkflow/`.

There is **no** database, message broker, Redis/PostgreSQL, or distributed
(Ray/Dask) backend. Run history, the version store, and the human-in-the-loop
review store are all in-memory. See the README's "What's Real vs Simulated"
section for what is actually implemented.

## Prerequisites

- Python 3.9+
- `pyyaml` — required for the compiler and engine. Without it the package still
  imports (schemas only) but `WorkflowEngine`, `FlowParser`, etc. are `None`.
- FastAPI + Uvicorn — only needed for the optional REST API (the `api` extra).

## Installation

Install the package in editable mode. Extras are declared in `pyproject.toml`:

```bash
# core + test tooling
pip install -e ".[dev]"

# also install FastAPI/uvicorn for the REST API
pip install -e ".[api,dev]"
```

Verify the install:

```bash
python -c "import aiworkflow; print(aiworkflow.__version__)"
```

## Configuration

The core library needs no configuration. The optional REST API reads a few
environment variables (stdlib only, no extra deps). All are optional:

| Env var | Default | Effect |
| --- | --- | --- |
| `API_KEYS` | *(unset)* | Comma-separated valid keys. Unset/empty disables auth (a startup warning is logged). |
| `RATE_LIMIT_PER_MINUTE` | `120` | Per-client request cap (by API key, else client IP). `0` disables. Over-limit returns `429` with `Retry-After`. |
| `REQUEST_TIMEOUT_SECONDS` | `30` | Per-request wall-clock budget. `0` disables. On timeout returns `504`. |

See `.env.example` for a template.

## Usage

### Define and run a flow programmatically

LLM and retrieval executors return deterministic mock output — no model API is
called (see the README).

```python
import asyncio
from aiworkflow import WorkflowEngine, FlowDefinition, Node, NodeType, NodeConfig

flow = FlowDefinition(
    name="qa-flow",
    version="1.0.0",
    nodes=[
        Node(id="retriever", type=NodeType.RETRIEVAL,
             config=NodeConfig(extra={"top_k": 3}),
             inputs={"query": "{{inputs.question}}"}),
        Node(id="generator", type=NodeType.LLM,
             config=NodeConfig(model="gpt-4", prompt_template="Q: {{question}}"),
             dependencies=["retriever"],
             inputs={"question": "{{inputs.question}}"}),
    ],
    outputs={"answer": "generator"},
)

engine = WorkflowEngine(max_parallel=5)

async def main():
    run = await engine.run_flow(flow_definition=flow, inputs={"question": "What is a DAG?"})
    print(run.status, run.outputs)

asyncio.run(main())
```

### Register a flow from YAML and execute by name

```python
import asyncio
from aiworkflow import WorkflowEngine

flow_yaml = """
name: qa_workflow
version: "1.0.0"
description: Q&A workflow with retrieval and generation

nodes:
  - id: retriever
    type: retrieval
    inputs:
      query: "{{inputs.question}}"
    config:
      extra:
        top_k: 5

  - id: generator
    type: llm
    dependencies:
      - retriever
    inputs:
      question: "{{inputs.question}}"
    config:
      model: gpt-4
      prompt_template: |
        Question: {{question}}
        Answer:

outputs:
  answer: generator
"""

engine = WorkflowEngine()
flow = engine.register_flow(flow_yaml)
run = asyncio.run(engine.execute(flow.name, inputs={"question": "What is machine learning?"}))
print(run.status, run.outputs)
```

The package also ships an `EXAMPLE_FLOW` YAML string you can register directly:

```python
from aiworkflow import WorkflowEngine, EXAMPLE_FLOW

engine = WorkflowEngine()
flow = engine.register_flow(EXAMPLE_FLOW)
```

### Built-in node types

Valid `type` values (from `NodeType` in `schemas.py`): `llm`, `retrieval`,
`tool`, `branch`, `transform`, `subflow`, `human_review`, `data`, `process`,
`model`, `conditional`, `validation`. There is no `rerank`, `web_search`, or
`merge` node type — model those with `transform`/`tool` nodes or a custom
executor.

### Custom node executors

`BaseNodeExecutor.execute` receives the `Node` and a resolved `inputs` dict:

```python
from aiworkflow.nodes import BaseNodeExecutor

class DoubleExecutor(BaseNodeExecutor):
    async def execute(self, node, inputs):
        return {"processed": [x * 2 for x in inputs.get("data", [])]}

engine.register_node_executor("doubler", DoubleExecutor())
# reference it from a node via Node(..., executor="doubler")
```

### Batch execution

```python
results = await engine.execute_batch(
    flow_name="qa_workflow",
    inputs_list=[{"question": "What is ML?"}, {"question": "What is DL?"}],
)
for run in results:
    print(run.status, run.outputs)
```

### Versioning and run history

```python
# Versioning is on by default; register_flow saves a content-hashed version.
flow_v1 = engine.get_flow("qa_workflow", version="1.0.0")

# Run history (in-memory)
for run in engine.get_run_history(flow_name="qa_workflow", limit=100):
    print(run.run_id, run.status)

single = engine.get_run("some-run-id")
```

### Configured engine factory

```python
from aiworkflow import create_engine

engine = create_engine(
    max_parallel=10,
    enable_versioning=True,
    enable_optimization=True,
)
```

## Running the REST API

The FastAPI app is exported as `aiworkflow.api:app`:

```bash
uvicorn aiworkflow.api:app --reload
```

With auth enabled, send the key as `Authorization: Bearer <key>` or
`X-API-Key: <key>`:

```bash
API_KEYS=mysecret uvicorn aiworkflow.api:app
curl -H "Authorization: Bearer mysecret" http://localhost:8000/flows
```

`/health`, `/`, and the docs (`/docs`, `/redoc`, `/openapi.json`) stay open.

## Testing

```bash
# run the suite (asyncio_mode = auto)
pytest

# with coverage
pytest --cov=aiworkflow

# a single file
pytest tests/test_engine.py -v
```

No external services are required — LLM/retrieval calls are mocked in-process.

## Project Structure

```
28-ai-workflow-engine/
├── src/aiworkflow/
│   ├── schemas.py         # core dataclasses and enums
│   ├── engine.py          # WorkflowEngine, create_engine, EXAMPLE_FLOW
│   ├── compiler/          # parser, validator, DAG builder, optimizer
│   ├── executor/          # Scheduler, AsyncScheduler
│   ├── nodes/             # node executors and the registry
│   ├── retry/             # retry strategies, circuit breaker, RetryManager
│   ├── versioning/        # FlowVersionManager, MigrationManager
│   ├── enterprise/        # HITL review, secrets, visualization
│   └── api/               # FastAPI app
├── tests/                 # test suite
├── docs/                  # ARCHITECTURE.md, API.md, BLUEPRINT.md, this file
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

## Resources

- [YAML Specification](https://yaml.org/spec/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Uvicorn](https://www.uvicorn.org/)
</content>
</invoke>
