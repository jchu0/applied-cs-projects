# Project 29: Model Routing Layer - Setup Guide

## Overview

A capacity-aware scheduling and routing layer for LLM inference clusters. Requests arrive
through a FastAPI gateway, land on per-model priority queues, and are routed to workers by
one of six strategies. Everything runs in-process with pure Python + NumPy — no external
inference backend, database, or message broker is required to run or test it.

This guide covers only what the code actually does. For the full design, see
[BLUEPRINT.md](BLUEPRINT.md); for a feature overview, see the top-level
[README](../README.md).

## Prerequisites

- Python 3.11+ (the package metadata declares `>=3.9`; the codebase uses 3.11+ syntax such
  as `str | None` in module scope).
- No external services. Worker execution, GPU metrics, and health pings are simulated in
  process. The `docker-compose.yml` at the project root only provisions Redis/PostgreSQL for
  experimentation — nothing in the running code connects to them.

## Installation

From the project root (`29-model-routing-layer/`):

```bash
pip install -e ".[dev]"
```

This installs the `modelrouter` package (from `src/`) plus the test/lint tooling. The API
server dependencies (`fastapi`, `uvicorn`) are pulled in by the `dev` extra transitively via
the package's runtime deps and the `api` extra is available if you want them explicitly:

```bash
pip install -e ".[api]"   # fastapi + uvicorn + aiohttp, if not already present
```

Optional extras defined in `pyproject.toml`: `ml`, `llm`, `api`, `ratelimit`, `queue`,
`database`, `gpu`, `observability`, `full`. None are required for the core router or the
served HTTP endpoints — they exist for experimentation and are not wired into the running
code.

## Running the API server

The FastAPI app is created by a factory, `modelrouter.api:create_app`:

```bash
uvicorn modelrouter.api:create_app --factory --host 0.0.0.0 --port 8000
```

Interactive docs are served at `http://localhost:8000/docs`.

### Endpoints

All endpoints except `/health` (and the docs) are subject to the opt-in HTTP-layer auth /
rate-limit / timeout described below.

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness check; returns `{"status": "ok"}`. Always open. |
| POST | `/v1/inference` | Submit an inference request. |
| POST | `/workers/register` | Register a worker. |
| GET | `/workers` | List registered workers. |
| GET | `/capacity` | Aggregate capacity for all models. |
| GET | `/capacity/{model}` | Capacity for one model. |
| GET | `/queue/stats` | Per-model queue statistics. |
| GET | `/metrics` | Recent request metrics (`?limit=` optional, default 100). |

### Example: register a worker and submit a request

```bash
curl -X POST localhost:8000/workers/register \
  -H 'content-type: application/json' \
  -d '{"worker_id":"w1","host":"localhost","port":8080,"models":["gpt-4"]}'

curl -X POST localhost:8000/v1/inference \
  -H 'content-type: application/json' \
  -d '{"model":"gpt-4","prompt":"Hello","priority":"NORMAL"}'
```

The `/v1/inference` body maps to `InferenceSubmitRequest` in `api.py`:
`model` (str, required), `prompt` (str, required), `max_tokens` (int, default 100),
`temperature` (float, default 0.7), `priority` (str, default `"NORMAL"`),
`tenant_id` (str, default `"default"`), `sla_deadline_ms` (int, optional),
`metadata` (object, optional). Valid `priority` values are `CRITICAL`, `HIGH`, `NORMAL`,
`LOW`, `BATCH`.

## Configuration (environment variables)

The only environment variables the running code reads are the opt-in HTTP-layer hardening
knobs in `src/modelrouter/security.py`. All are optional.

| Env var | Default | Effect |
|---------|---------|--------|
| `API_KEYS` | *(unset)* | Comma-separated valid keys. When unset/empty, auth is **disabled** (a single startup warning is logged). When set, every endpoint except `/health`, `/`, and the docs (`/docs`, `/redoc`, `/openapi.json`) requires a key via `Authorization: Bearer <key>` or `X-API-Key: <key>`; missing/invalid returns 401. |
| `RATE_LIMIT_PER_MINUTE` | `120` | In-process sliding-window request cap keyed by API key (or client IP if none). `0` disables. Over-limit returns 429 with a `Retry-After` header. |
| `REQUEST_TIMEOUT_SECONDS` | `30` | Per-request timeout; on expiry returns 504. `0` disables. |

```bash
API_KEYS=secret-key uvicorn modelrouter.api:create_app --factory --port 8000 &

curl -X POST localhost:8000/v1/inference \
  -H 'content-type: application/json' \
  -H 'Authorization: Bearer secret-key' \
  -d '{"model":"gpt-4","prompt":"Hello","priority":"NORMAL"}'
```

> The `.env.example` at the project root lists many additional variables
> (`OPENAI_API_KEY`, `REDIS_HOST`, `DATABASE_URL`, `PROMETHEUS_PORT`, etc.). Those are
> placeholders for a hypothetical production deployment and are **not** read by the current
> code. Only the three variables above have any effect.

## Programmatic usage

The router can be driven directly, without the HTTP layer:

```python
import asyncio
from modelrouter import create_router, InferenceRequest, Priority, generate_id

async def main():
    router = create_router(routing_strategy="least_loaded", max_queue_depth=1000)

    await router.register_worker(
        worker_id="worker-1", host="localhost", port=8080,
        models=["gpt-4"], token_budget=100_000,
    )
    router.register_tenant(
        tenant_id="tenant-1", name="Demo", api_key="test-key",
    )

    request = InferenceRequest(
        request_id=generate_id(), tenant_id="tenant-1", model="gpt-4",
        prompt="What is 2 + 2?", max_tokens=50, temperature=0.7,
        priority=Priority.NORMAL, estimated_tokens=65,
    )
    response = await router.submit(request)
    print(response.worker_id, response.tokens_used)

    # Switch to the learned RL policy at runtime
    router.set_routing_strategy("rl")

asyncio.run(main())
```

`create_router(routing_strategy=..., max_queue_depth=...)` is the only public factory; both
arguments have defaults (`"least_loaded"`, `1000`).

### Routing strategies

Set via `create_router(routing_strategy=...)` or `router.set_routing_strategy(...)`. The six
strategies implemented in `scheduler/router.py` are:

- `least_loaded` — worker with the lowest current load (default).
- `round_robin` — rotate across eligible workers.
- `token_based` — factor in per-worker token budget.
- `sla_based` — bias toward meeting the request's SLA deadline.
- `weighted` — weighted selection.
- `rl` — the learned NumPy DQN policy (`RLRouter`).

## Testing

```bash
pytest tests/ -v
```

The suite covers each routing strategy, queue ordering, rate limiting, quota and preemption
logic, the RL and congestion networks, the FastAPI endpoints, and the HTTP-layer hardening.
No external services are needed.

If you run tests without installing the package, put `src/` on the path:

```bash
PYTHONPATH=src pytest tests/ -v
```

## What's simulated

Worker execution is mocked — `ModelRouter._execute_on_worker` sleeps briefly and returns a
synthetic `InferenceResponse` instead of calling a real backend. GPU metrics
(`GPUMonitor.collect_metrics`) and the worker ping in `HealthChecker` return fixed values;
production would use `pynvml` / HTTP. The registry, rate limiter, and quota stores are
in-memory. See the "What's Real vs Simulated" section of the [README](../README.md) for the
full breakdown.

## Project structure

```
29-model-routing-layer/
  src/modelrouter/
    api.py            # FastAPI app + endpoints (create_app factory)
    router.py         # ModelRouter orchestrator + create_router
    schemas.py        # Dataclasses and enums
    security.py       # HTTP-layer API-key auth, rate limit, timeout
    gateway/          # Per-tenant auth, rate limiting, token estimation
    scheduler/        # Queue, cost, routing engine (six strategies)
    registry/         # Worker registry, health, GPU monitor
    enterprise/       # Quotas, preemption, traffic splitting
    optimization/     # RL router, congestion prediction
  tests/              # Pytest suite
  docs/
    BLUEPRINT.md      # Full architecture and design
    SETUP.md          # This file
```
</content>
</invoke>
