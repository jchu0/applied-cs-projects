"""FastAPI REST API for Model Routing Layer."""

from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .router import ModelRouter, create_router
from .schemas import InferenceRequest, Priority, CongestionThrottled, generate_id


# --- Request/Response models ---

class InferenceSubmitRequest(BaseModel):
    """Request body for inference submission."""
    model: str
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    priority: str = "NORMAL"
    tenant_id: str = "default"
    sla_deadline_ms: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkerRegisterRequest(BaseModel):
    """Request body for worker registration."""
    worker_id: str
    host: str
    port: int
    models: list[str]
    token_budget: int = 100000


# --- Factory ---

def create_app(router: ModelRouter | None = None) -> FastAPI:
    """Create FastAPI application.

    Args:
        router: Optional pre-configured ModelRouter. Creates default if None.

    Returns:
        Configured FastAPI app
    """
    if router is None:
        router = create_router()

    app = FastAPI(title="Model Routing Layer", version="0.1.0")
    app.state.router = router

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/v1/inference")
    async def submit_inference(body: InferenceSubmitRequest):
        try:
            priority = Priority[body.priority]
        except KeyError:
            raise HTTPException(status_code=422, detail=f"Invalid priority: {body.priority}")

        sla_map = {
            Priority.CRITICAL: 1000,
            Priority.HIGH: 3000,
            Priority.NORMAL: 10000,
            Priority.LOW: 30000,
            Priority.BATCH: 300000,
        }

        request = InferenceRequest(
            request_id=generate_id(),
            tenant_id=body.tenant_id,
            model=body.model,
            prompt=body.prompt,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            priority=priority,
            sla_deadline_ms=body.sla_deadline_ms or sla_map[priority],
            metadata=body.metadata,
            estimated_tokens=len(body.prompt) // 4 + body.max_tokens,
        )

        try:
            response = await app.state.router.submit(request)
        except CongestionThrottled as exc:
            raise HTTPException(status_code=503, detail=str(exc))

        return {
            "request_id": response.request_id,
            "text": response.text,
            "tokens_used": response.tokens_used,
            "latency_ms": response.latency_ms,
            "worker_id": response.worker_id,
            "model": response.model,
        }

    @app.post("/workers/register")
    async def register_worker(body: WorkerRegisterRequest):
        worker_id = await app.state.router.register_worker(
            worker_id=body.worker_id,
            host=body.host,
            port=body.port,
            models=body.models,
            token_budget=body.token_budget,
        )
        return {"worker_id": worker_id}

    @app.get("/workers")
    async def list_workers():
        workers = await app.state.router.registry.get_workers(status=None)
        return {
            "workers": [
                {
                    "worker_id": w.worker_id,
                    "host": w.host,
                    "port": w.port,
                    "models": w.models,
                    "status": w.status,
                    "current_load": w.current_load,
                }
                for w in workers
            ]
        }

    @app.get("/capacity/{model}")
    async def get_model_capacity(model: str):
        capacity = await app.state.router.get_capacity(model)
        return asdict(capacity)

    @app.get("/capacity")
    async def get_all_capacity():
        capacity = await app.state.router.get_capacity()
        return {model: asdict(info) for model, info in capacity.items()}

    @app.get("/queue/stats")
    async def get_queue_stats():
        return await app.state.router.get_queue_stats()

    @app.get("/metrics")
    async def get_metrics(limit: int = 100):
        return app.state.router.get_metrics(limit=limit)

    return app
