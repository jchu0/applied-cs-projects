"""FastAPI REST API for the workflow engine.

Exposes flow registration, (inline and registered) execution, run history, and
human-in-the-loop review resolution. FastAPI is an optional dependency (the
``api`` extra); importing this module requires it, but the core engine does not.

Build an app with :func:`create_app`, optionally injecting a configured engine::

    from aiworkflow.api import create_app
    app = create_app()
    # uvicorn aiworkflow.api:app  (module-level `app` is created lazily below)
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..engine import WorkflowEngine
from ..enterprise import to_mermaid, to_dot


# --------------------------------------------------------------- request models

class RegisterFlowRequest(BaseModel):
    spec: dict = Field(..., description="Flow definition (YAML/JSON-as-dict)")
    description: str = ""


class RunRequest(BaseModel):
    inputs: dict = Field(default_factory=dict)


class InlineRunRequest(BaseModel):
    spec: dict
    inputs: dict = Field(default_factory=dict)


class ReviewDecision(BaseModel):
    reviewer: Optional[str] = None
    comment: str = ""
    data: dict = Field(default_factory=dict)


# --------------------------------------------------------------- serialization

def _safe(obj: Any) -> Any:
    """Coerce a value into something JSON-serializable, stringifying unknowns."""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    return str(obj)


def _serialize_run(run) -> dict:
    """Render a FlowRun as a JSON-safe dict."""
    return {
        "run_id": run.run_id,
        "flow_id": run.flow_id,
        "flow_version": run.flow_version,
        "status": run.status.value,
        "inputs": _safe(run.inputs),
        "outputs": _safe(run.outputs),
        "error": run.error,
        "start_time": run.start_time.isoformat() if run.start_time else None,
        "end_time": run.end_time.isoformat() if run.end_time else None,
        "node_executions": [
            {
                "node_id": e.node_id,
                "status": e.status.value,
                "latency_ms": e.latency_ms,
                "attempts": e.attempts,
                "error": e.error,
            }
            for e in run.node_executions
        ],
    }


# ------------------------------------------------------------------- app factory

def create_app(engine: Optional[WorkflowEngine] = None) -> FastAPI:
    """Create a FastAPI app bound to a workflow engine.

    Args:
        engine: an existing engine to expose. If None, a fresh one is created.
    """
    engine = engine or WorkflowEngine()
    app = FastAPI(title="AI Workflow Engine", version="0.1.0")
    app.state.engine = engine

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "flows": len(engine._flows)}

    # --- flows -------------------------------------------------------------
    @app.post("/flows", status_code=201)
    def register_flow(req: RegisterFlowRequest) -> dict:
        try:
            flow = engine.register_flow(req.spec, req.description)
        except Exception as exc:  # parse/validation errors -> 400
            raise HTTPException(status_code=400, detail=str(exc))
        return {"name": flow.name, "version": flow.version, "nodes": len(flow.nodes)}

    @app.get("/flows")
    def list_flows() -> list[dict]:
        return [
            {"name": f.name, "version": f.version, "nodes": len(f.nodes)}
            for f in engine._flows.values()
        ]

    @app.get("/flows/{name}")
    def get_flow(name: str) -> dict:
        try:
            flow = engine.get_flow(name)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return {
            "name": flow.name,
            "version": flow.version,
            "description": flow.description,
            "nodes": [{"id": n.id, "type": n.type.value, "dependencies": n.dependencies} for n in flow.nodes],
            "outputs": flow.outputs,
        }

    @app.get("/flows/{name}/diagram")
    def flow_diagram(name: str, fmt: str = "mermaid") -> dict:
        try:
            flow = engine.get_flow(name)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        diagram = to_dot(flow) if fmt == "dot" else to_mermaid(flow)
        return {"format": fmt, "diagram": diagram}

    @app.post("/flows/{name}/run")
    async def run_registered(name: str, req: RunRequest) -> dict:
        try:
            engine.get_flow(name)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        run = await engine.execute(name, req.inputs)
        return _serialize_run(run)

    # --- inline runs -------------------------------------------------------
    @app.post("/runs")
    async def run_inline(req: InlineRunRequest) -> dict:
        try:
            flow = engine.parser.parse_dict(req.spec)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        run = await engine.run_flow(flow_definition=flow, inputs=req.inputs)
        return _serialize_run(run)

    @app.get("/runs")
    def list_runs(flow_name: Optional[str] = None, limit: int = 100) -> list[dict]:
        return [_serialize_run(r) for r in engine.get_run_history(flow_name, limit)]

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        run = engine.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        return _serialize_run(run)

    # --- human-in-the-loop reviews ----------------------------------------
    @app.get("/reviews")
    def list_reviews() -> list[dict]:
        return [r.to_dict() for r in engine.review_store.list_pending()]

    @app.get("/reviews/{review_id}")
    def get_review(review_id: str) -> dict:
        review = engine.review_store.get(review_id)
        if review is None:
            raise HTTPException(status_code=404, detail=f"Review not found: {review_id}")
        return review.to_dict()

    @app.post("/reviews/{review_id}/approve")
    def approve_review(review_id: str, decision: ReviewDecision) -> dict:
        return _resolve_review(engine, review_id, "approve", decision)

    @app.post("/reviews/{review_id}/reject")
    def reject_review(review_id: str, decision: ReviewDecision) -> dict:
        return _resolve_review(engine, review_id, "reject", decision)

    return app


def _resolve_review(engine, review_id, action, decision: ReviewDecision) -> dict:
    method = getattr(engine.review_store, action)
    try:
        review = method(
            review_id, reviewer=decision.reviewer, comment=decision.comment, data=decision.data
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Review not found: {review_id}")
    except ValueError as exc:  # already resolved
        raise HTTPException(status_code=409, detail=str(exc))
    return review.to_dict()


# Module-level app for `uvicorn aiworkflow.api:app`.
app = create_app()
