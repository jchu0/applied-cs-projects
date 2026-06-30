"""Health check endpoints."""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("")
async def health_check() -> dict[str, Any]:
    """Basic health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/ready")
async def readiness_check(request: Request) -> dict[str, Any]:
    """Readiness check including component status."""
    state = request.app.state.orchestrator

    scheduler_stats = await state.scheduler.get_stats()
    job_stats = await state.job_manager.get_stats()

    return {
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "scheduler": {
                "running": scheduler_stats.get("running", False),
                "queue_size": scheduler_stats.get("queue", {}).get("size", 0),
            },
            "job_manager": {
                "total_jobs": job_stats.get("total_jobs", 0),
            },
        },
    }


@router.get("/live")
async def liveness_check() -> dict[str, str]:
    """Liveness check."""
    return {"status": "alive"}


@router.get("/stats")
async def get_stats(request: Request) -> dict[str, Any]:
    """Get comprehensive system statistics."""
    state = request.app.state.orchestrator

    return {
        "scheduler": await state.scheduler.get_stats(),
        "job_manager": await state.job_manager.get_stats(),
        "allocator": await state.allocator.get_stats(),
        "gpu_manager": await state.gpu_manager.get_stats(),
        "checkpoint_manager": await state.checkpoint_manager.get_stats(),
        "experiment_tracker": await state.experiment_tracker.get_stats(),
        "timestamp": datetime.utcnow().isoformat(),
    }
