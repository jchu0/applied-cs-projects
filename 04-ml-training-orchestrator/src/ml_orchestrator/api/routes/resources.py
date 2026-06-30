"""Resource management API endpoints."""

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel, Field

from ml_orchestrator.core.models import ResourceRequest, WorkerInfo, WorkerStatus


router = APIRouter()


class RegisterWorkerRequest(BaseModel):
    """Request to register a worker."""

    hostname: str
    ip_address: str
    port: int = 8000
    cpus: int = Field(ge=1)
    memory_gb: float = Field(ge=0.5)
    gpus: int = Field(default=0, ge=0)
    gpu_type: Optional[str] = None
    gpu_memory_gb: Optional[float] = None
    labels: dict[str, str] = Field(default_factory=dict)


class WorkerResponse(BaseModel):
    """Worker response model."""

    id: str
    hostname: str
    ip_address: str
    port: int
    status: str
    cpus: int
    memory_gb: float
    gpus: int
    allocated_cpus: int
    allocated_memory_gb: float
    allocated_gpus: int
    current_jobs: list[str]
    labels: dict[str, str]

    @classmethod
    def from_worker(cls, worker: WorkerInfo) -> "WorkerResponse":
        return cls(
            id=worker.id,
            hostname=worker.hostname,
            ip_address=worker.ip_address,
            port=worker.port,
            status=worker.status.value,
            cpus=worker.resources.cpus,
            memory_gb=worker.resources.memory_gb,
            gpus=worker.resources.gpus,
            allocated_cpus=worker.allocated_resources.cpus,
            allocated_memory_gb=worker.allocated_resources.memory_gb,
            allocated_gpus=worker.allocated_resources.gpus,
            current_jobs=worker.current_jobs,
            labels=worker.labels,
        )


@router.post("/workers", response_model=WorkerResponse)
async def register_worker(
    request: Request,
    body: RegisterWorkerRequest,
) -> WorkerResponse:
    """Register a new worker node."""
    state = request.app.state.orchestrator

    resources = ResourceRequest(
        cpus=body.cpus,
        memory_gb=body.memory_gb,
        gpus=body.gpus,
        gpu_type=body.gpu_type,
        gpu_memory_gb=body.gpu_memory_gb,
    )

    worker = WorkerInfo(
        hostname=body.hostname,
        ip_address=body.ip_address,
        port=body.port,
        resources=resources,
        status=WorkerStatus.READY,
        labels=body.labels,
    )

    await state.scheduler.register_worker(worker)
    await state.allocator.register_worker(worker)

    return WorkerResponse.from_worker(worker)


@router.get("/workers", response_model=list[WorkerResponse])
async def list_workers(
    request: Request,
    status: Optional[str] = Query(None),
    healthy_only: bool = Query(False),
) -> list[WorkerResponse]:
    """List all workers."""
    state = request.app.state.orchestrator

    if healthy_only:
        workers = await state.scheduler.get_healthy_workers()
    else:
        workers = await state.scheduler.list_workers()

    if status:
        try:
            worker_status = WorkerStatus(status)
            workers = [w for w in workers if w.status == worker_status]
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

    return [WorkerResponse.from_worker(w) for w in workers]


@router.get("/workers/{worker_id}", response_model=WorkerResponse)
async def get_worker(
    request: Request,
    worker_id: str,
) -> WorkerResponse:
    """Get worker by ID."""
    state = request.app.state.orchestrator

    worker = await state.scheduler.get_worker(worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail=f"Worker not found: {worker_id}")

    return WorkerResponse.from_worker(worker)


@router.delete("/workers/{worker_id}")
async def unregister_worker(
    request: Request,
    worker_id: str,
) -> dict[str, bool]:
    """Unregister a worker."""
    state = request.app.state.orchestrator

    await state.scheduler.unregister_worker(worker_id)
    await state.allocator.unregister_worker(worker_id)

    return {"success": True}


@router.post("/workers/{worker_id}/heartbeat")
async def worker_heartbeat(
    request: Request,
    worker_id: str,
) -> dict[str, bool]:
    """Update worker heartbeat."""
    state = request.app.state.orchestrator

    success = await state.scheduler.worker_heartbeat(worker_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Worker not found: {worker_id}")

    return {"success": True}


@router.get("/utilization")
async def get_utilization(
    request: Request,
) -> dict[str, Any]:
    """Get current resource utilization."""
    state = request.app.state.orchestrator

    utilization = await state.scheduler.get_resource_utilization()
    allocator_stats = await state.allocator.get_stats()

    return {
        "utilization": utilization,
        "resources": allocator_stats.get("resources", {}),
        "fragmentation_score": await state.allocator.get_fragmentation_score(),
    }


@router.get("/gpus")
async def get_gpu_status(
    request: Request,
) -> dict[str, Any]:
    """Get GPU status."""
    state = request.app.state.orchestrator

    return await state.gpu_manager.get_stats()


@router.get("/gpus/available")
async def get_available_gpus(
    request: Request,
    gpu_type: Optional[str] = Query(None),
    min_memory_gb: Optional[float] = Query(None),
) -> dict[str, Any]:
    """Get available GPUs."""
    state = request.app.state.orchestrator

    gpus = await state.gpu_manager.get_available_gpus(
        gpu_type=gpu_type,
        min_memory_gb=min_memory_gb,
    )

    return {
        "count": len(gpus),
        "gpus": [g.to_dict() for g in gpus],
    }


@router.get("/capacity")
async def get_capacity(
    request: Request,
) -> dict[str, Any]:
    """Get total and available capacity."""
    state = request.app.state.orchestrator

    total = await state.allocator.get_total_resources()
    available = await state.allocator.get_available_resources()

    return {
        "total": total.model_dump(),
        "available": available.model_dump(),
    }
