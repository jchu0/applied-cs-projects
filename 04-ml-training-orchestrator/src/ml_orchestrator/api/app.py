"""FastAPI application factory."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog

from ml_orchestrator.core.job_manager import JobManager
from ml_orchestrator.scheduling.scheduler import Scheduler
from ml_orchestrator.resources.allocator import ResourceAllocator
from ml_orchestrator.resources.gpu_manager import GPUManager
from ml_orchestrator.checkpoint.manager import CheckpointManager
from ml_orchestrator.experiment.tracker import ExperimentTracker
from ml_orchestrator.api.routes import jobs, resources, experiments, health
from ml_orchestrator.api.security import configure_security, require_api_key


logger = structlog.get_logger(__name__)


class AppState:
    """Application state container."""

    def __init__(self):
        self.job_manager: JobManager = JobManager()
        self.scheduler: Scheduler = Scheduler(self.job_manager)
        self.allocator: ResourceAllocator = ResourceAllocator()
        self.gpu_manager: GPUManager = GPUManager()
        self.checkpoint_manager: CheckpointManager = CheckpointManager()
        self.experiment_tracker: ExperimentTracker = ExperimentTracker()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    logger.info("Starting ML Training Orchestrator API")

    # Initialize state
    app.state.orchestrator = AppState()

    # Start background services
    await app.state.orchestrator.scheduler.start()

    yield

    # Shutdown
    logger.info("Shutting down ML Training Orchestrator API")
    await app.state.orchestrator.scheduler.stop()


def create_app(
    title: str = "ML Training Orchestrator",
    version: str = "0.1.0",
    debug: bool = False,
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        title: API title
        version: API version
        debug: Enable debug mode

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title=title,
        version=version,
        description="Distributed ML Training Orchestration Platform",
        lifespan=lifespan,
        debug=debug,
    )

    # Permissive CORS -- dev-only default. Lock down allow_origins/methods before
    # exposing this beyond a trusted network.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Hardening baseline: rate limiting + request timeout middleware, plus the
    # auth-status startup warning. All opt-in via env (see api/security.py).
    configure_security(app)

    # Opt-in API-key auth applied to the data-plane routers only; health and the
    # docs/openapi endpoints stay open.
    protected = [Depends(require_api_key)]

    # Include routers
    app.include_router(health.router, prefix="/health", tags=["Health"])
    app.include_router(
        jobs.router, prefix="/api/v1/jobs", tags=["Jobs"], dependencies=protected
    )
    app.include_router(
        resources.router,
        prefix="/api/v1/resources",
        tags=["Resources"],
        dependencies=protected,
    )
    app.include_router(
        experiments.router,
        prefix="/api/v1/experiments",
        tags=["Experiments"],
        dependencies=protected,
    )

    return app


# Default app instance
app = create_app()
