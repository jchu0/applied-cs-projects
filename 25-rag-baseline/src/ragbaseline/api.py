"""FastAPI REST API for RAG system."""

import asyncio
import json
import time
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .schemas import Document, RAGConfig
from .parsers import DocumentIngestion
from .embeddings import get_embedding_model
from .vectorstore import get_vector_store
from .chunking import SentenceChunker
from .index import RAGIndex
from .pipeline import RAGPipeline, get_llm_provider
from .enterprise import TenantManager, RetrievalLogger, UsageTracker
from .retrieval import HybridRetriever, get_reranker, MetadataFilter
from .security import (
    RateLimitMiddleware,
    SlidingWindowRateLimiter,
    TimeoutMiddleware,
    auth_enabled,
    logger as security_logger,
    rate_limit_per_minute,
    request_timeout_seconds,
    require_api_key,
)


# =============================================================================
# Pydantic Models
# =============================================================================

class QueryRequest(BaseModel):
    """Request for RAG query."""
    question: str = Field(..., description="Question to answer")
    tenant_id: str = Field(default="default", description="Tenant ID")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results")
    filter: Optional[dict] = Field(default=None, description="Metadata filter")
    filter_string: Optional[str] = Field(default=None, description="Filter query string")
    rerank: bool = Field(default=False, description="Enable reranking")
    stream: bool = Field(default=False, description="Stream response")


class QueryResponse(BaseModel):
    """Response from RAG query."""
    answer: str
    sources: list[dict]
    query: str
    latency_ms: float


class IngestRequest(BaseModel):
    """Request for document ingestion."""
    tenant_id: str = Field(default="default", description="Tenant ID")
    file_path: str = Field(..., description="Path to file")
    metadata: Optional[dict] = Field(default=None, description="Additional metadata")


class IngestResponse(BaseModel):
    """Response from document ingestion."""
    status: str
    document_id: str
    chunks: int


class IngestTextRequest(BaseModel):
    """Request for text ingestion."""
    tenant_id: str = Field(default="default", description="Tenant ID")
    content: str = Field(..., description="Text content")
    metadata: Optional[dict] = Field(default=None, description="Document metadata")
    source: str = Field(default="", description="Source identifier")


class SearchRequest(BaseModel):
    """Request for search without generation."""
    query: str = Field(..., description="Search query")
    tenant_id: str = Field(default="default", description="Tenant ID")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results")
    filter: Optional[dict] = Field(default=None, description="Metadata filter")
    hybrid: bool = Field(default=False, description="Use hybrid search")
    alpha: float = Field(default=0.5, ge=0, le=1, description="Vector weight for hybrid")


class SearchResponse(BaseModel):
    """Response from search."""
    results: list[dict]
    latency_ms: float


class TenantInfo(BaseModel):
    """Tenant information."""
    tenant_id: str
    document_count: int
    created_at: Optional[str]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str


class AnalyticsResponse(BaseModel):
    """Analytics response."""
    total_queries: int
    tenants: int
    latency: Optional[dict]


# =============================================================================
# Application Setup
# =============================================================================

def create_app(
    base_directory: str = "./data",
    default_llm_provider: str = "mock",
    default_llm_model: str = None,
) -> FastAPI:
    """Create FastAPI application.

    Args:
        base_directory: Base directory for data storage
        default_llm_provider: Default LLM provider
        default_llm_model: Default LLM model

    Returns:
        Configured FastAPI app
    """
    app = FastAPI(
        title="RAG Baseline API",
        description="Production-ready Retrieval-Augmented Generation API",
        version="1.0.0",
        dependencies=[Depends(require_api_key)],
    )

    # Warn once at startup when auth is disabled (no API_KEYS configured).
    if not auth_enabled():
        security_logger.warning("API auth disabled (set API_KEYS to enable)")

    # Production hardening middleware (opt-in via env; see security.py).
    # Order matters: outermost added last. We add timeout first (inner) then
    # rate limiting (outer) so rejected requests never consume a handler slot.
    app.add_middleware(
        TimeoutMiddleware,
        timeout_seconds=request_timeout_seconds(),
    )
    app.add_middleware(
        RateLimitMiddleware,
        limiter=SlidingWindowRateLimiter(rate_limit_per_minute()),
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize services
    base_path = Path(base_directory)
    base_path.mkdir(parents=True, exist_ok=True)

    tenant_manager = TenantManager(
        base_directory=str(base_path / "tenants"),
    )

    retrieval_logger = RetrievalLogger(
        log_directory=str(base_path / "logs"),
    )

    usage_tracker = UsageTracker(
        storage_path=str(base_path / "usage"),
    )

    document_ingestion = DocumentIngestion()

    # Store in app state
    app.state.tenant_manager = tenant_manager
    app.state.retrieval_logger = retrieval_logger
    app.state.usage_tracker = usage_tracker
    app.state.document_ingestion = document_ingestion
    app.state.default_llm_provider = default_llm_provider
    app.state.default_llm_model = default_llm_model

    # =============================================================================
    # API Endpoints
    # =============================================================================

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(status="healthy", version="1.0.0")

    @app.post("/query", response_model=QueryResponse)
    async def query(
        request: QueryRequest,
        background_tasks: BackgroundTasks,
    ):
        """Execute RAG query.

        This endpoint retrieves relevant documents and generates an answer
        using an LLM.
        """
        start_time = time.time()

        # Get tenant index
        index = app.state.tenant_manager.get_tenant_index(request.tenant_id)

        # Get LLM provider
        llm_provider = get_llm_provider(
            provider_type=app.state.default_llm_provider,
            model=app.state.default_llm_model,
        )

        # Build config
        config = RAGConfig(
            top_k=request.top_k,
            rerank=request.rerank,
        )

        # Build pipeline
        pipeline = RAGPipeline(index, llm_provider, config)

        # Parse filter
        filter_dict = request.filter
        if request.filter_string:
            filter_dict = MetadataFilter.parse(request.filter_string)

        # Execute query
        result = await pipeline.query(request.question, filter=filter_dict)

        latency_ms = (time.time() - start_time) * 1000

        # Log query in background
        background_tasks.add_task(
            app.state.retrieval_logger.log_query,
            query=request.question,
            results=result.sources,
            response=result.answer,
            tenant_id=request.tenant_id,
            latency_ms=latency_ms,
        )

        # Track usage
        background_tasks.add_task(
            app.state.usage_tracker.track_query,
            tenant_id=request.tenant_id,
        )

        return QueryResponse(
            answer=result.answer,
            sources=[s.to_dict() for s in result.sources],
            query=result.query,
            latency_ms=latency_ms,
        )

    @app.post("/query/stream")
    async def query_stream(request: QueryRequest):
        """Execute RAG query with streaming response.

        Returns a Server-Sent Events stream of the response.
        """
        # Get tenant index
        index = app.state.tenant_manager.get_tenant_index(request.tenant_id)

        # Get LLM provider
        llm_provider = get_llm_provider(
            provider_type=app.state.default_llm_provider,
            model=app.state.default_llm_model,
        )

        # Build config
        config = RAGConfig(
            top_k=request.top_k,
            rerank=request.rerank,
        )

        # Build pipeline
        pipeline = RAGPipeline(index, llm_provider, config)

        # Parse filter
        filter_dict = request.filter
        if request.filter_string:
            filter_dict = MetadataFilter.parse(request.filter_string)

        async def generate():
            async for chunk in pipeline.query_stream(
                request.question,
                filter=filter_dict,
            ):
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
        )

    @app.post("/search", response_model=SearchResponse)
    async def search(request: SearchRequest):
        """Search for relevant documents without generation.

        This endpoint performs retrieval only, without LLM generation.
        """
        start_time = time.time()

        # Get tenant index
        index = app.state.tenant_manager.get_tenant_index(request.tenant_id)

        # Perform search
        results = index.search(
            query=request.query,
            k=request.top_k,
            filter=request.filter,
        )

        latency_ms = (time.time() - start_time) * 1000

        return SearchResponse(
            results=[r.to_dict() for r in results],
            latency_ms=latency_ms,
        )

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest_document(
        request: IngestRequest,
        background_tasks: BackgroundTasks,
    ):
        """Ingest a document from file.

        Supports PDF, HTML, and Markdown files.
        """
        file_path = Path(request.file_path)

        if not file_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {request.file_path}"
            )

        try:
            # Parse document
            document = app.state.document_ingestion.ingest(file_path)

            # Add custom metadata
            if request.metadata:
                document.metadata.update(request.metadata)

            # Get tenant index
            index = app.state.tenant_manager.get_tenant_index(request.tenant_id)

            # Count chunks before indexing
            chunks = index.chunker.chunk(document)

            # Index document
            index.index_document(document)

            # Track usage
            background_tasks.add_task(
                app.state.usage_tracker.track_document,
                tenant_id=request.tenant_id,
            )

            return IngestResponse(
                status="success",
                document_id=document.id,
                chunks=len(chunks),
            )

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Ingestion failed: {str(e)}"
            )

    @app.post("/ingest/text", response_model=IngestResponse)
    async def ingest_text(
        request: IngestTextRequest,
        background_tasks: BackgroundTasks,
    ):
        """Ingest text content directly.

        Use this endpoint to ingest raw text without a file.
        """
        from .schemas import generate_id

        # Create document
        document = Document(
            id=generate_id(request.content),
            content=request.content,
            metadata=request.metadata or {},
            source=request.source,
        )

        # Get tenant index
        index = app.state.tenant_manager.get_tenant_index(request.tenant_id)

        # Count chunks
        chunks = index.chunker.chunk(document)

        # Index document
        index.index_document(document)

        # Track usage
        background_tasks.add_task(
            app.state.usage_tracker.track_document,
            tenant_id=request.tenant_id,
        )

        return IngestResponse(
            status="success",
            document_id=document.id,
            chunks=len(chunks),
        )

    @app.get("/tenants", response_model=list[str])
    async def list_tenants():
        """List all tenants."""
        return app.state.tenant_manager.list_tenants()

    @app.get("/tenants/{tenant_id}", response_model=TenantInfo)
    async def get_tenant(tenant_id: str):
        """Get tenant information."""
        info = app.state.tenant_manager.get_tenant_info(tenant_id)

        return TenantInfo(
            tenant_id=tenant_id,
            document_count=info.get("document_count", 0),
            created_at=info.get("created_at"),
        )

    @app.delete("/tenants/{tenant_id}")
    async def delete_tenant(tenant_id: str):
        """Delete tenant and all data.

        Warning: This permanently deletes all tenant data!
        """
        app.state.tenant_manager.delete_tenant(tenant_id)
        return {"status": "deleted", "tenant_id": tenant_id}

    @app.get("/analytics", response_model=AnalyticsResponse)
    async def get_analytics():
        """Get retrieval analytics."""
        analytics = app.state.retrieval_logger.get_analytics()

        return AnalyticsResponse(
            total_queries=analytics.get("total_queries", 0),
            tenants=analytics.get("tenants", 0),
            latency=analytics.get("latency"),
        )

    @app.get("/usage/{tenant_id}")
    async def get_usage(tenant_id: str):
        """Get usage metrics for tenant."""
        return app.state.usage_tracker.get_usage(tenant_id)

    return app


# Create default app instance
app = create_app()


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Run the API server."""
    import uvicorn

    uvicorn.run(
        "ragbaseline.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
