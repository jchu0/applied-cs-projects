"""FastAPI REST API for Advanced RAG system."""

import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..schemas import Document, generate_id
from ..pipeline import create_pipeline, RAGPipeline
from .security import install_hardening, require_api_key


# =============================================================================
# Pydantic Models
# =============================================================================

class QueryRequest(BaseModel):
    """Request for RAG query."""
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    filter: Optional[dict] = None
    rewrite_query: bool = True
    include_citations: bool = True


class CitationResponse(BaseModel):
    """Citation in response."""
    source_id: str
    title: str
    excerpt: str
    relevance: float


class QueryResponse(BaseModel):
    """Response from RAG query."""
    answer: str
    citations: list[CitationResponse]
    confidence: float
    latency_ms: float


class IndexRequest(BaseModel):
    """Request to index documents."""
    documents: list[dict]


class IndexResponse(BaseModel):
    """Response from indexing."""
    indexed: int
    errors: list[str]


class SearchRequest(BaseModel):
    """Request for search without generation."""
    query: str
    top_k: int = Field(default=10, ge=1, le=100)
    filter: Optional[dict] = None


class SearchResponse(BaseModel):
    """Response from search."""
    results: list[dict]
    latency_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    documents_indexed: int


# =============================================================================
# Application
# =============================================================================

def create_app(
    data_directory: str = "./data",
) -> FastAPI:
    """Create FastAPI application.

    Args:
        data_directory: Directory for data storage

    Returns:
        Configured FastAPI app
    """
    app = FastAPI(
        title="Advanced RAG API",
        description="Production-grade Retrieval-Augmented Generation with neural reranking",
        version="1.0.0",
        # Global API-key auth (opt-in via API_KEYS); open paths self-exempt.
        dependencies=[Depends(require_api_key)],
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Production hardening: auth warning, rate limiting, request timeouts.
    install_hardening(app)

    # Initialize pipeline
    pipeline = create_pipeline()

    # Store in app state
    app.state.pipeline = pipeline
    app.state.data_directory = Path(data_directory)
    app.state.data_directory.mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # Endpoints
    # ==========================================================================

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            documents_indexed=app.state.pipeline.retriever.count,
        )

    @app.post("/v1/query", response_model=QueryResponse)
    async def query(
        request: QueryRequest,
        background_tasks: BackgroundTasks,
    ):
        """Execute RAG query.

        This endpoint performs retrieval, reranking, and generation.
        """
        try:
            result = await app.state.pipeline.execute(
                query=request.query,
                top_k=request.top_k,
                filter_dict=request.filter,
                rewrite_query=request.rewrite_query,
            )

            citations = []
            if request.include_citations:
                for c in result.answer.citations:
                    citations.append(CitationResponse(
                        source_id=c.source_id,
                        title=c.source_title,
                        excerpt=c.quoted_text[:200],
                        relevance=c.relevance_score,
                    ))

            return QueryResponse(
                answer=result.answer.answer,
                citations=citations,
                confidence=result.answer.confidence,
                latency_ms=result.latency_ms,
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/search", response_model=SearchResponse)
    async def search(request: SearchRequest):
        """Search without generation.

        Returns retrieved and reranked documents without LLM generation.
        """
        start_time = time.time()

        try:
            # Just retrieval and reranking
            retrieval_results = app.state.pipeline.retriever.search(
                request.query,
                top_k=request.top_k * 2,
                filter_dict=request.filter,
            )

            reranked = await app.state.pipeline.reranker.rerank(
                request.query,
                retrieval_results,
                top_k=request.top_k,
            )

            results = [r.to_dict() for r in reranked]
            latency_ms = (time.time() - start_time) * 1000

            return SearchResponse(
                results=results,
                latency_ms=latency_ms,
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/index", response_model=IndexResponse)
    async def index_documents(request: IndexRequest):
        """Index documents into the RAG system."""
        indexed = 0
        errors = []

        for doc_data in request.documents:
            try:
                doc = Document(
                    id=doc_data.get("id") or generate_id(doc_data["content"]),
                    content=doc_data["content"],
                    metadata=doc_data.get("metadata", {}),
                )
                app.state.pipeline.add_documents([doc])
                indexed += 1

            except Exception as e:
                errors.append(f"Error indexing document: {str(e)}")

        return IndexResponse(indexed=indexed, errors=errors)

    @app.delete("/v1/documents/{doc_id}")
    async def delete_document(doc_id: str):
        """Delete a document by ID."""
        try:
            app.state.pipeline.delete_documents([doc_id])
            return {"status": "deleted", "id": doc_id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/v1/stats")
    async def get_stats():
        """Get system statistics."""
        return {
            "documents_indexed": app.state.pipeline.retriever.count,
            "config": app.state.pipeline.config,
        }

    return app


# Default app instance
app = create_app()


def main():
    """Run the API server."""
    import uvicorn
    uvicorn.run(
        "advancedrag.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
