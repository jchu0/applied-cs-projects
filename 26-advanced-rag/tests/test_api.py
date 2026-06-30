"""Tests for FastAPI REST API endpoints."""

from fastapi.testclient import TestClient

from advancedrag.api.main import create_app
from advancedrag.schemas import Document


def _make_client_with_docs() -> TestClient:
    """Create a TestClient with pre-indexed documents."""
    app = create_app()
    pipeline = app.state.pipeline

    docs = [
        Document(id="doc-1", content="Python is a programming language used for web development and data science.", metadata={"title": "Python Overview"}),
        Document(id="doc-2", content="Machine learning models learn from data to make predictions.", metadata={"title": "ML Basics"}),
        Document(id="doc-3", content="Neural networks are inspired by biological neurons in the brain.", metadata={"title": "Neural Networks"}),
    ]
    pipeline.add_documents(docs)

    return TestClient(app)


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_returns_200(self):
        app = create_app()
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_includes_status_and_version(self):
        app = create_app()
        client = TestClient(app)
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"

    def test_documents_indexed_count(self):
        client = _make_client_with_docs()
        data = client.get("/health").json()
        assert data["documents_indexed"] == 3


# ---------------------------------------------------------------------------
# Query endpoint
# ---------------------------------------------------------------------------

class TestQueryEndpoint:
    """Tests for POST /v1/query."""

    def test_valid_query(self):
        client = _make_client_with_docs()
        resp = client.post("/v1/query", json={"query": "What is Python?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "confidence" in data
        assert "latency_ms" in data

    def test_query_with_citations(self):
        client = _make_client_with_docs()
        data = client.post("/v1/query", json={
            "query": "What is machine learning?",
            "include_citations": True,
        }).json()
        assert isinstance(data["citations"], list)

    def test_query_without_rewrite(self):
        client = _make_client_with_docs()
        resp = client.post("/v1/query", json={
            "query": "neural networks",
            "rewrite_query": False,
        })
        assert resp.status_code == 200

    def test_empty_query_rejected(self):
        client = _make_client_with_docs()
        resp = client.post("/v1/query", json={"query": ""})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Index / Delete endpoints
# ---------------------------------------------------------------------------

class TestIndexEndpoints:
    """Tests for POST /v1/index and DELETE /v1/documents/{id}."""

    def test_index_documents(self):
        app = create_app()
        client = TestClient(app)
        resp = client.post("/v1/index", json={
            "documents": [
                {"content": "First document", "metadata": {"title": "Doc 1"}},
                {"content": "Second document", "metadata": {"title": "Doc 2"}},
            ]
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["indexed"] == 2
        assert data["errors"] == []

    def test_index_empty_list(self):
        app = create_app()
        client = TestClient(app)
        resp = client.post("/v1/index", json={"documents": []})
        assert resp.status_code == 200
        assert resp.json()["indexed"] == 0

    def test_delete_document(self):
        client = _make_client_with_docs()
        resp = client.delete("/v1/documents/doc-1")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"


# ---------------------------------------------------------------------------
# Search endpoint
# ---------------------------------------------------------------------------

class TestSearchEndpoint:
    """Tests for POST /v1/search."""

    def test_search_returns_results(self):
        client = _make_client_with_docs()
        resp = client.post("/v1/search", json={"query": "programming"})
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert "latency_ms" in data

    def test_search_with_top_k(self):
        client = _make_client_with_docs()
        resp = client.post("/v1/search", json={"query": "data", "top_k": 2})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Stats endpoint
# ---------------------------------------------------------------------------

class TestStatsEndpoint:
    """Tests for GET /v1/stats."""

    def test_stats_returns_count(self):
        client = _make_client_with_docs()
        resp = client.get("/v1/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["documents_indexed"] == 3
        assert "config" in data
