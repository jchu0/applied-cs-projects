"""Tests for FastAPI REST API layer."""

import pytest
import asyncio

from fastapi.testclient import TestClient

from modelrouter import create_app, create_router, ModelRouter


def _make_app_with_worker() -> TestClient:
    """Create a TestClient with a router that has one worker + tenant."""
    router = create_router()
    asyncio.run(_register_worker_and_tenant(router))
    app = create_app(router)
    return TestClient(app)


async def _register_worker_and_tenant(router: ModelRouter):
    await router.register_worker(
        worker_id="worker-1",
        host="localhost",
        port=8080,
        models=["gpt-4", "gpt-3.5-turbo"],
        token_budget=100000,
    )
    router.register_tenant(
        tenant_id="tenant-1",
        name="Test Tenant",
        api_key="test-key",
    )


# ---------------------------------------------------------------------------
# TestHealthEndpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_returns_200(self):
        app = create_app()
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_includes_status_field(self):
        app = create_app()
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# TestInferenceEndpoint
# ---------------------------------------------------------------------------

class TestInferenceEndpoint:
    """Tests for POST /v1/inference."""

    def test_submit_valid_request(self):
        client = _make_app_with_worker()
        resp = client.post("/v1/inference", json={
            "model": "gpt-4",
            "prompt": "Hello",
            "max_tokens": 50,
            "tenant_id": "tenant-1",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "request_id" in data
        assert data["model"] == "gpt-4"
        assert data["worker_id"] == "worker-1"

    def test_missing_fields_returns_422(self):
        client = _make_app_with_worker()
        resp = client.post("/v1/inference", json={"prompt": "Hello"})
        assert resp.status_code == 422

    def test_congestion_throttled_returns_503(self):
        """When congestion predictor fires, LOW-priority returns 503."""
        from unittest.mock import patch
        router = create_router()
        asyncio.run(_register_worker_and_tenant(router))
        app = create_app(router)
        client = TestClient(app)

        with patch.object(router.congestion_predictor, "should_throttle", return_value=True):
            resp = client.post("/v1/inference", json={
                "model": "gpt-4",
                "prompt": "Test",
                "max_tokens": 50,
                "tenant_id": "tenant-1",
                "priority": "LOW",
            })
            assert resp.status_code == 503

    def test_response_has_correct_fields(self):
        client = _make_app_with_worker()
        resp = client.post("/v1/inference", json={
            "model": "gpt-4",
            "prompt": "Hi",
            "max_tokens": 10,
            "tenant_id": "tenant-1",
        })
        data = resp.json()
        for field in ("request_id", "text", "tokens_used", "latency_ms", "worker_id", "model"):
            assert field in data


# ---------------------------------------------------------------------------
# TestWorkerEndpoints
# ---------------------------------------------------------------------------

class TestWorkerEndpoints:
    """Tests for worker registration and listing."""

    def test_register_worker(self):
        app = create_app()
        client = TestClient(app)
        resp = client.post("/workers/register", json={
            "worker_id": "w-1",
            "host": "10.0.0.1",
            "port": 9090,
            "models": ["gpt-4"],
        })
        assert resp.status_code == 200
        assert resp.json()["worker_id"] == "w-1"

    def test_list_workers(self):
        client = _make_app_with_worker()
        resp = client.get("/workers")
        assert resp.status_code == 200
        workers = resp.json()["workers"]
        assert len(workers) == 1
        assert workers[0]["worker_id"] == "worker-1"

    def test_list_workers_empty(self):
        app = create_app()
        client = TestClient(app)
        resp = client.get("/workers")
        assert resp.status_code == 200
        assert resp.json()["workers"] == []


# ---------------------------------------------------------------------------
# TestCapacityEndpoints
# ---------------------------------------------------------------------------

class TestCapacityEndpoints:
    """Tests for GET /capacity and GET /capacity/{model}."""

    def test_get_all_capacity(self):
        client = _make_app_with_worker()
        resp = client.get("/capacity")
        assert resp.status_code == 200
        data = resp.json()
        assert "gpt-4" in data

    def test_get_model_capacity(self):
        client = _make_app_with_worker()
        resp = client.get("/capacity/gpt-4")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_workers"] == 1
        assert data["healthy_workers"] == 1

    def test_unknown_model_returns_zero(self):
        client = _make_app_with_worker()
        resp = client.get("/capacity/nonexistent")
        assert resp.status_code == 200
        assert resp.json()["total_workers"] == 0


# ---------------------------------------------------------------------------
# TestQueueAndMetrics
# ---------------------------------------------------------------------------

class TestQueueAndMetrics:
    """Tests for GET /queue/stats and GET /metrics."""

    def test_get_queue_stats(self):
        client = _make_app_with_worker()
        resp = client.get("/queue/stats")
        assert resp.status_code == 200
        assert isinstance(resp.json(), dict)

    def test_get_metrics_empty(self):
        app = create_app()
        client = TestClient(app)
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_metrics_after_requests(self):
        client = _make_app_with_worker()
        # Fire a request so a metric is recorded
        client.post("/v1/inference", json={
            "model": "gpt-4",
            "prompt": "Hello",
            "max_tokens": 10,
            "tenant_id": "tenant-1",
        })
        resp = client.get("/metrics")
        assert resp.status_code == 200
        metrics = resp.json()
        assert len(metrics) >= 1
        assert "request_id" in metrics[0]
