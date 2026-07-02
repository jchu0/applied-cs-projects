"""Tests for HTTP-layer hardening: API-key auth, rate limiting, timeouts."""

import asyncio

from fastapi.testclient import TestClient

from modelrouter import create_app, create_router, ModelRouter


def _make_app_with_worker() -> ModelRouter:
    """Create a router with one worker + tenant (auth is env-driven, not router-driven)."""
    router = create_router()
    asyncio.run(_register_worker_and_tenant(router))
    return router


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


_INFERENCE_BODY = {
    "model": "gpt-4",
    "prompt": "Hello",
    "max_tokens": 10,
    "tenant_id": "tenant-1",
}


# ---------------------------------------------------------------------------
# TestApiKeyAuth
# ---------------------------------------------------------------------------

class TestApiKeyAuth:
    """API-key auth is opt-in via the API_KEYS env var."""

    def test_missing_key_returns_401_when_enabled(self, monkeypatch):
        monkeypatch.setenv("API_KEYS", "secret-key")
        client = TestClient(create_app(_make_app_with_worker()))
        resp = client.post("/v1/inference", json=_INFERENCE_BODY)
        assert resp.status_code == 401
        assert resp.headers.get("www-authenticate") == "Bearer"

    def test_bad_key_returns_401_when_enabled(self, monkeypatch):
        monkeypatch.setenv("API_KEYS", "secret-key")
        client = TestClient(create_app(_make_app_with_worker()))
        resp = client.post(
            "/v1/inference",
            json=_INFERENCE_BODY,
            headers={"Authorization": "Bearer wrong"},
        )
        assert resp.status_code == 401

    def test_valid_bearer_key_returns_200(self, monkeypatch):
        monkeypatch.setenv("API_KEYS", "secret-key,other-key")
        client = TestClient(create_app(_make_app_with_worker()))
        resp = client.post(
            "/v1/inference",
            json=_INFERENCE_BODY,
            headers={"Authorization": "Bearer secret-key"},
        )
        assert resp.status_code == 200
        assert resp.json()["worker_id"] == "worker-1"

    def test_valid_x_api_key_header_returns_200(self, monkeypatch):
        monkeypatch.setenv("API_KEYS", "secret-key")
        client = TestClient(create_app(_make_app_with_worker()))
        resp = client.post(
            "/v1/inference",
            json=_INFERENCE_BODY,
            headers={"X-API-Key": "secret-key"},
        )
        assert resp.status_code == 200

    def test_auth_disabled_when_unset(self, monkeypatch):
        monkeypatch.delenv("API_KEYS", raising=False)
        client = TestClient(create_app(_make_app_with_worker()))
        resp = client.post("/v1/inference", json=_INFERENCE_BODY)
        assert resp.status_code == 200

    def test_health_and_openapi_open_without_key(self, monkeypatch):
        monkeypatch.setenv("API_KEYS", "secret-key")
        client = TestClient(create_app(_make_app_with_worker()))
        assert client.get("/health").status_code == 200
        assert client.get("/openapi.json").status_code == 200


# ---------------------------------------------------------------------------
# TestRateLimit
# ---------------------------------------------------------------------------

class TestRateLimit:
    """In-process sliding-window rate limiting via RATE_LIMIT_PER_MINUTE."""

    def test_over_limit_returns_429(self, monkeypatch):
        monkeypatch.delenv("API_KEYS", raising=False)
        monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")
        client = TestClient(create_app(_make_app_with_worker()))

        assert client.post("/v1/inference", json=_INFERENCE_BODY).status_code == 200
        assert client.post("/v1/inference", json=_INFERENCE_BODY).status_code == 200
        resp = client.post("/v1/inference", json=_INFERENCE_BODY)
        assert resp.status_code == 429
        assert "Retry-After" in resp.headers

    def test_health_exempt_from_rate_limit(self, monkeypatch):
        monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1")
        client = TestClient(create_app(_make_app_with_worker()))
        for _ in range(5):
            assert client.get("/health").status_code == 200
