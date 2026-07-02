"""Tests for API hardening: auth, rate limiting, timeouts."""

import pytest
from fastapi.testclient import TestClient

from advancedrag.api.main import create_app


# ---------------------------------------------------------------------------
# API-key authentication
# ---------------------------------------------------------------------------

class TestApiKeyAuth:
    """Tests for opt-in API-key authentication."""

    def test_401_when_auth_enabled_and_key_missing(self, monkeypatch):
        monkeypatch.setenv("API_KEYS", "secret-key")
        client = TestClient(create_app())
        resp = client.post("/v1/search", json={"query": "python"})
        assert resp.status_code == 401
        assert resp.headers.get("WWW-Authenticate") == "Bearer"

    def test_401_when_key_invalid(self, monkeypatch):
        monkeypatch.setenv("API_KEYS", "secret-key")
        client = TestClient(create_app())
        resp = client.post(
            "/v1/search",
            json={"query": "python"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401

    def test_200_with_valid_bearer_key(self, monkeypatch):
        monkeypatch.setenv("API_KEYS", "secret-key,other-key")
        client = TestClient(create_app())
        resp = client.post(
            "/v1/search",
            json={"query": "python"},
            headers={"Authorization": "Bearer secret-key"},
        )
        assert resp.status_code == 200

    def test_200_with_valid_x_api_key(self, monkeypatch):
        monkeypatch.setenv("API_KEYS", "secret-key")
        client = TestClient(create_app())
        resp = client.post(
            "/v1/search",
            json={"query": "python"},
            headers={"X-API-Key": "secret-key"},
        )
        assert resp.status_code == 200

    def test_health_and_openapi_open_without_key(self, monkeypatch):
        monkeypatch.setenv("API_KEYS", "secret-key")
        client = TestClient(create_app())
        assert client.get("/health").status_code == 200
        assert client.get("/openapi.json").status_code == 200


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class TestRateLimiting:
    """Tests for in-process sliding-window rate limiting."""

    def test_429_when_over_limit(self, monkeypatch):
        monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")
        client = TestClient(create_app())
        # First two allowed, third exceeds.
        assert client.post("/v1/search", json={"query": "a"}).status_code == 200
        assert client.post("/v1/search", json={"query": "b"}).status_code == 200
        resp = client.post("/v1/search", json={"query": "c"})
        assert resp.status_code == 429
        assert "Retry-After" in resp.headers

    def test_health_exempt_from_rate_limit(self, monkeypatch):
        monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1")
        client = TestClient(create_app())
        for _ in range(5):
            assert client.get("/health").status_code == 200


# ---------------------------------------------------------------------------
# Auth disabled by default (env unset)
# ---------------------------------------------------------------------------

class TestAuthDisabledByDefault:
    """With no env configured, endpoints stay open."""

    def test_no_key_required(self, monkeypatch):
        monkeypatch.delenv("API_KEYS", raising=False)
        client = TestClient(create_app())
        resp = client.post("/v1/search", json={"query": "python"})
        assert resp.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
