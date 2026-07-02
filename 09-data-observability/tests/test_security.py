"""Tests for the API hardening baseline (auth / rate limiting / timeout)."""

import pytest

# Try to import fastapi test client, skip if not available
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient
from observability.api import app, _tables, _anomalies, _alerts, _lineage_graph


@pytest.fixture(autouse=True)
def clear_state():
    """Clear global state before each test."""
    _tables.clear()
    _anomalies.clear()
    _alerts.clear()
    _lineage_graph._nodes.clear()
    _lineage_graph._edges.clear()
    app.state.rate_limiter.reset()
    yield


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestApiKeyAuth:
    """Tests for opt-in API-key authentication."""

    def test_401_when_auth_enabled_and_key_missing(self, client, monkeypatch):
        """A protected route returns 401 when a key is required but absent."""
        monkeypatch.setenv("API_KEYS", "secret-key")
        response = client.get("/tables")
        assert response.status_code == 401
        assert response.headers.get("WWW-Authenticate") == "Bearer"

    def test_401_when_auth_enabled_and_key_bad(self, client, monkeypatch):
        """A protected route returns 401 for an invalid key."""
        monkeypatch.setenv("API_KEYS", "secret-key")
        response = client.get("/tables", headers={"X-API-Key": "wrong"})
        assert response.status_code == 401

    def test_200_with_valid_bearer_key(self, client, monkeypatch):
        """A protected route succeeds with a valid Bearer token."""
        monkeypatch.setenv("API_KEYS", "secret-key,other-key")
        response = client.get(
            "/tables", headers={"Authorization": "Bearer secret-key"}
        )
        assert response.status_code == 200

    def test_200_with_valid_x_api_key(self, client, monkeypatch):
        """A protected route succeeds with a valid X-API-Key header."""
        monkeypatch.setenv("API_KEYS", "secret-key")
        response = client.get("/tables", headers={"X-API-Key": "secret-key"})
        assert response.status_code == 200

    def test_open_paths_reachable_without_key(self, client, monkeypatch):
        """Health and OpenAPI stay open even when auth is enabled."""
        monkeypatch.setenv("API_KEYS", "secret-key")
        assert client.get("/health").status_code == 200
        assert client.get("/openapi.json").status_code == 200


class TestRateLimiting:
    """Tests for in-process rate limiting."""

    def test_429_when_over_limit(self, client, monkeypatch):
        """Requests beyond the per-minute limit return 429 with Retry-After."""
        monkeypatch.delenv("API_KEYS", raising=False)
        monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "3")

        statuses = [client.get("/tables").status_code for _ in range(5)]
        assert statuses[:3] == [200, 200, 200]
        assert 429 in statuses[3:]

        limited = client.get("/tables")
        assert limited.status_code == 429
        assert "Retry-After" in limited.headers

    def test_health_exempt_from_rate_limit(self, client, monkeypatch):
        """Health endpoint is never rate limited."""
        monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1")
        for _ in range(5):
            assert client.get("/health").status_code == 200
