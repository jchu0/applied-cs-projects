"""Tests for API-key auth, rate limiting, and request timeouts.

These cover the production hardening layered in front of the existing tenant
isolation. Env vars are set per-test via monkeypatch so the default (keyless,
no-limit) behaviour exercised by the rest of the suite is unaffected.
"""

import importlib

import pytest
from fastapi.testclient import TestClient


def _build_client(tmp_path):
    """Build a fresh app + TestClient reading the current environment.

    The app factory reads env vars at construction time, so callers must set
    API_KEYS / RATE_LIMIT_PER_MINUTE / REQUEST_TIMEOUT_SECONDS before calling.
    """
    from ragbaseline import api as api_module

    # Reload so module-level state (and the security env reads) are fresh.
    importlib.reload(api_module)
    app = api_module.create_app(base_directory=str(tmp_path / "data"))
    return TestClient(app)


# ============================================================================
# API-key auth
# ============================================================================

class TestApiKeyAuth:
    """Auth is opt-in via API_KEYS and applied to non-open endpoints."""

    def test_401_when_auth_enabled_and_key_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("API_KEYS", "secret-key-1,secret-key-2")
        client = _build_client(tmp_path)

        resp = client.get("/tenants")
        assert resp.status_code == 401
        assert resp.headers.get("www-authenticate") == "Bearer"

    def test_401_when_auth_enabled_and_key_bad(self, tmp_path, monkeypatch):
        monkeypatch.setenv("API_KEYS", "secret-key-1")
        client = _build_client(tmp_path)

        resp = client.get("/tenants", headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 401

    def test_200_with_valid_bearer_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("API_KEYS", "secret-key-1")
        client = _build_client(tmp_path)

        resp = client.get(
            "/tenants", headers={"Authorization": "Bearer secret-key-1"}
        )
        assert resp.status_code == 200

    def test_200_with_valid_x_api_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("API_KEYS", "secret-key-1")
        client = _build_client(tmp_path)

        resp = client.get("/tenants", headers={"X-API-Key": "secret-key-1"})
        assert resp.status_code == 200

    def test_health_and_openapi_open_with_no_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("API_KEYS", "secret-key-1")
        client = _build_client(tmp_path)

        assert client.get("/health").status_code == 200
        assert client.get("/openapi.json").status_code == 200

    def test_auth_disabled_by_default(self, tmp_path, monkeypatch):
        monkeypatch.delenv("API_KEYS", raising=False)
        client = _build_client(tmp_path)

        # No key required when API_KEYS is unset.
        assert client.get("/tenants").status_code == 200


# ============================================================================
# Rate limiting
# ============================================================================

class TestRateLimiting:
    """In-process sliding-window limiter returns 429 with Retry-After."""

    def test_429_when_over_limit(self, tmp_path, monkeypatch):
        monkeypatch.delenv("API_KEYS", raising=False)
        monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")
        client = _build_client(tmp_path)

        assert client.get("/tenants").status_code == 200
        assert client.get("/tenants").status_code == 200
        resp = client.get("/tenants")
        assert resp.status_code == 429
        assert "Retry-After" in resp.headers

    def test_health_exempt_from_rate_limit(self, tmp_path, monkeypatch):
        monkeypatch.delenv("API_KEYS", raising=False)
        monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1")
        client = _build_client(tmp_path)

        # Exceed the limit on health many times; it should never be limited.
        for _ in range(5):
            assert client.get("/health").status_code == 200


# ============================================================================
# Request timeout
# ============================================================================

class TestRequestTimeout:
    """Timeout middleware returns 504 and exempts the streaming endpoint."""

    def test_504_on_timeout(self, tmp_path, monkeypatch):
        monkeypatch.delenv("API_KEYS", raising=False)
        monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "0.05")

        from ragbaseline import api as api_module

        importlib.reload(api_module)
        app = api_module.create_app(base_directory=str(tmp_path / "data"))

        # Register a slow handler that should trip the timeout middleware.
        @app.get("/slow-test")
        async def slow():  # pragma: no cover - registered for the test only
            import asyncio

            await asyncio.sleep(1.0)
            return {"ok": True}

        client = TestClient(app)
        resp = client.get("/slow-test")
        assert resp.status_code == 504
