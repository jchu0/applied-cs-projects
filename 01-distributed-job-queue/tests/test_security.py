"""Tests for the production hardening baseline (auth, rate limits, timeouts)."""

import pytest

# Skip if required dependencies are not available
pytest.importorskip("fastapi")
pytest.importorskip("redis")
pytest.importorskip("croniter")

from fastapi.testclient import TestClient

from jobqueue.api import create_app


def _client(monkeypatch, **env):
    """Build a TestClient (as a context manager) with the given env applied.

    Env is set before create_app() so the hardening config is read correctly.
    The caller must use the returned client as a context manager so the app
    lifespan runs and the broker is initialized.
    """
    for key in ("API_KEYS", "RATE_LIMIT_PER_MINUTE", "REQUEST_TIMEOUT_SECONDS"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return TestClient(create_app())


def test_401_when_auth_enabled_and_key_missing_or_bad(monkeypatch):
    """Protected routes return 401 when auth is on and no/bad key is given."""
    with _client(monkeypatch, API_KEYS="secret-key") as client:
        # No key at all.
        resp = client.get("/queues")
        assert resp.status_code == 401
        assert resp.headers.get("WWW-Authenticate") == "Bearer"

        # Wrong key.
        resp = client.get("/queues", headers={"X-API-Key": "wrong"})
        assert resp.status_code == 401


def test_200_with_valid_key(monkeypatch):
    """A valid key (Bearer or X-API-Key) reaches the protected route."""
    with _client(monkeypatch, API_KEYS="secret-key,other-key") as client:
        resp = client.get("/queues", headers={"Authorization": "Bearer secret-key"})
        assert resp.status_code == 200

        resp = client.get("/queues", headers={"X-API-Key": "other-key"})
        assert resp.status_code == 200


def test_internal_routes_require_auth(monkeypatch):
    """Worker-facing /internal/* routes are protected too."""
    with _client(monkeypatch, API_KEYS="secret-key") as client:
        resp = client.post("/internal/heartbeat?worker_id=w1")
        assert resp.status_code == 401

        resp = client.post(
            "/internal/heartbeat?worker_id=w1",
            headers={"X-API-Key": "secret-key"},
        )
        assert resp.status_code == 200


def test_429_when_over_limit(monkeypatch):
    """Exceeding a low per-minute rate limit yields 429 with Retry-After."""
    with _client(monkeypatch, RATE_LIMIT_PER_MINUTE="3") as client:
        statuses = [client.get("/queues").status_code for _ in range(5)]
        assert statuses[:3] == [200, 200, 200]
        assert 429 in statuses[3:]

        resp = client.get("/queues")
        assert resp.status_code == 429
        assert int(resp.headers["Retry-After"]) >= 1


def test_health_and_openapi_open_without_key(monkeypatch):
    """Health and /openapi.json stay reachable with no key when auth is on."""
    with _client(monkeypatch, API_KEYS="secret-key") as client:
        assert client.get("/health").status_code == 200
        assert client.get("/openapi.json").status_code == 200
