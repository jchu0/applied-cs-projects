"""Tests for the API hardening baseline: auth, rate limiting, timeouts."""

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from ml_orchestrator.api.app import create_app


JOB_BODY = {
    "name": "secure-job",
    "user_id": "user-1",
    "config": {
        "script_path": "/train.py",
        "epochs": 1,
        "batch_size": 32,
        "learning_rate": 0.001,
        "timeout_hours": 1.0,
    },
    "resources": {"cpus": 2, "memory_gb": 8.0, "gpus": 0},
    "priority": 50,
}


def test_missing_or_bad_key_401_when_auth_enabled(monkeypatch):
    monkeypatch.setenv("API_KEYS", "secret-key")
    with TestClient(create_app()) as client:
        # No key -> 401 with WWW-Authenticate.
        resp = client.get("/api/v1/jobs")
        assert resp.status_code == 401
        assert "WWW-Authenticate" in resp.headers

        # Wrong key -> 401 (try both header styles).
        assert client.get(
            "/api/v1/jobs", headers={"X-API-Key": "nope"}
        ).status_code == 401
        assert client.get(
            "/api/v1/jobs", headers={"Authorization": "Bearer nope"}
        ).status_code == 401


def test_valid_key_200_via_both_header_styles(monkeypatch):
    monkeypatch.setenv("API_KEYS", "secret-key,other-key")
    with TestClient(create_app()) as client:
        assert client.get(
            "/api/v1/jobs", headers={"X-API-Key": "secret-key"}
        ).status_code == 200
        assert client.get(
            "/api/v1/jobs", headers={"Authorization": "Bearer other-key"}
        ).status_code == 200

        # A protected write path also works with a valid key.
        resp = client.post(
            "/api/v1/jobs", json=JOB_BODY, headers={"X-API-Key": "secret-key"}
        )
        assert resp.status_code == 200, resp.text


def test_health_and_openapi_open_without_key(monkeypatch):
    monkeypatch.setenv("API_KEYS", "secret-key")
    with TestClient(create_app()) as client:
        assert client.get("/health").json()["status"] == "healthy"
        assert client.get("/health/live").status_code == 200
        assert client.get("/openapi.json").status_code == 200
        assert client.get("/docs").status_code == 200


def test_rate_limit_429_over_low_limit(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "3")
    # Auth off so we key by client IP.
    monkeypatch.delenv("API_KEYS", raising=False)
    with TestClient(create_app()) as client:
        codes = [client.get("/api/v1/jobs").status_code for _ in range(5)]
        assert 429 in codes
        # First few succeed, then we trip the limit.
        assert codes[:3] == [200, 200, 200]
        limited = client.get("/api/v1/jobs")
        assert limited.status_code == 429
        assert "Retry-After" in limited.headers
