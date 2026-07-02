"""Tests for API hardening: auth, rate limiting, request timeout."""

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from syntheticdata.api import create_api


def _client() -> TestClient:
    return TestClient(create_api())


def test_401_when_auth_enabled_and_key_missing(monkeypatch):
    monkeypatch.setenv("API_KEYS", "secret-key")
    client = _client()
    resp = client.get("/domains")
    assert resp.status_code == 401
    assert resp.headers.get("WWW-Authenticate") == "Bearer"


def test_401_when_key_bad(monkeypatch):
    monkeypatch.setenv("API_KEYS", "secret-key")
    client = _client()
    resp = client.get("/domains", headers={"X-API-Key": "wrong"})
    assert resp.status_code == 401


def test_200_with_valid_bearer_key(monkeypatch):
    monkeypatch.setenv("API_KEYS", "secret-key")
    client = _client()
    resp = client.get("/domains", headers={"Authorization": "Bearer secret-key"})
    assert resp.status_code == 200


def test_200_with_valid_x_api_key(monkeypatch):
    monkeypatch.setenv("API_KEYS", "a-key,b-key")
    client = _client()
    resp = client.get("/domains", headers={"X-API-Key": "b-key"})
    assert resp.status_code == 200


def test_health_and_openapi_open_without_key(monkeypatch):
    monkeypatch.setenv("API_KEYS", "secret-key")
    client = _client()
    assert client.get("/health").status_code == 200
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/").status_code == 200


def test_429_when_over_low_limit(monkeypatch):
    monkeypatch.delenv("API_KEYS", raising=False)
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "3")
    client = _client()

    statuses = [client.get("/domains").status_code for _ in range(5)]
    assert 429 in statuses
    # First 3 should succeed, then limited.
    assert statuses[:3] == [200, 200, 200]
    limited = next(r for r in [client.get("/domains") for _ in range(1)])
    assert limited.status_code == 429
    assert "Retry-After" in limited.headers


def test_health_exempt_from_rate_limit(monkeypatch):
    monkeypatch.delenv("API_KEYS", raising=False)
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")
    client = _client()
    # Well over the limit, but health is exempt.
    for _ in range(10):
        assert client.get("/health").status_code == 200


def test_existing_behavior_with_auth_off(monkeypatch):
    monkeypatch.delenv("API_KEYS", raising=False)
    monkeypatch.delenv("RATE_LIMIT_PER_MINUTE", raising=False)
    client = _client()
    assert client.get("/domains").status_code == 200
