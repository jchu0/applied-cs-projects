"""Tests for the opt-in API hardening layer (auth, rate limiting)."""

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from aiworkflow.api import create_app
from aiworkflow.engine import WorkflowEngine
from aiworkflow.enterprise import HumanReviewStore


def make_client():
    store = HumanReviewStore()
    engine = WorkflowEngine(
        enable_versioning=False, enable_optimization=False, review_store=store
    )
    return TestClient(create_app(engine))


def test_401_when_auth_enabled_and_key_missing_or_bad(monkeypatch):
    monkeypatch.setenv("API_KEYS", "secret-key")
    client = make_client()

    missing = client.get("/flows")
    assert missing.status_code == 401
    assert "WWW-Authenticate" in missing.headers

    bad = client.get("/flows", headers={"X-API-Key": "wrong"})
    assert bad.status_code == 401


def test_200_with_valid_key(monkeypatch):
    monkeypatch.setenv("API_KEYS", "secret-key,other-key")
    client = make_client()

    bearer = client.get("/flows", headers={"Authorization": "Bearer secret-key"})
    assert bearer.status_code == 200

    header = client.get("/flows", headers={"X-API-Key": "other-key"})
    assert header.status_code == 200


def test_429_when_over_low_limit(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "3")
    client = make_client()

    responses = [client.get("/flows") for _ in range(5)]
    statuses = [r.status_code for r in responses]
    assert statuses.count(200) == 3
    limited = [r for r in responses if r.status_code == 429]
    assert limited, "expected at least one 429"
    assert "Retry-After" in limited[0].headers


def test_health_and_openapi_open_without_key(monkeypatch):
    monkeypatch.setenv("API_KEYS", "secret-key")
    client = make_client()

    assert client.get("/health").status_code == 200
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200
