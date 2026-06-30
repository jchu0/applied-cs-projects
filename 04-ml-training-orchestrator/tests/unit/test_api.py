"""Tests for the FastAPI REST layer (via TestClient)."""

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from ml_orchestrator.api.app import create_app


JOB_BODY = {
    "name": "api-job",
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


@pytest.fixture
def client():
    # Context manager runs startup/shutdown (which starts/stops the scheduler).
    with TestClient(create_app()) as c:
        yield c


def test_health_endpoints(client):
    assert client.get("/health").json()["status"] == "healthy"
    assert client.get("/health/live").json()["status"] == "alive"
    assert client.get("/health/ready").json()["status"] == "ready"


def test_health_stats_aggregates_components(client):
    stats = client.get("/health/stats").json()
    for key in ("scheduler", "job_manager", "allocator", "gpu_manager"):
        assert key in stats


def test_submit_list_get_job(client):
    resp = client.post("/api/v1/jobs", json=JOB_BODY)
    assert resp.status_code == 200, resp.text
    job = resp.json()
    assert job["name"] == "api-job"
    job_id = job["id"]

    listed = client.get("/api/v1/jobs").json()
    assert any(j["id"] == job_id for j in listed)

    fetched = client.get(f"/api/v1/jobs/{job_id}").json()
    assert fetched["id"] == job_id


def test_get_missing_job_404(client):
    assert client.get("/api/v1/jobs/does-not-exist").status_code == 404


def test_update_job_priority(client):
    job_id = client.post("/api/v1/jobs", json=JOB_BODY).json()["id"]
    resp = client.patch(f"/api/v1/jobs/{job_id}", json={"priority": 75})
    assert resp.status_code == 200
    assert resp.json()["priority"] == 75


def test_submit_invalid_job_422(client):
    # Missing required config.script_path -> request validation error
    bad = {"name": "x", "user_id": "u", "config": {"epochs": 1}}
    assert client.post("/api/v1/jobs", json=bad).status_code == 422


def test_resources_and_experiments_list(client):
    # List endpoints should respond (empty collections initially).
    assert client.get("/api/v1/resources/workers").status_code in (200, 404)
    assert client.get("/api/v1/experiments").status_code in (200, 404)
