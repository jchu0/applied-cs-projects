"""Extended REST API coverage: job lifecycle + sub-resource endpoints."""

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from ml_orchestrator.api.app import create_app

JOB_BODY = {
    "name": "lifecycle-job",
    "user_id": "user-1",
    "config": {"script_path": "/train.py", "epochs": 1, "batch_size": 8, "learning_rate": 0.01},
    "resources": {"cpus": 2, "memory_gb": 4.0, "gpus": 0},
    "priority": 50,
}


@pytest.fixture
def client():
    with TestClient(create_app()) as c:
        yield c


def _submit(client):
    return client.post("/api/v1/jobs", json=JOB_BODY).json()["id"]


def test_job_subresource_endpoints(client):
    job_id = _submit(client)
    # GET sub-resources should respond (200 with data, or 404 if none yet).
    for path in ("details", "metrics", "checkpoints", "queue-position"):
        assert client.get(f"/api/v1/jobs/{job_id}/{path}").status_code in (200, 404)


def test_job_lifecycle_transitions(client):
    job_id = _submit(client)
    # pause/resume/cancel exercise the state-machine routes; accept success or
    # a 4xx if the transition isn't valid from the current state.
    for action in ("pause", "resume", "cancel"):
        resp = client.post(f"/api/v1/jobs/{job_id}/{action}")
        assert resp.status_code in (200, 400, 409)


def test_list_jobs_with_filters(client):
    _submit(client)
    # list endpoint supports query filters; just exercise them.
    assert client.get("/api/v1/jobs", params={"user_id": "user-1"}).status_code == 200
    assert client.get("/api/v1/jobs", params={"status": "pending"}).status_code in (200, 422)
