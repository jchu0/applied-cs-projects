"""Tests for the FastAPI REST layer."""

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from aiworkflow.api import create_app
from aiworkflow.engine import WorkflowEngine
from aiworkflow.enterprise import HumanReviewStore


def make_client(auto_approve: bool = False):
    store = HumanReviewStore(auto_approve=auto_approve)
    engine = WorkflowEngine(
        enable_versioning=False, enable_optimization=False, review_store=store
    )
    return TestClient(create_app(engine)), engine


ECHO_FLOW = {
    "name": "echo",
    "version": "1.0",
    "nodes": [{"id": "t", "type": "transform", "config": {"expression": ""}}],
    "outputs": {"out": "t"},
}

APPROVAL_FLOW = {
    "name": "approval",
    "version": "1.0",
    "nodes": [{"id": "review", "type": "human_review", "config": {"prompt": "ok?"}}],
    "outputs": {"decision": "review"},
}


def test_health():
    client, _ = make_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_register_list_get_flow():
    client, _ = make_client()
    resp = client.post("/flows", json={"spec": ECHO_FLOW, "description": "echo flow"})
    assert resp.status_code == 201
    assert resp.json()["name"] == "echo"

    listed = client.get("/flows").json()
    assert any(f["name"] == "echo" for f in listed)

    got = client.get("/flows/echo").json()
    assert got["name"] == "echo"
    assert got["nodes"][0]["id"] == "t"


def test_get_flow_404():
    client, _ = make_client()
    assert client.get("/flows/missing").status_code == 404


def test_register_invalid_flow_400():
    client, _ = make_client()
    # No 'nodes' key -> parse error -> 400
    resp = client.post("/flows", json={"spec": {"name": "bad"}})
    assert resp.status_code == 400


def test_run_registered_flow():
    client, _ = make_client()
    client.post("/flows", json={"spec": ECHO_FLOW})
    resp = client.post("/flows/echo/run", json={"inputs": {"x": 5}})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["outputs"]["out"]["x"] == 5


def test_run_inline_flow_and_history():
    client, _ = make_client()
    resp = client.post("/runs", json={"spec": ECHO_FLOW, "inputs": {"y": 9}})
    assert resp.status_code == 200
    run = resp.json()
    assert run["status"] == "completed"
    assert run["outputs"]["out"]["y"] == 9

    history = client.get("/runs").json()
    assert len(history) >= 1
    fetched = client.get(f"/runs/{run['run_id']}").json()
    assert fetched["run_id"] == run["run_id"]


def test_diagram_endpoint():
    client, _ = make_client()
    client.post("/flows", json={"spec": ECHO_FLOW})
    mer = client.get("/flows/echo/diagram").json()
    assert mer["format"] == "mermaid"
    assert "flowchart TD" in mer["diagram"]
    dot = client.get("/flows/echo/diagram", params={"fmt": "dot"}).json()
    assert "digraph" in dot["diagram"]


def test_review_lifecycle_via_api():
    client, engine = make_client()
    review = engine.review_store.create_review(node_id="n", payload={"a": 1}, prompt="approve?")

    pending = client.get("/reviews").json()
    assert any(r["id"] == review.id for r in pending)

    resp = client.post(
        f"/reviews/{review.id}/approve", json={"reviewer": "alice", "comment": "lgtm"}
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "approved"

    # No longer pending; re-resolving is a conflict.
    assert client.get("/reviews").json() == []
    again = client.post(f"/reviews/{review.id}/reject", json={"reviewer": "bob"})
    assert again.status_code == 409


def test_review_not_found_404():
    client, _ = make_client()
    assert client.post("/reviews/nope/approve", json={}).status_code == 404


def test_human_review_flow_auto_approves():
    client, _ = make_client(auto_approve=True)
    resp = client.post("/runs", json={"spec": APPROVAL_FLOW, "inputs": {"doc": "d"}})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["outputs"]["decision"]["approved"] is True
