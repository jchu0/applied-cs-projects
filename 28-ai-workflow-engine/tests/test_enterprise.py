"""Tests for enterprise features: HITL, secrets, and visualization."""

import asyncio

import pytest

from aiworkflow.engine import WorkflowEngine
from aiworkflow.schemas import RunStatus
from aiworkflow.enterprise import (
    HumanReviewStore,
    InMemorySecretProvider,
    EnvSecretProvider,
    ChainedSecretProvider,
    SecretResolver,
    to_mermaid,
    to_dot,
    run_to_mermaid,
)


APPROVAL_FLOW = {
    "name": "approval",
    "version": "1.0",
    "nodes": [{"id": "review", "type": "human_review", "config": {"prompt": "approve?"}}],
    "outputs": {"decision": "review"},
}


def engine(**kw):
    kw.setdefault("enable_versioning", False)
    kw.setdefault("enable_optimization", False)
    return WorkflowEngine(**kw)


# ---------------------------------------------------------------- HITL --------

async def test_human_review_auto_approve():
    eng = engine(review_store=HumanReviewStore(auto_approve=True))
    run = await eng.run_flow(
        flow_definition=eng.parser.parse_dict(APPROVAL_FLOW), inputs={"doc": "x"}
    )
    assert run.status == RunStatus.COMPLETED
    assert run.outputs["decision"]["approved"] is True


async def _resolve_pending(store, decision, **kw):
    """Concurrent helper: wait for a pending review, then resolve it."""
    for _ in range(400):
        pending = store.list_pending()
        if pending:
            getattr(store, decision)(pending[0].id, **kw)
            return pending[0]
        await asyncio.sleep(0.005)
    raise AssertionError("no pending review appeared")


async def test_human_review_manual_approval_unblocks_run():
    eng = engine()
    flow = eng.parser.parse_dict(APPROVAL_FLOW)
    # Run the flow and the (human) resolver concurrently on the same loop.
    run, resolved = await asyncio.gather(
        eng.run_flow(flow_definition=flow, inputs={"doc": "x"}),
        _resolve_pending(eng.review_store, "approve", reviewer="alice", comment="lgtm"),
    )
    assert resolved.prompt == "approve?"
    assert run.status == RunStatus.COMPLETED
    decision = run.outputs["decision"]
    assert decision["approved"] is True
    assert decision["reviewer"] == "alice"
    assert decision["comment"] == "lgtm"


async def test_human_review_rejection_fails_run():
    eng = engine()
    flow = eng.parser.parse_dict(APPROVAL_FLOW)
    run, _ = await asyncio.gather(
        eng.run_flow(flow_definition=flow, inputs={}),
        _resolve_pending(eng.review_store, "reject", reviewer="bob", comment="nope"),
    )
    assert run.status == RunStatus.FAILED
    assert "rejected" in (run.error or "").lower()


async def test_human_review_timeout():
    flow = {
        "name": "approval",
        "version": "1.0",
        "nodes": [
            {"id": "review", "type": "human_review", "config": {"timeout_seconds": 0.05}}
        ],
        "outputs": {"decision": "review"},
    }
    eng = engine()
    run = await eng.run_flow(flow_definition=eng.parser.parse_dict(flow), inputs={})
    assert run.status == RunStatus.FAILED
    assert "timed out" in (run.error or "").lower()


def test_review_store_double_resolve_raises():
    store = HumanReviewStore()
    req = store.create_review(node_id="n", payload={})
    store.approve(req.id, reviewer="a")
    with pytest.raises(ValueError):
        store.reject(req.id, reviewer="b")


# ------------------------------------------------------------- secrets --------

def test_inmemory_and_chained_providers():
    mem = InMemorySecretProvider({"API_KEY": "sk-123"})
    assert mem.get("API_KEY") == "sk-123"
    assert mem.get("MISSING") is None
    chained = ChainedSecretProvider(InMemorySecretProvider({}), mem)
    assert chained.get("API_KEY") == "sk-123"


def test_env_secret_provider(monkeypatch):
    monkeypatch.setenv("WF_TOKEN", "envval")
    provider = EnvSecretProvider(prefix="WF_")
    assert provider.get("TOKEN") == "envval"
    assert provider.get("NOPE") is None


def test_secret_resolver_resolves_and_masks():
    resolver = SecretResolver(InMemorySecretProvider({"OPENAI_KEY": "sk-secret"}))
    config = {"model": "gpt-4", "headers": {"Authorization": "Bearer ${secret:OPENAI_KEY}"}}
    resolved = resolver.resolve(config)
    assert resolved["headers"]["Authorization"] == "Bearer sk-secret"
    # masking redacts the resolved value from log text
    log_line = f"calling api with {resolved['headers']['Authorization']}"
    assert "sk-secret" not in resolver.mask(log_line)


def test_secret_resolver_strict_vs_lenient():
    strict = SecretResolver(InMemorySecretProvider({}), strict=True)
    with pytest.raises(KeyError):
        strict.resolve_string("${secret:UNKNOWN}")
    lenient = SecretResolver(InMemorySecretProvider({}), strict=False)
    assert lenient.resolve_string("${secret:UNKNOWN}") == "${secret:UNKNOWN}"


def test_secret_references_discovery():
    obj = {"a": "${secret:ONE}", "b": ["${secret:TWO}", "plain"]}
    assert SecretResolver.references(obj) == {"ONE", "TWO"}


# ----------------------------------------------------------------- viz --------

VIZ_FLOW = {
    "name": "viz",
    "version": "1.0",
    "nodes": [
        {"id": "a", "type": "transform", "config": {"expression": ""}},
        {"id": "b", "type": "transform", "config": {"expression": ""}, "dependencies": ["a"]},
    ],
    "outputs": {"out": "b"},
}


def test_to_mermaid_and_dot():
    eng = engine()
    flow = eng.parser.parse_dict(VIZ_FLOW)
    mer = to_mermaid(flow)
    assert "flowchart TD" in mer
    assert "a --> b" in mer
    dot = to_dot(flow)
    assert "digraph" in dot
    assert '"a" -> "b";' in dot


async def test_run_to_mermaid_colors_executed_nodes():
    eng = engine()
    flow = eng.parser.parse_dict(VIZ_FLOW)
    run = await eng.run_flow(flow_definition=flow, inputs={})
    assert run.status == RunStatus.COMPLETED
    diagram = run_to_mermaid(flow, run)
    assert "classDef done" in diagram
    assert "class a done" in diagram
