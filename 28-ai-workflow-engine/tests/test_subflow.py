"""Tests for SubflowNodeExecutor (nested-workflow composition)."""

import pytest

from aiworkflow.engine import WorkflowEngine
from aiworkflow.schemas import RunStatus, NodeType, Node, NodeConfig
from aiworkflow.nodes import SubflowNodeExecutor


def make_engine():
    # Disable versioning/optimization for deterministic, isolated runs.
    return WorkflowEngine(enable_versioning=False, enable_optimization=False)


# A trivial child flow: a single transform node with an empty expression
# echoes its inputs straight back, so we can assert on what the parent passed.
CHILD = {
    "name": "child",
    "version": "1.0",
    "nodes": [{"id": "echo", "type": "transform", "config": {"expression": ""}}],
    "outputs": {"echoed": "echo"},
}


async def test_subflow_runs_inline_nested_flow():
    engine = make_engine()
    parent = {
        "name": "parent",
        "version": "1.0",
        "nodes": [{"id": "sub", "type": "subflow", "config": {"flow": CHILD}}],
        "outputs": {"child": "sub"},
    }
    run = await engine.run_flow(
        flow_definition=engine.parser.parse_dict(parent), inputs={"x": 42}
    )
    assert run.status == RunStatus.COMPLETED
    assert run.outputs["child"]["echoed"]["x"] == 42


async def test_subflow_by_registered_name():
    engine = make_engine()
    engine.register_flow(CHILD)
    parent = {
        "name": "parent",
        "version": "1.0",
        "nodes": [{"id": "sub", "type": "subflow", "config": {"flow_name": "child"}}],
        "outputs": {"child": "sub"},
    }
    run = await engine.run_flow(
        flow_definition=engine.parser.parse_dict(parent), inputs={"x": 7}
    )
    assert run.status == RunStatus.COMPLETED
    assert run.outputs["child"]["echoed"]["x"] == 7


async def test_subflow_input_mapping_renames_inputs():
    engine = make_engine()
    parent = {
        "name": "parent",
        "version": "1.0",
        "nodes": [
            {
                "id": "sub",
                "type": "subflow",
                "config": {"flow": CHILD, "input_mapping": {"renamed": "{{x}}"}},
            }
        ],
        "outputs": {"child": "sub"},
    }
    run = await engine.run_flow(
        flow_definition=engine.parser.parse_dict(parent), inputs={"x": 99}
    )
    assert run.status == RunStatus.COMPLETED
    echoed = run.outputs["child"]["echoed"]
    assert echoed.get("renamed") == 99
    assert "x" not in echoed  # only mapped inputs are passed through


async def test_subflow_output_key_wraps_result():
    engine = make_engine()
    parent = {
        "name": "parent",
        "version": "1.0",
        "nodes": [
            {"id": "sub", "type": "subflow", "config": {"flow": CHILD, "output_key": "result"}}
        ],
        "outputs": {"child": "sub"},
    }
    run = await engine.run_flow(
        flow_definition=engine.parser.parse_dict(parent), inputs={"x": 1}
    )
    assert run.status == RunStatus.COMPLETED
    assert run.outputs["child"]["result"]["echoed"]["x"] == 1


async def test_subflow_recursion_limit_fails_cleanly():
    engine = make_engine()
    recur = {
        "name": "recur",
        "version": "1.0",
        "nodes": [
            {"id": "again", "type": "subflow", "config": {"flow_name": "recur", "max_depth": 3}}
        ],
        "outputs": {"r": "again"},
    }
    engine.register_flow(recur)
    run = await engine.run_flow(flow_definition=engine.get_flow("recur"), inputs={})
    assert run.status == RunStatus.FAILED
    assert "recursion limit" in (run.error or "").lower()


async def test_subflow_missing_config_fails():
    engine = make_engine()
    parent = {
        "name": "p",
        "version": "1.0",
        "nodes": [{"id": "sub", "type": "subflow", "config": {}}],
        "outputs": {"o": "sub"},
    }
    run = await engine.run_flow(
        flow_definition=engine.parser.parse_dict(parent), inputs={}
    )
    assert run.status == RunStatus.FAILED
    assert "flow" in (run.error or "").lower()


async def test_unbound_subflow_executor_raises():
    executor = SubflowNodeExecutor()  # no engine bound
    node = Node(id="s", type=NodeType.SUBFLOW, config=NodeConfig(extra={"flow": CHILD}))
    with pytest.raises(RuntimeError, match="engine"):
        await executor.execute(node, {})


async def test_subflow_registered_in_default_registry():
    # The engine should bind an engine-aware subflow executor.
    engine = make_engine()
    executor = engine.node_registry.get(NodeType.SUBFLOW)
    assert isinstance(executor, SubflowNodeExecutor)
    assert executor.engine is engine
