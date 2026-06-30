"""Pytest configuration and shared fixtures for AI Workflow Engine tests."""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock
import json

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Check for yaml availability
try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

from aiworkflow.schemas import (
    FlowDefinition,
    Node as NodeDefinition,  # Alias for compatibility
    NodeType,
    RunStatus
)

# Optional imports requiring yaml
if _HAS_YAML:
    from aiworkflow.engine import WorkflowEngine
else:
    WorkflowEngine = None


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_flow_definitions():
    """Collection of sample flow definitions for testing."""
    return {
        "simple": FlowDefinition(
            name="simple-flow",
            version="1.0.0",
            description="Simple linear flow",
            nodes=[
                NodeDefinition(
                    id="node1",
                    name="Node1",
                    type=NodeType.DATA,
                    config={"source": "data.csv"}
                ),
                NodeDefinition(
                    id="node2",
                    name="Node2",
                    type=NodeType.PROCESS,
                    dependencies=["node1"],
                    config={"operation": "transform"}
                )
            ],
            edges=[{"from": "node1", "to": "node2"}]
        ),
        "parallel": FlowDefinition(
            name="parallel-flow",
            version="1.0.0",
            description="Flow with parallel branches",
            nodes=[
                NodeDefinition(
                    id="start",
                    name="Start",
                    type=NodeType.DATA,
                    config={}
                ),
                NodeDefinition(
                    id="branch1",
                    name="Branch1",
                    type=NodeType.PROCESS,
                    dependencies=["start"],
                    config={}
                ),
                NodeDefinition(
                    id="branch2",
                    name="Branch2",
                    type=NodeType.PROCESS,
                    dependencies=["start"],
                    config={}
                ),
                NodeDefinition(
                    id="merge",
                    name="Merge",
                    type=NodeType.PROCESS,
                    dependencies=["branch1", "branch2"],
                    config={}
                )
            ],
            edges=[
                {"from": "start", "to": "branch1"},
                {"from": "start", "to": "branch2"},
                {"from": "branch1", "to": "merge"},
                {"from": "branch2", "to": "merge"}
            ]
        ),
        "conditional": FlowDefinition(
            name="conditional-flow",
            version="1.0.0",
            description="Flow with conditional branching",
            nodes=[
                NodeDefinition(
                    id="check",
                    name="Check",
                    type=NodeType.CONDITIONAL,
                    config={"condition": "score > 0.8"}
                ),
                NodeDefinition(
                    id="high_path",
                    name="HighPath",
                    type=NodeType.PROCESS,
                    dependencies=["check"],
                    config={},
                    condition="branch == 'high'"
                ),
                NodeDefinition(
                    id="low_path",
                    name="LowPath",
                    type=NodeType.PROCESS,
                    dependencies=["check"],
                    config={},
                    condition="branch == 'low'"
                )
            ],
            edges=[
                {"from": "check", "to": "high_path"},
                {"from": "check", "to": "low_path"}
            ]
        )
    }


@pytest.fixture
def mock_executors():
    """Collection of mock node executors."""
    executors = {}

    # Data loader executor
    async def data_loader(inputs):
        return {
            "data": {
                "features": [[1, 2], [3, 4]],
                "labels": [0, 1]
            }
        }
    executors["data_loader"] = data_loader

    # Preprocessor executor
    async def preprocessor(inputs):
        data = inputs.get("data", {})
        return {
            "processed_data": data,
            "metadata": {"preprocessed": True}
        }
    executors["preprocessor"] = preprocessor

    # Model trainer executor
    async def trainer(inputs):
        return {
            "model": "trained_model",
            "metrics": {"accuracy": 0.95}
        }
    executors["trainer"] = trainer

    # Conditional executor
    async def conditional(inputs):
        score = inputs.get("score", 0)
        if score > 0.8:
            return {"branch": "high", "next_node": "high_path"}
        return {"branch": "low", "next_node": "low_path"}
    executors["conditional"] = conditional

    return executors


@pytest.fixture
def temp_workflow_dir():
    """Create a temporary directory for workflow files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_workflow_files(temp_workflow_dir):
    """Create sample workflow definition files."""
    if not _HAS_YAML:
        pytest.skip("Requires yaml")
    # YAML workflow
    yaml_flow = {
        "name": "test-workflow",
        "version": "1.0.0",
        "description": "Test workflow from YAML",
        "nodes": [
            {
                "id": "node1",
                "name": "DataLoader",
                "type": "data",
                "config": {"source": "test.csv"}
            },
            {
                "id": "node2",
                "name": "Processor",
                "type": "process",
                "dependencies": ["node1"],
                "config": {"method": "transform"}
            }
        ],
        "edges": [
            {"from": "node1", "to": "node2"}
        ]
    }

    yaml_file = temp_workflow_dir / "workflow.yaml"
    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_flow, f)

    # JSON workflow
    json_flow = {
        "name": "json-workflow",
        "version": "2.0.0",
        "nodes": [
            {
                "id": "start",
                "name": "Start",
                "type": "data",
                "config": {}
            }
        ],
        "edges": []
    }

    json_file = temp_workflow_dir / "workflow.json"
    with open(json_file, 'w') as f:
        json.dump(json_flow, f)

    # DSL workflow
    dsl_content = """
    workflow ml_pipeline:
        node data_loader:
            type: data
            config:
                source: "database"

        node preprocessor:
            type: process
            depends_on: data_loader
            config:
                normalize: true

        node trainer:
            type: model
            depends_on: preprocessor
            config:
                algorithm: "xgboost"

        flow:
            data_loader -> preprocessor -> trainer
    """

    dsl_file = temp_workflow_dir / "workflow.dsl"
    dsl_file.write_text(dsl_content)

    return {
        "yaml": yaml_file,
        "json": json_file,
        "dsl": dsl_file
    }


@pytest.fixture
def mock_data_sources():
    """Mock data sources for testing."""
    return {
        "csv_data": [
            {"id": 1, "value": 100},
            {"id": 2, "value": 200}
        ],
        "database_result": {
            "users": [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"}
            ]
        },
        "api_response": {
            "status": "success",
            "data": {"items": [1, 2, 3]}
        }
    }


@pytest.fixture
def performance_metrics():
    """Helper for tracking performance metrics."""
    class PerformanceTracker:
        def __init__(self):
            self.metrics = {}

        def start(self, name):
            import time
            self.metrics[name] = {"start": time.time()}

        def end(self, name):
            import time
            if name in self.metrics:
                self.metrics[name]["end"] = time.time()
                self.metrics[name]["duration"] = (
                    self.metrics[name]["end"] - self.metrics[name]["start"]
                )

        def get_duration(self, name):
            return self.metrics.get(name, {}).get("duration", 0)

    return PerformanceTracker()


@pytest.fixture
def cleanup_registry():
    """Cleanup node executor registry after test."""
    from aiworkflow.nodes import NodeExecutorRegistry

    original_registry = {}

    def _cleanup(registry: NodeExecutorRegistry):
        # Save original state
        for key in list(registry._executors.keys()):
            original_registry[key] = registry._executors[key]

        yield

        # Restore original state
        registry._executors.clear()
        for key, value in original_registry.items():
            registry._executors[key] = value

    return _cleanup