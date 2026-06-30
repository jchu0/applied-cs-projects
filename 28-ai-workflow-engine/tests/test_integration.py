"""Integration tests for the AI Workflow Engine."""

import pytest
import asyncio
import json
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import time

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Optional imports requiring yaml
try:
    import yaml
    from aiworkflow.engine import WorkflowEngine
    from aiworkflow.compiler import FlowParser
    from aiworkflow.nodes import NodeExecutorRegistry
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

from aiworkflow.schemas import (
    FlowDefinition,
    Node as NodeDefinition,
    NodeType,
    RunStatus
)

pytestmark = pytest.mark.skipif(not _HAS_YAML, reason="Requires yaml")


class TestEndToEndWorkflow:
    """End-to-end workflow execution tests."""

    @pytest.fixture
    def engine(self):
        """Create a fully configured workflow engine."""
        engine = WorkflowEngine(
            max_parallel=5,
            enable_versioning=True,
            enable_optimization=True
        )

        # Register mock node executors
        registry = engine.node_registry

        # Mock data loader
        async def mock_data_loader(inputs):
            return {
                "data": {
                    "features": [[1, 2], [3, 4], [5, 6]],
                    "labels": [0, 1, 0]
                }
            }
        registry.register("data_loader", mock_data_loader)

        # Mock preprocessor
        async def mock_preprocessor(inputs):
            data = inputs.get("data", {})
            return {
                "processed_data": {
                    "features": data.get("features"),
                    "labels": data.get("labels"),
                    "metadata": {"preprocessed": True}
                }
            }
        registry.register("preprocessor", mock_preprocessor)

        # Mock model trainer
        async def mock_trainer(inputs):
            return {
                "model": "trained_model_object",
                "metrics": {
                    "accuracy": 0.95,
                    "loss": 0.05
                }
            }
        registry.register("model_trainer", mock_trainer)

        # Mock model evaluator
        async def mock_evaluator(inputs):
            return {
                "evaluation": {
                    "test_accuracy": 0.93,
                    "precision": 0.94,
                    "recall": 0.92
                }
            }
        registry.register("model_evaluator", mock_evaluator)

        return engine

    @pytest.mark.asyncio
    async def test_simple_ml_pipeline(self, engine):
        """Test a simple ML pipeline execution."""
        flow_def = FlowDefinition(
            name="ml-pipeline",
            version="1.0.0",
            description="Simple ML pipeline",
            nodes=[
                NodeDefinition(
                    id="loader",
                    name="DataLoader",
                    type=NodeType.DATA,
                    executor="data_loader",
                    config={"source": "dataset.csv"}
                ),
                NodeDefinition(
                    id="preprocess",
                    name="Preprocessor",
                    type=NodeType.PROCESS,
                    executor="preprocessor",
                    dependencies=["loader"],
                    config={"normalize": True}
                ),
                NodeDefinition(
                    id="train",
                    name="ModelTrainer",
                    type=NodeType.MODEL,
                    executor="model_trainer",
                    dependencies=["preprocess"],
                    config={"algorithm": "random_forest"}
                ),
                NodeDefinition(
                    id="evaluate",
                    name="ModelEvaluator",
                    type=NodeType.VALIDATION,
                    executor="model_evaluator",
                    dependencies=["train"],
                    config={}
                )
            ],
            edges=[
                {"from": "loader", "to": "preprocess"},
                {"from": "preprocess", "to": "train"},
                {"from": "train", "to": "evaluate"}
            ]
        )

        # Execute workflow
        run = await engine.run_flow(flow_def, inputs={})

        assert run.status == RunStatus.COMPLETED
        assert run.outputs is not None
        assert "evaluation" in run.outputs
        assert run.outputs["evaluation"]["test_accuracy"] == 0.93

    @pytest.mark.asyncio
    async def test_parallel_branches(self, engine):
        """Test workflow with parallel execution branches."""
        # Register additional mock executors
        async def mock_branch_a(inputs):
            await asyncio.sleep(0.1)  # Simulate work
            return {"branch_a_result": "completed_a"}

        async def mock_branch_b(inputs):
            await asyncio.sleep(0.1)  # Simulate work
            return {"branch_b_result": "completed_b"}

        async def mock_merger(inputs):
            return {
                "merged_result": {
                    "a": inputs.get("branch_a_result"),
                    "b": inputs.get("branch_b_result")
                }
            }

        engine.node_registry.register("branch_a", mock_branch_a)
        engine.node_registry.register("branch_b", mock_branch_b)
        engine.node_registry.register("merger", mock_merger)

        flow_def = FlowDefinition(
            name="parallel-flow",
            version="1.0.0",
            nodes=[
                NodeDefinition(
                    id="start",
                    name="Start",
                    type=NodeType.DATA,
                    executor="data_loader",
                    config={}
                ),
                NodeDefinition(
                    id="branch_a",
                    name="BranchA",
                    type=NodeType.PROCESS,
                    executor="branch_a",
                    dependencies=["start"],
                    config={}
                ),
                NodeDefinition(
                    id="branch_b",
                    name="BranchB",
                    type=NodeType.PROCESS,
                    executor="branch_b",
                    dependencies=["start"],
                    config={}
                ),
                NodeDefinition(
                    id="merge",
                    name="Merge",
                    type=NodeType.PROCESS,
                    executor="merger",
                    dependencies=["branch_a", "branch_b"],
                    config={}
                )
            ],
            edges=[
                {"from": "start", "to": "branch_a"},
                {"from": "start", "to": "branch_b"},
                {"from": "branch_a", "to": "merge"},
                {"from": "branch_b", "to": "merge"}
            ]
        )

        start_time = time.time()
        run = await engine.run_flow(flow_def, inputs={})
        execution_time = time.time() - start_time

        assert run.status == RunStatus.COMPLETED
        assert run.outputs["merged_result"]["a"] == "completed_a"
        assert run.outputs["merged_result"]["b"] == "completed_b"

        # Verify parallel execution (should be faster than sequential)
        assert execution_time < 0.3  # Both branches run in parallel

    @pytest.mark.asyncio
    async def test_workflow_with_failure_recovery(self, engine):
        """Test workflow execution with failure and recovery."""
        attempt_count = 0

        async def flaky_node(inputs):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception("Temporary failure")
            return {"result": "success"}

        engine.node_registry.register("flaky", flaky_node)

        flow_def = FlowDefinition(
            name="retry-flow",
            version="1.0.0",
            nodes=[
                NodeDefinition(
                    id="flaky_node",
                    name="FlakyNode",
                    type=NodeType.PROCESS,
                    executor="flaky",
                    config={},
                    retry_config={
                        "max_retries": 5,
                        "strategy": "exponential",
                        "base_delay": 0.01
                    }
                )
            ],
            edges=[]
        )

        run = await engine.run_flow(flow_def, inputs={})

        assert run.status == RunStatus.COMPLETED
        assert attempt_count == 3  # Failed twice, succeeded on third

    @pytest.mark.asyncio
    async def test_conditional_workflow(self, engine):
        """Test workflow with conditional branching."""
        async def condition_checker(inputs):
            score = inputs.get("score", 0)
            if score > 0.8:
                return {"branch": "high_score", "next_node": "advanced_processing"}
            else:
                return {"branch": "low_score", "next_node": "basic_processing"}

        async def advanced_processing(inputs):
            return {"result": "advanced_result"}

        async def basic_processing(inputs):
            return {"result": "basic_result"}

        engine.node_registry.register("condition", condition_checker)
        engine.node_registry.register("advanced", advanced_processing)
        engine.node_registry.register("basic", basic_processing)

        # Test high score path
        flow_def = FlowDefinition(
            name="conditional-flow",
            version="1.0.0",
            nodes=[
                NodeDefinition(
                    id="check",
                    name="ConditionCheck",
                    type=NodeType.CONDITIONAL,
                    executor="condition",
                    config={}
                ),
                NodeDefinition(
                    id="advanced_processing",
                    name="AdvancedProcessing",
                    type=NodeType.PROCESS,
                    executor="advanced",
                    dependencies=["check"],
                    config={},
                    condition="branch == 'high_score'"
                ),
                NodeDefinition(
                    id="basic_processing",
                    name="BasicProcessing",
                    type=NodeType.PROCESS,
                    executor="basic",
                    dependencies=["check"],
                    config={},
                    condition="branch == 'low_score'"
                )
            ],
            edges=[
                {"from": "check", "to": "advanced_processing"},
                {"from": "check", "to": "basic_processing"}
            ]
        )

        # Test high score branch
        run = await engine.run_flow(flow_def, inputs={"score": 0.9})
        assert run.outputs["result"] == "advanced_result"

        # Test low score branch
        run = await engine.run_flow(flow_def, inputs={"score": 0.5})
        assert run.outputs["result"] == "basic_result"


class TestWorkflowPersistence:
    """Test workflow state persistence and recovery."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.mark.asyncio
    async def test_save_and_load_workflow_state(self, temp_storage):
        """Test saving and loading workflow state."""
        engine = WorkflowEngine(enable_versioning=True)

        flow_def = FlowDefinition(
            name="persistent-flow",
            version="1.0.0",
            nodes=[
                NodeDefinition(
                    id="node1",
                    name="Node1",
                    type=NodeType.DATA,
                    config={"data": "test"}
                )
            ],
            edges=[]
        )

        # Save workflow
        state_file = temp_storage / "workflow_state.json"
        engine.save_state(flow_def, state_file)

        assert state_file.exists()

        # Load workflow
        loaded_flow = engine.load_state(state_file)

        assert loaded_flow.name == flow_def.name
        assert loaded_flow.version == flow_def.version
        assert len(loaded_flow.nodes) == len(flow_def.nodes)

    @pytest.mark.asyncio
    async def test_checkpoint_recovery(self, temp_storage):
        """Test workflow recovery from checkpoint."""
        engine = WorkflowEngine(
            enable_checkpointing=True,
            checkpoint_dir=str(temp_storage)
        )

        # Create a workflow that checkpoints after each node
        checkpoint_count = 0

        async def checkpointed_node(inputs):
            nonlocal checkpoint_count
            checkpoint_count += 1
            return {"checkpoint": checkpoint_count}

        engine.node_registry.register("checkpoint_node", checkpointed_node)

        flow_def = FlowDefinition(
            name="checkpoint-flow",
            version="1.0.0",
            nodes=[
                NodeDefinition(
                    id=f"node{i}",
                    name=f"Node{i}",
                    type=NodeType.PROCESS,
                    executor="checkpoint_node",
                    dependencies=[f"node{i-1}"] if i > 1 else [],
                    config={},
                    checkpoint=True
                )
                for i in range(1, 4)
            ],
            edges=[
                {"from": f"node{i}", "to": f"node{i+1}"}
                for i in range(1, 3)
            ]
        )

        # Start execution
        run_id = await engine.start_flow(flow_def, inputs={})

        # Simulate interruption after first node
        await asyncio.sleep(0.1)
        engine.interrupt_flow(run_id)

        # Check that checkpoint exists
        checkpoint_files = list(temp_storage.glob("*.checkpoint"))
        assert len(checkpoint_files) > 0

        # Resume from checkpoint
        resumed_run = await engine.resume_from_checkpoint(run_id)

        assert resumed_run.status == RunStatus.COMPLETED
        assert checkpoint_count == 3  # All nodes executed


class TestWorkflowFromFiles:
    """Test loading and executing workflows from files."""

    @pytest.fixture
    def temp_workflow_files(self):
        """Create temporary workflow definition files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create YAML workflow
            yaml_file = tmpdir / "workflow.yaml"
            yaml_content = """
            name: yaml-workflow
            version: 1.0.0
            description: Workflow from YAML
            nodes:
              - id: start
                name: Start
                type: data
                config:
                  source: input.csv
              - id: process
                name: Process
                type: process
                dependencies: [start]
                config:
                  operation: transform
            edges:
              - from: start
                to: process
            """
            yaml_file.write_text(yaml_content)

            # Create JSON workflow
            json_file = tmpdir / "workflow.json"
            json_content = {
                "name": "json-workflow",
                "version": "1.0.0",
                "description": "Workflow from JSON",
                "nodes": [
                    {
                        "id": "node1",
                        "name": "Node1",
                        "type": "data",
                        "config": {"key": "value"}
                    }
                ],
                "edges": []
            }
            json_file.write_text(json.dumps(json_content))

            # Create DSL workflow
            dsl_file = tmpdir / "workflow.dsl"
            dsl_content = """
            workflow dsl_workflow:
                node loader:
                    type: data
                    config:
                        source: "data.csv"

                node processor:
                    type: process
                    depends_on: loader
                    config:
                        method: "normalize"

                flow:
                    loader -> processor
            """
            dsl_file.write_text(dsl_content)

            yield tmpdir

    @pytest.mark.asyncio
    async def test_load_yaml_workflow(self, temp_workflow_files):
        """Test loading and executing YAML workflow."""
        parser = FlowParser()
        yaml_file = temp_workflow_files / "workflow.yaml"

        flow = parser.parse_file(str(yaml_file))

        assert flow.name == "yaml-workflow"
        assert len(flow.nodes) == 2

    @pytest.mark.asyncio
    async def test_load_json_workflow(self, temp_workflow_files):
        """Test loading and executing JSON workflow."""
        parser = FlowParser()
        json_file = temp_workflow_files / "workflow.json"

        flow = parser.parse_file(str(json_file))

        assert flow.name == "json-workflow"
        assert len(flow.nodes) == 1

    @pytest.mark.asyncio
    async def test_load_dsl_workflow(self, temp_workflow_files):
        """Test loading and executing DSL workflow."""
        parser = FlowParser()
        dsl_file = temp_workflow_files / "workflow.dsl"

        flow = parser.parse_file(str(dsl_file))

        assert flow.name == "dsl_workflow"
        assert len(flow.nodes) == 2


class TestPerformanceAndScaling:
    """Test performance and scaling characteristics."""

    @pytest.mark.asyncio
    async def test_large_workflow_execution(self):
        """Test execution of large workflow with many nodes."""
        engine = WorkflowEngine(max_parallel=10)

        # Register a simple executor
        async def simple_executor(inputs):
            await asyncio.sleep(0.01)  # Simulate minimal work
            return {"result": "done"}

        engine.node_registry.register("simple", simple_executor)

        # Create a large workflow
        num_nodes = 100
        nodes = []
        edges = []

        for i in range(num_nodes):
            node = NodeDefinition(
                id=f"node{i}",
                name=f"Node{i}",
                type=NodeType.PROCESS,
                executor="simple",
                dependencies=[f"node{i-1}"] if i > 0 else [],
                config={}
            )
            nodes.append(node)

            if i > 0:
                edges.append({"from": f"node{i-1}", "to": f"node{i}"})

        flow_def = FlowDefinition(
            name="large-flow",
            version="1.0.0",
            nodes=nodes,
            edges=edges
        )

        start_time = time.time()
        run = await engine.run_flow(flow_def, inputs={})
        execution_time = time.time() - start_time

        assert run.status == RunStatus.COMPLETED
        # Should complete in reasonable time despite many nodes
        assert execution_time < 5.0

    @pytest.mark.asyncio
    async def test_memory_efficiency(self):
        """Test memory efficiency with large data passing."""
        engine = WorkflowEngine()

        # Create executors that pass large data
        async def data_generator(inputs):
            # Generate large dataset
            return {"data": list(range(1000000))}

        async def data_processor(inputs):
            data = inputs.get("data", [])
            # Process without creating copies
            return {"result": len(data)}

        engine.node_registry.register("generator", data_generator)
        engine.node_registry.register("processor", data_processor)

        flow_def = FlowDefinition(
            name="memory-test",
            version="1.0.0",
            nodes=[
                NodeDefinition(
                    id="gen",
                    name="Generator",
                    type=NodeType.DATA,
                    executor="generator",
                    config={}
                ),
                NodeDefinition(
                    id="proc",
                    name="Processor",
                    type=NodeType.PROCESS,
                    executor="processor",
                    dependencies=["gen"],
                    config={}
                )
            ],
            edges=[{"from": "gen", "to": "proc"}]
        )

        run = await engine.run_flow(flow_def, inputs={})

        assert run.status == RunStatus.COMPLETED
        assert run.outputs["result"] == 1000000