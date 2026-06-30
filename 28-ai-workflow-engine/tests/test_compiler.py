"""Unit tests for the workflow compiler components."""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Optional imports requiring yaml
try:
    from aiworkflow.compiler import (
        FlowParser,
        FlowValidator,
        DAGBuilder,
        FlowOptimizer
    )
    from aiworkflow.compiler.parser import ParseError
    from aiworkflow.compiler.dag import CircularDependencyError
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

from aiworkflow.schemas import (
    FlowDefinition,
    Node as NodeDefinition,
    NodeType,
)

pytestmark = pytest.mark.skipif(not _HAS_YAML, reason="Requires yaml")


class TestFlowParser:
    """Test suite for FlowParser."""

    @pytest.fixture
    def parser(self):
        """Create a FlowParser instance."""
        return FlowParser()

    def test_parse_yaml_workflow(self, parser):
        """Test parsing YAML workflow definition."""
        yaml_content = """
        name: test-workflow
        version: 1.0.0
        description: Test workflow
        nodes:
          - id: node1
            name: DataLoader
            type: data
            config:
              source: test.csv
          - id: node2
            name: Processor
            type: process
            config:
              method: normalize
            dependencies: [node1]
        edges:
          - from: node1
            to: node2
        """

        result = parser.parse_yaml(yaml_content)

        assert result.name == "test-workflow"
        assert result.version == "1.0.0"
        assert len(result.nodes) == 2
        assert result.nodes[0].id == "node1"
        assert result.nodes[1].dependencies == ["node1"]

    def test_parse_json_workflow(self, parser):
        """Test parsing JSON workflow definition."""
        json_content = {
            "name": "test-workflow",
            "version": "1.0.0",
            "description": "Test workflow",
            "nodes": [
                {
                    "id": "node1",
                    "name": "DataLoader",
                    "type": "data",
                    "config": {"source": "test.csv"}
                }
            ],
            "edges": []
        }

        result = parser.parse_json(json_content)

        assert result.name == "test-workflow"
        assert len(result.nodes) == 1
        assert result.nodes[0].id == "node1"

    def test_parse_dsl_workflow(self, parser):
        """Test parsing DSL workflow definition."""
        dsl_content = """
        workflow test_pipeline:
            node data_loader:
                type: data
                config:
                    source: "database"
                    table: "users"

            node preprocessor:
                type: process
                depends_on: data_loader
                config:
                    operations: ["normalize", "encode"]

            node model_trainer:
                type: model
                depends_on: preprocessor
                config:
                    algorithm: "random_forest"
                    hyperparameters:
                        n_estimators: 100
                        max_depth: 10

            flow:
                data_loader -> preprocessor -> model_trainer
        """

        result = parser.parse_dsl(dsl_content)

        assert result.name == "test_pipeline"
        assert len(result.nodes) == 3
        assert result.nodes[0].name == "data_loader"
        assert result.nodes[1].dependencies == ["data_loader"]
        assert result.nodes[2].dependencies == ["preprocessor"]

    def test_parse_invalid_yaml(self, parser):
        """Test parsing invalid YAML content."""
        invalid_yaml = """
        name: test
        nodes
            - invalid syntax
        """

        with pytest.raises(ParseError):
            parser.parse_yaml(invalid_yaml)

    def test_parse_missing_required_fields(self, parser):
        """Test parsing with missing required fields."""
        incomplete_json = {
            "name": "test-workflow",
            # Missing nodes field
        }

        with pytest.raises(ParseError, match="Missing required field"):
            parser.parse_json(incomplete_json)


class TestFlowValidator:
    """Test suite for FlowValidator."""

    @pytest.fixture
    def validator(self):
        """Create a FlowValidator instance."""
        return FlowValidator()

    @pytest.fixture
    def valid_flow(self):
        """Create a valid flow definition."""
        return FlowDefinition(
            name="valid-flow",
            version="1.0.0",
            nodes=[
                NodeDefinition(
                    id="node1",
                    name="Node1",
                    type=NodeType.DATA,
                    config={"key": "value"}
                ),
                NodeDefinition(
                    id="node2",
                    name="Node2",
                    type=NodeType.PROCESS,
                    dependencies=["node1"],
                    config={}
                )
            ],
            edges=[{"from": "node1", "to": "node2"}]
        )

    def test_validate_valid_flow(self, validator, valid_flow):
        """Test validation of a valid flow."""
        result = validator.validate(valid_flow)
        assert result is True

    def test_validate_duplicate_node_ids(self, validator):
        """Test validation with duplicate node IDs."""
        invalid_flow = FlowDefinition(
            name="invalid-flow",
            version="1.0.0",
            nodes=[
                NodeDefinition(
                    id="node1",
                    name="Node1",
                    type=NodeType.DATA,
                    config={}
                ),
                NodeDefinition(
                    id="node1",  # Duplicate ID
                    name="Node2",
                    type=NodeType.PROCESS,
                    config={}
                )
            ],
            edges=[]
        )

        result = validator.validate(invalid_flow)
        assert result is False
        errors = validator.get_errors()
        assert any("duplicate" in str(e).lower() for e in errors)

    def test_validate_invalid_dependencies(self, validator):
        """Test validation with invalid node dependencies."""
        invalid_flow = FlowDefinition(
            name="invalid-flow",
            version="1.0.0",
            nodes=[
                NodeDefinition(
                    id="node1",
                    name="Node1",
                    type=NodeType.DATA,
                    dependencies=["non_existent_node"],  # Invalid dependency
                    config={}
                )
            ],
            edges=[]
        )

        result = validator.validate(invalid_flow)
        assert result is False
        errors = validator.get_errors()
        assert any("dependency" in str(e).lower() for e in errors)

    def test_validate_circular_dependencies(self, validator):
        """Test validation with circular dependencies."""
        circular_flow = FlowDefinition(
            name="circular-flow",
            version="1.0.0",
            nodes=[
                NodeDefinition(
                    id="node1",
                    name="Node1",
                    type=NodeType.PROCESS,
                    dependencies=["node2"],
                    config={}
                ),
                NodeDefinition(
                    id="node2",
                    name="Node2",
                    type=NodeType.PROCESS,
                    dependencies=["node1"],  # Circular dependency
                    config={}
                )
            ],
            edges=[
                {"from": "node1", "to": "node2"},
                {"from": "node2", "to": "node1"}
            ]
        )

        result = validator.validate(circular_flow)
        assert result is False
        errors = validator.get_errors()
        assert any("circular" in str(e).lower() for e in errors)

    def test_validate_empty_flow(self, validator):
        """Test validation of empty flow."""
        empty_flow = FlowDefinition(
            name="empty-flow",
            version="1.0.0",
            nodes=[],
            edges=[]
        )

        result = validator.validate(empty_flow)
        assert result is False
        errors = validator.get_errors()
        assert any("empty" in str(e).lower() for e in errors)


class TestDAGBuilder:
    """Test suite for DAGBuilder."""

    @pytest.fixture
    def builder(self):
        """Create a DAGBuilder instance."""
        return DAGBuilder()

    def test_build_simple_dag(self, builder):
        """Test building a simple DAG."""
        flow = FlowDefinition(
            name="simple-dag",
            version="1.0.0",
            nodes=[
                NodeDefinition(id="a", name="A", type=NodeType.DATA, config={}),
                NodeDefinition(id="b", name="B", type=NodeType.PROCESS,
                              dependencies=["a"], config={}),
                NodeDefinition(id="c", name="C", type=NodeType.MODEL,
                              dependencies=["b"], config={})
            ],
            edges=[
                {"from": "a", "to": "b"},
                {"from": "b", "to": "c"}
            ]
        )

        dag = builder.build(flow)

        assert dag.get_node("a") is not None
        assert dag.get_node("b") is not None
        assert dag.get_node("c") is not None
        assert dag.get_dependencies("b") == ["a"]
        assert dag.get_dependencies("c") == ["b"]

    def test_build_parallel_dag(self, builder):
        """Test building DAG with parallel branches."""
        flow = FlowDefinition(
            name="parallel-dag",
            version="1.0.0",
            nodes=[
                NodeDefinition(id="start", name="Start", type=NodeType.DATA, config={}),
                NodeDefinition(id="branch1", name="Branch1", type=NodeType.PROCESS,
                              dependencies=["start"], config={}),
                NodeDefinition(id="branch2", name="Branch2", type=NodeType.PROCESS,
                              dependencies=["start"], config={}),
                NodeDefinition(id="merge", name="Merge", type=NodeType.PROCESS,
                              dependencies=["branch1", "branch2"], config={})
            ],
            edges=[
                {"from": "start", "to": "branch1"},
                {"from": "start", "to": "branch2"},
                {"from": "branch1", "to": "merge"},
                {"from": "branch2", "to": "merge"}
            ]
        )

        dag = builder.build(flow)

        assert dag.get_dependencies("branch1") == ["start"]
        assert dag.get_dependencies("branch2") == ["start"]
        assert set(dag.get_dependencies("merge")) == {"branch1", "branch2"}

    def test_topological_sort(self, builder):
        """Test topological sorting of DAG nodes."""
        flow = FlowDefinition(
            name="topo-sort",
            version="1.0.0",
            nodes=[
                NodeDefinition(id="a", name="A", type=NodeType.DATA, config={}),
                NodeDefinition(id="b", name="B", type=NodeType.PROCESS,
                              dependencies=["a"], config={}),
                NodeDefinition(id="c", name="C", type=NodeType.PROCESS,
                              dependencies=["a"], config={}),
                NodeDefinition(id="d", name="D", type=NodeType.MODEL,
                              dependencies=["b", "c"], config={})
            ],
            edges=[
                {"from": "a", "to": "b"},
                {"from": "a", "to": "c"},
                {"from": "b", "to": "d"},
                {"from": "c", "to": "d"}
            ]
        )

        dag = builder.build(flow)
        sorted_nodes = dag.topological_sort()

        # Check that dependencies come before dependents
        node_positions = {node: i for i, node in enumerate(sorted_nodes)}
        assert node_positions["a"] < node_positions["b"]
        assert node_positions["a"] < node_positions["c"]
        assert node_positions["b"] < node_positions["d"]
        assert node_positions["c"] < node_positions["d"]

    def test_detect_circular_dependency(self, builder):
        """Test detection of circular dependencies."""
        flow = FlowDefinition(
            name="circular",
            version="1.0.0",
            nodes=[
                NodeDefinition(id="a", name="A", type=NodeType.PROCESS,
                              dependencies=["c"], config={}),
                NodeDefinition(id="b", name="B", type=NodeType.PROCESS,
                              dependencies=["a"], config={}),
                NodeDefinition(id="c", name="C", type=NodeType.PROCESS,
                              dependencies=["b"], config={})
            ],
            edges=[
                {"from": "c", "to": "a"},
                {"from": "a", "to": "b"},
                {"from": "b", "to": "c"}
            ]
        )

        with pytest.raises(CircularDependencyError):
            builder.build(flow)


class TestFlowOptimizer:
    """Test suite for FlowOptimizer."""

    @pytest.fixture
    def optimizer(self):
        """Create a FlowOptimizer instance."""
        return FlowOptimizer()

    def test_optimize_redundant_nodes(self, optimizer):
        """Test optimization removing redundant nodes."""
        flow = FlowDefinition(
            name="redundant-flow",
            version="1.0.0",
            nodes=[
                NodeDefinition(id="load1", name="Load1", type=NodeType.DATA,
                              config={"source": "same.csv"}),
                NodeDefinition(id="load2", name="Load2", type=NodeType.DATA,
                              config={"source": "same.csv"}),  # Redundant
                NodeDefinition(id="process", name="Process", type=NodeType.PROCESS,
                              dependencies=["load1", "load2"], config={})
            ],
            edges=[
                {"from": "load1", "to": "process"},
                {"from": "load2", "to": "process"}
            ]
        )

        optimized = optimizer.optimize(flow)

        # Should merge redundant data loading nodes
        assert len(optimized.nodes) < len(flow.nodes)

    def test_optimize_parallel_execution(self, optimizer):
        """Test optimization for parallel execution opportunities."""
        flow = FlowDefinition(
            name="sequential-flow",
            version="1.0.0",
            nodes=[
                NodeDefinition(id="load", name="Load", type=NodeType.DATA, config={}),
                NodeDefinition(id="proc1", name="Process1", type=NodeType.PROCESS,
                              dependencies=["load"], config={"independent": True}),
                NodeDefinition(id="proc2", name="Process2", type=NodeType.PROCESS,
                              dependencies=["load"], config={"independent": True}),
                NodeDefinition(id="merge", name="Merge", type=NodeType.PROCESS,
                              dependencies=["proc1", "proc2"], config={})
            ],
            edges=[
                {"from": "load", "to": "proc1"},
                {"from": "load", "to": "proc2"},
                {"from": "proc1", "to": "merge"},
                {"from": "proc2", "to": "merge"}
            ]
        )

        optimized = optimizer.optimize(flow)

        # Should identify parallel execution opportunities
        assert optimized.metadata.get("parallel_groups") is not None
        parallel_groups = optimized.metadata["parallel_groups"]
        assert any(set(["proc1", "proc2"]).issubset(set(group))
                  for group in parallel_groups)

    def test_optimize_cache_opportunities(self, optimizer):
        """Test optimization identifying caching opportunities."""
        flow = FlowDefinition(
            name="cacheable-flow",
            version="1.0.0",
            nodes=[
                NodeDefinition(id="expensive", name="ExpensiveOp",
                              type=NodeType.PROCESS,
                              config={"deterministic": True, "cost": "high"}),
                NodeDefinition(id="consumer1", name="Consumer1",
                              type=NodeType.PROCESS,
                              dependencies=["expensive"], config={}),
                NodeDefinition(id="consumer2", name="Consumer2",
                              type=NodeType.PROCESS,
                              dependencies=["expensive"], config={})
            ],
            edges=[
                {"from": "expensive", "to": "consumer1"},
                {"from": "expensive", "to": "consumer2"}
            ]
        )

        optimized = optimizer.optimize(flow)

        # Should mark expensive deterministic nodes as cacheable
        expensive_node = next(n for n in optimized.nodes if n.id == "expensive")
        assert expensive_node.metadata.get("cache_enabled", False) is True

    def test_optimize_empty_flow(self, optimizer):
        """Test optimization of empty flow."""
        empty_flow = FlowDefinition(
            name="empty",
            version="1.0.0",
            nodes=[],
            edges=[]
        )

        optimized = optimizer.optimize(empty_flow)

        assert optimized == empty_flow

    def test_optimize_already_optimal(self, optimizer):
        """Test optimization of already optimal flow."""
        optimal_flow = FlowDefinition(
            name="optimal",
            version="1.0.0",
            nodes=[
                NodeDefinition(id="a", name="A", type=NodeType.DATA, config={}),
                NodeDefinition(id="b", name="B", type=NodeType.PROCESS,
                              dependencies=["a"], config={})
            ],
            edges=[{"from": "a", "to": "b"}]
        )

        optimized = optimizer.optimize(optimal_flow)

        # Should not change an already optimal flow
        assert len(optimized.nodes) == len(optimal_flow.nodes)
        assert optimized.edges == optimal_flow.edges