"""Integration tests for DAG compiler edge cases.

Tests cover:
- Empty and single-node DAGs
- Deep linear chains
- Wide parallel DAGs (fan-out/fan-in)
- Diamond dependencies
- Complex multi-path dependencies
- Disconnected subgraphs
- Self-referential node handling
- Large-scale DAG performance
"""

import pytest
import asyncio
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aiworkflow.schemas import (
    FlowDefinition,
    Node,
    NodeType,
    NodeConfig,
    Edge,
)

# Optional imports requiring yaml
try:
    from aiworkflow.compiler.dag import DAGBuilder, FlowOptimizer
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

pytestmark = pytest.mark.skipif(not _HAS_YAML, reason="Requires yaml")


class TestEmptyAndMinimalDAGs:
    """Test edge cases for empty and minimal DAG structures."""

    def test_empty_dag(self):
        """Test building DAG with no nodes."""
        flow = FlowDefinition(
            name="empty-flow",
            version="1.0",
            nodes=[],
            edges=[]
        )

        dag = DAGBuilder(flow)

        assert dag.topological_sort() == []
        assert dag.get_execution_levels() == []

    def test_single_node_dag(self):
        """Test DAG with single node (no dependencies)."""
        flow = FlowDefinition(
            name="single-node-flow",
            version="1.0",
            nodes=[
                Node(
                    id="only_node",
                    type=NodeType.LLM,
                    config=NodeConfig()
                )
            ],
            edges=[]
        )

        dag = DAGBuilder(flow)
        topo_order = dag.topological_sort()
        levels = dag.get_execution_levels()

        assert topo_order == ["only_node"]
        assert levels == [["only_node"]]
        assert dag.get_dependencies("only_node") == set()
        assert dag.get_dependents("only_node") == set()

    def test_two_disconnected_nodes(self):
        """Test DAG with two nodes that have no connection."""
        flow = FlowDefinition(
            name="disconnected-flow",
            version="1.0",
            nodes=[
                Node(id="node_a", type=NodeType.LLM, config=NodeConfig()),
                Node(id="node_b", type=NodeType.LLM, config=NodeConfig()),
            ],
            edges=[]
        )

        dag = DAGBuilder(flow)
        levels = dag.get_execution_levels()

        # Both nodes should be in the first level (can run in parallel)
        assert len(levels) == 1
        assert set(levels[0]) == {"node_a", "node_b"}


class TestLinearChainDAGs:
    """Test linear chain DAG structures."""

    def test_simple_linear_chain(self):
        """Test simple A -> B -> C chain."""
        flow = FlowDefinition(
            name="linear-chain",
            version="1.0",
            nodes=[
                Node(id="a", type=NodeType.LLM, config=NodeConfig()),
                Node(id="b", type=NodeType.LLM, config=NodeConfig(), dependencies=["a"]),
                Node(id="c", type=NodeType.LLM, config=NodeConfig(), dependencies=["b"]),
            ],
            edges=[
                Edge(from_node="a", to_node="b"),
                Edge(from_node="b", to_node="c"),
            ]
        )

        dag = DAGBuilder(flow)
        topo_order = dag.topological_sort()
        levels = dag.get_execution_levels()

        # Check ordering
        assert topo_order.index("a") < topo_order.index("b")
        assert topo_order.index("b") < topo_order.index("c")

        # Each node should be in its own level
        assert len(levels) == 3
        assert levels[0] == ["a"]
        assert levels[1] == ["b"]
        assert levels[2] == ["c"]

    def test_deep_linear_chain(self):
        """Test deep linear chain (100 nodes)."""
        num_nodes = 100
        nodes = []
        edges = []

        for i in range(num_nodes):
            deps = [f"node_{i-1}"] if i > 0 else []
            nodes.append(Node(
                id=f"node_{i}",
                type=NodeType.LLM,
                config=NodeConfig(),
                dependencies=deps
            ))
            if i > 0:
                edges.append(Edge(from_node=f"node_{i-1}", to_node=f"node_{i}"))

        flow = FlowDefinition(
            name="deep-chain",
            version="1.0",
            nodes=nodes,
            edges=edges
        )

        dag = DAGBuilder(flow)
        topo_order = dag.topological_sort()
        levels = dag.get_execution_levels()

        # Should have 100 levels
        assert len(levels) == num_nodes

        # Verify ordering
        for i in range(num_nodes - 1):
            assert topo_order.index(f"node_{i}") < topo_order.index(f"node_{i+1}")


class TestParallelAndFanOutDAGs:
    """Test fan-out and fan-in DAG patterns."""

    def test_fan_out_pattern(self):
        """Test single node fanning out to multiple nodes."""
        flow = FlowDefinition(
            name="fan-out",
            version="1.0",
            nodes=[
                Node(id="source", type=NodeType.LLM, config=NodeConfig()),
                Node(id="branch_a", type=NodeType.LLM, config=NodeConfig(), dependencies=["source"]),
                Node(id="branch_b", type=NodeType.LLM, config=NodeConfig(), dependencies=["source"]),
                Node(id="branch_c", type=NodeType.LLM, config=NodeConfig(), dependencies=["source"]),
            ],
            edges=[
                Edge(from_node="source", to_node="branch_a"),
                Edge(from_node="source", to_node="branch_b"),
                Edge(from_node="source", to_node="branch_c"),
            ]
        )

        dag = DAGBuilder(flow)
        levels = dag.get_execution_levels()

        # First level: source, Second level: all branches
        assert len(levels) == 2
        assert levels[0] == ["source"]
        assert set(levels[1]) == {"branch_a", "branch_b", "branch_c"}

        # Check dependents
        assert dag.get_dependents("source") == {"branch_a", "branch_b", "branch_c"}

    def test_fan_in_pattern(self):
        """Test multiple nodes converging to a single node."""
        flow = FlowDefinition(
            name="fan-in",
            version="1.0",
            nodes=[
                Node(id="source_a", type=NodeType.LLM, config=NodeConfig()),
                Node(id="source_b", type=NodeType.LLM, config=NodeConfig()),
                Node(id="source_c", type=NodeType.LLM, config=NodeConfig()),
                Node(id="merge", type=NodeType.LLM, config=NodeConfig(),
                     dependencies=["source_a", "source_b", "source_c"]),
            ],
            edges=[
                Edge(from_node="source_a", to_node="merge"),
                Edge(from_node="source_b", to_node="merge"),
                Edge(from_node="source_c", to_node="merge"),
            ]
        )

        dag = DAGBuilder(flow)
        levels = dag.get_execution_levels()

        # First level: all sources, Second level: merge
        assert len(levels) == 2
        assert set(levels[0]) == {"source_a", "source_b", "source_c"}
        assert levels[1] == ["merge"]

        # Check dependencies
        assert dag.get_dependencies("merge") == {"source_a", "source_b", "source_c"}

    def test_wide_parallel_dag(self):
        """Test wide parallel DAG with many independent branches."""
        num_branches = 50
        nodes = [Node(id="source", type=NodeType.LLM, config=NodeConfig())]
        edges = []

        for i in range(num_branches):
            nodes.append(Node(
                id=f"branch_{i}",
                type=NodeType.LLM,
                config=NodeConfig(),
                dependencies=["source"]
            ))
            edges.append(Edge(from_node="source", to_node=f"branch_{i}"))

        # Add merge node
        nodes.append(Node(
            id="merge",
            type=NodeType.LLM,
            config=NodeConfig(),
            dependencies=[f"branch_{i}" for i in range(num_branches)]
        ))
        for i in range(num_branches):
            edges.append(Edge(from_node=f"branch_{i}", to_node="merge"))

        flow = FlowDefinition(
            name="wide-parallel",
            version="1.0",
            nodes=nodes,
            edges=edges
        )

        dag = DAGBuilder(flow)
        levels = dag.get_execution_levels()

        # Should have 3 levels: source, all branches, merge
        assert len(levels) == 3
        assert len(levels[1]) == num_branches

        # All branches can execute in parallel
        parallel_groups = dag.get_parallel_groups()
        assert len(parallel_groups[1]) == num_branches


class TestDiamondDependencies:
    """Test diamond dependency patterns."""

    def test_simple_diamond(self):
        """Test classic diamond pattern: A -> B,C -> D."""
        flow = FlowDefinition(
            name="diamond",
            version="1.0",
            nodes=[
                Node(id="top", type=NodeType.LLM, config=NodeConfig()),
                Node(id="left", type=NodeType.LLM, config=NodeConfig(), dependencies=["top"]),
                Node(id="right", type=NodeType.LLM, config=NodeConfig(), dependencies=["top"]),
                Node(id="bottom", type=NodeType.LLM, config=NodeConfig(), dependencies=["left", "right"]),
            ],
            edges=[
                Edge(from_node="top", to_node="left"),
                Edge(from_node="top", to_node="right"),
                Edge(from_node="left", to_node="bottom"),
                Edge(from_node="right", to_node="bottom"),
            ]
        )

        dag = DAGBuilder(flow)
        topo_order = dag.topological_sort()
        levels = dag.get_execution_levels()

        # Verify ordering constraints
        assert topo_order.index("top") < topo_order.index("left")
        assert topo_order.index("top") < topo_order.index("right")
        assert topo_order.index("left") < topo_order.index("bottom")
        assert topo_order.index("right") < topo_order.index("bottom")

        # Should have 3 levels
        assert len(levels) == 3
        assert levels[0] == ["top"]
        assert set(levels[1]) == {"left", "right"}
        assert levels[2] == ["bottom"]

    def test_nested_diamond(self):
        """Test nested diamond dependencies."""
        #       A
        #      / \
        #     B   C
        #    / \ / \
        #   D   E   F
        #    \ | /
        #      G
        flow = FlowDefinition(
            name="nested-diamond",
            version="1.0",
            nodes=[
                Node(id="a", type=NodeType.LLM, config=NodeConfig()),
                Node(id="b", type=NodeType.LLM, config=NodeConfig(), dependencies=["a"]),
                Node(id="c", type=NodeType.LLM, config=NodeConfig(), dependencies=["a"]),
                Node(id="d", type=NodeType.LLM, config=NodeConfig(), dependencies=["b"]),
                Node(id="e", type=NodeType.LLM, config=NodeConfig(), dependencies=["b", "c"]),
                Node(id="f", type=NodeType.LLM, config=NodeConfig(), dependencies=["c"]),
                Node(id="g", type=NodeType.LLM, config=NodeConfig(), dependencies=["d", "e", "f"]),
            ],
            edges=[
                Edge(from_node="a", to_node="b"),
                Edge(from_node="a", to_node="c"),
                Edge(from_node="b", to_node="d"),
                Edge(from_node="b", to_node="e"),
                Edge(from_node="c", to_node="e"),
                Edge(from_node="c", to_node="f"),
                Edge(from_node="d", to_node="g"),
                Edge(from_node="e", to_node="g"),
                Edge(from_node="f", to_node="g"),
            ]
        )

        dag = DAGBuilder(flow)
        levels = dag.get_execution_levels()

        # Should have 4 levels
        assert len(levels) == 4
        assert levels[0] == ["a"]
        assert set(levels[1]) == {"b", "c"}
        assert set(levels[2]) == {"d", "e", "f"}
        assert levels[3] == ["g"]


class TestAncestorQueries:
    """Test ancestor and transitive dependency queries."""

    def test_get_ancestors_linear(self):
        """Test getting ancestors in linear chain."""
        flow = FlowDefinition(
            name="linear",
            version="1.0",
            nodes=[
                Node(id="a", type=NodeType.LLM, config=NodeConfig()),
                Node(id="b", type=NodeType.LLM, config=NodeConfig(), dependencies=["a"]),
                Node(id="c", type=NodeType.LLM, config=NodeConfig(), dependencies=["b"]),
                Node(id="d", type=NodeType.LLM, config=NodeConfig(), dependencies=["c"]),
            ],
            edges=[
                Edge(from_node="a", to_node="b"),
                Edge(from_node="b", to_node="c"),
                Edge(from_node="c", to_node="d"),
            ]
        )

        dag = DAGBuilder(flow)

        assert dag.get_ancestors("a") == set()
        assert dag.get_ancestors("b") == {"a"}
        assert dag.get_ancestors("c") == {"a", "b"}
        assert dag.get_ancestors("d") == {"a", "b", "c"}

    def test_get_ancestors_diamond(self):
        """Test getting ancestors in diamond pattern."""
        flow = FlowDefinition(
            name="diamond",
            version="1.0",
            nodes=[
                Node(id="top", type=NodeType.LLM, config=NodeConfig()),
                Node(id="left", type=NodeType.LLM, config=NodeConfig(), dependencies=["top"]),
                Node(id="right", type=NodeType.LLM, config=NodeConfig(), dependencies=["top"]),
                Node(id="bottom", type=NodeType.LLM, config=NodeConfig(), dependencies=["left", "right"]),
            ],
            edges=[
                Edge(from_node="top", to_node="left"),
                Edge(from_node="top", to_node="right"),
                Edge(from_node="left", to_node="bottom"),
                Edge(from_node="right", to_node="bottom"),
            ]
        )

        dag = DAGBuilder(flow)

        assert dag.get_ancestors("top") == set()
        assert dag.get_ancestors("left") == {"top"}
        assert dag.get_ancestors("right") == {"top"}
        assert dag.get_ancestors("bottom") == {"top", "left", "right"}


class TestExecutionReadiness:
    """Test can_execute functionality."""

    def test_can_execute_no_deps(self):
        """Test that nodes with no dependencies can execute immediately."""
        flow = FlowDefinition(
            name="test",
            version="1.0",
            nodes=[
                Node(id="a", type=NodeType.LLM, config=NodeConfig()),
                Node(id="b", type=NodeType.LLM, config=NodeConfig(), dependencies=["a"]),
            ],
            edges=[Edge(from_node="a", to_node="b")]
        )

        dag = DAGBuilder(flow)

        # a can execute with empty completed set
        assert dag.can_execute("a", set()) is True
        # b cannot execute until a is completed
        assert dag.can_execute("b", set()) is False
        assert dag.can_execute("b", {"a"}) is True

    def test_can_execute_multiple_deps(self):
        """Test execution readiness with multiple dependencies."""
        flow = FlowDefinition(
            name="test",
            version="1.0",
            nodes=[
                Node(id="a", type=NodeType.LLM, config=NodeConfig()),
                Node(id="b", type=NodeType.LLM, config=NodeConfig()),
                Node(id="c", type=NodeType.LLM, config=NodeConfig(), dependencies=["a", "b"]),
            ],
            edges=[
                Edge(from_node="a", to_node="c"),
                Edge(from_node="b", to_node="c"),
            ]
        )

        dag = DAGBuilder(flow)

        # c needs both a and b
        assert dag.can_execute("c", set()) is False
        assert dag.can_execute("c", {"a"}) is False
        assert dag.can_execute("c", {"b"}) is False
        assert dag.can_execute("c", {"a", "b"}) is True


class TestFlowOptimizer:
    """Test the flow optimizer edge cases."""

    def test_optimize_empty_flow(self):
        """Test optimizing empty flow."""
        flow = FlowDefinition(name="empty", version="1.0", nodes=[], edges=[])

        optimizer = FlowOptimizer()
        optimized = optimizer.optimize(flow)

        assert optimized.nodes == []
        assert optimized.edges == []

    def test_optimize_single_node(self):
        """Test optimizing single node flow."""
        flow = FlowDefinition(
            name="single",
            version="1.0",
            nodes=[Node(id="only", type=NodeType.LLM, config=NodeConfig())],
            edges=[]
        )

        optimizer = FlowOptimizer()
        optimized = optimizer.optimize(flow)

        assert len(optimized.nodes) == 1
        assert optimized.nodes[0].id == "only"

    def test_optimize_parallel_levels(self):
        """Test that optimizer identifies parallel execution levels."""
        flow = FlowDefinition(
            name="parallel",
            version="1.0",
            nodes=[
                Node(id="source", type=NodeType.LLM, config=NodeConfig()),
                Node(id="branch_a", type=NodeType.LLM, config=NodeConfig(), dependencies=["source"]),
                Node(id="branch_b", type=NodeType.LLM, config=NodeConfig(), dependencies=["source"]),
                Node(id="merge", type=NodeType.LLM, config=NodeConfig(), dependencies=["branch_a", "branch_b"]),
            ],
            edges=[
                Edge(from_node="source", to_node="branch_a"),
                Edge(from_node="source", to_node="branch_b"),
                Edge(from_node="branch_a", to_node="merge"),
                Edge(from_node="branch_b", to_node="merge"),
            ]
        )

        optimizer = FlowOptimizer()
        optimized = optimizer.optimize(flow)

        # Optimizer should mark parallel levels in metadata
        node_map = {n.id: n for n in optimized.nodes}

        # branch_a and branch_b should have the same parallel level
        assert node_map["branch_a"].metadata.get("parallel_level") == node_map["branch_b"].metadata.get("parallel_level")


class TestDAGPerformance:
    """Performance tests for DAG operations."""

    def test_large_dag_build_performance(self):
        """Test building large DAG completes in reasonable time."""
        num_nodes = 1000
        nodes = []
        edges = []

        # Create a tree structure
        for i in range(num_nodes):
            parent_idx = (i - 1) // 2 if i > 0 else None
            deps = [f"node_{parent_idx}"] if parent_idx is not None else []
            nodes.append(Node(
                id=f"node_{i}",
                type=NodeType.LLM,
                config=NodeConfig(),
                dependencies=deps
            ))
            if parent_idx is not None:
                edges.append(Edge(from_node=f"node_{parent_idx}", to_node=f"node_{i}"))

        flow = FlowDefinition(
            name="large-tree",
            version="1.0",
            nodes=nodes,
            edges=edges
        )

        start_time = time.time()
        dag = DAGBuilder(flow)
        topo_order = dag.topological_sort()
        levels = dag.get_execution_levels()
        elapsed = time.time() - start_time

        assert elapsed < 1.0  # Should complete in under 1 second
        assert len(topo_order) == num_nodes
        assert len(levels) > 0

    def test_topological_sort_performance(self):
        """Test topological sort performance on complex graph."""
        num_layers = 10
        nodes_per_layer = 100
        nodes = []
        edges = []

        for layer in range(num_layers):
            for i in range(nodes_per_layer):
                node_id = f"layer{layer}_node{i}"
                if layer == 0:
                    deps = []
                else:
                    # Connect to random nodes in previous layer
                    deps = [f"layer{layer-1}_node{j}" for j in range(nodes_per_layer)]

                nodes.append(Node(
                    id=node_id,
                    type=NodeType.LLM,
                    config=NodeConfig(),
                    dependencies=deps[:5]  # Limit deps to avoid explosion
                ))

                for dep in deps[:5]:
                    edges.append(Edge(from_node=dep, to_node=node_id))

        flow = FlowDefinition(
            name="layered",
            version="1.0",
            nodes=nodes,
            edges=edges
        )

        start_time = time.time()
        dag = DAGBuilder(flow)
        topo_order = dag.topological_sort()
        elapsed = time.time() - start_time

        assert elapsed < 2.0  # Should complete in under 2 seconds
        assert len(topo_order) == num_layers * nodes_per_layer


class TestEdgeCaseNodeReferences:
    """Test edge cases with node references."""

    def test_duplicate_edges_handling(self):
        """Test handling of duplicate edges."""
        flow = FlowDefinition(
            name="dup-edges",
            version="1.0",
            nodes=[
                Node(id="a", type=NodeType.LLM, config=NodeConfig()),
                Node(id="b", type=NodeType.LLM, config=NodeConfig(), dependencies=["a"]),
            ],
            edges=[
                Edge(from_node="a", to_node="b"),
                Edge(from_node="a", to_node="b"),  # Duplicate
            ]
        )

        dag = DAGBuilder(flow)

        # Should still work correctly despite duplicate edge
        assert dag.get_dependencies("b") == {"a"}
        assert dag.get_dependents("a") == {"b"}

    def test_both_deps_and_edges(self):
        """Test flow with both dependencies and explicit edges."""
        flow = FlowDefinition(
            name="mixed",
            version="1.0",
            nodes=[
                Node(id="a", type=NodeType.LLM, config=NodeConfig()),
                Node(id="b", type=NodeType.LLM, config=NodeConfig(), dependencies=["a"]),
                Node(id="c", type=NodeType.LLM, config=NodeConfig()),
            ],
            edges=[
                # Explicit edge for c (no dependencies attribute)
                Edge(from_node="a", to_node="c"),
            ]
        )

        dag = DAGBuilder(flow)

        # Both b and c depend on a
        assert dag.get_dependents("a") == {"b", "c"}
        assert dag.get_dependencies("b") == {"a"}
        assert dag.get_dependencies("c") == {"a"}
