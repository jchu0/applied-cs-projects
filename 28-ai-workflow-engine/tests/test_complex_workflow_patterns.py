"""Integration tests for complex workflow patterns.

Tests cover:
- Parallel execution patterns
- Conditional branching workflows
- Loop-like patterns (iterative workflows)
- Nested subflow patterns
- Mixed pattern combinations
- Dynamic workflow modifications
"""

import pytest
import asyncio
import time
import sys
import os
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aiworkflow.schemas import (
    FlowDefinition,
    Node,
    NodeType,
    NodeConfig,
    Edge,
    FlowRun,
    RunStatus,
    NodeExecution,
    NodeStatus,
)

# Optional imports requiring yaml
try:
    from aiworkflow.compiler.dag import DAGBuilder
    from aiworkflow.executor.scheduler import Scheduler, AsyncScheduler
    from aiworkflow.nodes.base import (
        MockNodeExecutor,
        BranchNodeExecutor,
        TransformNodeExecutor,
        NodeExecutorRegistry,
    )
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

pytestmark = pytest.mark.skipif(not _HAS_YAML, reason="Requires yaml")


class TestParallelExecutionPatterns:
    """Test parallel execution workflow patterns."""

    @pytest.fixture
    def mock_executor(self):
        """Create a mock node executor that tracks execution."""
        execution_log = []

        class TrackingExecutor:
            async def execute(self, node, inputs):
                start = time.time()
                await asyncio.sleep(0.05)  # Simulate work
                end = time.time()
                execution_log.append({
                    "node_id": node.id,
                    "start": start,
                    "end": end,
                    "inputs": inputs
                })
                return {"result": f"output_{node.id}", "processed": True}

        executor = TrackingExecutor()
        executor.execution_log = execution_log
        return executor

    @pytest.mark.asyncio
    async def test_parallel_branches_execute_concurrently(self, mock_executor):
        """Test that parallel branches execute concurrently."""
        flow = FlowDefinition(
            name="parallel-test",
            version="1.0",
            nodes=[
                Node(id="source", type=NodeType.LLM, config=NodeConfig()),
                Node(id="branch_a", type=NodeType.LLM, config=NodeConfig(), dependencies=["source"]),
                Node(id="branch_b", type=NodeType.LLM, config=NodeConfig(), dependencies=["source"]),
                Node(id="branch_c", type=NodeType.LLM, config=NodeConfig(), dependencies=["source"]),
                Node(id="merge", type=NodeType.LLM, config=NodeConfig(),
                     dependencies=["branch_a", "branch_b", "branch_c"]),
            ],
            edges=[
                Edge(from_node="source", to_node="branch_a"),
                Edge(from_node="source", to_node="branch_b"),
                Edge(from_node="source", to_node="branch_c"),
                Edge(from_node="branch_a", to_node="merge"),
                Edge(from_node="branch_b", to_node="merge"),
                Edge(from_node="branch_c", to_node="merge"),
            ]
        )

        scheduler = Scheduler(mock_executor, max_parallel=10)
        start_time = time.time()
        result = await scheduler.execute_flow(flow, inputs={"data": "test"})
        total_time = time.time() - start_time

        assert result.status == RunStatus.COMPLETED

        # Verify parallel execution: total time should be ~3 phases
        # (source) + (branches in parallel) + (merge)
        # Each phase ~0.05s, so total should be ~0.15s, not 0.25s if sequential
        assert total_time < 0.3  # Should be significantly less than sequential

        # Verify branches executed in parallel by checking overlapping times
        branch_execs = [e for e in mock_executor.execution_log if e["node_id"].startswith("branch")]
        assert len(branch_execs) == 3

        # Check that branch executions overlap
        if len(branch_execs) >= 2:
            first = branch_execs[0]
            for other in branch_execs[1:]:
                # If parallel, there should be time overlap
                overlaps = first["start"] < other["end"] and other["start"] < first["end"]
                assert overlaps, "Branches should execute in parallel"

    @pytest.mark.asyncio
    async def test_parallel_with_semaphore_limit(self, mock_executor):
        """Test parallel execution respects semaphore limit."""
        # Create flow with 10 parallel nodes but limit to 2 concurrent
        nodes = [Node(id="source", type=NodeType.LLM, config=NodeConfig())]
        edges = []

        for i in range(10):
            nodes.append(Node(
                id=f"worker_{i}",
                type=NodeType.LLM,
                config=NodeConfig(),
                dependencies=["source"]
            ))
            edges.append(Edge(from_node="source", to_node=f"worker_{i}"))

        flow = FlowDefinition(
            name="semaphore-test",
            version="1.0",
            nodes=nodes,
            edges=edges
        )

        scheduler = Scheduler(mock_executor, max_parallel=2)
        result = await scheduler.execute_flow(flow, inputs={})

        assert result.status == RunStatus.COMPLETED

        # Verify never more than 2 concurrent executions in workers
        worker_execs = [e for e in mock_executor.execution_log if e["node_id"].startswith("worker")]

        max_concurrent = 0
        for exec1 in worker_execs:
            concurrent = 1
            for exec2 in worker_execs:
                if exec1 is not exec2:
                    # Check if overlapping
                    if exec1["start"] < exec2["end"] and exec2["start"] < exec1["end"]:
                        concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)

        assert max_concurrent <= 2, f"Max concurrent was {max_concurrent}, expected <= 2"


class TestConditionalWorkflowPatterns:
    """Test conditional branching workflow patterns."""

    @pytest.fixture
    def condition_executor(self):
        """Create executor that handles conditional logic."""
        class ConditionalExecutor:
            async def execute(self, node, inputs):
                if node.type == NodeType.BRANCH:
                    # Evaluate condition
                    condition = node.config.extra.get("condition", "")
                    branches = node.config.extra.get("branches", {})
                    default = node.config.extra.get("default")

                    # Simple condition evaluation
                    value = inputs.get("value")
                    if value is None:
                        value = inputs.get("condition_value")

                    if isinstance(value, str):
                        target = branches.get(value, default)
                    elif isinstance(value, bool):
                        target = branches.get("true" if value else "false", default)
                    elif isinstance(value, (int, float)):
                        if value > 0.8:
                            target = branches.get("high", default)
                        else:
                            target = branches.get("low", default)
                    else:
                        target = default

                    return {"selected_branch": target, "condition_value": value}
                else:
                    return {"result": f"processed_{node.id}", "input": inputs}

        return ConditionalExecutor()

    @pytest.mark.asyncio
    async def test_simple_conditional_branching(self, condition_executor):
        """Test simple if-else conditional branching."""
        flow = FlowDefinition(
            name="conditional-test",
            version="1.0",
            nodes=[
                Node(id="input", type=NodeType.LLM, config=NodeConfig()),
                Node(
                    id="condition",
                    type=NodeType.BRANCH,
                    config=NodeConfig(extra={
                        "condition": "{{value}}",
                        "branches": {"high": "high_path", "low": "low_path"},
                        "default": "low_path"
                    }),
                    dependencies=["input"]
                ),
                Node(id="high_path", type=NodeType.LLM, config=NodeConfig(), dependencies=["condition"]),
                Node(id="low_path", type=NodeType.LLM, config=NodeConfig(), dependencies=["condition"]),
            ],
            edges=[
                Edge(from_node="input", to_node="condition"),
                Edge(from_node="condition", to_node="high_path"),
                Edge(from_node="condition", to_node="low_path"),
            ]
        )

        scheduler = Scheduler(condition_executor, max_parallel=5)

        # Test high value path
        result_high = await scheduler.execute_flow(flow, inputs={"value": 0.9})
        assert result_high.status == RunStatus.COMPLETED

        # Test low value path
        result_low = await scheduler.execute_flow(flow, inputs={"value": 0.3})
        assert result_low.status == RunStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_multi_way_conditional(self, condition_executor):
        """Test multi-way switch/case conditional branching."""
        flow = FlowDefinition(
            name="multi-way-test",
            version="1.0",
            nodes=[
                Node(id="router", type=NodeType.BRANCH, config=NodeConfig(extra={
                    "condition": "{{category}}",
                    "branches": {
                        "billing": "billing_handler",
                        "technical": "technical_handler",
                        "general": "general_handler",
                    },
                    "default": "general_handler"
                })),
                Node(id="billing_handler", type=NodeType.LLM, config=NodeConfig(), dependencies=["router"]),
                Node(id="technical_handler", type=NodeType.LLM, config=NodeConfig(), dependencies=["router"]),
                Node(id="general_handler", type=NodeType.LLM, config=NodeConfig(), dependencies=["router"]),
            ],
            edges=[
                Edge(from_node="router", to_node="billing_handler"),
                Edge(from_node="router", to_node="technical_handler"),
                Edge(from_node="router", to_node="general_handler"),
            ]
        )

        scheduler = Scheduler(condition_executor, max_parallel=5)

        # Test each category
        for category in ["billing", "technical", "general"]:
            result = await scheduler.execute_flow(flow, inputs={"value": category})
            assert result.status == RunStatus.COMPLETED


class TestIterativeWorkflowPatterns:
    """Test loop-like and iterative workflow patterns."""

    @pytest.fixture
    def iteration_executor(self):
        """Create executor that supports iteration patterns."""
        class IterativeExecutor:
            def __init__(self):
                self.iteration_count = 0
                self.max_iterations = 5

            async def execute(self, node, inputs):
                if node.id == "iterator":
                    self.iteration_count += 1
                    should_continue = self.iteration_count < self.max_iterations
                    return {
                        "iteration": self.iteration_count,
                        "continue": should_continue,
                        "accumulated": inputs.get("accumulated", 0) + 1
                    }
                elif node.id == "processor":
                    return {
                        "processed_value": inputs.get("accumulated", 0) * 2,
                        "iteration": inputs.get("iteration", 0)
                    }
                else:
                    return {"result": f"node_{node.id}", **inputs}

        return IterativeExecutor()

    @pytest.mark.asyncio
    async def test_batch_processing_pattern(self, iteration_executor):
        """Test batch processing workflow pattern."""
        # Simulate batch processing: split -> process in parallel -> merge
        flow = FlowDefinition(
            name="batch-process",
            version="1.0",
            nodes=[
                Node(id="splitter", type=NodeType.TRANSFORM, config=NodeConfig()),
                Node(id="process_1", type=NodeType.LLM, config=NodeConfig(), dependencies=["splitter"]),
                Node(id="process_2", type=NodeType.LLM, config=NodeConfig(), dependencies=["splitter"]),
                Node(id="process_3", type=NodeType.LLM, config=NodeConfig(), dependencies=["splitter"]),
                Node(id="merger", type=NodeType.TRANSFORM, config=NodeConfig(),
                     dependencies=["process_1", "process_2", "process_3"]),
            ],
            edges=[
                Edge(from_node="splitter", to_node="process_1"),
                Edge(from_node="splitter", to_node="process_2"),
                Edge(from_node="splitter", to_node="process_3"),
                Edge(from_node="process_1", to_node="merger"),
                Edge(from_node="process_2", to_node="merger"),
                Edge(from_node="process_3", to_node="merger"),
            ]
        )

        scheduler = Scheduler(iteration_executor, max_parallel=5)
        result = await scheduler.execute_flow(flow, inputs={"batch_size": 3})

        assert result.status == RunStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_map_reduce_pattern(self):
        """Test map-reduce workflow pattern."""
        class MapReduceExecutor:
            async def execute(self, node, inputs):
                if node.id == "mapper":
                    # Simulate mapping: produce multiple outputs
                    data = inputs.get("data", [1, 2, 3, 4, 5])
                    return {"mapped": [x * 2 for x in data]}
                elif node.id == "reducer":
                    # Simulate reducing: aggregate results
                    mapped = inputs.get("mapped", [])
                    return {"reduced": sum(mapped)}
                else:
                    return {"result": node.id}

        flow = FlowDefinition(
            name="map-reduce",
            version="1.0",
            nodes=[
                Node(id="mapper", type=NodeType.TRANSFORM, config=NodeConfig()),
                Node(id="reducer", type=NodeType.TRANSFORM, config=NodeConfig(), dependencies=["mapper"]),
            ],
            edges=[Edge(from_node="mapper", to_node="reducer")]
        )

        scheduler = Scheduler(MapReduceExecutor(), max_parallel=5)
        result = await scheduler.execute_flow(flow, inputs={"data": [1, 2, 3, 4, 5]})

        assert result.status == RunStatus.COMPLETED


class TestNestedWorkflowPatterns:
    """Test nested and subflow patterns."""

    @pytest.mark.asyncio
    async def test_nested_parallel_groups(self):
        """Test nested groups of parallel executions."""
        class TimedExecutor:
            async def execute(self, node, inputs):
                await asyncio.sleep(0.02)
                return {"result": node.id}

        # Group 1 (parallel) -> Group 2 (parallel) -> Group 3 (parallel)
        flow = FlowDefinition(
            name="nested-parallel",
            version="1.0",
            nodes=[
                # Group 1
                Node(id="g1_a", type=NodeType.LLM, config=NodeConfig()),
                Node(id="g1_b", type=NodeType.LLM, config=NodeConfig()),
                # Group 2
                Node(id="g2_a", type=NodeType.LLM, config=NodeConfig(), dependencies=["g1_a", "g1_b"]),
                Node(id="g2_b", type=NodeType.LLM, config=NodeConfig(), dependencies=["g1_a", "g1_b"]),
                Node(id="g2_c", type=NodeType.LLM, config=NodeConfig(), dependencies=["g1_a", "g1_b"]),
                # Group 3
                Node(id="g3_a", type=NodeType.LLM, config=NodeConfig(), dependencies=["g2_a", "g2_b", "g2_c"]),
                Node(id="g3_b", type=NodeType.LLM, config=NodeConfig(), dependencies=["g2_a", "g2_b", "g2_c"]),
            ],
            edges=[
                Edge(from_node="g1_a", to_node="g2_a"),
                Edge(from_node="g1_a", to_node="g2_b"),
                Edge(from_node="g1_a", to_node="g2_c"),
                Edge(from_node="g1_b", to_node="g2_a"),
                Edge(from_node="g1_b", to_node="g2_b"),
                Edge(from_node="g1_b", to_node="g2_c"),
                Edge(from_node="g2_a", to_node="g3_a"),
                Edge(from_node="g2_a", to_node="g3_b"),
                Edge(from_node="g2_b", to_node="g3_a"),
                Edge(from_node="g2_b", to_node="g3_b"),
                Edge(from_node="g2_c", to_node="g3_a"),
                Edge(from_node="g2_c", to_node="g3_b"),
            ]
        )

        dag = DAGBuilder(flow)
        levels = dag.get_execution_levels()

        # Should have 3 levels
        assert len(levels) == 3
        assert set(levels[0]) == {"g1_a", "g1_b"}
        assert set(levels[1]) == {"g2_a", "g2_b", "g2_c"}
        assert set(levels[2]) == {"g3_a", "g3_b"}

        # Execute and verify timing
        scheduler = Scheduler(TimedExecutor(), max_parallel=10)
        start = time.time()
        result = await scheduler.execute_flow(flow, inputs={})
        elapsed = time.time() - start

        assert result.status == RunStatus.COMPLETED
        # 3 phases * ~0.02s each = ~0.06s (with overhead, should be under 0.2s)
        assert elapsed < 0.3


class TestMixedPatternWorkflows:
    """Test workflows combining multiple patterns."""

    @pytest.mark.asyncio
    async def test_parallel_with_conditional(self):
        """Test parallel execution with conditional merging."""
        class MixedExecutor:
            async def execute(self, node, inputs):
                await asyncio.sleep(0.01)

                if node.type == NodeType.BRANCH:
                    # Determine which results to use based on scores
                    path_a_score = inputs.get("path_a_score", 0.5)
                    path_b_score = inputs.get("path_b_score", 0.5)

                    if path_a_score > path_b_score:
                        return {"selected": "path_a", "score": path_a_score}
                    else:
                        return {"selected": "path_b", "score": path_b_score}

                elif node.id.startswith("path_a"):
                    return {"path_a_score": 0.9, "result": "a"}
                elif node.id.startswith("path_b"):
                    return {"path_b_score": 0.7, "result": "b"}
                else:
                    return {"result": node.id}

        flow = FlowDefinition(
            name="parallel-conditional",
            version="1.0",
            nodes=[
                Node(id="source", type=NodeType.LLM, config=NodeConfig()),
                # Parallel paths
                Node(id="path_a", type=NodeType.LLM, config=NodeConfig(), dependencies=["source"]),
                Node(id="path_b", type=NodeType.LLM, config=NodeConfig(), dependencies=["source"]),
                # Conditional selector
                Node(id="selector", type=NodeType.BRANCH, config=NodeConfig(),
                     dependencies=["path_a", "path_b"]),
                Node(id="final", type=NodeType.LLM, config=NodeConfig(), dependencies=["selector"]),
            ],
            edges=[
                Edge(from_node="source", to_node="path_a"),
                Edge(from_node="source", to_node="path_b"),
                Edge(from_node="path_a", to_node="selector"),
                Edge(from_node="path_b", to_node="selector"),
                Edge(from_node="selector", to_node="final"),
            ]
        )

        scheduler = Scheduler(MixedExecutor(), max_parallel=5)
        result = await scheduler.execute_flow(flow, inputs={})

        assert result.status == RunStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_diamond_with_transform(self):
        """Test diamond pattern with data transformation."""
        class TransformExecutor:
            async def execute(self, node, inputs):
                if node.id == "split":
                    data = inputs.get("data", [1, 2, 3, 4])
                    mid = len(data) // 2
                    return {"left": data[:mid], "right": data[mid:]}
                elif node.id == "left_transform":
                    return {"left_result": [x * 2 for x in inputs.get("left", [])]}
                elif node.id == "right_transform":
                    return {"right_result": [x * 3 for x in inputs.get("right", [])]}
                elif node.id == "merge":
                    left = inputs.get("left_result", [])
                    right = inputs.get("right_result", [])
                    return {"merged": left + right}
                return inputs

        flow = FlowDefinition(
            name="diamond-transform",
            version="1.0",
            nodes=[
                Node(id="split", type=NodeType.TRANSFORM, config=NodeConfig()),
                Node(id="left_transform", type=NodeType.TRANSFORM, config=NodeConfig(), dependencies=["split"]),
                Node(id="right_transform", type=NodeType.TRANSFORM, config=NodeConfig(), dependencies=["split"]),
                Node(id="merge", type=NodeType.TRANSFORM, config=NodeConfig(),
                     dependencies=["left_transform", "right_transform"]),
            ],
            edges=[
                Edge(from_node="split", to_node="left_transform"),
                Edge(from_node="split", to_node="right_transform"),
                Edge(from_node="left_transform", to_node="merge"),
                Edge(from_node="right_transform", to_node="merge"),
            ]
        )

        scheduler = Scheduler(TransformExecutor(), max_parallel=5)
        result = await scheduler.execute_flow(flow, inputs={"data": [1, 2, 3, 4]})

        assert result.status == RunStatus.COMPLETED


class TestAsyncSchedulerPatterns:
    """Test AsyncScheduler with batch and parallel execution."""

    @pytest.mark.asyncio
    async def test_parallel_flow_execution(self):
        """Test executing multiple flows in parallel."""
        class SimpleExecutor:
            async def execute(self, node, inputs):
                await asyncio.sleep(0.01)
                return {"result": f"{node.id}_{inputs.get('batch_id', 0)}"}

        flow = FlowDefinition(
            name="parallel-flows",
            version="1.0",
            nodes=[
                Node(id="step1", type=NodeType.LLM, config=NodeConfig()),
                Node(id="step2", type=NodeType.LLM, config=NodeConfig(), dependencies=["step1"]),
            ],
            edges=[Edge(from_node="step1", to_node="step2")]
        )

        scheduler = AsyncScheduler(SimpleExecutor(), max_parallel=10)

        # Execute 5 flows in parallel
        inputs_list = [{"batch_id": i} for i in range(5)]
        start = time.time()
        results = await scheduler.execute_parallel(flow, inputs_list)
        elapsed = time.time() - start

        assert len(results) == 5
        assert all(r.status == RunStatus.COMPLETED for r in results)
        # All 5 should execute in parallel, so time should be ~1 flow time, not 5x
        assert elapsed < 0.2


class TestInputResolutionPatterns:
    """Test input reference resolution patterns."""

    @pytest.mark.asyncio
    async def test_nested_input_reference(self):
        """Test resolving nested input references."""
        class EchoExecutor:
            async def execute(self, node, inputs):
                return {"echo": inputs, "node_id": node.id}

        flow = FlowDefinition(
            name="input-resolution",
            version="1.0",
            inputs={"user_query": "test query", "context": {"key": "value"}},
            nodes=[
                Node(
                    id="node1",
                    type=NodeType.LLM,
                    config=NodeConfig(),
                    inputs={"query": "{{inputs.user_query}}"}
                ),
                Node(
                    id="node2",
                    type=NodeType.LLM,
                    config=NodeConfig(),
                    inputs={"prev_result": "{{node1.echo}}"},
                    dependencies=["node1"]
                ),
            ],
            edges=[Edge(from_node="node1", to_node="node2")]
        )

        scheduler = Scheduler(EchoExecutor(), max_parallel=5)
        result = await scheduler.execute_flow(
            flow,
            inputs={"user_query": "actual query", "context": {"key": "value"}}
        )

        assert result.status == RunStatus.COMPLETED


class TestWorkflowStateTransitions:
    """Test workflow state transitions through execution."""

    @pytest.mark.asyncio
    async def test_node_execution_order_tracking(self):
        """Test that node execution order is properly tracked."""
        execution_order = []

        class OrderTrackingExecutor:
            async def execute(self, node, inputs):
                execution_order.append(node.id)
                return {"order": len(execution_order)}

        flow = FlowDefinition(
            name="order-tracking",
            version="1.0",
            nodes=[
                Node(id="a", type=NodeType.LLM, config=NodeConfig()),
                Node(id="b", type=NodeType.LLM, config=NodeConfig(), dependencies=["a"]),
                Node(id="c", type=NodeType.LLM, config=NodeConfig(), dependencies=["a"]),
                Node(id="d", type=NodeType.LLM, config=NodeConfig(), dependencies=["b", "c"]),
            ],
            edges=[
                Edge(from_node="a", to_node="b"),
                Edge(from_node="a", to_node="c"),
                Edge(from_node="b", to_node="d"),
                Edge(from_node="c", to_node="d"),
            ]
        )

        scheduler = Scheduler(OrderTrackingExecutor(), max_parallel=5)
        result = await scheduler.execute_flow(flow, inputs={})

        assert result.status == RunStatus.COMPLETED
        # a must come first
        assert execution_order[0] == "a"
        # d must come last
        assert execution_order[-1] == "d"
        # b and c must come between a and d
        assert "b" in execution_order[1:-1]
        assert "c" in execution_order[1:-1]
