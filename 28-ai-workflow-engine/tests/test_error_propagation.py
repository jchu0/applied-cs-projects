"""Integration tests for error propagation in workflows.

Tests cover:
- Node failure propagation
- Dependent node handling on upstream failure
- Partial completion states
- Error context preservation
- Exception chaining
- Cleanup on failure
- Recovery scenarios
- Error aggregation in parallel execution
"""

import pytest
import asyncio
import time
import sys
import os
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aiworkflow.schemas import (
    FlowDefinition,
    Node,
    NodeType,
    NodeConfig,
    Edge,
    FlowRun,
    RunStatus,
    NodeStatus,
    NodeExecution,
)

# Optional imports requiring yaml
try:
    from aiworkflow.compiler.dag import DAGBuilder
    from aiworkflow.executor.scheduler import Scheduler, AsyncScheduler
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

pytestmark = pytest.mark.skipif(not _HAS_YAML, reason="Requires yaml")


# Custom exception types for testing
class NodeExecutionError(Exception):
    """Error during node execution."""
    def __init__(self, node_id: str, message: str, cause: Exception = None):
        self.node_id = node_id
        self.cause = cause
        super().__init__(f"Node {node_id} failed: {message}")


class ValidationError(Exception):
    """Input validation error."""
    pass


class ResourceError(Exception):
    """Resource allocation error."""
    pass


class TimeoutError(Exception):
    """Execution timeout error."""
    pass


class TestNodeFailurePropagation:
    """Test error propagation from failing nodes."""

    @pytest.mark.asyncio
    async def test_single_node_failure_stops_flow(self):
        """Test that single node failure stops the workflow."""
        class FailingExecutor:
            async def execute(self, node, inputs):
                if node.id == "fail_node":
                    raise NodeExecutionError("fail_node", "Intentional failure")
                return {"result": "ok"}

        flow = FlowDefinition(
            name="fail-test",
            version="1.0",
            nodes=[
                Node(id="start", type=NodeType.LLM, config=NodeConfig()),
                Node(id="fail_node", type=NodeType.LLM, config=NodeConfig(), dependencies=["start"]),
                Node(id="never_runs", type=NodeType.LLM, config=NodeConfig(), dependencies=["fail_node"]),
            ],
            edges=[
                Edge(from_node="start", to_node="fail_node"),
                Edge(from_node="fail_node", to_node="never_runs"),
            ]
        )

        scheduler = Scheduler(FailingExecutor(), max_parallel=5)
        result = await scheduler.execute_flow(flow, inputs={})

        assert result.status == RunStatus.FAILED
        assert "fail_node" in result.error

    @pytest.mark.asyncio
    async def test_failure_in_parallel_branch_fails_flow(self):
        """Test that failure in one parallel branch fails the entire flow."""
        executed_nodes = set()

        class ParallelFailExecutor:
            async def execute(self, node, inputs):
                executed_nodes.add(node.id)
                await asyncio.sleep(0.01)  # Allow parallel execution to start

                if node.id == "branch_b":
                    raise NodeExecutionError("branch_b", "Branch B failed")

                return {"result": node.id}

        flow = FlowDefinition(
            name="parallel-fail",
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

        scheduler = Scheduler(ParallelFailExecutor(), max_parallel=5)
        result = await scheduler.execute_flow(flow, inputs={})

        assert result.status == RunStatus.FAILED
        assert "branch_b" in result.error
        # Merge should not have executed
        assert "merge" not in executed_nodes


class TestDependentNodeHandling:
    """Test handling of nodes dependent on failed upstream nodes."""

    @pytest.mark.asyncio
    async def test_downstream_nodes_not_executed(self):
        """Test that nodes downstream of failure are not executed."""
        execution_order = []

        class TrackingExecutor:
            async def execute(self, node, inputs):
                execution_order.append(node.id)
                if node.id == "fail":
                    raise Exception("Failure")
                return {"result": node.id}

        flow = FlowDefinition(
            name="downstream-test",
            version="1.0",
            nodes=[
                Node(id="a", type=NodeType.LLM, config=NodeConfig()),
                Node(id="fail", type=NodeType.LLM, config=NodeConfig(), dependencies=["a"]),
                Node(id="b", type=NodeType.LLM, config=NodeConfig(), dependencies=["fail"]),
                Node(id="c", type=NodeType.LLM, config=NodeConfig(), dependencies=["b"]),
            ],
            edges=[
                Edge(from_node="a", to_node="fail"),
                Edge(from_node="fail", to_node="b"),
                Edge(from_node="b", to_node="c"),
            ]
        )

        scheduler = Scheduler(TrackingExecutor(), max_parallel=5)
        result = await scheduler.execute_flow(flow, inputs={})

        assert result.status == RunStatus.FAILED
        assert "a" in execution_order
        assert "fail" in execution_order
        assert "b" not in execution_order
        assert "c" not in execution_order

    @pytest.mark.asyncio
    async def test_independent_branches_complete_after_failure(self):
        """Test that independent branches may complete even if one fails."""
        execution_log = []
        lock = asyncio.Lock()

        class IndependentBranchExecutor:
            async def execute(self, node, inputs):
                async with lock:
                    execution_log.append(("start", node.id, time.time()))

                if node.id == "fail_branch":
                    await asyncio.sleep(0.01)
                    async with lock:
                        execution_log.append(("fail", node.id, time.time()))
                    raise Exception("Failure")

                await asyncio.sleep(0.02)

                async with lock:
                    execution_log.append(("end", node.id, time.time()))

                return {"result": node.id}

        # Two independent paths from source
        flow = FlowDefinition(
            name="independent-branches",
            version="1.0",
            nodes=[
                Node(id="source", type=NodeType.LLM, config=NodeConfig()),
                Node(id="fail_branch", type=NodeType.LLM, config=NodeConfig(), dependencies=["source"]),
                Node(id="ok_branch", type=NodeType.LLM, config=NodeConfig(), dependencies=["source"]),
            ],
            edges=[
                Edge(from_node="source", to_node="fail_branch"),
                Edge(from_node="source", to_node="ok_branch"),
            ]
        )

        scheduler = Scheduler(IndependentBranchExecutor(), max_parallel=5)
        result = await scheduler.execute_flow(flow, inputs={})

        assert result.status == RunStatus.FAILED

        # Both branches should have started
        starts = [e for e in execution_log if e[0] == "start"]
        assert any(e[1] == "fail_branch" for e in starts)
        assert any(e[1] == "ok_branch" for e in starts)


class TestPartialCompletionStates:
    """Test tracking of partially completed workflows."""

    @pytest.mark.asyncio
    async def test_node_execution_records_on_failure(self):
        """Test that node execution records are preserved on failure."""
        class PartialExecutor:
            async def execute(self, node, inputs):
                if node.id == "c":
                    raise Exception("Node C failed")
                return {"result": node.id}

        flow = FlowDefinition(
            name="partial-completion",
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

        scheduler = Scheduler(PartialExecutor(), max_parallel=5)
        result = await scheduler.execute_flow(flow, inputs={})

        assert result.status == RunStatus.FAILED

        # Should have execution records for completed nodes
        completed_ids = [e.node_id for e in result.node_executions
                        if e.status == NodeStatus.COMPLETED]
        failed_ids = [e.node_id for e in result.node_executions
                     if e.status == NodeStatus.FAILED]

        assert "a" in completed_ids
        assert "b" in completed_ids
        assert "c" in failed_ids


class TestErrorContextPreservation:
    """Test preservation of error context and stack traces."""

    @pytest.mark.asyncio
    async def test_error_message_preserved(self):
        """Test that original error message is preserved."""
        error_message = "Specific error with details: code=42, resource=test"

        class DetailedErrorExecutor:
            async def execute(self, node, inputs):
                if node.id == "error_node":
                    raise ValueError(error_message)
                return {"result": node.id}

        flow = FlowDefinition(
            name="error-context",
            version="1.0",
            nodes=[
                Node(id="error_node", type=NodeType.LLM, config=NodeConfig()),
            ],
            edges=[]
        )

        scheduler = Scheduler(DetailedErrorExecutor(), max_parallel=5)
        result = await scheduler.execute_flow(flow, inputs={})

        assert result.status == RunStatus.FAILED
        # Original error message should be preserved
        assert "code=42" in result.error or "Specific error" in result.error

    @pytest.mark.asyncio
    async def test_nested_exception_handling(self):
        """Test handling of nested/chained exceptions."""
        class NestedException(Exception):
            pass

        class NestedErrorExecutor:
            async def execute(self, node, inputs):
                if node.id == "nested":
                    try:
                        raise ValueError("Inner error")
                    except ValueError as e:
                        raise NestedException("Outer error") from e
                return {"result": node.id}

        flow = FlowDefinition(
            name="nested-error",
            version="1.0",
            nodes=[
                Node(id="nested", type=NodeType.LLM, config=NodeConfig()),
            ],
            edges=[]
        )

        scheduler = Scheduler(NestedErrorExecutor(), max_parallel=5)
        result = await scheduler.execute_flow(flow, inputs={})

        assert result.status == RunStatus.FAILED


class TestExceptionTypes:
    """Test handling of different exception types."""

    @pytest.mark.asyncio
    async def test_validation_error_handling(self):
        """Test handling of validation errors."""
        class ValidationExecutor:
            async def execute(self, node, inputs):
                if not inputs.get("required_field"):
                    raise ValidationError("Missing required_field")
                return {"result": "ok"}

        flow = FlowDefinition(
            name="validation-test",
            version="1.0",
            nodes=[
                Node(id="validator", type=NodeType.LLM, config=NodeConfig()),
            ],
            edges=[]
        )

        scheduler = Scheduler(ValidationExecutor(), max_parallel=5)
        result = await scheduler.execute_flow(flow, inputs={})

        assert result.status == RunStatus.FAILED
        assert "required_field" in result.error

    @pytest.mark.asyncio
    async def test_timeout_error_handling(self):
        """Test handling of timeout errors."""
        class TimeoutExecutor:
            async def execute(self, node, inputs):
                raise TimeoutError("Operation timed out after 30s")

        flow = FlowDefinition(
            name="timeout-test",
            version="1.0",
            nodes=[
                Node(id="slow_node", type=NodeType.LLM, config=NodeConfig()),
            ],
            edges=[]
        )

        scheduler = Scheduler(TimeoutExecutor(), max_parallel=5)
        result = await scheduler.execute_flow(flow, inputs={})

        assert result.status == RunStatus.FAILED
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_resource_error_handling(self):
        """Test handling of resource errors."""
        class ResourceExecutor:
            async def execute(self, node, inputs):
                raise ResourceError("Insufficient memory to process request")

        flow = FlowDefinition(
            name="resource-test",
            version="1.0",
            nodes=[
                Node(id="resource_heavy", type=NodeType.LLM, config=NodeConfig()),
            ],
            edges=[]
        )

        scheduler = Scheduler(ResourceExecutor(), max_parallel=5)
        result = await scheduler.execute_flow(flow, inputs={})

        assert result.status == RunStatus.FAILED
        assert "memory" in result.error.lower()


class TestErrorAggregation:
    """Test error aggregation in parallel execution scenarios."""

    @pytest.mark.asyncio
    async def test_first_error_reported(self):
        """Test that first error encountered is reported."""
        first_error_node = None
        lock = asyncio.Lock()

        class MultiFailExecutor:
            async def execute(self, node, inputs):
                nonlocal first_error_node

                if node.id.startswith("fail"):
                    # Stagger failures
                    delay = int(node.id.split("_")[1]) * 0.01
                    await asyncio.sleep(delay)

                    async with lock:
                        if first_error_node is None:
                            first_error_node = node.id

                    raise Exception(f"{node.id} failed")

                return {"result": node.id}

        flow = FlowDefinition(
            name="multi-fail",
            version="1.0",
            nodes=[
                Node(id="source", type=NodeType.LLM, config=NodeConfig()),
                Node(id="fail_1", type=NodeType.LLM, config=NodeConfig(), dependencies=["source"]),
                Node(id="fail_2", type=NodeType.LLM, config=NodeConfig(), dependencies=["source"]),
                Node(id="fail_3", type=NodeType.LLM, config=NodeConfig(), dependencies=["source"]),
            ],
            edges=[
                Edge(from_node="source", to_node="fail_1"),
                Edge(from_node="source", to_node="fail_2"),
                Edge(from_node="source", to_node="fail_3"),
            ]
        )

        scheduler = Scheduler(MultiFailExecutor(), max_parallel=5)
        result = await scheduler.execute_flow(flow, inputs={})

        assert result.status == RunStatus.FAILED
        # Should report first error (fail_1)
        assert "fail_1" in result.error


class TestCleanupOnFailure:
    """Test cleanup and resource release on failure."""

    @pytest.mark.asyncio
    async def test_cleanup_called_on_failure(self):
        """Test that cleanup operations are performed on failure."""
        resources_acquired = []
        resources_released = []

        class CleanupExecutor:
            async def execute(self, node, inputs):
                resources_acquired.append(node.id)
                try:
                    if node.id == "fail":
                        raise Exception("Failure")
                    await asyncio.sleep(0.01)
                    return {"result": node.id}
                finally:
                    resources_released.append(node.id)

        flow = FlowDefinition(
            name="cleanup-test",
            version="1.0",
            nodes=[
                Node(id="a", type=NodeType.LLM, config=NodeConfig()),
                Node(id="fail", type=NodeType.LLM, config=NodeConfig(), dependencies=["a"]),
            ],
            edges=[Edge(from_node="a", to_node="fail")]
        )

        scheduler = Scheduler(CleanupExecutor(), max_parallel=5)
        result = await scheduler.execute_flow(flow, inputs={})

        assert result.status == RunStatus.FAILED
        # All acquired resources should be released
        for resource in resources_acquired:
            assert resource in resources_released


class TestRecoveryScenarios:
    """Test error recovery scenarios."""

    @pytest.mark.asyncio
    async def test_flow_can_rerun_after_failure(self):
        """Test that a flow can be rerun successfully after failure."""
        attempt = [0]

        class RecoverableExecutor:
            async def execute(self, node, inputs):
                attempt[0] += 1
                if attempt[0] <= 2 and node.id == "maybe_fail":
                    raise Exception("First attempt fails")
                return {"result": node.id}

        flow = FlowDefinition(
            name="recovery-test",
            version="1.0",
            nodes=[
                Node(id="start", type=NodeType.LLM, config=NodeConfig()),
                Node(id="maybe_fail", type=NodeType.LLM, config=NodeConfig(), dependencies=["start"]),
                Node(id="end", type=NodeType.LLM, config=NodeConfig(), dependencies=["maybe_fail"]),
            ],
            edges=[
                Edge(from_node="start", to_node="maybe_fail"),
                Edge(from_node="maybe_fail", to_node="end"),
            ]
        )

        scheduler = Scheduler(RecoverableExecutor(), max_parallel=5)

        # First run fails
        result1 = await scheduler.execute_flow(flow, inputs={})
        assert result1.status == RunStatus.FAILED

        # Second run succeeds
        result2 = await scheduler.execute_flow(flow, inputs={})
        assert result2.status == RunStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_partial_results_available_on_failure(self):
        """Test that partial results from completed nodes are available."""
        class PartialResultExecutor:
            async def execute(self, node, inputs):
                if node.id == "fail":
                    raise Exception("Failure")
                return {"value": f"result_{node.id}"}

        flow = FlowDefinition(
            name="partial-results",
            version="1.0",
            nodes=[
                Node(id="a", type=NodeType.LLM, config=NodeConfig()),
                Node(id="b", type=NodeType.LLM, config=NodeConfig(), dependencies=["a"]),
                Node(id="fail", type=NodeType.LLM, config=NodeConfig(), dependencies=["b"]),
            ],
            edges=[
                Edge(from_node="a", to_node="b"),
                Edge(from_node="b", to_node="fail"),
            ]
        )

        scheduler = Scheduler(PartialResultExecutor(), max_parallel=5)
        result = await scheduler.execute_flow(flow, inputs={})

        assert result.status == RunStatus.FAILED

        # Check that completed nodes have results
        completed_execs = [e for e in result.node_executions
                         if e.status == NodeStatus.COMPLETED]
        assert len(completed_execs) == 2


class TestBatchErrorHandling:
    """Test error handling in batch execution scenarios."""

    @pytest.mark.asyncio
    async def test_batch_partial_failure(self):
        """Test handling of partial failures in batch execution."""
        fail_indices = {2, 5, 8}

        class BatchExecutor:
            async def execute(self, node, inputs):
                batch_id = inputs.get("batch_id", 0)
                if batch_id in fail_indices:
                    raise Exception(f"Batch {batch_id} failed")
                return {"result": f"batch_{batch_id}"}

        flow = FlowDefinition(
            name="batch-fail",
            version="1.0",
            nodes=[
                Node(id="process", type=NodeType.LLM, config=NodeConfig()),
            ],
            edges=[]
        )

        scheduler = AsyncScheduler(BatchExecutor(), max_parallel=10)

        inputs_list = [{"batch_id": i} for i in range(10)]
        results = await scheduler.execute_parallel(flow, inputs_list)

        assert len(results) == 10

        successful = [r for r in results if r.status == RunStatus.COMPLETED]
        failed = [r for r in results if r.status == RunStatus.FAILED]

        assert len(failed) == len(fail_indices)
        assert len(successful) == 10 - len(fail_indices)

    @pytest.mark.asyncio
    async def test_batch_all_fail(self):
        """Test handling when all batch items fail."""
        class AllFailExecutor:
            async def execute(self, node, inputs):
                raise Exception("All fail")

        flow = FlowDefinition(
            name="all-fail",
            version="1.0",
            nodes=[
                Node(id="process", type=NodeType.LLM, config=NodeConfig()),
            ],
            edges=[]
        )

        scheduler = AsyncScheduler(AllFailExecutor(), max_parallel=10)

        inputs_list = [{"batch_id": i} for i in range(5)]
        results = await scheduler.execute_parallel(flow, inputs_list)

        assert len(results) == 5
        assert all(r.status == RunStatus.FAILED for r in results)


class TestErrorInDifferentPhases:
    """Test errors occurring in different workflow phases."""

    @pytest.mark.asyncio
    async def test_error_in_first_node(self):
        """Test error in the very first node."""
        class FirstNodeFailExecutor:
            async def execute(self, node, inputs):
                if node.id == "first":
                    raise Exception("First node fails")
                return {"result": node.id}

        flow = FlowDefinition(
            name="first-fail",
            version="1.0",
            nodes=[
                Node(id="first", type=NodeType.LLM, config=NodeConfig()),
                Node(id="second", type=NodeType.LLM, config=NodeConfig(), dependencies=["first"]),
            ],
            edges=[Edge(from_node="first", to_node="second")]
        )

        scheduler = Scheduler(FirstNodeFailExecutor(), max_parallel=5)
        result = await scheduler.execute_flow(flow, inputs={})

        assert result.status == RunStatus.FAILED
        assert len([e for e in result.node_executions if e.status == NodeStatus.COMPLETED]) == 0

    @pytest.mark.asyncio
    async def test_error_in_last_node(self):
        """Test error in the very last node."""
        class LastNodeFailExecutor:
            async def execute(self, node, inputs):
                if node.id == "last":
                    raise Exception("Last node fails")
                return {"result": node.id}

        flow = FlowDefinition(
            name="last-fail",
            version="1.0",
            nodes=[
                Node(id="first", type=NodeType.LLM, config=NodeConfig()),
                Node(id="second", type=NodeType.LLM, config=NodeConfig(), dependencies=["first"]),
                Node(id="last", type=NodeType.LLM, config=NodeConfig(), dependencies=["second"]),
            ],
            edges=[
                Edge(from_node="first", to_node="second"),
                Edge(from_node="second", to_node="last"),
            ]
        )

        scheduler = Scheduler(LastNodeFailExecutor(), max_parallel=5)
        result = await scheduler.execute_flow(flow, inputs={})

        assert result.status == RunStatus.FAILED
        # First two nodes should have completed
        completed = [e for e in result.node_executions if e.status == NodeStatus.COMPLETED]
        assert len(completed) == 2

    @pytest.mark.asyncio
    async def test_error_in_middle_of_parallel_phase(self):
        """Test error occurring in middle of parallel execution phase."""
        execution_times = {}
        lock = asyncio.Lock()

        class MidParallelFailExecutor:
            async def execute(self, node, inputs):
                start = time.time()

                if node.id == "p3":
                    await asyncio.sleep(0.02)  # Fail in the middle
                    async with lock:
                        execution_times[node.id] = (start, time.time())
                    raise Exception("Middle parallel node fails")

                await asyncio.sleep(0.05)

                async with lock:
                    execution_times[node.id] = (start, time.time())

                return {"result": node.id}

        nodes = [Node(id="source", type=NodeType.LLM, config=NodeConfig())]
        for i in range(5):
            nodes.append(Node(
                id=f"p{i}",
                type=NodeType.LLM,
                config=NodeConfig(),
                dependencies=["source"]
            ))

        flow = FlowDefinition(
            name="mid-parallel-fail",
            version="1.0",
            nodes=nodes,
            edges=[Edge(from_node="source", to_node=f"p{i}") for i in range(5)]
        )

        scheduler = Scheduler(MidParallelFailExecutor(), max_parallel=5)
        result = await scheduler.execute_flow(flow, inputs={})

        assert result.status == RunStatus.FAILED
        assert "p3" in result.error
