"""Integration tests for scheduler behavior under load.

Tests cover:
- High concurrency execution
- Resource contention scenarios
- Memory pressure handling
- Throughput benchmarks
- Latency under load
- Semaphore behavior at limits
- Large workflow execution
- Batch processing performance
"""

import pytest
import asyncio
import time
import sys
import os
import gc
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aiworkflow.schemas import (
    FlowDefinition,
    Node,
    NodeType,
    NodeConfig,
    Edge,
    FlowRun,
    RunStatus,
)

# Optional imports requiring yaml
try:
    from aiworkflow.compiler.dag import DAGBuilder
    from aiworkflow.executor.scheduler import Scheduler, AsyncScheduler
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

pytestmark = pytest.mark.skipif(not _HAS_YAML, reason="Requires yaml")


class TestHighConcurrencyExecution:
    """Test scheduler behavior under high concurrency."""

    @pytest.fixture
    def concurrent_executor(self):
        """Create executor that tracks concurrent executions."""
        class ConcurrentTracker:
            def __init__(self):
                self.current_concurrent = 0
                self.max_concurrent = 0
                self.lock = asyncio.Lock()
                self.execution_times = []

            async def execute(self, node, inputs):
                async with self.lock:
                    self.current_concurrent += 1
                    self.max_concurrent = max(self.max_concurrent, self.current_concurrent)

                start = time.time()
                await asyncio.sleep(0.02)  # Simulate work
                elapsed = time.time() - start

                async with self.lock:
                    self.current_concurrent -= 1
                    self.execution_times.append(elapsed)

                return {"node_id": node.id, "result": "ok"}

        return ConcurrentTracker()

    @pytest.mark.asyncio
    async def test_respects_max_parallel_limit(self, concurrent_executor):
        """Test that scheduler respects max_parallel setting."""
        max_parallel = 5

        # Create flow with many parallel nodes
        nodes = [Node(id="source", type=NodeType.LLM, config=NodeConfig())]
        edges = []

        for i in range(20):
            nodes.append(Node(
                id=f"worker_{i}",
                type=NodeType.LLM,
                config=NodeConfig(),
                dependencies=["source"]
            ))
            edges.append(Edge(from_node="source", to_node=f"worker_{i}"))

        flow = FlowDefinition(
            name="high-concurrency",
            version="1.0",
            nodes=nodes,
            edges=edges
        )

        scheduler = Scheduler(concurrent_executor, max_parallel=max_parallel)
        result = await scheduler.execute_flow(flow, inputs={})

        assert result.status == RunStatus.COMPLETED
        # Should never exceed max_parallel during worker execution
        assert concurrent_executor.max_concurrent <= max_parallel

    @pytest.mark.asyncio
    async def test_all_nodes_complete_under_load(self, concurrent_executor):
        """Test all nodes complete even under high load."""
        num_workers = 50

        nodes = [Node(id="source", type=NodeType.LLM, config=NodeConfig())]
        edges = []

        for i in range(num_workers):
            nodes.append(Node(
                id=f"worker_{i}",
                type=NodeType.LLM,
                config=NodeConfig(),
                dependencies=["source"]
            ))
            edges.append(Edge(from_node="source", to_node=f"worker_{i}"))

        # Add merge node
        nodes.append(Node(
            id="merge",
            type=NodeType.LLM,
            config=NodeConfig(),
            dependencies=[f"worker_{i}" for i in range(num_workers)]
        ))
        for i in range(num_workers):
            edges.append(Edge(from_node=f"worker_{i}", to_node="merge"))

        flow = FlowDefinition(
            name="load-test",
            version="1.0",
            nodes=nodes,
            edges=edges
        )

        scheduler = Scheduler(concurrent_executor, max_parallel=10)
        result = await scheduler.execute_flow(flow, inputs={})

        assert result.status == RunStatus.COMPLETED
        # All nodes should have executed (source + workers + merge)
        assert len(concurrent_executor.execution_times) == num_workers + 2


class TestResourceContention:
    """Test scheduler behavior under resource contention."""

    @pytest.mark.asyncio
    async def test_semaphore_fairness(self):
        """Test that semaphore provides fair access to all nodes."""
        execution_order = []
        lock = asyncio.Lock()

        class FairnessExecutor:
            async def execute(self, node, inputs):
                async with lock:
                    execution_order.append(node.id)

                await asyncio.sleep(random.uniform(0.01, 0.03))
                return {"result": node.id}

        # Create many parallel nodes
        nodes = [Node(id="source", type=NodeType.LLM, config=NodeConfig())]
        edges = []

        for i in range(10):
            nodes.append(Node(
                id=f"node_{i}",
                type=NodeType.LLM,
                config=NodeConfig(),
                dependencies=["source"]
            ))
            edges.append(Edge(from_node="source", to_node=f"node_{i}"))

        flow = FlowDefinition(
            name="fairness-test",
            version="1.0",
            nodes=nodes,
            edges=edges
        )

        scheduler = Scheduler(FairnessExecutor(), max_parallel=3)
        result = await scheduler.execute_flow(flow, inputs={})

        assert result.status == RunStatus.COMPLETED
        # All parallel nodes should execute
        parallel_executions = [e for e in execution_order if e.startswith("node_")]
        assert len(parallel_executions) == 10

    @pytest.mark.asyncio
    async def test_no_deadlock_under_contention(self):
        """Test that scheduler doesn't deadlock under resource contention."""
        class ContentiousExecutor:
            def __init__(self):
                self.resources = asyncio.Semaphore(2)

            async def execute(self, node, inputs):
                async with self.resources:
                    await asyncio.sleep(0.01)
                    return {"result": "ok"}

        # Create a complex dependency graph
        nodes = [
            Node(id="a", type=NodeType.LLM, config=NodeConfig()),
            Node(id="b", type=NodeType.LLM, config=NodeConfig()),
            Node(id="c", type=NodeType.LLM, config=NodeConfig(), dependencies=["a"]),
            Node(id="d", type=NodeType.LLM, config=NodeConfig(), dependencies=["a"]),
            Node(id="e", type=NodeType.LLM, config=NodeConfig(), dependencies=["b"]),
            Node(id="f", type=NodeType.LLM, config=NodeConfig(), dependencies=["b"]),
            Node(id="g", type=NodeType.LLM, config=NodeConfig(), dependencies=["c", "d", "e", "f"]),
        ]

        edges = [
            Edge(from_node="a", to_node="c"),
            Edge(from_node="a", to_node="d"),
            Edge(from_node="b", to_node="e"),
            Edge(from_node="b", to_node="f"),
            Edge(from_node="c", to_node="g"),
            Edge(from_node="d", to_node="g"),
            Edge(from_node="e", to_node="g"),
            Edge(from_node="f", to_node="g"),
        ]

        flow = FlowDefinition(
            name="contention-test",
            version="1.0",
            nodes=nodes,
            edges=edges
        )

        scheduler = Scheduler(ContentiousExecutor(), max_parallel=2)

        # Should complete without deadlock within timeout
        try:
            result = await asyncio.wait_for(
                scheduler.execute_flow(flow, inputs={}),
                timeout=5.0
            )
            assert result.status == RunStatus.COMPLETED
        except asyncio.TimeoutError:
            pytest.fail("Scheduler deadlocked")


class TestThroughputBenchmarks:
    """Test scheduler throughput under various conditions."""

    @pytest.mark.asyncio
    async def test_throughput_linear_chain(self):
        """Benchmark throughput for linear chain execution."""
        class FastExecutor:
            async def execute(self, node, inputs):
                return {"result": node.id}

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
            name="throughput-linear",
            version="1.0",
            nodes=nodes,
            edges=edges
        )

        scheduler = Scheduler(FastExecutor(), max_parallel=10)

        start = time.time()
        result = await scheduler.execute_flow(flow, inputs={})
        elapsed = time.time() - start

        assert result.status == RunStatus.COMPLETED

        nodes_per_second = num_nodes / elapsed
        # Should achieve reasonable throughput
        assert nodes_per_second > 100, f"Throughput too low: {nodes_per_second:.1f} nodes/sec"

    @pytest.mark.asyncio
    async def test_throughput_wide_parallel(self):
        """Benchmark throughput for wide parallel execution."""
        class FastExecutor:
            async def execute(self, node, inputs):
                await asyncio.sleep(0.001)  # Minimal work
                return {"result": node.id}

        num_parallel = 100

        nodes = [Node(id="source", type=NodeType.LLM, config=NodeConfig())]
        edges = []

        for i in range(num_parallel):
            nodes.append(Node(
                id=f"worker_{i}",
                type=NodeType.LLM,
                config=NodeConfig(),
                dependencies=["source"]
            ))
            edges.append(Edge(from_node="source", to_node=f"worker_{i}"))

        flow = FlowDefinition(
            name="throughput-parallel",
            version="1.0",
            nodes=nodes,
            edges=edges
        )

        scheduler = Scheduler(FastExecutor(), max_parallel=50)

        start = time.time()
        result = await scheduler.execute_flow(flow, inputs={})
        elapsed = time.time() - start

        assert result.status == RunStatus.COMPLETED

        # With 100 parallel nodes and max_parallel=50, should complete in ~2 batches
        # Each batch takes ~0.001s, so total should be well under 1s
        assert elapsed < 1.0, f"Parallel execution too slow: {elapsed:.2f}s"


class TestLatencyUnderLoad:
    """Test scheduler latency characteristics under load."""

    @pytest.mark.asyncio
    async def test_first_node_latency(self):
        """Test that first node starts quickly even under load."""
        first_node_start = None
        request_time = None

        class LatencyExecutor:
            async def execute(self, node, inputs):
                nonlocal first_node_start
                if first_node_start is None:
                    first_node_start = time.time()
                await asyncio.sleep(0.01)
                return {"result": node.id}

        # Large parallel flow
        nodes = [Node(id="source", type=NodeType.LLM, config=NodeConfig())]
        for i in range(50):
            nodes.append(Node(
                id=f"worker_{i}",
                type=NodeType.LLM,
                config=NodeConfig(),
                dependencies=["source"]
            ))

        flow = FlowDefinition(
            name="latency-test",
            version="1.0",
            nodes=nodes,
            edges=[]
        )

        scheduler = Scheduler(LatencyExecutor(), max_parallel=10)

        request_time = time.time()
        await scheduler.execute_flow(flow, inputs={})

        startup_latency = first_node_start - request_time
        # First node should start within 50ms
        assert startup_latency < 0.05, f"Startup latency too high: {startup_latency:.3f}s"

    @pytest.mark.asyncio
    async def test_node_execution_latency_consistency(self):
        """Test that node execution latencies are consistent."""
        latencies = []

        class MeasuredExecutor:
            async def execute(self, node, inputs):
                start = time.time()
                await asyncio.sleep(0.01)  # Fixed work
                latencies.append(time.time() - start)
                return {"result": node.id}

        nodes = [Node(id=f"node_{i}", type=NodeType.LLM, config=NodeConfig())
                 for i in range(20)]

        flow = FlowDefinition(
            name="consistency-test",
            version="1.0",
            nodes=nodes,
            edges=[]
        )

        scheduler = Scheduler(MeasuredExecutor(), max_parallel=5)
        await scheduler.execute_flow(flow, inputs={})

        # Calculate latency statistics
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)

        # Latencies should be reasonably consistent
        variance = max_latency - min_latency
        assert variance < avg_latency, f"Latency variance too high: {variance:.4f}s"


class TestLargeWorkflowExecution:
    """Test scheduler with large workflow graphs."""

    @pytest.mark.asyncio
    async def test_thousand_node_workflow(self):
        """Test execution of workflow with 1000 nodes."""
        class SimpleExecutor:
            async def execute(self, node, inputs):
                return {"result": node.id}

        # Create tree structure
        num_nodes = 1000
        nodes = []
        edges = []

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
            name="large-workflow",
            version="1.0",
            nodes=nodes,
            edges=edges
        )

        scheduler = Scheduler(SimpleExecutor(), max_parallel=100)

        start = time.time()
        result = await scheduler.execute_flow(flow, inputs={})
        elapsed = time.time() - start

        assert result.status == RunStatus.COMPLETED
        # Should complete 1000 nodes in reasonable time
        assert elapsed < 5.0, f"Large workflow too slow: {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_deep_dependency_chain(self):
        """Test workflow with very deep dependency chain."""
        class SimpleExecutor:
            async def execute(self, node, inputs):
                return {"result": node.id}

        depth = 200
        nodes = []
        edges = []

        for i in range(depth):
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

        scheduler = Scheduler(SimpleExecutor(), max_parallel=10)

        start = time.time()
        result = await scheduler.execute_flow(flow, inputs={})
        elapsed = time.time() - start

        assert result.status == RunStatus.COMPLETED
        assert elapsed < 2.0


class TestBatchProcessingPerformance:
    """Test batch processing performance with AsyncScheduler."""

    @pytest.mark.asyncio
    async def test_batch_execution_performance(self):
        """Test performance of batch flow execution."""
        class BatchExecutor:
            async def execute(self, node, inputs):
                await asyncio.sleep(0.001)
                return {"batch_id": inputs.get("batch_id"), "result": node.id}

        flow = FlowDefinition(
            name="batch-flow",
            version="1.0",
            nodes=[
                Node(id="step1", type=NodeType.LLM, config=NodeConfig()),
                Node(id="step2", type=NodeType.LLM, config=NodeConfig(), dependencies=["step1"]),
            ],
            edges=[Edge(from_node="step1", to_node="step2")]
        )

        scheduler = AsyncScheduler(BatchExecutor(), max_parallel=50)

        num_batches = 100
        inputs_list = [{"batch_id": i} for i in range(num_batches)]

        start = time.time()
        results = await scheduler.execute_parallel(flow, inputs_list)
        elapsed = time.time() - start

        assert len(results) == num_batches
        assert all(r.status == RunStatus.COMPLETED for r in results)

        # 100 batches with high parallelism should complete quickly
        batches_per_second = num_batches / elapsed
        assert batches_per_second > 50, f"Batch throughput too low: {batches_per_second:.1f}/sec"

    @pytest.mark.asyncio
    async def test_batch_with_failures(self):
        """Test batch execution handles individual failures gracefully."""
        fail_indices = {10, 25, 50, 75}

        class FailingExecutor:
            async def execute(self, node, inputs):
                batch_id = inputs.get("batch_id", 0)
                if batch_id in fail_indices and node.id == "step2":
                    raise Exception(f"Batch {batch_id} failed")
                return {"result": "ok"}

        flow = FlowDefinition(
            name="batch-fail-flow",
            version="1.0",
            nodes=[
                Node(id="step1", type=NodeType.LLM, config=NodeConfig()),
                Node(id="step2", type=NodeType.LLM, config=NodeConfig(), dependencies=["step1"]),
            ],
            edges=[Edge(from_node="step1", to_node="step2")]
        )

        scheduler = AsyncScheduler(FailingExecutor(), max_parallel=20)

        num_batches = 100
        inputs_list = [{"batch_id": i} for i in range(num_batches)]

        results = await scheduler.execute_parallel(flow, inputs_list)

        assert len(results) == num_batches

        successful = [r for r in results if r.status == RunStatus.COMPLETED]
        failed = [r for r in results if r.status == RunStatus.FAILED]

        assert len(failed) == len(fail_indices)
        assert len(successful) == num_batches - len(fail_indices)


class TestSchedulerStress:
    """Stress tests for scheduler stability."""

    @pytest.mark.asyncio
    async def test_rapid_flow_submission(self):
        """Test rapid submission of many flows."""
        class SimpleExecutor:
            async def execute(self, node, inputs):
                return {"result": node.id}

        flow = FlowDefinition(
            name="rapid-flow",
            version="1.0",
            nodes=[Node(id="single", type=NodeType.LLM, config=NodeConfig())],
            edges=[]
        )

        scheduler = Scheduler(SimpleExecutor(), max_parallel=100)

        # Submit 500 flows rapidly
        num_flows = 500
        tasks = []

        start = time.time()
        for i in range(num_flows):
            task = scheduler.execute_flow(flow, inputs={"flow_id": i})
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start

        assert len(results) == num_flows
        assert all(r.status == RunStatus.COMPLETED for r in results)

        flows_per_second = num_flows / elapsed
        assert flows_per_second > 100, f"Submission rate too low: {flows_per_second:.1f}/sec"

    @pytest.mark.asyncio
    async def test_mixed_workload_stability(self):
        """Test stability under mixed workload patterns."""
        class MixedExecutor:
            async def execute(self, node, inputs):
                # Variable execution time
                delay = random.uniform(0.001, 0.02)
                await asyncio.sleep(delay)
                return {"result": node.id}

        # Create flows of varying complexity
        flows = []

        # Simple flow
        flows.append(FlowDefinition(
            name="simple",
            version="1.0",
            nodes=[Node(id="single", type=NodeType.LLM, config=NodeConfig())],
            edges=[]
        ))

        # Parallel flow
        parallel_nodes = [Node(id="source", type=NodeType.LLM, config=NodeConfig())]
        for i in range(5):
            parallel_nodes.append(Node(
                id=f"p_{i}",
                type=NodeType.LLM,
                config=NodeConfig(),
                dependencies=["source"]
            ))
        flows.append(FlowDefinition(
            name="parallel",
            version="1.0",
            nodes=parallel_nodes,
            edges=[]
        ))

        # Chain flow
        chain_nodes = []
        for i in range(5):
            deps = [f"c_{i-1}"] if i > 0 else []
            chain_nodes.append(Node(
                id=f"c_{i}",
                type=NodeType.LLM,
                config=NodeConfig(),
                dependencies=deps
            ))
        flows.append(FlowDefinition(
            name="chain",
            version="1.0",
            nodes=chain_nodes,
            edges=[]
        ))

        scheduler = Scheduler(MixedExecutor(), max_parallel=20)

        # Execute mixed workload
        tasks = []
        for _ in range(50):
            flow = random.choice(flows)
            tasks.append(scheduler.execute_flow(flow, inputs={}))

        results = await asyncio.gather(*tasks)

        assert len(results) == 50
        assert all(r.status == RunStatus.COMPLETED for r in results)


class TestMemoryBehavior:
    """Test memory behavior under load (basic tests without heavy monitoring)."""

    @pytest.mark.asyncio
    async def test_no_memory_leak_simple(self):
        """Basic test that many executions don't cause obvious issues."""
        class LightExecutor:
            async def execute(self, node, inputs):
                return {"result": "ok"}

        flow = FlowDefinition(
            name="memory-test",
            version="1.0",
            nodes=[
                Node(id="a", type=NodeType.LLM, config=NodeConfig()),
                Node(id="b", type=NodeType.LLM, config=NodeConfig(), dependencies=["a"]),
            ],
            edges=[Edge(from_node="a", to_node="b")]
        )

        scheduler = Scheduler(LightExecutor(), max_parallel=10)

        # Execute many times
        for _ in range(100):
            result = await scheduler.execute_flow(flow, inputs={})
            assert result.status == RunStatus.COMPLETED

        # Force garbage collection
        gc.collect()

        # If we get here without error, basic memory handling is working
        assert True

    @pytest.mark.asyncio
    async def test_large_result_handling(self):
        """Test handling of large results doesn't cause issues."""
        class LargeResultExecutor:
            async def execute(self, node, inputs):
                # Generate moderately large result
                large_data = list(range(10000))
                return {"data": large_data}

        flow = FlowDefinition(
            name="large-result",
            version="1.0",
            nodes=[
                Node(id="gen", type=NodeType.LLM, config=NodeConfig()),
                Node(id="use", type=NodeType.LLM, config=NodeConfig(), dependencies=["gen"]),
            ],
            edges=[Edge(from_node="gen", to_node="use")]
        )

        scheduler = Scheduler(LargeResultExecutor(), max_parallel=5)

        # Execute with large results
        for _ in range(20):
            result = await scheduler.execute_flow(flow, inputs={})
            assert result.status == RunStatus.COMPLETED

        gc.collect()
        assert True
