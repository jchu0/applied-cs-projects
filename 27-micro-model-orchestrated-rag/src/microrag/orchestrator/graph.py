"""Dynamic computation graph for orchestrating SLM pipeline."""

from typing import Any
import time
import asyncio

from ..schemas import (
    NodeType,
    GraphNode,
    ExecutionResult,
    generate_id,
)
from .registry import ComponentRegistry


class DynamicComputationGraph:
    """Builds and executes dynamic computation graphs for RAG."""

    def __init__(self, registry: ComponentRegistry):
        """Initialize graph.

        Args:
            registry: Component registry for resolving components
        """
        self.registry = registry
        self.nodes: dict[str, GraphNode] = {}
        self.execution_history: list[ExecutionResult] = []

    def add_node(
        self,
        node_id: str,
        node_type: NodeType,
        component: str,
        config: dict = None,
        dependencies: list[str] = None,
        fallback: str = None
    ):
        """Add a node to the graph.

        Args:
            node_id: Unique node identifier
            node_type: Type of node
            component: Component name in registry
            config: Node configuration
            dependencies: List of dependency node IDs
            fallback: Optional fallback node ID
        """
        self.nodes[node_id] = GraphNode(
            id=node_id,
            node_type=node_type,
            component=component,
            config=config or {},
            dependencies=dependencies or [],
            fallback=fallback
        )

    async def execute(
        self,
        inputs: dict[str, Any],
        trace_id: str = None
    ) -> dict[str, Any]:
        """Execute graph with topological ordering.

        Args:
            inputs: Input values keyed by node ID
            trace_id: Optional trace ID for distributed tracing

        Returns:
            Results keyed by node ID
        """
        trace_id = trace_id or generate_id()

        # Build execution order
        execution_order = self._topological_sort()

        # Initialize results with inputs
        results = {**inputs}

        for node_id in execution_order:
            node = self.nodes[node_id]

            # Gather dependencies
            dep_results = {
                dep: results[dep]
                for dep in node.dependencies
                if dep in results
            }

            # Execute node
            result = await self._execute_node(node, dep_results, trace_id)
            results[node_id] = result.output
            self.execution_history.append(result)

        return results

    async def _execute_node(
        self,
        node: GraphNode,
        inputs: dict[str, Any],
        trace_id: str
    ) -> ExecutionResult:
        """Execute a single node with tracing.

        Args:
            node: Node to execute
            inputs: Input values from dependencies
            trace_id: Trace ID

        Returns:
            Execution result
        """
        start_time = time.time()

        try:
            # Execute based on node type
            if node.node_type == NodeType.MERGE:
                # Merge nodes don't need a component from registry
                output = self._merge_results(inputs)
            elif node.node_type == NodeType.SLM:
                component = self.registry.get(node.component)
                if hasattr(component, 'process'):
                    output = await component.process(**inputs, **node.config)
                else:
                    output = await component(**inputs, **node.config)
            elif node.node_type == NodeType.BRANCH:
                component = self.registry.get(node.component)
                output = await self._execute_branch(component, inputs, node.config)
            else:
                component = self.registry.get(node.component)
                output = await component(**inputs)

            latency = (time.time() - start_time) * 1000
            quality = await self._compute_quality(node, output)

            return ExecutionResult(
                node_id=node.id,
                output=output,
                latency_ms=latency,
                quality_score=quality,
                metadata={
                    "trace_id": trace_id,
                    "component": node.component,
                    "input_keys": list(inputs.keys())
                }
            )

        except Exception as e:
            # Try fallback
            if node.fallback:
                return await self._execute_fallback(node, inputs, trace_id, e)
            raise

    async def _execute_branch(
        self,
        component: Any,
        inputs: dict[str, Any],
        config: dict
    ) -> Any:
        """Execute branching logic."""
        condition = config.get("condition")
        branches = config.get("branches", {})

        # Evaluate condition
        result = await component.evaluate(condition, inputs)

        # Select branch
        branch_id = branches.get(result, branches.get("default"))
        if branch_id and branch_id in self.nodes:
            branch_node = self.nodes[branch_id]
            return await self._execute_node(branch_node, inputs, generate_id())

        return None

    def _merge_results(self, inputs: dict[str, Any]) -> dict:
        """Merge results from multiple branches."""
        merged = {}
        for key, value in inputs.items():
            if isinstance(value, dict):
                merged.update(value)
            else:
                merged[key] = value
        return merged

    async def _execute_fallback(
        self,
        node: GraphNode,
        inputs: dict[str, Any],
        trace_id: str,
        error: Exception
    ) -> ExecutionResult:
        """Execute fallback node."""
        fallback_node = self.nodes.get(node.fallback)
        if fallback_node:
            result = await self._execute_node(fallback_node, inputs, trace_id)
            result.metadata["fallback_from"] = node.id
            result.metadata["original_error"] = str(error)
            return result

        raise error

    async def _compute_quality(self, node: GraphNode, output: Any) -> float:
        """Compute quality score for output."""
        # Basic quality estimation
        if output is None:
            return 0.0
        if isinstance(output, list) and len(output) == 0:
            return 0.0
        return 0.8  # Default quality

    def _topological_sort(self) -> list[str]:
        """Topological sort of nodes.

        Returns:
            List of node IDs in execution order
        """
        visited = set()
        order = []

        def visit(node_id):
            if node_id in visited:
                return
            if node_id not in self.nodes:
                return

            visited.add(node_id)

            # Visit dependencies first
            for dep in self.nodes[node_id].dependencies:
                visit(dep)

            order.append(node_id)

        for node_id in self.nodes:
            visit(node_id)

        return order

    def get_execution_summary(self) -> dict:
        """Get summary of graph execution."""
        if not self.execution_history:
            return {}

        total_latency = sum(r.latency_ms for r in self.execution_history)
        avg_quality = sum(r.quality_score for r in self.execution_history) / len(self.execution_history)

        return {
            "total_latency_ms": total_latency,
            "avg_quality_score": avg_quality,
            "node_count": len(self.execution_history),
            "nodes": [
                {
                    "id": r.node_id,
                    "latency_ms": r.latency_ms,
                    "quality": r.quality_score
                }
                for r in self.execution_history
            ]
        }


class GraphBuilder:
    """Fluent API for building computation graphs."""

    def __init__(self, registry: ComponentRegistry):
        """Initialize builder.

        Args:
            registry: Component registry
        """
        self.graph = DynamicComputationGraph(registry)

    def slm(
        self,
        node_id: str,
        component: str,
        dependencies: list[str] = None,
        fallback: str = None,
        **config
    ) -> 'GraphBuilder':
        """Add SLM node.

        Args:
            node_id: Node identifier
            component: Component name
            dependencies: Dependency node IDs
            fallback: Fallback node ID
            **config: Node configuration

        Returns:
            Self for chaining
        """
        self.graph.add_node(
            node_id,
            NodeType.SLM,
            component,
            config,
            dependencies,
            fallback
        )
        return self

    def branch(
        self,
        node_id: str,
        condition: str,
        branches: dict[str, str],
        dependencies: list[str] = None
    ) -> 'GraphBuilder':
        """Add branch node.

        Args:
            node_id: Node identifier
            condition: Condition to evaluate
            branches: Map of condition results to node IDs
            dependencies: Dependency node IDs

        Returns:
            Self for chaining
        """
        self.graph.add_node(
            node_id,
            NodeType.BRANCH,
            "branch_executor",
            {"condition": condition, "branches": branches},
            dependencies
        )
        return self

    def merge(
        self,
        node_id: str,
        dependencies: list[str]
    ) -> 'GraphBuilder':
        """Add merge node.

        Args:
            node_id: Node identifier
            dependencies: Nodes to merge

        Returns:
            Self for chaining
        """
        self.graph.add_node(
            node_id,
            NodeType.MERGE,
            "merge",
            {},
            dependencies
        )
        return self

    def build(self) -> DynamicComputationGraph:
        """Build and return the graph.

        Returns:
            Constructed computation graph
        """
        return self.graph


def build_rag_graph(registry: ComponentRegistry) -> DynamicComputationGraph:
    """Build standard RAG computation graph.

    Args:
        registry: Component registry

    Returns:
        RAG computation graph
    """
    return (
        GraphBuilder(registry)
        .slm("retrieve", "retriever_slm", dependencies=["query"])
        .slm("rerank", "reranker_slm", dependencies=["query", "retrieve"])
        .slm("summarize", "summarizer_slm", dependencies=["query", "rerank"])
        .slm("compress_cot", "cot_compressor_slm", dependencies=["query", "summarize"])
        .slm("stabilize", "answer_stabilizer_slm", dependencies=["query", "compress_cot"])
        .build()
    )


def build_indexing_graph(registry: ComponentRegistry) -> DynamicComputationGraph:
    """Build document indexing graph.

    Args:
        registry: Component registry

    Returns:
        Indexing computation graph
    """
    return (
        GraphBuilder(registry)
        .slm("chunk", "chunker_slm", dependencies=["document"])
        .slm("embed", "embedder_slm", dependencies=["chunk"])
        .build()
    )
