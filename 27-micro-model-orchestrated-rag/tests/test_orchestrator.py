"""Tests for orchestrator components: graph, registry, and tracing."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from microrag.schemas import (
    NodeType,
    GraphNode,
    ExecutionResult,
    Span,
)
from microrag.orchestrator import (
    ComponentRegistry,
    SLMRegistry,
    DynamicComputationGraph,
    GraphBuilder,
    PipelineTracer,
    InMemoryExporter,
    ConsoleExporter,
    JSONExporter,
    TracingContext,
    build_rag_graph,
    build_indexing_graph,
)


# ============================================================================
# ComponentRegistry Tests
# ============================================================================

class TestComponentRegistry:
    """Tests for ComponentRegistry."""

    def test_register_and_get_component(self, empty_registry):
        """Test registering and retrieving a component."""
        mock_component = MagicMock()
        empty_registry.register("test_component", mock_component)

        retrieved = empty_registry.get("test_component")
        assert retrieved is mock_component

    def test_register_with_metadata(self, empty_registry):
        """Test registering component with metadata."""
        mock_component = MagicMock()
        metadata = {"task": "test", "version": "1.0"}

        empty_registry.register("test_component", mock_component, metadata)

        assert empty_registry.get_metadata("test_component") == metadata

    def test_get_nonexistent_component_raises_error(self, empty_registry):
        """Test that getting a non-existent component raises KeyError."""
        with pytest.raises(KeyError) as exc_info:
            empty_registry.get("nonexistent")

        assert "nonexistent" in str(exc_info.value)

    def test_has_component(self, empty_registry):
        """Test checking component existence."""
        mock_component = MagicMock()
        empty_registry.register("test_component", mock_component)

        assert empty_registry.has("test_component") is True
        assert empty_registry.has("nonexistent") is False

    def test_list_components(self, empty_registry):
        """Test listing all registered components."""
        empty_registry.register("comp1", MagicMock())
        empty_registry.register("comp2", MagicMock())

        components = empty_registry.list_components()
        assert "comp1" in components
        assert "comp2" in components
        assert len(components) == 2

    def test_unregister_component(self, empty_registry):
        """Test unregistering a component."""
        empty_registry.register("test_component", MagicMock())
        empty_registry.unregister("test_component")

        assert empty_registry.has("test_component") is False

    def test_clear_registry(self, empty_registry):
        """Test clearing all components."""
        empty_registry.register("comp1", MagicMock())
        empty_registry.register("comp2", MagicMock())

        empty_registry.clear()

        assert empty_registry.list_components() == []


class TestSLMRegistry:
    """Tests for SLMRegistry specialized registry."""

    def test_register_slm_with_task_metadata(self):
        """Test registering SLM with task-specific metadata."""
        registry = SLMRegistry()
        mock_slm = MagicMock()

        registry.register_slm(
            "test_slm",
            mock_slm,
            task="embedding",
            base_model="test-model"
        )

        metadata = registry.get_metadata("test_slm")
        assert metadata["task"] == "embedding"
        assert metadata["base_model"] == "test-model"

    def test_get_slms_by_task(self):
        """Test filtering SLMs by task type."""
        registry = SLMRegistry()

        registry.register_slm("emb1", MagicMock(), task="embedding", base_model="m1")
        registry.register_slm("emb2", MagicMock(), task="embedding", base_model="m2")
        registry.register_slm("ret1", MagicMock(), task="retrieval", base_model="m3")

        embedding_slms = registry.get_by_task("embedding")
        assert len(embedding_slms) == 2
        assert "emb1" in embedding_slms
        assert "emb2" in embedding_slms

        retrieval_slms = registry.get_by_task("retrieval")
        assert len(retrieval_slms) == 1
        assert "ret1" in retrieval_slms


# ============================================================================
# DynamicComputationGraph Tests
# ============================================================================

class TestDynamicComputationGraph:
    """Tests for DynamicComputationGraph."""

    def test_add_node(self, mock_registry):
        """Test adding nodes to the graph."""
        graph = DynamicComputationGraph(mock_registry)

        graph.add_node(
            node_id="test_node",
            node_type=NodeType.SLM,
            component="retriever_slm",
            config={"top_k": 10},
            dependencies=["query"]
        )

        assert "test_node" in graph.nodes
        assert graph.nodes["test_node"].node_type == NodeType.SLM
        assert graph.nodes["test_node"].component == "retriever_slm"

    def test_topological_sort_simple(self, mock_registry):
        """Test topological sort with simple dependencies."""
        graph = DynamicComputationGraph(mock_registry)

        graph.add_node("a", NodeType.SLM, "retriever_slm", dependencies=[])
        graph.add_node("b", NodeType.SLM, "retriever_slm", dependencies=["a"])
        graph.add_node("c", NodeType.SLM, "retriever_slm", dependencies=["b"])

        order = graph._topological_sort()

        # a should come before b, b before c
        assert order.index("a") < order.index("b")
        assert order.index("b") < order.index("c")

    def test_topological_sort_diamond(self, mock_registry):
        """Test topological sort with diamond dependency pattern."""
        graph = DynamicComputationGraph(mock_registry)

        # Diamond pattern: a -> b, a -> c, b -> d, c -> d
        graph.add_node("a", NodeType.SLM, "retriever_slm", dependencies=[])
        graph.add_node("b", NodeType.SLM, "retriever_slm", dependencies=["a"])
        graph.add_node("c", NodeType.SLM, "retriever_slm", dependencies=["a"])
        graph.add_node("d", NodeType.SLM, "retriever_slm", dependencies=["b", "c"])

        order = graph._topological_sort()

        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    @pytest.mark.asyncio
    async def test_execute_simple_graph(self, simple_graph, sample_query):
        """Test executing a simple graph."""
        results = await simple_graph.execute({"query": sample_query})

        assert "query" in results
        assert "retrieve" in results
        assert "rerank" in results

    @pytest.mark.asyncio
    async def test_execute_full_rag_graph(self, full_rag_graph, sample_query):
        """Test executing a full RAG graph."""
        results = await full_rag_graph.execute({"query": sample_query})

        assert "query" in results
        assert "retrieve" in results
        assert "rerank" in results
        assert "summarize" in results
        assert "compress_cot" in results
        assert "stabilize" in results

    @pytest.mark.asyncio
    async def test_execute_records_history(self, simple_graph, sample_query):
        """Test that execution records history."""
        await simple_graph.execute({"query": sample_query})

        assert len(simple_graph.execution_history) > 0
        for result in simple_graph.execution_history:
            assert isinstance(result, ExecutionResult)
            assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_get_execution_summary(self, simple_graph, sample_query):
        """Test getting execution summary."""
        await simple_graph.execute({"query": sample_query})

        summary = simple_graph.get_execution_summary()

        assert "total_latency_ms" in summary
        assert "avg_quality_score" in summary
        assert "node_count" in summary
        assert "nodes" in summary
        assert summary["node_count"] > 0

    @pytest.mark.asyncio
    async def test_merge_node_combines_results(self, mock_registry):
        """Test that merge nodes combine results from multiple branches."""
        graph = DynamicComputationGraph(mock_registry)

        # Create two branches that merge
        graph.add_node("branch1", NodeType.SLM, "retriever_slm", dependencies=["query"])
        graph.add_node("branch2", NodeType.SLM, "retriever_slm", dependencies=["query"])
        graph.add_node("merged", NodeType.MERGE, "merge", dependencies=["branch1", "branch2"])

        results = await graph.execute({"query": "test query"})

        assert "merged" in results

    def test_compute_quality_returns_default(self, mock_registry):
        """Test quality computation returns default values."""
        graph = DynamicComputationGraph(mock_registry)

        # Test with None output
        node = GraphNode(
            id="test",
            node_type=NodeType.SLM,
            component="test",
            config={},
            dependencies=[]
        )

        quality = asyncio.get_event_loop().run_until_complete(
            graph._compute_quality(node, None)
        )
        assert quality == 0.0

        # Test with empty list
        quality = asyncio.get_event_loop().run_until_complete(
            graph._compute_quality(node, [])
        )
        assert quality == 0.0

        # Test with valid output
        quality = asyncio.get_event_loop().run_until_complete(
            graph._compute_quality(node, {"result": "value"})
        )
        assert quality == 0.8  # Default quality


# ============================================================================
# GraphBuilder Tests
# ============================================================================

class TestGraphBuilder:
    """Tests for GraphBuilder fluent API."""

    def test_builder_creates_slm_nodes(self, mock_registry):
        """Test that builder creates SLM nodes correctly."""
        graph = (
            GraphBuilder(mock_registry)
            .slm("node1", "retriever_slm")
            .build()
        )

        assert "node1" in graph.nodes
        assert graph.nodes["node1"].node_type == NodeType.SLM

    def test_builder_chains_multiple_nodes(self, mock_registry):
        """Test fluent chaining of multiple nodes."""
        graph = (
            GraphBuilder(mock_registry)
            .slm("node1", "retriever_slm")
            .slm("node2", "reranker_slm", dependencies=["node1"])
            .slm("node3", "summarizer_slm", dependencies=["node2"])
            .build()
        )

        assert len(graph.nodes) == 3
        assert graph.nodes["node2"].dependencies == ["node1"]
        assert graph.nodes["node3"].dependencies == ["node2"]

    def test_builder_branch_node(self, mock_registry):
        """Test creating branch nodes."""
        # First register a branch executor
        mock_registry.register("branch_executor", MagicMock())

        graph = (
            GraphBuilder(mock_registry)
            .branch(
                "branch_node",
                condition="is_complex",
                branches={"true": "complex_path", "false": "simple_path"},
                dependencies=["input"]
            )
            .build()
        )

        assert "branch_node" in graph.nodes
        assert graph.nodes["branch_node"].node_type == NodeType.BRANCH

    def test_builder_merge_node(self, mock_registry):
        """Test creating merge nodes."""
        graph = (
            GraphBuilder(mock_registry)
            .slm("path1", "retriever_slm")
            .slm("path2", "retriever_slm")
            .merge("merged", dependencies=["path1", "path2"])
            .build()
        )

        assert "merged" in graph.nodes
        assert graph.nodes["merged"].node_type == NodeType.MERGE
        assert set(graph.nodes["merged"].dependencies) == {"path1", "path2"}

    def test_builder_with_fallback(self, mock_registry):
        """Test creating nodes with fallback configuration."""
        graph = (
            GraphBuilder(mock_registry)
            .slm("primary", "retriever_slm", fallback="backup")
            .slm("backup", "retriever_slm")
            .build()
        )

        assert graph.nodes["primary"].fallback == "backup"


# ============================================================================
# Graph Factory Functions Tests
# ============================================================================

class TestGraphFactoryFunctions:
    """Tests for graph factory functions."""

    def test_build_rag_graph(self, mock_registry):
        """Test RAG graph factory function."""
        graph = build_rag_graph(mock_registry)

        assert "retrieve" in graph.nodes
        assert "rerank" in graph.nodes
        assert "summarize" in graph.nodes
        assert "compress_cot" in graph.nodes
        assert "stabilize" in graph.nodes

    def test_build_indexing_graph(self, mock_registry):
        """Test indexing graph factory function."""
        graph = build_indexing_graph(mock_registry)

        assert "chunk" in graph.nodes
        assert "embed" in graph.nodes


# ============================================================================
# PipelineTracer Tests
# ============================================================================

class TestPipelineTracer:
    """Tests for PipelineTracer."""

    def test_start_trace_generates_id(self, tracer):
        """Test that start_trace generates a valid trace ID."""
        trace_id = tracer.start_trace()

        assert trace_id is not None
        assert len(trace_id) == 12  # generate_id produces 12 char hex

    def test_trace_step_context_manager(self, tracer, in_memory_exporter):
        """Test trace_step context manager."""
        trace_id = tracer.start_trace()

        with tracer.trace_step("test_step", trace_id) as span:
            assert span.name == "test_step"
            assert span.trace_id == trace_id

        # Check that span was exported
        exported_spans = in_memory_exporter.get_traces()
        assert len(exported_spans) == 1
        assert exported_spans[0].name == "test_step"
        assert exported_spans[0].duration_ms is not None

    def test_trace_step_with_parent(self, tracer, in_memory_exporter):
        """Test trace_step with parent span."""
        trace_id = tracer.start_trace()

        with tracer.trace_step("parent", trace_id) as parent_span:
            with tracer.trace_step("child", trace_id, parent_span.span_id) as child_span:
                assert child_span.parent_id == parent_span.span_id

    def test_add_attribute(self, tracer):
        """Test adding attributes to spans."""
        trace_id = tracer.start_trace()

        with tracer.trace_step("test", trace_id) as span:
            tracer.add_attribute(span, "key", "value")
            tracer.add_attribute(span, "count", 42)

            assert span.attributes["key"] == "value"
            assert span.attributes["count"] == 42

    def test_record_error(self, tracer):
        """Test recording errors on spans."""
        trace_id = tracer.start_trace()

        with tracer.trace_step("test", trace_id) as span:
            error = ValueError("Test error")
            tracer.record_error(span, error)

            assert span.attributes["error"] is True
            assert span.attributes["error_type"] == "ValueError"
            assert span.attributes["error_message"] == "Test error"

    def test_get_active_spans(self, tracer):
        """Test getting active spans."""
        trace_id = tracer.start_trace()

        with tracer.trace_step("outer", trace_id) as outer_span:
            active = tracer.get_active_spans()
            assert len(active) == 1
            assert active[0].name == "outer"

            with tracer.trace_step("inner", trace_id) as inner_span:
                active = tracer.get_active_spans()
                assert len(active) == 2


class TestTraceExporters:
    """Tests for trace exporters."""

    def test_in_memory_exporter(self):
        """Test InMemoryExporter stores spans."""
        exporter = InMemoryExporter()
        span = Span(
            trace_id="trace1",
            span_id="span1",
            parent_id=None,
            name="test",
            start_time=1.0,
            end_time=2.0,
            duration_ms=1000.0
        )

        exporter.export(span)

        traces = exporter.get_traces()
        assert len(traces) == 1
        assert traces[0].name == "test"

    def test_in_memory_exporter_clear(self):
        """Test clearing InMemoryExporter."""
        exporter = InMemoryExporter()
        span = Span(
            trace_id="trace1",
            span_id="span1",
            parent_id=None,
            name="test",
            start_time=1.0
        )

        exporter.export(span)
        exporter.clear()

        assert len(exporter.get_traces()) == 0

    def test_json_exporter_stores_spans(self, tmp_path):
        """Test JSONExporter stores spans."""
        filepath = tmp_path / "traces.json"
        exporter = JSONExporter(str(filepath))

        span = Span(
            trace_id="trace1",
            span_id="span1",
            parent_id=None,
            name="test",
            start_time=1.0,
            end_time=2.0,
            duration_ms=1000.0,
            attributes={"key": "value"}
        )

        exporter.export(span)
        exporter.flush()

        assert filepath.exists()


class TestTracingContext:
    """Tests for TracingContext."""

    @pytest.mark.asyncio
    async def test_tracing_context_generates_trace_id(self, tracer):
        """Test that TracingContext generates trace ID on entry."""
        context = TracingContext(tracer)

        async with context:
            assert context.trace_id is not None

    @pytest.mark.asyncio
    async def test_tracing_context_step(self, tracer, in_memory_exporter):
        """Test using step within TracingContext."""
        context = TracingContext(tracer)

        async with context:
            with context.step("step1") as span:
                assert span.trace_id == context.trace_id

        exported = in_memory_exporter.get_traces()
        assert len(exported) == 1
        assert exported[0].name == "step1"


# ============================================================================
# Fallback Tests
# ============================================================================

class TestFallbackExecution:
    """Tests for fallback execution in graphs."""

    @pytest.mark.asyncio
    async def test_fallback_on_error(self, mock_registry, mock_async_component):
        """Test that fallback is executed when primary fails."""
        # Create a failing component and a backup
        failing_component = mock_async_component(should_fail=True)
        backup_component = mock_async_component(return_value={"backup": True})

        mock_registry.register("failing_slm", failing_component)
        mock_registry.register("backup_slm", backup_component)

        graph = (
            GraphBuilder(mock_registry)
            .slm("primary", "failing_slm", dependencies=["query"], fallback="backup")
            .slm("backup", "backup_slm", dependencies=["query"])
            .build()
        )

        results = await graph.execute({"query": "test"})

        # The backup should have been called
        assert backup_component.call_count >= 1

    @pytest.mark.asyncio
    async def test_error_without_fallback_raises(self, mock_registry, mock_async_component):
        """Test that error without fallback raises exception."""
        failing_component = mock_async_component(should_fail=True)
        mock_registry.register("failing_slm", failing_component)

        graph = (
            GraphBuilder(mock_registry)
            .slm("primary", "failing_slm", dependencies=["query"])
            .build()
        )

        with pytest.raises(RuntimeError):
            await graph.execute({"query": "test"})
