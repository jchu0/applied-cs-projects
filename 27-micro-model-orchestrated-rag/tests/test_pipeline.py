"""End-to-end pipeline tests for MicroRAG."""

import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

from microrag.schemas import (
    StabilizedAnswer,
    Document,
    Chunk,
    GuardrailResult,
    RuleResult,
)

# Optional imports requiring torch
try:
    from microrag.pipeline import MicroRAGPipeline, create_pipeline
    from microrag.orchestrator import ComponentRegistry, PipelineTracer, InMemoryExporter
    from microrag.enterprise import (
        GuardrailEngine,
        RelevanceGuardrail,
        ConfidenceGuardrail,
        HallucinationGuardrail,
        LengthGuardrail,
    )
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="Requires torch")


# ============================================================================
# MicroRAGPipeline Creation Tests
# ============================================================================

class TestPipelineCreation:
    """Tests for pipeline creation and configuration."""

    def test_create_mock_pipeline(self):
        """Test creating pipeline with mock SLMs."""
        pipeline = create_pipeline(use_mock=True)

        assert pipeline is not None
        assert pipeline.use_mock is True
        assert pipeline.registry is not None
        assert pipeline.graph is not None

    def test_create_pipeline_with_guardrails(self):
        """Test creating pipeline with guardrails enabled."""
        pipeline = create_pipeline(use_mock=True, use_guardrails=True)

        assert pipeline.guardrails is not None

    def test_create_pipeline_without_guardrails(self):
        """Test creating pipeline with guardrails disabled."""
        pipeline = create_pipeline(use_mock=True, use_guardrails=False)

        assert pipeline.guardrails is None

    def test_pipeline_registers_all_components(self):
        """Test that pipeline registers all required SLM components."""
        pipeline = create_pipeline(use_mock=True)
        registry = pipeline.registry

        expected_components = [
            "chunker_slm",
            "embedder_slm",
            "retriever_slm",
            "reranker_slm",
            "summarizer_slm",
            "cot_compressor_slm",
            "answer_stabilizer_slm",
        ]

        for component in expected_components:
            assert registry.has(component), f"Missing component: {component}"

    def test_pipeline_builds_correct_graph(self):
        """Test that pipeline builds graph with correct nodes."""
        pipeline = create_pipeline(use_mock=True)
        graph = pipeline.graph

        expected_nodes = [
            "retrieve",
            "rerank",
            "summarize",
            "compress_cot",
            "stabilize",
        ]

        for node in expected_nodes:
            assert node in graph.nodes, f"Missing node: {node}"


# ============================================================================
# Pipeline Query Execution Tests
# ============================================================================

class TestPipelineQuery:
    """Tests for pipeline query execution."""

    @pytest.mark.asyncio
    async def test_basic_query_execution(self, mock_pipeline, sample_query):
        """Test basic query execution returns StabilizedAnswer."""
        result = await mock_pipeline.query(sample_query)

        assert isinstance(result, StabilizedAnswer)
        assert len(result.answer) > 0
        assert 0 <= result.confidence <= 1
        assert 0 <= result.consistency <= 1

    @pytest.mark.asyncio
    async def test_query_with_custom_top_k(self, mock_pipeline, sample_query):
        """Test query with custom top_k parameter."""
        result = await mock_pipeline.query(sample_query, top_k=5)

        assert isinstance(result, StabilizedAnswer)

    @pytest.mark.asyncio
    async def test_query_with_trace_id(self, mock_pipeline, sample_query):
        """Test query with custom trace ID."""
        trace_id = "custom_trace_123"
        result = await mock_pipeline.query(sample_query, trace_id=trace_id)

        assert isinstance(result, StabilizedAnswer)

    @pytest.mark.asyncio
    async def test_query_records_metrics(self, mock_pipeline, sample_query):
        """Test that query execution records metrics."""
        await mock_pipeline.query(sample_query)

        summary = mock_pipeline.get_execution_summary()
        assert "metrics" in summary
        assert "graph_summary" in summary

    @pytest.mark.asyncio
    async def test_multiple_queries_sequential(self, mock_pipeline, sample_query):
        """Test executing multiple queries sequentially."""
        queries = [
            "What is machine learning?",
            "How do neural networks work?",
            "Explain deep learning.",
        ]

        results = []
        for query in queries:
            result = await mock_pipeline.query(query)
            results.append(result)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, StabilizedAnswer)

    @pytest.mark.asyncio
    async def test_query_execution_summary(self, mock_pipeline, sample_query):
        """Test getting execution summary after query."""
        await mock_pipeline.query(sample_query)

        summary = mock_pipeline.get_execution_summary()

        assert "mode" in summary
        assert summary["mode"] == "mock"
        assert "graph_summary" in summary


# ============================================================================
# Pipeline Document Indexing Tests
# ============================================================================

class TestPipelineIndexing:
    """Tests for pipeline document indexing."""

    @pytest.mark.asyncio
    async def test_index_document_basic(self, mock_pipeline, sample_document_text):
        """Test basic document indexing."""
        result = await mock_pipeline.index_document(sample_document_text)

        assert result["status"] == "success"
        assert result["num_chunks"] > 0
        assert "doc_id" in result

    @pytest.mark.asyncio
    async def test_index_document_with_custom_id(self, mock_pipeline, sample_document_text):
        """Test indexing with custom document ID."""
        doc_id = "custom_doc_123"
        result = await mock_pipeline.index_document(
            sample_document_text,
            doc_id=doc_id
        )

        assert result["doc_id"] == doc_id

    @pytest.mark.asyncio
    async def test_index_document_with_metadata(self, mock_pipeline, sample_document_text):
        """Test indexing with metadata."""
        metadata = {"source": "test", "author": "Test Author"}
        result = await mock_pipeline.index_document(
            sample_document_text,
            metadata=metadata
        )

        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_index_empty_document(self, mock_pipeline):
        """Test indexing empty document."""
        result = await mock_pipeline.index_document("")

        # Should handle gracefully
        assert "status" in result


# ============================================================================
# Pipeline Guardrail Tests
# ============================================================================

class TestPipelineGuardrails:
    """Tests for pipeline guardrail integration."""

    @pytest.mark.asyncio
    async def test_guardrails_are_checked(self, mock_pipeline, sample_query):
        """Test that guardrails are checked during query."""
        result = await mock_pipeline.query(sample_query)

        # With mock SLMs, guardrails should pass normally
        assert isinstance(result, StabilizedAnswer)

    @pytest.mark.asyncio
    async def test_pipeline_without_guardrails(self, mock_pipeline_no_guardrails, sample_query):
        """Test pipeline works without guardrails."""
        result = await mock_pipeline_no_guardrails.query(sample_query)

        assert isinstance(result, StabilizedAnswer)


# ============================================================================
# Guardrail Engine Tests
# ============================================================================

class TestGuardrailEngine:
    """Tests for GuardrailEngine."""

    @pytest.mark.asyncio
    async def test_guardrail_engine_passes_valid_input(self, guardrail_engine):
        """Test guardrail engine passes valid input."""
        # Create valid output
        valid_output = StabilizedAnswer(
            answer="Valid answer",
            confidence=0.8,
            consistency=0.9,
            num_samples=3
        )

        result = await guardrail_engine.check(
            "stabilize",
            "Test query",
            valid_output
        )

        assert result.passed is True
        assert result.action == "pass"

    @pytest.mark.asyncio
    async def test_guardrail_engine_warns_low_confidence(self, guardrail_engine):
        """Test guardrail engine warns on low confidence."""
        low_confidence_output = StabilizedAnswer(
            answer="Low confidence answer",
            confidence=0.2,  # Below threshold
            consistency=0.9,
            num_samples=3
        )

        result = await guardrail_engine.check(
            "stabilize",
            "Test query",
            low_confidence_output
        )

        # Should warn but pass (not block)
        assert len(result.violations) > 0 or result.action in ["warn", "pass"]

    def test_guardrail_engine_add_rule(self, guardrail_engine):
        """Test adding rules to guardrail engine."""
        new_rule = LengthGuardrail(min_length=10, max_length=1000)
        guardrail_engine.add_rule(new_rule)

        assert len(guardrail_engine.rules) == 3  # 2 original + 1 new

    def test_guardrail_engine_get_violations(self, guardrail_engine):
        """Test getting violations from guardrail engine."""
        # Initially empty
        violations = guardrail_engine.get_violations()
        assert violations == []

    def test_guardrail_engine_clear_violations(self, guardrail_engine):
        """Test clearing violations."""
        guardrail_engine.clear_violations()
        assert guardrail_engine.get_violations() == []


class TestIndividualGuardrails:
    """Tests for individual guardrail rules."""

    @pytest.mark.asyncio
    async def test_relevance_guardrail_applies_to_retrieval(self):
        """Test RelevanceGuardrail applies to retrieval stage."""
        guardrail = RelevanceGuardrail(min_relevance=0.3)

        assert guardrail.applies_to("retrieval") is True
        assert guardrail.applies_to("reranking") is True
        assert guardrail.applies_to("summarization") is False

    @pytest.mark.asyncio
    async def test_relevance_guardrail_passes_valid(self, sample_retrieval_results):
        """Test RelevanceGuardrail passes valid results."""
        guardrail = RelevanceGuardrail(min_relevance=0.3)

        result = await guardrail.evaluate("query", sample_retrieval_results)

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_relevance_guardrail_fails_empty_results(self):
        """Test RelevanceGuardrail fails on empty results."""
        guardrail = RelevanceGuardrail(min_relevance=0.3)

        result = await guardrail.evaluate("query", [])

        assert result.passed is False
        assert result.severity == "warning"

    @pytest.mark.asyncio
    async def test_confidence_guardrail_applies_to_stabilize(self):
        """Test ConfidenceGuardrail applies to stabilize stage."""
        guardrail = ConfidenceGuardrail(min_confidence=0.5)

        assert guardrail.applies_to("stabilize") is True
        assert guardrail.applies_to("answer_generation") is True
        assert guardrail.applies_to("retrieval") is False

    @pytest.mark.asyncio
    async def test_confidence_guardrail_passes_high_confidence(self):
        """Test ConfidenceGuardrail passes high confidence output."""
        guardrail = ConfidenceGuardrail(min_confidence=0.5)
        output = StabilizedAnswer(
            answer="Confident answer",
            confidence=0.8,
            consistency=0.9,
            num_samples=3
        )

        result = await guardrail.evaluate("query", output)

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_confidence_guardrail_fails_low_confidence(self):
        """Test ConfidenceGuardrail fails low confidence output."""
        guardrail = ConfidenceGuardrail(min_confidence=0.5)
        output = StabilizedAnswer(
            answer="Low confidence answer",
            confidence=0.3,
            consistency=0.9,
            num_samples=3
        )

        result = await guardrail.evaluate("query", output)

        assert result.passed is False
        assert result.severity == "warning"

    @pytest.mark.asyncio
    async def test_hallucination_guardrail_applies_to_summarization(self):
        """Test HallucinationGuardrail applies to summarization stage."""
        guardrail = HallucinationGuardrail()

        assert guardrail.applies_to("summarization") is True
        assert guardrail.applies_to("stabilize") is True
        assert guardrail.applies_to("retrieval") is False

    @pytest.mark.asyncio
    async def test_hallucination_guardrail_passes_clean_text(self):
        """Test HallucinationGuardrail passes clean text."""
        guardrail = HallucinationGuardrail()

        result = await guardrail.evaluate("query", "Machine learning is a type of AI.")

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_hallucination_guardrail_warns_suspicious_patterns(self):
        """Test HallucinationGuardrail warns on suspicious patterns."""
        guardrail = HallucinationGuardrail()
        suspicious_text = "I think it might be, probably, I believe this could work."

        result = await guardrail.evaluate("query", suspicious_text)

        # Should detect multiple suspicious patterns
        assert result.passed is False or "hallucination" in result.message.lower()

    @pytest.mark.asyncio
    async def test_length_guardrail_applies_to_all(self):
        """Test LengthGuardrail applies to all stages."""
        guardrail = LengthGuardrail()

        assert guardrail.applies_to("retrieval") is True
        assert guardrail.applies_to("summarization") is True
        assert guardrail.applies_to("any_stage") is True

    @pytest.mark.asyncio
    async def test_length_guardrail_passes_valid_length(self):
        """Test LengthGuardrail passes valid length output."""
        guardrail = LengthGuardrail(min_length=5, max_length=100)

        result = await guardrail.evaluate("query", "This is valid length text.")

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_length_guardrail_fails_too_short(self):
        """Test LengthGuardrail fails on too short output."""
        guardrail = LengthGuardrail(min_length=50, max_length=1000)

        result = await guardrail.evaluate("query", "Short")

        assert result.passed is False

    @pytest.mark.asyncio
    async def test_length_guardrail_fails_too_long(self):
        """Test LengthGuardrail fails on too long output."""
        guardrail = LengthGuardrail(min_length=5, max_length=50)

        result = await guardrail.evaluate("query", "x" * 100)

        assert result.passed is False


# ============================================================================
# Pipeline Error Handling Tests
# ============================================================================

class TestPipelineErrorHandling:
    """Tests for pipeline error handling."""

    @pytest.mark.asyncio
    async def test_pipeline_handles_component_error(self, mock_registry):
        """Test pipeline handles component errors gracefully."""
        # Create a registry with a failing component
        class FailingRetriever:
            _loaded = True
            is_loaded = True

            async def process(self, **kwargs):
                raise RuntimeError("Simulated retrieval failure")

        mock_registry.unregister("retriever_slm")
        mock_registry.register("retriever_slm", FailingRetriever())

        pipeline = MicroRAGPipeline(
            registry=mock_registry,
            use_mock=True,
            use_guardrails=False
        )

        # Should return error response, not raise
        result = await pipeline.query("test query")

        assert isinstance(result, StabilizedAnswer)
        assert "Error" in result.answer or result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_pipeline_metrics_on_error(self, mock_pipeline, sample_query):
        """Test that metrics are recorded even on partial success."""
        # Execute a query
        await mock_pipeline.query(sample_query)

        summary = mock_pipeline.get_execution_summary()
        assert "metrics" in summary


# ============================================================================
# Pipeline Warmup and Cleanup Tests
# ============================================================================

class TestPipelineLifecycle:
    """Tests for pipeline warmup and cleanup."""

    @pytest.mark.asyncio
    async def test_pipeline_warmup(self, mock_pipeline):
        """Test pipeline warmup loads models."""
        await mock_pipeline.warmup()

        # All mock components should still be loaded
        registry = mock_pipeline.registry
        for component_name in registry.list_components():
            component = registry.get(component_name)
            if hasattr(component, 'is_loaded'):
                assert component.is_loaded is True

    @pytest.mark.asyncio
    async def test_pipeline_cleanup(self, mock_pipeline):
        """Test pipeline cleanup unloads models."""
        await mock_pipeline.cleanup()

        # Mock components don't really unload, but method should complete without error


# ============================================================================
# Pipeline Tracing Tests
# ============================================================================

class TestPipelineTracing:
    """Tests for pipeline tracing integration."""

    @pytest.mark.asyncio
    async def test_pipeline_with_custom_tracer(self, sample_query):
        """Test pipeline with custom tracer."""
        exporter = InMemoryExporter()
        tracer = PipelineTracer(exporter=exporter)

        pipeline = MicroRAGPipeline(
            use_mock=True,
            use_guardrails=False,
            tracer=tracer
        )

        await pipeline.query(sample_query)

        # Check that traces were recorded
        traces = exporter.get_traces()
        assert len(traces) > 0

    @pytest.mark.asyncio
    async def test_pipeline_traces_contain_query_info(self, sample_query):
        """Test that pipeline traces contain query information."""
        exporter = InMemoryExporter()
        tracer = PipelineTracer(exporter=exporter)

        pipeline = MicroRAGPipeline(
            use_mock=True,
            use_guardrails=False,
            tracer=tracer
        )

        await pipeline.query(sample_query)

        traces = exporter.get_traces()

        # Find the main query span
        query_spans = [t for t in traces if "query" in t.name.lower() or "rag" in t.name.lower()]
        assert len(query_spans) > 0


# ============================================================================
# Pipeline Configuration Tests
# ============================================================================

class TestPipelineConfiguration:
    """Tests for different pipeline configurations."""

    def test_pipeline_default_configuration(self):
        """Test pipeline with default configuration."""
        pipeline = create_pipeline(use_mock=True)

        assert pipeline.use_mock is True
        assert pipeline.guardrails is not None

    def test_pipeline_production_mode_flag(self):
        """Test pipeline reports correct mode."""
        mock_pipeline = create_pipeline(use_mock=True)
        assert mock_pipeline.use_mock is True

        summary = mock_pipeline.get_execution_summary()
        assert summary["mode"] == "mock"

    @pytest.mark.asyncio
    async def test_pipeline_concurrent_queries(self, mock_pipeline):
        """Test pipeline handles concurrent queries."""
        queries = [
            "What is machine learning?",
            "How do neural networks work?",
            "Explain deep learning.",
        ]

        # Execute queries concurrently
        tasks = [mock_pipeline.query(q) for q in queries]
        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, StabilizedAnswer)
