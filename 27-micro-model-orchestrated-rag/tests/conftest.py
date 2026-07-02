"""Pytest fixtures for Micro-Model Orchestrated RAG tests."""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import pytest
import numpy as np
from typing import List

# Import schemas (no torch required)
from microrag.schemas import (
    Chunk,
    Document,
    RetrievalResult,
    RerankResult,
    RoutingDecision,
    StabilizedAnswer,
    GraphNode,
    ExecutionResult,
    NodeType,
    Span,
    GuardrailResult,
    RuleResult,
)

# Import torch-dependent components (optional)
try:
    from microrag.orchestrator import (
        ComponentRegistry,
        DynamicComputationGraph,
        GraphBuilder,
        PipelineTracer,
        InMemoryExporter,
    )
    from microrag.slm import (
        MockChunkerSLM,
        MockEmbedderSLM,
        MockRetrieverSLM,
        MockRerankerSLM,
        MockSummarizerSLM,
        MockCoTCompressorSLM,
        MockAnswerStabilizerSLM,
        BaseSLM,
    )
    from microrag.enterprise import (
        GuardrailEngine,
        RelevanceGuardrail,
        ConfidenceGuardrail,
        HallucinationGuardrail,
        LengthGuardrail,
    )
    from microrag.pipeline import MicroRAGPipeline, create_pipeline
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    ComponentRegistry = None
    DynamicComputationGraph = None
    GraphBuilder = None
    PipelineTracer = None
    InMemoryExporter = None
    MockChunkerSLM = None
    MockEmbedderSLM = None
    MockRetrieverSLM = None
    MockRerankerSLM = None
    MockSummarizerSLM = None
    MockCoTCompressorSLM = None
    MockAnswerStabilizerSLM = None
    BaseSLM = None
    GuardrailEngine = None
    RelevanceGuardrail = None
    ConfidenceGuardrail = None
    HallucinationGuardrail = None
    LengthGuardrail = None
    MicroRAGPipeline = None
    create_pipeline = None


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_document_text():
    """Sample document text for testing."""
    return """
    Machine learning is a subset of artificial intelligence that enables systems
    to learn and improve from experience without being explicitly programmed.
    It focuses on developing computer programs that can access data and use it
    to learn for themselves.

    The process of learning begins with observations or data, such as examples,
    direct experience, or instruction. It looks for patterns in data and makes
    better decisions in the future based on the examples provided.

    Deep learning is a subset of machine learning that uses neural networks with
    many layers. These deep neural networks can learn complex patterns from large
    amounts of data, enabling breakthroughs in image recognition, natural language
    processing, and more.
    """


@pytest.fixture
def sample_query():
    """Sample query for testing."""
    return "What is machine learning and how does it work?"


@pytest.fixture
def sample_chunks() -> List[Chunk]:
    """Sample chunks for testing."""
    return [
        Chunk(
            content="Machine learning is a subset of artificial intelligence.",
            start_idx=0,
            end_idx=50,
            chunk_type="paragraph",
            semantic_score=0.85,
            metadata={"chunk_idx": 0}
        ),
        Chunk(
            content="The process of learning begins with observations or data.",
            start_idx=51,
            end_idx=100,
            chunk_type="paragraph",
            semantic_score=0.78,
            metadata={"chunk_idx": 1}
        ),
        Chunk(
            content="Deep learning uses neural networks with many layers.",
            start_idx=101,
            end_idx=150,
            chunk_type="paragraph",
            semantic_score=0.82,
            metadata={"chunk_idx": 2}
        ),
    ]


@pytest.fixture
def sample_documents() -> List[Document]:
    """Sample documents for testing."""
    return [
        Document(
            id="doc_1",
            content="Machine learning enables systems to learn from data.",
            metadata={"source": "article", "author": "John Doe"}
        ),
        Document(
            id="doc_2",
            content="Neural networks are inspired by the human brain.",
            metadata={"source": "textbook", "author": "Jane Smith"}
        ),
        Document(
            id="doc_3",
            content="Deep learning has revolutionized computer vision.",
            metadata={"source": "paper", "year": 2023}
        ),
    ]


@pytest.fixture
def sample_retrieval_results(sample_documents) -> List[RetrievalResult]:
    """Sample retrieval results for testing."""
    return [
        RetrievalResult(
            document=sample_documents[0],
            score=0.95,
            retriever_type="dense",
            rank=0
        ),
        RetrievalResult(
            document=sample_documents[1],
            score=0.85,
            retriever_type="dense",
            rank=1
        ),
        RetrievalResult(
            document=sample_documents[2],
            score=0.75,
            retriever_type="sparse",
            rank=2
        ),
    ]


@pytest.fixture
def sample_rerank_results(sample_documents) -> List[RerankResult]:
    """Sample rerank results for testing."""
    return [
        RerankResult(
            document=sample_documents[0],
            original_rank=0,
            new_rank=0,
            relevance_score=0.92,
            features={"cross_encoder_score": 0.92}
        ),
        RerankResult(
            document=sample_documents[2],
            original_rank=2,
            new_rank=1,
            relevance_score=0.88,
            features={"cross_encoder_score": 0.88}
        ),
        RerankResult(
            document=sample_documents[1],
            original_rank=1,
            new_rank=2,
            relevance_score=0.80,
            features={"cross_encoder_score": 0.80}
        ),
    ]


@pytest.fixture
def sample_stabilized_answer():
    """Sample stabilized answer for testing."""
    return StabilizedAnswer(
        answer="Machine learning is a type of AI that learns from data.",
        confidence=0.85,
        consistency=0.90,
        num_samples=3,
        reasoning="Selected from 3 consistent samples using semantic voting."
    )


# ============================================================================
# Registry Fixtures
# ============================================================================

requires_torch = pytest.mark.skipif(not _HAS_TORCH, reason="Requires ML extras (transformers)")

@pytest.fixture
def mock_registry():
    """Create a component registry with mock SLMs."""
    if not _HAS_TORCH:
        pytest.skip("Requires ML extras (transformers)")
    registry = ComponentRegistry()

    # Register mock components
    registry.register("chunker_slm", MockChunkerSLM())
    registry.register("embedder_slm", MockEmbedderSLM())
    registry.register("retriever_slm", MockRetrieverSLM())
    registry.register("reranker_slm", MockRerankerSLM())
    registry.register("summarizer_slm", MockSummarizerSLM())
    registry.register("cot_compressor_slm", MockCoTCompressorSLM())
    registry.register("answer_stabilizer_slm", MockAnswerStabilizerSLM())

    return registry


@pytest.fixture
def empty_registry():
    """Create an empty component registry."""
    if not _HAS_TORCH:
        pytest.skip("Requires ML extras (transformers)")
    return ComponentRegistry()


# ============================================================================
# Graph Fixtures
# ============================================================================

@pytest.fixture
def simple_graph(mock_registry):
    """Create a simple computation graph for testing."""
    return (
        GraphBuilder(mock_registry)
        .slm("retrieve", "retriever_slm", dependencies=["query"])
        .slm("rerank", "reranker_slm", dependencies=["query", "retrieve"])
        .build()
    )


@pytest.fixture
def full_rag_graph(mock_registry):
    """Create a full RAG computation graph for testing."""
    return (
        GraphBuilder(mock_registry)
        .slm("retrieve", "retriever_slm", dependencies=["query"])
        .slm("rerank", "reranker_slm", dependencies=["query", "retrieve"])
        .slm("summarize", "summarizer_slm", dependencies=["query", "rerank"])
        .slm("compress_cot", "cot_compressor_slm", dependencies=["query", "summarize"])
        .slm("stabilize", "answer_stabilizer_slm", dependencies=["query", "compress_cot"])
        .build()
    )


@pytest.fixture
def indexing_graph(mock_registry):
    """Create an indexing computation graph for testing."""
    return (
        GraphBuilder(mock_registry)
        .slm("chunk", "chunker_slm", dependencies=["document"])
        .slm("embed", "embedder_slm", dependencies=["chunk"])
        .build()
    )


# ============================================================================
# Tracing Fixtures
# ============================================================================

@pytest.fixture
def in_memory_exporter():
    """Create an in-memory trace exporter for testing."""
    if not _HAS_TORCH:
        pytest.skip("Requires ML extras (transformers)")
    return InMemoryExporter()


@pytest.fixture
def tracer(in_memory_exporter):
    """Create a pipeline tracer with in-memory exporter."""
    if not _HAS_TORCH:
        pytest.skip("Requires ML extras (transformers)")
    return PipelineTracer(exporter=in_memory_exporter)


# ============================================================================
# Guardrail Fixtures
# ============================================================================

@pytest.fixture
def guardrail_engine():
    """Create a guardrail engine with default rules."""
    if not _HAS_TORCH:
        pytest.skip("Requires ML extras (transformers)")
    return GuardrailEngine([
        RelevanceGuardrail(min_relevance=0.3),
        ConfidenceGuardrail(min_confidence=0.5),
    ])


@pytest.fixture
def strict_guardrail_engine():
    """Create a guardrail engine with strict rules."""
    if not _HAS_TORCH:
        pytest.skip("Requires ML extras (transformers)")
    return GuardrailEngine([
        RelevanceGuardrail(min_relevance=0.7),
        ConfidenceGuardrail(min_confidence=0.8),
        HallucinationGuardrail(max_hallucination_ratio=0.05),
        LengthGuardrail(min_length=20, max_length=5000),
    ])


# ============================================================================
# Pipeline Fixtures
# ============================================================================

@pytest.fixture
def mock_pipeline():
    """Create a pipeline with mock SLMs."""
    if not _HAS_TORCH:
        pytest.skip("Requires ML extras (transformers)")
    return create_pipeline(use_mock=True, use_guardrails=True)


@pytest.fixture
def mock_pipeline_no_guardrails():
    """Create a pipeline with mock SLMs and no guardrails."""
    if not _HAS_TORCH:
        pytest.skip("Requires ML extras (transformers)")
    return create_pipeline(use_mock=True, use_guardrails=False)


# ============================================================================
# Embedding Fixtures
# ============================================================================

@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing."""
    np.random.seed(42)
    return np.random.randn(5, 384).astype(np.float32)


@pytest.fixture
def normalized_embeddings(sample_embeddings):
    """Normalized sample embeddings."""
    norms = np.linalg.norm(sample_embeddings, axis=1, keepdims=True)
    return sample_embeddings / (norms + 1e-9)


# ============================================================================
# Helper Functions
# ============================================================================

class MockAsyncComponent:
    """Mock async component for testing graph execution."""

    def __init__(self, return_value=None, should_fail=False):
        self.return_value = return_value
        self.should_fail = should_fail
        self.call_count = 0
        self.last_kwargs = None

    async def process(self, **kwargs):
        self.call_count += 1
        self.last_kwargs = kwargs

        if self.should_fail:
            raise RuntimeError("Mock component failure")

        return self.return_value if self.return_value is not None else kwargs

    async def __call__(self, **kwargs):
        return await self.process(**kwargs)


@pytest.fixture
def mock_async_component():
    """Create a mock async component factory."""
    def _create(return_value=None, should_fail=False):
        return MockAsyncComponent(return_value, should_fail)
    return _create
