"""
Micro-Model Orchestrated RAG System

A revolutionary RAG system where every stage is powered by specialized Small
Language Models (SLMs), orchestrated through a dynamic computation graph.
"""

from .schemas import (
    NodeType,
    QueryIntent,
    Chunk,
    Document,
    RetrievalResult,
    RerankResult,
    RoutingDecision,
    StabilizedAnswer,
    GraphNode,
    ExecutionResult,
    Span,
    GuardrailResult,
    ModelInfo,
    SelectionConstraints,
    generate_id,
)
# Optional imports requiring torch/transformers
try:
    from .pipeline import MicroRAGPipeline, create_pipeline
    from .orchestrator import (
        ComponentRegistry,
        SLMRegistry,
        component_registry,
        slm_registry,
        DynamicComputationGraph,
        GraphBuilder,
        build_rag_graph,
        build_indexing_graph,
        PipelineTracer,
        get_tracer,
        set_tracer,
    )
    from .slm import (
        BaseSLM,
        ChunkerSLM,
        EmbedderSLM,
        RetrieverSLM,
        RerankerSLM,
        SummarizerSLM,
        CoTCompressorSLM,
        AnswerStabilizerSLM,
    )
    from .enterprise import (
        ModelSelector,
        FallbackChain,
        GuardrailEngine,
        RelevanceGuardrail,
        HallucinationGuardrail,
        get_metrics,
    )
    _HAS_TORCH = True
except ImportError:
    # Create stubs for when torch is not available
    MicroRAGPipeline = None
    create_pipeline = None
    ComponentRegistry = None
    SLMRegistry = None
    component_registry = None
    slm_registry = None
    DynamicComputationGraph = None
    GraphBuilder = None
    build_rag_graph = None
    build_indexing_graph = None
    PipelineTracer = None
    get_tracer = None
    set_tracer = None
    BaseSLM = None
    ChunkerSLM = None
    EmbedderSLM = None
    RetrieverSLM = None
    RerankerSLM = None
    SummarizerSLM = None
    CoTCompressorSLM = None
    AnswerStabilizerSLM = None
    ModelSelector = None
    FallbackChain = None
    GuardrailEngine = None
    RelevanceGuardrail = None
    HallucinationGuardrail = None
    get_metrics = None
    _HAS_TORCH = False

__version__ = "1.0.0"
__all__ = [
    # Schemas
    "NodeType",
    "QueryIntent",
    "Chunk",
    "Document",
    "RetrievalResult",
    "RerankResult",
    "RoutingDecision",
    "StabilizedAnswer",
    "GraphNode",
    "ExecutionResult",
    "Span",
    "GuardrailResult",
    "ModelInfo",
    "SelectionConstraints",
    "generate_id",
    # Pipeline
    "MicroRAGPipeline",
    "create_pipeline",
    # Orchestrator
    "ComponentRegistry",
    "SLMRegistry",
    "component_registry",
    "slm_registry",
    "DynamicComputationGraph",
    "GraphBuilder",
    "build_rag_graph",
    "build_indexing_graph",
    "PipelineTracer",
    "get_tracer",
    "set_tracer",
    # SLMs
    "BaseSLM",
    "ChunkerSLM",
    "EmbedderSLM",
    "RetrieverSLM",
    "RerankerSLM",
    "SummarizerSLM",
    "CoTCompressorSLM",
    "AnswerStabilizerSLM",
    # Enterprise
    "ModelSelector",
    "FallbackChain",
    "GuardrailEngine",
    "RelevanceGuardrail",
    "HallucinationGuardrail",
    "get_metrics",
]
