"""Orchestrator module for Micro-Model RAG."""

from .registry import (
    ComponentRegistry,
    SLMRegistry,
    component_registry,
    slm_registry,
)
from .graph import (
    DynamicComputationGraph,
    GraphBuilder,
    build_rag_graph,
    build_indexing_graph,
)
from .tracing import (
    PipelineTracer,
    TraceExporter,
    ConsoleExporter,
    JSONExporter,
    InMemoryExporter,
    TracingContext,
    get_tracer,
    set_tracer,
)

__all__ = [
    # Registry
    "ComponentRegistry",
    "SLMRegistry",
    "component_registry",
    "slm_registry",
    # Graph
    "DynamicComputationGraph",
    "GraphBuilder",
    "build_rag_graph",
    "build_indexing_graph",
    # Tracing
    "PipelineTracer",
    "TraceExporter",
    "ConsoleExporter",
    "JSONExporter",
    "InMemoryExporter",
    "TracingContext",
    "get_tracer",
    "set_tracer",
]
