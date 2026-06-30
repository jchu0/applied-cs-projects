"""Core data models for Micro-Model Orchestrated RAG."""

from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum
import uuid
import time


def generate_id() -> str:
    """Generate unique identifier."""
    return uuid.uuid4().hex[:12]


class NodeType(Enum):
    """Types of computation graph nodes."""
    SLM = "slm"
    RETRIEVAL = "retrieval"
    TRANSFORM = "transform"
    BRANCH = "branch"
    MERGE = "merge"


class QueryIntent(Enum):
    """Query intent types."""
    FACTUAL = "factual"
    COMPARISON = "comparison"
    PROCEDURAL = "procedural"
    EXPLORATORY = "exploratory"
    CLARIFICATION = "clarification"


@dataclass
class Chunk:
    """Document chunk with metadata."""
    content: str
    start_idx: int
    end_idx: int
    chunk_type: str  # paragraph, section, code_block, table
    semantic_score: float
    metadata: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"chunk_{self.start_idx}_{self.end_idx}"


@dataclass
class Document:
    """Document with content and metadata."""
    id: str
    content: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[Any] = None


@dataclass
class RetrievalResult:
    """Result from retrieval."""
    document: Document
    score: float
    retriever_type: str
    rank: int


@dataclass
class RerankResult:
    """Result from reranking."""
    document: Document
    original_rank: int
    new_rank: int
    relevance_score: float
    features: dict = field(default_factory=dict)


@dataclass
class RoutingDecision:
    """Query routing decision."""
    retrievers: list[str]
    transformed_query: str
    filters: dict = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class StabilizedAnswer:
    """Answer after stabilization."""
    answer: str
    confidence: float
    consistency: float
    num_samples: int
    reasoning: str = ""


@dataclass
class GraphNode:
    """Node in computation graph."""
    id: str
    node_type: NodeType
    component: str
    config: dict = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    fallback: Optional[str] = None


@dataclass
class ExecutionResult:
    """Result of node execution."""
    node_id: str
    output: Any
    latency_ms: float
    quality_score: float
    metadata: dict = field(default_factory=dict)


@dataclass
class Span:
    """Tracing span."""
    trace_id: str
    span_id: str
    parent_id: Optional[str]
    name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    attributes: dict = field(default_factory=dict)


@dataclass
class GuardrailResult:
    """Result of guardrail check."""
    passed: bool
    action: str  # pass, warn, block
    violations: list = field(default_factory=list)


@dataclass
class RuleResult:
    """Result of a single guardrail rule."""
    passed: bool
    severity: str = "info"
    message: str = ""


@dataclass
class ModelInfo:
    """Information about a model."""
    name: str
    task: str
    size_mb: float
    avg_latency_ms: float
    quality_score: float
    cost_per_1k: float = 0.0


@dataclass
class SelectionConstraints:
    """Constraints for model selection."""
    max_latency_ms: Optional[float] = None
    min_quality: Optional[float] = None
    max_cost: Optional[float] = None
    preferred_models: list[str] = field(default_factory=list)


@dataclass
class PerformanceStats:
    """Performance statistics for a model."""
    avg_latency: float = 0.0
    avg_quality: float = 0.0
    error_rate: float = 0.0
    call_count: int = 0


@dataclass
class CanaryEvaluation:
    """Canary evaluation result."""
    canary_id: str
    passed: bool
    checks: list
    recommendation: str
