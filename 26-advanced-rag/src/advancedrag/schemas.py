"""Core data schemas for Advanced RAG system."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import hashlib


class QueryIntent(Enum):
    """Classification of query intent."""
    FACTUAL = "factual"
    COMPARISON = "comparison"
    PROCEDURAL = "procedural"
    EXPLORATORY = "exploratory"
    CLARIFICATION = "clarification"


@dataclass
class Document:
    """Document in the RAG system."""
    id: str
    content: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[list] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
        }


@dataclass
class RetrievalResult:
    """Result from retrieval."""
    document: Document
    score: float
    retriever_type: str
    rank: int

    def to_dict(self) -> dict:
        return {
            "document": self.document.to_dict(),
            "score": self.score,
            "retriever_type": self.retriever_type,
            "rank": self.rank,
        }


@dataclass
class RerankResult:
    """Result from reranking."""
    document: Document
    original_rank: int
    new_rank: int
    relevance_score: float
    features: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "document": self.document.to_dict(),
            "original_rank": self.original_rank,
            "new_rank": self.new_rank,
            "relevance_score": self.relevance_score,
            "features": self.features,
        }


@dataclass
class RewrittenQuery:
    """Rewritten query with expansions."""
    original: str
    rewritten: str
    intent: QueryIntent
    expansions: list[str]
    sub_queries: list[str]
    confidence: float
    metadata: dict = field(default_factory=dict)


@dataclass
class Citation:
    """Citation from answer."""
    source_id: str
    source_title: str
    quoted_text: str
    relevance_score: float


@dataclass
class ConstructedContext:
    """Constructed context for generation."""
    content: str
    source_documents: list[Document]
    compression_ratio: float
    token_count: int
    metadata: dict = field(default_factory=dict)


@dataclass
class GeneratedAnswer:
    """Generated answer with citations."""
    answer: str
    citations: list[Citation]
    confidence: float
    hallucination_flags: list[str]
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "citations": [
                {
                    "source_id": c.source_id,
                    "source_title": c.source_title,
                    "quoted_text": c.quoted_text,
                    "relevance_score": c.relevance_score,
                }
                for c in self.citations
            ],
            "confidence": self.confidence,
            "hallucination_flags": self.hallucination_flags,
            "metadata": self.metadata,
        }


@dataclass
class RAGResult:
    """Complete RAG pipeline result."""
    query: str
    rewritten_query: Optional[RewrittenQuery]
    retrieval_results: list[RetrievalResult]
    reranked_results: list[RerankResult]
    context: ConstructedContext
    answer: GeneratedAnswer
    latency_ms: float
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer.to_dict(),
            "sources": [r.to_dict() for r in self.reranked_results],
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }


def generate_id(content: str) -> str:
    """Generate deterministic ID from content."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]
