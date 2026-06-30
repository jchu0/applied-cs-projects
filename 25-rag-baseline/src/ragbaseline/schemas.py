"""Core data schemas for RAG system."""

from dataclasses import dataclass, field
from typing import Optional
import hashlib


@dataclass
class Document:
    """Ingested document."""

    id: str
    content: str
    metadata: dict = field(default_factory=dict)
    source: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            source=data.get("source", ""),
        )


@dataclass
class Chunk:
    """Document chunk for embedding."""

    id: str
    content: str
    document_id: str
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)
    start_char: int = 0
    end_char: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            document_id=data.get("document_id", ""),
            chunk_index=data.get("chunk_index", 0),
            metadata=data.get("metadata", {}),
            start_char=data.get("start_char", 0),
            end_char=data.get("end_char", 0),
        )


@dataclass
class SearchResult:
    """Result from vector search."""

    id: str
    score: float = 0.0
    content: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SearchResult":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            score=data.get("score", 0.0),
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
        )

    def __gt__(self, other: "SearchResult") -> bool:
        return self.score > other.score

    def __lt__(self, other: "SearchResult") -> bool:
        return self.score < other.score

    def __ge__(self, other: "SearchResult") -> bool:
        return self.score >= other.score

    def __le__(self, other: "SearchResult") -> bool:
        return self.score <= other.score


@dataclass
class RAGResponse:
    """Response from RAG pipeline."""

    query: str
    answer: str
    sources: list[SearchResult]
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        import json
        return json.dumps(self.to_dict())


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""

    # Retrieval
    top_k: int = 5
    rerank: bool = False
    rerank_top_k: int = 3

    # Generation
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1024
    stream: bool = False

    # Prompt
    system_prompt: str = """You are a helpful assistant that answers questions based on the provided context.
If the context doesn't contain enough information to answer, say so.
Always cite your sources by referencing the document titles or filenames."""


def generate_id(content: str) -> str:
    """Generate deterministic ID from content."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]
