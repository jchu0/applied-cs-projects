"""Plugin registry for pluggable RAG components."""

from typing import Type
from ..reranking.reranker import BaseReranker


class RerankerRegistry:
    """Registry for pluggable rerankers."""

    def __init__(self):
        self._rerankers: dict[str, Type[BaseReranker]] = {}
        self._instances: dict[str, BaseReranker] = {}

    def register(self, name: str, reranker_class: Type[BaseReranker]):
        """Register a reranker class.

        Args:
            name: Unique name for the reranker
            reranker_class: The reranker class to register
        """
        if not issubclass(reranker_class, BaseReranker):
            raise ValueError(f"{reranker_class} must be a subclass of BaseReranker")
        self._rerankers[name] = reranker_class

    def get(self, name: str, **kwargs) -> BaseReranker:
        """Get a reranker instance.

        Args:
            name: Registered reranker name
            **kwargs: Arguments to pass to the reranker constructor

        Returns:
            Reranker instance

        Raises:
            ValueError: If reranker not found
        """
        if name not in self._rerankers:
            available = ", ".join(self._rerankers.keys())
            raise ValueError(f"Unknown reranker: {name}. Available: {available}")

        return self._rerankers[name](**kwargs)

    def get_cached(self, name: str, **kwargs) -> BaseReranker:
        """Get a cached reranker instance.

        Args:
            name: Registered reranker name
            **kwargs: Arguments to pass to the reranker constructor

        Returns:
            Cached reranker instance
        """
        cache_key = f"{name}:{hash(frozenset(kwargs.items()))}"

        if cache_key not in self._instances:
            self._instances[cache_key] = self.get(name, **kwargs)

        return self._instances[cache_key]

    def list_available(self) -> list[str]:
        """List available reranker names.

        Returns:
            List of registered reranker names
        """
        return list(self._rerankers.keys())

    def unregister(self, name: str):
        """Unregister a reranker.

        Args:
            name: Reranker name to unregister
        """
        self._rerankers.pop(name, None)
        # Also clear cached instances
        to_remove = [k for k in self._instances if k.startswith(f"{name}:")]
        for key in to_remove:
            del self._instances[key]

    def clear_cache(self):
        """Clear all cached instances."""
        self._instances.clear()


class RetrieverRegistry:
    """Registry for pluggable retrievers."""

    def __init__(self):
        self._retrievers: dict[str, type] = {}

    def register(self, name: str, retriever_class: type):
        """Register a retriever class."""
        self._retrievers[name] = retriever_class

    def get(self, name: str, **kwargs):
        """Get a retriever instance."""
        if name not in self._retrievers:
            available = ", ".join(self._retrievers.keys())
            raise ValueError(f"Unknown retriever: {name}. Available: {available}")
        return self._retrievers[name](**kwargs)

    def list_available(self) -> list[str]:
        """List available retriever names."""
        return list(self._retrievers.keys())


class EmbeddingRegistry:
    """Registry for embedding models."""

    def __init__(self):
        self._embeddings: dict[str, type] = {}

    def register(self, name: str, embedding_class: type):
        """Register an embedding class."""
        self._embeddings[name] = embedding_class

    def get(self, name: str, **kwargs):
        """Get an embedding instance."""
        if name not in self._embeddings:
            available = ", ".join(self._embeddings.keys())
            raise ValueError(f"Unknown embedding: {name}. Available: {available}")
        return self._embeddings[name](**kwargs)

    def list_available(self) -> list[str]:
        """List available embedding names."""
        return list(self._embeddings.keys())


# Global registry instances
reranker_registry = RerankerRegistry()
retriever_registry = RetrieverRegistry()
embedding_registry = EmbeddingRegistry()


def setup_default_registries():
    """Register default components in registries."""
    from ..reranking.reranker import (
        CrossEncoderReranker,
        SLMReranker,
        MockReranker,
        MultiStageReranker,
    )
    from ..retrieval.bm25 import BM25Index
    from ..retrieval.vector import (
        VectorRetriever,
        MockEmbedding,
        SentenceTransformerEmbedding,
    )
    from ..retrieval.hybrid import HybridRetriever

    # Register rerankers
    reranker_registry.register("cross-encoder", CrossEncoderReranker)
    reranker_registry.register("slm", SLMReranker)
    reranker_registry.register("mock", MockReranker)
    reranker_registry.register("multi-stage", MultiStageReranker)

    # Register retrievers
    retriever_registry.register("bm25", BM25Index)
    retriever_registry.register("vector", VectorRetriever)
    retriever_registry.register("hybrid", HybridRetriever)

    # Register embeddings
    embedding_registry.register("mock", MockEmbedding)
    embedding_registry.register("sentence-transformer", SentenceTransformerEmbedding)
