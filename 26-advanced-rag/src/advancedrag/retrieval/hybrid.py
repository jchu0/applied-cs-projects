"""Hybrid retrieval combining BM25 and vector search."""

from collections import defaultdict
from typing import Optional
import numpy as np

from ..schemas import Document, RetrievalResult
from .bm25 import BM25Index
from .vector import VectorRetriever, EmbeddingModel, VectorStore


class HybridRetriever:
    """Combines BM25 and vector retrieval with fusion strategies."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
        fusion_strategy: str = "rrf",  # rrf, linear
    ):
        self.bm25_index = BM25Index()
        self.vector_retriever = VectorRetriever(embedding_model, vector_store)

        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.fusion_strategy = fusion_strategy

    def add_documents(self, documents: list[Document]):
        """Add documents to both indices."""
        self.bm25_index.add_documents(documents)
        self.vector_retriever.add_documents(documents)

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: dict = None,
        alpha: float = None,
    ) -> list[RetrievalResult]:
        """Hybrid search combining BM25 and vector results.

        Args:
            query: Search query
            top_k: Number of results
            filter_dict: Metadata filter
            alpha: Override vector weight (0-1)

        Returns:
            Fused retrieval results
        """
        if alpha is not None:
            vector_weight = alpha
            bm25_weight = 1 - alpha
        else:
            vector_weight = self.vector_weight
            bm25_weight = self.bm25_weight

        # Retrieve from both sources
        fetch_k = top_k * 3

        bm25_results = self.bm25_index.search(query, fetch_k, filter_dict)
        vector_results = self.vector_retriever.search(query, fetch_k, filter_dict)

        # Fuse results
        if self.fusion_strategy == "rrf":
            fused = self._reciprocal_rank_fusion(
                bm25_results, vector_results,
                bm25_weight, vector_weight
            )
        else:
            fused = self._linear_combination(
                bm25_results, vector_results,
                bm25_weight, vector_weight
            )

        return fused[:top_k]

    def _reciprocal_rank_fusion(
        self,
        bm25_results: list[RetrievalResult],
        vector_results: list[RetrievalResult],
        bm25_weight: float,
        vector_weight: float,
        k: int = 60,
    ) -> list[RetrievalResult]:
        """RRF fusion: score = sum(weight / (k + rank))."""
        doc_scores: dict[str, float] = defaultdict(float)
        doc_results: dict[str, RetrievalResult] = {}

        # Process BM25 results
        for result in bm25_results:
            doc_id = result.document.id
            rrf_score = bm25_weight / (k + result.rank + 1)
            doc_scores[doc_id] += rrf_score
            doc_results[doc_id] = result

        # Process vector results
        for result in vector_results:
            doc_id = result.document.id
            rrf_score = vector_weight / (k + result.rank + 1)
            doc_scores[doc_id] += rrf_score
            if doc_id not in doc_results:
                doc_results[doc_id] = result

        # Sort by fused score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        return [
            RetrievalResult(
                document=doc_results[doc_id].document,
                score=score,
                retriever_type="hybrid_rrf",
                rank=i,
            )
            for i, (doc_id, score) in enumerate(sorted_docs)
        ]

    def _linear_combination(
        self,
        bm25_results: list[RetrievalResult],
        vector_results: list[RetrievalResult],
        bm25_weight: float,
        vector_weight: float,
    ) -> list[RetrievalResult]:
        """Linear combination with normalized scores."""
        # Normalize scores
        bm25_normalized = self._normalize_scores(bm25_results)
        vector_normalized = self._normalize_scores(vector_results)

        doc_scores: dict[str, float] = defaultdict(float)
        doc_results: dict[str, RetrievalResult] = {}

        for result, norm_score in zip(bm25_results, bm25_normalized):
            doc_id = result.document.id
            doc_scores[doc_id] += norm_score * bm25_weight
            doc_results[doc_id] = result

        for result, norm_score in zip(vector_results, vector_normalized):
            doc_id = result.document.id
            doc_scores[doc_id] += norm_score * vector_weight
            if doc_id not in doc_results:
                doc_results[doc_id] = result

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        return [
            RetrievalResult(
                document=doc_results[doc_id].document,
                score=score,
                retriever_type="hybrid_linear",
                rank=i,
            )
            for i, (doc_id, score) in enumerate(sorted_docs)
        ]

    def _normalize_scores(self, results: list[RetrievalResult]) -> list[float]:
        """Normalize scores to [0, 1]."""
        if not results:
            return []

        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [1.0] * len(scores)

        return [(s - min_score) / (max_score - min_score) for s in scores]

    def delete(self, doc_ids: list[str]):
        """Delete documents from both indices."""
        self.bm25_index.delete(doc_ids)
        self.vector_retriever.delete(doc_ids)

    @property
    def count(self) -> int:
        """Get number of indexed documents."""
        return self.bm25_index.count
