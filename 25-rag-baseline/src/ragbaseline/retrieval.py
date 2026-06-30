"""Advanced retrieval features: hybrid search, reranking, and filtering."""

import math
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
import numpy as np

from .schemas import SearchResult
from .embeddings import EmbeddingModel
from .vectorstore import VectorStore


@dataclass
class OldRetrievalConfig:
    """Configuration for retrieval (legacy)."""

    # Vector search
    vector_weight: float = 0.5

    # BM25 parameters
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # Reranking
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Filtering
    enable_filters: bool = True


class BM25:
    """BM25 implementation for keyword-based retrieval."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.k1 = k1
        self.b = b

        # Index data
        self.doc_ids: list[str] = []
        self.documents: list[str] = []
        self.metadatas: list[dict] = []

        # BM25 statistics
        self.doc_freqs: dict[str, int] = defaultdict(int)
        self.doc_lens: list[int] = []
        self.avg_doc_len: float = 0
        self.term_freqs: list[dict[str, int]] = []
        self.n_docs: int = 0

    def add_documents(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict] = None,
    ):
        """Add documents to BM25 index."""
        if metadatas is None:
            metadatas = [{}] * len(documents)

        for doc_id, doc, metadata in zip(ids, documents, metadatas):
            self.doc_ids.append(doc_id)
            self.documents.append(doc)
            self.metadatas.append(metadata)

            # Tokenize
            terms = self._tokenize(doc)
            self.doc_lens.append(len(terms))

            # Count term frequencies
            term_freq = defaultdict(int)
            for term in terms:
                term_freq[term] += 1
            self.term_freqs.append(dict(term_freq))

            # Update document frequencies
            for term in set(terms):
                self.doc_freqs[term] += 1

        self.n_docs = len(self.documents)
        self.avg_doc_len = sum(self.doc_lens) / self.n_docs if self.n_docs > 0 else 0

    def search(
        self,
        query: str,
        k: int = 5,
        filter: dict = None,
    ) -> list[SearchResult]:
        """Search using BM25 scoring."""
        if self.n_docs == 0:
            return []

        query_terms = self._tokenize(query)
        scores = []

        for idx in range(self.n_docs):
            # Apply filter
            if filter and not self._match_filter(self.metadatas[idx], filter):
                scores.append(-1)
                continue

            score = self._score_document(idx, query_terms)
            scores.append(score)

        # Get top k
        scores_array = np.array(scores)
        top_indices = np.argsort(scores_array)[-k:][::-1]

        results = []
        for idx in top_indices:
            if scores[idx] >= 0:
                results.append(SearchResult(
                    id=self.doc_ids[idx],
                    content=self.documents[idx],
                    score=float(scores[idx]),
                    metadata=self.metadatas[idx],
                ))

        return results

    def _score_document(self, doc_idx: int, query_terms: list[str]) -> float:
        """Calculate BM25 score for a document."""
        score = 0.0
        doc_len = self.doc_lens[doc_idx]
        term_freqs = self.term_freqs[doc_idx]

        for term in query_terms:
            if term not in self.doc_freqs:
                continue

            # IDF
            df = self.doc_freqs[term]
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)

            # TF with length normalization
            tf = term_freqs.get(term, 0)
            tf_norm = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            )

            score += idf * tf_norm

        return score

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization."""
        # Lowercase and split on non-alphanumeric
        import re
        return re.findall(r'\w+', text.lower())

    def _match_filter(self, metadata: dict, filter: dict) -> bool:
        """Check if metadata matches filter."""
        for key, value in filter.items():
            if key not in metadata:
                return False

            if isinstance(value, dict):
                for op, op_value in value.items():
                    if op == "$in":
                        if metadata[key] not in op_value:
                            return False
                    elif op == "$eq":
                        if metadata[key] != op_value:
                            return False
                    elif op == "$ne":
                        if metadata[key] == op_value:
                            return False
                    elif op == "$gte":
                        if metadata[key] < op_value:
                            return False
                    elif op == "$gt":
                        if metadata[key] <= op_value:
                            return False
                    elif op == "$lte":
                        if metadata[key] > op_value:
                            return False
                    elif op == "$lt":
                        if metadata[key] >= op_value:
                            return False
                    elif op == "$contains":
                        if op_value not in metadata[key]:
                            return False
            else:
                if metadata[key] != value:
                    return False

        return True

    def delete(self, ids: list[str]):
        """Delete documents by ID."""
        ids_set = set(ids)
        indices_to_keep = [
            i for i, doc_id in enumerate(self.doc_ids)
            if doc_id not in ids_set
        ]

        self.doc_ids = [self.doc_ids[i] for i in indices_to_keep]
        self.documents = [self.documents[i] for i in indices_to_keep]
        self.metadatas = [self.metadatas[i] for i in indices_to_keep]
        self.doc_lens = [self.doc_lens[i] for i in indices_to_keep]
        self.term_freqs = [self.term_freqs[i] for i in indices_to_keep]

        # Rebuild document frequencies
        self.doc_freqs = defaultdict(int)
        for term_freq in self.term_freqs:
            for term in term_freq:
                self.doc_freqs[term] += 1

        self.n_docs = len(self.documents)
        self.avg_doc_len = sum(self.doc_lens) / self.n_docs if self.n_docs > 0 else 0


class LegacyHybridRetriever:
    """Combine vector search with BM25 for hybrid retrieval (legacy)."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        config: OldRetrievalConfig = None,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.config = config or OldRetrievalConfig()

        # BM25 index
        self.bm25 = BM25(
            k1=self.config.bm25_k1,
            b=self.config.bm25_b,
        )

    def add(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        documents: list[str],
        metadatas: list[dict],
    ):
        """Add documents to both indices."""
        # Add to vector store
        self.vector_store.add(ids, embeddings, documents, metadatas)

        # Add to BM25
        self.bm25.add_documents(ids, documents, metadatas)

    def search(
        self,
        query: str,
        k: int = 5,
        filter: dict = None,
        alpha: float = None,
    ) -> list[SearchResult]:
        """Hybrid search combining vector and BM25.

        Args:
            query: Search query
            k: Number of results to return
            filter: Metadata filter
            alpha: Weight for vector search (0-1). If None, uses config.

        Returns:
            Combined search results
        """
        if alpha is None:
            alpha = self.config.vector_weight

        # Vector search
        query_embedding = self.embedding_model.encode([query])
        vector_results = self.vector_store.search(
            query_embedding,
            k=k * 2,  # Get more for fusion
            filter=filter,
        )

        # BM25 search
        bm25_results = self.bm25.search(query, k=k * 2, filter=filter)

        # Reciprocal Rank Fusion (RRF)
        combined_scores = self._rrf_fusion(
            vector_results,
            bm25_results,
            alpha=alpha,
        )

        # Sort by combined score
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x.score,
            reverse=True,
        )

        return sorted_results[:k]

    def _rrf_fusion(
        self,
        vector_results: list[SearchResult],
        bm25_results: list[SearchResult],
        alpha: float = 0.5,
        k: int = 60,  # RRF constant
    ) -> dict[str, SearchResult]:
        """Reciprocal Rank Fusion for combining results."""
        combined = {}

        # Process vector results
        for rank, result in enumerate(vector_results, 1):
            rrf_score = alpha * (1 / (k + rank))

            if result.id in combined:
                combined[result.id].score += rrf_score
            else:
                combined[result.id] = SearchResult(
                    id=result.id,
                    content=result.content,
                    score=rrf_score,
                    metadata=result.metadata,
                )

        # Process BM25 results
        for rank, result in enumerate(bm25_results, 1):
            rrf_score = (1 - alpha) * (1 / (k + rank))

            if result.id in combined:
                combined[result.id].score += rrf_score
            else:
                combined[result.id] = SearchResult(
                    id=result.id,
                    content=result.content,
                    score=rrf_score,
                    metadata=result.metadata,
                )

        return combined

    def delete(self, ids: list[str]):
        """Delete documents from both indices."""
        self.vector_store.delete(ids)
        self.bm25.delete(ids)


class Reranker(ABC):
    """Base class for rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = None,
    ) -> list[SearchResult]:
        """Rerank search results."""
        pass


class CrossEncoderReranker(Reranker):
    """Reranker using cross-encoder models."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers required. "
                "Install with: pip install sentence-transformers"
            )

        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = None,
    ) -> list[SearchResult]:
        """Rerank results using cross-encoder."""
        if not results:
            return []

        # Score query-document pairs
        pairs = [[query, r.content] for r in results]
        scores = self.model.predict(pairs)

        # Create new results with reranker scores
        reranked = []
        for result, score in zip(results, scores):
            reranked.append(SearchResult(
                id=result.id,
                content=result.content,
                score=float(score),
                metadata=result.metadata,
            ))

        # Sort by score
        reranked.sort(key=lambda x: x.score, reverse=True)

        if top_k:
            return reranked[:top_k]

        return reranked


class CohereReranker(Reranker):
    """Reranker using Cohere API."""

    def __init__(
        self,
        model: str = "rerank-english-v2.0",
        api_key: str = None,
    ):
        try:
            import cohere
        except ImportError:
            raise ImportError("cohere required. Install with: pip install cohere")

        self.client = cohere.Client(api_key)
        self.model = model

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = None,
    ) -> list[SearchResult]:
        """Rerank results using Cohere."""
        if not results:
            return []

        documents = [r.content for r in results]

        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=documents,
            top_n=top_k or len(documents),
        )

        # Create reranked results
        reranked = []
        for item in response.results:
            original = results[item.index]
            reranked.append(SearchResult(
                id=original.id,
                content=original.content,
                score=item.relevance_score,
                metadata=original.metadata,
            ))

        return reranked


class MetadataFilter:
    """Advanced metadata filtering with query syntax."""

    @staticmethod
    def parse(filter_string: str) -> dict:
        """Parse filter string into filter dict.

        Examples:
            "file_type:pdf" -> {"file_type": "pdf"}
            "date:>=2024-01-01" -> {"date": {"$gte": "2024-01-01"}}
            "category:in:pricing,sales" -> {"category": {"$in": ["pricing", "sales"]}}
        """
        if not filter_string:
            return {}

        filters = {}
        parts = filter_string.split(" AND ")

        for part in parts:
            part = part.strip()
            if ":" not in part:
                continue

            # Parse key:value or key:op:value
            segments = part.split(":", 2)
            key = segments[0].strip()

            if len(segments) == 2:
                # Simple equality
                filters[key] = segments[1].strip()
            elif len(segments) == 3:
                # Operator syntax
                op = segments[1].strip()
                value = segments[2].strip()

                if op == "in":
                    filters[key] = {"$in": [v.strip() for v in value.split(",")]}
                elif op == ">=":
                    filters[key] = {"$gte": value}
                elif op == ">":
                    filters[key] = {"$gt": value}
                elif op == "<=":
                    filters[key] = {"$lte": value}
                elif op == "<":
                    filters[key] = {"$lt": value}
                elif op == "!=":
                    filters[key] = {"$ne": value}
                elif op == "contains":
                    filters[key] = {"$contains": value}

        return filters

    @staticmethod
    def validate(filter_dict: dict, metadata: dict) -> bool:
        """Check if metadata matches filter."""
        for key, value in filter_dict.items():
            if key not in metadata:
                return False

            meta_value = metadata[key]

            if isinstance(value, dict):
                for op, op_value in value.items():
                    if op == "$in":
                        if meta_value not in op_value:
                            return False
                    elif op == "$eq":
                        if meta_value != op_value:
                            return False
                    elif op == "$ne":
                        if meta_value == op_value:
                            return False
                    elif op == "$gte":
                        if meta_value < op_value:
                            return False
                    elif op == "$gt":
                        if meta_value <= op_value:
                            return False
                    elif op == "$lte":
                        if meta_value > op_value:
                            return False
                    elif op == "$lt":
                        if meta_value >= op_value:
                            return False
                    elif op == "$contains":
                        if op_value not in str(meta_value):
                            return False
            else:
                if meta_value != value:
                    return False

        return True


def get_reranker(
    reranker_type: str = "cross-encoder",
    **kwargs,
) -> Reranker:
    """Factory function to get reranker.

    Args:
        reranker_type: Type of reranker
            - "cross-encoder": Cross-encoder model
            - "cohere": Cohere API
        **kwargs: Additional arguments

    Returns:
        Reranker instance
    """
    if reranker_type == "cross-encoder":
        return CrossEncoderReranker(**kwargs)
    elif reranker_type == "cohere":
        return CohereReranker(**kwargs)
    else:
        raise ValueError(f"Unknown reranker type: {reranker_type}")


# Async retrieval classes for test compatibility

class Retriever(ABC):
    """Abstract base class for async retrievers."""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        filter: dict = None,
    ) -> list[SearchResult]:
        """Retrieve relevant documents."""
        pass

    async def retrieve_batch(
        self,
        queries: list[str],
    ) -> list[list[SearchResult]]:
        """Retrieve for multiple queries."""
        results = []
        for query in queries:
            results.append(await self.retrieve(query))
        return results


class VectorRetriever(Retriever):
    """Async vector-based retriever."""

    def __init__(
        self,
        vectorstore,
        embedder,
        k: int = 5,
        score_threshold: float = 0.0,
    ):
        self.vectorstore = vectorstore
        self.embedder = embedder
        self.k = k
        self.score_threshold = score_threshold

    async def retrieve(
        self,
        query: str,
        filter: dict = None,
    ) -> list[SearchResult]:
        """Retrieve using vector similarity."""
        if not query.strip():
            return []

        embedding = await self.embedder.embed(query)
        results = await self.vectorstore.search(
            embedding,
            k=self.k,
            filter=filter,
        )

        # Apply score threshold
        if self.score_threshold > 0:
            results = [r for r in results if r.score >= self.score_threshold]

        return results

    async def retrieve_batch(
        self,
        queries: list[str],
    ) -> list[list[SearchResult]]:
        """Batch retrieval."""
        embeddings = await self.embedder.embed_batch(queries)
        results = await self.vectorstore.search_batch(embeddings, k=self.k)
        return results


class AsyncHybridRetriever(Retriever):
    """Hybrid retriever combining multiple strategies."""

    def __init__(
        self,
        retrievers: list,
        weights: list[float] = None,
        fusion_method: str = "weighted",
    ):
        self.retrievers = retrievers
        self.weights = weights or [1.0 / len(retrievers)] * len(retrievers)
        self.fusion_method = fusion_method

    async def retrieve(
        self,
        query: str,
        filter: dict = None,
    ) -> list[SearchResult]:
        """Retrieve and fuse results."""
        all_results = []
        for retriever in self.retrievers:
            results = await retriever.retrieve(query, filter=filter)
            all_results.append(results)

        if self.fusion_method == "reciprocal_rank":
            return self._reciprocal_rank_fusion(all_results)
        elif self.fusion_method == "max":
            return self._max_fusion(all_results)
        else:
            return self._weighted_fusion(all_results, self.weights)

    def _weighted_fusion(
        self,
        results_lists: list[list[SearchResult]],
        weights: list[float],
    ) -> list[SearchResult]:
        """Weighted score fusion."""
        combined = {}

        for results, weight in zip(results_lists, weights):
            for result in results:
                if result.id in combined:
                    combined[result.id].score += result.score * weight
                else:
                    combined[result.id] = SearchResult(
                        id=result.id,
                        score=result.score * weight,
                        content=result.content,
                        metadata=result.metadata,
                    )

        return sorted(combined.values(), key=lambda x: x.score, reverse=True)

    def _reciprocal_rank_fusion(
        self,
        results_lists: list[list[SearchResult]],
        k: int = 60,
    ) -> list[SearchResult]:
        """RRF fusion with tie-breaking by last rank position."""
        combined = {}
        last_rank = {}  # Track last rank for tie-breaking

        for results in results_lists:
            for rank, result in enumerate(results, 1):
                rrf_score = 1 / (k + rank)
                if result.id in combined:
                    combined[result.id].score += rrf_score
                else:
                    combined[result.id] = SearchResult(
                        id=result.id,
                        score=rrf_score,
                        content=result.content,
                        metadata=result.metadata,
                    )
                # Always update last_rank to track the most recent rank
                last_rank[result.id] = rank

        # Sort by RRF score (descending), then by last rank (ascending) for ties
        # Lower last_rank means appeared higher in later lists
        return sorted(
            combined.values(),
            key=lambda x: (-x.score, last_rank[x.id])
        )

    def _max_fusion(
        self,
        results_lists: list[list[SearchResult]],
    ) -> list[SearchResult]:
        """Max score fusion."""
        combined = {}

        for results in results_lists:
            for result in results:
                if result.id not in combined or result.score > combined[result.id].score:
                    combined[result.id] = result

        return sorted(combined.values(), key=lambda x: x.score, reverse=True)


# Alias for test compatibility - LegacyHybridRetriever has sync add()/search() methods
# AsyncHybridRetriever is for async usage with the Retriever protocol
HybridRetriever = LegacyHybridRetriever


class RerankedRetriever(Retriever):
    """Retriever with reranking."""

    def __init__(
        self,
        base_retriever,
        reranker,
        top_k: int = None,
        fallback_on_error: bool = False,
    ):
        self.base_retriever = base_retriever
        self.reranker = reranker
        self.top_k = top_k
        self.fallback_on_error = fallback_on_error

    async def retrieve(
        self,
        query: str,
        filter: dict = None,
    ) -> list[SearchResult]:
        """Retrieve and rerank."""
        results = await self.base_retriever.retrieve(query, filter=filter)

        try:
            reranked = await self.reranker.rerank(query, results)

            # Map reranked scores back to results
            reranked_results = []
            for idx, score in reranked:
                result = results[idx]
                reranked_results.append(SearchResult(
                    id=result.id,
                    score=score,
                    content=result.content,
                    metadata=result.metadata,
                ))

            if self.top_k:
                return reranked_results[:self.top_k]
            return reranked_results
        except Exception as e:
            if self.fallback_on_error:
                return results
            raise


class FusionRetriever(Retriever):
    """Fusion retriever with configurable strategies."""

    def __init__(
        self,
        retrievers: list,
        weights: list[float] = None,
        k: int = 60,
    ):
        self.retrievers = retrievers
        self.weights = weights or ([1.0 / len(retrievers)] if retrievers else [])
        self.k = k

    async def retrieve(
        self,
        query: str,
        filter: dict = None,
    ) -> list[SearchResult]:
        """Retrieve with RRF fusion."""
        all_results = []
        for retriever in self.retrievers:
            results = await retriever.retrieve(query, filter=filter)
            all_results.append(results)

        return self._reciprocal_rank_fusion(all_results, self.k)

    def _reciprocal_rank_fusion(
        self,
        results_lists: list[list[SearchResult]],
        k: int = 60,
    ) -> list[SearchResult]:
        """RRF fusion with tie-breaking by last rank position."""
        combined = {}
        last_rank = {}  # Track last rank for tie-breaking

        for results in results_lists:
            for rank, result in enumerate(results, 1):
                rrf_score = 1 / (k + rank)
                if result.id in combined:
                    combined[result.id].score += rrf_score
                else:
                    combined[result.id] = SearchResult(
                        id=result.id,
                        score=rrf_score,
                        content=result.content,
                        metadata=result.metadata,
                    )
                # Always update last_rank to track the most recent rank
                last_rank[result.id] = rank

        # Sort by RRF score (descending), then by last rank (ascending) for ties
        # Lower last_rank means appeared higher in later lists
        return sorted(
            combined.values(),
            key=lambda x: (-x.score, last_rank[x.id])
        )

    def _weighted_fusion(
        self,
        results_lists: list[list[SearchResult]],
        weights: list[float],
    ) -> list[SearchResult]:
        """Weighted fusion."""
        combined = {}

        for results, weight in zip(results_lists, weights):
            for result in results:
                if result.id in combined:
                    combined[result.id].score += result.score * weight
                else:
                    combined[result.id] = SearchResult(
                        id=result.id,
                        score=result.score * weight,
                        content=result.content,
                        metadata=result.metadata,
                    )

        return sorted(combined.values(), key=lambda x: x.score, reverse=True)


class MMRRetriever(Retriever):
    """Maximal Marginal Relevance retriever for diversity."""

    def __init__(
        self,
        base_retriever,
        embedder,
        lambda_mult: float = 0.5,
        k: int = 5,
    ):
        self.base_retriever = base_retriever
        self.embedder = embedder
        self.lambda_mult = lambda_mult
        self.k = k

    async def retrieve(
        self,
        query: str,
        filter: dict = None,
    ) -> list[SearchResult]:
        """Retrieve with MMR diversity."""
        # Get initial results
        results = await self.base_retriever.retrieve(query, filter=filter)

        if not results or self.lambda_mult == 1.0:
            return results[:self.k]

        # Get embeddings for results
        embeddings = {}
        for result in results:
            emb = await self.embedder.embed(result.id)
            embeddings[result.id] = emb

        query_emb = await self.embedder.embed(query)

        # MMR selection
        selected = []
        remaining = list(results)

        while len(selected) < self.k and remaining:
            best_score = float('-inf')
            best_idx = 0

            for idx, result in enumerate(remaining):
                emb = embeddings[result.id]

                # Relevance score
                relevance = result.score

                # Diversity penalty
                max_sim = 0.0
                for sel in selected:
                    sel_emb = embeddings[sel.id]
                    sim = self._cosine_similarity(emb, sel_emb)
                    max_sim = max(max_sim, sim)

                # MMR score
                mmr_score = self.lambda_mult * relevance - (1 - self.lambda_mult) * max_sim

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            selected.append(remaining.pop(best_idx))

        return selected

    def _cosine_similarity(self, a, b) -> float:
        """Calculate cosine similarity."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


@dataclass
class RetrievalConfig:
    """Configuration for retrieval."""

    retriever_type: str = "vector"
    k: int = 5
    score_threshold: float = 0.0
    reranking_enabled: bool = False
    mmr_lambda: float = 0.5
    vector_weight: float = 0.5  # For hybrid retrieval
    bm25_k1: float = 1.5  # BM25 term saturation parameter
    bm25_b: float = 0.75  # BM25 document length normalization

    def __post_init__(self):
        """Validate config."""
        if self.k <= 0:
            raise ValueError("k must be positive")
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("score_threshold must be between 0 and 1")
        if not 0.0 <= self.mmr_lambda <= 1.0:
            raise ValueError("mmr_lambda must be between 0 and 1")
        if not 0.0 <= self.vector_weight <= 1.0:
            raise ValueError("vector_weight must be between 0 and 1")

    @classmethod
    def from_dict(cls, data: dict) -> "RetrievalConfig":
        """Create config from dictionary."""
        return cls(**data)
