"""Neural reranking implementations."""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

from ..schemas import RetrievalResult, RerankResult


class BaseReranker(ABC):
    """Base class for rerankers."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int = None,
    ) -> list[RerankResult]:
        """Rerank candidates."""
        pass


class CrossEncoderReranker(BaseReranker):
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

    async def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int = None,
    ) -> list[RerankResult]:
        """Rerank using cross-encoder scores."""
        if not candidates:
            return []

        # Prepare query-document pairs
        pairs = [(query, c.document.content) for c in candidates]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Sort by score
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        # Create results
        results = []
        for new_rank, (candidate, score) in enumerate(scored):
            if top_k and new_rank >= top_k:
                break

            results.append(RerankResult(
                document=candidate.document,
                original_rank=candidate.rank,
                new_rank=new_rank,
                relevance_score=float(score),
                features={"cross_encoder_score": float(score)},
            ))

        return results


class SLMReranker(BaseReranker):
    """Small Language Model reranker."""

    def __init__(self, llm_client, batch_size: int = 5):
        self.llm = llm_client
        self.batch_size = batch_size

    async def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int = None,
    ) -> list[RerankResult]:
        """Rerank using LLM scoring."""
        if not candidates:
            return []

        # Score in batches
        all_scores = []

        for i in range(0, len(candidates), self.batch_size):
            batch = candidates[i:i + self.batch_size]
            scores = await self._score_batch(query, batch)
            all_scores.extend(scores)

        # Sort by score
        scored = list(zip(candidates, all_scores))
        scored.sort(key=lambda x: x[1]["relevance"], reverse=True)

        # Create results
        results = []
        for new_rank, (candidate, scores) in enumerate(scored):
            if top_k and new_rank >= top_k:
                break

            results.append(RerankResult(
                document=candidate.document,
                original_rank=candidate.rank,
                new_rank=new_rank,
                relevance_score=scores["relevance"],
                features=scores,
            ))

        return results

    async def _score_batch(
        self,
        query: str,
        batch: list[RetrievalResult]
    ) -> list[dict]:
        """Score a batch of documents."""
        docs_text = "\n\n".join([
            f"Document {i+1}: {c.document.content[:500]}"
            for i, c in enumerate(batch)
        ])

        prompt = f"""Rate how well each document answers the query.
Score from 0-10 for relevance.

Query: {query}

{docs_text}

Return scores as JSON array: [{{"relevance": score}}, ...]"""

        response = await self.llm.generate(prompt)

        # Parse scores (simplified)
        try:
            import json
            scores = json.loads(response)
            return scores
        except:
            # Default scores
            return [{"relevance": 5.0} for _ in batch]


class MultiStageReranker(BaseReranker):
    """Multi-stage reranking pipeline."""

    def __init__(
        self,
        stage1_reranker: BaseReranker,
        stage2_reranker: Optional[BaseReranker] = None,
        stage1_top_k: int = 50,
        diversity_lambda: float = 0.7,
    ):
        self.stage1 = stage1_reranker
        self.stage2 = stage2_reranker
        self.stage1_top_k = stage1_top_k
        self.diversity_lambda = diversity_lambda

    async def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int = None,
    ) -> list[RerankResult]:
        """Multi-stage reranking with optional diversity."""
        # Stage 1: Fast coarse reranking
        stage1_results = await self.stage1.rerank(
            query, candidates, self.stage1_top_k
        )

        # Stage 2: Fine-grained reranking (if available)
        if self.stage2:
            # Convert back to retrieval results
            stage1_candidates = [
                RetrievalResult(
                    document=r.document,
                    score=r.relevance_score,
                    retriever_type="stage1",
                    rank=r.new_rank,
                )
                for r in stage1_results
            ]

            stage2_results = await self.stage2.rerank(
                query, stage1_candidates, top_k * 2 if top_k else None
            )
        else:
            stage2_results = stage1_results

        # Apply diversity filtering
        diverse_results = self._mmr_diversity(stage2_results, top_k or len(stage2_results))

        return diverse_results

    def _mmr_diversity(
        self,
        results: list[RerankResult],
        top_k: int
    ) -> list[RerankResult]:
        """Maximal Marginal Relevance for diversity."""
        if not results:
            return []

        selected = []
        remaining = list(results)

        while len(selected) < top_k and remaining:
            if not selected:
                # First: highest relevance
                best = max(remaining, key=lambda x: x.relevance_score)
            else:
                # MMR: balance relevance and diversity
                best = max(
                    remaining,
                    key=lambda x: (
                        self.diversity_lambda * x.relevance_score -
                        (1 - self.diversity_lambda) * self._max_similarity(x, selected)
                    )
                )

            selected.append(best)
            remaining.remove(best)

        # Update ranks
        for i, result in enumerate(selected):
            result.new_rank = i

        return selected

    def _max_similarity(
        self,
        candidate: RerankResult,
        selected: list[RerankResult]
    ) -> float:
        """Compute max similarity to selected documents."""
        if not selected:
            return 0.0

        # Simple text overlap similarity
        candidate_words = set(candidate.document.content.lower().split())

        max_sim = 0.0
        for sel in selected:
            sel_words = set(sel.document.content.lower().split())
            if candidate_words and sel_words:
                overlap = len(candidate_words & sel_words)
                sim = overlap / max(len(candidate_words), len(sel_words))
                max_sim = max(max_sim, sim)

        return max_sim


class MockReranker(BaseReranker):
    """Mock reranker for testing."""

    async def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int = None,
    ) -> list[RerankResult]:
        """Mock reranking - just converts to RerankResults."""
        results = []
        for i, candidate in enumerate(candidates):
            if top_k and i >= top_k:
                break

            results.append(RerankResult(
                document=candidate.document,
                original_rank=candidate.rank,
                new_rank=i,
                relevance_score=candidate.score,
                features={"mock": True},
            ))

        return results
