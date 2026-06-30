"""Reranker SLM for cross-attention relevance scoring."""

from typing import List, Optional
import numpy as np
import torch
from sentence_transformers import CrossEncoder

from .base import BaseSLM, CrossEncoderMixin, GenerativeModelMixin, logger
from ..schemas import RetrievalResult, RerankResult


class RerankerSLM(BaseSLM, CrossEncoderMixin, GenerativeModelMixin):
    """SLM reranker with cross-attention scoring using actual models."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        llm_model: str = "Qwen/Qwen2-1.5B-Instruct",
        default_top_k: int = 10,
        batch_size: int = 32
    ):
        super().__init__(model_name)
        self.llm_model_name = llm_model
        self.default_top_k = default_top_k
        self.batch_size = batch_size
        self.cross_encoder = None
        self.llm = None

    def _load_model(self):
        """Load cross-encoder for reranking and optionally LLM for explanations."""
        # Load cross-encoder model for reranking
        try:
            self.cross_encoder = CrossEncoder(
                self.model_name,
                max_length=512,
                device=self.device
            )
            logger.info(f"Loaded cross-encoder: {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to load {self.model_name}: {str(e)}")
            # Fallback to a smaller cross-encoder
            fallback = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            self.cross_encoder = CrossEncoder(
                fallback,
                max_length=512,
                device=self.device
            )
            logger.info(f"Using fallback cross-encoder: {fallback}")

    async def process(
        self,
        query: str = None,
        retrieve: List[RetrievalResult] = None,
        top_k: int = None,
        **kwargs
    ) -> List[RerankResult]:
        """Rerank retrieval results using cross-encoder model.

        Args:
            query: Original query
            retrieve: Retrieval results to rerank
            top_k: Number of results to return

        Returns:
            Reranked results
        """
        if not query or not retrieve:
            return []

        if not self._loaded:
            self.load()

        top_k = top_k or self.default_top_k
        top_k = min(top_k, len(retrieve))

        # Prepare query-document pairs for the cross-encoder
        pairs = []
        for result in retrieve:
            pairs.append([query, result.document.content])

        # Score all pairs with the cross-encoder
        scores = self._score_pairs_batch(pairs)

        # Create (result, score) tuples and sort
        scored_results = list(zip(retrieve, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Create reranked results
        reranked = []
        for new_rank, (result, score) in enumerate(scored_results[:top_k]):
            reranked.append(RerankResult(
                document=result.document,
                original_rank=result.rank,
                new_rank=new_rank,
                relevance_score=float(score),
                features={
                    "cross_encoder_score": float(score),
                    "original_score": result.score,
                    "retriever_type": result.retriever_type
                }
            ))

        return reranked

    def _score_pairs_batch(self, pairs: List[List[str]]) -> np.ndarray:
        """Score query-document pairs in batches for efficiency."""
        if not pairs:
            return np.array([])

        all_scores = []

        # Process in batches to avoid memory issues
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]

            try:
                # Get scores from cross-encoder
                with torch.no_grad():
                    batch_scores = self.cross_encoder.predict(
                        batch,
                        show_progress_bar=False
                    )

                # Apply sigmoid to convert to probabilities
                batch_scores = 1 / (1 + np.exp(-batch_scores))
                all_scores.extend(batch_scores)

            except Exception as e:
                logger.warning(f"Error scoring batch: {str(e)}")
                # Fallback to simple similarity scoring
                for pair in batch:
                    score = self._fallback_score(pair[0], pair[1])
                    all_scores.append(score)

        return np.array(all_scores)

    def _fallback_score(self, query: str, document: str) -> float:
        """Simple fallback scoring when cross-encoder fails."""
        query_terms = set(query.lower().split())
        doc_terms = set(document.lower().split())

        if not query_terms:
            return 0.0

        # Compute term overlap
        overlap = len(query_terms & doc_terms)
        score = overlap / len(query_terms)

        # Boost for exact phrase match
        if query.lower() in document.lower():
            score = min(1.0, score + 0.3)

        return score


class MockRerankerSLM(BaseSLM):
    """Mock reranker for testing."""

    def __init__(self):
        super().__init__("mock")
        self._loaded = True

    async def process(
        self,
        query: str = None,
        retrieve: List[RetrievalResult] = None,
        top_k: int = 10,
        **kwargs
    ) -> List[RerankResult]:
        if not retrieve:
            return []

        # Simply reverse the ranking as mock behavior
        results = []
        for i, candidate in enumerate(retrieve[:top_k]):
            results.append(RerankResult(
                document=candidate.document,
                original_rank=candidate.rank,
                new_rank=i,
                relevance_score=1.0 - (i * 0.1),
                features={"mock": True}
            ))

        return results


class ExplainableReranker(RerankerSLM):
    """Reranker that generates relevance explanations using an LLM."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        llm_model: str = "Qwen/Qwen2-1.5B-Instruct",
        default_top_k: int = 10
    ):
        super().__init__(model_name, llm_model, default_top_k)
        self.llm_loaded = False

    def _load_llm(self):
        """Load the LLM for generating explanations."""
        if not self.llm_loaded:
            try:
                self._load_generative_model(self.llm_model_name, load_in_8bit=True)
                self.llm_loaded = True
                logger.info(f"Loaded LLM for explanations: {self.llm_model_name}")
            except Exception as e:
                logger.warning(f"Failed to load LLM for explanations: {str(e)}")

    async def process(
        self,
        query: str = None,
        retrieve: List[RetrievalResult] = None,
        top_k: int = None,
        explain_top: int = 5,
        **kwargs
    ) -> List[RerankResult]:
        """Rerank with explanations for top results.

        Args:
            query: Query string
            retrieve: Candidates to rerank
            top_k: Number to return
            explain_top: Number of results to explain

        Returns:
            Reranked results with explanations
        """
        # Get base reranking
        results = await super().process(
            query=query,
            retrieve=retrieve,
            top_k=top_k,
            **kwargs
        )

        # Generate explanations for top results if LLM is available
        if explain_top > 0:
            self._load_llm()

            for i, result in enumerate(results[:explain_top]):
                if self.llm_loaded:
                    explanation = await self._generate_llm_explanation(
                        query,
                        result.document.content,
                        result.relevance_score
                    )
                else:
                    explanation = self._generate_heuristic_explanation(
                        query,
                        result.document.content,
                        result.relevance_score
                    )

                result.features["explanation"] = explanation
                result.features["explanation_rank"] = i + 1

        return results

    async def _generate_llm_explanation(
        self,
        query: str,
        content: str,
        score: float
    ) -> str:
        """Generate relevance explanation using LLM."""
        # Truncate content if too long
        max_content_len = 300
        if len(content) > max_content_len:
            content = content[:max_content_len] + "..."

        prompt = f"""Explain why this document is relevant to the query (be concise):

Query: {query}

Document excerpt: {content}

Relevance score: {score:.3f}

Explanation (1-2 sentences):"""

        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=400
                ).to(self.device)

                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_new_tokens=50,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

                explanation = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()

            return explanation if explanation else "Highly relevant based on semantic similarity."

        except Exception as e:
            logger.warning(f"Failed to generate LLM explanation: {str(e)}")
            return self._generate_heuristic_explanation(query, content, score)

    def _generate_heuristic_explanation(
        self,
        query: str,
        content: str,
        score: float
    ) -> str:
        """Generate simple heuristic explanation."""
        query_terms = set(query.lower().split())
        content_terms = set(content.lower().split()[:100])  # Check first 100 words
        matched = query_terms & content_terms

        if matched:
            matched_str = ", ".join(list(matched)[:5])
            return f"Document matches query terms: {matched_str} (score: {score:.3f})"
        elif score > 0.8:
            return f"High semantic similarity to query topic (score: {score:.3f})"
        elif score > 0.5:
            return f"Moderate relevance to query context (score: {score:.3f})"
        else:
            return f"Potentially relevant based on cross-encoder scoring (score: {score:.3f})"


class DiversityReranker(RerankerSLM):
    """Reranker that promotes diversity using MMR (Maximal Marginal Relevance)."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        default_top_k: int = 10,
        lambda_param: float = 0.7
    ):
        super().__init__(model_name)
        self.embedding_model_name = embedding_model
        self.default_top_k = default_top_k
        self.lambda_param = lambda_param  # Balance between relevance and diversity
        self.embedder = None

    def _load_model(self):
        """Load both cross-encoder and embedding model."""
        super()._load_model()

        # Load embedding model for diversity computation
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer(self.embedding_model_name)
        self.embedder.to(self.device)

    async def process(
        self,
        query: str = None,
        retrieve: List[RetrievalResult] = None,
        top_k: int = None,
        lambda_param: float = None,
        **kwargs
    ) -> List[RerankResult]:
        """Rerank with diversity using MMR algorithm.

        Args:
            query: Query string
            retrieve: Candidates to rerank
            top_k: Number to return
            lambda_param: Balance parameter (higher = more relevance, lower = more diversity)

        Returns:
            Diverse reranked results
        """
        if not query or not retrieve:
            return []

        if not self._loaded:
            self.load()

        top_k = top_k or self.default_top_k
        lambda_param = lambda_param or self.lambda_param

        # First get relevance scores
        pairs = [[query, r.document.content] for r in retrieve]
        relevance_scores = self._score_pairs_batch(pairs)

        # Get document embeddings for diversity
        doc_texts = [r.document.content for r in retrieve]
        doc_embeddings = self.embedder.encode(
            doc_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        # Apply MMR algorithm
        selected_indices = []
        selected_embeddings = []

        for _ in range(min(top_k, len(retrieve))):
            mmr_scores = []

            for i, (rel_score, doc_emb) in enumerate(zip(relevance_scores, doc_embeddings)):
                if i in selected_indices:
                    mmr_scores.append(-float('inf'))
                    continue

                # Relevance part
                relevance = lambda_param * rel_score

                # Diversity part (negative of max similarity to selected docs)
                if selected_embeddings:
                    similarities = np.dot(
                        np.array(selected_embeddings),
                        doc_emb
                    )
                    max_sim = np.max(similarities)
                    diversity = (1 - lambda_param) * (1 - max_sim)
                else:
                    diversity = 0

                mmr_score = relevance + diversity
                mmr_scores.append(mmr_score)

            # Select document with highest MMR score
            best_idx = np.argmax(mmr_scores)
            selected_indices.append(best_idx)
            selected_embeddings.append(doc_embeddings[best_idx])

        # Create reranked results
        reranked = []
        for new_rank, idx in enumerate(selected_indices):
            result = retrieve[idx]
            reranked.append(RerankResult(
                document=result.document,
                original_rank=result.rank,
                new_rank=new_rank,
                relevance_score=float(relevance_scores[idx]),
                features={
                    "mmr_selected": True,
                    "original_score": result.score,
                    "diversity_considered": True
                }
            ))

        return reranked