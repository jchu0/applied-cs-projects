# Project 26: Advanced RAG System

## Executive Summary

A production-grade Retrieval-Augmented Generation system implementing multi-stage retrieval pipelines, intelligent query rewriting, LLM-based reranking, and comprehensive hallucination detection. This system provides enterprise-ready evaluation frameworks with pluggable components and multi-tenant support.

> **Concepts covered:** [§04 RAG systems](../../04-ai-engineering/02-llm-applications/rag/rag-systems.md) · [§04 Vector stores](../../04-ai-engineering/03-vector-databases/vector-stores/vector-stores.md). For the simpler baseline, see [Project 25](../25-rag-baseline/); for SLM-orchestrated variation, [Project 27](../27-micro-model-orchestrated-rag/). Map: [`CONCEPT_TO_PROJECT_MAP.md`](../CONCEPT_TO_PROJECT_MAP.md).

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Core Components](#core-components)
3. [Technical Specifications](#technical-specifications)
4. [Data Flow Architecture](#data-flow-architecture)
5. [API Design](#api-design)
6. [Implementation Phases](#implementation-phases)
7. [Enterprise Features](#enterprise-features)
8. [Evaluation Framework](#evaluation-framework)
9. [Deployment Architecture](#deployment-architecture)
10. [Stretch Goals](#stretch-goals)

---

## System Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Advanced RAG System                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │    Query     │───▶│   Hybrid     │───▶│   Neural     │───▶│  Context   │ │
│  │   Rewriter   │    │  Retriever   │    │  Reranker    │    │ Constructor│ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └────────────┘ │
│         │                   │                   │                    │       │
│         ▼                   ▼                   ▼                    ▼       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │   Intent     │    │   BM25 +     │    │Cross-Encoder │    │Compression │ │
│  │  Detection   │    │   Vector     │    │   + SLM      │    │  Engine    │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └────────────┘ │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         LLM Answerer with Citations                    │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  │  │
│  │  │  Citation   │  │Hallucination│  │  Answer     │  │  Confidence  │  │  │
│  │  │  Extractor  │  │  Detector   │  │  Generator  │  │   Scorer     │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └──────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      Evaluation & Monitoring Layer                     │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  │  │
│  │  │  Retrieval  │  │   Answer    │  │   E2E       │  │   A/B Test   │  │  │
│  │  │   Metrics   │  │   Quality   │  │  Pipeline   │  │   Framework  │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └──────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

```
User Query
    │
    ▼
┌───────────────────┐
│  Query Analysis   │ ─── Intent Classification
│  & Rewriting      │ ─── Query Expansion
└─────────┬─────────┘ ─── Ambiguity Resolution
          │
          ▼
┌───────────────────┐     ┌─────────────────┐
│  Hybrid Retrieval │────▶│  Candidate Pool │
│  BM25 + Vector    │     │  (Top-K merged) │
└───────────────────┘     └────────┬────────┘
                                   │
                                   ▼
                    ┌─────────────────────────┐
                    │   Multi-Stage Reranking │
                    │   1. Cross-Encoder      │
                    │   2. SLM Relevance      │
                    │   3. Diversity Filter   │
                    └───────────┬─────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │   Context Construction  │
                    │   - Compression         │
                    │   - Deduplication       │
                    │   - Ordering            │
                    └───────────┬─────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │   LLM Answer Generation │
                    │   + Citation Extraction │
                    │   + Hallucination Check │
                    └───────────┬─────────────┘
                                │
                                ▼
                          Final Response
                          with Citations
```

---

## Core Components

### 1. Query Rewriter

The query rewriter transforms user queries into optimized retrieval queries through multiple strategies.

#### Architecture

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class QueryIntent(Enum):
    FACTUAL = "factual"
    COMPARISON = "comparison"
    PROCEDURAL = "procedural"
    EXPLORATORY = "exploratory"
    CLARIFICATION = "clarification"

@dataclass
class RewrittenQuery:
    original: str
    rewritten: str
    intent: QueryIntent
    expansions: List[str]
    sub_queries: List[str]
    confidence: float
    metadata: dict

class QueryRewriter(ABC):
    @abstractmethod
    async def rewrite(self, query: str, context: Optional[dict] = None) -> RewrittenQuery:
        pass

class LLMQueryRewriter(QueryRewriter):
    """LLM-based query rewriting with chain-of-thought reasoning."""

    def __init__(self, llm_client, prompt_template: str):
        self.llm = llm_client
        self.template = prompt_template

    async def rewrite(self, query: str, context: Optional[dict] = None) -> RewrittenQuery:
        # Intent detection
        intent = await self._detect_intent(query)

        # Query expansion based on intent
        expansions = await self._expand_query(query, intent)

        # Generate sub-queries for complex questions
        sub_queries = await self._decompose_query(query, intent)

        # Main rewrite
        rewritten = await self._rewrite_for_retrieval(query, intent, context)

        return RewrittenQuery(
            original=query,
            rewritten=rewritten,
            intent=intent,
            expansions=expansions,
            sub_queries=sub_queries,
            confidence=await self._compute_confidence(query, rewritten),
            metadata={"context_used": context is not None}
        )

    async def _detect_intent(self, query: str) -> QueryIntent:
        prompt = f"""Classify the intent of this query:
        Query: {query}

        Categories:
        - FACTUAL: Seeking specific facts or data
        - COMPARISON: Comparing entities or concepts
        - PROCEDURAL: How-to or step-by-step
        - EXPLORATORY: Open-ended exploration
        - CLARIFICATION: Seeking explanation

        Return only the category name."""

        response = await self.llm.generate(prompt)
        return QueryIntent(response.strip().lower())

    async def _expand_query(self, query: str, intent: QueryIntent) -> List[str]:
        prompt = f"""Generate 3-5 semantically related search queries for:
        Original: {query}
        Intent: {intent.value}

        Focus on synonyms, related concepts, and alternative phrasings.
        Return one query per line."""

        response = await self.llm.generate(prompt)
        return [q.strip() for q in response.split('\n') if q.strip()]

    async def _decompose_query(self, query: str, intent: QueryIntent) -> List[str]:
        if intent not in [QueryIntent.COMPARISON, QueryIntent.PROCEDURAL]:
            return []

        prompt = f"""Break down this complex query into atomic sub-queries:
        Query: {query}

        Each sub-query should be independently answerable.
        Return one sub-query per line."""

        response = await self.llm.generate(prompt)
        return [q.strip() for q in response.split('\n') if q.strip()]


class HybridQueryRewriter(QueryRewriter):
    """Combines rule-based and LLM-based rewriting strategies."""

    def __init__(self, llm_rewriter: LLMQueryRewriter, rules: List[RewriteRule]):
        self.llm_rewriter = llm_rewriter
        self.rules = rules

    async def rewrite(self, query: str, context: Optional[dict] = None) -> RewrittenQuery:
        # Apply rule-based transformations first
        rule_transformed = self._apply_rules(query)

        # Then apply LLM rewriting
        result = await self.llm_rewriter.rewrite(rule_transformed, context)

        return result

    def _apply_rules(self, query: str) -> str:
        result = query
        for rule in self.rules:
            result = rule.apply(result)
        return result
```

#### Rewrite Strategies

| Strategy | Use Case | Example |
|----------|----------|---------|
| Synonym Expansion | Improve recall | "ML" → "machine learning, ML, deep learning" |
| Acronym Resolution | Clarify ambiguity | "NLP tasks" → "natural language processing tasks" |
| Temporal Contextualization | Add time context | "latest research" → "research from 2024-2025" |
| Entity Disambiguation | Resolve ambiguity | "Python" → "Python programming language" |
| Query Decomposition | Complex queries | "Compare X and Y" → ["What is X?", "What is Y?", "Differences"] |

### 2. Hybrid Retriever (BM25 + Vector)

The hybrid retriever combines lexical and semantic search for optimal recall and precision.

#### Architecture

```python
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from abc import ABC, abstractmethod

@dataclass
class Document:
    id: str
    content: str
    metadata: dict
    embedding: Optional[np.ndarray] = None

@dataclass
class RetrievalResult:
    document: Document
    score: float
    retriever_type: str
    rank: int

class BaseRetriever(ABC):
    @abstractmethod
    async def retrieve(self, query: str, top_k: int) -> List[RetrievalResult]:
        pass

class BM25Retriever(BaseRetriever):
    """Lexical retrieval using BM25 algorithm."""

    def __init__(self, index_path: str, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.index = self._load_index(index_path)

    async def retrieve(self, query: str, top_k: int) -> List[RetrievalResult]:
        # Tokenize query
        tokens = self._tokenize(query)

        # Compute BM25 scores
        scores = self._compute_bm25_scores(tokens)

        # Get top-k results
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for rank, idx in enumerate(top_indices):
            doc = self.index.get_document(idx)
            results.append(RetrievalResult(
                document=doc,
                score=scores[idx],
                retriever_type="bm25",
                rank=rank
            ))

        return results

class VectorRetriever(BaseRetriever):
    """Dense retrieval using vector similarity."""

    def __init__(self, embedding_model, vector_store, similarity_metric: str = "cosine"):
        self.embedder = embedding_model
        self.vector_store = vector_store
        self.metric = similarity_metric

    async def retrieve(self, query: str, top_k: int) -> List[RetrievalResult]:
        # Embed query
        query_embedding = await self.embedder.embed(query)

        # Search vector store
        results = await self.vector_store.search(
            query_embedding,
            top_k=top_k,
            metric=self.metric
        )

        return [
            RetrievalResult(
                document=r.document,
                score=r.similarity,
                retriever_type="vector",
                rank=i
            )
            for i, r in enumerate(results)
        ]

class HybridRetriever(BaseRetriever):
    """Combines BM25 and vector retrieval with fusion strategies."""

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        vector_retriever: VectorRetriever,
        fusion_strategy: str = "rrf",  # rrf, linear, learned
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5
    ):
        self.bm25 = bm25_retriever
        self.vector = vector_retriever
        self.fusion_strategy = fusion_strategy
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

    async def retrieve(self, query: str, top_k: int) -> List[RetrievalResult]:
        # Retrieve from both sources (typically fetch more than top_k)
        fetch_k = top_k * 3

        bm25_results, vector_results = await asyncio.gather(
            self.bm25.retrieve(query, fetch_k),
            self.vector.retrieve(query, fetch_k)
        )

        # Fuse results
        if self.fusion_strategy == "rrf":
            fused = self._reciprocal_rank_fusion(bm25_results, vector_results)
        elif self.fusion_strategy == "linear":
            fused = self._linear_combination(bm25_results, vector_results)
        else:
            fused = self._learned_fusion(bm25_results, vector_results)

        return fused[:top_k]

    def _reciprocal_rank_fusion(
        self,
        results1: List[RetrievalResult],
        results2: List[RetrievalResult],
        k: int = 60
    ) -> List[RetrievalResult]:
        """RRF fusion: score = sum(1 / (k + rank))"""

        doc_scores = {}
        doc_results = {}

        for result in results1:
            doc_id = result.document.id
            rrf_score = 1 / (k + result.rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score * self.bm25_weight
            doc_results[doc_id] = result

        for result in results2:
            doc_id = result.document.id
            rrf_score = 1 / (k + result.rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score * self.vector_weight
            if doc_id not in doc_results:
                doc_results[doc_id] = result

        # Sort by fused score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        return [
            RetrievalResult(
                document=doc_results[doc_id].document,
                score=score,
                retriever_type="hybrid",
                rank=i
            )
            for i, (doc_id, score) in enumerate(sorted_docs)
        ]

    def _linear_combination(
        self,
        results1: List[RetrievalResult],
        results2: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Linear combination with normalized scores."""

        # Normalize scores to [0, 1]
        scores1 = self._normalize_scores(results1)
        scores2 = self._normalize_scores(results2)

        doc_scores = {}
        doc_results = {}

        for result, norm_score in zip(results1, scores1):
            doc_id = result.document.id
            doc_scores[doc_id] = norm_score * self.bm25_weight
            doc_results[doc_id] = result

        for result, norm_score in zip(results2, scores2):
            doc_id = result.document.id
            current = doc_scores.get(doc_id, 0)
            doc_scores[doc_id] = current + norm_score * self.vector_weight
            if doc_id not in doc_results:
                doc_results[doc_id] = result

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        return [
            RetrievalResult(
                document=doc_results[doc_id].document,
                score=score,
                retriever_type="hybrid",
                rank=i
            )
            for i, (doc_id, score) in enumerate(sorted_docs)
        ]
```

#### Fusion Strategy Comparison

| Strategy | Pros | Cons | Best For |
|----------|------|------|----------|
| RRF | No score normalization needed | Ignores score magnitude | General use |
| Linear | Uses score information | Requires normalization | Calibrated scores |
| Learned | Optimal weights | Needs training data | High-stakes apps |

### 3. Neural Reranker

Multi-stage reranking using cross-encoders and small language models for fine-grained relevance scoring.

#### Architecture

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
import torch
import torch.nn as nn

@dataclass
class RerankResult:
    document: Document
    original_rank: int
    new_rank: int
    relevance_score: float
    features: dict

class BaseReranker(ABC):
    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: int
    ) -> List[RerankResult]:
        pass

class CrossEncoderReranker(BaseReranker):
    """Cross-encoder reranking using transformer models."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    async def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: int
    ) -> List[RerankResult]:
        # Prepare query-document pairs
        pairs = [(query, c.document.content) for c in candidates]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Sort by score
        scored_candidates = list(zip(candidates, scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        results = []
        for new_rank, (candidate, score) in enumerate(scored_candidates[:top_k]):
            results.append(RerankResult(
                document=candidate.document,
                original_rank=candidate.rank,
                new_rank=new_rank,
                relevance_score=float(score),
                features={"cross_encoder_score": float(score)}
            ))

        return results

class SLMReranker(BaseReranker):
    """Small Language Model reranker with chain-of-thought scoring."""

    def __init__(self, slm_client, batch_size: int = 10):
        self.slm = slm_client
        self.batch_size = batch_size

    async def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: int
    ) -> List[RerankResult]:
        # Score in batches for efficiency
        all_scores = []

        for i in range(0, len(candidates), self.batch_size):
            batch = candidates[i:i + self.batch_size]
            scores = await self._score_batch(query, batch)
            all_scores.extend(scores)

        # Combine with original scores
        scored_candidates = list(zip(candidates, all_scores))
        scored_candidates.sort(key=lambda x: x[1]["relevance"], reverse=True)

        results = []
        for new_rank, (candidate, scores) in enumerate(scored_candidates[:top_k]):
            results.append(RerankResult(
                document=candidate.document,
                original_rank=candidate.rank,
                new_rank=new_rank,
                relevance_score=scores["relevance"],
                features=scores
            ))

        return results

    async def _score_batch(self, query: str, batch: List[RetrievalResult]) -> List[dict]:
        prompt = f"""Score each document's relevance to the query.

Query: {query}

Documents:
{self._format_documents(batch)}

For each document, provide:
1. Relevance (0-10): How well it answers the query
2. Coverage (0-10): Information completeness
3. Freshness (0-10): Temporal relevance if applicable

Return JSON array with scores."""

        response = await self.slm.generate(prompt)
        return self._parse_scores(response)

class MultiStageReranker(BaseReranker):
    """Multi-stage reranking pipeline with progressive filtering."""

    def __init__(
        self,
        stage1_reranker: BaseReranker,  # Fast, coarse (e.g., small cross-encoder)
        stage2_reranker: BaseReranker,  # Slow, fine (e.g., SLM)
        stage1_top_k: int = 50,
        diversity_filter: Optional[DiversityFilter] = None
    ):
        self.stage1 = stage1_reranker
        self.stage2 = stage2_reranker
        self.stage1_top_k = stage1_top_k
        self.diversity_filter = diversity_filter

    async def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: int
    ) -> List[RerankResult]:
        # Stage 1: Fast coarse reranking
        stage1_results = await self.stage1.rerank(query, candidates, self.stage1_top_k)

        # Convert to retrieval results for stage 2
        stage1_candidates = [
            RetrievalResult(
                document=r.document,
                score=r.relevance_score,
                retriever_type="stage1",
                rank=r.new_rank
            )
            for r in stage1_results
        ]

        # Stage 2: Fine-grained reranking
        stage2_results = await self.stage2.rerank(query, stage1_candidates, top_k * 2)

        # Optional diversity filtering
        if self.diversity_filter:
            stage2_results = self.diversity_filter.filter(stage2_results, top_k)

        return stage2_results[:top_k]

class DiversityFilter:
    """Ensures result diversity using MMR or clustering."""

    def __init__(self, method: str = "mmr", lambda_param: float = 0.7):
        self.method = method
        self.lambda_param = lambda_param

    def filter(self, results: List[RerankResult], top_k: int) -> List[RerankResult]:
        if self.method == "mmr":
            return self._mmr_filter(results, top_k)
        else:
            return self._cluster_filter(results, top_k)

    def _mmr_filter(self, results: List[RerankResult], top_k: int) -> List[RerankResult]:
        """Maximal Marginal Relevance for diversity."""
        selected = []
        remaining = list(results)

        while len(selected) < top_k and remaining:
            if not selected:
                # First document: highest relevance
                best = max(remaining, key=lambda x: x.relevance_score)
            else:
                # MMR: balance relevance and diversity
                best = max(
                    remaining,
                    key=lambda x: (
                        self.lambda_param * x.relevance_score -
                        (1 - self.lambda_param) * self._max_similarity(x, selected)
                    )
                )

            selected.append(best)
            remaining.remove(best)

        # Update ranks
        for i, result in enumerate(selected):
            result.new_rank = i

        return selected
```

### 4. Context Constructor

Assembles and compresses retrieved documents into optimal context for LLM generation.

#### Architecture

```python
from dataclasses import dataclass
from typing import List, Optional
from abc import ABC, abstractmethod

@dataclass
class ConstructedContext:
    content: str
    source_documents: List[Document]
    compression_ratio: float
    token_count: int
    metadata: dict

class ContextConstructor:
    """Constructs optimized context from retrieved documents."""

    def __init__(
        self,
        max_tokens: int = 4000,
        compressor: Optional[ContextCompressor] = None,
        deduplicator: Optional[Deduplicator] = None,
        orderer: Optional[ContextOrderer] = None
    ):
        self.max_tokens = max_tokens
        self.compressor = compressor or LLMCompressor()
        self.deduplicator = deduplicator or SemanticDeduplicator()
        self.orderer = orderer or RelevanceOrderer()

    async def construct(
        self,
        query: str,
        reranked_results: List[RerankResult]
    ) -> ConstructedContext:
        # Extract documents
        documents = [r.document for r in reranked_results]

        # Step 1: Deduplicate overlapping content
        deduplicated = await self.deduplicator.deduplicate(documents)

        # Step 2: Compress each document
        compressed = await self._compress_documents(query, deduplicated)

        # Step 3: Order for optimal comprehension
        ordered = self.orderer.order(query, compressed)

        # Step 4: Assemble within token budget
        context, used_docs = self._assemble_context(ordered)

        original_tokens = sum(self._count_tokens(d.content) for d in documents)
        final_tokens = self._count_tokens(context)

        return ConstructedContext(
            content=context,
            source_documents=used_docs,
            compression_ratio=final_tokens / original_tokens if original_tokens > 0 else 1.0,
            token_count=final_tokens,
            metadata={
                "original_count": len(documents),
                "used_count": len(used_docs),
                "dedup_removed": len(documents) - len(deduplicated)
            }
        )

    async def _compress_documents(
        self,
        query: str,
        documents: List[Document]
    ) -> List[Document]:
        compressed = []
        for doc in documents:
            compressed_content = await self.compressor.compress(query, doc.content)
            compressed.append(Document(
                id=doc.id,
                content=compressed_content,
                metadata={**doc.metadata, "compressed": True}
            ))
        return compressed

    def _assemble_context(self, documents: List[Document]) -> Tuple[str, List[Document]]:
        context_parts = []
        used_docs = []
        current_tokens = 0

        for i, doc in enumerate(documents):
            doc_tokens = self._count_tokens(doc.content)

            if current_tokens + doc_tokens > self.max_tokens:
                # Try to fit partial content
                remaining = self.max_tokens - current_tokens
                if remaining > 100:  # Minimum useful content
                    truncated = self._truncate_to_tokens(doc.content, remaining)
                    context_parts.append(f"[Source {i+1}]\n{truncated}\n")
                    used_docs.append(doc)
                break

            context_parts.append(f"[Source {i+1}]\n{doc.content}\n")
            used_docs.append(doc)
            current_tokens += doc_tokens

        return "\n".join(context_parts), used_docs

class LLMCompressor:
    """Uses LLM to compress documents while preserving query-relevant information."""

    def __init__(self, llm_client, target_ratio: float = 0.3):
        self.llm = llm_client
        self.target_ratio = target_ratio

    async def compress(self, query: str, content: str) -> str:
        prompt = f"""Compress the following text while preserving all information relevant to the query.

Query: {query}

Text to compress:
{content}

Requirements:
1. Keep all facts, numbers, and specific details relevant to the query
2. Remove redundant phrases and filler content
3. Maintain logical flow and coherence
4. Target compression: {int(self.target_ratio * 100)}% of original length

Compressed text:"""

        compressed = await self.llm.generate(prompt)
        return compressed.strip()

class SemanticDeduplicator:
    """Removes semantically duplicate content across documents."""

    def __init__(self, similarity_threshold: float = 0.85):
        self.threshold = similarity_threshold
        self.embedder = None  # Lazy load

    async def deduplicate(self, documents: List[Document]) -> List[Document]:
        if len(documents) <= 1:
            return documents

        # Chunk documents into passages
        all_passages = []
        for doc in documents:
            passages = self._chunk_document(doc)
            all_passages.extend([(doc.id, p) for p in passages])

        # Embed all passages
        embeddings = await self._embed_passages([p for _, p in all_passages])

        # Find and remove duplicates
        keep_indices = self._find_unique_passages(embeddings)

        # Reconstruct documents
        doc_passages = {}
        for i in keep_indices:
            doc_id, passage = all_passages[i]
            if doc_id not in doc_passages:
                doc_passages[doc_id] = []
            doc_passages[doc_id].append(passage)

        # Rebuild documents
        deduplicated = []
        for doc in documents:
            if doc.id in doc_passages:
                new_content = "\n\n".join(doc_passages[doc.id])
                deduplicated.append(Document(
                    id=doc.id,
                    content=new_content,
                    metadata=doc.metadata
                ))

        return deduplicated
```

### 5. LLM Answerer with Citations

Generates grounded answers with inline citations and confidence scoring.

#### Architecture

```python
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class Citation:
    source_id: str
    source_title: str
    quoted_text: str
    relevance_score: float

@dataclass
class GeneratedAnswer:
    answer: str
    citations: List[Citation]
    confidence: float
    hallucination_flags: List[str]
    metadata: dict

class LLMAnswerer:
    """Generates answers with citations and hallucination detection."""

    def __init__(
        self,
        llm_client,
        citation_extractor: CitationExtractor,
        hallucination_detector: HallucinationDetector
    ):
        self.llm = llm_client
        self.citation_extractor = citation_extractor
        self.hallucination_detector = hallucination_detector

    async def generate(
        self,
        query: str,
        context: ConstructedContext
    ) -> GeneratedAnswer:
        # Generate answer with inline citations
        raw_answer = await self._generate_with_citations(query, context)

        # Extract structured citations
        answer_text, citations = await self.citation_extractor.extract(
            raw_answer,
            context.source_documents
        )

        # Detect hallucinations
        hallucination_flags = await self.hallucination_detector.detect(
            query,
            answer_text,
            context.content
        )

        # Compute confidence
        confidence = self._compute_confidence(
            citations,
            hallucination_flags,
            context
        )

        return GeneratedAnswer(
            answer=answer_text,
            citations=citations,
            confidence=confidence,
            hallucination_flags=hallucination_flags,
            metadata={
                "sources_used": len(context.source_documents),
                "context_tokens": context.token_count
            }
        )

    async def _generate_with_citations(
        self,
        query: str,
        context: ConstructedContext
    ) -> str:
        prompt = f"""Answer the question based ONLY on the provided sources.

Question: {query}

Sources:
{context.content}

Instructions:
1. Answer the question directly and comprehensively
2. Use [1], [2], etc. to cite sources inline
3. If information is not in the sources, say "I don't have information about..."
4. Never make up information not in the sources
5. If sources conflict, mention the discrepancy

Answer:"""

        return await self.llm.generate(prompt)

    def _compute_confidence(
        self,
        citations: List[Citation],
        hallucination_flags: List[str],
        context: ConstructedContext
    ) -> float:
        # Base confidence from citation coverage
        citation_score = min(len(citations) / 3, 1.0) * 0.4

        # Penalty for hallucinations
        hallucination_penalty = len(hallucination_flags) * 0.15

        # Bonus for high-relevance sources
        relevance_bonus = sum(c.relevance_score for c in citations) / max(len(citations), 1) * 0.3

        # Bonus for good context compression
        compression_bonus = (1 - context.compression_ratio) * 0.1

        confidence = citation_score + relevance_bonus + compression_bonus - hallucination_penalty
        return max(0, min(1, confidence))

class HallucinationDetector:
    """Detects unsupported claims in generated answers."""

    def __init__(self, llm_client, nli_model=None):
        self.llm = llm_client
        self.nli_model = nli_model

    async def detect(
        self,
        query: str,
        answer: str,
        context: str
    ) -> List[str]:
        # Extract claims from answer
        claims = await self._extract_claims(answer)

        # Verify each claim against context
        unsupported_claims = []
        for claim in claims:
            is_supported = await self._verify_claim(claim, context)
            if not is_supported:
                unsupported_claims.append(claim)

        return unsupported_claims

    async def _extract_claims(self, answer: str) -> List[str]:
        prompt = f"""Extract all factual claims from this answer.

Answer: {answer}

List each distinct factual claim on a separate line.
Only include claims that can be verified (not opinions or hedged statements).

Claims:"""

        response = await self.llm.generate(prompt)
        return [c.strip() for c in response.split('\n') if c.strip()]

    async def _verify_claim(self, claim: str, context: str) -> bool:
        if self.nli_model:
            # Use NLI model for fast verification
            return self._nli_verify(claim, context)

        # Fall back to LLM verification
        prompt = f"""Determine if the following claim is supported by the context.

Claim: {claim}

Context:
{context}

Answer with only "SUPPORTED" or "NOT SUPPORTED"."""

        response = await self.llm.generate(prompt)
        return "SUPPORTED" in response.upper()

class CitationExtractor:
    """Extracts and validates citations from generated text."""

    def __init__(self, fuzzy_threshold: float = 0.8):
        self.fuzzy_threshold = fuzzy_threshold

    async def extract(
        self,
        answer: str,
        source_documents: List[Document]
    ) -> Tuple[str, List[Citation]]:
        import re

        # Find citation markers [1], [2], etc.
        pattern = r'\[(\d+)\]'
        markers = re.findall(pattern, answer)

        citations = []
        for marker in set(markers):
            idx = int(marker) - 1
            if 0 <= idx < len(source_documents):
                doc = source_documents[idx]

                # Find quoted text near citation
                quoted = self._find_quoted_text(answer, marker, doc.content)

                citations.append(Citation(
                    source_id=doc.id,
                    source_title=doc.metadata.get("title", f"Source {marker}"),
                    quoted_text=quoted,
                    relevance_score=self._compute_relevance(quoted, doc.content)
                ))

        return answer, citations
```

---

## Technical Specifications

### Performance Requirements

| Metric | Target | Notes |
|--------|--------|-------|
| Query Rewriting Latency | < 200ms | P95 |
| Retrieval Latency | < 100ms | P95, hybrid |
| Reranking Latency | < 500ms | P95, 100 docs |
| End-to-End Latency | < 3s | P95, with generation |
| Throughput | 100 QPS | Per instance |
| Retrieval Recall@100 | > 0.90 | On eval set |
| Answer Accuracy | > 0.85 | Human eval |

### Storage Requirements

| Component | Storage Type | Size Estimate |
|-----------|--------------|---------------|
| BM25 Index | Disk (SSD) | 2-5x raw text |
| Vector Store | Memory/Disk | ~1.5KB per doc (768d) |
| Document Store | PostgreSQL | ~2x raw text |
| Metadata Store | PostgreSQL | ~100 bytes per doc |
| Cache Layer | Redis | 10GB per instance |

### Model Specifications

| Component | Model Options | Size | Notes |
|-----------|---------------|------|-------|
| Query Embedder | all-MiniLM-L6-v2 | 80MB | Fast, good quality |
| Document Embedder | e5-large-v2 | 1.3GB | Higher quality |
| Cross-Encoder | ms-marco-MiniLM-L-12 | 120MB | Stage 1 reranking |
| SLM Reranker | Phi-3-mini, Qwen2-1.5B | 1-4GB | Stage 2 reranking |
| Generator LLM | GPT-4, Claude, Llama | API/70GB | Answer generation |

---

## Data Flow Architecture

### Request Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        Request Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. INPUT VALIDATION                                            │
│     ├─ Query sanitization                                       │
│     ├─ Length validation                                        │
│     └─ Tenant authentication                                    │
│                                                                  │
│  2. QUERY PROCESSING                                            │
│     ├─ Intent detection                                         │
│     ├─ Query rewriting                                          │
│     ├─ Query expansion                                          │
│     └─ Sub-query generation                                     │
│                                                                  │
│  3. RETRIEVAL                                                   │
│     ├─ Parallel BM25 + Vector search                           │
│     ├─ Result fusion (RRF/Linear)                              │
│     └─ Initial candidate pool                                   │
│                                                                  │
│  4. RERANKING                                                   │
│     ├─ Stage 1: Cross-encoder (fast)                           │
│     ├─ Stage 2: SLM scoring (detailed)                         │
│     └─ Diversity filtering (MMR)                                │
│                                                                  │
│  5. CONTEXT CONSTRUCTION                                        │
│     ├─ Semantic deduplication                                   │
│     ├─ Content compression                                      │
│     ├─ Relevance ordering                                       │
│     └─ Token budget assembly                                    │
│                                                                  │
│  6. GENERATION                                                  │
│     ├─ Answer generation with citations                         │
│     ├─ Citation extraction                                      │
│     ├─ Hallucination detection                                  │
│     └─ Confidence scoring                                       │
│                                                                  │
│  7. POST-PROCESSING                                             │
│     ├─ Response formatting                                      │
│     ├─ Metrics logging                                          │
│     └─ Cache update                                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Caching Strategy

```python
class RAGCacheManager:
    """Multi-level caching for RAG pipeline."""

    def __init__(self, redis_client, local_cache_size: int = 1000):
        self.redis = redis_client
        self.local_cache = LRUCache(local_cache_size)

    async def get_or_compute_embedding(self, text: str, embedder) -> np.ndarray:
        cache_key = f"emb:{self._hash(text)}"

        # L1: Local cache
        if cache_key in self.local_cache:
            return self.local_cache[cache_key]

        # L2: Redis cache
        cached = await self.redis.get(cache_key)
        if cached:
            embedding = np.frombuffer(cached, dtype=np.float32)
            self.local_cache[cache_key] = embedding
            return embedding

        # Compute
        embedding = await embedder.embed(text)

        # Store in both caches
        await self.redis.setex(cache_key, 3600, embedding.tobytes())
        self.local_cache[cache_key] = embedding

        return embedding

    async def get_or_compute_retrieval(
        self,
        query: str,
        retriever,
        top_k: int,
        ttl: int = 300
    ) -> List[RetrievalResult]:
        cache_key = f"ret:{self._hash(query)}:{top_k}"

        cached = await self.redis.get(cache_key)
        if cached:
            return self._deserialize_results(cached)

        results = await retriever.retrieve(query, top_k)

        await self.redis.setex(cache_key, ttl, self._serialize_results(results))

        return results
```

---

## API Design

### REST API

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional

app = FastAPI(title="Advanced RAG API")

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    include_citations: bool = True
    include_confidence: bool = True
    tenant_id: Optional[str] = None
    config_override: Optional[dict] = None

class CitationResponse(BaseModel):
    source_id: str
    title: str
    excerpt: str
    relevance: float

class RAGResponse(BaseModel):
    answer: str
    citations: List[CitationResponse]
    confidence: float
    processing_time_ms: int
    metadata: dict

@app.post("/v1/query", response_model=RAGResponse)
async def query_endpoint(
    request: QueryRequest,
    background_tasks: BackgroundTasks
):
    start_time = time.time()

    try:
        # Load tenant configuration
        config = await load_tenant_config(request.tenant_id, request.config_override)

        # Execute RAG pipeline
        result = await rag_pipeline.execute(
            query=request.query,
            top_k=request.top_k,
            config=config
        )

        # Log metrics asynchronously
        background_tasks.add_task(
            log_metrics,
            query=request.query,
            result=result,
            latency=time.time() - start_time
        )

        return RAGResponse(
            answer=result.answer,
            citations=[
                CitationResponse(
                    source_id=c.source_id,
                    title=c.source_title,
                    excerpt=c.quoted_text[:200],
                    relevance=c.relevance_score
                )
                for c in result.citations
            ] if request.include_citations else [],
            confidence=result.confidence if request.include_confidence else 0,
            processing_time_ms=int((time.time() - start_time) * 1000),
            metadata=result.metadata
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/index")
async def index_documents(documents: List[DocumentInput]):
    """Index new documents into the RAG system."""
    results = await indexing_pipeline.index_batch(documents)
    return {"indexed": len(results), "errors": [r for r in results if r.error]}

@app.get("/v1/evaluate/{eval_id}")
async def get_evaluation_results(eval_id: str):
    """Get results of an evaluation run."""
    results = await evaluation_store.get(eval_id)
    if not results:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return results
```

### Streaming API

```python
from fastapi.responses import StreamingResponse
import asyncio

@app.post("/v1/query/stream")
async def stream_query(request: QueryRequest):
    """Stream RAG response with intermediate results."""

    async def generate():
        # Stream retrieval results
        retrieval_results = await retriever.retrieve(request.query, request.top_k * 3)
        yield f"data: {json.dumps({'type': 'retrieval', 'count': len(retrieval_results)})}\n\n"

        # Stream reranking progress
        reranked = await reranker.rerank(request.query, retrieval_results, request.top_k)
        yield f"data: {json.dumps({'type': 'reranking', 'top_docs': [r.document.id for r in reranked[:3]]})}\n\n"

        # Stream answer tokens
        context = await context_constructor.construct(request.query, reranked)

        async for token in llm_answerer.stream_generate(request.query, context):
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        # Final metadata
        yield f"data: {json.dumps({'type': 'done', 'citations': len(reranked)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

---

## Implementation Phases

### Phase 1: Core Retrieval (Weeks 1-3)

**Objectives:**
- Implement BM25 and vector retrieval
- Basic hybrid fusion (RRF)
- Simple query interface

**Deliverables:**
- [ ] BM25 indexing and search
- [ ] Vector store integration (FAISS/Milvus)
- [ ] Embedding pipeline
- [ ] RRF fusion implementation
- [ ] Basic REST API
- [ ] Unit tests for retrieval

**Success Criteria:**
- Retrieval latency < 100ms P95
- Recall@100 > 0.85

### Phase 2: Query Enhancement (Weeks 4-5)

**Objectives:**
- Implement query rewriting
- Add query expansion
- Intent detection

**Deliverables:**
- [ ] LLM-based query rewriter
- [ ] Intent classification
- [ ] Query expansion module
- [ ] Sub-query decomposition
- [ ] Integration tests

**Success Criteria:**
- Query rewriting improves recall by 10%+
- Latency overhead < 200ms

### Phase 3: Neural Reranking (Weeks 6-8)

**Objectives:**
- Cross-encoder reranking
- SLM-based scoring
- Multi-stage pipeline

**Deliverables:**
- [ ] Cross-encoder integration
- [ ] SLM reranker implementation
- [ ] Multi-stage orchestration
- [ ] Diversity filtering (MMR)
- [ ] Benchmarking suite

**Success Criteria:**
- MRR improvement > 15%
- Reranking latency < 500ms

### Phase 4: Context & Generation (Weeks 9-11)

**Objectives:**
- Context construction pipeline
- Citation generation
- Hallucination detection

**Deliverables:**
- [ ] Document compression
- [ ] Semantic deduplication
- [ ] Citation extraction
- [ ] Hallucination detector
- [ ] Confidence scoring

**Success Criteria:**
- Context compression > 50%
- Hallucination detection F1 > 0.80

### Phase 5: Enterprise Features (Weeks 12-15)

**Objectives:**
- Multi-tenant support
- Pluggable rerankers
- Evaluation framework

**Deliverables:**
- [ ] Tenant configuration system
- [ ] Plugin architecture for rerankers
- [ ] Comprehensive eval suite
- [ ] A/B testing framework
- [ ] Admin dashboard

**Success Criteria:**
- Support 10+ tenants
- Evaluation coverage > 90%

### Phase 6: Optimization & Hardening (Weeks 16-18)

**Objectives:**
- Performance optimization
- Production hardening
- Documentation

**Deliverables:**
- [ ] Caching layer optimization
- [ ] Batch processing
- [ ] Monitoring & alerting
- [ ] Load testing
- [ ] API documentation

**Success Criteria:**
- 100 QPS throughput
- 99.9% availability

---

## Enterprise Features

### Multi-Tenant Configuration

```python
@dataclass
class TenantConfig:
    tenant_id: str

    # Retrieval settings
    retrieval_top_k: int = 100
    bm25_weight: float = 0.4
    vector_weight: float = 0.6
    fusion_strategy: str = "rrf"

    # Reranking settings
    reranker_type: str = "cross-encoder"  # or "slm", "ensemble"
    reranking_top_k: int = 10
    use_diversity_filter: bool = True

    # Generation settings
    llm_model: str = "gpt-4"
    max_context_tokens: int = 4000
    include_citations: bool = True

    # Quality settings
    hallucination_check: bool = True
    min_confidence_threshold: float = 0.3

    # Rate limiting
    requests_per_minute: int = 100
    max_concurrent: int = 10

class TenantConfigManager:
    """Manages per-tenant configurations with hot-reload support."""

    def __init__(self, config_store):
        self.store = config_store
        self.cache = {}

    async def get_config(self, tenant_id: str) -> TenantConfig:
        if tenant_id in self.cache:
            return self.cache[tenant_id]

        config_data = await self.store.get(tenant_id)
        if not config_data:
            return TenantConfig(tenant_id=tenant_id)  # Defaults

        config = TenantConfig(**config_data)
        self.cache[tenant_id] = config
        return config

    async def update_config(self, tenant_id: str, updates: dict):
        current = await self.get_config(tenant_id)
        for key, value in updates.items():
            if hasattr(current, key):
                setattr(current, key, value)

        await self.store.set(tenant_id, asdict(current))
        self.cache[tenant_id] = current
```

### Pluggable Reranker System

```python
class RerankerRegistry:
    """Registry for pluggable rerankers."""

    def __init__(self):
        self._rerankers = {}

    def register(self, name: str, reranker_class: type):
        self._rerankers[name] = reranker_class

    def get(self, name: str, **kwargs) -> BaseReranker:
        if name not in self._rerankers:
            raise ValueError(f"Unknown reranker: {name}")
        return self._rerankers[name](**kwargs)

    def list_available(self) -> List[str]:
        return list(self._rerankers.keys())

# Built-in rerankers
registry = RerankerRegistry()
registry.register("cross-encoder", CrossEncoderReranker)
registry.register("slm", SLMReranker)
registry.register("bge", BGEReranker)
registry.register("cohere", CohereReranker)

# Custom reranker example
class CustomBusinessReranker(BaseReranker):
    """Domain-specific reranker with business rules."""

    async def rerank(self, query, candidates, top_k):
        # Apply business logic
        boosted = self._apply_business_rules(candidates)
        # Then neural reranking
        return await self.neural_reranker.rerank(query, boosted, top_k)

registry.register("custom-business", CustomBusinessReranker)
```

---

## Evaluation Framework

### Comprehensive Evaluation Suite

```python
from dataclasses import dataclass
from typing import List, Dict
import pandas as pd

@dataclass
class EvaluationResult:
    metric_name: str
    value: float
    confidence_interval: tuple
    metadata: dict

class RAGEvaluator:
    """Comprehensive evaluation framework for RAG systems."""

    def __init__(self, rag_pipeline):
        self.pipeline = rag_pipeline
        self.metrics = {}

    def register_metric(self, name: str, metric_fn):
        self.metrics[name] = metric_fn

    async def evaluate(
        self,
        test_set: List[dict],
        metrics: List[str] = None
    ) -> Dict[str, EvaluationResult]:
        metrics = metrics or list(self.metrics.keys())

        # Run pipeline on test set
        predictions = []
        for item in test_set:
            result = await self.pipeline.execute(item["query"])
            predictions.append({
                "query": item["query"],
                "expected": item.get("expected_answer"),
                "expected_docs": item.get("relevant_docs", []),
                "predicted": result
            })

        # Compute metrics
        results = {}
        for metric_name in metrics:
            if metric_name in self.metrics:
                result = self.metrics[metric_name](predictions)
                results[metric_name] = result

        return results

# Retrieval metrics
def retrieval_recall_at_k(predictions: List[dict], k: int = 10) -> EvaluationResult:
    recalls = []
    for pred in predictions:
        expected = set(pred["expected_docs"])
        retrieved = set([r.document.id for r in pred["predicted"].retrieval_results[:k]])
        recall = len(expected & retrieved) / len(expected) if expected else 1.0
        recalls.append(recall)

    return EvaluationResult(
        metric_name=f"recall@{k}",
        value=np.mean(recalls),
        confidence_interval=_bootstrap_ci(recalls),
        metadata={"k": k, "n_samples": len(recalls)}
    )

def retrieval_mrr(predictions: List[dict]) -> EvaluationResult:
    mrrs = []
    for pred in predictions:
        expected = set(pred["expected_docs"])
        for rank, result in enumerate(pred["predicted"].retrieval_results):
            if result.document.id in expected:
                mrrs.append(1 / (rank + 1))
                break
        else:
            mrrs.append(0)

    return EvaluationResult(
        metric_name="mrr",
        value=np.mean(mrrs),
        confidence_interval=_bootstrap_ci(mrrs),
        metadata={"n_samples": len(mrrs)}
    )

# Answer quality metrics
def answer_faithfulness(predictions: List[dict], llm_judge) -> EvaluationResult:
    """Measures if answer is faithful to retrieved context."""
    scores = []
    for pred in predictions:
        answer = pred["predicted"].answer
        context = pred["predicted"].context.content

        score = llm_judge.evaluate_faithfulness(answer, context)
        scores.append(score)

    return EvaluationResult(
        metric_name="faithfulness",
        value=np.mean(scores),
        confidence_interval=_bootstrap_ci(scores),
        metadata={"n_samples": len(scores)}
    )

def answer_relevance(predictions: List[dict], llm_judge) -> EvaluationResult:
    """Measures if answer is relevant to query."""
    scores = []
    for pred in predictions:
        query = pred["query"]
        answer = pred["predicted"].answer

        score = llm_judge.evaluate_relevance(query, answer)
        scores.append(score)

    return EvaluationResult(
        metric_name="relevance",
        value=np.mean(scores),
        confidence_interval=_bootstrap_ci(scores),
        metadata={"n_samples": len(scores)}
    )

# RAGAS-style metrics
def context_precision(predictions: List[dict]) -> EvaluationResult:
    """Proportion of retrieved context that is relevant."""
    pass

def context_recall(predictions: List[dict]) -> EvaluationResult:
    """Proportion of relevant information that was retrieved."""
    pass

def answer_correctness(predictions: List[dict], ground_truth) -> EvaluationResult:
    """Semantic similarity to ground truth answers."""
    pass
```

### A/B Testing Framework

```python
class ABTestManager:
    """Manages A/B tests for RAG configurations."""

    def __init__(self, storage):
        self.storage = storage
        self.active_tests = {}

    async def create_test(
        self,
        name: str,
        control_config: dict,
        treatment_config: dict,
        traffic_split: float = 0.5,
        metrics: List[str] = None
    ) -> str:
        test_id = generate_id()

        test = {
            "id": test_id,
            "name": name,
            "control": control_config,
            "treatment": treatment_config,
            "traffic_split": traffic_split,
            "metrics": metrics or ["latency", "relevance", "faithfulness"],
            "status": "active",
            "start_time": datetime.utcnow(),
            "results": {"control": [], "treatment": []}
        }

        await self.storage.save(test_id, test)
        self.active_tests[test_id] = test

        return test_id

    async def get_variant(self, test_id: str, user_id: str) -> str:
        """Deterministically assign user to variant."""
        test = self.active_tests.get(test_id)
        if not test:
            return "control"

        # Consistent hashing for user assignment
        hash_val = hash(f"{test_id}:{user_id}") % 100
        return "treatment" if hash_val < test["traffic_split"] * 100 else "control"

    async def record_result(
        self,
        test_id: str,
        variant: str,
        metrics: dict
    ):
        """Record metrics for a request in the test."""
        test = self.active_tests.get(test_id)
        if test:
            test["results"][variant].append(metrics)

    async def analyze_test(self, test_id: str) -> dict:
        """Statistical analysis of test results."""
        test = await self.storage.get(test_id)

        analysis = {}
        for metric in test["metrics"]:
            control_vals = [r[metric] for r in test["results"]["control"]]
            treatment_vals = [r[metric] for r in test["results"]["treatment"]]

            # T-test for significance
            t_stat, p_value = stats.ttest_ind(control_vals, treatment_vals)

            analysis[metric] = {
                "control_mean": np.mean(control_vals),
                "treatment_mean": np.mean(treatment_vals),
                "lift": (np.mean(treatment_vals) - np.mean(control_vals)) / np.mean(control_vals),
                "p_value": p_value,
                "significant": p_value < 0.05
            }

        return analysis
```

---

## Deployment Architecture

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: advanced-rag
  namespace: ml-platform
spec:
  replicas: 3
  selector:
    matchLabels:
      app: advanced-rag
  template:
    metadata:
      labels:
        app: advanced-rag
    spec:
      containers:
      - name: rag-api
        image: advanced-rag:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        env:
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: redis-url
        - name: VECTOR_STORE_URL
          valueFrom:
            configMapKeyRef:
              name: rag-config
              key: vector-store-url
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

      - name: reranker-sidecar
        image: reranker-service:latest
        ports:
        - containerPort: 8001
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
            nvidia.com/gpu: 1
          limits:
            memory: "4Gi"
            nvidia.com/gpu: 1
---
apiVersion: v1
kind: Service
metadata:
  name: advanced-rag
spec:
  selector:
    app: advanced-rag
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: advanced-rag-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: advanced-rag
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: requests_per_second
      target:
        type: AverageValue
        averageValue: "50"
```

### Monitoring Setup

```python
from prometheus_client import Counter, Histogram, Gauge

# Metrics definitions
REQUEST_COUNT = Counter(
    'rag_requests_total',
    'Total RAG requests',
    ['tenant_id', 'status']
)

REQUEST_LATENCY = Histogram(
    'rag_request_latency_seconds',
    'RAG request latency',
    ['stage'],  # retrieval, reranking, generation
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

RETRIEVAL_RESULTS = Histogram(
    'rag_retrieval_results',
    'Number of retrieval results',
    buckets=[10, 25, 50, 100, 200]
)

CONFIDENCE_SCORE = Histogram(
    'rag_confidence_score',
    'Answer confidence scores',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

HALLUCINATION_COUNT = Counter(
    'rag_hallucinations_total',
    'Detected hallucinations',
    ['severity']
)

CACHE_HIT_RATE = Gauge(
    'rag_cache_hit_rate',
    'Cache hit rate',
    ['cache_type']
)
```

---

## Stretch Goals

### Multi-Hop Retrieval

```python
class MultiHopRetriever:
    """Iterative retrieval for complex multi-hop questions."""

    def __init__(self, base_retriever, llm_client, max_hops: int = 3):
        self.retriever = base_retriever
        self.llm = llm_client
        self.max_hops = max_hops

    async def retrieve(self, query: str, top_k: int) -> List[RetrievalResult]:
        all_results = []
        current_query = query
        context = ""

        for hop in range(self.max_hops):
            # Retrieve for current query
            results = await self.retriever.retrieve(current_query, top_k)
            all_results.extend(results)

            # Update context
            context += "\n".join([r.document.content for r in results])

            # Check if we need another hop
            needs_more, next_query = await self._check_completeness(
                query, context, current_query
            )

            if not needs_more:
                break

            current_query = next_query

        # Deduplicate and re-score
        return self._deduplicate_and_rank(all_results)

    async def _check_completeness(
        self,
        original_query: str,
        context: str,
        last_query: str
    ) -> Tuple[bool, str]:
        prompt = f"""Given the original question and retrieved context, determine if more information is needed.

Original Question: {original_query}
Last Search Query: {last_query}

Retrieved Context:
{context[:2000]}

If the context fully answers the question, respond with: COMPLETE
If more information is needed, respond with: NEED_MORE: <follow-up search query>"""

        response = await self.llm.generate(prompt)

        if "COMPLETE" in response:
            return False, ""
        elif "NEED_MORE:" in response:
            next_query = response.split("NEED_MORE:")[1].strip()
            return True, next_query

        return False, ""
```

### Evidence Consistency Scoring

```python
class EvidenceConsistencyScorer:
    """Scores consistency of evidence across multiple sources."""

    def __init__(self, llm_client, nli_model=None):
        self.llm = llm_client
        self.nli_model = nli_model

    async def score_consistency(
        self,
        query: str,
        documents: List[Document]
    ) -> dict:
        # Extract key claims from each document
        doc_claims = {}
        for doc in documents:
            claims = await self._extract_claims(doc.content, query)
            doc_claims[doc.id] = claims

        # Build claim-evidence matrix
        all_claims = set()
        for claims in doc_claims.values():
            all_claims.update(claims)

        # Check each claim against each document
        consistency_matrix = {}
        for claim in all_claims:
            consistency_matrix[claim] = {}
            for doc_id, doc in [(d.id, d) for d in documents]:
                support = await self._check_support(claim, doc.content)
                consistency_matrix[claim][doc_id] = support

        # Compute consistency scores
        scores = self._compute_scores(consistency_matrix)

        return {
            "overall_consistency": scores["overall"],
            "claim_scores": scores["per_claim"],
            "conflicting_claims": scores["conflicts"],
            "well_supported_claims": scores["supported"]
        }

    def _compute_scores(self, matrix: dict) -> dict:
        conflicts = []
        supported = []

        for claim, supports in matrix.items():
            support_values = list(supports.values())

            # Check for conflicts (some support, some contradict)
            if "supports" in support_values and "contradicts" in support_values:
                conflicts.append(claim)
            elif support_values.count("supports") >= 2:
                supported.append(claim)

        overall = 1 - (len(conflicts) / len(matrix)) if matrix else 1.0

        return {
            "overall": overall,
            "per_claim": {c: self._claim_score(s) for c, s in matrix.items()},
            "conflicts": conflicts,
            "supported": supported
        }
```

---

## File Structure

```
26-advanced-rag/
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── routes.py
│   │   └── middleware.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── bm25.py
│   │   ├── vector.py
│   │   ├── hybrid.py
│   │   └── fusion.py
│   ├── reranking/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── cross_encoder.py
│   │   ├── slm.py
│   │   └── diversity.py
│   ├── query/
│   │   ├── __init__.py
│   │   ├── rewriter.py
│   │   ├── intent.py
│   │   └── expansion.py
│   ├── context/
│   │   ├── __init__.py
│   │   ├── constructor.py
│   │   ├── compressor.py
│   │   └── deduplicator.py
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── answerer.py
│   │   ├── citations.py
│   │   └── hallucination.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── evaluator.py
│   │   └── ab_testing.py
│   └── utils/
│       ├── __init__.py
│       ├── cache.py
│       ├── config.py
│       └── logging.py
├── config/
│   ├── default.yaml
│   ├── production.yaml
│   └── tenants/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── evaluation/
├── docs/
│   ├── api.md
│   └── deployment.md
├── BLUEPRINT.md
├── PROGRESS.md
└── SESSION_CONTEXT.md
```

---

## Dependencies

```toml
[project]
name = "advanced-rag"
version = "0.1.0"
dependencies = [
    # Core
    "fastapi>=0.100.0",
    "uvicorn>=0.23.0",
    "pydantic>=2.0.0",

    # ML/NLP
    "torch>=2.0.0",
    "transformers>=4.30.0",
    "sentence-transformers>=2.2.0",

    # Vector stores
    "faiss-cpu>=1.7.4",
    "pymilvus>=2.3.0",

    # Search
    "rank-bm25>=0.2.2",
    "elasticsearch>=8.0.0",

    # LLM clients
    "openai>=1.0.0",
    "anthropic>=0.5.0",

    # Storage
    "redis>=4.5.0",
    "asyncpg>=0.28.0",
    "sqlalchemy>=2.0.0",

    # Monitoring
    "prometheus-client>=0.17.0",
    "structlog>=23.1.0",

    # Evaluation
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scipy>=1.10.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "mypy>=1.0.0",
]
```

---

## Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| End-to-End Latency P95 | < 3s | Prometheus histogram |
| Retrieval Recall@10 | > 0.85 | Offline evaluation |
| Answer Faithfulness | > 0.90 | LLM judge evaluation |
| Answer Relevance | > 0.85 | LLM judge evaluation |
| Hallucination Rate | < 5% | Automated detection |
| System Availability | 99.9% | Uptime monitoring |
| User Satisfaction | > 4.0/5 | User feedback |

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM API failures | High | Fallback models, caching, circuit breakers |
| Vector store scaling | Medium | Sharding, approximate search |
| Hallucination in answers | High | Multi-stage detection, confidence thresholds |
| High latency | Medium | Caching, async processing, model optimization |
| Data privacy | High | Tenant isolation, encryption, audit logs |

---

## References

- [RAGAS Evaluation Framework](https://github.com/explodinggradients/ragas)
- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [Cross-Encoder Reranking](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- [LLM-based Query Rewriting](https://arxiv.org/abs/2305.14283)
- [Hallucination Detection Survey](https://arxiv.org/abs/2311.05232)
