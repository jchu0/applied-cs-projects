# Project 27: Micro-Model Orchestrated RAG (SLM Family)

## FLAGSHIP PROJECT

## Executive Summary

A revolutionary RAG system where every stage is powered by specialized Small Language Models (SLMs), orchestrated through a dynamic computation graph. Each SLM is fine-tuned for its specific task (chunking, embedding, retrieval, reranking, summarization, CoT compression, answer stabilization), enabling superior quality while maintaining low latency. The orchestrator dynamically selects models, manages guardrails, and tracks latency/quality metrics with per-step tracing.

> **Concepts covered:** [§04 RAG systems](../../04-ai-engineering/02-llm-applications/rag/rag-systems.md) · [§04 Custom models / fine-tuning](../../04-ai-engineering/07-custom-models/) · [§04 Embeddings](../../04-ai-engineering/03-vector-databases/embeddings/embeddings.md). For baseline and advanced retrieval pipelines, see [Project 25](../25-rag-baseline/) and [Project 26](../26-advanced-rag/). Map: [`CONCEPT_TO_PROJECT_MAP.md`](../CONCEPT_TO_PROJECT_MAP.md).

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [SLM Component Specifications](#slm-component-specifications)
3. [Orchestration Engine](#orchestration-engine)
4. [Dynamic Computation Graphs](#dynamic-computation-graphs)
5. [Model Selection & Routing](#model-selection--routing)
6. [Dataset Design Per Model](#dataset-design-per-model)
7. [Enterprise Features](#enterprise-features)
8. [Implementation Phases](#implementation-phases)
9. [Stretch Goals](#stretch-goals)

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    Micro-Model Orchestrated RAG System                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                        Dynamic Orchestrator                                │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │  │
│  │  │   Graph     │  │   Model     │  │   Quality   │  │    Guardrail    │  │  │
│  │  │   Builder   │  │   Selector  │  │   Tracker   │  │    Engine       │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                    Specialized SLM Pipeline                                │  │
│  │                                                                            │  │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐               │  │
│  │  │ Chunker  │──▶│ Embedder │──▶│ Retriever│──▶│ Reranker │               │  │
│  │  │   SLM    │   │   SLM    │   │   SLM    │   │   SLM    │               │  │
│  │  └──────────┘   └──────────┘   └──────────┘   └──────────┘               │  │
│  │       │              │              │              │                      │  │
│  │       ▼              ▼              ▼              ▼                      │  │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐               │  │
│  │  │Semantic  │   │ Domain   │   │  Query   │   │Cross-Attn│               │  │
│  │  │Boundaries│   │ Adapted  │   │ Routing  │   │ Scoring  │               │  │
│  │  └──────────┘   └──────────┘   └──────────┘   └──────────┘               │  │
│  │                                                                            │  │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐                               │  │
│  │  │Summarizer│──▶│   CoT    │──▶│  Answer  │                               │  │
│  │  │   SLM    │   │Compressor│   │Stabilizer│                               │  │
│  │  └──────────┘   └──────────┘   └──────────┘                               │  │
│  │       │              │              │                                      │  │
│  │       ▼              ▼              ▼                                      │  │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐                               │  │
│  │  │ Faithful │   │ Reasoning│   │Consistency│                               │  │
│  │  │Extraction│   │ Distill  │   │ Voting   │                               │  │
│  │  └──────────┘   └──────────┘   └──────────┘                               │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                    Per-Step Tracing & Metrics                              │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │  │
│  │  │  Latency    │  │   Quality   │  │   Cost      │  │   Distributed   │  │  │
│  │  │  Tracking   │  │   Scores    │  │   Compute   │  │   Tracing       │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### SLM Pipeline Flow

```
Document Ingestion:
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Raw Docs   │───▶│  Chunker    │───▶│  Embedder   │───▶ Vector Store
│             │    │  SLM        │    │  SLM        │
└─────────────┘    └─────────────┘    └─────────────┘

Query Processing:
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Query     │───▶│  Retriever  │───▶│  Reranker   │───▶│ Summarizer  │
│             │    │  SLM        │    │  SLM        │    │    SLM      │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                │
                   ┌─────────────┐    ┌─────────────┐           │
                   │   Answer    │◀───│    CoT      │◀──────────┘
                   │ Stabilizer  │    │ Compressor  │
                   └─────────────┘    └─────────────┘
```

---

## SLM Component Specifications

### 1. Chunker SLM

Semantic-aware document chunking using a fine-tuned SLM that understands document structure and semantic boundaries.

#### Model Architecture

```python
from dataclasses import dataclass
from typing import List, Optional
import torch
import torch.nn as nn

@dataclass
class Chunk:
    content: str
    start_idx: int
    end_idx: int
    chunk_type: str  # paragraph, section, code_block, table
    semantic_score: float
    metadata: dict

class ChunkerSLM(nn.Module):
    """SLM for semantic-aware document chunking."""

    def __init__(
        self,
        base_model: str = "Qwen/Qwen2-0.5B-Instruct",
        max_chunk_tokens: int = 512,
        min_chunk_tokens: int = 50
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=2  # boundary / not-boundary
        )
        self.max_tokens = max_chunk_tokens
        self.min_tokens = min_chunk_tokens

    def forward(self, input_ids, attention_mask):
        """Predict boundary probabilities for each token position."""
        outputs = self.model(input_ids, attention_mask=attention_mask)
        boundary_probs = torch.softmax(outputs.logits, dim=-1)[:, :, 1]
        return boundary_probs

    async def chunk_document(self, document: str) -> List[Chunk]:
        # Tokenize document
        tokens = self.tokenizer(
            document,
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        # Get boundary predictions
        with torch.no_grad():
            boundary_probs = self.forward(
                tokens["input_ids"],
                tokens["attention_mask"]
            )

        # Find optimal chunk boundaries
        boundaries = self._find_boundaries(
            boundary_probs[0],
            tokens["offset_mapping"][0]
        )

        # Create chunks
        chunks = []
        for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            content = document[start:end]
            chunk_type = self._classify_chunk_type(content)

            chunks.append(Chunk(
                content=content,
                start_idx=start,
                end_idx=end,
                chunk_type=chunk_type,
                semantic_score=self._compute_semantic_score(content),
                metadata={"chunk_idx": i}
            ))

        return chunks

    def _find_boundaries(
        self,
        probs: torch.Tensor,
        offset_mapping: torch.Tensor
    ) -> List[int]:
        """Find optimal chunk boundaries using dynamic programming."""
        boundaries = [0]
        current_tokens = 0

        for i, (prob, (start, end)) in enumerate(zip(probs, offset_mapping)):
            current_tokens += 1

            # Check if we should split
            if current_tokens >= self.min_tokens and prob > 0.5:
                boundaries.append(end.item())
                current_tokens = 0
            elif current_tokens >= self.max_tokens:
                # Force split at max length
                boundaries.append(end.item())
                current_tokens = 0

        # Add final boundary
        if offset_mapping[-1][1].item() not in boundaries:
            boundaries.append(offset_mapping[-1][1].item())

        return boundaries

    def _classify_chunk_type(self, content: str) -> str:
        """Classify the type of chunk content."""
        if "```" in content or content.strip().startswith("def "):
            return "code_block"
        elif "|" in content and "-" in content:
            return "table"
        elif content.strip().startswith("#"):
            return "section"
        else:
            return "paragraph"

class AdaptiveChunker:
    """Adapts chunking strategy based on document type and query patterns."""

    def __init__(self, chunker_slm: ChunkerSLM):
        self.slm = chunker_slm
        self.strategies = {
            "technical": {"max_tokens": 400, "min_tokens": 100},
            "narrative": {"max_tokens": 600, "min_tokens": 150},
            "code": {"max_tokens": 300, "min_tokens": 50},
            "legal": {"max_tokens": 500, "min_tokens": 200}
        }

    async def chunk(
        self,
        document: str,
        doc_type: Optional[str] = None
    ) -> List[Chunk]:
        # Auto-detect document type if not provided
        if not doc_type:
            doc_type = await self._detect_document_type(document)

        # Apply strategy
        strategy = self.strategies.get(doc_type, self.strategies["technical"])
        self.slm.max_tokens = strategy["max_tokens"]
        self.slm.min_tokens = strategy["min_tokens"]

        return await self.slm.chunk_document(document)
```

### 2. Embedder SLM

Domain-adapted embedding model with contrastive fine-tuning.

#### Architecture

```python
class EmbedderSLM(nn.Module):
    """Domain-adapted embedding SLM with contrastive training."""

    def __init__(
        self,
        base_model: str = "BAAI/bge-small-en-v1.5",
        embedding_dim: int = 384,
        pooling: str = "mean"  # mean, cls, last
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.embedding_dim = embedding_dim
        self.pooling = pooling

        # Domain adaptation layers
        self.domain_adapter = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        outputs = self.encoder(input_ids, attention_mask=attention_mask)

        # Pooling
        if self.pooling == "mean":
            embeddings = self._mean_pooling(outputs.last_hidden_state, attention_mask)
        elif self.pooling == "cls":
            embeddings = outputs.last_hidden_state[:, 0]
        else:
            embeddings = outputs.last_hidden_state[:, -1]

        # Domain adaptation
        embeddings = self.domain_adapter(embeddings)

        # Normalize
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    async def embed_batch(self, texts: List[str]) -> np.ndarray:
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        with torch.no_grad():
            embeddings = self.forward(
                tokens["input_ids"],
                tokens["attention_mask"]
            )

        return embeddings.cpu().numpy()

    def _mean_pooling(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

class MultiDomainEmbedder:
    """Routes to specialized embedders based on content domain."""

    def __init__(self, embedders: Dict[str, EmbedderSLM], router: DomainRouter):
        self.embedders = embedders
        self.router = router

    async def embed(self, text: str) -> np.ndarray:
        domain = await self.router.route(text)
        embedder = self.embedders.get(domain, self.embedders["general"])
        return await embedder.embed_batch([text])
```

### 3. Retriever SLM

Intelligent query routing and multi-strategy retrieval.

#### Architecture

```python
class RetrieverSLM:
    """SLM-powered retrieval with query understanding and routing."""

    def __init__(
        self,
        query_router: QueryRouterSLM,
        retrievers: Dict[str, BaseRetriever],
        fusion_weights: Dict[str, float]
    ):
        self.query_router = query_router
        self.retrievers = retrievers
        self.fusion_weights = fusion_weights

    async def retrieve(
        self,
        query: str,
        top_k: int = 100
    ) -> List[RetrievalResult]:
        # SLM analyzes query and routes to appropriate retrievers
        routing_decision = await self.query_router.route(query)

        # Execute retrieval based on routing
        results = []
        for retriever_name in routing_decision.retrievers:
            weight = self.fusion_weights.get(retriever_name, 1.0)
            retriever = self.retrievers[retriever_name]

            retriever_results = await retriever.retrieve(
                routing_decision.transformed_query,
                top_k
            )

            # Apply routing weight
            for r in retriever_results:
                r.score *= weight

            results.extend(retriever_results)

        # Fuse and deduplicate
        return self._fuse_results(results, top_k)

class QueryRouterSLM(nn.Module):
    """SLM that routes queries to appropriate retrievers."""

    def __init__(self, base_model: str = "Qwen/Qwen2-0.5B-Instruct"):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

    async def route(self, query: str) -> RoutingDecision:
        prompt = f"""Analyze this query and determine the best retrieval strategy.

Query: {query}

Output JSON with:
- retrievers: list of retriever names to use ["dense", "sparse", "hybrid"]
- transformed_query: optimized query for retrieval
- filters: any metadata filters to apply

JSON:"""

        tokens = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = self.model.generate(
                tokens["input_ids"],
                max_new_tokens=200,
                temperature=0.1
            )

        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return self._parse_routing_decision(response)
```

### 4. Reranker SLM

Cross-attention based relevance scoring with explanation generation.

#### Architecture

```python
class RerankerSLM(nn.Module):
    """SLM reranker with cross-attention scoring."""

    def __init__(
        self,
        base_model: str = "Qwen/Qwen2-1.5B-Instruct",
        num_labels: int = 1
    ):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=num_labels
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

    async def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: int
    ) -> List[RerankResult]:
        # Prepare query-document pairs
        pairs = []
        for candidate in candidates:
            pairs.append(f"Query: {query}\nDocument: {candidate.document.content}")

        # Batch scoring
        tokens = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**tokens)
            scores = outputs.logits.squeeze(-1).sigmoid()

        # Sort by score
        scored = list(zip(candidates, scores.tolist()))
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for new_rank, (candidate, score) in enumerate(scored[:top_k]):
            results.append(RerankResult(
                document=candidate.document,
                original_rank=candidate.rank,
                new_rank=new_rank,
                relevance_score=score,
                features={"slm_score": score}
            ))

        return results

class ExplainableReranker(RerankerSLM):
    """Reranker that also generates relevance explanations."""

    async def rerank_with_explanations(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: int
    ) -> List[RerankResult]:
        results = await self.rerank(query, candidates, top_k)

        # Generate explanations for top results
        for result in results[:5]:  # Top 5 only
            explanation = await self._generate_explanation(
                query,
                result.document.content
            )
            result.features["explanation"] = explanation

        return results
```

### 5. Summarizer SLM

Faithful extractive and abstractive summarization.

#### Architecture

```python
class SummarizerSLM:
    """Faithful summarization with extraction + abstraction."""

    def __init__(
        self,
        base_model: str = "Qwen/Qwen2-1.5B-Instruct",
        max_summary_length: int = 200
    ):
        self.model = AutoModelForCausalLM.from_pretrained(base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.max_length = max_summary_length

    async def summarize(
        self,
        documents: List[Document],
        query: str,
        method: str = "hybrid"  # extractive, abstractive, hybrid
    ) -> str:
        if method == "extractive":
            return await self._extractive_summarize(documents, query)
        elif method == "abstractive":
            return await self._abstractive_summarize(documents, query)
        else:
            return await self._hybrid_summarize(documents, query)

    async def _extractive_summarize(
        self,
        documents: List[Document],
        query: str
    ) -> str:
        """Extract key sentences relevant to query."""
        prompt = f"""Extract the most relevant sentences for this query.

Query: {query}

Documents:
{self._format_documents(documents)}

Extract 3-5 sentences that directly answer the query.
Sentences:"""

        return await self._generate(prompt)

    async def _abstractive_summarize(
        self,
        documents: List[Document],
        query: str
    ) -> str:
        """Generate abstract summary."""
        prompt = f"""Summarize these documents to answer the query.

Query: {query}

Documents:
{self._format_documents(documents)}

Provide a concise summary that:
1. Directly addresses the query
2. Uses only information from the documents
3. Is factually accurate

Summary:"""

        return await self._generate(prompt)

    async def _hybrid_summarize(
        self,
        documents: List[Document],
        query: str
    ) -> str:
        """Extract then abstract."""
        # First extract
        extracted = await self._extractive_summarize(documents, query)

        # Then abstract
        prompt = f"""Rewrite these extracted sentences into a coherent summary.

Query: {query}

Extracted sentences:
{extracted}

Coherent summary:"""

        return await self._generate(prompt)
```

### 6. Chain-of-Thought (CoT) Compressor SLM

Distills complex reasoning into compact form.

#### Architecture

```python
class CoTCompressorSLM:
    """Compresses chain-of-thought reasoning while preserving logic."""

    def __init__(
        self,
        base_model: str = "Qwen/Qwen2-1.5B-Instruct",
        compression_ratio: float = 0.3
    ):
        self.model = AutoModelForCausalLM.from_pretrained(base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.compression_ratio = compression_ratio

    async def compress(
        self,
        reasoning_chain: str,
        query: str
    ) -> str:
        """Compress reasoning while preserving key logical steps."""
        prompt = f"""Compress this reasoning chain while preserving the logical flow.

Query: {query}

Full reasoning:
{reasoning_chain}

Compress to {int(self.compression_ratio * 100)}% of original length.
Keep:
1. Key logical steps
2. Critical evidence
3. Final conclusions

Compressed reasoning:"""

        tokens = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            output = self.model.generate(
                tokens["input_ids"],
                max_new_tokens=int(len(reasoning_chain.split()) * self.compression_ratio),
                temperature=0.1
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    async def distill_reasoning(
        self,
        full_reasoning: str,
        target_steps: int = 3
    ) -> List[str]:
        """Distill reasoning into key steps."""
        prompt = f"""Distill this reasoning into exactly {target_steps} key steps.

Full reasoning:
{full_reasoning}

Format each step as:
Step N: [Key insight or logical move]

Steps:"""

        response = await self._generate(prompt)
        return self._parse_steps(response)
```

### 7. Answer Stabilizer SLM

Ensures consistent and confident final answers.

#### Architecture

```python
class AnswerStabilizerSLM:
    """Stabilizes answers through consistency checking and voting."""

    def __init__(
        self,
        base_model: str = "Qwen/Qwen2-1.5B-Instruct",
        num_samples: int = 3,
        temperature: float = 0.7
    ):
        self.model = AutoModelForCausalLM.from_pretrained(base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.num_samples = num_samples
        self.temperature = temperature

    async def stabilize(
        self,
        query: str,
        context: str,
        initial_answer: str
    ) -> StabilizedAnswer:
        """Generate multiple samples and find consensus."""
        # Generate diverse answers
        samples = await self._generate_samples(query, context)

        # Check consistency
        consistency_score = self._compute_consistency(samples)

        # Vote for best answer
        if consistency_score > 0.8:
            # High consistency - use majority
            final_answer = self._majority_vote(samples)
        else:
            # Low consistency - use refinement
            final_answer = await self._refine_answer(
                query, context, samples
            )

        # Compute confidence
        confidence = self._compute_confidence(
            final_answer,
            samples,
            consistency_score
        )

        return StabilizedAnswer(
            answer=final_answer,
            confidence=confidence,
            consistency=consistency_score,
            num_samples=len(samples),
            reasoning=self._generate_reasoning(final_answer, samples)
        )

    async def _generate_samples(
        self,
        query: str,
        context: str
    ) -> List[str]:
        """Generate multiple answer samples with temperature."""
        prompt = f"""Answer this question based on the context.

Context:
{context}

Question: {query}

Answer:"""

        samples = []
        tokens = self.tokenizer(prompt, return_tensors="pt")

        for _ in range(self.num_samples):
            with torch.no_grad():
                output = self.model.generate(
                    tokens["input_ids"],
                    max_new_tokens=200,
                    temperature=self.temperature,
                    do_sample=True
                )
            sample = self.tokenizer.decode(output[0], skip_special_tokens=True)
            samples.append(sample)

        return samples

    def _compute_consistency(self, samples: List[str]) -> float:
        """Compute semantic consistency across samples."""
        # Use embedding similarity
        embeddings = self._embed_samples(samples)

        # Pairwise similarities
        similarities = []
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                sim = cosine_similarity(
                    embeddings[i].reshape(1, -1),
                    embeddings[j].reshape(1, -1)
                )[0][0]
                similarities.append(sim)

        return np.mean(similarities) if similarities else 1.0

    def _majority_vote(self, samples: List[str]) -> str:
        """Select answer with highest semantic similarity to others."""
        embeddings = self._embed_samples(samples)

        # Score each sample by average similarity to others
        scores = []
        for i in range(len(samples)):
            sim_sum = 0
            for j in range(len(samples)):
                if i != j:
                    sim_sum += cosine_similarity(
                        embeddings[i].reshape(1, -1),
                        embeddings[j].reshape(1, -1)
                    )[0][0]
            scores.append(sim_sum / (len(samples) - 1))

        # Return sample with highest average similarity
        best_idx = np.argmax(scores)
        return samples[best_idx]

    async def _refine_answer(
        self,
        query: str,
        context: str,
        samples: List[str]
    ) -> str:
        """Refine answer when samples disagree."""
        prompt = f"""Multiple answers were generated for this question. Synthesize the best answer.

Question: {query}

Context:
{context}

Generated answers:
{self._format_samples(samples)}

Synthesize the most accurate and complete answer:"""

        return await self._generate(prompt)
```

---

## Orchestration Engine

### Dynamic Computation Graph

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum
import asyncio

class NodeType(Enum):
    SLM = "slm"
    RETRIEVAL = "retrieval"
    TRANSFORM = "transform"
    BRANCH = "branch"
    MERGE = "merge"

@dataclass
class GraphNode:
    id: str
    node_type: NodeType
    component: str
    config: dict
    dependencies: List[str]
    fallback: Optional[str] = None

@dataclass
class ExecutionResult:
    node_id: str
    output: Any
    latency_ms: float
    quality_score: float
    metadata: dict

class DynamicComputationGraph:
    """Builds and executes dynamic computation graphs for RAG."""

    def __init__(self, component_registry: ComponentRegistry):
        self.registry = component_registry
        self.nodes: Dict[str, GraphNode] = {}
        self.execution_history: List[ExecutionResult] = []

    def add_node(
        self,
        node_id: str,
        node_type: NodeType,
        component: str,
        config: dict = None,
        dependencies: List[str] = None,
        fallback: str = None
    ):
        self.nodes[node_id] = GraphNode(
            id=node_id,
            node_type=node_type,
            component=component,
            config=config or {},
            dependencies=dependencies or [],
            fallback=fallback
        )

    async def execute(
        self,
        inputs: Dict[str, Any],
        trace_id: str = None
    ) -> Dict[str, Any]:
        """Execute graph with topological ordering."""
        # Build execution order
        execution_order = self._topological_sort()

        results = {**inputs}

        for node_id in execution_order:
            node = self.nodes[node_id]

            # Gather dependencies
            dep_results = {
                dep: results[dep]
                for dep in node.dependencies
                if dep in results
            }

            # Execute node
            result = await self._execute_node(node, dep_results, trace_id)
            results[node_id] = result.output
            self.execution_history.append(result)

        return results

    async def _execute_node(
        self,
        node: GraphNode,
        inputs: Dict[str, Any],
        trace_id: str
    ) -> ExecutionResult:
        """Execute a single node with tracing."""
        start_time = time.time()

        try:
            # Get component
            component = self.registry.get(node.component)

            # Execute
            if node.node_type == NodeType.SLM:
                output = await component.process(**inputs, **node.config)
            elif node.node_type == NodeType.BRANCH:
                output = await self._execute_branch(component, inputs, node.config)
            else:
                output = await component(**inputs)

            latency = (time.time() - start_time) * 1000
            quality = await self._compute_quality(node, output)

            return ExecutionResult(
                node_id=node.id,
                output=output,
                latency_ms=latency,
                quality_score=quality,
                metadata={
                    "trace_id": trace_id,
                    "component": node.component,
                    "input_keys": list(inputs.keys())
                }
            )

        except Exception as e:
            # Try fallback
            if node.fallback:
                return await self._execute_fallback(node, inputs, trace_id, e)
            raise

    def _topological_sort(self) -> List[str]:
        """Topological sort of nodes."""
        visited = set()
        order = []

        def visit(node_id):
            if node_id in visited:
                return
            visited.add(node_id)
            for dep in self.nodes[node_id].dependencies:
                visit(dep)
            order.append(node_id)

        for node_id in self.nodes:
            visit(node_id)

        return order

class GraphBuilder:
    """Fluent API for building computation graphs."""

    def __init__(self, registry: ComponentRegistry):
        self.graph = DynamicComputationGraph(registry)

    def slm(
        self,
        node_id: str,
        component: str,
        dependencies: List[str] = None,
        **config
    ) -> 'GraphBuilder':
        self.graph.add_node(
            node_id,
            NodeType.SLM,
            component,
            config,
            dependencies
        )
        return self

    def branch(
        self,
        node_id: str,
        condition: str,
        branches: Dict[str, str],
        dependencies: List[str] = None
    ) -> 'GraphBuilder':
        self.graph.add_node(
            node_id,
            NodeType.BRANCH,
            "branch_executor",
            {"condition": condition, "branches": branches},
            dependencies
        )
        return self

    def build(self) -> DynamicComputationGraph:
        return self.graph

# Example usage
def build_rag_graph(registry: ComponentRegistry) -> DynamicComputationGraph:
    """Build standard RAG computation graph."""
    return (
        GraphBuilder(registry)
        .slm("chunk", "chunker_slm", dependencies=["document"])
        .slm("embed", "embedder_slm", dependencies=["chunk"])
        .slm("retrieve", "retriever_slm", dependencies=["query", "embed"])
        .slm("rerank", "reranker_slm", dependencies=["query", "retrieve"])
        .slm("summarize", "summarizer_slm", dependencies=["query", "rerank"])
        .slm("compress_cot", "cot_compressor_slm", dependencies=["summarize"])
        .slm("stabilize", "answer_stabilizer_slm", dependencies=["query", "compress_cot"])
        .build()
    )
```

### Model Selection & Routing

```python
class ModelSelector:
    """Intelligent model selection based on task and context."""

    def __init__(
        self,
        models: Dict[str, ModelInfo],
        selection_strategy: str = "quality_latency_balance"
    ):
        self.models = models
        self.strategy = selection_strategy
        self.performance_tracker = PerformanceTracker()

    async def select(
        self,
        task: str,
        context: Dict[str, Any],
        constraints: SelectionConstraints = None
    ) -> str:
        """Select best model for task given constraints."""
        candidates = self._filter_candidates(task, constraints)

        if self.strategy == "quality_latency_balance":
            return self._balance_selection(candidates, context)
        elif self.strategy == "lowest_latency":
            return self._latency_selection(candidates)
        elif self.strategy == "highest_quality":
            return self._quality_selection(candidates)
        else:
            return self._learned_selection(candidates, context)

    def _balance_selection(
        self,
        candidates: List[str],
        context: Dict[str, Any]
    ) -> str:
        """Balance quality and latency."""
        scores = {}
        for model_name in candidates:
            info = self.models[model_name]
            perf = self.performance_tracker.get_stats(model_name)

            # Normalize metrics
            quality_score = perf.avg_quality / 10  # 0-1
            latency_score = 1 - (perf.avg_latency / 1000)  # Lower is better

            # Weight based on context
            if context.get("latency_sensitive"):
                scores[model_name] = 0.3 * quality_score + 0.7 * latency_score
            else:
                scores[model_name] = 0.7 * quality_score + 0.3 * latency_score

        return max(scores, key=scores.get)

class FallbackChain:
    """Manages model fallback chains."""

    def __init__(self, chains: Dict[str, List[str]]):
        self.chains = chains  # task -> [primary, fallback1, fallback2, ...]

    async def execute_with_fallback(
        self,
        task: str,
        executor: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute with automatic fallback on failure."""
        chain = self.chains.get(task, [])

        for model_name in chain:
            try:
                return await executor(model_name, *args, **kwargs)
            except Exception as e:
                logger.warning(f"Model {model_name} failed: {e}, trying fallback")
                continue

        raise RuntimeError(f"All models in fallback chain failed for {task}")
```

### Guardrail Engine

```python
class GuardrailEngine:
    """Applies guardrails at each pipeline stage."""

    def __init__(self, rules: List[GuardrailRule]):
        self.rules = rules
        self.violations = []

    async def check(
        self,
        stage: str,
        input_data: Any,
        output_data: Any
    ) -> GuardrailResult:
        """Check guardrails for a pipeline stage."""
        violations = []

        for rule in self.rules:
            if rule.applies_to(stage):
                result = await rule.evaluate(input_data, output_data)
                if not result.passed:
                    violations.append(result)

        if violations:
            # Log violations
            self.violations.extend(violations)

            # Determine action
            if any(v.severity == "critical" for v in violations):
                return GuardrailResult(
                    passed=False,
                    action="block",
                    violations=violations
                )
            elif any(v.severity == "warning" for v in violations):
                return GuardrailResult(
                    passed=True,
                    action="warn",
                    violations=violations
                )

        return GuardrailResult(passed=True, action="pass", violations=[])

class GuardrailRule(ABC):
    """Base class for guardrail rules."""

    @abstractmethod
    def applies_to(self, stage: str) -> bool:
        pass

    @abstractmethod
    async def evaluate(self, input_data: Any, output_data: Any) -> RuleResult:
        pass

class RelevanceGuardrail(GuardrailRule):
    """Ensures retrieved content is relevant to query."""

    def __init__(self, min_relevance: float = 0.3):
        self.min_relevance = min_relevance

    def applies_to(self, stage: str) -> bool:
        return stage in ["retrieval", "reranking"]

    async def evaluate(self, input_data, output_data) -> RuleResult:
        # Check if any results meet minimum relevance
        if hasattr(output_data, '__iter__'):
            max_relevance = max(r.score for r in output_data)
            if max_relevance < self.min_relevance:
                return RuleResult(
                    passed=False,
                    severity="warning",
                    message=f"Max relevance {max_relevance:.2f} below threshold {self.min_relevance}"
                )
        return RuleResult(passed=True)

class HallucinationGuardrail(GuardrailRule):
    """Checks for hallucinations in generated content."""

    def applies_to(self, stage: str) -> bool:
        return stage in ["summarization", "answer_generation"]

    async def evaluate(self, input_data, output_data) -> RuleResult:
        # Use NLI to check if output is supported by input
        # ...
        pass
```

---

## Dataset Design Per Model

### Training Data Requirements

| SLM Component | Dataset Type | Size | Key Features |
|---------------|--------------|------|--------------|
| Chunker | Boundary annotations | 50K docs | Multi-domain, human annotated boundaries |
| Embedder | Contrastive pairs | 1M pairs | Query-passage pairs, hard negatives |
| Retriever | Query routing | 100K queries | Multi-retriever labels, performance |
| Reranker | Relevance grades | 500K pairs | Fine-grained relevance (0-3 scale) |
| Summarizer | Faithful summaries | 100K | Source-summary alignment |
| CoT Compressor | Reasoning chains | 50K | Full/compressed reasoning pairs |
| Answer Stabilizer | Consistency data | 100K | Multi-sample answers, consensus labels |

### Chunker Training Data

```python
@dataclass
class ChunkerTrainingExample:
    document: str
    boundary_positions: List[int]  # Token positions
    chunk_types: List[str]
    quality_score: float

def create_chunker_dataset():
    """Create training data for chunker SLM."""
    examples = []

    for doc in documents:
        # Get human-annotated boundaries
        boundaries = get_human_annotations(doc)

        # Create labels for each token
        tokens = tokenize(doc)
        labels = [1 if i in boundaries else 0 for i in range(len(tokens))]

        examples.append(ChunkerTrainingExample(
            document=doc,
            boundary_positions=boundaries,
            chunk_types=get_chunk_types(doc, boundaries),
            quality_score=get_annotation_quality(doc)
        ))

    return examples
```

### Reranker Training Data

```python
@dataclass
class RerankerTrainingExample:
    query: str
    positive_doc: str
    negative_docs: List[str]
    relevance_grade: int  # 0-3

def create_reranker_dataset():
    """Create training data with hard negatives."""
    examples = []

    for query, relevant_docs in query_relevance_pairs:
        # Get BM25 negatives (hard negatives)
        bm25_negatives = bm25_search(query, exclude=relevant_docs)

        # Get random negatives (easy negatives)
        random_negatives = sample_random_docs(exclude=relevant_docs)

        for pos_doc, grade in relevant_docs:
            examples.append(RerankerTrainingExample(
                query=query,
                positive_doc=pos_doc,
                negative_docs=bm25_negatives[:5] + random_negatives[:5],
                relevance_grade=grade
            ))

    return examples
```

---

## Enterprise Features

### Per-Step Tracing

```python
class PipelineTracer:
    """Distributed tracing for pipeline execution."""

    def __init__(self, exporter: TraceExporter):
        self.exporter = exporter

    @contextmanager
    def trace_step(
        self,
        step_name: str,
        trace_id: str,
        parent_id: str = None
    ):
        span_id = generate_span_id()
        start_time = time.time()

        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_id=parent_id,
            name=step_name,
            start_time=start_time,
            attributes={}
        )

        try:
            yield span
        finally:
            span.end_time = time.time()
            span.duration_ms = (span.end_time - span.start_time) * 1000
            self.exporter.export(span)

    def add_attribute(self, span: Span, key: str, value: Any):
        span.attributes[key] = value
```

### Quality Metrics Dashboard

```python
class QualityMetricsCollector:
    """Collects and aggregates quality metrics."""

    def __init__(self):
        self.metrics = defaultdict(list)

    def record(
        self,
        component: str,
        metric_name: str,
        value: float,
        timestamp: float = None
    ):
        self.metrics[f"{component}.{metric_name}"].append({
            "value": value,
            "timestamp": timestamp or time.time()
        })

    def get_aggregates(self, component: str = None) -> Dict[str, Dict]:
        """Get aggregated metrics."""
        results = {}

        for key, values in self.metrics.items():
            if component and not key.startswith(component):
                continue

            vals = [v["value"] for v in values]
            results[key] = {
                "mean": np.mean(vals),
                "std": np.std(vals),
                "min": np.min(vals),
                "max": np.max(vals),
                "p50": np.percentile(vals, 50),
                "p95": np.percentile(vals, 95),
                "count": len(vals)
            }

        return results
```

### Canary Rollout

```python
class CanaryRolloutManager:
    """Manages canary rollouts for model updates."""

    def __init__(self, storage, metrics_collector):
        self.storage = storage
        self.metrics = metrics_collector

    async def create_canary(
        self,
        component: str,
        new_model: str,
        traffic_percentage: float = 5.0,
        success_criteria: Dict[str, float] = None
    ) -> str:
        canary_id = generate_id()

        canary = {
            "id": canary_id,
            "component": component,
            "new_model": new_model,
            "traffic_percentage": traffic_percentage,
            "success_criteria": success_criteria or {
                "latency_p95_increase": 10,  # Max 10% increase
                "quality_decrease": 5,  # Max 5% decrease
                "error_rate_increase": 1  # Max 1% increase
            },
            "status": "active",
            "start_time": datetime.utcnow()
        }

        await self.storage.save(canary_id, canary)
        return canary_id

    async def evaluate_canary(self, canary_id: str) -> CanaryEvaluation:
        """Evaluate canary performance against baseline."""
        canary = await self.storage.get(canary_id)

        # Get metrics for canary and baseline
        canary_metrics = self.metrics.get_aggregates(f"canary.{canary_id}")
        baseline_metrics = self.metrics.get_aggregates(f"baseline.{canary['component']}")

        # Check criteria
        checks = []
        for criterion, threshold in canary["success_criteria"].items():
            passed = self._check_criterion(
                criterion,
                threshold,
                canary_metrics,
                baseline_metrics
            )
            checks.append({"criterion": criterion, "passed": passed})

        return CanaryEvaluation(
            canary_id=canary_id,
            passed=all(c["passed"] for c in checks),
            checks=checks,
            recommendation="promote" if all(c["passed"] for c in checks) else "rollback"
        )
```

---

## Implementation Phases

### Phase 1: Foundation (Weeks 1-4)

**Objectives:**
- Core orchestration engine
- Base SLM integration framework
- Computation graph execution

**Deliverables:**
- [ ] Component registry system
- [ ] Dynamic computation graph builder
- [ ] Basic graph execution
- [ ] Tracing infrastructure
- [ ] Unit tests

### Phase 2: Core SLMs (Weeks 5-10)

**Objectives:**
- Implement all 7 SLM components
- Basic fine-tuning pipelines
- Integration with orchestrator

**Week 5-6: Chunker + Embedder**
- [ ] Chunker SLM implementation
- [ ] Boundary prediction model
- [ ] Embedder SLM implementation
- [ ] Domain adaptation layers

**Week 7-8: Retriever + Reranker**
- [ ] Retriever SLM with routing
- [ ] Reranker SLM implementation
- [ ] Multi-stage reranking

**Week 9-10: Summarizer + CoT + Stabilizer**
- [ ] Summarizer SLM
- [ ] CoT Compressor
- [ ] Answer Stabilizer
- [ ] End-to-end pipeline tests

### Phase 3: Model Selection & Quality (Weeks 11-14)

**Objectives:**
- Intelligent model selection
- Quality tracking
- Guardrail system

**Deliverables:**
- [ ] Model selector implementation
- [ ] Fallback chain system
- [ ] Guardrail engine
- [ ] Quality metrics collector
- [ ] Per-step tracing

### Phase 4: Enterprise Features (Weeks 15-18)

**Objectives:**
- Production hardening
- Canary rollout
- Advanced monitoring

**Deliverables:**
- [ ] Canary rollout manager
- [ ] A/B testing for models
- [ ] Comprehensive dashboards
- [ ] Load testing
- [ ] Documentation

### Phase 5: Optimization (Weeks 19-22)

**Objectives:**
- Performance optimization
- Model distillation
- Cost optimization

**Deliverables:**
- [ ] Batch inference optimization
- [ ] Model quantization
- [ ] Caching strategies
- [ ] Cost tracking

---

## Stretch Goals

### RL on Pipeline Rewards

```python
class PipelineRewardModel:
    """Learn to optimize pipeline decisions using RL."""

    def __init__(
        self,
        policy_network: nn.Module,
        value_network: nn.Module
    ):
        self.policy = policy_network
        self.value = value_network

    def compute_reward(
        self,
        execution_trace: List[ExecutionResult],
        final_quality: float
    ) -> float:
        """Compute reward for pipeline execution."""
        # Quality reward
        quality_reward = final_quality

        # Latency penalty
        total_latency = sum(r.latency_ms for r in execution_trace)
        latency_penalty = -0.001 * total_latency

        # Cost penalty
        total_cost = sum(self._compute_cost(r) for r in execution_trace)
        cost_penalty = -0.01 * total_cost

        return quality_reward + latency_penalty + cost_penalty

    async def update_policy(
        self,
        trajectories: List[PipelineTrajectory]
    ):
        """Update policy using PPO."""
        for trajectory in trajectories:
            # Compute advantages
            advantages = self._compute_advantages(trajectory)

            # Policy update
            policy_loss = self._compute_policy_loss(trajectory, advantages)
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # Value update
            value_loss = self._compute_value_loss(trajectory)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
```

### Policy Engine for Retrieval Depth

```python
class RetrievalDepthPolicy:
    """Learns optimal retrieval depth based on query complexity."""

    def __init__(self, policy_model):
        self.policy = policy_model

    async def decide_depth(
        self,
        query: str,
        initial_results: List[RetrievalResult]
    ) -> int:
        """Decide how many retrieval stages to run."""
        # Extract features
        features = self._extract_features(query, initial_results)

        # Policy prediction
        depth_probs = self.policy.predict(features)

        # Sample or argmax
        if self.training:
            depth = torch.multinomial(depth_probs, 1).item()
        else:
            depth = torch.argmax(depth_probs).item()

        return depth + 1  # 1-indexed

    def _extract_features(
        self,
        query: str,
        results: List[RetrievalResult]
    ) -> torch.Tensor:
        """Extract features for depth decision."""
        return torch.tensor([
            len(query.split()),  # Query length
            np.mean([r.score for r in results]),  # Avg relevance
            np.std([r.score for r in results]),  # Relevance variance
            self._compute_coverage(query, results),  # Topic coverage
        ])
```

### Automatic Pipeline Optimization

```python
class AutoPipelineOptimizer:
    """Automatically optimizes pipeline configuration."""

    def __init__(
        self,
        search_space: Dict[str, Any],
        objective: str = "quality_latency_pareto"
    ):
        self.search_space = search_space
        self.objective = objective
        self.trials = []

    async def optimize(
        self,
        n_trials: int = 100,
        eval_dataset: List[dict] = None
    ) -> Dict[str, Any]:
        """Run optimization trials."""
        for i in range(n_trials):
            # Sample configuration
            config = self._sample_config()

            # Evaluate
            metrics = await self._evaluate_config(config, eval_dataset)

            # Record trial
            self.trials.append({
                "config": config,
                "metrics": metrics,
                "score": self._compute_score(metrics)
            })

        # Return best configuration
        best_trial = max(self.trials, key=lambda t: t["score"])
        return best_trial["config"]

    def _sample_config(self) -> Dict[str, Any]:
        """Sample configuration from search space."""
        config = {}
        for param, space in self.search_space.items():
            if space["type"] == "categorical":
                config[param] = random.choice(space["values"])
            elif space["type"] == "float":
                config[param] = random.uniform(space["min"], space["max"])
            elif space["type"] == "int":
                config[param] = random.randint(space["min"], space["max"])
        return config
```

---

## File Structure

```
27-micro-model-orchestrated-rag/
├── src/
│   ├── __init__.py
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   ├── graph.py
│   │   ├── executor.py
│   │   ├── selector.py
│   │   └── guardrails.py
│   ├── slm/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── retriever.py
│   │   ├── reranker.py
│   │   ├── summarizer.py
│   │   ├── cot_compressor.py
│   │   └── answer_stabilizer.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── datasets.py
│   │   ├── trainers.py
│   │   └── evaluation.py
│   ├── enterprise/
│   │   ├── __init__.py
│   │   ├── tracing.py
│   │   ├── metrics.py
│   │   └── canary.py
│   └── api/
│       ├── __init__.py
│       └── main.py
├── models/
│   └── checkpoints/
├── data/
│   └── training/
├── config/
├── tests/
├── docs/
├── BLUEPRINT.md
├── PROGRESS.md
└── SESSION_CONTEXT.md
```

---

## Success Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| End-to-End Latency P95 | < 2s | With all SLMs |
| Answer Quality (human eval) | > 4.2/5 | Compared to single-LLM |
| Faithfulness | > 0.95 | No hallucinations |
| Cost per Query | < $0.01 | SLM cost efficiency |
| Model Selection Accuracy | > 90% | Optimal model chosen |
| Guardrail Catch Rate | > 95% | Policy violations caught |

---

## References

- [Small Language Models Survey](https://arxiv.org/abs/2402.02315)
- [Specialized Model Routing](https://arxiv.org/abs/2308.00951)
- [Chain-of-Thought Compression](https://arxiv.org/abs/2305.08291)
- [Self-Consistency Improves CoT](https://arxiv.org/abs/2203.11171)
- [Dynamic Computation Graphs](https://pytorch.org/docs/stable/notes/autograd.html)
