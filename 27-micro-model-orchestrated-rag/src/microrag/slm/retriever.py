"""Retriever SLM for intelligent query routing and retrieval."""

from typing import Optional, List, Dict
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss

from .base import BaseSLM, GenerativeModelMixin, EmbeddingModelMixin, logger
from ..schemas import RetrievalResult, RoutingDecision, Document


class RetrieverSLM(BaseSLM, GenerativeModelMixin, EmbeddingModelMixin):
    """SLM-powered retrieval with query understanding and routing."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-0.5B-Instruct",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        default_top_k: int = 100,
        use_faiss: bool = True
    ):
        super().__init__(model_name)
        self.embedding_model_name = embedding_model
        self.default_top_k = default_top_k
        self.use_faiss = use_faiss
        self._documents: List[Document] = []
        self._embeddings: Optional[np.ndarray] = None
        self._index = None
        self.embedding_model = None

    def _load_model(self):
        """Load both generative model for query understanding and embedding model."""
        # Load generative model for query understanding/routing
        self._load_generative_model(self.model_name, load_in_8bit=True)

        # Load embedding model for dense retrieval
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.embedding_model.to(self.device)

    def index_documents(self, documents: List[Document], embeddings: np.ndarray):
        """Index documents with embeddings using FAISS for efficient retrieval.

        Args:
            documents: List of documents
            embeddings: Document embeddings
        """
        self._documents = documents
        self._embeddings = embeddings

        if self.use_faiss and len(documents) > 0:
            # Create FAISS index for efficient similarity search
            embedding_dim = embeddings.shape[1]

            # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
            self._index = faiss.IndexFlatIP(embedding_dim)

            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)

            # Add embeddings to index
            self._index.add(embeddings.astype(np.float32))

            logger.info(f"Indexed {len(documents)} documents with FAISS")

    async def process(
        self,
        query: str = None,
        top_k: int = None,
        **kwargs
    ) -> List[RetrievalResult]:
        """Retrieve relevant documents with intelligent query routing.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of retrieval results
        """
        if not query:
            return []

        if not self._loaded:
            self.load()

        top_k = top_k or self.default_top_k

        # Route query to appropriate retrieval strategy
        routing = await self._route_query(query)

        # Execute retrieval based on routing decision
        results = await self._retrieve(
            routing.transformed_query,
            top_k,
            routing.filters,
            routing.retrievers
        )

        return results

    async def _route_query(self, query: str) -> RoutingDecision:
        """Route query using language model for understanding.

        Args:
            query: Original query

        Returns:
            Routing decision
        """
        # Use LLM to understand query intent and determine retrieval strategy
        prompt = f"""Analyze this search query and determine the best retrieval strategy:

Query: {query}

Determine:
1. Is this a factual question (needs dense retrieval)?
2. Does it contain specific keywords/entities (needs sparse retrieval)?
3. Does it need time-based filtering?
4. Is query expansion needed?

Output format:
Strategy: [dense/sparse/hybrid]
Needs expansion: [yes/no]
Key entities: [list any specific entities]
"""

        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256
                ).to(self.device)

                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_new_tokens=50,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

                response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip().lower()

            # Parse response to determine retrieval strategy
            retrievers = ["dense"]  # Default to dense

            if "hybrid" in response or ("dense" in response and "sparse" in response):
                retrievers = ["dense", "sparse"]
            elif "sparse" in response:
                retrievers = ["sparse"]

            # Determine if query expansion is needed
            needs_expansion = "yes" in response

            # Transform query if needed
            transformed_query = await self._expand_query(query) if needs_expansion else query

            return RoutingDecision(
                retrievers=retrievers,
                transformed_query=transformed_query,
                filters=self._extract_filters(query),
                confidence=0.85
            )

        except Exception as e:
            logger.warning(f"Failed to route query with LLM: {str(e)}")
            # Fallback to heuristic routing
            return self._heuristic_routing(query)

    def _heuristic_routing(self, query: str) -> RoutingDecision:
        """Fallback heuristic routing when LLM fails."""
        retrievers = ["dense"]

        # Add sparse for queries with specific entities or technical terms
        if any(word[0].isupper() for word in query.split() if word):
            retrievers.append("sparse")

        # Check for question words that might benefit from expansion
        question_words = ["what", "how", "why", "when", "where", "who"]
        needs_expansion = any(query.lower().startswith(word) for word in question_words)

        return RoutingDecision(
            retrievers=retrievers,
            transformed_query=query,
            filters={},
            confidence=0.7
        )

    async def _expand_query(self, query: str) -> str:
        """Expand query using language model."""
        prompt = f"""Expand this query with related terms and synonyms to improve retrieval:

Original query: {query}

Expanded query (include original terms plus related terms):"""

        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128
                ).to(self.device)

                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_new_tokens=30,
                    temperature=0.5,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

                expanded = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()

            # Combine original and expanded
            if expanded and len(expanded) > 5:
                return f"{query} {expanded}"

        except Exception as e:
            logger.warning(f"Failed to expand query: {str(e)}")

        return query

    async def _retrieve(
        self,
        query: str,
        top_k: int,
        filters: Dict,
        retrievers: List[str]
    ) -> List[RetrievalResult]:
        """Execute retrieval using selected strategies.

        Args:
            query: Query string (possibly expanded)
            top_k: Number of results
            filters: Metadata filters
            retrievers: List of retriever types to use

        Returns:
            Retrieval results
        """
        if not self._documents:
            return []

        results = []

        if "dense" in retrievers:
            dense_results = await self._dense_retrieval(query, top_k * 2)  # Get more for fusion
            results.extend(dense_results)

        if "sparse" in retrievers:
            sparse_results = await self._sparse_retrieval(query, top_k)
            results.extend(sparse_results)

        # Deduplicate and merge scores if using hybrid
        if len(retrievers) > 1:
            results = self._reciprocal_rank_fusion(results)

        # Apply filters
        if filters:
            results = self._apply_filters(results, filters)

        # Sort by score and limit to top_k
        results.sort(key=lambda x: x.score, reverse=True)
        results = results[:top_k]

        # Update ranks
        for i, result in enumerate(results):
            result.rank = i

        return results

    async def _dense_retrieval(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Perform dense retrieval using embeddings."""
        if self._index is None or not self.embedding_model:
            return []

        # Encode query
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Search with FAISS
        scores, indices = self._index.search(
            query_embedding.astype(np.float32),
            min(top_k, len(self._documents))
        )

        # Create results
        results = []
        for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < len(self._documents):
                results.append(RetrievalResult(
                    document=self._documents[idx],
                    score=float(score),
                    retriever_type="dense",
                    rank=i
                ))

        return results

    async def _sparse_retrieval(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Perform keyword-based sparse retrieval."""
        if not self._documents:
            return []

        # Simple BM25-style scoring
        query_terms = set(query.lower().split())
        results = []

        for i, doc in enumerate(self._documents):
            doc_terms = set(doc.content.lower().split())
            overlap = len(query_terms & doc_terms)

            if overlap > 0:
                # Simple TF-IDF style score
                score = overlap / (len(query_terms) + 1)

                # Boost for exact phrase match
                if query.lower() in doc.content.lower():
                    score *= 2

                results.append(RetrievalResult(
                    document=doc,
                    score=score,
                    retriever_type="sparse",
                    rank=len(results)
                ))

        # Sort and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def _reciprocal_rank_fusion(
        self,
        results: List[RetrievalResult],
        k: int = 60
    ) -> List[RetrievalResult]:
        """Merge results from multiple retrievers using reciprocal rank fusion."""
        # Group by document ID
        doc_scores = {}

        for result in results:
            doc_id = result.document.id
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    'document': result.document,
                    'score': 0,
                    'sources': []
                }

            # RRF score: 1 / (k + rank)
            rrf_score = 1.0 / (k + result.rank + 1)
            doc_scores[doc_id]['score'] += rrf_score
            doc_scores[doc_id]['sources'].append(result.retriever_type)

        # Create merged results
        merged = []
        for doc_id, info in doc_scores.items():
            merged.append(RetrievalResult(
                document=info['document'],
                score=info['score'],
                retriever_type="+".join(set(info['sources'])),
                rank=0  # Will be updated
            ))

        return merged

    def _apply_filters(
        self,
        results: List[RetrievalResult],
        filters: Dict
    ) -> List[RetrievalResult]:
        """Apply metadata filters to results."""
        filtered = []

        for result in results:
            passes = True
            for key, value in filters.items():
                if key in result.document.metadata:
                    if result.document.metadata[key] != value:
                        passes = False
                        break

            if passes:
                filtered.append(result)

        return filtered

    def _extract_filters(self, query: str) -> Dict:
        """Extract metadata filters from query."""
        filters = {}

        # Simple date extraction (could be enhanced)
        import re
        year_match = re.search(r'\b(20\d{2})\b', query)
        if year_match:
            filters['year'] = int(year_match.group(1))

        # Could add more filter extraction logic here

        return filters


class MockRetrieverSLM(BaseSLM):
    """Mock retriever for testing."""

    def __init__(self):
        super().__init__("mock")
        self._loaded = True

    async def process(
        self,
        query: str = None,
        top_k: int = 10,
        **kwargs
    ) -> List[RetrievalResult]:
        if not query:
            return []

        # Return mock results
        results = []
        for i in range(min(top_k, 5)):
            doc = Document(
                id=f"doc_{i}",
                content=f"Mock document {i} relevant to: {query}",
                metadata={"source": "mock"}
            )
            results.append(RetrievalResult(
                document=doc,
                score=1.0 - (i * 0.1),
                retriever_type="mock",
                rank=i
            ))

        return results


class QueryRouterSLM(BaseSLM, GenerativeModelMixin):
    """SLM that routes queries to appropriate retrievers."""

    def __init__(self, model_name: str = "Qwen/Qwen2-0.5B-Instruct"):
        super().__init__(model_name)

    def _load_model(self):
        """Load the generative model for query understanding."""
        self._load_generative_model(self.model_name, load_in_8bit=True)

    async def process(self, query: str = None, **kwargs) -> RoutingDecision:
        """Route query using language model.

        Args:
            query: Query string

        Returns:
            Routing decision
        """
        if not query:
            return RoutingDecision(
                retrievers=["dense"],
                transformed_query="",
                confidence=0.0
            )

        if not self._loaded:
            self.load()

        # Analyze query with LLM
        prompt = f"""Analyze this query and determine the best retrieval approach:

Query: {query}

Consider:
1. Would semantic search (dense retrieval) work well?
2. Are there specific keywords/entities that need exact matching (sparse retrieval)?
3. Should we use both approaches (hybrid)?

Answer with one word: dense, sparse, or hybrid
Answer:"""

        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=200
                ).to(self.device)

                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )

                response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip().lower()

            # Parse response
            if "hybrid" in response:
                retrievers = ["dense", "sparse"]
            elif "sparse" in response:
                retrievers = ["sparse"]
            else:
                retrievers = ["dense"]

            return RoutingDecision(
                retrievers=retrievers,
                transformed_query=query,
                filters={},
                confidence=0.9
            )

        except Exception as e:
            logger.warning(f"Failed to route query with LLM: {str(e)}")
            # Fallback
            return RoutingDecision(
                retrievers=["dense"],
                transformed_query=query,
                confidence=0.5
            )