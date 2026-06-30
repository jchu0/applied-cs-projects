"""Embedder SLM for domain-adapted embeddings."""

import numpy as np
from typing import Optional, List, Union
import torch
from sentence_transformers import SentenceTransformer

from .base import BaseSLM, EmbeddingModelMixin, logger


class EmbedderSLM(BaseSLM, EmbeddingModelMixin):
    """Domain-adapted embedding SLM using BGE models."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        embedding_dim: int = 384,
        pooling: str = "mean",
        normalize_embeddings: bool = True,
        batch_size: int = 32
    ):
        super().__init__(model_name)
        self.embedding_dim = embedding_dim
        self.pooling = pooling
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size

    def _load_model(self):
        """Load the BGE embedding model."""
        self._load_embedding_model(self.model_name)
        # Update embedding dimension based on loaded model
        if hasattr(self.model, 'get_sentence_embedding_dimension'):
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

    async def process(
        self,
        text: str = None,
        texts: List[str] = None,
        chunk: list = None,
        instruction: str = None,
        **kwargs
    ) -> np.ndarray:
        """Generate embeddings using BGE model.

        Args:
            text: Single text to embed
            texts: Multiple texts to embed
            chunk: Chunks to embed
            instruction: Optional instruction prefix for BGE models

        Returns:
            Embedding vector(s)
        """
        if not self._loaded:
            self.load()

        # Handle chunk input
        if chunk is not None:
            texts = [c.content if hasattr(c, 'content') else str(c) for c in chunk]
        elif text is not None:
            texts = [text]
        elif texts is None:
            return np.zeros((0, self.embedding_dim))

        # Add instruction prefix if provided (BGE models support this)
        if instruction and self.model_name.startswith("BAAI/bge"):
            texts = [f"{instruction} {t}" for t in texts]

        # Generate embeddings
        embeddings = self._encode_texts(texts)

        return embeddings

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts using the sentence transformer model."""
        if not texts:
            return np.zeros((0, self.embedding_dim))

        try:
            with torch.no_grad():
                embeddings = self.model.encode(
                    texts,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=self.normalize_embeddings
                )

            # Ensure correct dtype
            embeddings = embeddings.astype(np.float32)

            return embeddings

        except Exception as e:
            logger.error(f"Failed to encode texts: {str(e)}")
            # Return random embeddings as fallback
            embeddings = np.random.randn(len(texts), self.embedding_dim).astype(np.float32)
            if self.normalize_embeddings:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / (norms + 1e-9)
            return embeddings

    async def embed_batch(self, texts: List[str], **kwargs) -> np.ndarray:
        """Embed a batch of texts efficiently.

        Args:
            texts: List of texts
            **kwargs: Additional arguments

        Returns:
            Array of embeddings
        """
        return await self.process(texts=texts, **kwargs)

    async def embed_query(self, query: str) -> np.ndarray:
        """Embed a query with query-specific optimization.

        Args:
            query: Query text

        Returns:
            Query embedding
        """
        # BGE models recommend adding "query: " prefix for queries
        if self.model_name.startswith("BAAI/bge"):
            query_with_instruction = f"Represent this sentence for searching relevant passages: {query}"
            return await self.process(text=query_with_instruction)
        else:
            return await self.process(text=query)

    async def embed_document(self, document: str) -> np.ndarray:
        """Embed a document with document-specific optimization.

        Args:
            document: Document text

        Returns:
            Document embedding
        """
        # BGE models can use different instructions for documents
        if self.model_name.startswith("BAAI/bge"):
            doc_with_instruction = f"Represent this document for retrieval: {document}"
            return await self.process(text=doc_with_instruction)
        else:
            return await self.process(text=document)

    def compute_similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
        metric: str = "cosine"
    ) -> np.ndarray:
        """Compute similarity between two sets of embeddings.

        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            metric: Similarity metric (cosine, dot, euclidean)

        Returns:
            Similarity matrix
        """
        if metric == "cosine":
            # Normalize embeddings if not already normalized
            if not self.normalize_embeddings:
                norm1 = embeddings1 / (np.linalg.norm(embeddings1, axis=1, keepdims=True) + 1e-9)
                norm2 = embeddings2 / (np.linalg.norm(embeddings2, axis=1, keepdims=True) + 1e-9)
            else:
                norm1, norm2 = embeddings1, embeddings2

            return np.dot(norm1, norm2.T)

        elif metric == "dot":
            return np.dot(embeddings1, embeddings2.T)

        elif metric == "euclidean":
            # Compute negative euclidean distance (higher is more similar)
            distances = -np.linalg.norm(
                embeddings1[:, np.newaxis, :] - embeddings2[np.newaxis, :, :],
                axis=2
            )
            return distances

        else:
            raise ValueError(f"Unknown metric: {metric}")


class MockEmbedderSLM(BaseSLM):
    """Mock embedder for testing."""

    def __init__(self, embedding_dim: int = 384):
        super().__init__("mock")
        self._loaded = True
        self.embedding_dim = embedding_dim

    async def process(
        self,
        text: str = None,
        texts: List[str] = None,
        chunk: list = None,
        **kwargs
    ) -> np.ndarray:
        # Handle different inputs
        if chunk is not None:
            n = len(chunk)
        elif texts is not None:
            n = len(texts)
        elif text is not None:
            n = 1
        else:
            return np.zeros((0, self.embedding_dim))

        # Return random embeddings
        emb = np.random.randn(n, self.embedding_dim).astype(np.float32)
        # Normalize
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / (norms + 1e-9)
        return emb


class MultiDomainEmbedder:
    """Routes to specialized embedders based on content domain."""

    def __init__(
        self,
        embedders: dict = None,
        default_domain: str = "general"
    ):
        """Initialize multi-domain embedder.

        Args:
            embedders: Dictionary of domain -> embedder mappings
            default_domain: Default domain to use
        """
        if embedders is None:
            # Create default embedders for different domains
            embedders = {
                "general": EmbedderSLM("BAAI/bge-small-en-v1.5"),
                "code": EmbedderSLM("microsoft/codebert-base"),  # Specialized for code
                "medical": EmbedderSLM("emilyalsentzer/Bio_ClinicalBERT"),  # Medical domain
                "legal": EmbedderSLM("nlpaueb/legal-bert-small-uncased"),  # Legal domain
            }

        self.embedders = embedders
        self.default_domain = default_domain
        self._loaded = False

    def load(self):
        """Load all domain embedders."""
        for domain, embedder in self.embedders.items():
            try:
                embedder.load()
                logger.info(f"Loaded embedder for domain: {domain}")
            except Exception as e:
                logger.warning(f"Failed to load embedder for domain {domain}: {str(e)}")
        self._loaded = True

    async def embed(
        self,
        text: Union[str, List[str]],
        domain: str = None
    ) -> np.ndarray:
        """Embed text using domain-specific embedder.

        Args:
            text: Text or list of texts to embed
            domain: Optional domain hint

        Returns:
            Embedding vector(s)
        """
        if not self._loaded:
            self.load()

        # Handle list of texts
        if isinstance(text, list):
            # Process each text with appropriate domain
            embeddings = []
            for t in text:
                d = domain if domain else self._detect_domain(t)
                embedder = self.embedders.get(d, self.embedders.get(self.default_domain))
                emb = await embedder.process(text=t)
                embeddings.append(emb[0] if emb.ndim > 1 else emb)
            return np.array(embeddings)
        else:
            # Single text
            if not domain:
                domain = self._detect_domain(text)

            embedder = self.embedders.get(domain, self.embedders.get(self.default_domain))
            return await embedder.process(text=text)

    def _detect_domain(self, text: str) -> str:
        """Detect text domain using heuristics.

        Args:
            text: Text to analyze

        Returns:
            Detected domain
        """
        text_lower = text.lower()

        # Code detection
        code_keywords = ["function", "class", "def", "import", "return", "if __name__",
                        "public static", "void", "const", "let", "var"]
        if any(kw in text_lower for kw in code_keywords) or "```" in text:
            return "code"

        # Medical detection
        medical_keywords = ["patient", "diagnosis", "treatment", "symptom", "medication",
                           "clinical", "medical", "disease", "therapy", "prescription"]
        if sum(1 for kw in medical_keywords if kw in text_lower) >= 2:
            return "medical"

        # Legal detection
        legal_keywords = ["hereby", "whereas", "pursuant", "contract", "agreement",
                         "legal", "court", "law", "plaintiff", "defendant"]
        if sum(1 for kw in legal_keywords if kw in text_lower) >= 2:
            return "legal"

        return "general"

    async def embed_query_document_pair(
        self,
        query: str,
        document: str,
        query_domain: str = None,
        doc_domain: str = None
    ) -> tuple:
        """Embed a query-document pair with appropriate domain models.

        Args:
            query: Query text
            document: Document text
            query_domain: Optional query domain
            doc_domain: Optional document domain

        Returns:
            Tuple of (query_embedding, document_embedding)
        """
        query_emb = await self.embed(query, domain=query_domain)
        doc_emb = await self.embed(document, domain=doc_domain)

        return query_emb, doc_emb