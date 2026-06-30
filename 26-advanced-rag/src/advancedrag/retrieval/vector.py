"""Vector retrieval implementation."""

from abc import ABC, abstractmethod
from typing import Optional, Protocol
import numpy as np

from ..schemas import Document, RetrievalResult


class EmbeddingModel(Protocol):
    """Protocol for embedding models."""
    dimension: int

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        ...


class VectorStore(ABC):
    """Base class for vector stores."""

    @abstractmethod
    def add(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        documents: list[str],
        metadatas: list[dict],
    ):
        """Add documents to store."""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        k: int,
        filter_dict: dict = None,
    ) -> list[tuple[str, str, dict, float]]:
        """Search for similar documents."""
        pass

    @abstractmethod
    def delete(self, ids: list[str]):
        """Delete documents by ID."""
        pass


class SimpleVectorStore(VectorStore):
    """Simple in-memory vector store using NumPy."""

    def __init__(self):
        self.embeddings: Optional[np.ndarray] = None
        self.ids: list[str] = []
        self.documents: list[str] = []
        self.metadatas: list[dict] = []

    def add(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        documents: list[str],
        metadatas: list[dict],
    ):
        """Add documents to store."""
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

        self.ids.extend(ids)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filter_dict: dict = None,
    ) -> list[tuple[str, str, dict, float]]:
        """Search for similar documents."""
        if self.embeddings is None or len(self.ids) == 0:
            return []

        # Ensure query is 1D
        if query_embedding.ndim == 2:
            query_embedding = query_embedding[0]

        # Normalize for cosine similarity
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        emb_norms = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)

        # Compute similarities
        similarities = np.dot(emb_norms, query_norm)

        # Apply filter if provided
        if filter_dict:
            mask = np.ones(len(self.ids), dtype=bool)
            for i, metadata in enumerate(self.metadatas):
                if not self._match_filter(metadata, filter_dict):
                    mask[i] = False
            similarities = similarities * mask

        # Get top k
        top_indices = np.argsort(similarities)[-k:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append((
                    self.ids[idx],
                    self.documents[idx],
                    self.metadatas[idx],
                    float(similarities[idx]),
                ))

        return results

    def delete(self, ids: list[str]):
        """Delete documents by ID."""
        ids_set = set(ids)
        indices_to_keep = [
            i for i, doc_id in enumerate(self.ids)
            if doc_id not in ids_set
        ]

        if not indices_to_keep:
            self.embeddings = None
            self.ids = []
            self.documents = []
            self.metadatas = []
            return

        self.embeddings = self.embeddings[indices_to_keep]
        self.ids = [self.ids[i] for i in indices_to_keep]
        self.documents = [self.documents[i] for i in indices_to_keep]
        self.metadatas = [self.metadatas[i] for i in indices_to_keep]

    def _match_filter(self, metadata: dict, filter_dict: dict) -> bool:
        """Check if metadata matches filter."""
        for key, value in filter_dict.items():
            if key not in metadata:
                return False
            if isinstance(value, dict):
                for op, op_value in value.items():
                    if op == "$in" and metadata[key] not in op_value:
                        return False
                    elif op == "$eq" and metadata[key] != op_value:
                        return False
                    elif op == "$gte" and metadata[key] < op_value:
                        return False
                    elif op == "$lte" and metadata[key] > op_value:
                        return False
            elif metadata[key] != value:
                return False
        return True

    @property
    def count(self) -> int:
        """Get number of documents."""
        return len(self.ids)


class VectorRetriever:
    """Vector retrieval using embedding similarity."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def add_documents(self, documents: list[Document]):
        """Add documents to the retriever."""
        if not documents:
            return

        texts = [doc.content for doc in documents]
        embeddings = self.embedding_model.encode(texts)

        self.vector_store.add(
            ids=[doc.id for doc in documents],
            embeddings=embeddings,
            documents=texts,
            metadatas=[doc.metadata for doc in documents],
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: dict = None,
    ) -> list[RetrievalResult]:
        """Search for similar documents."""
        # Embed query
        query_embedding = self.embedding_model.encode([query])

        # Search vector store
        results = self.vector_store.search(query_embedding, top_k, filter_dict)

        # Convert to RetrievalResults
        retrieval_results = []
        for rank, (doc_id, content, metadata, score) in enumerate(results):
            retrieval_results.append(RetrievalResult(
                document=Document(id=doc_id, content=content, metadata=metadata),
                score=score,
                retriever_type="vector",
                rank=rank,
            ))

        return retrieval_results

    def delete(self, doc_ids: list[str]):
        """Delete documents by ID."""
        self.vector_store.delete(doc_ids)


class MockEmbedding:
    """Mock embedding for testing."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._cache = {}

    def encode(self, texts: list[str]) -> np.ndarray:
        """Generate deterministic mock embeddings."""
        embeddings = []

        for text in texts:
            if text not in self._cache:
                np.random.seed(hash(text) % (2**32))
                embedding = np.random.randn(self.dimension)
                embedding = embedding / np.linalg.norm(embedding)
                self._cache[text] = embedding

            embeddings.append(self._cache[text])

        return np.array(embeddings)


class SentenceTransformerEmbedding:
    """Embedding using sentence-transformers."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers required. "
                "Install with: pip install sentence-transformers"
            )

        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
