"""Vector store implementations for similarity search."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, Optional, Any
from pathlib import Path
import json
import pickle
import numpy as np

from .schemas import SearchResult


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    async def add(
        self,
        ids: list[str],
        vectors: list[np.ndarray],
        metadata: list[dict] = None,
    ):
        """Add vectors to store."""
        pass

    @abstractmethod
    async def search(
        self,
        query: np.ndarray,
        k: int = 5,
        filter: dict = None,
    ) -> list[SearchResult]:
        """Search for similar vectors."""
        pass

    @abstractmethod
    async def delete(self, ids: list[str]):
        """Delete vectors by ID."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Return number of vectors in store."""
        pass

    async def update(
        self,
        ids: list[str],
        vectors: list[np.ndarray],
        metadata: list[dict] = None,
    ):
        """Update existing vectors."""
        await self.delete(ids)
        await self.add(ids, vectors, metadata)

    async def clear(self):
        """Clear all vectors from store."""
        pass

    async def add_batch(
        self,
        ids: list[str],
        vectors: list[np.ndarray],
        metadata: list[dict] = None,
    ):
        """Add vectors in batch."""
        await self.add(ids, vectors, metadata)

    async def search_batch(
        self,
        queries: list[np.ndarray],
        k: int = 5,
    ) -> list[list[SearchResult]]:
        """Search for multiple queries."""
        results = []
        for query in queries:
            results.append(await self.search(query, k))
        return results

    async def save(self, path: Path):
        """Save store to disk."""
        pass

    async def load(self, path: Path):
        """Load store from disk."""
        pass

    @classmethod
    def from_config(cls, config: "VectorStoreConfig") -> "VectorStore":
        """Create store from config."""
        if config.type == "in_memory":
            return InMemoryVectorStore(dimension=config.dimension)
        elif config.type == "chroma":
            return ChromaVectorStore(
                collection_name=config.collection_name,
                persist_directory=config.persist_directory,
            )
        elif config.type == "qdrant":
            return QdrantVectorStore(
                collection_name=config.collection_name,
                host=config.host,
                port=config.port,
                dimension=config.dimension,
            )
        elif config.type == "pinecone":
            return PineconeVectorStore(
                index_name=config.index_name,
                api_key=config.api_key,
                environment=config.environment,
            )
        else:
            raise ValueError(f"Unknown vector store type: {config.type}")


@dataclass
class VectorStoreConfig:
    """Configuration for vector stores."""

    type: str = "in_memory"
    dimension: int = 384
    collection_name: str = "documents"
    persist_directory: str = "./data"
    host: str = "localhost"
    port: int = 6333
    index_name: str = "documents"
    api_key: str = ""
    environment: str = ""


class InMemoryVectorStore(VectorStore):
    """In-memory vector store using NumPy."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.vectors: dict[str, np.ndarray] = {}
        self.metadata: dict[str, dict] = {}

    async def add(
        self,
        ids: list[str],
        vectors: list[np.ndarray],
        metadata: list[dict] = None,
    ):
        """Add vectors to store."""
        if metadata is None:
            metadata = [{}] * len(ids)

        for i, (id_, vec) in enumerate(zip(ids, vectors)):
            self.vectors[id_] = np.array(vec)
            self.metadata[id_] = metadata[i]

    async def search(
        self,
        query: np.ndarray,
        k: int = 5,
        filter: dict = None,
    ) -> list[SearchResult]:
        """Search for similar vectors."""
        if not self.vectors:
            return []

        query = np.array(query).flatten()
        scores = []

        for id_, vec in self.vectors.items():
            # Apply filter
            if filter:
                meta = self.metadata.get(id_, {})
                if not self._matches_filter(meta, filter):
                    continue

            similarity = self._cosine_similarity(query, vec)
            scores.append((id_, similarity))

        # Sort by similarity
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for id_, score in scores[:k]:
            results.append(SearchResult(
                id=id_,
                score=float(score),
                metadata=self.metadata.get(id_, {}),
            ))

        return results

    async def delete(self, ids: list[str]):
        """Delete vectors by ID."""
        for id_ in ids:
            self.vectors.pop(id_, None)
            self.metadata.pop(id_, None)

    def size(self) -> int:
        """Return number of vectors."""
        return len(self.vectors)

    async def clear(self):
        """Clear all vectors."""
        self.vectors.clear()
        self.metadata.clear()

    async def save(self, path: Path):
        """Save store to disk."""
        path = Path(path)
        with open(path, 'wb') as f:
            pickle.dump({
                'vectors': self.vectors,
                'metadata': self.metadata,
                'dimension': self.dimension,
            }, f)

    async def load(self, path: Path):
        """Load store from disk."""
        path = Path(path)
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.vectors = data['vectors']
            self.metadata = data['metadata']
            self.dimension = data.get('dimension', self.dimension)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _matches_filter(self, metadata: dict, filter: dict) -> bool:
        """Check if metadata matches filter."""
        for key, value in filter.items():
            if key not in metadata:
                return False
            if isinstance(value, dict):
                for op, op_value in value.items():
                    if op == "$in" and metadata[key] not in op_value:
                        return False
                    elif op == "$eq" and metadata[key] != op_value:
                        return False
                    elif op == "$ne" and metadata[key] == op_value:
                        return False
            elif metadata[key] != value:
                return False
        return True


class QdrantVectorStore(VectorStore):
    """Vector store using Qdrant."""

    def __init__(
        self,
        collection_name: str = "documents",
        host: str = "localhost",
        port: int = 6333,
        dimension: int = 384,
    ):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            self._qdrant_models = __import__('qdrant_client.models', fromlist=['models'])
        except ImportError:
            raise ImportError("qdrant-client required. Install with: pip install qdrant-client")

        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.dimension = dimension

        # Create collection
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )

    async def add(
        self,
        ids: list[str],
        vectors: list[np.ndarray],
        metadata: list[dict] = None,
    ):
        """Add vectors to Qdrant."""
        from qdrant_client.models import PointStruct

        if metadata is None:
            metadata = [{}] * len(ids)

        points = [
            PointStruct(id=id_, vector=vec.tolist(), payload=meta)
            for id_, vec, meta in zip(ids, vectors, metadata)
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    async def search(
        self,
        query: np.ndarray,
        k: int = 5,
        filter: dict = None,
    ) -> list[SearchResult]:
        """Search in Qdrant."""
        query_filter = None
        if filter:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            conditions = [
                FieldCondition(key=key, match=MatchValue(value=value))
                for key, value in filter.items()
            ]
            query_filter = Filter(must=conditions)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query.tolist(),
            limit=k,
            query_filter=query_filter,
        )

        return [
            SearchResult(
                id=str(r.id),
                score=r.score,
                metadata=r.payload or {},
            )
            for r in results
        ]

    async def delete(self, ids: list[str]):
        """Delete from Qdrant."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=ids,
        )

    def size(self) -> int:
        """Return collection size."""
        info = self.client.get_collection(self.collection_name)
        return info.points_count


class PineconeVectorStore(VectorStore):
    """Vector store using Pinecone."""

    def __init__(
        self,
        index_name: str = "documents",
        api_key: str = None,
        environment: str = "us-east-1",
    ):
        try:
            import pinecone
            self._pinecone = pinecone
        except ImportError:
            raise ImportError("pinecone-client required. Install with: pip install pinecone-client")

        pinecone.init(api_key=api_key, environment=environment)
        self.index = pinecone.Index(index_name)
        self.index_name = index_name

    async def add(
        self,
        ids: list[str],
        vectors: list[np.ndarray],
        metadata: list[dict] = None,
    ):
        """Add vectors to Pinecone."""
        if metadata is None:
            metadata = [{}] * len(ids)

        upserts = [
            (id_, vec.tolist(), meta)
            for id_, vec, meta in zip(ids, vectors, metadata)
        ]

        self.index.upsert(upserts)

    async def search(
        self,
        query: np.ndarray,
        k: int = 5,
        filter: dict = None,
    ) -> list[SearchResult]:
        """Search in Pinecone."""
        results = self.index.query(
            vector=query.tolist(),
            top_k=k,
            filter=filter,
            include_metadata=True,
        )

        return [
            SearchResult(
                id=match["id"],
                score=match["score"],
                metadata=match.get("metadata", {}),
            )
            for match in results.get("matches", [])
        ]

    async def delete(self, ids: list[str]):
        """Delete from Pinecone."""
        self.index.delete(ids=ids)

    def size(self) -> int:
        """Return index size."""
        stats = self.index.describe_index_stats()
        return stats.get("total_vector_count", 0)


class ChromaVectorStore(VectorStore):
    """Vector store using ChromaDB."""

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = "./chroma_db",
    ):
        try:
            import chromadb
            self._chromadb = chromadb
        except ImportError:
            raise ImportError("chromadb required. Install with: pip install chromadb")

        # Create persistent client
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.persist_directory = persist_directory

    async def add(
        self,
        ids: list[str],
        vectors: list[np.ndarray],
        metadata: list[dict] = None,
    ):
        """Add vectors to collection."""
        if metadata is None:
            metadata = [{}] * len(ids)

        embeddings = [v.tolist() if isinstance(v, np.ndarray) else v for v in vectors]
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadata,
        )

    async def search(
        self,
        query: np.ndarray,
        k: int = 5,
        filter: dict = None,
    ) -> list[SearchResult]:
        """Search for similar vectors."""
        query_vec = query.tolist() if isinstance(query, np.ndarray) else query
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=k,
            where=filter,
            include=["metadatas", "distances"],
        )

        search_results = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                search_results.append(SearchResult(
                    id=results["ids"][0][i],
                    score=1 - results["distances"][0][i],  # Convert distance to score
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                ))

        return search_results

    async def delete(self, ids: list[str]):
        """Delete vectors by ID."""
        self.collection.delete(ids=ids)

    def size(self) -> int:
        """Get number of vectors in collection."""
        return self.collection.count()


class SimpleVectorStore:
    """Simple in-memory vector store using NumPy."""

    def __init__(self):
        self.embeddings = None
        self.ids = []
        self.documents = []
        self.metadatas = []

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
        k: int = 5,
        filter: dict = None,
    ) -> list[SearchResult]:
        """Search for similar documents."""
        if self.embeddings is None or len(self.ids) == 0:
            return []

        # Ensure query is 1D
        if query_embedding.ndim == 2:
            query_embedding = query_embedding[0]

        # Normalize for proper cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        emb_norms = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10)

        # Compute cosine similarity
        similarities = np.dot(emb_norms, query_norm)

        # Apply filter if provided
        mask = None
        if filter:
            mask = self._apply_filter(filter)
            # Use -inf for filtered items so they sort to the bottom
            similarities = np.where(mask > 0, similarities, -np.inf)

        # Get top k
        top_indices = np.argsort(similarities)[-k:][::-1]

        results = []
        for idx in top_indices:
            # Skip filtered out items (those with -inf similarity)
            if mask is not None and mask[idx] == 0:
                continue
            results.append(SearchResult(
                id=self.ids[idx],
                content=self.documents[idx],
                metadata=self.metadatas[idx],
                score=float(similarities[idx]),
            ))

        return results

    def delete(self, ids: list[str]):
        """Delete documents by ID."""
        indices_to_keep = [
            i for i, doc_id in enumerate(self.ids)
            if doc_id not in ids
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

    def _apply_filter(self, filter: dict) -> np.ndarray:
        """Apply metadata filter."""
        mask = np.ones(len(self.ids))

        for key, value in filter.items():
            for i, metadata in enumerate(self.metadatas):
                if key not in metadata:
                    mask[i] = 0
                elif isinstance(value, dict):
                    # Handle operators like $in, $eq, $gte
                    for op, op_value in value.items():
                        if op == "$in":
                            if metadata[key] not in op_value:
                                mask[i] = 0
                        elif op == "$eq":
                            if metadata[key] != op_value:
                                mask[i] = 0
                        elif op == "$gte":
                            if metadata[key] < op_value:
                                mask[i] = 0
                        elif op == "$lte":
                            if metadata[key] > op_value:
                                mask[i] = 0
                else:
                    if metadata[key] != value:
                        mask[i] = 0

        return mask

    def count(self) -> int:
        """Get number of documents."""
        return len(self.ids)

    def save(self, path: str):
        """Save store to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        np.save(path / "embeddings.npy", self.embeddings)
        with open(path / "metadata.json", "w") as f:
            json.dump({
                "ids": self.ids,
                "documents": self.documents,
                "metadatas": self.metadatas,
            }, f)

    def load(self, path: str):
        """Load store from disk."""
        path = Path(path)

        self.embeddings = np.load(path / "embeddings.npy")
        with open(path / "metadata.json") as f:
            data = json.load(f)
            self.ids = data["ids"]
            self.documents = data["documents"]
            self.metadatas = data["metadatas"]


def get_vector_store(
    store_type: str = "simple",
    **kwargs,
) -> VectorStore:
    """Factory function to get vector store.

    Args:
        store_type: Type of vector store
            - "simple": In-memory NumPy-based store
            - "chroma": ChromaDB persistent store
        **kwargs: Additional arguments for store

    Returns:
        Vector store instance
    """
    if store_type == "simple":
        return SimpleVectorStore()
    elif store_type == "chroma":
        return ChromaVectorStore(**kwargs)
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")
