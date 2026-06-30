"""Tests for vector store module."""

import asyncio
import numpy as np
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import List, Dict, Any

from ragbaseline.vectorstore import (
    VectorStore,
    InMemoryVectorStore,
    ChromaVectorStore,
    QdrantVectorStore,
    PineconeVectorStore,
    SearchResult,
    VectorStoreConfig,
)


class TestVectorStore:
    """Test base vector store interface."""

    def test_abstract_methods(self):
        """Test that abstract methods are enforced."""
        with pytest.raises(TypeError):
            VectorStore()


class TestInMemoryVectorStore:
    """Test in-memory vector store implementation."""

    @pytest.fixture
    def store(self):
        """Create in-memory vector store."""
        return InMemoryVectorStore(dimension=3)

    @pytest.fixture
    def sample_vectors(self):
        """Create sample vectors for testing."""
        return [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.5, 0.5, 0.0]),
        ]

    @pytest.mark.asyncio
    async def test_add_vectors(self, store, sample_vectors):
        """Test adding vectors to store."""
        ids = ["vec1", "vec2", "vec3", "vec4"]
        metadata = [
            {"type": "doc", "source": "file1"},
            {"type": "doc", "source": "file2"},
            {"type": "query", "source": "user"},
            {"type": "doc", "source": "file1"},
        ]

        await store.add(ids, sample_vectors, metadata)

        # Verify vectors were added
        assert store.size() == 4
        assert all(id in store.vectors for id in ids)

    @pytest.mark.asyncio
    async def test_search_similar(self, store, sample_vectors):
        """Test similarity search."""
        # Add vectors
        ids = ["vec1", "vec2", "vec3", "vec4"]
        await store.add(ids, sample_vectors, [{}] * 4)

        # Search for similar vectors to [1, 0, 0]
        query = np.array([1.0, 0.0, 0.0])
        results = await store.search(query, k=2)

        assert len(results) == 2
        assert results[0].id == "vec1"  # Exact match
        assert results[0].score == pytest.approx(1.0)
        assert results[1].id == "vec4"  # Next closest

    @pytest.mark.asyncio
    async def test_search_with_filter(self, store, sample_vectors):
        """Test filtered similarity search."""
        ids = ["vec1", "vec2", "vec3", "vec4"]
        metadata = [
            {"type": "doc"},
            {"type": "doc"},
            {"type": "query"},
            {"type": "doc"},
        ]

        await store.add(ids, sample_vectors, metadata)

        # Search only among docs
        query = np.array([0.0, 0.0, 1.0])
        results = await store.search(
            query,
            k=2,
            filter={"type": "doc"}
        )

        # Should not include vec3 (type=query)
        assert all(r.id != "vec3" for r in results)
        assert all(r.metadata["type"] == "doc" for r in results)

    @pytest.mark.asyncio
    async def test_delete_vectors(self, store, sample_vectors):
        """Test deleting vectors from store."""
        ids = ["vec1", "vec2", "vec3", "vec4"]
        await store.add(ids, sample_vectors, [{}] * 4)

        # Delete specific vectors
        await store.delete(["vec2", "vec3"])

        assert store.size() == 2
        assert "vec1" in store.vectors
        assert "vec4" in store.vectors
        assert "vec2" not in store.vectors
        assert "vec3" not in store.vectors

    @pytest.mark.asyncio
    async def test_update_vectors(self, store):
        """Test updating existing vectors."""
        # Add initial vector
        await store.add(
            ["vec1"],
            [np.array([1.0, 0.0, 0.0])],
            [{"version": 1}]
        )

        # Update vector
        await store.update(
            ["vec1"],
            [np.array([0.0, 1.0, 0.0])],
            [{"version": 2}]
        )

        # Verify update
        results = await store.search(np.array([0.0, 1.0, 0.0]), k=1)
        assert results[0].id == "vec1"
        assert results[0].metadata["version"] == 2

    @pytest.mark.asyncio
    async def test_clear_store(self, store, sample_vectors):
        """Test clearing all vectors from store."""
        ids = ["vec1", "vec2", "vec3", "vec4"]
        await store.add(ids, sample_vectors, [{}] * 4)

        assert store.size() == 4

        await store.clear()

        assert store.size() == 0
        assert len(store.vectors) == 0

    @pytest.mark.asyncio
    async def test_batch_operations(self, store):
        """Test batch add and search operations."""
        # Batch add
        batch_size = 100
        ids = [f"vec{i}" for i in range(batch_size)]
        vectors = [np.random.randn(3) for _ in range(batch_size)]
        metadata = [{"index": i} for i in range(batch_size)]

        await store.add_batch(ids, vectors, metadata)

        assert store.size() == batch_size

        # Batch search
        queries = [np.random.randn(3) for _ in range(10)]
        results = await store.search_batch(queries, k=5)

        assert len(results) == 10
        assert all(len(r) <= 5 for r in results)

    def test_cosine_similarity(self, store):
        """Test cosine similarity calculation."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        vec3 = np.array([0.0, 1.0, 0.0])
        vec4 = np.array([-1.0, 0.0, 0.0])

        # Same vector
        assert store._cosine_similarity(vec1, vec2) == pytest.approx(1.0)

        # Orthogonal vectors
        assert store._cosine_similarity(vec1, vec3) == pytest.approx(0.0)

        # Opposite vectors
        assert store._cosine_similarity(vec1, vec4) == pytest.approx(-1.0)

    @pytest.mark.asyncio
    async def test_persistence(self, store, sample_vectors, tmp_path):
        """Test saving and loading vector store."""
        ids = ["vec1", "vec2", "vec3", "vec4"]
        metadata = [{"index": i} for i in range(4)]
        await store.add(ids, sample_vectors, metadata)

        # Save to disk
        save_path = tmp_path / "vectors.pkl"
        await store.save(save_path)

        # Load into new store
        new_store = InMemoryVectorStore(dimension=3)
        await new_store.load(save_path)

        assert new_store.size() == 4
        assert all(id in new_store.vectors for id in ids)


class TestChromaVectorStore:
    """Test Chroma vector store implementation."""

    @pytest.fixture
    def mock_chroma_client(self):
        """Mock Chroma client."""
        mock = MagicMock()
        client = MagicMock()
        collection = MagicMock()
        client.get_or_create_collection.return_value = collection
        mock.Client.return_value = client
        with patch.dict("sys.modules", {"chromadb": mock}):
            yield collection

    @pytest.mark.asyncio
    async def test_initialization(self, mock_chroma_client):
        """Test Chroma store initialization."""
        store = ChromaVectorStore(
            collection_name="test_collection",
            persist_directory="/tmp/chroma"
        )

        assert store.collection is not None
        assert store.collection == mock_chroma_client

    @pytest.mark.asyncio
    async def test_add_documents(self, mock_chroma_client):
        """Test adding documents to Chroma."""
        store = ChromaVectorStore(collection_name="test")

        ids = ["doc1", "doc2"]
        vectors = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        metadata = [{"source": "file1"}, {"source": "file2"}]

        await store.add(ids, vectors, metadata)

        mock_chroma_client.add.assert_called_once()
        call_args = mock_chroma_client.add.call_args[1]
        assert call_args["ids"] == ids
        assert len(call_args["embeddings"]) == 2
        assert call_args["metadatas"] == metadata

    @pytest.mark.asyncio
    async def test_search(self, mock_chroma_client):
        """Test searching in Chroma store."""
        mock_chroma_client.query.return_value = {
            "ids": [["doc1", "doc2"]],
            "distances": [[0.1, 0.3]],
            "metadatas": [[{"source": "file1"}, {"source": "file2"}]],
        }

        store = ChromaVectorStore(collection_name="test")
        query = np.array([1, 2, 3])
        results = await store.search(query, k=2)

        assert len(results) == 2
        assert results[0].id == "doc1"
        assert results[0].score == pytest.approx(0.9)  # 1 - distance
        assert results[0].metadata["source"] == "file1"

    @pytest.mark.asyncio
    async def test_delete(self, mock_chroma_client):
        """Test deleting from Chroma store."""
        store = ChromaVectorStore(collection_name="test")
        await store.delete(["doc1", "doc2"])

        mock_chroma_client.delete.assert_called_once_with(
            ids=["doc1", "doc2"]
        )


class TestQdrantVectorStore:
    """Test Qdrant vector store implementation."""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Mock Qdrant client."""
        mock_qdrant_module = MagicMock()
        mock_models = MagicMock()
        mock_models.Distance = MagicMock()
        mock_models.VectorParams = MagicMock()
        mock_qdrant_module.models = mock_models
        client = MagicMock()
        mock_qdrant_module.QdrantClient.return_value = client
        with patch.dict("sys.modules", {
            "qdrant_client": mock_qdrant_module,
            "qdrant_client.models": mock_models,
        }):
            yield client

    @pytest.mark.asyncio
    async def test_initialization(self, mock_qdrant_client):
        """Test Qdrant store initialization."""
        store = QdrantVectorStore(
            collection_name="test_collection",
            host="localhost",
            port=6333,
            dimension=384
        )

        mock_qdrant_client.recreate_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_points(self, mock_qdrant_client):
        """Test adding points to Qdrant."""
        store = QdrantVectorStore(
            collection_name="test",
            dimension=3
        )

        ids = ["pt1", "pt2"]
        vectors = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        metadata = [{"tag": "A"}, {"tag": "B"}]

        await store.add(ids, vectors, metadata)

        mock_qdrant_client.upsert.assert_called_once()
        call_args = mock_qdrant_client.upsert.call_args[1]
        assert call_args["collection_name"] == "test"
        assert len(call_args["points"]) == 2

    @pytest.mark.asyncio
    async def test_search_with_filter(self, mock_qdrant_client):
        """Test searching with filters in Qdrant."""
        # Create mock ScoredPoint objects
        scored_point_1 = MagicMock()
        scored_point_1.id = "pt1"
        scored_point_1.score = 0.95
        scored_point_1.payload = {"tag": "A"}

        scored_point_2 = MagicMock()
        scored_point_2.id = "pt2"
        scored_point_2.score = 0.85
        scored_point_2.payload = {"tag": "A"}

        mock_qdrant_client.search.return_value = [scored_point_1, scored_point_2]

        store = QdrantVectorStore(collection_name="test", dimension=3)
        query = np.array([1, 2, 3])
        results = await store.search(
            query,
            k=2,
            filter={"tag": "A"}
        )

        assert len(results) == 2
        assert all(r.metadata["tag"] == "A" for r in results)


class TestPineconeVectorStore:
    """Test Pinecone vector store implementation."""

    @pytest.fixture
    def mock_pinecone(self):
        """Mock Pinecone client."""
        mock = MagicMock()
        index = MagicMock()
        mock.Index.return_value = index
        with patch.dict("sys.modules", {"pinecone": mock}):
            yield mock, index

    @pytest.mark.asyncio
    async def test_initialization(self, mock_pinecone):
        """Test Pinecone store initialization."""
        mock_client, mock_index = mock_pinecone

        store = PineconeVectorStore(
            index_name="test_index",
            api_key="test-key",
            environment="us-east-1"
        )

        mock_client.init.assert_called_once_with(
            api_key="test-key",
            environment="us-east-1"
        )

    @pytest.mark.asyncio
    async def test_upsert_vectors(self, mock_pinecone):
        """Test upserting vectors to Pinecone."""
        _, mock_index = mock_pinecone

        store = PineconeVectorStore(
            index_name="test",
            api_key="key"
        )
        store.index = mock_index

        ids = ["vec1", "vec2"]
        vectors = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        metadata = [{"type": "doc"}, {"type": "query"}]

        await store.add(ids, vectors, metadata)

        mock_index.upsert.assert_called_once()
        upserted = mock_index.upsert.call_args[0][0]
        assert len(upserted) == 2
        assert upserted[0][0] == "vec1"

    @pytest.mark.asyncio
    async def test_query(self, mock_pinecone):
        """Test querying Pinecone index."""
        _, mock_index = mock_pinecone

        mock_index.query.return_value = {
            "matches": [
                {"id": "vec1", "score": 0.95, "metadata": {"type": "doc"}},
                {"id": "vec2", "score": 0.85, "metadata": {"type": "doc"}},
            ]
        }

        store = PineconeVectorStore(index_name="test", api_key="key")
        store.index = mock_index

        query = np.array([1, 2, 3])
        results = await store.search(query, k=2)

        assert len(results) == 2
        assert results[0].id == "vec1"
        assert results[0].score == 0.95


class TestVectorStoreFactory:
    """Test vector store factory pattern."""

    def test_create_in_memory_store(self):
        """Test creating in-memory store."""
        config = VectorStoreConfig(
            type="in_memory",
            dimension=384
        )
        store = VectorStore.from_config(config)

        assert isinstance(store, InMemoryVectorStore)
        assert store.dimension == 384

    def test_create_chroma_store(self):
        """Test creating Chroma store."""
        mock_chroma = MagicMock()
        mock_client = MagicMock()
        mock_chroma.Client.return_value = mock_client
        with patch.dict("sys.modules", {"chromadb": mock_chroma}):
            config = VectorStoreConfig(
                type="chroma",
                collection_name="test",
                persist_directory="/tmp"
            )
            store = VectorStore.from_config(config)

            assert isinstance(store, ChromaVectorStore)

    def test_invalid_store_type(self):
        """Test error handling for invalid store type."""
        config = VectorStoreConfig(
            type="invalid_store",
            dimension=384
        )

        with pytest.raises(ValueError, match="Unknown vector store type"):
            VectorStore.from_config(config)


class TestSearchResult:
    """Test SearchResult data class."""

    def test_search_result_creation(self):
        """Test creating search result."""
        result = SearchResult(
            id="doc1",
            score=0.95,
            content="Test content",
            metadata={"source": "file.txt"}
        )

        assert result.id == "doc1"
        assert result.score == 0.95
        assert result.content == "Test content"
        assert result.metadata["source"] == "file.txt"

    def test_search_result_comparison(self):
        """Test comparing search results by score."""
        result1 = SearchResult("doc1", 0.95)
        result2 = SearchResult("doc2", 0.85)
        result3 = SearchResult("doc3", 0.95)

        assert result1 > result2
        assert result1 >= result3
        assert result2 < result1

    def test_search_result_serialization(self):
        """Test serializing search result."""
        result = SearchResult(
            id="doc1",
            score=0.95,
            content="Content",
            metadata={"key": "value"}
        )

        # To dict
        result_dict = result.to_dict()
        assert result_dict["id"] == "doc1"
        assert result_dict["score"] == 0.95

        # From dict
        restored = SearchResult.from_dict(result_dict)
        assert restored.id == result.id
        assert restored.score == result.score


if __name__ == "__main__":
    pytest.main([__file__, "-v"])