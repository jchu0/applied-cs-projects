"""Tests for embeddings module."""

import asyncio
import numpy as np
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from ragbaseline.embeddings import (
    EmbeddingProvider,
    OpenAIEmbedder,
    LocalEmbedder,
    CachedEmbedder,
    BatchedEmbedder,
)


class TestEmbeddingProvider:
    """Test base embedding provider interface."""

    def test_abstract_methods(self):
        """Test that abstract methods are enforced."""
        with pytest.raises(TypeError):
            # Cannot instantiate abstract class
            EmbeddingProvider()


class TestOpenAIEmbedder:
    """Test OpenAI embedding provider."""

    @pytest.fixture
    def mock_openai(self):
        """Mock OpenAI client."""
        with patch.dict("sys.modules", {"openai": MagicMock()}) as modules:
            import sys
            mock = sys.modules["openai"]
            mock_client = MagicMock()
            mock.AsyncOpenAI.return_value = mock_client
            yield mock_client

    @pytest.mark.asyncio
    async def test_embed_single(self, mock_openai):
        """Test embedding single text."""
        # Setup
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_openai.embeddings.create = AsyncMock(return_value=mock_response)

        embedder = OpenAIEmbedder(api_key="test-key")

        # Execute
        result = await embedder.embed("test text")

        # Assert
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        np.testing.assert_array_equal(result, [0.1, 0.2, 0.3])

    @pytest.mark.asyncio
    async def test_embed_batch(self, mock_openai):
        """Test batch embedding."""
        # Setup
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2]),
            MagicMock(embedding=[0.3, 0.4]),
        ]
        mock_openai.embeddings.create = AsyncMock(return_value=mock_response)

        embedder = OpenAIEmbedder(api_key="test-key")

        # Execute
        result = await embedder.embed_batch(["text1", "text2"])

        # Assert
        assert len(result) == 2
        assert all(isinstance(r, np.ndarray) for r in result)
        np.testing.assert_array_equal(result[0], [0.1, 0.2])
        np.testing.assert_array_equal(result[1], [0.3, 0.4])

    @pytest.mark.asyncio
    async def test_dimension_property(self, mock_openai):
        """Test dimension property returns correct value."""
        embedder = OpenAIEmbedder(model="text-embedding-ada-002")
        assert embedder.dimension == 1536

        embedder = OpenAIEmbedder(model="text-embedding-3-small")
        assert embedder.dimension == 1536

    @pytest.mark.asyncio
    async def test_error_handling(self, mock_openai):
        """Test error handling for API failures."""
        mock_openai.embeddings.create = AsyncMock(
            side_effect=Exception("API Error")
        )

        embedder = OpenAIEmbedder(api_key="test-key")

        with pytest.raises(Exception, match="API Error"):
            await embedder.embed("test text")


class TestLocalEmbedder:
    """Test local embedding provider."""

    @pytest.fixture
    def mock_sentence_transformer(self):
        """Mock sentence transformer."""
        mock_st_module = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_model.get_sentence_embedding_dimension.return_value = 3
        mock_st_module.SentenceTransformer.return_value = mock_model
        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            yield mock_model

    def test_initialization(self, mock_sentence_transformer):
        """Test local embedder initialization."""
        embedder = LocalEmbedder(model_name="test-model")
        assert embedder.model is not None
        assert embedder.dimension == 3

    @pytest.mark.asyncio
    async def test_embed_single(self, mock_sentence_transformer):
        """Test embedding single text with local model."""
        embedder = LocalEmbedder(model_name="test-model")

        result = await embedder.embed("test text")

        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        mock_sentence_transformer.encode.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_batch(self, mock_sentence_transformer):
        """Test batch embedding with local model."""
        mock_sentence_transformer.encode.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ])

        embedder = LocalEmbedder(model_name="test-model")

        result = await embedder.embed_batch(["text1", "text2"])

        assert len(result) == 2
        assert all(isinstance(r, np.ndarray) for r in result)
        assert result[0].shape == (3,)
        assert result[1].shape == (3,)

    @pytest.mark.asyncio
    async def test_normalize_embeddings(self, mock_sentence_transformer):
        """Test embedding normalization."""
        embedder = LocalEmbedder(model_name="test-model", normalize=True)

        # Mock non-normalized embedding
        mock_sentence_transformer.encode.return_value = np.array([[3.0, 4.0, 0.0]])

        result = await embedder.embed("test text")

        # Check normalization (should have unit length)
        norm = np.linalg.norm(result)
        assert pytest.approx(norm) == 1.0


class TestCachedEmbedder:
    """Test cached embedding provider."""

    @pytest.fixture
    def base_embedder(self):
        """Create mock base embedder."""
        embedder = Mock(spec=EmbeddingProvider)
        embedder.embed = AsyncMock(return_value=np.array([0.1, 0.2, 0.3]))
        embedder.embed_batch = AsyncMock(return_value=[
            np.array([0.1, 0.2]),
            np.array([0.3, 0.4]),
        ])
        embedder.dimension = 3
        return embedder

    @pytest.mark.asyncio
    async def test_cache_hit(self, base_embedder):
        """Test cache hit for repeated text."""
        cached = CachedEmbedder(base_embedder, cache_size=10)

        # First call
        result1 = await cached.embed("test text")
        assert base_embedder.embed.call_count == 1

        # Second call (should hit cache)
        result2 = await cached.embed("test text")
        assert base_embedder.embed.call_count == 1  # Not called again

        # Results should be identical
        np.testing.assert_array_equal(result1, result2)

    @pytest.mark.asyncio
    async def test_cache_miss(self, base_embedder):
        """Test cache miss for different texts."""
        cached = CachedEmbedder(base_embedder, cache_size=10)

        await cached.embed("text1")
        await cached.embed("text2")

        assert base_embedder.embed.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_eviction(self, base_embedder):
        """Test LRU cache eviction."""
        cached = CachedEmbedder(base_embedder, cache_size=2)

        # Fill cache
        await cached.embed("text1")
        await cached.embed("text2")
        await cached.embed("text3")  # Should evict text1

        # text1 should be evicted, so this should call base embedder
        base_embedder.embed.reset_mock()
        await cached.embed("text1")
        assert base_embedder.embed.call_count == 1

    @pytest.mark.asyncio
    async def test_batch_caching(self, base_embedder):
        """Test caching for batch operations."""
        cached = CachedEmbedder(base_embedder, cache_size=10)

        # First batch
        texts = ["text1", "text2", "text3"]
        await cached.embed_batch(texts)
        assert base_embedder.embed_batch.call_count == 1

        # Partially cached batch
        base_embedder.embed_batch.reset_mock()
        texts2 = ["text1", "text4"]  # text1 cached, text4 not
        await cached.embed_batch(texts2)

        # Should only embed text4
        assert base_embedder.embed_batch.call_count == 1
        called_texts = base_embedder.embed_batch.call_args[0][0]
        assert called_texts == ["text4"]


class TestBatchedEmbedder:
    """Test batched embedding provider."""

    @pytest.fixture
    def base_embedder(self):
        """Create mock base embedder."""
        embedder = Mock(spec=EmbeddingProvider)

        async def mock_embed_batch(texts):
            return [np.array([i, i+1, i+2]) for i in range(len(texts))]

        embedder.embed_batch = AsyncMock(side_effect=mock_embed_batch)
        embedder.dimension = 3
        return embedder

    @pytest.mark.asyncio
    async def test_batch_accumulation(self, base_embedder):
        """Test that embedder accumulates requests into batches."""
        batched = BatchedEmbedder(
            base_embedder,
            batch_size=3,
            batch_timeout=0.1
        )

        # Create multiple concurrent requests
        tasks = [
            batched.embed(f"text{i}")
            for i in range(5)
        ]

        results = await asyncio.gather(*tasks)

        # Should have made 2 batch calls (3 + 2)
        assert base_embedder.embed_batch.call_count == 2
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_batch_timeout(self, base_embedder):
        """Test batch timeout triggers processing."""
        batched = BatchedEmbedder(
            base_embedder,
            batch_size=10,  # Large batch size
            batch_timeout=0.05  # Short timeout
        )

        # Create fewer requests than batch size
        tasks = [
            batched.embed(f"text{i}")
            for i in range(2)
        ]

        results = await asyncio.gather(*tasks)

        # Should process due to timeout, not batch size
        assert base_embedder.embed_batch.call_count == 1
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_error_propagation(self, base_embedder):
        """Test that errors are properly propagated."""
        base_embedder.embed_batch = AsyncMock(
            side_effect=Exception("Embedding failed")
        )

        batched = BatchedEmbedder(base_embedder, batch_size=2)

        with pytest.raises(Exception, match="Embedding failed"):
            await batched.embed("test text")


class TestIntegration:
    """Integration tests for embedding providers."""

    @pytest.mark.asyncio
    async def test_cached_batched_combination(self):
        """Test combining cached and batched embedders."""
        # Create mock base embedder
        base = Mock(spec=EmbeddingProvider)

        # Return embeddings based on what texts are sent
        async def mock_embed_batch(texts):
            return [np.array([hash(t) % 100, hash(t) % 100 + 1]) for t in texts]

        base.embed_batch = AsyncMock(side_effect=mock_embed_batch)
        base.dimension = 2

        # Stack cached on top of batched
        batched = BatchedEmbedder(base, batch_size=10, batch_timeout=0.01)
        cached = CachedEmbedder(batched, cache_size=10)

        # Make sequential requests to test caching properly
        result1 = await cached.embed("text1")
        result2 = await cached.embed("text2")
        result3 = await cached.embed("text1")  # Cache hit

        # Results for same text should match
        np.testing.assert_array_equal(result1, result3)

        # Verify base was called for unique texts only
        # (at least once, caching should prevent repeat calls for text1)
        call_count_before = base.embed_batch.call_count
        await cached.embed("text1")  # Another cache hit
        assert base.embed_batch.call_count == call_count_before

    @pytest.mark.asyncio
    async def test_dimension_consistency(self):
        """Test dimension property is consistent across wrappers."""
        base = Mock(spec=EmbeddingProvider)
        base.dimension = 768

        batched = BatchedEmbedder(base)
        assert batched.dimension == 768

        cached = CachedEmbedder(batched)
        assert cached.dimension == 768


if __name__ == "__main__":
    pytest.main([__file__, "-v"])