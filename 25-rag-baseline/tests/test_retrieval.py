"""Tests for retrieval module."""

import asyncio
import numpy as np
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Optional

from ragbaseline.retrieval import (
    Retriever,
    VectorRetriever,
    AsyncHybridRetriever as HybridRetriever,  # Use async version for these tests
    RerankedRetriever,
    FusionRetriever,
    MMRRetriever,
    SearchResult,
    RetrievalConfig,
)


class TestRetriever:
    """Test base retriever interface."""

    def test_abstract_methods(self):
        """Test that abstract methods are enforced."""
        with pytest.raises(TypeError):
            Retriever()


class TestVectorRetriever:
    """Test vector-based retriever."""

    @pytest.fixture
    def mock_vectorstore(self):
        """Create mock vector store."""
        store = Mock()
        store.search = AsyncMock(return_value=[
            SearchResult("doc1", 0.95, "Content 1", {"source": "file1"}),
            SearchResult("doc2", 0.85, "Content 2", {"source": "file2"}),
            SearchResult("doc3", 0.75, "Content 3", {"source": "file3"}),
        ])
        return store

    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedder."""
        embedder = Mock()
        embedder.embed = AsyncMock(return_value=np.array([1, 2, 3]))
        return embedder

    @pytest.mark.asyncio
    async def test_basic_retrieval(self, mock_vectorstore, mock_embedder):
        """Test basic vector retrieval."""
        retriever = VectorRetriever(
            vectorstore=mock_vectorstore,
            embedder=mock_embedder,
            k=3
        )

        results = await retriever.retrieve("test query")

        assert len(results) == 3
        assert results[0].id == "doc1"
        assert results[0].score == 0.95
        mock_embedder.embed.assert_called_once_with("test query")
        mock_vectorstore.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieval_with_filter(self, mock_vectorstore, mock_embedder):
        """Test retrieval with metadata filter."""
        retriever = VectorRetriever(
            vectorstore=mock_vectorstore,
            embedder=mock_embedder,
            k=2
        )

        results = await retriever.retrieve(
            "test query",
            filter={"source": "file1"}
        )

        # Check filter was passed to vectorstore
        call_args = mock_vectorstore.search.call_args[1]
        assert call_args["filter"] == {"source": "file1"}

    @pytest.mark.asyncio
    async def test_score_threshold(self, mock_vectorstore, mock_embedder):
        """Test filtering results by score threshold."""
        retriever = VectorRetriever(
            vectorstore=mock_vectorstore,
            embedder=mock_embedder,
            k=5,
            score_threshold=0.8
        )

        results = await retriever.retrieve("test query")

        # Only results with score >= 0.8
        assert len(results) == 2
        assert all(r.score >= 0.8 for r in results)

    @pytest.mark.asyncio
    async def test_empty_query_handling(self, mock_vectorstore, mock_embedder):
        """Test handling of empty query."""
        retriever = VectorRetriever(
            vectorstore=mock_vectorstore,
            embedder=mock_embedder
        )

        results = await retriever.retrieve("")

        assert len(results) == 0
        mock_embedder.embed.assert_not_called()

    @pytest.mark.asyncio
    async def test_batch_retrieval(self, mock_vectorstore, mock_embedder):
        """Test batch retrieval of multiple queries."""
        mock_embedder.embed_batch = AsyncMock(return_value=[
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
        ])
        mock_vectorstore.search_batch = AsyncMock(return_value=[
            [SearchResult("doc1", 0.95), SearchResult("doc2", 0.85)],
            [SearchResult("doc3", 0.90), SearchResult("doc4", 0.80)],
        ])

        retriever = VectorRetriever(
            vectorstore=mock_vectorstore,
            embedder=mock_embedder,
            k=2
        )

        results = await retriever.retrieve_batch(["query1", "query2"])

        assert len(results) == 2
        assert len(results[0]) == 2
        assert len(results[1]) == 2


class TestHybridRetriever:
    """Test hybrid retrieval combining multiple strategies."""

    @pytest.fixture
    def mock_vector_retriever(self):
        """Create mock vector retriever."""
        retriever = Mock(spec=VectorRetriever)
        retriever.retrieve = AsyncMock(return_value=[
            SearchResult("doc1", 0.9, "Vector result 1"),
            SearchResult("doc2", 0.8, "Vector result 2"),
        ])
        return retriever

    @pytest.fixture
    def mock_keyword_retriever(self):
        """Create mock keyword retriever."""
        retriever = Mock()
        retriever.retrieve = AsyncMock(return_value=[
            SearchResult("doc2", 0.85, "Keyword result 2"),
            SearchResult("doc3", 0.75, "Keyword result 3"),
        ])
        return retriever

    @pytest.mark.asyncio
    async def test_score_fusion(self, mock_vector_retriever, mock_keyword_retriever):
        """Test score fusion in hybrid retrieval."""
        hybrid = HybridRetriever(
            retrievers=[mock_vector_retriever, mock_keyword_retriever],
            weights=[0.6, 0.4]
        )

        results = await hybrid.retrieve("test query")

        # doc2 should be highest (appears in both)
        assert results[0].id == "doc2"
        # Score should be weighted combination
        expected_score = (0.8 * 0.6 + 0.85 * 0.4)
        assert results[0].score == pytest.approx(expected_score, rel=1e-2)

    @pytest.mark.asyncio
    async def test_reciprocal_rank_fusion(self, mock_vector_retriever, mock_keyword_retriever):
        """Test reciprocal rank fusion strategy."""
        hybrid = HybridRetriever(
            retrievers=[mock_vector_retriever, mock_keyword_retriever],
            fusion_method="reciprocal_rank"
        )

        results = await hybrid.retrieve("test query")

        assert len(results) == 3  # doc1, doc2, doc3
        # doc2 should score highest (rank 1 in keyword, rank 2 in vector)
        assert results[0].id == "doc2"

    @pytest.mark.asyncio
    async def test_max_fusion(self, mock_vector_retriever, mock_keyword_retriever):
        """Test max score fusion strategy."""
        hybrid = HybridRetriever(
            retrievers=[mock_vector_retriever, mock_keyword_retriever],
            fusion_method="max"
        )

        results = await hybrid.retrieve("test query")

        # Each doc should have its maximum score
        doc1 = next((r for r in results if r.id == "doc1"), None)
        assert doc1.score == 0.9  # Only from vector

        doc2 = next((r for r in results if r.id == "doc2"), None)
        assert doc2.score == 0.85  # Max of 0.8 and 0.85

    @pytest.mark.asyncio
    async def test_deduplication(self, mock_vector_retriever, mock_keyword_retriever):
        """Test deduplication of results."""
        hybrid = HybridRetriever(
            retrievers=[mock_vector_retriever, mock_keyword_retriever]
        )

        results = await hybrid.retrieve("test query")

        # Check no duplicate IDs
        ids = [r.id for r in results]
        assert len(ids) == len(set(ids))

        # doc2 should appear only once
        doc2_count = sum(1 for r in results if r.id == "doc2")
        assert doc2_count == 1


class TestRerankedRetriever:
    """Test retriever with reranking."""

    @pytest.fixture
    def mock_base_retriever(self):
        """Create mock base retriever."""
        retriever = Mock()
        retriever.retrieve = AsyncMock(return_value=[
            SearchResult("doc1", 0.7, "Content about cats"),
            SearchResult("doc2", 0.6, "Content about dogs"),
            SearchResult("doc3", 0.5, "Content about birds"),
        ])
        return retriever

    @pytest.fixture
    def mock_reranker(self):
        """Create mock reranker model."""
        reranker = Mock()
        reranker.rerank = AsyncMock(return_value=[
            (2, 0.95),  # doc3 reranked highest
            (0, 0.85),  # doc1 second
            (1, 0.75),  # doc2 third
        ])
        return reranker

    @pytest.mark.asyncio
    async def test_basic_reranking(self, mock_base_retriever, mock_reranker):
        """Test basic reranking of results."""
        reranked_retriever = RerankedRetriever(
            base_retriever=mock_base_retriever,
            reranker=mock_reranker,
            top_k=2
        )

        results = await reranked_retriever.retrieve("birds query")

        assert len(results) == 2
        assert results[0].id == "doc3"  # Reranked highest
        assert results[0].score == 0.95
        assert results[1].id == "doc1"
        assert results[1].score == 0.85

    @pytest.mark.asyncio
    async def test_preserve_metadata(self, mock_base_retriever, mock_reranker):
        """Test that metadata is preserved during reranking."""
        reranked_retriever = RerankedRetriever(
            base_retriever=mock_base_retriever,
            reranker=mock_reranker
        )

        results = await reranked_retriever.retrieve("test query")

        # Check original content is preserved
        doc3 = next((r for r in results if r.id == "doc3"), None)
        assert doc3.content == "Content about birds"

    @pytest.mark.asyncio
    async def test_reranker_failure_fallback(self, mock_base_retriever):
        """Test fallback to original results if reranker fails."""
        failing_reranker = Mock()
        failing_reranker.rerank = AsyncMock(side_effect=Exception("Reranker error"))

        reranked_retriever = RerankedRetriever(
            base_retriever=mock_base_retriever,
            reranker=failing_reranker,
            fallback_on_error=True
        )

        results = await reranked_retriever.retrieve("test query")

        # Should return original results
        assert len(results) == 3
        assert results[0].id == "doc1"
        assert results[0].score == 0.7


class TestMMRRetriever:
    """Test Maximal Marginal Relevance retriever."""

    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedder for MMR."""
        embedder = Mock()
        embedder.embed = AsyncMock(side_effect=lambda text: {
            "doc1": np.array([1, 0, 0]),
            "doc2": np.array([0.9, 0.1, 0]),
            "doc3": np.array([0, 1, 0]),
            "doc4": np.array([0, 0, 1]),
        }.get(text, np.random.randn(3)))
        return embedder

    @pytest.fixture
    def base_results(self):
        """Create base retrieval results."""
        return [
            SearchResult("doc1", 0.95, "Content 1"),
            SearchResult("doc2", 0.93, "Content 2"),  # Similar to doc1
            SearchResult("doc3", 0.90, "Content 3"),
            SearchResult("doc4", 0.88, "Content 4"),
        ]

    @pytest.mark.asyncio
    async def test_mmr_diversity(self, mock_embedder, base_results):
        """Test MMR promotes diversity."""
        mock_retriever = Mock()
        mock_retriever.retrieve = AsyncMock(return_value=base_results)

        mmr_retriever = MMRRetriever(
            base_retriever=mock_retriever,
            embedder=mock_embedder,
            lambda_mult=0.5,  # Balance relevance and diversity
            k=3
        )

        results = await mmr_retriever.retrieve("test query")

        assert len(results) == 3
        # Should select diverse documents
        ids = [r.id for r in results]
        assert "doc1" in ids  # Highest relevance
        assert "doc3" in ids or "doc4" in ids  # Diverse from doc1

        # doc2 might be excluded due to similarity to doc1
        if "doc2" not in ids:
            assert "doc3" in ids and "doc4" in ids

    @pytest.mark.asyncio
    async def test_mmr_lambda_extremes(self, mock_embedder, base_results):
        """Test MMR with extreme lambda values."""
        mock_retriever = Mock()
        mock_retriever.retrieve = AsyncMock(return_value=base_results)

        # Lambda = 1.0 (pure relevance)
        mmr_high = MMRRetriever(
            base_retriever=mock_retriever,
            embedder=mock_embedder,
            lambda_mult=1.0,
            k=3
        )

        results_high = await mmr_high.retrieve("test query")
        # Should return top 3 by relevance
        assert [r.id for r in results_high] == ["doc1", "doc2", "doc3"]

        # Lambda = 0.0 (pure diversity)
        mmr_low = MMRRetriever(
            base_retriever=mock_retriever,
            embedder=mock_embedder,
            lambda_mult=0.0,
            k=3
        )

        results_low = await mmr_low.retrieve("test query")
        # Should maximize diversity
        ids_low = [r.id for r in results_low]
        assert "doc1" in ids_low  # First pick
        # Should pick most different documents next


class TestFusionRetriever:
    """Test fusion strategies for combining results."""

    def test_reciprocal_rank_fusion_scoring(self):
        """Test RRF scoring calculation."""
        retriever = FusionRetriever(retrievers=[], k=60)

        results_lists = [
            [SearchResult("A", 0.9), SearchResult("B", 0.8), SearchResult("C", 0.7)],
            [SearchResult("B", 0.85), SearchResult("A", 0.75), SearchResult("D", 0.65)],
        ]

        fused = retriever._reciprocal_rank_fusion(results_lists, k=60)

        # B should score highest (rank 2 + rank 1)
        assert fused[0].id == "B"

        # A should be second (rank 1 + rank 2)
        assert fused[1].id == "A"

        # Verify RRF scores
        b_score = 1/(1+60) + 1/(2+60)  # ranks 1 and 2
        assert fused[0].score == pytest.approx(b_score, rel=1e-3)

    def test_weighted_fusion(self):
        """Test weighted fusion of scores."""
        retriever = FusionRetriever(retrievers=[], weights=[0.7, 0.3])

        results_lists = [
            [SearchResult("A", 0.9), SearchResult("B", 0.5)],
            [SearchResult("A", 0.7), SearchResult("C", 0.8)],
        ]

        fused = retriever._weighted_fusion(results_lists, [0.7, 0.3])

        # A should have combined score
        a_result = next((r for r in fused if r.id == "A"), None)
        expected_a = 0.9 * 0.7 + 0.7 * 0.3
        assert a_result.score == pytest.approx(expected_a)

        # C should have score from second list only
        c_result = next((r for r in fused if r.id == "C"), None)
        assert c_result.score == pytest.approx(0.8 * 0.3)


class TestRetrievalConfig:
    """Test retrieval configuration."""

    def test_config_creation(self):
        """Test creating retrieval config."""
        config = RetrievalConfig(
            retriever_type="hybrid",
            k=10,
            score_threshold=0.7,
            reranking_enabled=True,
            mmr_lambda=0.5
        )

        assert config.retriever_type == "hybrid"
        assert config.k == 10
        assert config.score_threshold == 0.7

    def test_config_validation(self):
        """Test config validation."""
        # Invalid k value
        with pytest.raises(ValueError):
            RetrievalConfig(k=0)

        # Invalid score threshold
        with pytest.raises(ValueError):
            RetrievalConfig(score_threshold=1.5)

        # Invalid lambda value
        with pytest.raises(ValueError):
            RetrievalConfig(mmr_lambda=2.0)

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "retriever_type": "vector",
            "k": 5,
            "score_threshold": 0.8
        }

        config = RetrievalConfig.from_dict(config_dict)

        assert config.retriever_type == "vector"
        assert config.k == 5
        assert config.score_threshold == 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])