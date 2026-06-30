"""Tests for reranking module."""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch

from advancedrag.schemas import Document, RetrievalResult, RerankResult
from advancedrag.reranking.reranker import (
    BaseReranker,
    MockReranker,
    MultiStageReranker,
    SLMReranker,
)


class TestMockReranker:
    """Test mock reranker functionality."""

    @pytest.fixture
    def mock_reranker(self):
        """Create mock reranker instance."""
        return MockReranker()

    @pytest.fixture
    def sample_candidates(self):
        """Create sample retrieval results for testing."""
        documents = [
            Document(id="doc1", content="Machine learning is a subset of AI.", metadata={"category": "tech"}),
            Document(id="doc2", content="Deep learning uses neural networks.", metadata={"category": "tech"}),
            Document(id="doc3", content="Cooking recipes for pasta.", metadata={"category": "food"}),
        ]
        return [
            RetrievalResult(document=documents[0], score=0.9, retriever_type="bm25", rank=0),
            RetrievalResult(document=documents[1], score=0.7, retriever_type="bm25", rank=1),
            RetrievalResult(document=documents[2], score=0.5, retriever_type="bm25", rank=2),
        ]

    @pytest.mark.asyncio
    async def test_basic_reranking(self, mock_reranker, sample_candidates):
        """Test basic reranking preserves order for mock."""
        query = "What is machine learning?"

        results = await mock_reranker.rerank(query, sample_candidates)

        assert len(results) == 3
        assert all(isinstance(r, RerankResult) for r in results)
        # Mock reranker preserves order
        assert results[0].document.id == "doc1"
        assert results[1].document.id == "doc2"
        assert results[2].document.id == "doc3"

    @pytest.mark.asyncio
    async def test_top_k_limiting(self, mock_reranker, sample_candidates):
        """Test top_k parameter limits results."""
        query = "What is machine learning?"

        results = await mock_reranker.rerank(query, sample_candidates, top_k=2)

        assert len(results) == 2
        assert results[0].document.id == "doc1"
        assert results[1].document.id == "doc2"

    @pytest.mark.asyncio
    async def test_empty_candidates(self, mock_reranker):
        """Test handling of empty candidates."""
        query = "What is machine learning?"

        results = await mock_reranker.rerank(query, [])

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_rerank_result_structure(self, mock_reranker, sample_candidates):
        """Test that RerankResult has correct structure."""
        query = "What is machine learning?"

        results = await mock_reranker.rerank(query, sample_candidates)

        result = results[0]
        assert hasattr(result, 'document')
        assert hasattr(result, 'original_rank')
        assert hasattr(result, 'new_rank')
        assert hasattr(result, 'relevance_score')
        assert hasattr(result, 'features')
        assert result.features.get("mock") is True

    @pytest.mark.asyncio
    async def test_ranks_are_updated(self, mock_reranker, sample_candidates):
        """Test that new_rank values are correctly assigned."""
        query = "What is machine learning?"

        results = await mock_reranker.rerank(query, sample_candidates)

        for i, result in enumerate(results):
            assert result.new_rank == i
            assert result.original_rank == sample_candidates[i].rank


class TestSLMReranker:
    """Test SLM-based reranker."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        client = Mock()
        client.generate = AsyncMock(return_value='[{"relevance": 9.0}, {"relevance": 7.0}]')
        return client

    @pytest.fixture
    def slm_reranker(self, mock_llm_client):
        """Create SLM reranker with mock LLM."""
        return SLMReranker(llm_client=mock_llm_client, batch_size=2)

    @pytest.fixture
    def sample_candidates(self):
        """Create sample retrieval results."""
        documents = [
            Document(id="doc1", content="Machine learning is a subset of AI.", metadata={}),
            Document(id="doc2", content="Deep learning uses neural networks.", metadata={}),
        ]
        return [
            RetrievalResult(document=documents[0], score=0.9, retriever_type="vector", rank=0),
            RetrievalResult(document=documents[1], score=0.7, retriever_type="vector", rank=1),
        ]

    @pytest.mark.asyncio
    async def test_slm_reranking_scores_documents(self, slm_reranker, sample_candidates):
        """Test SLM reranker scores documents using LLM."""
        query = "What is machine learning?"

        results = await slm_reranker.rerank(query, sample_candidates)

        assert len(results) == 2
        # Higher scored doc should be first
        assert results[0].relevance_score >= results[1].relevance_score

    @pytest.mark.asyncio
    async def test_slm_reranking_orders_by_relevance(self, slm_reranker, mock_llm_client, sample_candidates):
        """Test SLM reranker orders by LLM scores."""
        # Make second doc score higher
        mock_llm_client.generate = AsyncMock(return_value='[{"relevance": 5.0}, {"relevance": 9.0}]')

        query = "What is deep learning?"
        results = await slm_reranker.rerank(query, sample_candidates)

        # doc2 should be first due to higher relevance
        assert results[0].document.id == "doc2"
        assert results[0].relevance_score == 9.0

    @pytest.mark.asyncio
    async def test_slm_handles_invalid_json(self, mock_llm_client, sample_candidates):
        """Test fallback when LLM returns invalid JSON."""
        mock_llm_client.generate = AsyncMock(return_value='invalid json response')
        reranker = SLMReranker(llm_client=mock_llm_client, batch_size=2)

        query = "What is machine learning?"
        results = await reranker.rerank(query, sample_candidates)

        # Should return default scores
        assert len(results) == 2
        assert all(r.relevance_score == 5.0 for r in results)

    @pytest.mark.asyncio
    async def test_slm_batch_processing(self, mock_llm_client):
        """Test SLM processes documents in batches."""
        mock_llm_client.generate = AsyncMock(
            side_effect=[
                '[{"relevance": 8.0}, {"relevance": 7.0}]',
                '[{"relevance": 6.0}]'
            ]
        )
        reranker = SLMReranker(llm_client=mock_llm_client, batch_size=2)

        documents = [
            Document(id=f"doc{i}", content=f"Content {i}", metadata={})
            for i in range(3)
        ]
        candidates = [
            RetrievalResult(document=doc, score=0.5, retriever_type="bm25", rank=i)
            for i, doc in enumerate(documents)
        ]

        query = "test query"
        results = await reranker.rerank(query, candidates)

        # Should have made 2 LLM calls (batch of 2, batch of 1)
        assert mock_llm_client.generate.call_count == 2
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_slm_empty_candidates(self, slm_reranker):
        """Test SLM reranker with empty candidates."""
        results = await slm_reranker.rerank("query", [])
        assert len(results) == 0


class TestMultiStageReranker:
    """Test multi-stage reranking pipeline."""

    @pytest.fixture
    def sample_candidates(self):
        """Create sample retrieval results."""
        documents = [
            Document(id="doc1", content="Machine learning AI algorithms", metadata={}),
            Document(id="doc2", content="Deep learning neural networks training", metadata={}),
            Document(id="doc3", content="Cooking recipes for pasta dishes", metadata={}),
            Document(id="doc4", content="Natural language processing text", metadata={}),
        ]
        return [
            RetrievalResult(document=doc, score=0.9 - i * 0.1, retriever_type="hybrid", rank=i)
            for i, doc in enumerate(documents)
        ]

    @pytest.mark.asyncio
    async def test_single_stage_reranking(self, sample_candidates):
        """Test multi-stage with only stage1."""
        stage1 = MockReranker()
        reranker = MultiStageReranker(stage1_reranker=stage1)

        query = "What is machine learning?"
        results = await reranker.rerank(query, sample_candidates)

        assert len(results) == len(sample_candidates)

    @pytest.mark.asyncio
    async def test_two_stage_reranking(self, sample_candidates):
        """Test two-stage reranking pipeline."""
        stage1 = MockReranker()
        stage2 = MockReranker()
        reranker = MultiStageReranker(
            stage1_reranker=stage1,
            stage2_reranker=stage2,
            stage1_top_k=3
        )

        query = "What is machine learning?"
        results = await reranker.rerank(query, sample_candidates, top_k=2)

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_stage1_top_k_filtering(self, sample_candidates):
        """Test that stage1 filters to stage1_top_k."""
        stage1 = MockReranker()
        reranker = MultiStageReranker(
            stage1_reranker=stage1,
            stage1_top_k=2
        )

        query = "What is machine learning?"
        results = await reranker.rerank(query, sample_candidates)

        # Should only pass stage1_top_k results
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_mmr_diversity_selection(self):
        """Test MMR diversity in result selection."""
        # Create documents with similar content
        documents = [
            Document(id="doc1", content="machine learning algorithms data", metadata={}),
            Document(id="doc2", content="machine learning algorithms training", metadata={}),  # Similar to doc1
            Document(id="doc3", content="cooking recipes pasta dishes", metadata={}),  # Different
        ]
        candidates = [
            RetrievalResult(document=doc, score=0.9 - i * 0.05, retriever_type="hybrid", rank=i)
            for i, doc in enumerate(documents)
        ]

        stage1 = MockReranker()
        reranker = MultiStageReranker(
            stage1_reranker=stage1,
            diversity_lambda=0.3  # Lower lambda = more diversity preference
        )

        query = "test query"
        results = await reranker.rerank(query, candidates, top_k=2)

        # With high diversity preference, doc3 (different content) might be selected
        # even though it has lower relevance
        assert len(results) == 2
        doc_ids = [r.document.id for r in results]
        # doc1 should still be first (highest relevance)
        assert doc_ids[0] == "doc1"

    @pytest.mark.asyncio
    async def test_mmr_empty_input(self):
        """Test MMR with empty input."""
        stage1 = MockReranker()
        reranker = MultiStageReranker(stage1_reranker=stage1)

        results = await reranker.rerank("query", [])

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_ranks_updated_after_mmr(self, sample_candidates):
        """Test that ranks are correctly updated after MMR."""
        stage1 = MockReranker()
        reranker = MultiStageReranker(
            stage1_reranker=stage1,
            diversity_lambda=0.7
        )

        query = "test query"
        results = await reranker.rerank(query, sample_candidates)

        for i, result in enumerate(results):
            assert result.new_rank == i


class TestMaxSimilarity:
    """Test max similarity computation for MMR diversity."""

    def test_max_similarity_computation(self):
        """Test _max_similarity helper method."""
        stage1 = MockReranker()
        reranker = MultiStageReranker(stage1_reranker=stage1)

        # Create similar documents
        doc1 = Document(id="doc1", content="machine learning algorithms", metadata={})
        doc2 = Document(id="doc2", content="machine learning models", metadata={})
        doc3 = Document(id="doc3", content="cooking pasta recipes", metadata={})

        candidate = RerankResult(
            document=doc1,
            original_rank=0,
            new_rank=0,
            relevance_score=0.9,
            features={}
        )

        selected = [
            RerankResult(document=doc2, original_rank=1, new_rank=0, relevance_score=0.8, features={}),
        ]

        sim1 = reranker._max_similarity(candidate, selected)

        # doc1 and doc2 share "machine", "learning" -> should have some similarity
        assert sim1 > 0

        # Test with dissimilar document
        candidate_diff = RerankResult(
            document=doc3,
            original_rank=2,
            new_rank=0,
            relevance_score=0.7,
            features={}
        )

        sim2 = reranker._max_similarity(candidate_diff, selected)

        # doc3 is about cooking, should have lower similarity
        assert sim2 < sim1

    def test_max_similarity_empty_selected(self):
        """Test max similarity with no selected documents."""
        stage1 = MockReranker()
        reranker = MultiStageReranker(stage1_reranker=stage1)

        doc1 = Document(id="doc1", content="test content", metadata={})
        candidate = RerankResult(
            document=doc1,
            original_rank=0,
            new_rank=0,
            relevance_score=0.9,
            features={}
        )

        sim = reranker._max_similarity(candidate, [])

        assert sim == 0.0


class TestRerankResultDataclass:
    """Test RerankResult data structure."""

    def test_rerank_result_creation(self):
        """Test creating a RerankResult."""
        doc = Document(id="doc1", content="Test content", metadata={"key": "value"})
        result = RerankResult(
            document=doc,
            original_rank=0,
            new_rank=1,
            relevance_score=0.95,
            features={"model": "test"}
        )

        assert result.document.id == "doc1"
        assert result.original_rank == 0
        assert result.new_rank == 1
        assert result.relevance_score == 0.95
        assert result.features["model"] == "test"

    def test_rerank_result_to_dict(self):
        """Test RerankResult serialization."""
        doc = Document(id="doc1", content="Test content", metadata={"key": "value"})
        result = RerankResult(
            document=doc,
            original_rank=0,
            new_rank=1,
            relevance_score=0.95,
            features={"model": "test"}
        )

        result_dict = result.to_dict()

        assert result_dict["original_rank"] == 0
        assert result_dict["new_rank"] == 1
        assert result_dict["relevance_score"] == 0.95
        assert result_dict["document"]["id"] == "doc1"

    def test_rerank_result_default_features(self):
        """Test RerankResult with default features."""
        doc = Document(id="doc1", content="Test", metadata={})
        result = RerankResult(
            document=doc,
            original_rank=0,
            new_rank=0,
            relevance_score=0.5
        )

        assert result.features == {}


class TestRerankerOrdering:
    """Test that rerankers correctly order results by score."""

    @pytest.fixture
    def unsorted_candidates(self):
        """Create candidates in non-sorted order."""
        documents = [
            Document(id="doc_low", content="Low relevance content", metadata={}),
            Document(id="doc_high", content="High relevance content", metadata={}),
            Document(id="doc_mid", content="Medium relevance content", metadata={}),
        ]
        return [
            RetrievalResult(document=documents[0], score=0.3, retriever_type="bm25", rank=0),
            RetrievalResult(document=documents[1], score=0.9, retriever_type="bm25", rank=1),
            RetrievalResult(document=documents[2], score=0.6, retriever_type="bm25", rank=2),
        ]

    @pytest.mark.asyncio
    async def test_slm_reranker_orders_by_score(self, unsorted_candidates):
        """Test SLM reranker orders by LLM-assigned scores."""
        mock_llm = Mock()
        mock_llm.generate = AsyncMock(
            return_value='[{"relevance": 3.0}, {"relevance": 9.0}, {"relevance": 6.0}]'
        )
        reranker = SLMReranker(llm_client=mock_llm, batch_size=5)

        results = await reranker.rerank("query", unsorted_candidates)

        # Should be ordered by relevance: doc_high (9), doc_mid (6), doc_low (3)
        assert results[0].document.id == "doc_high"
        assert results[1].document.id == "doc_mid"
        assert results[2].document.id == "doc_low"
        assert results[0].relevance_score > results[1].relevance_score > results[2].relevance_score


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
