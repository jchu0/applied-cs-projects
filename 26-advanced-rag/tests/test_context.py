"""Tests for context construction module."""

import pytest
from unittest.mock import AsyncMock, Mock

from advancedrag.schemas import Document, RerankResult, ConstructedContext
from advancedrag.context.constructor import (
    ContextConstructor,
    LLMCompressor,
    SemanticDeduplicator,
)


class TestSemanticDeduplicator:
    """Test SemanticDeduplicator."""

    @pytest.fixture
    def deduplicator(self):
        return SemanticDeduplicator(similarity_threshold=0.85)

    @pytest.mark.asyncio
    async def test_identical_documents_deduplicated(self, deduplicator):
        """Identical documents should be deduplicated to one."""
        docs = [
            Document(id="d1", content="Machine learning is a subset of AI", metadata={}),
            Document(id="d2", content="Machine learning is a subset of AI", metadata={}),
        ]
        result = await deduplicator.deduplicate(docs)
        assert len(result) == 1
        assert result[0].id == "d1"

    @pytest.mark.asyncio
    async def test_near_duplicate_removed(self, deduplicator):
        """Near-duplicates above threshold should be removed."""
        docs = [
            Document(id="d1", content="Machine learning is a subset of artificial intelligence", metadata={}),
            Document(id="d2", content="Machine learning is a subset of artificial intelligence today", metadata={}),
        ]
        result = await deduplicator.deduplicate(docs)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_unique_documents_kept(self, deduplicator):
        """Unique documents should all be kept."""
        docs = [
            Document(id="d1", content="Python is a programming language used for data science", metadata={}),
            Document(id="d2", content="The weather forecast predicts rain tomorrow afternoon", metadata={}),
        ]
        result = await deduplicator.deduplicate(docs)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_single_document(self, deduplicator):
        """Single document should be returned as-is."""
        docs = [Document(id="d1", content="Some content here", metadata={})]
        result = await deduplicator.deduplicate(docs)
        assert len(result) == 1
        assert result[0].id == "d1"


class TestLLMCompressor:
    """Test LLMCompressor."""

    @pytest.mark.asyncio
    async def test_extractive_compression_no_llm(self):
        """Without LLM, should use extractive compression."""
        compressor = LLMCompressor(llm_client=None, target_ratio=0.5)
        content = "Machine learning models train on data. The weather is nice. ML algorithms are powerful. Dogs are friendly."
        result = await compressor.compress("machine learning", content)
        # Should select sentences relevant to "machine learning"
        assert len(result) > 0
        assert len(result) <= len(content)

    @pytest.mark.asyncio
    async def test_llm_compression_with_mock(self):
        """With LLM client, should call LLM for compression."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value="ML models train on data.")
        compressor = LLMCompressor(llm_client=mock_llm, target_ratio=0.5)
        result = await compressor.compress("machine learning", "Some long text about machine learning models that train on data.")
        assert result == "ML models train on data."
        mock_llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_target_ratio_affects_output(self):
        """Different target ratios should produce different output lengths."""
        content = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."
        compressor_half = LLMCompressor(llm_client=None, target_ratio=0.5)
        compressor_quarter = LLMCompressor(llm_client=None, target_ratio=0.25)
        result_half = await compressor_half.compress("sentence", content)
        result_quarter = await compressor_quarter.compress("sentence", content)
        assert len(result_half) >= len(result_quarter)


class TestContextConstructor:
    """Test ContextConstructor."""

    def _make_rerank_result(self, doc_id, content, rank=0, score=0.9):
        return RerankResult(
            document=Document(id=doc_id, content=content, metadata={}),
            original_rank=rank,
            new_rank=rank,
            relevance_score=score,
        )

    @pytest.mark.asyncio
    async def test_full_construction_flow(self):
        """Test the full context construction pipeline."""
        constructor = ContextConstructor(max_tokens=1000)
        results = [
            self._make_rerank_result("d1", "First document about AI research", rank=0),
            self._make_rerank_result("d2", "Second document about neural networks", rank=1),
        ]
        context = await constructor.construct("AI research", results)
        assert isinstance(context, ConstructedContext)
        assert len(context.source_documents) > 0
        assert context.token_count > 0
        assert "d1" in [d.id for d in context.source_documents]

    @pytest.mark.asyncio
    async def test_empty_input(self):
        """Empty input should return empty context."""
        constructor = ContextConstructor(max_tokens=1000)
        context = await constructor.construct("query", [])
        assert context.content == ""
        assert len(context.source_documents) == 0
        assert context.compression_ratio == 1.0
        assert context.token_count == 0

    @pytest.mark.asyncio
    async def test_token_budget_respected(self):
        """Context should not exceed token budget."""
        constructor = ContextConstructor(max_tokens=10)
        results = [
            self._make_rerank_result("d1", " ".join(["word"] * 50), rank=0),
            self._make_rerank_result("d2", " ".join(["word"] * 50), rank=1),
        ]
        context = await constructor.construct("query", results)
        # Token count should be constrained
        assert context.token_count <= 20  # Some overhead from formatting

    @pytest.mark.asyncio
    async def test_dedup_integration(self):
        """Deduplicator should remove duplicate documents."""
        constructor = ContextConstructor(max_tokens=1000)
        results = [
            self._make_rerank_result("d1", "Machine learning is great for data analysis", rank=0),
            self._make_rerank_result("d2", "Machine learning is great for data analysis", rank=1),
        ]
        context = await constructor.construct("machine learning", results)
        assert context.metadata.get("dedup_removed", 0) >= 1

    @pytest.mark.asyncio
    async def test_compression_stats(self):
        """Context should include compression statistics."""
        constructor = ContextConstructor(max_tokens=1000)
        results = [
            self._make_rerank_result("d1", "Document about artificial intelligence and machine learning research", rank=0),
        ]
        context = await constructor.construct("AI", results)
        assert "original_count" in context.metadata
        assert "used_count" in context.metadata
        assert context.compression_ratio > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
