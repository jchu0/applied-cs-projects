"""Tests for SLM components: chunker, embedder, retriever, reranker, summarizer, etc."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock

from microrag.schemas import (
    Chunk,
    Document,
    RetrievalResult,
    RerankResult,
    RoutingDecision,
    StabilizedAnswer,
)

# Optional imports requiring torch
try:
    from microrag.slm import (
        BaseSLM,
        MockSLM,
        MockChunkerSLM,
        MockEmbedderSLM,
        MockRetrieverSLM,
        MockRerankerSLM,
        MockSummarizerSLM,
        MockCoTCompressorSLM,
        MockAnswerStabilizerSLM,
    )
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="Requires ML extras (transformers)")


# ============================================================================
# BaseSLM Tests
# ============================================================================

class TestBaseSLM:
    """Tests for BaseSLM base class."""

    def test_mock_slm_is_loaded_by_default(self):
        """Test that MockSLM is loaded by default."""
        slm = MockSLM()
        assert slm.is_loaded is True

    @pytest.mark.asyncio
    async def test_mock_slm_process(self):
        """Test MockSLM process method."""
        slm = MockSLM()
        result = await slm.process(key1="value1", key2="value2")

        assert result["mock"] is True
        assert "key1" in result["inputs"]
        assert "key2" in result["inputs"]

    @pytest.mark.asyncio
    async def test_mock_slm_callable(self):
        """Test MockSLM can be called as function."""
        slm = MockSLM()
        result = await slm(key="value")

        assert result["mock"] is True


# ============================================================================
# MockChunkerSLM Tests
# ============================================================================

class TestMockChunkerSLM:
    """Tests for MockChunkerSLM."""

    @pytest.mark.asyncio
    async def test_chunker_empty_document(self):
        """Test chunker with empty document."""
        chunker = MockChunkerSLM()
        chunks = await chunker.process(document="")

        assert chunks == []

    @pytest.mark.asyncio
    async def test_chunker_none_document(self):
        """Test chunker with None document."""
        chunker = MockChunkerSLM()
        chunks = await chunker.process(document=None)

        assert chunks == []

    @pytest.mark.asyncio
    async def test_chunker_single_paragraph(self):
        """Test chunker with single paragraph."""
        chunker = MockChunkerSLM()
        text = "This is a single paragraph about machine learning."

        chunks = await chunker.process(document=text)

        assert len(chunks) == 1
        assert isinstance(chunks[0], Chunk)
        assert chunks[0].content == text.strip()
        assert chunks[0].chunk_type == "paragraph"

    @pytest.mark.asyncio
    async def test_chunker_multiple_paragraphs(self, sample_document_text):
        """Test chunker splits multiple paragraphs."""
        chunker = MockChunkerSLM()
        chunks = await chunker.process(document=sample_document_text)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.semantic_score > 0

    @pytest.mark.asyncio
    async def test_chunker_metadata(self):
        """Test chunker adds metadata to chunks."""
        chunker = MockChunkerSLM()
        text = "Paragraph one.\n\nParagraph two."

        chunks = await chunker.process(document=text)

        for i, chunk in enumerate(chunks):
            assert "chunk_idx" in chunk.metadata


# ============================================================================
# MockEmbedderSLM Tests
# ============================================================================

class TestMockEmbedderSLM:
    """Tests for MockEmbedderSLM."""

    @pytest.mark.asyncio
    async def test_embedder_single_text(self):
        """Test embedding a single text."""
        embedder = MockEmbedderSLM(embedding_dim=384)
        embedding = await embedder.process(text="Test text")

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1, 384)

    @pytest.mark.asyncio
    async def test_embedder_multiple_texts(self):
        """Test embedding multiple texts."""
        embedder = MockEmbedderSLM(embedding_dim=384)
        texts = ["Text one", "Text two", "Text three"]

        embeddings = await embedder.process(texts=texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 384)

    @pytest.mark.asyncio
    async def test_embedder_chunks(self, sample_chunks):
        """Test embedding chunks."""
        embedder = MockEmbedderSLM(embedding_dim=384)
        embeddings = await embedder.process(chunk=sample_chunks)

        assert embeddings.shape == (len(sample_chunks), 384)

    @pytest.mark.asyncio
    async def test_embedder_empty_input(self):
        """Test embedder with empty input."""
        embedder = MockEmbedderSLM(embedding_dim=384)
        embedding = await embedder.process()

        assert embedding.shape == (0, 384)

    @pytest.mark.asyncio
    async def test_embedder_normalized_output(self):
        """Test that embeddings are normalized."""
        embedder = MockEmbedderSLM(embedding_dim=384)
        embeddings = await embedder.process(texts=["Text one", "Text two"])

        # Check that embeddings are normalized (L2 norm ~1)
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(2), decimal=5)


# ============================================================================
# MockRetrieverSLM Tests
# ============================================================================

class TestMockRetrieverSLM:
    """Tests for MockRetrieverSLM."""

    @pytest.mark.asyncio
    async def test_retriever_basic_query(self, sample_query):
        """Test basic retrieval."""
        retriever = MockRetrieverSLM()
        results = await retriever.process(query=sample_query)

        assert len(results) > 0
        for result in results:
            assert isinstance(result, RetrievalResult)
            assert result.score > 0

    @pytest.mark.asyncio
    async def test_retriever_empty_query(self):
        """Test retriever with empty query."""
        retriever = MockRetrieverSLM()
        results = await retriever.process(query="")

        assert results == []

    @pytest.mark.asyncio
    async def test_retriever_none_query(self):
        """Test retriever with None query."""
        retriever = MockRetrieverSLM()
        results = await retriever.process(query=None)

        assert results == []

    @pytest.mark.asyncio
    async def test_retriever_top_k(self, sample_query):
        """Test retriever respects top_k parameter."""
        retriever = MockRetrieverSLM()
        results = await retriever.process(query=sample_query, top_k=3)

        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_retriever_results_sorted(self, sample_query):
        """Test that results are sorted by score (descending)."""
        retriever = MockRetrieverSLM()
        results = await retriever.process(query=sample_query)

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_retriever_results_have_documents(self, sample_query):
        """Test that retrieval results contain documents."""
        retriever = MockRetrieverSLM()
        results = await retriever.process(query=sample_query)

        for result in results:
            assert result.document is not None
            assert result.document.id is not None
            assert result.document.content is not None


# ============================================================================
# MockRerankerSLM Tests
# ============================================================================

class TestMockRerankerSLM:
    """Tests for MockRerankerSLM."""

    @pytest.mark.asyncio
    async def test_reranker_basic(self, sample_query, sample_retrieval_results):
        """Test basic reranking."""
        reranker = MockRerankerSLM()
        results = await reranker.process(
            query=sample_query,
            retrieve=sample_retrieval_results
        )

        assert len(results) > 0
        for result in results:
            assert isinstance(result, RerankResult)
            assert result.relevance_score > 0

    @pytest.mark.asyncio
    async def test_reranker_empty_retrieve(self, sample_query):
        """Test reranker with empty retrieval results."""
        reranker = MockRerankerSLM()
        results = await reranker.process(query=sample_query, retrieve=[])

        assert results == []

    @pytest.mark.asyncio
    async def test_reranker_none_retrieve(self, sample_query):
        """Test reranker with None retrieval results."""
        reranker = MockRerankerSLM()
        results = await reranker.process(query=sample_query, retrieve=None)

        assert results == []

    @pytest.mark.asyncio
    async def test_reranker_top_k(self, sample_query, sample_retrieval_results):
        """Test reranker respects top_k parameter."""
        reranker = MockRerankerSLM()
        results = await reranker.process(
            query=sample_query,
            retrieve=sample_retrieval_results,
            top_k=2
        )

        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_reranker_assigns_new_ranks(self, sample_query, sample_retrieval_results):
        """Test that reranker assigns new ranks starting from 0."""
        reranker = MockRerankerSLM()
        results = await reranker.process(
            query=sample_query,
            retrieve=sample_retrieval_results
        )

        new_ranks = [r.new_rank for r in results]
        expected_ranks = list(range(len(results)))
        assert new_ranks == expected_ranks

    @pytest.mark.asyncio
    async def test_reranker_preserves_original_ranks(self, sample_query, sample_retrieval_results):
        """Test that reranker preserves original ranks."""
        reranker = MockRerankerSLM()
        results = await reranker.process(
            query=sample_query,
            retrieve=sample_retrieval_results
        )

        for result, original in zip(results, sample_retrieval_results):
            assert result.original_rank == original.rank


# ============================================================================
# MockSummarizerSLM Tests
# ============================================================================

class TestMockSummarizerSLM:
    """Tests for MockSummarizerSLM."""

    @pytest.mark.asyncio
    async def test_summarizer_basic(self, sample_query, sample_rerank_results):
        """Test basic summarization."""
        summarizer = MockSummarizerSLM()
        summary = await summarizer.process(
            query=sample_query,
            rerank=sample_rerank_results
        )

        assert isinstance(summary, str)
        assert len(summary) > 0

    @pytest.mark.asyncio
    async def test_summarizer_empty_rerank(self, sample_query):
        """Test summarizer with empty rerank results."""
        summarizer = MockSummarizerSLM()
        summary = await summarizer.process(query=sample_query, rerank=[])

        assert "No documents to summarize" in summary

    @pytest.mark.asyncio
    async def test_summarizer_includes_document_count(self, sample_query, sample_rerank_results):
        """Test that summary mentions document count."""
        summarizer = MockSummarizerSLM()
        summary = await summarizer.process(
            query=sample_query,
            rerank=sample_rerank_results
        )

        assert str(len(sample_rerank_results)) in summary


# ============================================================================
# MockCoTCompressorSLM Tests
# ============================================================================

class TestMockCoTCompressorSLM:
    """Tests for MockCoTCompressorSLM."""

    @pytest.mark.asyncio
    async def test_compressor_basic(self, sample_query):
        """Test basic compression."""
        compressor = MockCoTCompressorSLM()
        text = "This is a long text that needs compression. " * 10

        compressed = await compressor.process(
            query=sample_query,
            summarize=text
        )

        assert isinstance(compressed, str)
        assert len(compressed) > 0
        assert len(compressed.split()) < len(text.split())

    @pytest.mark.asyncio
    async def test_compressor_empty_input(self, sample_query):
        """Test compressor with empty input."""
        compressor = MockCoTCompressorSLM()
        compressed = await compressor.process(query=sample_query, summarize="")

        assert compressed == ""

    @pytest.mark.asyncio
    async def test_compressor_none_input(self, sample_query):
        """Test compressor with None input."""
        compressor = MockCoTCompressorSLM()
        compressed = await compressor.process(query=sample_query, summarize=None)

        assert compressed == ""

    @pytest.mark.asyncio
    async def test_compressor_short_text_preserved(self, sample_query):
        """Test that very short text is minimally compressed."""
        compressor = MockCoTCompressorSLM()
        short_text = "Short text."

        compressed = await compressor.process(
            query=sample_query,
            summarize=short_text
        )

        # Should have at least some content
        assert len(compressed) > 0


# ============================================================================
# MockAnswerStabilizerSLM Tests
# ============================================================================

class TestMockAnswerStabilizerSLM:
    """Tests for MockAnswerStabilizerSLM."""

    @pytest.mark.asyncio
    async def test_stabilizer_basic(self, sample_query):
        """Test basic answer stabilization."""
        stabilizer = MockAnswerStabilizerSLM()
        result = await stabilizer.process(
            query=sample_query,
            compress_cot="Some context about machine learning."
        )

        assert isinstance(result, StabilizedAnswer)
        assert len(result.answer) > 0
        assert 0 <= result.confidence <= 1
        assert 0 <= result.consistency <= 1

    @pytest.mark.asyncio
    async def test_stabilizer_empty_query(self):
        """Test stabilizer with empty query."""
        stabilizer = MockAnswerStabilizerSLM()
        result = await stabilizer.process(query="", compress_cot="Context")

        assert isinstance(result, StabilizedAnswer)

    @pytest.mark.asyncio
    async def test_stabilizer_none_query(self):
        """Test stabilizer with None query."""
        stabilizer = MockAnswerStabilizerSLM()
        result = await stabilizer.process(query=None, compress_cot="Context")

        assert isinstance(result, StabilizedAnswer)
        assert "No query" in result.answer

    @pytest.mark.asyncio
    async def test_stabilizer_metadata(self, sample_query):
        """Test that stabilizer returns proper metadata."""
        stabilizer = MockAnswerStabilizerSLM()
        result = await stabilizer.process(
            query=sample_query,
            compress_cot="Context"
        )

        assert result.num_samples > 0
        assert len(result.reasoning) > 0


# ============================================================================
# SLM Loading/Unloading Tests (Mock)
# ============================================================================

class TestSLMLifecycle:
    """Tests for SLM loading/unloading lifecycle."""

    def test_mock_slm_starts_loaded(self):
        """Test that mock SLMs start in loaded state."""
        slms = [
            MockChunkerSLM(),
            MockEmbedderSLM(),
            MockRetrieverSLM(),
            MockRerankerSLM(),
            MockSummarizerSLM(),
            MockCoTCompressorSLM(),
            MockAnswerStabilizerSLM(),
        ]

        for slm in slms:
            assert slm.is_loaded is True

    @pytest.mark.asyncio
    async def test_slm_process_without_loading(self):
        """Test that mock SLMs can process without explicit load."""
        chunker = MockChunkerSLM()
        # Should work without calling load()
        result = await chunker.process(document="Test document")
        assert result is not None


# ============================================================================
# Integration Tests
# ============================================================================

class TestSLMIntegration:
    """Integration tests for SLM components working together."""

    @pytest.mark.asyncio
    async def test_chunker_to_embedder_flow(self, sample_document_text):
        """Test chunker output flows to embedder."""
        chunker = MockChunkerSLM()
        embedder = MockEmbedderSLM(embedding_dim=384)

        # Chunk the document
        chunks = await chunker.process(document=sample_document_text)
        assert len(chunks) > 0

        # Embed the chunks
        embeddings = await embedder.process(chunk=chunks)
        assert embeddings.shape[0] == len(chunks)

    @pytest.mark.asyncio
    async def test_retriever_to_reranker_flow(self, sample_query):
        """Test retriever output flows to reranker."""
        retriever = MockRetrieverSLM()
        reranker = MockRerankerSLM()

        # Retrieve documents
        retrieved = await retriever.process(query=sample_query, top_k=10)
        assert len(retrieved) > 0

        # Rerank documents
        reranked = await reranker.process(query=sample_query, retrieve=retrieved)
        assert len(reranked) > 0
        assert len(reranked) <= len(retrieved)

    @pytest.mark.asyncio
    async def test_reranker_to_summarizer_flow(self, sample_query, sample_retrieval_results):
        """Test reranker output flows to summarizer."""
        reranker = MockRerankerSLM()
        summarizer = MockSummarizerSLM()

        # Rerank
        reranked = await reranker.process(
            query=sample_query,
            retrieve=sample_retrieval_results
        )

        # Summarize
        summary = await summarizer.process(query=sample_query, rerank=reranked)
        assert isinstance(summary, str)
        assert len(summary) > 0

    @pytest.mark.asyncio
    async def test_full_query_pipeline(self, sample_query):
        """Test full query processing pipeline with mocks."""
        retriever = MockRetrieverSLM()
        reranker = MockRerankerSLM()
        summarizer = MockSummarizerSLM()
        compressor = MockCoTCompressorSLM()
        stabilizer = MockAnswerStabilizerSLM()

        # Step 1: Retrieve
        retrieved = await retriever.process(query=sample_query, top_k=10)
        assert len(retrieved) > 0

        # Step 2: Rerank
        reranked = await reranker.process(query=sample_query, retrieve=retrieved)
        assert len(reranked) > 0

        # Step 3: Summarize
        summary = await summarizer.process(query=sample_query, rerank=reranked)
        assert len(summary) > 0

        # Step 4: Compress
        compressed = await compressor.process(query=sample_query, summarize=summary)
        assert isinstance(compressed, str)

        # Step 5: Stabilize
        answer = await stabilizer.process(query=sample_query, compress_cot=compressed)
        assert isinstance(answer, StabilizedAnswer)
        assert answer.confidence > 0


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestSLMEdgeCases:
    """Tests for SLM edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_chunker_special_characters(self):
        """Test chunker handles special characters."""
        chunker = MockChunkerSLM()
        text = "Text with special chars: !@#$%^&*()\n\nAnother paragraph."

        chunks = await chunker.process(document=text)
        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_chunker_unicode_text(self):
        """Test chunker handles unicode text."""
        chunker = MockChunkerSLM()
        text = "English text with unicode. Additional content here."

        chunks = await chunker.process(document=text)
        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_embedder_very_long_text(self):
        """Test embedder handles very long text."""
        embedder = MockEmbedderSLM(embedding_dim=384)
        long_text = "Word " * 10000  # 10k words

        embedding = await embedder.process(text=long_text)
        assert embedding.shape == (1, 384)

    @pytest.mark.asyncio
    async def test_retriever_special_query(self):
        """Test retriever handles special characters in query."""
        retriever = MockRetrieverSLM()
        query = "What is 'machine learning' (ML)?"

        results = await retriever.process(query=query)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_reranker_single_document(self, sample_query, sample_documents):
        """Test reranker with single document."""
        reranker = MockRerankerSLM()
        single_result = [RetrievalResult(
            document=sample_documents[0],
            score=0.9,
            retriever_type="dense",
            rank=0
        )]

        results = await reranker.process(
            query=sample_query,
            retrieve=single_result
        )

        assert len(results) == 1
        assert results[0].new_rank == 0
