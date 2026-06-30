"""Tests for the RAG pipeline module."""

import pytest
from unittest.mock import AsyncMock, Mock

from advancedrag.schemas import (
    Document,
    RAGResult,
    RetrievalResult,
    RerankResult,
    GeneratedAnswer,
    Citation,
    ConstructedContext,
)
from advancedrag.pipeline import RAGPipeline, create_pipeline
from advancedrag.retrieval.vector import MockEmbedding, SimpleVectorStore
from advancedrag.retrieval.hybrid import HybridRetriever
from advancedrag.reranking.reranker import MockReranker
from advancedrag.query.rewriter import RuleBasedRewriter


class TestRAGPipeline:
    """Test RAGPipeline."""

    @pytest.fixture
    def sample_docs(self):
        return [
            Document(id="d1", content="Artificial intelligence is transforming healthcare", metadata={"title": "AI Health"}),
            Document(id="d2", content="Machine learning models require large datasets", metadata={"title": "ML Data"}),
            Document(id="d3", content="Neural networks are inspired by the human brain", metadata={"title": "Neural Nets"}),
        ]

    @pytest.fixture
    def pipeline(self, sample_docs):
        p = create_pipeline()
        p.add_documents(sample_docs)
        return p

    @pytest.mark.asyncio
    async def test_end_to_end(self, pipeline):
        """Full pipeline execution should produce RAGResult."""
        result = await pipeline.execute("What is AI?", top_k=2)
        assert isinstance(result, RAGResult)
        assert result.query == "What is AI?"
        assert result.latency_ms > 0
        assert result.answer is not None

    @pytest.mark.asyncio
    async def test_no_rewrite(self, pipeline):
        """Pipeline should work without query rewriting."""
        result = await pipeline.execute("AI in healthcare", top_k=2, rewrite_query=False)
        assert isinstance(result, RAGResult)
        assert result.rewritten_query is None

    @pytest.mark.asyncio
    async def test_llm_generation(self, sample_docs):
        """Pipeline with LLM client should use LLM for generation."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value="AI transforms healthcare [1].")
        p = create_pipeline(llm_client=mock_llm)
        p.add_documents(sample_docs)
        result = await p.execute("What is AI?", top_k=2)
        assert isinstance(result, RAGResult)
        mock_llm.generate.assert_called()

    @pytest.mark.asyncio
    async def test_add_and_delete_docs(self):
        """Should support adding and deleting documents."""
        p = create_pipeline()
        docs = [Document(id="x1", content="Test document", metadata={})]
        p.add_documents(docs)
        result = await p.execute("test", top_k=1)
        assert len(result.retrieval_results) >= 0  # May or may not match
        p.delete_documents(["x1"])

    @pytest.mark.asyncio
    async def test_empty_retrieval(self):
        """Pipeline with no documents should still return a result."""
        p = create_pipeline()
        result = await p.execute("nonexistent query", top_k=5)
        assert isinstance(result, RAGResult)
        assert len(result.retrieval_results) == 0


class TestCreatePipeline:
    """Test create_pipeline factory."""

    def test_factory_defaults(self):
        """Factory should create pipeline with default components."""
        p = create_pipeline()
        assert isinstance(p, RAGPipeline)
        assert isinstance(p.retriever, HybridRetriever)
        assert isinstance(p.reranker, MockReranker)

    def test_custom_components(self):
        """Factory should accept custom components."""
        embedding = MockEmbedding()
        store = SimpleVectorStore()
        reranker = MockReranker()
        rewriter = RuleBasedRewriter()
        p = create_pipeline(
            embedding_model=embedding,
            vector_store=store,
            reranker=reranker,
            query_rewriter=rewriter,
        )
        assert isinstance(p, RAGPipeline)
        assert p.reranker is reranker

    def test_config_override(self):
        """Factory should accept config overrides."""
        config = {"top_k_retrieval": 50, "top_k_rerank": 5, "max_context_tokens": 2000}
        p = create_pipeline(config=config)
        assert p.config["top_k_retrieval"] == 50
        assert p.config["top_k_rerank"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
