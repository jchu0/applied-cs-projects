"""Tests for RAG pipeline module."""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any

from ragbaseline.pipeline import (
    RAGPipeline,
    LLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    LocalLLMProvider,
    PipelineConfig,
    RAGResponse,
    StreamingRAGPipeline,
)
from ragbaseline.schemas import SearchResult


class TestLLMProvider:
    """Test base LLM provider interface."""

    def test_abstract_methods(self):
        """Test that abstract methods are enforced."""
        with pytest.raises(TypeError):
            LLMProvider()


class TestOpenAIProvider:
    """Test OpenAI LLM provider."""

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client."""
        with patch.dict("sys.modules", {"openai": MagicMock()}) as modules:
            import sys
            mock_openai = sys.modules["openai"]
            client = MagicMock()
            mock_openai.AsyncOpenAI.return_value = client
            yield client

    @pytest.mark.asyncio
    async def test_generate(self, mock_openai_client):
        """Test text generation with OpenAI."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Generated response"))
        ]
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        provider = OpenAIProvider(api_key="test-key")

        messages = [{"role": "user", "content": "Hello"}]
        response = await provider.generate(messages, temperature=0.7)

        assert response == "Generated response"
        mock_openai_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_stream(self, mock_openai_client):
        """Test streaming generation with OpenAI."""
        # Mock streaming response
        async def mock_stream():
            chunks = ["Hello", " ", "world", "!"]
            for chunk in chunks:
                mock_chunk = MagicMock()
                mock_chunk.choices = [
                    MagicMock(delta=MagicMock(content=chunk))
                ]
                yield mock_chunk

        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_stream()
        )

        provider = OpenAIProvider(api_key="test-key")
        messages = [{"role": "user", "content": "Hello"}]

        chunks = []
        async for chunk in provider.generate_stream(messages):
            chunks.append(chunk)

        assert "".join(chunks) == "Hello world!"

    @pytest.mark.asyncio
    async def test_error_handling(self, mock_openai_client):
        """Test error handling in OpenAI provider."""
        mock_openai_client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )

        provider = OpenAIProvider(api_key="test-key")

        with pytest.raises(Exception, match="API Error"):
            await provider.generate([{"role": "user", "content": "test"}])


class TestRAGPipeline:
    """Test RAG pipeline implementation."""

    @pytest.fixture
    def mock_retriever(self):
        """Create mock retriever."""
        retriever = Mock()
        retriever.retrieve = AsyncMock(return_value=[
            SearchResult("doc1", 0.9, "Paris is the capital of France."),
            SearchResult("doc2", 0.8, "France is in Western Europe."),
            SearchResult("doc3", 0.7, "The Eiffel Tower is in Paris."),
        ])
        return retriever

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM provider."""
        llm = Mock(spec=LLMProvider)
        llm.generate = AsyncMock(
            return_value="Paris is the capital of France, located in Western Europe."
        )
        llm.generate_stream = AsyncMock()
        return llm

    @pytest.mark.asyncio
    async def test_basic_query(self, mock_retriever, mock_llm):
        """Test basic RAG query pipeline."""
        pipeline = RAGPipeline(
            retriever=mock_retriever,
            llm_provider=mock_llm
        )

        response = await pipeline.query("What is the capital of France?")

        assert response.answer == "Paris is the capital of France, located in Western Europe."
        assert len(response.sources) == 3
        assert response.sources[0].id == "doc1"
        assert response.query == "What is the capital of France?"

        # Verify retriever was called
        mock_retriever.retrieve.assert_called_once_with(
            "What is the capital of France?"
        )

        # Verify LLM was called with context
        mock_llm.generate.assert_called_once()
        messages = mock_llm.generate.call_args[0][0]
        assert any("Paris is the capital" in str(m) for m in messages)

    @pytest.mark.asyncio
    async def test_no_relevant_docs(self, mock_llm):
        """Test handling when no relevant documents found."""
        empty_retriever = Mock()
        empty_retriever.retrieve = AsyncMock(return_value=[])

        pipeline = RAGPipeline(
            retriever=empty_retriever,
            llm_provider=mock_llm
        )

        response = await pipeline.query("Unknown query")

        # Should still call LLM but with minimal/no context
        mock_llm.generate.assert_called_once()
        messages = mock_llm.generate.call_args[0][0]
        # The message should contain the query and have minimal context
        messages_str = str(messages).lower()
        assert "unknown query" in messages_str

    @pytest.mark.asyncio
    async def test_custom_prompt_template(self, mock_retriever, mock_llm):
        """Test using custom prompt template."""
        custom_template = """
        Context: {context}
        Question: {query}
        Provide a brief answer.
        """

        pipeline = RAGPipeline(
            retriever=mock_retriever,
            llm_provider=mock_llm,
            prompt_template=custom_template
        )

        await pipeline.query("Test question")

        # Verify custom template was used
        messages = mock_llm.generate.call_args[0][0]
        assert "Provide a brief answer" in str(messages)

    @pytest.mark.asyncio
    async def test_metadata_filtering(self, mock_retriever, mock_llm):
        """Test query with metadata filtering."""
        pipeline = RAGPipeline(
            retriever=mock_retriever,
            llm_provider=mock_llm
        )

        await pipeline.query(
            "What is the capital?",
            filter={"source": "geography"}
        )

        # Verify filter was passed to retriever
        call_args = mock_retriever.retrieve.call_args
        assert call_args[1].get("filter") == {"source": "geography"}

    @pytest.mark.asyncio
    async def test_context_size_limit(self, mock_retriever, mock_llm):
        """Test limiting context size."""
        # Create many results
        many_results = [
            SearchResult(f"doc{i}", 0.9 - i*0.01, f"Content {i}" * 100)
            for i in range(20)
        ]
        mock_retriever.retrieve = AsyncMock(return_value=many_results)

        pipeline = RAGPipeline(
            retriever=mock_retriever,
            llm_provider=mock_llm,
            max_context_tokens=500
        )

        response = await pipeline.query("Test query")

        # Should truncate to fit context limit
        messages = mock_llm.generate.call_args[0][0]
        context_str = str(messages)
        assert len(context_str) < 5000  # Approximate check

    @pytest.mark.asyncio
    async def test_response_caching(self, mock_retriever, mock_llm):
        """Test caching of responses."""
        pipeline = RAGPipeline(
            retriever=mock_retriever,
            llm_provider=mock_llm,
            enable_cache=True
        )

        # First query
        response1 = await pipeline.query("What is the capital?")
        assert mock_llm.generate.call_count == 1

        # Same query (should use cache)
        response2 = await pipeline.query("What is the capital?")
        assert mock_llm.generate.call_count == 1  # Not called again
        assert response1.answer == response2.answer

        # Different query (should not use cache)
        await pipeline.query("What is the population?")
        assert mock_llm.generate.call_count == 2


class TestStreamingRAGPipeline:
    """Test streaming RAG pipeline."""

    @pytest.fixture
    def mock_retriever(self):
        """Create mock retriever."""
        retriever = Mock()
        retriever.retrieve = AsyncMock(return_value=[
            SearchResult("doc1", 0.9, "Paris is the capital of France."),
            SearchResult("doc2", 0.8, "France is in Western Europe."),
            SearchResult("doc3", 0.7, "The Eiffel Tower is in Paris."),
        ])
        return retriever

    @pytest.fixture
    def mock_streaming_llm(self):
        """Create mock streaming LLM."""
        llm = Mock(spec=LLMProvider)

        async def mock_stream(messages, **kwargs):
            chunks = ["The ", "answer ", "is ", "42."]
            for chunk in chunks:
                yield chunk
                await asyncio.sleep(0.01)

        llm.generate_stream = mock_stream
        return llm

    @pytest.mark.asyncio
    async def test_streaming_response(self, mock_retriever, mock_streaming_llm):
        """Test streaming RAG response."""
        pipeline = StreamingRAGPipeline(
            retriever=mock_retriever,
            llm_provider=mock_streaming_llm
        )

        chunks = []
        async for chunk in pipeline.stream_query("What is the answer?"):
            chunks.append(chunk)

        assert "".join(chunks) == "The answer is 42."

    @pytest.mark.asyncio
    async def test_streaming_with_sources(self, mock_retriever, mock_streaming_llm):
        """Test that sources are included with streaming."""
        pipeline = StreamingRAGPipeline(
            retriever=mock_retriever,
            llm_provider=mock_streaming_llm,
            include_sources_in_stream=True
        )

        response_parts = []
        async for part in pipeline.stream_query("Test query"):
            response_parts.append(part)

        # Should include both chunks and sources
        text_chunks = [p for p in response_parts if isinstance(p, str)]
        source_data = [p for p in response_parts if isinstance(p, dict)]

        assert len(text_chunks) > 0
        assert len(source_data) > 0  # Sources included


class TestPipelineConfig:
    """Test pipeline configuration."""

    def test_config_creation(self):
        """Test creating pipeline config."""
        config = PipelineConfig(
            retriever_config={"k": 5},
            llm_config={"model": "gpt-4"},
            max_context_tokens=2000,
            enable_cache=True
        )

        assert config.retriever_config["k"] == 5
        assert config.llm_config["model"] == "gpt-4"
        assert config.max_context_tokens == 2000
        assert config.enable_cache is True

    def test_config_validation(self):
        """Test config validation."""
        # Invalid max tokens
        with pytest.raises(ValueError):
            PipelineConfig(max_context_tokens=-1)

        # Invalid temperature
        with pytest.raises(ValueError):
            PipelineConfig(llm_config={"temperature": 2.5})

    @pytest.mark.skipif(
        not __import__("importlib.util", fromlist=["find_spec"]).find_spec("yaml"),
        reason="pyyaml not installed"
    )
    def test_config_from_file(self, tmp_path):
        """Test loading config from file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
        retriever_config:
          k: 10
          score_threshold: 0.7
        llm_config:
          model: gpt-3.5-turbo
          temperature: 0.5
        max_context_tokens: 1500
        """)

        config = PipelineConfig.from_file(config_file)

        assert config.retriever_config["k"] == 10
        assert config.llm_config["temperature"] == 0.5


class TestRAGResponse:
    """Test RAG response data structure."""

    def test_response_creation(self):
        """Test creating RAG response."""
        sources = [
            SearchResult("doc1", 0.9, "Content 1"),
            SearchResult("doc2", 0.8, "Content 2"),
        ]

        response = RAGResponse(
            query="Test query",
            answer="Test answer",
            sources=sources,
            metadata={"time": 0.5}
        )

        assert response.query == "Test query"
        assert response.answer == "Test answer"
        assert len(response.sources) == 2
        assert response.metadata["time"] == 0.5

    def test_response_serialization(self):
        """Test serializing RAG response."""
        sources = [SearchResult("doc1", 0.9, "Content")]
        response = RAGResponse(
            query="Q",
            answer="A",
            sources=sources
        )

        # To dict
        response_dict = response.to_dict()
        assert response_dict["query"] == "Q"
        assert response_dict["answer"] == "A"
        assert len(response_dict["sources"]) == 1

        # To JSON
        json_str = response.to_json()
        assert "query" in json_str
        assert "answer" in json_str


class TestIntegrationScenarios:
    """Test end-to-end integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_reranking(self):
        """Test complete pipeline with reranking."""
        # Setup mock components
        retriever = Mock()
        retriever.retrieve = AsyncMock(return_value=[
            SearchResult("doc1", 0.6, "Initial result 1"),
            SearchResult("doc2", 0.5, "Initial result 2"),
        ])

        reranker = Mock()
        reranker.rerank = AsyncMock(return_value=[
            SearchResult("doc2", 0.9, "Initial result 2"),
            SearchResult("doc1", 0.7, "Initial result 1"),
        ])

        llm = Mock(spec=LLMProvider)
        llm.generate = AsyncMock(return_value="Reranked answer")

        # Create pipeline with reranking
        pipeline = RAGPipeline(
            retriever=retriever,
            llm_provider=llm,
            reranker=reranker
        )

        response = await pipeline.query("Test query")

        assert response.answer == "Reranked answer"
        # Sources should be reranked
        assert response.sources[0].id == "doc2"
        assert response.sources[0].score == 0.9

    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test pipeline error recovery."""
        # Retriever that fails first time
        retriever = Mock()
        retriever.retrieve = AsyncMock(
            side_effect=[Exception("Network error"), []]
        )

        llm = Mock(spec=LLMProvider)
        llm.generate = AsyncMock(return_value="Fallback answer")

        pipeline = RAGPipeline(
            retriever=retriever,
            llm_provider=llm,
            retry_on_error=True
        )

        # Should retry and use fallback
        response = await pipeline.query("Test query")

        assert response.answer == "Fallback answer"
        assert retriever.retrieve.call_count == 2  # Initial + retry


if __name__ == "__main__":
    pytest.main([__file__, "-v"])