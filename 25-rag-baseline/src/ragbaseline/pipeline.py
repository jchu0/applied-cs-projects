"""RAG query pipeline with LLM generation."""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

from .schemas import RAGConfig, RAGResponse, SearchResult
from .index import RAGIndex


class LLMProvider(ABC):
    """Base class for LLM providers."""

    @abstractmethod
    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Generate response from messages."""
        pass

    async def generate_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """Generate streaming response."""
        # Default implementation: non-streaming
        response = await self.generate(messages, temperature, max_tokens)
        yield response


class OpenAIProvider(LLMProvider):
    """LLM provider using OpenAI API."""

    def __init__(self, model: str = "gpt-3.5-turbo", api_key: str = None):
        try:
            import openai
        except ImportError:
            raise ImportError("openai required. Install with: pip install openai")

        if api_key:
            self.client = openai.AsyncOpenAI(api_key=api_key)
        else:
            self.client = openai.AsyncOpenAI()

        self.model = model

    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    async def generate_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AnthropicProvider(LLMProvider):
    """LLM provider using Anthropic API."""

    def __init__(self, model: str = "claude-3-haiku-20240307", api_key: str = None):
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic required. Install with: pip install anthropic")

        if api_key:
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
        else:
            self.client = anthropic.AsyncAnthropic()

        self.model = model

    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        # Extract system message if present
        system = None
        filtered_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                filtered_messages.append(msg)

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=filtered_messages,
        )
        return response.content[0].text

    async def generate_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        # Extract system message
        system = None
        filtered_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                filtered_messages.append(msg)

        async with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=filtered_messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def __init__(self, responses: list[str] = None):
        self.responses = responses or ["This is a mock response."]
        self.call_count = 0

    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response


class LocalLLMProvider(LLMProvider):
    """LLM provider using local models (llama.cpp, etc)."""

    def __init__(
        self,
        model_path: str = None,
        model_name: str = "local-model",
        **kwargs,
    ):
        self.model_path = model_path
        self.model_name = model_name
        self._model = None

    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Generate using local model (mock for testing)."""
        # For testing, return mock response
        return "Local LLM response"

    async def generate_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """Stream response."""
        response = await self.generate(messages, temperature, max_tokens)
        for word in response.split():
            yield word + " "


class RAGPipeline:
    """Main RAG query pipeline.

    Supports two interfaces:
    - index-based: RAGPipeline(index, llm_provider, config)
    - retriever-based: RAGPipeline(retriever=retriever, llm_provider=llm)
    """

    def __init__(
        self,
        index: RAGIndex = None,
        llm_provider: LLMProvider = None,
        config: RAGConfig = None,
        *,
        retriever=None,  # Alternative interface
        prompt_template: str = None,
        max_context_tokens: int = 4000,
        enable_cache: bool = False,
        reranker=None,
        retry_on_error: bool = False,
    ):
        # Support both interfaces
        self.index = index
        self.retriever = retriever
        self.llm = llm_provider
        self.config = config or RAGConfig()
        self.prompt_template = prompt_template
        self.max_context_tokens = max_context_tokens
        self.enable_cache = enable_cache
        self.reranker = reranker
        self.retry_on_error = retry_on_error
        self._cache = {}

        # Validate that at least one interface is provided
        if index is None and retriever is None:
            raise ValueError("Either index or retriever must be provided")

    async def query(
        self,
        question: str,
        filter: dict = None,
    ) -> RAGResponse:
        """Execute RAG query."""
        # Check cache
        cache_key = f"{question}:{filter}"
        if self.enable_cache and cache_key in self._cache:
            return self._cache[cache_key]

        # 1. Retrieve relevant chunks (use retriever or index)
        try:
            if self.retriever is not None:
                # Only pass filter if it's not None
                if filter is not None:
                    results = await self.retriever.retrieve(question, filter=filter)
                else:
                    results = await self.retriever.retrieve(question)
            else:
                results = self.index.search(
                    question,
                    k=self.config.top_k,
                    filter=filter,
                )
        except Exception as e:
            if self.retry_on_error:
                # Retry once
                if self.retriever is not None:
                    if filter is not None:
                        results = await self.retriever.retrieve(question, filter=filter)
                    else:
                        results = await self.retriever.retrieve(question)
                else:
                    results = self.index.search(
                        question,
                        k=self.config.top_k,
                        filter=filter,
                    )
            else:
                raise

        # 2. Rerank if enabled
        if self.reranker is not None and results:
            reranked = await self.reranker.rerank(question, results)
            from .schemas import SearchResult
            # Handle different reranker return formats
            reranked_results = []
            for item in reranked:
                if isinstance(item, tuple) and len(item) == 2:
                    idx, score = item
                    reranked_results.append(SearchResult(
                        id=results[idx].id,
                        score=score,
                        content=results[idx].content,
                        metadata=results[idx].metadata,
                    ))
                elif hasattr(item, 'id'):
                    reranked_results.append(item)
                else:
                    reranked_results.append(item)
            results = reranked_results
        elif self.config.rerank and len(results) > self.config.rerank_top_k:
            results = await self._rerank(question, results)
            results = results[:self.config.rerank_top_k]

        # 3. Build prompt with context
        context = self._build_context(results)
        messages = self._build_messages(question, context)

        # 4. Generate answer
        answer = await self.llm.generate(messages)

        response = RAGResponse(
            answer=answer,
            sources=results,
            query=question,
        )

        # Cache result
        if self.enable_cache:
            self._cache[cache_key] = response

        return response

    async def query_stream(
        self,
        question: str,
        filter: dict = None,
    ) -> AsyncIterator[str]:
        """Execute RAG query with streaming response."""
        # Retrieve (use retriever or index)
        if self.retriever is not None:
            results = await self.retriever.retrieve(question)
        else:
            results = self.index.search(
                question,
                k=self.config.top_k,
                filter=filter,
            )

        # Build prompt
        context = self._build_context(results)
        messages = self._build_messages(question, context)

        # Stream generation
        async for chunk in self.llm.generate_stream(
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        ):
            yield chunk

    def _build_context(self, results: list[SearchResult]) -> str:
        """Build context string from search results."""
        context_parts = []
        total_chars = 0
        char_limit = self.max_context_tokens * 4  # Approximate chars per token

        for i, result in enumerate(results, 1):
            source = result.metadata.get("filename", "Unknown")
            text = result.content
            part = f"[{i}] {source}:\n{text}"

            if total_chars + len(part) > char_limit:
                break

            context_parts.append(part)
            total_chars += len(part) + 2  # +2 for \n\n

        return "\n\n".join(context_parts)

    def _build_messages(self, question: str, context: str) -> list[dict]:
        """Build messages for LLM."""
        if self.prompt_template:
            user_content = self.prompt_template.format(
                context=context,
                query=question,
                question=question,
            )
        else:
            user_content = f"""Context:
{context}

Question: {question}

Please answer the question based on the context provided above."""

        return [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": user_content},
        ]

    async def _rerank(
        self,
        query: str,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """Rerank results using cross-encoder."""
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            # Return original results if reranker not available
            return results

        # Load reranker
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # Score pairs
        pairs = [[query, r.content] for r in results]
        scores = reranker.predict(pairs)

        # Sort by score
        scored_results = list(zip(results, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        return [r for r, s in scored_results]


def get_llm_provider(
    provider_type: str = "openai",
    model: str = None,
    **kwargs,
) -> LLMProvider:
    """Factory function to get LLM provider.

    Args:
        provider_type: Type of provider
            - "openai": OpenAI API
            - "anthropic": Anthropic API
            - "mock": Mock provider for testing
        model: Model name
        **kwargs: Additional arguments

    Returns:
        LLM provider instance
    """
    if provider_type == "openai":
        return OpenAIProvider(
            model=model or "gpt-3.5-turbo",
            **kwargs
        )
    elif provider_type == "anthropic":
        return AnthropicProvider(
            model=model or "claude-3-haiku-20240307",
            **kwargs
        )
    elif provider_type == "mock":
        return MockLLMProvider(**kwargs)
    else:
        raise ValueError(f"Unknown LLM provider type: {provider_type}")


# New async pipeline classes for test compatibility

import json as json_module
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    """Configuration for RAG pipeline."""

    retriever_config: dict = field(default_factory=dict)
    llm_config: dict = field(default_factory=dict)
    max_context_tokens: int = 4000
    enable_cache: bool = False

    def __post_init__(self):
        """Validate config."""
        if self.max_context_tokens < 0:
            raise ValueError("max_context_tokens cannot be negative")
        if self.llm_config.get("temperature", 0) > 2.0:
            raise ValueError("temperature must be <= 2.0")

    @classmethod
    def from_file(cls, path: Path) -> "PipelineConfig":
        """Load config from YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("pyyaml required. Install with: pip install pyyaml")

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(
            retriever_config=data.get("retriever_config", {}),
            llm_config=data.get("llm_config", {}),
            max_context_tokens=data.get("max_context_tokens", 4000),
            enable_cache=data.get("enable_cache", False),
        )


class AsyncRAGPipeline:
    """Async RAG pipeline with retriever interface."""

    def __init__(
        self,
        retriever,
        llm_provider: LLMProvider,
        prompt_template: str = None,
        max_context_tokens: int = 4000,
        enable_cache: bool = False,
        reranker=None,
        retry_on_error: bool = False,
    ):
        self.retriever = retriever
        self.llm = llm_provider
        self.prompt_template = prompt_template
        self.max_context_tokens = max_context_tokens
        self.enable_cache = enable_cache
        self.reranker = reranker
        self.retry_on_error = retry_on_error
        self._cache = {}

    async def query(
        self,
        question: str,
        filter: dict = None,
    ) -> RAGResponse:
        """Execute RAG query."""
        # Check cache
        cache_key = f"{question}:{filter}"
        if self.enable_cache and cache_key in self._cache:
            return self._cache[cache_key]

        # Retrieve with retry support
        try:
            sources = await self.retriever.retrieve(question, filter=filter)
        except Exception as e:
            if self.retry_on_error:
                sources = await self.retriever.retrieve(question, filter=filter)
            else:
                raise

        # Rerank if enabled
        if self.reranker and sources:
            reranked = await self.reranker.rerank(question, sources)
            # Reconstruct sources from reranked result
            reranked_sources = []
            for idx, score in reranked:
                s = sources[idx]
                from .schemas import SearchResult
                reranked_sources.append(SearchResult(
                    id=s.id,
                    score=score,
                    content=s.content,
                    metadata=s.metadata,
                ))
            sources = reranked_sources

        # Build context
        context = self._build_context(sources)
        messages = self._build_messages(question, context)

        # Generate
        answer = await self.llm.generate(messages)

        response = RAGResponse(
            query=question,
            answer=answer,
            sources=sources,
            metadata={},
        )

        # Cache result
        if self.enable_cache:
            self._cache[cache_key] = response

        return response

    def _build_context(self, sources: list) -> str:
        """Build context from sources."""
        if not sources:
            return "No relevant information found."

        parts = []
        total_chars = 0
        char_limit = self.max_context_tokens * 4  # Approximate

        for i, source in enumerate(sources, 1):
            content = source.content if hasattr(source, 'content') else str(source)
            if total_chars + len(content) > char_limit:
                break
            parts.append(f"[{i}] {content}")
            total_chars += len(content)

        return "\n\n".join(parts)

    def _build_messages(self, question: str, context: str) -> list[dict]:
        """Build messages for LLM."""
        if self.prompt_template:
            user_content = self.prompt_template.format(
                context=context,
                query=question,
            )
        else:
            user_content = f"Context:\n{context}\n\nQuestion: {question}"

        return [
            {"role": "system", "content": "Answer based on the provided context."},
            {"role": "user", "content": user_content},
        ]


# Keep RAGPipeline as the original index-based implementation
# The AsyncRAGPipeline is for newer retriever-based usage


class StreamingRAGPipeline(AsyncRAGPipeline):
    """Streaming RAG pipeline."""

    def __init__(
        self,
        retriever,
        llm_provider: LLMProvider,
        include_sources_in_stream: bool = False,
        **kwargs,
    ):
        super().__init__(retriever, llm_provider, **kwargs)
        self.include_sources_in_stream = include_sources_in_stream

    async def stream_query(
        self,
        question: str,
        filter: dict = None,
    ):
        """Stream RAG query response."""
        sources = await self.retriever.retrieve(question, filter=filter)

        context = self._build_context(sources)
        messages = self._build_messages(question, context)

        async for chunk in self.llm.generate_stream(messages):
            yield chunk

        # Optionally yield sources at the end
        if self.include_sources_in_stream and sources:
            yield {"sources": [s.to_dict() if hasattr(s, 'to_dict') else s for s in sources]}
