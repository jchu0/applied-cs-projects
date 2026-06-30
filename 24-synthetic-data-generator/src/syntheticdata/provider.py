"""Model provider interfaces for LLM generation."""

from abc import ABC, abstractmethod
from typing import Optional
import json
import asyncio


class ModelProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        """Generate completion from messages."""
        pass

    @abstractmethod
    async def generate_json(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> dict:
        """Generate JSON response from messages."""
        pass


class OpenAIProvider(ModelProvider):
    """OpenAI API provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        base_url: Optional[str] = None,
    ):
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        self.model = model

    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        """Generate completion using OpenAI API."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return response.choices[0].message.content

    async def generate_json(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> dict:
        """Generate JSON response using OpenAI API."""
        response = await self.generate(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            **kwargs,
        )
        return json.loads(response)


class AnthropicProvider(ModelProvider):
    """Anthropic Claude API provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-sonnet-20240229",
    ):
        try:
            from anthropic import AsyncAnthropic
            self.client = AsyncAnthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")

        self.model = model

    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        """Generate completion using Anthropic API."""
        # Extract system message if present
        system = None
        chat_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                chat_messages.append(msg)

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=chat_messages,
            temperature=temperature,
        )
        return response.content[0].text

    async def generate_json(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> dict:
        """Generate JSON response using Anthropic API."""
        # Add JSON instruction to the last message
        modified_messages = messages.copy()
        if modified_messages:
            last_msg = modified_messages[-1].copy()
            last_msg["content"] += "\n\nRespond with valid JSON only."
            modified_messages[-1] = last_msg

        response = await self.generate(
            modified_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        # Extract JSON from response
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to find JSON in response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise


class MockProvider(ModelProvider):
    """Mock provider for testing."""

    def __init__(self, responses: Optional[list[str]] = None, model: str = "mock"):
        self.responses = responses or []
        self.model = model
        self.call_count = 0
        self.calls = []

    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        """Return mock response."""
        self.calls.append({
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })

        if self.responses:
            response = self.responses[self.call_count % len(self.responses)]
        else:
            # Detect what format is needed based on the prompt
            prompt = str(messages)
            if "instruction" in prompt.lower() or "INSTRUCTION" in prompt:
                response = json.dumps({
                    "instruction": "Write a function to calculate the sum of two numbers",
                    "input": "a = 5, b = 3",
                    "output": "The sum is 8",
                    "explanation": "This is a mock instruction response.",
                })
            elif "conversation" in prompt.lower() or "CONVERSATION" in prompt:
                response = json.dumps({
                    "messages": [
                        {"role": "user", "content": "Hello, how are you?"},
                        {"role": "assistant", "content": "I'm doing well, thank you!"},
                    ],
                    "system_prompt": "You are a helpful assistant.",
                })
            else:
                # Default RAG_QA format
                response = json.dumps({
                    "question": "What is the main topic?",
                    "answer": "The main topic is synthetic data generation.",
                    "reasoning": "This is a mock response.",
                })

        self.call_count += 1
        return response

    async def generate_json(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> dict:
        """Return mock JSON response."""
        response = await self.generate(messages, temperature, max_tokens, **kwargs)
        return json.loads(response)


class RateLimitedProvider(ModelProvider):
    """Wrapper that adds rate limiting to any provider."""

    def __init__(
        self,
        provider: ModelProvider,
        requests_per_minute: int = 60,
    ):
        self.provider = provider
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0.0
        self._lock = asyncio.Lock()

    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        """Generate with rate limiting."""
        async with self._lock:
            # Wait if needed
            import time
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)

            self.last_request_time = time.time()

        return await self.provider.generate(
            messages, temperature, max_tokens, **kwargs
        )

    async def generate_json(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> dict:
        """Generate JSON with rate limiting."""
        response = await self.generate(messages, temperature, max_tokens, **kwargs)
        return json.loads(response)


# Alias for backward compatibility
MockModelProvider = MockProvider
