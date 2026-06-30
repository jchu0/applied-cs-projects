"""Embedding models for text vectorization."""

import asyncio
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Protocol, Optional
import numpy as np


class EmbeddingModel(Protocol):
    """Protocol for embedding models."""

    dimension: int

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        ...


class EmbeddingProvider(ABC):
    """Abstract base class for async embedding providers."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass

    @abstractmethod
    async def embed(self, text: str) -> np.ndarray:
        """Embed single text."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed multiple texts."""
        pass


class OpenAIEmbedder(EmbeddingProvider):
    """Async OpenAI embedding provider."""

    def __init__(
        self,
        model: str = "text-embedding-ada-002",
        api_key: str = None,
    ):
        try:
            import openai
            self._openai = openai
        except ImportError:
            raise ImportError("openai required. Install with: pip install openai")

        if api_key:
            self.client = openai.AsyncOpenAI(api_key=api_key)
        else:
            self.client = openai.AsyncOpenAI()

        self.model = model
        # Dimensions for OpenAI models
        if "small" in model or "ada" in model:
            self._dimension = 1536
        elif "large" in model:
            self._dimension = 3072
        else:
            self._dimension = 1536

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> np.ndarray:
        """Embed single text."""
        response = await self.client.embeddings.create(
            input=[text],
            model=self.model,
        )
        return np.array(response.data[0].embedding)

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed multiple texts."""
        response = await self.client.embeddings.create(
            input=texts,
            model=self.model,
        )
        return [np.array(item.embedding) for item in response.data]


class LocalEmbedder(EmbeddingProvider):
    """Local embedding provider using sentence-transformers."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        normalize: bool = False,
    ):
        try:
            from sentence_transformers import SentenceTransformer
            self._SentenceTransformer = SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers required. "
                "Install with: pip install sentence-transformers"
            )

        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()
        self.normalize = normalize

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> np.ndarray:
        """Embed single text."""
        embedding = self.model.encode([text])[0]
        if self.normalize:
            embedding = embedding / np.linalg.norm(embedding)
        return embedding

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed multiple texts."""
        embeddings = self.model.encode(texts)
        if self.normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return [emb for emb in embeddings]


class CachedEmbedder(EmbeddingProvider):
    """Caching wrapper for embedding providers."""

    def __init__(self, base_embedder: EmbeddingProvider, cache_size: int = 1000):
        self.base_embedder = base_embedder
        self.cache_size = cache_size
        self._cache: OrderedDict = OrderedDict()

    @property
    def dimension(self) -> int:
        return self.base_embedder.dimension

    async def embed(self, text: str) -> np.ndarray:
        """Embed with caching."""
        if text in self._cache:
            # Move to end (LRU)
            self._cache.move_to_end(text)
            return self._cache[text]

        embedding = await self.base_embedder.embed(text)
        self._cache[text] = embedding

        # Evict if over capacity
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

        return embedding

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed batch with caching."""
        results = [None] * len(texts)
        uncached_texts = []
        uncached_indices = []

        # Check cache
        for i, text in enumerate(texts):
            if text in self._cache:
                self._cache.move_to_end(text)
                results[i] = self._cache[text]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Embed uncached texts
        if uncached_texts:
            new_embeddings = await self.base_embedder.embed_batch(uncached_texts)
            for idx, text, emb in zip(uncached_indices, uncached_texts, new_embeddings):
                self._cache[text] = emb
                results[idx] = emb

                # Evict if over capacity
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)

        return results


class BatchedEmbedder(EmbeddingProvider):
    """Batching wrapper for embedding providers."""

    def __init__(
        self,
        base_embedder: EmbeddingProvider,
        batch_size: int = 32,
        batch_timeout: float = 0.1,
    ):
        self.base_embedder = base_embedder
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout

        self._pending: list = []
        self._lock = asyncio.Lock()
        self._batch_event = asyncio.Event()

    @property
    def dimension(self) -> int:
        return self.base_embedder.dimension

    async def embed(self, text: str) -> np.ndarray:
        """Embed single text with batching."""
        future = asyncio.Future()

        async with self._lock:
            self._pending.append((text, future))

            if len(self._pending) >= self.batch_size:
                await self._process_batch()
            else:
                # Schedule batch processing after timeout
                asyncio.create_task(self._schedule_batch())

        return await future

    async def _schedule_batch(self):
        """Schedule batch processing after timeout."""
        await asyncio.sleep(self.batch_timeout)
        async with self._lock:
            if self._pending:
                await self._process_batch()

    async def _process_batch(self):
        """Process pending batch."""
        if not self._pending:
            return

        batch = self._pending[:self.batch_size]
        self._pending = self._pending[self.batch_size:]

        texts = [item[0] for item in batch]
        futures = [item[1] for item in batch]

        try:
            embeddings = await self.base_embedder.embed_batch(texts)
            for future, embedding in zip(futures, embeddings):
                if not future.done():
                    future.set_result(embedding)
        except Exception as e:
            for future in futures:
                if not future.done():
                    future.set_exception(e)

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed batch directly."""
        return await self.base_embedder.embed_batch(texts)


class SentenceTransformerEmbedding:
    """Embedding using sentence-transformers library."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers required. "
                "Install with: pip install sentence-transformers"
            )

        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )


class OpenAIEmbedding:
    """Embedding using OpenAI API."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str = None,
    ):
        try:
            import openai
        except ImportError:
            raise ImportError("openai required. Install with: pip install openai")

        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            self.client = openai.OpenAI()

        self.model = model
        # Dimensions for OpenAI models
        if "small" in model:
            self.dimension = 1536
        elif "large" in model:
            self.dimension = 3072
        elif "ada" in model:
            self.dimension = 1536
        else:
            self.dimension = 1536

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        # Handle empty input
        if not texts:
            return np.array([])

        # OpenAI API limit is 8191 tokens per text
        # Batch if needed
        batch_size = 100
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            response = self.client.embeddings.create(
                input=batch,
                model=self.model,
            )

            embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(embeddings)

        return np.array(all_embeddings)


class HuggingFaceEmbedding:
    """Embedding using HuggingFace transformers."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = None,
    ):
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
        except ImportError:
            raise ImportError(
                "transformers and torch required. "
                "Install with: pip install transformers torch"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.to(self.device)
        self.model.eval()

        # Get dimension from model config
        self.dimension = self.model.config.hidden_size

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        import torch

        if not texts:
            return np.array([])

        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token or mean pooling
            embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token

        # Normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()


class MockEmbedding:
    """Mock embedding for testing."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._cache = {}

    def encode(self, texts: list[str]) -> np.ndarray:
        """Generate deterministic mock embeddings."""
        embeddings = []

        for text in texts:
            if text not in self._cache:
                # Generate deterministic embedding from text hash
                np.random.seed(hash(text) % (2**32))
                embedding = np.random.randn(self.dimension)
                embedding = embedding / np.linalg.norm(embedding)
                self._cache[text] = embedding

            embeddings.append(self._cache[text])

        return np.array(embeddings)


def get_embedding_model(
    model_type: str = "sentence-transformers",
    model_name: str = None,
    **kwargs,
) -> EmbeddingModel:
    """Factory function to get embedding model.

    Args:
        model_type: Type of embedding model
            - "sentence-transformers": Uses sentence-transformers library
            - "openai": Uses OpenAI API
            - "huggingface": Uses HuggingFace transformers
            - "mock": Mock embeddings for testing
        model_name: Model name/identifier
        **kwargs: Additional arguments for model

    Returns:
        Embedding model instance
    """
    if model_type == "sentence-transformers":
        return SentenceTransformerEmbedding(
            model_name=model_name or "BAAI/bge-small-en-v1.5"
        )
    elif model_type == "openai":
        return OpenAIEmbedding(
            model=model_name or "text-embedding-3-small",
            **kwargs
        )
    elif model_type == "huggingface":
        return HuggingFaceEmbedding(
            model_name=model_name or "BAAI/bge-small-en-v1.5",
            **kwargs
        )
    elif model_type == "mock":
        return MockEmbedding(**kwargs)
    else:
        raise ValueError(f"Unknown embedding model type: {model_type}")
