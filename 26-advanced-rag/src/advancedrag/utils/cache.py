"""Multi-level caching for RAG pipeline optimization."""

from typing import Optional, Any
from collections import OrderedDict
import hashlib
import json
import time
import numpy as np


class LRUCache:
    """Thread-safe LRU cache implementation."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: OrderedDict = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def set(self, key: str, value: Any):
        """Set item in cache."""
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value

        # Evict oldest if over capacity
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def delete(self, key: str):
        """Delete item from cache."""
        self._cache.pop(key, None)

    def clear(self):
        """Clear all items."""
        self._cache.clear()

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)


class TTLCache:
    """Cache with time-to-live expiration."""

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: dict[str, tuple[Any, float]] = {}

    def get(self, key: str) -> Optional[Any]:
        """Get item if not expired."""
        if key in self._cache:
            value, expiry = self._cache[key]
            if time.time() < expiry:
                return value
            # Expired, remove it
            del self._cache[key]
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set item with TTL."""
        ttl = ttl or self.default_ttl
        expiry = time.time() + ttl
        self._cache[key] = (value, expiry)

        # Evict expired and oldest if over capacity
        self._cleanup()

    def _cleanup(self):
        """Remove expired entries and enforce size limit."""
        now = time.time()

        # Remove expired
        expired = [k for k, (_, exp) in self._cache.items() if exp <= now]
        for k in expired:
            del self._cache[k]

        # Remove oldest if still over capacity
        while len(self._cache) > self.max_size:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]

    def clear(self):
        """Clear all items."""
        self._cache.clear()

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None


class RAGCacheManager:
    """Multi-level caching for RAG pipeline."""

    def __init__(
        self,
        redis_client=None,
        local_cache_size: int = 1000,
        default_ttl: int = 3600,
    ):
        """Initialize cache manager.

        Args:
            redis_client: Optional Redis client for L2 cache
            local_cache_size: Size of local L1 cache
            default_ttl: Default TTL in seconds
        """
        self.redis = redis_client
        self.local_cache = TTLCache(local_cache_size, default_ttl)
        self.default_ttl = default_ttl

        # Cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "l1_hits": 0,
            "l2_hits": 0,
        }

    def _hash(self, text: str) -> str:
        """Generate hash for cache key."""
        return hashlib.md5(text.encode()).hexdigest()

    async def get_or_compute_embedding(
        self,
        text: str,
        embedder,
        ttl: Optional[int] = None,
    ) -> np.ndarray:
        """Get embedding from cache or compute it.

        Args:
            text: Text to embed
            embedder: Embedding model
            ttl: Optional TTL override

        Returns:
            Embedding vector
        """
        cache_key = f"emb:{self._hash(text)}"

        # L1: Local cache
        cached = self.local_cache.get(cache_key)
        if cached is not None:
            self.stats["hits"] += 1
            self.stats["l1_hits"] += 1
            return cached

        # L2: Redis cache
        if self.redis:
            redis_data = await self.redis.get(cache_key)
            if redis_data:
                embedding = np.frombuffer(redis_data, dtype=np.float32)
                self.local_cache.set(cache_key, embedding, ttl)
                self.stats["hits"] += 1
                self.stats["l2_hits"] += 1
                return embedding

        # Compute embedding
        self.stats["misses"] += 1
        embedding = await embedder.embed(text)

        # Store in both caches
        self.local_cache.set(cache_key, embedding, ttl)
        if self.redis:
            await self.redis.setex(
                cache_key,
                ttl or self.default_ttl,
                embedding.tobytes()
            )

        return embedding

    async def get_or_compute_retrieval(
        self,
        query: str,
        retriever,
        top_k: int,
        ttl: int = 300,
    ) -> list:
        """Get retrieval results from cache or compute.

        Args:
            query: Search query
            retriever: Retriever to use
            top_k: Number of results
            ttl: Cache TTL

        Returns:
            Retrieval results
        """
        cache_key = f"ret:{self._hash(query)}:{top_k}"

        # L1: Local cache
        cached = self.local_cache.get(cache_key)
        if cached is not None:
            self.stats["hits"] += 1
            self.stats["l1_hits"] += 1
            return cached

        # L2: Redis cache
        if self.redis:
            redis_data = await self.redis.get(cache_key)
            if redis_data:
                results = json.loads(redis_data)
                self.local_cache.set(cache_key, results, ttl)
                self.stats["hits"] += 1
                self.stats["l2_hits"] += 1
                return results

        # Compute
        self.stats["misses"] += 1
        results = await retriever.retrieve(query, top_k)

        # Cache as serialized form
        serialized = self._serialize_results(results)
        self.local_cache.set(cache_key, results, ttl)
        if self.redis:
            await self.redis.setex(cache_key, ttl, serialized)

        return results

    async def get_or_compute_answer(
        self,
        query: str,
        context_hash: str,
        answerer,
        context,
        ttl: int = 600,
    ):
        """Get generated answer from cache or compute.

        Args:
            query: User query
            context_hash: Hash of context content
            answerer: Answer generator
            context: Constructed context
            ttl: Cache TTL

        Returns:
            Generated answer
        """
        cache_key = f"ans:{self._hash(query)}:{context_hash}"

        # L1: Local cache
        cached = self.local_cache.get(cache_key)
        if cached is not None:
            self.stats["hits"] += 1
            self.stats["l1_hits"] += 1
            return cached

        # Compute (don't cache answers in Redis - too variable)
        self.stats["misses"] += 1
        answer = await answerer.generate(query, context)
        self.local_cache.set(cache_key, answer, ttl)

        return answer

    def _serialize_results(self, results: list) -> str:
        """Serialize retrieval results for caching."""
        serialized = []
        for r in results:
            serialized.append({
                "doc_id": r.document.id,
                "content": r.document.content,
                "metadata": r.document.metadata,
                "score": r.score,
                "rank": r.rank,
            })
        return json.dumps(serialized)

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0

        return {
            **self.stats,
            "total_requests": total,
            "hit_rate": hit_rate,
            "l1_hit_rate": self.stats["l1_hits"] / total if total > 0 else 0,
        }

    def clear(self):
        """Clear all caches."""
        self.local_cache.clear()
        # Note: Redis cache not cleared - do manually if needed

    def reset_stats(self):
        """Reset cache statistics."""
        self.stats = {
            "hits": 0,
            "misses": 0,
            "l1_hits": 0,
            "l2_hits": 0,
        }
