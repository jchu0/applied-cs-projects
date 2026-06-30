"""Tests for utility modules: caching, batch processing, monitoring."""

import asyncio
import time
import numpy as np

from advancedrag import (
    LRUCache,
    TTLCache,
    RAGCacheManager,
    BatchProcessor,
    MetricsCollector,
    get_collector,
    parallel_map,
)
from advancedrag.utils.batch import AsyncBatcher, chunked_parallel


# ---------------------------------------------------------------------------
# LRUCache
# ---------------------------------------------------------------------------

class TestLRUCache:
    """Tests for LRUCache."""

    def test_get_set(self):
        cache = LRUCache(max_size=10)
        cache.set("a", 1)
        assert cache.get("a") == 1

    def test_get_missing_returns_none(self):
        cache = LRUCache()
        assert cache.get("missing") is None

    def test_eviction_on_overflow(self):
        cache = LRUCache(max_size=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_lru_ordering(self):
        """Accessing an item makes it most-recently-used."""
        cache = LRUCache(max_size=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.get("a")  # touch a -> b is now LRU
        cache.set("c", 3)
        assert cache.get("b") is None
        assert cache.get("a") == 1

    def test_overwrite_existing_key(self):
        cache = LRUCache(max_size=5)
        cache.set("k", "old")
        cache.set("k", "new")
        assert cache.get("k") == "new"
        assert len(cache) == 1

    def test_delete(self):
        cache = LRUCache()
        cache.set("x", 42)
        cache.delete("x")
        assert cache.get("x") is None

    def test_clear(self):
        cache = LRUCache()
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert len(cache) == 0

    def test_contains(self):
        cache = LRUCache()
        cache.set("k", 1)
        assert "k" in cache
        assert "missing" not in cache

    def test_len(self):
        cache = LRUCache()
        assert len(cache) == 0
        cache.set("a", 1)
        cache.set("b", 2)
        assert len(cache) == 2


# ---------------------------------------------------------------------------
# TTLCache
# ---------------------------------------------------------------------------

class TestTTLCache:
    """Tests for TTLCache."""

    def test_get_set(self):
        cache = TTLCache(default_ttl=60)
        cache.set("k", "v")
        assert cache.get("k") == "v"

    def test_expired_entry_returns_none(self):
        cache = TTLCache(default_ttl=60)
        cache.set("k", "v", ttl=1)
        # Manually expire the entry
        cache._cache["k"] = (cache._cache["k"][0], time.time() - 1)
        assert cache.get("k") is None

    def test_custom_ttl(self):
        cache = TTLCache(default_ttl=1)
        cache.set("k", "v", ttl=3600)
        assert cache.get("k") == "v"

    def test_contains_checks_expiry(self):
        cache = TTLCache(default_ttl=60)
        cache.set("k", "v", ttl=1)
        # Manually expire the entry
        cache._cache["k"] = (cache._cache["k"][0], time.time() - 1)
        assert "k" not in cache

    def test_eviction_on_overflow(self):
        cache = TTLCache(max_size=2, default_ttl=3600)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        # One of a/b should be evicted (oldest by expiry)
        remaining = sum(1 for k in ["a", "b", "c"] if cache.get(k) is not None)
        assert remaining <= 2


# ---------------------------------------------------------------------------
# RAGCacheManager
# ---------------------------------------------------------------------------

class TestRAGCacheManager:
    """Tests for RAGCacheManager."""

    def test_stats_initial(self):
        mgr = RAGCacheManager()
        stats = mgr.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["total_requests"] == 0

    def test_get_or_compute_embedding_miss_then_hit(self):
        async def run():
            class MockEmbedder:
                async def embed(self, text):
                    return np.array([1.0, 2.0, 3.0], dtype=np.float32)

            mgr = RAGCacheManager()
            embedder = MockEmbedder()

            # First call = miss
            result = await mgr.get_or_compute_embedding("hello", embedder)
            assert mgr.stats["misses"] == 1

            # Second call = L1 hit
            result2 = await mgr.get_or_compute_embedding("hello", embedder)
            assert mgr.stats["l1_hits"] == 1
            np.testing.assert_array_equal(result, result2)

        asyncio.run(run())

    def test_get_or_compute_answer_caches(self):
        async def run():
            class MockAnswerer:
                async def generate(self, query, context):
                    return "answer"

            mgr = RAGCacheManager()
            answerer = MockAnswerer()

            await mgr.get_or_compute_answer("q", "ctx_hash", answerer, None)
            assert mgr.stats["misses"] == 1

            await mgr.get_or_compute_answer("q", "ctx_hash", answerer, None)
            assert mgr.stats["l1_hits"] == 1

        asyncio.run(run())

    def test_reset_stats(self):
        mgr = RAGCacheManager()
        mgr.stats["hits"] = 10
        mgr.reset_stats()
        assert mgr.stats["hits"] == 0

    def test_clear(self):
        mgr = RAGCacheManager()
        mgr.local_cache.set("k", "v")
        mgr.clear()
        assert mgr.local_cache.get("k") is None

    def test_hit_rate_computation(self):
        mgr = RAGCacheManager()
        mgr.stats["hits"] = 3
        mgr.stats["misses"] = 7
        stats = mgr.get_stats()
        assert stats["hit_rate"] == 0.3
        assert stats["total_requests"] == 10


# ---------------------------------------------------------------------------
# BatchProcessor
# ---------------------------------------------------------------------------

class TestBatchProcessor:
    """Tests for BatchProcessor."""

    def test_process_batches(self):
        async def run():
            processor = BatchProcessor(batch_size=3)

            async def double(batch):
                return [x * 2 for x in batch]

            result = await processor.process([1, 2, 3, 4, 5], double)
            assert result == [2, 4, 6, 8, 10]

        asyncio.run(run())

    def test_empty_input(self):
        async def run():
            processor = BatchProcessor()
            result = await processor.process([], lambda x: x)
            assert result == []

        asyncio.run(run())

    def test_process_with_index(self):
        async def run():
            processor = BatchProcessor()

            async def label(idx, item):
                return f"{idx}:{item}"

            result = await processor.process_with_index(["a", "b", "c"], label)
            assert result == ["0:a", "1:b", "2:c"]

        asyncio.run(run())


# ---------------------------------------------------------------------------
# AsyncBatcher
# ---------------------------------------------------------------------------

class TestAsyncBatcher:
    """Tests for AsyncBatcher."""

    def test_batch_full_triggers_processing(self):
        async def run():
            async def process_fn(items):
                return [x.upper() for x in items]

            batcher = AsyncBatcher(batch_size=2, process_fn=process_fn)

            # Submit 2 items to fill the batch
            results = await asyncio.gather(
                batcher.submit("a"),
                batcher.submit("b"),
            )
            assert set(results) == {"A", "B"}

        asyncio.run(run())

    def test_flush_processes_partial_batch(self):
        async def run():
            async def process_fn(items):
                return [x * 2 for x in items]

            batcher = AsyncBatcher(batch_size=100, process_fn=process_fn, timeout=10)

            # Submit one item (won't fill batch)
            task = asyncio.create_task(batcher.submit(5))
            await asyncio.sleep(0.01)
            await batcher.flush()
            result = await task
            assert result == 10

        asyncio.run(run())


# ---------------------------------------------------------------------------
# parallel_map / chunked_parallel
# ---------------------------------------------------------------------------

class TestParallelMap:
    """Tests for parallel_map and chunked_parallel."""

    def test_parallel_map(self):
        async def run():
            async def square(x):
                return x ** 2

            result = await parallel_map([1, 2, 3, 4], square, max_concurrent=2)
            assert list(result) == [1, 4, 9, 16]

        asyncio.run(run())

    def test_chunked_parallel(self):
        async def run():
            async def process_chunk(chunk):
                return [x + 1 for x in chunk]

            result = await chunked_parallel([10, 20, 30], process_chunk, chunk_size=2)
            assert result == [11, 21, 31]

        asyncio.run(run())


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------

class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_get_collector(self):
        collector = get_collector("tenant-1")
        assert isinstance(collector, MetricsCollector)
        assert collector.tenant_id == "tenant-1"

    def test_track_request_success(self):
        collector = MetricsCollector("t1")
        with collector.track_request():
            pass  # no error

    def test_track_request_error_propagates(self):
        collector = MetricsCollector("t1")
        try:
            with collector.track_request():
                raise ValueError("boom")
        except ValueError:
            pass  # expected

    def test_track_retrieval(self):
        collector = MetricsCollector()
        with collector.track_retrieval("bm25"):
            pass

    def test_track_reranking(self):
        collector = MetricsCollector()
        with collector.track_reranking("cross-encoder"):
            pass

    def test_track_generation(self):
        collector = MetricsCollector()
        with collector.track_generation():
            pass

    def test_record_methods(self):
        collector = MetricsCollector()
        collector.record_retrieval_count(10)
        collector.record_confidence(0.9)
        collector.record_citations(3)
        collector.record_hallucination("low")
        collector.record_error("timeout", "retrieval")
        collector.update_cache_metrics("l1", 0.8, 500)
