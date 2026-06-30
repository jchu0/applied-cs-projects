"""Tests for task deduplication layer."""

import asyncio
import pytest
import time

from jobqueue.deduplication import (
    DeduplicationLayer,
    DeduplicationEntry,
    InMemoryDeduplicationStore,
)


@pytest.fixture
def store():
    """Create a fresh in-memory store for each test."""
    return InMemoryDeduplicationStore()


@pytest.fixture
def dedup(store):
    """Create a deduplication layer with the store."""
    return DeduplicationLayer(store=store, default_ttl_seconds=60)


class TestInMemoryDeduplicationStore:
    """Tests for InMemoryDeduplicationStore."""

    async def test_set_and_get(self, store):
        """Test basic set and get operations."""
        entry = DeduplicationEntry(task_id="task-123", created_at=time.time())
        await store.set("key1", entry)

        retrieved = await store.get("key1")
        assert retrieved is not None
        assert retrieved.task_id == "task-123"

    async def test_get_nonexistent(self, store):
        """Test getting a nonexistent key returns None."""
        result = await store.get("nonexistent")
        assert result is None

    async def test_exists(self, store):
        """Test exists check."""
        entry = DeduplicationEntry(task_id="task-123", created_at=time.time())
        await store.set("key1", entry)

        assert await store.exists("key1") is True
        assert await store.exists("nonexistent") is False

    async def test_delete(self, store):
        """Test delete operation."""
        entry = DeduplicationEntry(task_id="task-123", created_at=time.time())
        await store.set("key1", entry)

        assert await store.delete("key1") is True
        assert await store.exists("key1") is False
        assert await store.delete("key1") is False  # Already deleted

    async def test_ttl_expiration(self, store):
        """Test that entries expire based on TTL."""
        entry = DeduplicationEntry(task_id="task-123", created_at=time.time())
        await store.set("key1", entry, ttl_seconds=0.1)  # 100ms TTL

        # Should exist immediately
        assert await store.exists("key1") is True

        # Wait for expiration
        await asyncio.sleep(0.15)

        # Should be expired now
        assert await store.exists("key1") is False
        assert await store.get("key1") is None

    async def test_cleanup_expired(self, store):
        """Test cleanup of expired entries."""
        entry1 = DeduplicationEntry(task_id="task-1", created_at=time.time())
        entry2 = DeduplicationEntry(task_id="task-2", created_at=time.time())
        entry3 = DeduplicationEntry(task_id="task-3", created_at=time.time())

        await store.set("key1", entry1, ttl_seconds=0.05)
        await store.set("key2", entry2, ttl_seconds=0.05)
        await store.set("key3", entry3, ttl_seconds=10)  # Long TTL

        await asyncio.sleep(0.1)

        removed = await store.cleanup_expired()
        assert removed == 2

        assert await store.exists("key1") is False
        assert await store.exists("key2") is False
        assert await store.exists("key3") is True

    async def test_size(self, store):
        """Test size property."""
        assert store.size == 0

        entry = DeduplicationEntry(task_id="task-123", created_at=time.time())
        await store.set("key1", entry)
        await store.set("key2", entry)

        assert store.size == 2

        await store.delete("key1")
        assert store.size == 1


class TestDeduplicationLayer:
    """Tests for DeduplicationLayer."""

    async def test_is_duplicate_false(self, dedup):
        """Test is_duplicate returns False for new key."""
        assert await dedup.is_duplicate("new-key") is False

    async def test_is_duplicate_true(self, dedup):
        """Test is_duplicate returns True for existing key."""
        await dedup.register("existing-key", "task-123")
        assert await dedup.is_duplicate("existing-key") is True

    async def test_get_task_id(self, dedup):
        """Test getting task ID by idempotency key."""
        await dedup.register("key1", "task-123")

        task_id = await dedup.get_task_id("key1")
        assert task_id == "task-123"

    async def test_get_task_id_not_found(self, dedup):
        """Test getting task ID for nonexistent key."""
        task_id = await dedup.get_task_id("nonexistent")
        assert task_id is None

    async def test_register_new_key(self, dedup):
        """Test registering a new idempotency key."""
        result = await dedup.register("new-key", "task-123")
        assert result is True
        assert await dedup.is_duplicate("new-key") is True

    async def test_register_duplicate_key(self, dedup):
        """Test registering a duplicate key fails."""
        await dedup.register("key1", "task-123")
        result = await dedup.register("key1", "task-456")
        assert result is False

        # Original task ID should be preserved
        task_id = await dedup.get_task_id("key1")
        assert task_id == "task-123"

    async def test_unregister(self, dedup):
        """Test unregistering an idempotency key."""
        await dedup.register("key1", "task-123")

        result = await dedup.unregister("key1")
        assert result is True
        assert await dedup.is_duplicate("key1") is False

    async def test_unregister_not_found(self, dedup):
        """Test unregistering a nonexistent key."""
        result = await dedup.unregister("nonexistent")
        assert result is False

    async def test_get_or_create_new(self, dedup):
        """Test get_or_create with new key creates task."""
        task_counter = [0]

        def factory():
            task_counter[0] += 1
            return {"id": f"task-{task_counter[0]}"}

        task_id, is_new = await dedup.get_or_create(
            "key1",
            factory=factory,
            task_id_extractor=lambda t: t["id"],
        )

        assert task_id == "task-1"
        assert is_new is True
        assert task_counter[0] == 1

    async def test_get_or_create_duplicate(self, dedup):
        """Test get_or_create with existing key returns existing."""
        task_counter = [0]

        def factory():
            task_counter[0] += 1
            return {"id": f"task-{task_counter[0]}"}

        # First call creates
        task_id1, is_new1 = await dedup.get_or_create(
            "key1",
            factory=factory,
            task_id_extractor=lambda t: t["id"],
        )

        # Second call returns existing
        task_id2, is_new2 = await dedup.get_or_create(
            "key1",
            factory=factory,
            task_id_extractor=lambda t: t["id"],
        )

        assert task_id1 == task_id2
        assert is_new1 is True
        assert is_new2 is False
        assert task_counter[0] == 1  # Factory called only once

    async def test_ttl_expiration(self, dedup):
        """Test that keys expire based on TTL."""
        await dedup.register("key1", "task-123", ttl_seconds=0.1)

        assert await dedup.is_duplicate("key1") is True

        await asyncio.sleep(0.15)

        assert await dedup.is_duplicate("key1") is False

    async def test_extend_ttl(self, dedup):
        """Test extending TTL of an existing key."""
        await dedup.register("key1", "task-123", ttl_seconds=0.1)

        await asyncio.sleep(0.05)

        # Extend TTL
        result = await dedup.extend_ttl("key1", 1.0)
        assert result is True

        await asyncio.sleep(0.1)

        # Should still exist after original TTL
        assert await dedup.is_duplicate("key1") is True

    async def test_extend_ttl_not_found(self, dedup):
        """Test extending TTL of nonexistent key."""
        result = await dedup.extend_ttl("nonexistent", 1.0)
        assert result is False

    async def test_get_stats(self, dedup):
        """Test getting deduplication statistics."""
        await dedup.register("key1", "task-1")
        await dedup.register("key2", "task-2")

        stats = await dedup.get_stats()
        assert stats["entries"] == 2
        assert stats["default_ttl"] == 60
        assert "cleanup_interval" in stats

    async def test_cleanup_task_lifecycle(self, dedup):
        """Test starting and stopping cleanup task."""
        await dedup.start_cleanup_task()
        assert dedup._cleanup_task is not None

        await dedup.stop_cleanup_task()
        assert dedup._cleanup_task is None


class TestDeduplicationConcurrency:
    """Tests for concurrent deduplication operations."""

    async def test_concurrent_register(self, dedup):
        """Test concurrent registration of same key."""
        results = await asyncio.gather(*[
            dedup.register("key1", f"task-{i}")
            for i in range(10)
        ])

        # Only one should succeed
        assert sum(results) == 1

        # Should have one entry
        stats = await dedup.get_stats()
        assert stats["entries"] == 1

    async def test_concurrent_get_or_create(self, dedup):
        """Test concurrent get_or_create with same key."""
        call_count = [0]

        def factory():
            call_count[0] += 1
            return {"id": f"task-{call_count[0]}"}

        results = await asyncio.gather(*[
            dedup.get_or_create(
                "key1",
                factory=factory,
                task_id_extractor=lambda t: t["id"],
            )
            for _ in range(10)
        ])

        task_ids = [r[0] for r in results]
        is_new_flags = [r[1] for r in results]

        # All should return same task_id
        assert len(set(task_ids)) == 1

        # Only one should be marked as new
        assert sum(is_new_flags) == 1

        # Factory should only be called once
        assert call_count[0] == 1

    async def test_concurrent_register_different_keys(self, store):
        """Test concurrent registration of different keys."""
        dedup = DeduplicationLayer(store=store, default_ttl_seconds=60)

        async def register_key(i):
            return await dedup.register(f"key-{i}", f"task-{i}")

        results = await asyncio.gather(*[
            register_key(i)
            for i in range(100)
        ])

        # All should succeed
        assert all(results)

        # Should have 100 entries
        stats = await dedup.get_stats()
        assert stats["entries"] == 100


class TestDeduplicationEdgeCases:
    """Tests for edge cases in deduplication."""

    async def test_empty_key(self, dedup):
        """Test handling of empty idempotency key."""
        result = await dedup.register("", "task-123")
        assert result is True

        assert await dedup.is_duplicate("") is True
        assert await dedup.get_task_id("") == "task-123"

    async def test_special_characters_in_key(self, dedup):
        """Test handling of special characters in key."""
        special_key = "key:with/special#chars?and=values"

        result = await dedup.register(special_key, "task-123")
        assert result is True

        assert await dedup.is_duplicate(special_key) is True
        assert await dedup.get_task_id(special_key) == "task-123"

    async def test_long_key(self, dedup):
        """Test handling of very long key."""
        long_key = "x" * 10000

        result = await dedup.register(long_key, "task-123")
        assert result is True

        assert await dedup.is_duplicate(long_key) is True

    async def test_zero_ttl(self, store):
        """Test behavior with zero TTL (immediate expiration)."""
        dedup = DeduplicationLayer(store=store, default_ttl_seconds=0)

        # Should still register initially
        result = await dedup.register("key1", "task-123")
        assert result is True

        # But may be immediately expired on next check
        # (depending on timing)
        await asyncio.sleep(0.01)
        # After sleep, should be expired

    async def test_very_short_ttl(self, store):
        """Test behavior with very short TTL."""
        dedup = DeduplicationLayer(store=store, default_ttl_seconds=0.001)

        await dedup.register("key1", "task-123")

        await asyncio.sleep(0.01)

        # Should be expired
        assert await dedup.is_duplicate("key1") is False

    async def test_custom_ttl_per_key(self, dedup):
        """Test different TTL values per key."""
        await dedup.register("short-ttl", "task-1", ttl_seconds=0.1)
        await dedup.register("long-ttl", "task-2", ttl_seconds=10)

        await asyncio.sleep(0.15)

        # Short TTL should be expired
        assert await dedup.is_duplicate("short-ttl") is False

        # Long TTL should still exist
        assert await dedup.is_duplicate("long-ttl") is True


class TestDeduplicationWithBroker:
    """Integration tests with broker."""

    async def test_broker_idempotency_consistency(self, dedup):
        """Test that dedup layer matches broker behavior."""
        from jobqueue.broker import InMemoryBroker
        from jobqueue.models import Task

        broker = InMemoryBroker()

        # Use dedup layer to check before enqueuing
        idempotency_key = "unique-task-key"

        if await dedup.is_duplicate(idempotency_key):
            existing_id = await dedup.get_task_id(idempotency_key)
            task = await broker.get_task(existing_id)
        else:
            task = Task(name="test", idempotency_key=idempotency_key)
            await broker.enqueue(task)
            await dedup.register(idempotency_key, task.id)

        # Second attempt should detect duplicate
        assert await dedup.is_duplicate(idempotency_key) is True
        assert await dedup.get_task_id(idempotency_key) == task.id
