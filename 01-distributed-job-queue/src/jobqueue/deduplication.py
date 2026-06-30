"""Task deduplication layer for idempotent task processing."""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, TypeVar

import structlog

logger = structlog.get_logger()

T = TypeVar("T")


@dataclass
class DeduplicationEntry:
    """Entry in the deduplication store."""
    task_id: str
    created_at: float
    expires_at: Optional[float] = None


class DeduplicationStore(ABC):
    """Abstract base class for deduplication storage backends."""

    @abstractmethod
    async def get(self, key: str) -> Optional[DeduplicationEntry]:
        """Get deduplication entry by key."""
        pass

    @abstractmethod
    async def set(
        self, key: str, entry: DeduplicationEntry, ttl_seconds: Optional[float] = None
    ) -> None:
        """Set deduplication entry with optional TTL."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete deduplication entry by key."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if deduplication entry exists."""
        pass

    @abstractmethod
    async def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count of removed entries."""
        pass


class InMemoryDeduplicationStore(DeduplicationStore):
    """In-memory deduplication store for development and testing."""

    def __init__(self):
        self._entries: dict[str, DeduplicationEntry] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[DeduplicationEntry]:
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None

            # Check expiration
            if entry.expires_at and time.time() > entry.expires_at:
                del self._entries[key]
                return None

            return entry

    async def set(
        self, key: str, entry: DeduplicationEntry, ttl_seconds: Optional[float] = None
    ) -> None:
        async with self._lock:
            if ttl_seconds:
                entry.expires_at = time.time() + ttl_seconds
            self._entries[key] = entry

    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self._entries:
                del self._entries[key]
                return True
            return False

    async def exists(self, key: str) -> bool:
        entry = await self.get(key)
        return entry is not None

    async def cleanup_expired(self) -> int:
        async with self._lock:
            now = time.time()
            expired_keys = [
                key
                for key, entry in self._entries.items()
                if entry.expires_at and entry.expires_at < now
            ]
            for key in expired_keys:
                del self._entries[key]
            return len(expired_keys)

    @property
    def size(self) -> int:
        """Get current number of entries."""
        return len(self._entries)


class DeduplicationLayer:
    """
    High-level deduplication layer for idempotent task processing.

    Provides methods to:
    - Check if a task is a duplicate
    - Register new idempotency keys
    - Get or create tasks atomically
    - Handle TTL-based expiration

    Example usage:
        dedup = DeduplicationLayer(store, default_ttl_seconds=3600)

        # Check for duplicates
        if await dedup.is_duplicate("my-key"):
            existing_id = await dedup.get_task_id("my-key")
            return existing_task_by_id(existing_id)

        # Or use atomic get_or_create
        task, is_new = await dedup.get_or_create(
            "my-key",
            factory=lambda: create_new_task()
        )
    """

    def __init__(
        self,
        store: Optional[DeduplicationStore] = None,
        default_ttl_seconds: float = 3600,  # 1 hour default
        cleanup_interval_seconds: float = 300,  # 5 minutes
    ):
        """
        Initialize deduplication layer.

        Args:
            store: Backend store (defaults to InMemoryDeduplicationStore)
            default_ttl_seconds: Default TTL for idempotency keys
            cleanup_interval_seconds: Interval for expired entry cleanup
        """
        self._store = store or InMemoryDeduplicationStore()
        self._default_ttl = default_ttl_seconds
        self._cleanup_interval = cleanup_interval_seconds
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None

        logger.info(
            "DeduplicationLayer initialized",
            default_ttl=default_ttl_seconds,
            cleanup_interval=cleanup_interval_seconds,
        )

    async def start_cleanup_task(self) -> None:
        """Start background cleanup task for expired entries."""
        if self._cleanup_task is not None:
            return

        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(self._cleanup_interval)
                    removed = await self._store.cleanup_expired()
                    if removed > 0:
                        logger.debug("Cleaned up expired dedup entries", count=removed)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("Error in dedup cleanup", error=str(e))

        self._cleanup_task = asyncio.create_task(cleanup_loop())

    async def stop_cleanup_task(self) -> None:
        """Stop background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

    async def is_duplicate(self, idempotency_key: str) -> bool:
        """
        Check if an idempotency key already exists.

        Args:
            idempotency_key: The unique key to check

        Returns:
            True if key exists (duplicate), False otherwise
        """
        return await self._store.exists(idempotency_key)

    async def get_task_id(self, idempotency_key: str) -> Optional[str]:
        """
        Get the task ID associated with an idempotency key.

        Args:
            idempotency_key: The unique key

        Returns:
            Task ID if found, None otherwise
        """
        entry = await self._store.get(idempotency_key)
        return entry.task_id if entry else None

    async def register(
        self,
        idempotency_key: str,
        task_id: str,
        ttl_seconds: Optional[float] = None,
    ) -> bool:
        """
        Register a new idempotency key with a task ID.

        Args:
            idempotency_key: The unique key
            task_id: The task ID to associate
            ttl_seconds: Optional TTL override

        Returns:
            True if registered, False if key already exists
        """
        async with self._lock:
            if await self._store.exists(idempotency_key):
                return False

            entry = DeduplicationEntry(
                task_id=task_id,
                created_at=time.time(),
            )
            await self._store.set(
                idempotency_key, entry, ttl_seconds or self._default_ttl
            )
            logger.debug(
                "Registered idempotency key",
                key=idempotency_key,
                task_id=task_id,
            )
            return True

    async def unregister(self, idempotency_key: str) -> bool:
        """
        Remove an idempotency key.

        Args:
            idempotency_key: The key to remove

        Returns:
            True if removed, False if not found
        """
        return await self._store.delete(idempotency_key)

    async def get_or_create(
        self,
        idempotency_key: str,
        factory: Callable[[], T],
        task_id_extractor: Callable[[T], str],
        ttl_seconds: Optional[float] = None,
    ) -> tuple[Optional[str], bool]:
        """
        Atomically get existing task ID or create new task.

        Args:
            idempotency_key: The unique key
            factory: Function to create new task if not duplicate
            task_id_extractor: Function to extract task ID from factory result
            ttl_seconds: Optional TTL override

        Returns:
            Tuple of (task_id, is_new) where is_new is True if factory was called
        """
        async with self._lock:
            # Check for existing
            entry = await self._store.get(idempotency_key)
            if entry:
                logger.debug(
                    "Duplicate detected",
                    key=idempotency_key,
                    existing_task_id=entry.task_id,
                )
                return entry.task_id, False

            # Create new
            result = factory()
            task_id = task_id_extractor(result)

            new_entry = DeduplicationEntry(
                task_id=task_id,
                created_at=time.time(),
            )
            await self._store.set(
                idempotency_key, new_entry, ttl_seconds or self._default_ttl
            )

            logger.debug(
                "Created new task",
                key=idempotency_key,
                task_id=task_id,
            )
            return task_id, True

    async def extend_ttl(
        self, idempotency_key: str, additional_seconds: float
    ) -> bool:
        """
        Extend the TTL of an existing idempotency key.

        Args:
            idempotency_key: The key to extend
            additional_seconds: Additional time to add

        Returns:
            True if extended, False if key not found
        """
        async with self._lock:
            entry = await self._store.get(idempotency_key)
            if not entry:
                return False

            if entry.expires_at:
                entry.expires_at += additional_seconds
            else:
                entry.expires_at = time.time() + additional_seconds

            await self._store.set(idempotency_key, entry)
            return True

    async def get_stats(self) -> dict:
        """
        Get deduplication statistics.

        Returns:
            Dictionary with statistics
        """
        if isinstance(self._store, InMemoryDeduplicationStore):
            return {
                "entries": self._store.size,
                "default_ttl": self._default_ttl,
                "cleanup_interval": self._cleanup_interval,
            }
        return {
            "default_ttl": self._default_ttl,
            "cleanup_interval": self._cleanup_interval,
        }
