"""Batch processing utilities for RAG pipeline."""

import asyncio
from typing import TypeVar, Callable, Awaitable

T = TypeVar("T")
R = TypeVar("R")


class BatchProcessor:
    """Process items in batches for efficiency."""

    def __init__(
        self,
        batch_size: int = 32,
        max_concurrent: int = 4,
    ):
        """Initialize batch processor.

        Args:
            batch_size: Items per batch
            max_concurrent: Maximum concurrent batches
        """
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def process(
        self,
        items: list[T],
        process_fn: Callable[[list[T]], Awaitable[list[R]]],
    ) -> list[R]:
        """Process items in batches.

        Args:
            items: Items to process
            process_fn: Async function to process a batch

        Returns:
            Processed results in order
        """
        if not items:
            return []

        # Create batches
        batches = [
            items[i:i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]

        # Process batches concurrently
        async def process_batch(batch: list[T]) -> list[R]:
            async with self.semaphore:
                return await process_fn(batch)

        results = await asyncio.gather(*[
            process_batch(batch) for batch in batches
        ])

        # Flatten results
        return [item for batch_result in results for item in batch_result]

    async def process_with_index(
        self,
        items: list[T],
        process_fn: Callable[[int, T], Awaitable[R]],
    ) -> list[R]:
        """Process items individually with index tracking.

        Args:
            items: Items to process
            process_fn: Async function taking (index, item)

        Returns:
            Processed results in order
        """
        if not items:
            return []

        async def process_item(idx: int, item: T) -> tuple[int, R]:
            async with self.semaphore:
                result = await process_fn(idx, item)
                return (idx, result)

        # Process all items
        indexed_results = await asyncio.gather(*[
            process_item(i, item) for i, item in enumerate(items)
        ])

        # Sort by index and extract results
        sorted_results = sorted(indexed_results, key=lambda x: x[0])
        return [r for _, r in sorted_results]


class AsyncBatcher:
    """Accumulate items and process in batches when threshold reached."""

    def __init__(
        self,
        batch_size: int,
        process_fn: Callable[[list[T]], Awaitable[list[R]]],
        timeout: float = 1.0,
    ):
        """Initialize async batcher.

        Args:
            batch_size: Target batch size
            process_fn: Function to process batches
            timeout: Max wait time before processing partial batch
        """
        self.batch_size = batch_size
        self.process_fn = process_fn
        self.timeout = timeout

        self._queue: list[tuple[T, asyncio.Future]] = []
        self._lock = asyncio.Lock()
        self._timer_task: asyncio.Task | None = None

    async def submit(self, item: T) -> R:
        """Submit item for batched processing.

        Args:
            item: Item to process

        Returns:
            Processing result
        """
        future: asyncio.Future = asyncio.Future()

        async with self._lock:
            self._queue.append((item, future))

            # Process if batch is full
            if len(self._queue) >= self.batch_size:
                await self._process_queue()
            else:
                # Start timer if not running
                if self._timer_task is None or self._timer_task.done():
                    self._timer_task = asyncio.create_task(self._timer())

        return await future

    async def _timer(self):
        """Process partial batch after timeout."""
        await asyncio.sleep(self.timeout)
        async with self._lock:
            if self._queue:
                await self._process_queue()

    async def _process_queue(self):
        """Process all items in queue."""
        if not self._queue:
            return

        # Get items and futures
        items = [item for item, _ in self._queue]
        futures = [future for _, future in self._queue]
        self._queue = []

        # Cancel timer
        if self._timer_task:
            self._timer_task.cancel()
            self._timer_task = None

        try:
            # Process batch
            results = await self.process_fn(items)

            # Resolve futures
            for future, result in zip(futures, results):
                if not future.done():
                    future.set_result(result)
        except Exception as e:
            # Propagate error to all futures
            for future in futures:
                if not future.done():
                    future.set_exception(e)

    async def flush(self):
        """Process any remaining items."""
        async with self._lock:
            await self._process_queue()


class EmbeddingBatcher(AsyncBatcher):
    """Specialized batcher for embeddings."""

    def __init__(self, embedder, batch_size: int = 32, timeout: float = 0.1):
        """Initialize embedding batcher.

        Args:
            embedder: Embedding model with batch_embed method
            batch_size: Target batch size
            timeout: Max wait time
        """
        self.embedder = embedder

        async def embed_batch(texts: list[str]):
            return await embedder.batch_embed(texts)

        super().__init__(batch_size, embed_batch, timeout)

    async def embed(self, text: str):
        """Get embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        return await self.submit(text)


async def parallel_map(
    items: list[T],
    fn: Callable[[T], Awaitable[R]],
    max_concurrent: int = 10,
) -> list[R]:
    """Map function over items with concurrency limit.

    Args:
        items: Items to process
        fn: Async function to apply
        max_concurrent: Maximum concurrent executions

    Returns:
        Results in order
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_fn(item: T) -> R:
        async with semaphore:
            return await fn(item)

    return await asyncio.gather(*[limited_fn(item) for item in items])


async def chunked_parallel(
    items: list[T],
    fn: Callable[[list[T]], Awaitable[list[R]]],
    chunk_size: int = 100,
    max_concurrent: int = 4,
) -> list[R]:
    """Process items in chunks with parallelism.

    Args:
        items: Items to process
        fn: Async function taking a chunk
        chunk_size: Items per chunk
        max_concurrent: Maximum concurrent chunks

    Returns:
        Flattened results
    """
    processor = BatchProcessor(chunk_size, max_concurrent)
    return await processor.process(items, fn)
