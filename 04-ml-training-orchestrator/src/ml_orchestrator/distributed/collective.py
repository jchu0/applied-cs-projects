"""Collective communication operations for distributed training."""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Generic, Optional, TypeVar
from uuid import uuid4
import structlog


logger = structlog.get_logger(__name__)


T = TypeVar("T")


class ReductionOp(str, Enum):
    """Reduction operations for collective communications."""

    SUM = "sum"
    PRODUCT = "product"
    MIN = "min"
    MAX = "max"
    AVG = "avg"


class CollectiveStatus(str, Enum):
    """Status of a collective operation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class CollectiveResult(Generic[T]):
    """Result of a collective operation."""

    operation_id: str
    status: CollectiveStatus
    result: Optional[T] = None
    error: Optional[str] = None
    participants: list[str] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def duration_ms(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return None


class CollectiveOperation(ABC):
    """Base class for collective operations."""

    def __init__(
        self,
        operation_id: Optional[str] = None,
        timeout_seconds: float = 30.0,
    ):
        self.operation_id = operation_id or str(uuid4())
        self.timeout = timeout_seconds
        self._participants: dict[str, Any] = {}  # worker_id -> data
        self._result: Optional[Any] = None
        self._status = CollectiveStatus.PENDING
        self._lock = asyncio.Lock()
        self._complete_event = asyncio.Event()
        self._created_at = datetime.utcnow()

    @property
    def status(self) -> CollectiveStatus:
        return self._status

    @property
    def is_complete(self) -> bool:
        return self._status in (
            CollectiveStatus.COMPLETED,
            CollectiveStatus.FAILED,
            CollectiveStatus.TIMEOUT,
        )

    @abstractmethod
    async def contribute(self, worker_id: str, data: Any) -> bool:
        """Worker contributes data to the operation."""
        pass

    @abstractmethod
    async def get_result(self, worker_id: str) -> CollectiveResult:
        """Get result of the operation for a worker."""
        pass

    async def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait for operation to complete."""
        timeout = timeout or self.timeout
        try:
            await asyncio.wait_for(self._complete_event.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            self._status = CollectiveStatus.TIMEOUT
            return False


class AllReduceOp(CollectiveOperation):
    """
    AllReduce: Reduce data from all workers and distribute result to all.

    Every worker contributes data, and all receive the reduced result.
    """

    def __init__(
        self,
        world_size: int,
        reduction: ReductionOp = ReductionOp.SUM,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._world_size = world_size
        self._reduction = reduction

    async def contribute(self, worker_id: str, data: Any) -> bool:
        """Contribute data to reduction."""
        async with self._lock:
            if worker_id in self._participants:
                return False  # Already contributed

            self._participants[worker_id] = data

            if self._status == CollectiveStatus.PENDING:
                self._status = CollectiveStatus.IN_PROGRESS

            # Check if all participants have contributed
            if len(self._participants) >= self._world_size:
                await self._compute_result()
                self._complete_event.set()

            return True

    async def _compute_result(self) -> None:
        """Compute the reduced result."""
        values = list(self._participants.values())

        if not values:
            self._status = CollectiveStatus.FAILED
            return

        try:
            if self._reduction == ReductionOp.SUM:
                self._result = sum(values)
            elif self._reduction == ReductionOp.PRODUCT:
                result = values[0]
                for v in values[1:]:
                    result = result * v
                self._result = result
            elif self._reduction == ReductionOp.MIN:
                self._result = min(values)
            elif self._reduction == ReductionOp.MAX:
                self._result = max(values)
            elif self._reduction == ReductionOp.AVG:
                self._result = sum(values) / len(values)

            self._status = CollectiveStatus.COMPLETED
            logger.debug(
                "allreduce_complete",
                op_id=self.operation_id,
                participants=len(self._participants),
                reduction=self._reduction.value,
            )

        except Exception as e:
            self._status = CollectiveStatus.FAILED
            logger.error("allreduce_failed", op_id=self.operation_id, error=str(e))

    async def get_result(self, worker_id: str) -> CollectiveResult:
        """Get result (same for all workers in AllReduce)."""
        return CollectiveResult(
            operation_id=self.operation_id,
            status=self._status,
            result=self._result,
            participants=list(self._participants.keys()),
            started_at=self._created_at,
            completed_at=datetime.utcnow() if self.is_complete else None,
        )


class AllGatherOp(CollectiveOperation):
    """
    AllGather: Gather data from all workers and distribute to all.

    Each worker contributes unique data, all receive list of all data.
    """

    def __init__(self, world_size: int, **kwargs: Any):
        super().__init__(**kwargs)
        self._world_size = world_size

    async def contribute(self, worker_id: str, data: Any) -> bool:
        """Contribute data to gather."""
        async with self._lock:
            if worker_id in self._participants:
                return False

            self._participants[worker_id] = data

            if self._status == CollectiveStatus.PENDING:
                self._status = CollectiveStatus.IN_PROGRESS

            if len(self._participants) >= self._world_size:
                self._result = list(self._participants.values())
                self._status = CollectiveStatus.COMPLETED
                self._complete_event.set()
                logger.debug(
                    "allgather_complete",
                    op_id=self.operation_id,
                    participants=len(self._participants),
                )

            return True

    async def get_result(self, worker_id: str) -> CollectiveResult:
        """Get gathered results (same for all workers)."""
        return CollectiveResult(
            operation_id=self.operation_id,
            status=self._status,
            result=self._result,
            participants=list(self._participants.keys()),
            started_at=self._created_at,
            completed_at=datetime.utcnow() if self.is_complete else None,
        )


class BroadcastOp(CollectiveOperation):
    """
    Broadcast: Distribute data from root to all workers.

    Only root contributes data, all workers receive it.
    """

    def __init__(
        self,
        world_size: int,
        root_worker_id: str,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._world_size = world_size
        self._root_id = root_worker_id
        self._receivers_ready: set[str] = set()

    async def contribute(self, worker_id: str, data: Any) -> bool:
        """Root contributes data to broadcast."""
        async with self._lock:
            if worker_id != self._root_id:
                # Non-root workers register as ready to receive
                self._receivers_ready.add(worker_id)
                if (
                    self._result is not None
                    and len(self._receivers_ready) >= self._world_size - 1
                ):
                    self._status = CollectiveStatus.COMPLETED
                    self._complete_event.set()
                return True

            if self._result is not None:
                return False  # Already broadcast

            self._result = data
            self._status = CollectiveStatus.IN_PROGRESS
            self._participants[worker_id] = data

            # Check if all receivers ready
            if len(self._receivers_ready) >= self._world_size - 1:
                self._status = CollectiveStatus.COMPLETED
                self._complete_event.set()
                logger.debug(
                    "broadcast_complete",
                    op_id=self.operation_id,
                    root=self._root_id,
                )

            return True

    async def get_result(self, worker_id: str) -> CollectiveResult:
        """Get broadcast data."""
        return CollectiveResult(
            operation_id=self.operation_id,
            status=self._status,
            result=self._result,
            participants=list(self._participants.keys()),
            started_at=self._created_at,
            completed_at=datetime.utcnow() if self.is_complete else None,
        )


class ReduceScatterOp(CollectiveOperation):
    """
    ReduceScatter: Reduce data and scatter chunks to workers.

    Each worker receives a portion of the reduced result.
    """

    def __init__(
        self,
        world_size: int,
        reduction: ReductionOp = ReductionOp.SUM,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._world_size = world_size
        self._reduction = reduction
        self._worker_ranks: dict[str, int] = {}

    async def contribute(self, worker_id: str, data: list[Any]) -> bool:
        """
        Contribute data to reduce-scatter.

        Data should be a list with world_size elements.
        """
        async with self._lock:
            if worker_id in self._participants:
                return False

            if not isinstance(data, list) or len(data) != self._world_size:
                logger.error(
                    "reducescatter_invalid_data",
                    worker_id=worker_id,
                    expected_len=self._world_size,
                    actual_len=len(data) if isinstance(data, list) else "not_list",
                )
                return False

            rank = len(self._participants)
            self._worker_ranks[worker_id] = rank
            self._participants[worker_id] = data

            if self._status == CollectiveStatus.PENDING:
                self._status = CollectiveStatus.IN_PROGRESS

            if len(self._participants) >= self._world_size:
                await self._compute_result()
                self._complete_event.set()

            return True

    async def _compute_result(self) -> None:
        """Compute reduced and scattered result."""
        try:
            # Collect all data
            all_data = list(self._participants.values())

            # Reduce across workers for each chunk
            chunks = []
            for i in range(self._world_size):
                values = [data[i] for data in all_data]

                if self._reduction == ReductionOp.SUM:
                    chunks.append(sum(values))
                elif self._reduction == ReductionOp.AVG:
                    chunks.append(sum(values) / len(values))
                elif self._reduction == ReductionOp.MIN:
                    chunks.append(min(values))
                elif self._reduction == ReductionOp.MAX:
                    chunks.append(max(values))

            self._result = chunks
            self._status = CollectiveStatus.COMPLETED
            logger.debug(
                "reducescatter_complete",
                op_id=self.operation_id,
                participants=len(self._participants),
            )

        except Exception as e:
            self._status = CollectiveStatus.FAILED
            logger.error("reducescatter_failed", op_id=self.operation_id, error=str(e))

    async def get_result(self, worker_id: str) -> CollectiveResult:
        """Get scattered result for this worker."""
        if worker_id not in self._worker_ranks:
            return CollectiveResult(
                operation_id=self.operation_id,
                status=CollectiveStatus.FAILED,
                error="Worker not registered",
            )

        # Each worker gets their corresponding chunk
        rank = self._worker_ranks[worker_id]
        result = self._result[rank] if self._result else None

        return CollectiveResult(
            operation_id=self.operation_id,
            status=self._status,
            result=result,
            participants=list(self._participants.keys()),
            started_at=self._created_at,
            completed_at=datetime.utcnow() if self.is_complete else None,
        )


class CollectiveManager:
    """
    Manages collective operations for distributed training jobs.
    """

    def __init__(self):
        self._operations: dict[str, CollectiveOperation] = {}
        self._job_operations: dict[str, list[str]] = {}  # job_id -> op_ids
        self._lock = asyncio.Lock()

    async def create_allreduce(
        self,
        job_id: str,
        world_size: int,
        reduction: ReductionOp = ReductionOp.SUM,
        timeout: float = 30.0,
    ) -> AllReduceOp:
        """Create an AllReduce operation."""
        async with self._lock:
            op = AllReduceOp(
                world_size=world_size,
                reduction=reduction,
                timeout_seconds=timeout,
            )
            self._operations[op.operation_id] = op
            if job_id not in self._job_operations:
                self._job_operations[job_id] = []
            self._job_operations[job_id].append(op.operation_id)
            return op

    async def create_allgather(
        self,
        job_id: str,
        world_size: int,
        timeout: float = 30.0,
    ) -> AllGatherOp:
        """Create an AllGather operation."""
        async with self._lock:
            op = AllGatherOp(world_size=world_size, timeout_seconds=timeout)
            self._operations[op.operation_id] = op
            if job_id not in self._job_operations:
                self._job_operations[job_id] = []
            self._job_operations[job_id].append(op.operation_id)
            return op

    async def create_broadcast(
        self,
        job_id: str,
        world_size: int,
        root_worker_id: str,
        timeout: float = 30.0,
    ) -> BroadcastOp:
        """Create a Broadcast operation."""
        async with self._lock:
            op = BroadcastOp(
                world_size=world_size,
                root_worker_id=root_worker_id,
                timeout_seconds=timeout,
            )
            self._operations[op.operation_id] = op
            if job_id not in self._job_operations:
                self._job_operations[job_id] = []
            self._job_operations[job_id].append(op.operation_id)
            return op

    async def create_reducescatter(
        self,
        job_id: str,
        world_size: int,
        reduction: ReductionOp = ReductionOp.SUM,
        timeout: float = 30.0,
    ) -> ReduceScatterOp:
        """Create a ReduceScatter operation."""
        async with self._lock:
            op = ReduceScatterOp(
                world_size=world_size,
                reduction=reduction,
                timeout_seconds=timeout,
            )
            self._operations[op.operation_id] = op
            if job_id not in self._job_operations:
                self._job_operations[job_id] = []
            self._job_operations[job_id].append(op.operation_id)
            return op

    async def get_operation(self, operation_id: str) -> Optional[CollectiveOperation]:
        """Get an operation by ID."""
        async with self._lock:
            return self._operations.get(operation_id)

    async def cleanup_job(self, job_id: str) -> int:
        """Clean up operations for a job."""
        async with self._lock:
            op_ids = self._job_operations.pop(job_id, [])
            for op_id in op_ids:
                self._operations.pop(op_id, None)
            return len(op_ids)

    async def get_stats(self) -> dict[str, Any]:
        """Get manager statistics."""
        async with self._lock:
            by_status = {}
            for op in self._operations.values():
                status = op.status.value
                by_status[status] = by_status.get(status, 0) + 1

            return {
                "total_operations": len(self._operations),
                "active_jobs": len(self._job_operations),
                "by_status": by_status,
            }
