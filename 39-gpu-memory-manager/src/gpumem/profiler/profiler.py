"""Memory profiling and diagnostics."""

import logging
import threading
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict
import json

from ..core.memory import MemoryBlock, MemoryStats, DeviceType

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of memory events."""
    ALLOC = auto()
    FREE = auto()
    OOM = auto()
    CACHE_HIT = auto()
    CACHE_MISS = auto()
    DEFRAG = auto()
    SNAPSHOT = auto()


@dataclass
class MemoryEvent:
    """A single memory event."""
    event_type: EventType
    timestamp: float
    ptr: int = 0
    size: int = 0
    device: DeviceType = DeviceType.CPU
    device_id: int = 0
    stream: int = 0
    tag: str = ""
    stacktrace: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AllocationTrace:
    """Trace of a single allocation."""
    ptr: int
    size: int
    device: DeviceType
    device_id: int
    timestamp: float
    freed_at: Optional[float] = None
    stacktrace: str = ""
    tag: str = ""

    @property
    def lifetime(self) -> float:
        """Get allocation lifetime in seconds."""
        if self.freed_at:
            return self.freed_at - self.timestamp
        return time.time() - self.timestamp

    @property
    def is_active(self) -> bool:
        return self.freed_at is None


@dataclass
class MemorySnapshot:
    """Snapshot of memory state at a point in time."""
    timestamp: float
    stats: MemoryStats
    active_allocations: List[AllocationTrace]
    events_since_last: List[MemoryEvent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "stats": {
                "allocated": self.stats.allocated,
                "reserved": self.stats.reserved,
                "active": self.stats.active,
                "inactive": self.stats.inactive,
                "peak_allocated": self.stats.peak_allocated,
                "num_allocs": self.stats.num_allocs,
                "num_frees": self.stats.num_frees,
            },
            "num_active_allocations": len(self.active_allocations),
            "total_active_size": sum(a.size for a in self.active_allocations),
            "metadata": self.metadata,
        }


class MemoryTimeline:
    """Timeline of memory events and snapshots."""

    def __init__(self, max_events: int = 10000):
        self._events: List[MemoryEvent] = []
        self._snapshots: List[MemorySnapshot] = []
        self._max_events = max_events
        self._lock = threading.Lock()

    def add_event(self, event: MemoryEvent):
        """Add event to timeline."""
        with self._lock:
            self._events.append(event)

            # Trim if too many events
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events // 2:]

    def add_snapshot(self, snapshot: MemorySnapshot):
        """Add snapshot to timeline."""
        with self._lock:
            self._snapshots.append(snapshot)

    def get_events(
        self,
        start_time: float = 0,
        end_time: float = None,
        event_types: List[EventType] = None
    ) -> List[MemoryEvent]:
        """Get events in time range."""
        with self._lock:
            if end_time is None:
                end_time = time.time()

            events = [
                e for e in self._events
                if start_time <= e.timestamp <= end_time
            ]

            if event_types:
                events = [e for e in events if e.event_type in event_types]

            return events

    def get_snapshots(self) -> List[MemorySnapshot]:
        """Get all snapshots."""
        with self._lock:
            return list(self._snapshots)

    def clear(self):
        """Clear timeline."""
        with self._lock:
            self._events.clear()
            self._snapshots.clear()

    def export_json(self, filepath: str):
        """Export timeline to JSON."""
        with self._lock:
            data = {
                "events": [
                    {
                        "type": e.event_type.name,
                        "timestamp": e.timestamp,
                        "ptr": e.ptr,
                        "size": e.size,
                        "device": e.device.name,
                        "tag": e.tag,
                    }
                    for e in self._events
                ],
                "snapshots": [s.to_dict() for s in self._snapshots],
            }

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)


class MemoryProfiler:
    """
    Memory profiler for tracking allocations and usage.

    Features:
    - Allocation tracking with stacktraces
    - Memory snapshots
    - Leak detection
    - Usage statistics
    - Export to various formats
    """

    def __init__(self, record_stacktraces: bool = False):
        self._lock = threading.Lock()
        self._enabled = False
        self._record_stacktraces = record_stacktraces

        # Tracking data
        self._allocations: Dict[int, AllocationTrace] = {}
        self._timeline = MemoryTimeline()
        self._stats = MemoryStats()

        # Aggregated statistics
        self._size_histogram: Dict[int, int] = defaultdict(int)
        self._tag_stats: Dict[str, int] = defaultdict(int)
        self._device_stats: Dict[DeviceType, int] = defaultdict(int)

    def enable(self):
        """Enable profiling."""
        self._enabled = True
        logger.info("Memory profiler enabled")

    def disable(self):
        """Disable profiling."""
        self._enabled = False
        logger.info("Memory profiler disabled")

    def record_alloc(self, block: MemoryBlock):
        """Record an allocation."""
        if not self._enabled:
            return

        with self._lock:
            # Create trace
            trace = AllocationTrace(
                ptr=block.ptr,
                size=block.size,
                device=block.device,
                device_id=block.device_id,
                timestamp=time.time(),
                tag=block.tag,
                stacktrace=self._get_stacktrace() if self._record_stacktraces else ""
            )
            self._allocations[block.ptr] = trace

            # Record event
            event = MemoryEvent(
                event_type=EventType.ALLOC,
                timestamp=trace.timestamp,
                ptr=block.ptr,
                size=block.size,
                device=block.device,
                device_id=block.device_id,
                tag=block.tag,
                stacktrace=trace.stacktrace
            )
            self._timeline.add_event(event)

            # Update statistics
            self._stats.allocated += block.size
            self._stats.num_allocs += 1
            self._update_histogram(block.size)
            self._tag_stats[block.tag] += block.size
            self._device_stats[block.device] += block.size

    def record_free(self, block: MemoryBlock):
        """Record a deallocation."""
        if not self._enabled:
            return

        with self._lock:
            if block.ptr in self._allocations:
                trace = self._allocations[block.ptr]
                trace.freed_at = time.time()

                # Record event
                event = MemoryEvent(
                    event_type=EventType.FREE,
                    timestamp=trace.freed_at,
                    ptr=block.ptr,
                    size=block.size,
                    device=block.device,
                    device_id=block.device_id
                )
                self._timeline.add_event(event)

                # Update statistics
                self._stats.allocated -= block.size
                self._stats.num_frees += 1

                # Remove from active allocations
                del self._allocations[block.ptr]

    def record_oom(self, size: int, device: DeviceType, device_id: int):
        """Record out of memory event."""
        if not self._enabled:
            return

        with self._lock:
            event = MemoryEvent(
                event_type=EventType.OOM,
                timestamp=time.time(),
                size=size,
                device=device,
                device_id=device_id,
                stacktrace=self._get_stacktrace() if self._record_stacktraces else ""
            )
            self._timeline.add_event(event)
            self._stats.num_ooms += 1

    def take_snapshot(self, tag: str = "") -> MemorySnapshot:
        """Take a snapshot of current memory state."""
        with self._lock:
            snapshot = MemorySnapshot(
                timestamp=time.time(),
                stats=MemoryStats(
                    allocated=self._stats.allocated,
                    reserved=self._stats.reserved,
                    active=self._stats.active,
                    inactive=self._stats.inactive,
                    num_allocs=self._stats.num_allocs,
                    num_frees=self._stats.num_frees,
                    num_ooms=self._stats.num_ooms,
                    peak_allocated=self._stats.peak_allocated
                ),
                active_allocations=list(self._allocations.values()),
                metadata={"tag": tag}
            )
            self._timeline.add_snapshot(snapshot)
            return snapshot

    def get_active_allocations(self) -> List[AllocationTrace]:
        """Get all active allocations."""
        with self._lock:
            return list(self._allocations.values())

    def find_leaks(self, min_lifetime: float = 60.0) -> List[AllocationTrace]:
        """Find potential memory leaks (long-lived allocations)."""
        with self._lock:
            leaks = []
            current_time = time.time()

            for trace in self._allocations.values():
                if current_time - trace.timestamp > min_lifetime:
                    leaks.append(trace)

            return sorted(leaks, key=lambda t: t.lifetime, reverse=True)

    def get_top_allocations(self, n: int = 10) -> List[AllocationTrace]:
        """Get top N allocations by size."""
        with self._lock:
            allocations = list(self._allocations.values())
            return sorted(allocations, key=lambda t: t.size, reverse=True)[:n]

    def get_allocation_by_tag(self, tag: str) -> List[AllocationTrace]:
        """Get allocations with specific tag."""
        with self._lock:
            return [t for t in self._allocations.values() if t.tag == tag]

    def get_stats(self) -> MemoryStats:
        """Get current statistics."""
        return self._stats

    def get_size_histogram(self) -> Dict[str, int]:
        """Get allocation size histogram."""
        with self._lock:
            return {
                "<1KB": self._size_histogram[0],
                "1KB-1MB": self._size_histogram[1],
                "1MB-10MB": self._size_histogram[2],
                "10MB-100MB": self._size_histogram[3],
                ">100MB": self._size_histogram[4],
            }

    def get_tag_summary(self) -> Dict[str, int]:
        """Get memory usage by tag."""
        with self._lock:
            return dict(self._tag_stats)

    def get_device_summary(self) -> Dict[str, int]:
        """Get memory usage by device."""
        with self._lock:
            return {d.name: size for d, size in self._device_stats.items()}

    def get_timeline(self) -> MemoryTimeline:
        """Get event timeline."""
        return self._timeline

    def _get_stacktrace(self) -> str:
        """Get current stacktrace."""
        return ''.join(traceback.format_stack()[:-2])

    def _update_histogram(self, size: int):
        """Update size histogram."""
        if size < 1024:
            self._size_histogram[0] += 1
        elif size < 1024 * 1024:
            self._size_histogram[1] += 1
        elif size < 10 * 1024 * 1024:
            self._size_histogram[2] += 1
        elif size < 100 * 1024 * 1024:
            self._size_histogram[3] += 1
        else:
            self._size_histogram[4] += 1

    def reset(self):
        """Reset profiler state."""
        with self._lock:
            self._allocations.clear()
            self._timeline.clear()
            self._stats = MemoryStats()
            self._size_histogram.clear()
            self._tag_stats.clear()
            self._device_stats.clear()

    def print_summary(self):
        """Print profiling summary."""
        with self._lock:
            print("\n=== Memory Profiler Summary ===")
            print(f"Active allocations: {len(self._allocations)}")
            print(f"Total allocated: {self._stats.allocated / 1e6:.2f} MB")
            print(f"Peak allocated: {self._stats.peak_allocated / 1e6:.2f} MB")
            print(f"Alloc count: {self._stats.num_allocs}")
            print(f"Free count: {self._stats.num_frees}")
            print(f"OOM count: {self._stats.num_ooms}")

            print("\nSize histogram:")
            for bucket, count in self.get_size_histogram().items():
                print(f"  {bucket}: {count}")

            print("\nTop 5 allocations by size:")
            for trace in self.get_top_allocations(5):
                print(f"  {trace.size / 1e6:.2f} MB - {trace.tag or 'unnamed'}")

    def export_report(self, filepath: str):
        """Export detailed report to JSON."""
        with self._lock:
            report = {
                "summary": {
                    "active_allocations": len(self._allocations),
                    "total_allocated": self._stats.allocated,
                    "peak_allocated": self._stats.peak_allocated,
                    "num_allocs": self._stats.num_allocs,
                    "num_frees": self._stats.num_frees,
                    "num_ooms": self._stats.num_ooms,
                },
                "size_histogram": self.get_size_histogram(),
                "tag_summary": self.get_tag_summary(),
                "device_summary": self.get_device_summary(),
                "top_allocations": [
                    {
                        "ptr": t.ptr,
                        "size": t.size,
                        "tag": t.tag,
                        "lifetime": t.lifetime,
                    }
                    for t in self.get_top_allocations(20)
                ],
                "potential_leaks": [
                    {
                        "ptr": t.ptr,
                        "size": t.size,
                        "tag": t.tag,
                        "lifetime": t.lifetime,
                    }
                    for t in self.find_leaks()[:10]
                ],
            }

            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)

            logger.info(f"Report exported to {filepath}")
