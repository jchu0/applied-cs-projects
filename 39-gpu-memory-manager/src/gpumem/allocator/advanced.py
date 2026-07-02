"""Advanced memory management features.

Phase 6 features:
- P2P memory transfers
- Prefetch operations
- Stream synchronization
- CPU offloading automation
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
import heapq

from ..core.memory import (
    MemoryBlock,
    MemoryConfig,
    MemoryStats,
    DeviceType,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Transfer Types and Events
# =============================================================================

class TransferDirection(Enum):
    """Direction of memory transfer."""
    HOST_TO_DEVICE = auto()
    DEVICE_TO_HOST = auto()
    DEVICE_TO_DEVICE = auto()


@dataclass
class TransferEvent:
    """Represents a memory transfer event."""
    source_ptr: int
    dest_ptr: int
    size: int
    direction: TransferDirection
    source_device: int = 0
    dest_device: int = 0
    stream: int = 0
    timestamp: float = field(default_factory=time.time)
    completed: bool = False
    duration_ms: float = 0.0


@dataclass
class StreamEvent:
    """Represents a stream synchronization event."""
    stream: int
    event_id: int
    recorded_at: float = field(default_factory=time.time)
    synchronized: bool = False


# =============================================================================
# GPU Topology Detection and Routing
# =============================================================================

class TopologyType(Enum):
    """GPU interconnect topology types."""
    LINEAR = auto()      # Linear chain (PCIe)
    RING = auto()        # Ring topology
    FULL_MESH = auto()   # Full NVLink mesh (DGX-style)
    HYBRID = auto()      # Mixed NVLink/PCIe
    CUSTOM = auto()      # User-defined topology


@dataclass
class GPULink:
    """Represents a link between two GPUs."""
    src: int
    dst: int
    bandwidth_gbps: float  # GB/s
    latency_us: float  # microseconds
    link_type: str  # "nvlink", "pcie", "nvswitch"
    bidirectional: bool = True


class GPUTopology:
    """
    Represents GPU interconnect topology for optimal transfer routing.

    Supports:
    - NVLink mesh topologies (DGX, HGX)
    - PCIe topologies with root complex awareness
    - NUMA node mapping
    - Multi-hop path computation with Dijkstra's algorithm
    """

    # Standard topology templates
    NVLINK_BANDWIDTH = 200.0  # GB/s per NVLink
    PCIE_GEN4_BANDWIDTH = 32.0  # GB/s PCIe Gen4 x16
    PCIE_GEN5_BANDWIDTH = 64.0  # GB/s PCIe Gen5 x16
    NVSWITCH_BANDWIDTH = 400.0  # GB/s NVSwitch

    NVLINK_LATENCY = 1.0  # microseconds
    PCIE_LATENCY = 3.0  # microseconds
    NVSWITCH_LATENCY = 0.5  # microseconds

    def __init__(self, num_devices: int, topology_type: TopologyType = TopologyType.LINEAR):
        self._num_devices = num_devices
        self._topology_type = topology_type
        self._links: Dict[Tuple[int, int], GPULink] = {}
        self._numa_mapping: Dict[int, int] = {}  # GPU -> NUMA node
        self._adjacency: Dict[int, List[int]] = {i: [] for i in range(num_devices)}

        # Initialize based on topology type
        self._initialize_topology()

    def _initialize_topology(self):
        """Initialize topology based on type."""
        if self._topology_type == TopologyType.LINEAR:
            self._init_linear_topology()
        elif self._topology_type == TopologyType.RING:
            self._init_ring_topology()
        elif self._topology_type == TopologyType.FULL_MESH:
            self._init_full_mesh_topology()
        elif self._topology_type == TopologyType.HYBRID:
            self._init_hybrid_topology()

        # Default NUMA mapping (2 GPUs per NUMA node)
        for i in range(self._num_devices):
            self._numa_mapping[i] = i // 2

    def _add_link(self, src: int, dst: int, bandwidth: float, latency: float,
                  link_type: str, bidirectional: bool = True):
        """Add a link between GPUs."""
        self._links[(src, dst)] = GPULink(src, dst, bandwidth, latency, link_type, bidirectional)
        self._adjacency[src].append(dst)
        if bidirectional:
            self._links[(dst, src)] = GPULink(dst, src, bandwidth, latency, link_type, True)
            self._adjacency[dst].append(src)

    def _init_linear_topology(self):
        """Initialize linear PCIe topology."""
        for i in range(self._num_devices - 1):
            self._add_link(i, i + 1, self.PCIE_GEN4_BANDWIDTH, self.PCIE_LATENCY, "pcie")

    def _init_ring_topology(self):
        """Initialize ring topology with NVLink."""
        for i in range(self._num_devices):
            next_gpu = (i + 1) % self._num_devices
            self._add_link(i, next_gpu, self.NVLINK_BANDWIDTH, self.NVLINK_LATENCY, "nvlink")

    def _init_full_mesh_topology(self):
        """Initialize full NVLink mesh (DGX-style)."""
        for i in range(self._num_devices):
            for j in range(i + 1, self._num_devices):
                self._add_link(i, j, self.NVLINK_BANDWIDTH, self.NVLINK_LATENCY, "nvlink")

    def _init_hybrid_topology(self):
        """Initialize hybrid topology (NVLink for adjacent, PCIe for others)."""
        # NVLink for adjacent GPUs (simulating NVLink bridges)
        for i in range(self._num_devices - 1):
            self._add_link(i, i + 1, self.NVLINK_BANDWIDTH, self.NVLINK_LATENCY, "nvlink")

        # PCIe for non-adjacent (through CPU)
        for i in range(self._num_devices):
            for j in range(i + 2, self._num_devices):
                self._add_link(i, j, self.PCIE_GEN4_BANDWIDTH, self.PCIE_LATENCY, "pcie")

    def get_link(self, src: int, dst: int) -> Optional[GPULink]:
        """Get link between two GPUs."""
        return self._links.get((src, dst))

    def get_neighbors(self, gpu: int) -> List[int]:
        """Get directly connected GPUs."""
        return self._adjacency.get(gpu, [])

    def get_numa_node(self, gpu: int) -> int:
        """Get NUMA node for a GPU."""
        return self._numa_mapping.get(gpu, 0)

    def set_numa_mapping(self, gpu: int, numa_node: int):
        """Set NUMA node for a GPU."""
        self._numa_mapping[gpu] = numa_node

    def compute_optimal_path(self, src: int, dst: int) -> Tuple[List[int], float, float]:
        """
        Compute optimal path from src to dst using Dijkstra's algorithm.

        Returns:
            Tuple of (path, total_bandwidth, total_latency)
            - path: List of GPU IDs from src to dst (inclusive)
            - total_bandwidth: Effective bandwidth of path (limited by slowest link)
            - total_latency: Sum of link latencies
        """
        if src == dst:
            return [src], float('inf'), 0.0

        if src >= self._num_devices or dst >= self._num_devices:
            return [], 0.0, float('inf')

        # Dijkstra's algorithm (minimize latency, track bandwidth)
        # Priority queue: (latency, current_gpu, path, min_bandwidth)
        pq = [(0.0, src, [src], float('inf'))]
        visited: Set[int] = set()

        while pq:
            latency, current, path, min_bw = heapq.heappop(pq)

            if current == dst:
                return path, min_bw, latency

            if current in visited:
                continue
            visited.add(current)

            for neighbor in self._adjacency.get(current, []):
                if neighbor not in visited:
                    link = self._links.get((current, neighbor))
                    if link:
                        new_latency = latency + link.latency_us
                        new_bw = min(min_bw, link.bandwidth_gbps)
                        heapq.heappush(pq, (new_latency, neighbor, path + [neighbor], new_bw))

        # No path found - fallback to direct (assume PCIe through CPU)
        return [src, dst], self.PCIE_GEN4_BANDWIDTH / 2, self.PCIE_LATENCY * 2

    def get_transfer_cost(self, src: int, dst: int, size_bytes: int) -> float:
        """
        Get transfer cost (time in milliseconds) for a given transfer.

        Considers:
        - Path bandwidth (limited by slowest link)
        - Hop latencies
        - NUMA penalties
        """
        path, bandwidth_gbps, latency_us = self.compute_optimal_path(src, dst)

        if not path or bandwidth_gbps <= 0:
            return float('inf')

        # Transfer time = latency + (size / bandwidth)
        bandwidth_bytes_per_sec = bandwidth_gbps * 1e9
        transfer_time_sec = size_bytes / bandwidth_bytes_per_sec
        transfer_time_ms = transfer_time_sec * 1000
        latency_ms = latency_us / 1000

        # NUMA penalty if crossing NUMA boundaries
        numa_penalty = 0.0
        src_numa = self._numa_mapping.get(src, 0)
        dst_numa = self._numa_mapping.get(dst, 0)
        if src_numa != dst_numa:
            numa_penalty = 0.1  # 0.1ms NUMA crossing penalty

        return transfer_time_ms + latency_ms + numa_penalty

    def to_dict(self) -> Dict[str, Any]:
        """Serialize topology to dictionary."""
        return {
            'num_devices': self._num_devices,
            'topology_type': self._topology_type.name,
            'links': [
                {
                    'src': link.src,
                    'dst': link.dst,
                    'bandwidth_gbps': link.bandwidth_gbps,
                    'latency_us': link.latency_us,
                    'link_type': link.link_type,
                }
                for link in self._links.values()
            ],
            'numa_mapping': self._numa_mapping,
        }


# =============================================================================
# P2P Transfer Manager
# =============================================================================

class P2PTransferManager:
    """
    Manages peer-to-peer memory transfers between GPUs.

    Features:
    - Direct GPU-to-GPU transfers
    - Transfer queue management
    - Bandwidth monitoring
    - Topology-aware routing with multi-hop support
    - NUMA-aware transfer optimization
    """

    def __init__(
        self,
        num_devices: int = 2,
        topology: Optional[GPUTopology] = None,
        topology_type: TopologyType = TopologyType.HYBRID
    ):
        self._lock = threading.Lock()
        self._num_devices = num_devices

        # GPU topology for routing decisions
        self._topology = topology or GPUTopology(num_devices, topology_type)

        # P2P access matrix (derived from topology)
        self._p2p_enabled: Dict[Tuple[int, int], bool] = {}
        self._initialize_p2p_access()

        # Transfer tracking
        self._pending_transfers: deque = deque()
        self._completed_transfers: List[TransferEvent] = []
        self._transfer_id = 0

        # Bandwidth tracking (bytes/sec) - derived from topology
        self._bandwidth_matrix: Dict[Tuple[int, int], float] = {}
        self._initialize_bandwidth()

        # Multi-hop transfer tracking
        self._hop_count: Dict[Tuple[int, int], int] = {}
        self._optimal_paths: Dict[Tuple[int, int], List[int]] = {}
        self._precompute_paths()

        # Statistics
        self._total_bytes_transferred = 0
        self._total_transfers = 0
        self._multi_hop_transfers = 0

    def _initialize_p2p_access(self):
        """Initialize P2P access matrix from topology."""
        for i in range(self._num_devices):
            for j in range(self._num_devices):
                if i == j:
                    self._p2p_enabled[(i, j)] = True
                else:
                    # Check if path exists in topology
                    path, bw, _ = self._topology.compute_optimal_path(i, j)
                    self._p2p_enabled[(i, j)] = len(path) > 0 and bw > 0

    def _initialize_bandwidth(self):
        """Initialize bandwidth estimates from topology."""
        for i in range(self._num_devices):
            for j in range(self._num_devices):
                if i == j:
                    self._bandwidth_matrix[(i, j)] = 900e9  # Same device (900 GB/s)
                else:
                    # Get effective bandwidth from topology path
                    _, bw_gbps, _ = self._topology.compute_optimal_path(i, j)
                    self._bandwidth_matrix[(i, j)] = bw_gbps * 1e9  # Convert to bytes/sec

    def _precompute_paths(self):
        """Precompute optimal paths for all GPU pairs."""
        for i in range(self._num_devices):
            for j in range(self._num_devices):
                if i != j:
                    path, _, _ = self._topology.compute_optimal_path(i, j)
                    self._optimal_paths[(i, j)] = path
                    self._hop_count[(i, j)] = len(path) - 1 if path else 0

    def can_access_peer(self, src_device: int, dst_device: int) -> bool:
        """Check if P2P access is enabled between devices."""
        return self._p2p_enabled.get((src_device, dst_device), False)

    def enable_peer_access(self, src_device: int, dst_device: int) -> bool:
        """Enable P2P access between devices."""
        with self._lock:
            self._p2p_enabled[(src_device, dst_device)] = True
            self._p2p_enabled[(dst_device, src_device)] = True
            return True

    def disable_peer_access(self, src_device: int, dst_device: int):
        """Disable P2P access between devices."""
        with self._lock:
            self._p2p_enabled[(src_device, dst_device)] = False
            self._p2p_enabled[(dst_device, src_device)] = False

    def transfer(
        self,
        src_ptr: int,
        dst_ptr: int,
        size: int,
        src_device: int,
        dst_device: int,
        stream: int = 0,
        async_transfer: bool = True
    ) -> Optional[TransferEvent]:
        """
        Initiate a P2P memory transfer.

        Args:
            src_ptr: Source memory pointer
            dst_ptr: Destination memory pointer
            size: Size in bytes
            src_device: Source GPU ID
            dst_device: Destination GPU ID
            stream: CUDA stream for async transfer
            async_transfer: Whether to perform async transfer

        Returns:
            TransferEvent on success, None on failure
        """
        with self._lock:
            if not self.can_access_peer(src_device, dst_device):
                logger.warning(
                    f"P2P access not enabled between device {src_device} "
                    f"and device {dst_device}"
                )
                return None

            event = TransferEvent(
                source_ptr=src_ptr,
                dest_ptr=dst_ptr,
                size=size,
                direction=TransferDirection.DEVICE_TO_DEVICE,
                source_device=src_device,
                dest_device=dst_device,
                stream=stream
            )

            # Simulate transfer
            bandwidth = self._bandwidth_matrix.get(
                (src_device, dst_device),
                12e9
            )
            event.duration_ms = (size / bandwidth) * 1000

            if async_transfer:
                self._pending_transfers.append(event)
            else:
                # Simulate blocking transfer
                time.sleep(event.duration_ms / 1000)
                event.completed = True
                self._completed_transfers.append(event)

            self._transfer_id += 1
            self._total_bytes_transferred += size
            self._total_transfers += 1

            return event

    def wait_for_transfer(self, event: TransferEvent) -> bool:
        """Wait for a transfer to complete."""
        if event.completed:
            return True

        # Simulate waiting
        remaining_time = event.duration_ms / 1000
        if remaining_time > 0:
            time.sleep(remaining_time)

        event.completed = True
        return True

    def get_bandwidth(self, src_device: int, dst_device: int) -> float:
        """Get estimated bandwidth between devices in bytes/sec."""
        return self._bandwidth_matrix.get((src_device, dst_device), 0)

    def get_statistics(self) -> Dict[str, Any]:
        """Get transfer statistics."""
        return {
            'total_bytes_transferred': self._total_bytes_transferred,
            'total_transfers': self._total_transfers,
            'pending_transfers': len(self._pending_transfers),
            'completed_transfers': len(self._completed_transfers),
            'multi_hop_transfers': self._multi_hop_transfers,
            'topology_type': self._topology._topology_type.name,
        }

    # -------------------------------------------------------------------------
    # Topology-Aware Routing Methods
    # -------------------------------------------------------------------------

    def get_optimal_path(self, src_device: int, dst_device: int) -> List[int]:
        """
        Get the optimal path from src to dst GPU.

        Returns:
            List of GPU IDs representing the path (inclusive of src and dst).
        """
        if src_device == dst_device:
            return [src_device]
        return self._optimal_paths.get((src_device, dst_device), [src_device, dst_device])

    def get_hop_count(self, src_device: int, dst_device: int) -> int:
        """Get number of hops required for transfer."""
        if src_device == dst_device:
            return 0
        return self._hop_count.get((src_device, dst_device), 1)

    def get_transfer_cost(self, src_device: int, dst_device: int, size: int) -> float:
        """
        Get estimated transfer time in milliseconds.

        Uses topology-aware cost estimation including:
        - Path bandwidth (limited by slowest link)
        - Hop latencies
        - NUMA crossing penalties
        """
        return self._topology.get_transfer_cost(src_device, dst_device, size)

    def is_direct_path(self, src_device: int, dst_device: int) -> bool:
        """Check if transfer uses a direct (single-hop) path."""
        return self.get_hop_count(src_device, dst_device) <= 1

    def get_topology(self) -> GPUTopology:
        """Get the GPU topology instance."""
        return self._topology

    def set_topology(self, topology: GPUTopology):
        """
        Update the GPU topology and recompute routing.

        Args:
            topology: New GPUTopology instance
        """
        with self._lock:
            self._topology = topology
            self._initialize_p2p_access()
            self._initialize_bandwidth()
            self._precompute_paths()

    def transfer_with_routing(
        self,
        src_ptr: int,
        dst_ptr: int,
        size: int,
        src_device: int,
        dst_device: int,
        stream: int = 0,
        use_multi_hop: bool = True
    ) -> List[TransferEvent]:
        """
        Transfer with topology-aware routing.

        For multi-hop transfers, creates intermediate transfers through
        the optimal path. Returns a list of transfer events.

        Args:
            src_ptr: Source memory pointer
            dst_ptr: Destination memory pointer
            size: Size in bytes
            src_device: Source GPU ID
            dst_device: Destination GPU ID
            stream: CUDA stream for transfers
            use_multi_hop: Whether to use multi-hop routing if needed

        Returns:
            List of TransferEvents for each hop
        """
        path = self.get_optimal_path(src_device, dst_device)

        if len(path) <= 2 or not use_multi_hop:
            # Direct transfer
            event = self.transfer(src_ptr, dst_ptr, size, src_device, dst_device, stream)
            return [event] if event else []

        # Multi-hop transfer
        events = []
        current_ptr = src_ptr

        for i in range(len(path) - 1):
            hop_src = path[i]
            hop_dst = path[i + 1]

            # For intermediate hops, we'd need intermediate buffers
            # In simulation, we track the transfer chain
            is_final = (i == len(path) - 2)
            hop_dst_ptr = dst_ptr if is_final else current_ptr  # Simplified

            event = self.transfer(
                src_ptr=current_ptr,
                dst_ptr=hop_dst_ptr,
                size=size,
                src_device=hop_src,
                dst_device=hop_dst,
                stream=stream,
                async_transfer=True
            )

            if event:
                events.append(event)
            else:
                logger.warning(f"Multi-hop transfer failed at hop {hop_src} -> {hop_dst}")
                break

        if events:
            self._multi_hop_transfers += 1

        return events

    def get_numa_distance(self, src_device: int, dst_device: int) -> int:
        """
        Get NUMA distance between two GPUs.

        Returns:
            0 if same NUMA node, positive integer for distance
        """
        src_numa = self._topology.get_numa_node(src_device)
        dst_numa = self._topology.get_numa_node(dst_device)
        return abs(src_numa - dst_numa)


# =============================================================================
# Prefetch Manager
# =============================================================================

class PrefetchPriority(Enum):
    """Prefetch priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class PrefetchRequest:
    """Represents a prefetch request."""
    block: MemoryBlock
    target_device: DeviceType
    target_device_id: int = 0
    priority: PrefetchPriority = PrefetchPriority.NORMAL
    stream: int = 0
    callback: Optional[Callable] = None
    submitted_at: float = field(default_factory=time.time)
    completed: bool = False

    def __lt__(self, other):
        # Higher priority first
        return self.priority.value > other.priority.value


class PrefetchManager:
    """
    Manages proactive memory prefetching.

    Features:
    - Async prefetch to GPU
    - Prefetch to CPU (eviction)
    - Priority queue management
    - Prefetch scheduling
    """

    def __init__(self, max_pending: int = 100):
        self._lock = threading.Lock()
        self._max_pending = max_pending

        # Priority queue for prefetch requests
        self._pending: List[PrefetchRequest] = []
        self._in_flight: List[PrefetchRequest] = []
        self._completed: List[PrefetchRequest] = []

        # Device state tracking
        self._device_memory_used: Dict[Tuple[DeviceType, int], int] = {}
        self._device_memory_limit: Dict[Tuple[DeviceType, int], int] = {}

        # Statistics
        self._stats = {
            'prefetches_to_device': 0,
            'prefetches_to_host': 0,
            'bytes_prefetched': 0,
            'prefetch_hits': 0,
            'prefetch_misses': 0,
        }

    def prefetch_to_device(
        self,
        block: MemoryBlock,
        device_id: int = 0,
        stream: int = 0,
        priority: PrefetchPriority = PrefetchPriority.NORMAL,
        callback: Optional[Callable] = None
    ) -> PrefetchRequest:
        """
        Prefetch memory block to GPU.

        Args:
            block: Memory block to prefetch
            device_id: Target GPU ID
            stream: CUDA stream for async prefetch
            priority: Prefetch priority
            callback: Optional callback on completion

        Returns:
            PrefetchRequest for tracking
        """
        with self._lock:
            request = PrefetchRequest(
                block=block,
                target_device=DeviceType.CUDA,
                target_device_id=device_id,
                priority=priority,
                stream=stream,
                callback=callback
            )

            heapq.heappush(self._pending, request)
            self._stats['prefetches_to_device'] += 1

            # Process if possible
            self._process_pending()

            return request

    def prefetch_to_host(
        self,
        block: MemoryBlock,
        stream: int = 0,
        priority: PrefetchPriority = PrefetchPriority.NORMAL,
        callback: Optional[Callable] = None
    ) -> PrefetchRequest:
        """
        Prefetch memory block to CPU (evict from GPU).

        Args:
            block: Memory block to prefetch to host
            stream: CUDA stream for async prefetch
            priority: Prefetch priority
            callback: Optional callback on completion

        Returns:
            PrefetchRequest for tracking
        """
        with self._lock:
            request = PrefetchRequest(
                block=block,
                target_device=DeviceType.CPU,
                target_device_id=0,
                priority=priority,
                stream=stream,
                callback=callback
            )

            heapq.heappush(self._pending, request)
            self._stats['prefetches_to_host'] += 1

            self._process_pending()

            return request

    def _process_pending(self):
        """Process pending prefetch requests."""
        while self._pending and len(self._in_flight) < self._max_pending:
            request = heapq.heappop(self._pending)

            # Simulate prefetch initiation
            self._in_flight.append(request)
            self._stats['bytes_prefetched'] += request.block.size

    def wait(self, request: PrefetchRequest, timeout_ms: float = 0) -> bool:
        """
        Wait for a prefetch request to complete.

        Args:
            request: Prefetch request to wait for
            timeout_ms: Timeout in milliseconds (0 = infinite)

        Returns:
            True if completed, False on timeout
        """
        if request.completed:
            return True

        # Simulate wait
        start = time.time()
        while not request.completed:
            elapsed_ms = (time.time() - start) * 1000
            if timeout_ms > 0 and elapsed_ms >= timeout_ms:
                return False

            # Simulate completion
            if request in self._in_flight:
                self._in_flight.remove(request)
                request.completed = True
                self._completed.append(request)

                if request.callback:
                    request.callback(request)

            time.sleep(0.001)

        return True

    def cancel(self, request: PrefetchRequest) -> bool:
        """Cancel a pending prefetch request."""
        with self._lock:
            if request in self._pending:
                self._pending.remove(request)
                heapq.heapify(self._pending)
                return True
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get prefetch statistics."""
        return {
            **self._stats,
            'pending_requests': len(self._pending),
            'in_flight_requests': len(self._in_flight),
        }


# =============================================================================
# Stream Synchronization
# =============================================================================

class StreamSynchronizer:
    """
    Manages CUDA stream synchronization.

    Features:
    - Stream-to-stream synchronization
    - Event-based synchronization
    - Multi-stream barriers
    - Synchronization graph
    """

    def __init__(self, num_streams: int = 16):
        self._lock = threading.Lock()
        self._num_streams = num_streams

        # Event tracking
        self._events: Dict[int, List[StreamEvent]] = {
            i: [] for i in range(num_streams)
        }
        self._next_event_id = 0

        # Stream dependencies
        self._dependencies: Dict[int, Set[int]] = {
            i: set() for i in range(num_streams)
        }

        # Synchronization state
        self._stream_timestamps: Dict[int, float] = {
            i: 0.0 for i in range(num_streams)
        }

    def record_event(self, stream: int) -> StreamEvent:
        """
        Record an event on a stream.

        Args:
            stream: Stream ID to record event on

        Returns:
            StreamEvent for later synchronization
        """
        with self._lock:
            event = StreamEvent(
                stream=stream,
                event_id=self._next_event_id
            )
            self._next_event_id += 1

            self._events[stream].append(event)
            self._stream_timestamps[stream] = time.time()

            return event

    def wait_event(self, stream: int, event: StreamEvent):
        """
        Make a stream wait for an event.

        Args:
            stream: Stream that should wait
            event: Event to wait for
        """
        with self._lock:
            self._dependencies[stream].add(event.stream)

    def synchronize_stream(self, stream: int):
        """
        Synchronize a stream (block until all operations complete).

        Args:
            stream: Stream ID to synchronize
        """
        with self._lock:
            # In simulation, just mark all events as synchronized
            # Real implementation would wait for GPU operations

            # Mark all events as synchronized
            for event in self._events[stream]:
                event.synchronized = True

            self._stream_timestamps[stream] = time.time()

            # Clear dependencies after sync
            self._dependencies[stream] = set()

    def synchronize_all(self):
        """Synchronize all streams (global barrier)."""
        with self._lock:
            for stream in range(self._num_streams):
                # Mark all events as synchronized
                for event in self._events[stream]:
                    event.synchronized = True
                self._stream_timestamps[stream] = time.time()
                self._dependencies[stream] = set()

    def create_barrier(self, streams: List[int]) -> int:
        """
        Create a synchronization barrier for multiple streams.

        Args:
            streams: List of stream IDs to synchronize

        Returns:
            Barrier ID for tracking
        """
        with self._lock:
            # Record events on all streams
            events = []
            for stream in streams:
                event = StreamEvent(
                    stream=stream,
                    event_id=self._next_event_id
                )
                self._next_event_id += 1
                self._events[stream].append(event)
                self._stream_timestamps[stream] = time.time()
                events.append(event)

            # Make all streams wait for each other
            for stream in streams:
                for event in events:
                    if event.stream != stream:
                        self._dependencies[stream].add(event.stream)

            return self._next_event_id

    def query_stream(self, stream: int) -> bool:
        """
        Query if a stream has completed all operations.

        Args:
            stream: Stream ID to query

        Returns:
            True if stream is idle
        """
        with self._lock:
            if not self._events[stream]:
                return True

            return all(e.synchronized for e in self._events[stream])

    def get_stream_dependencies(self, stream: int) -> Set[int]:
        """Get streams that this stream depends on."""
        return self._dependencies.get(stream, set())


# =============================================================================
# CPU Offloader
# =============================================================================

@dataclass
class OffloadEntry:
    """Tracks an offloaded memory block."""
    block: MemoryBlock
    cpu_ptr: int
    gpu_ptr: int
    size: int
    offloaded_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0


class CPUOffloader:
    """
    Automated CPU offloading for memory management.

    Features:
    - Automatic offload on memory pressure
    - LRU-based eviction
    - Prefetch on access
    - Memory budget management
    """

    def __init__(
        self,
        gpu_memory_limit: int = 8 * 1024 * 1024 * 1024,  # 8GB
        cpu_memory_limit: int = 32 * 1024 * 1024 * 1024,  # 32GB
        offload_threshold: float = 0.8,
        prefetch_on_access: bool = True
    ):
        self._lock = threading.Lock()
        self._gpu_limit = gpu_memory_limit
        self._cpu_limit = cpu_memory_limit
        self._offload_threshold = offload_threshold
        self._prefetch_on_access = prefetch_on_access

        # Memory tracking
        self._gpu_used = 0
        self._cpu_used = 0

        # Offload tracking
        self._on_gpu: Dict[int, OffloadEntry] = {}  # gpu_ptr -> entry
        self._on_cpu: Dict[int, OffloadEntry] = {}  # cpu_ptr -> entry

        # LRU tracking
        self._access_order: List[int] = []  # gpu_ptrs in access order

        # CPU pointer allocation
        self._next_cpu_ptr = 0x400000

        # Statistics
        self._stats = {
            'offloads': 0,
            'reloads': 0,
            'bytes_offloaded': 0,
            'bytes_reloaded': 0,
            'automatic_offloads': 0,
        }

    def register(self, block: MemoryBlock):
        """
        Register a GPU memory block for offload management.

        Args:
            block: GPU memory block to manage
        """
        with self._lock:
            if block.ptr in self._on_gpu:
                return

            entry = OffloadEntry(
                block=block,
                cpu_ptr=0,  # Not allocated yet
                gpu_ptr=block.ptr,
                size=block.size
            )

            self._on_gpu[block.ptr] = entry
            self._gpu_used += block.size
            self._access_order.append(block.ptr)

            # Check if offload needed
            self._maybe_offload()

    def unregister(self, block: MemoryBlock):
        """
        Unregister a memory block from offload management.

        Args:
            block: Memory block to unregister
        """
        with self._lock:
            if block.ptr in self._on_gpu:
                entry = self._on_gpu.pop(block.ptr)
                self._gpu_used -= entry.size
                if block.ptr in self._access_order:
                    self._access_order.remove(block.ptr)

            # Also clean up any offloaded version
            for cpu_ptr, entry in list(self._on_cpu.items()):
                if entry.gpu_ptr == block.ptr:
                    del self._on_cpu[cpu_ptr]
                    self._cpu_used -= entry.size

    def access(self, block: MemoryBlock) -> int:
        """
        Record access to a memory block and ensure it's on GPU.

        Args:
            block: Memory block being accessed

        Returns:
            GPU pointer (may have been reloaded)
        """
        with self._lock:
            gpu_ptr = block.ptr

            # Check if on GPU
            if gpu_ptr in self._on_gpu:
                entry = self._on_gpu[gpu_ptr]
                entry.access_count += 1
                entry.last_accessed = time.time()

                # Update LRU order
                if gpu_ptr in self._access_order:
                    self._access_order.remove(gpu_ptr)
                self._access_order.append(gpu_ptr)

                return gpu_ptr

            # Check if offloaded to CPU
            for cpu_ptr, entry in self._on_cpu.items():
                if entry.gpu_ptr == gpu_ptr:
                    if self._prefetch_on_access:
                        return self._reload(entry)
                    return gpu_ptr

            # Not found
            return gpu_ptr

    def offload(self, block: MemoryBlock) -> Optional[int]:
        """
        Manually offload a block to CPU.

        Args:
            block: Memory block to offload

        Returns:
            CPU pointer where data was offloaded
        """
        with self._lock:
            if block.ptr not in self._on_gpu:
                return None

            return self._do_offload(block.ptr)

    def reload(self, cpu_ptr: int) -> Optional[int]:
        """
        Reload an offloaded block back to GPU.

        Args:
            cpu_ptr: CPU pointer of offloaded data

        Returns:
            GPU pointer where data was reloaded
        """
        with self._lock:
            if cpu_ptr not in self._on_cpu:
                return None

            entry = self._on_cpu[cpu_ptr]
            return self._reload(entry)

    def _maybe_offload(self):
        """Check if automatic offload is needed."""
        if self._gpu_used > self._gpu_limit * self._offload_threshold:
            # Offload LRU blocks until under threshold
            target = int(self._gpu_limit * self._offload_threshold * 0.8)

            while self._gpu_used > target and self._access_order:
                lru_ptr = self._access_order[0]
                self._do_offload(lru_ptr)
                self._stats['automatic_offloads'] += 1

    def _do_offload(self, gpu_ptr: int) -> int:
        """Perform the actual offload operation."""
        entry = self._on_gpu.pop(gpu_ptr)

        # Allocate CPU memory
        cpu_ptr = self._next_cpu_ptr
        self._next_cpu_ptr += entry.size

        entry.cpu_ptr = cpu_ptr
        self._on_cpu[cpu_ptr] = entry

        # Update tracking
        self._gpu_used -= entry.size
        self._cpu_used += entry.size

        if gpu_ptr in self._access_order:
            self._access_order.remove(gpu_ptr)

        self._stats['offloads'] += 1
        self._stats['bytes_offloaded'] += entry.size

        logger.debug(f"Offloaded {entry.size} bytes from GPU {gpu_ptr} to CPU {cpu_ptr}")

        return cpu_ptr

    def _reload(self, entry: OffloadEntry) -> int:
        """Reload data from CPU to GPU."""
        # Move back to GPU tracking
        del self._on_cpu[entry.cpu_ptr]
        self._on_gpu[entry.gpu_ptr] = entry

        # Update tracking
        self._cpu_used -= entry.size
        self._gpu_used += entry.size

        entry.last_accessed = time.time()
        entry.access_count += 1
        self._access_order.append(entry.gpu_ptr)

        self._stats['reloads'] += 1
        self._stats['bytes_reloaded'] += entry.size

        # May need to offload other blocks
        self._maybe_offload()

        logger.debug(f"Reloaded {entry.size} bytes from CPU to GPU {entry.gpu_ptr}")

        return entry.gpu_ptr

    def get_gpu_usage(self) -> float:
        """Get current GPU memory usage as fraction."""
        return self._gpu_used / self._gpu_limit

    def get_cpu_usage(self) -> float:
        """Get current CPU memory usage as fraction."""
        return self._cpu_used / self._cpu_limit

    def get_statistics(self) -> Dict[str, Any]:
        """Get offloading statistics."""
        return {
            **self._stats,
            'gpu_used': self._gpu_used,
            'gpu_limit': self._gpu_limit,
            'gpu_usage_pct': self._gpu_used / self._gpu_limit * 100,
            'cpu_used': self._cpu_used,
            'cpu_limit': self._cpu_limit,
            'blocks_on_gpu': len(self._on_gpu),
            'blocks_on_cpu': len(self._on_cpu),
        }

    def get_offloaded_blocks(self) -> List[OffloadEntry]:
        """Get list of blocks currently offloaded to CPU."""
        return list(self._on_cpu.values())


# =============================================================================
# Unified Memory Manager with All Phase 6 Features
# =============================================================================

class AdvancedMemoryManager:
    """
    Unified memory manager with all advanced features.

    Combines:
    - P2P transfers
    - Prefetching
    - Stream synchronization
    - CPU offloading
    """

    def __init__(
        self,
        num_gpus: int = 2,
        num_streams: int = 16,
        gpu_memory_limit: int = 8 * 1024 * 1024 * 1024
    ):
        self.p2p = P2PTransferManager(num_gpus)
        self.prefetch = PrefetchManager()
        self.streams = StreamSynchronizer(num_streams)
        self.offloader = CPUOffloader(gpu_memory_limit=gpu_memory_limit)

        self._lock = threading.Lock()

    def transfer_between_gpus(
        self,
        src_ptr: int,
        dst_ptr: int,
        size: int,
        src_gpu: int,
        dst_gpu: int,
        stream: int = 0
    ) -> Optional[TransferEvent]:
        """Transfer memory between GPUs."""
        return self.p2p.transfer(
            src_ptr, dst_ptr, size,
            src_gpu, dst_gpu, stream
        )

    def prefetch_to_gpu(
        self,
        block: MemoryBlock,
        gpu_id: int = 0,
        stream: int = 0
    ) -> PrefetchRequest:
        """Prefetch block to GPU."""
        return self.prefetch.prefetch_to_device(
            block, gpu_id, stream
        )

    def prefetch_to_cpu(
        self,
        block: MemoryBlock,
        stream: int = 0
    ) -> PrefetchRequest:
        """Prefetch block to CPU (evict from GPU)."""
        return self.prefetch.prefetch_to_host(block, stream)

    def synchronize(self, stream: int):
        """Synchronize a stream."""
        self.streams.synchronize_stream(stream)

    def synchronize_all(self):
        """Synchronize all streams."""
        self.streams.synchronize_all()

    def manage_block(self, block: MemoryBlock):
        """Register block for automatic offload management."""
        self.offloader.register(block)

    def access_block(self, block: MemoryBlock) -> int:
        """Access a block, reloading if necessary."""
        return self.offloader.access(block)

    def get_all_statistics(self) -> Dict[str, Any]:
        """Get combined statistics from all managers."""
        return {
            'p2p': self.p2p.get_statistics(),
            'prefetch': self.prefetch.get_statistics(),
            'offloader': self.offloader.get_statistics(),
        }


# =============================================================================
# Multi-GPU Load Balancing Strategy
# =============================================================================

class LoadBalanceStrategy(Enum):
    """Strategy for distributing allocations across GPUs."""
    ROUND_ROBIN = auto()      # Distribute evenly in round-robin fashion
    LEAST_UTILIZED = auto()   # Allocate on GPU with most free memory
    LOCALITY_AWARE = auto()   # Consider data locality for transfers
    MEMORY_PRESSURE = auto()  # Avoid GPUs under memory pressure
    MANUAL = auto()           # User specifies device explicitly


@dataclass
class GPUDeviceState:
    """State information for a single GPU device."""
    device_id: int
    total_memory: int
    allocated_memory: int = 0
    reserved_memory: int = 0
    num_allocations: int = 0
    is_available: bool = True
    memory_pressure: float = 0.0  # 0.0 to 1.0

    @property
    def free_memory(self) -> int:
        return self.total_memory - self.allocated_memory

    @property
    def utilization(self) -> float:
        return self.allocated_memory / self.total_memory if self.total_memory > 0 else 0.0


# =============================================================================
# Multi-GPU Allocator
# =============================================================================

class MultiGPUAllocator:
    """
    Multi-GPU memory allocator with load balancing.

    Features:
    - Per-device allocators
    - P2P access management
    - Load-balanced allocation
    - Cross-GPU memory transfers
    - Unified statistics
    """

    def __init__(
        self,
        num_gpus: int = 2,
        memory_per_gpu: int = 8 * 1024 * 1024 * 1024,  # 8GB default
        strategy: LoadBalanceStrategy = LoadBalanceStrategy.LEAST_UTILIZED,
        enable_p2p: bool = True
    ):
        from .allocator import CachingAllocator

        self._lock = threading.RLock()
        self._num_gpus = num_gpus
        self._strategy = strategy

        # Create per-device allocators
        self._allocators: Dict[int, Any] = {}
        self._device_states: Dict[int, GPUDeviceState] = {}

        for device_id in range(num_gpus):
            config = MemoryConfig(
                device_type=DeviceType.CUDA,
                device_id=device_id,
                max_memory=memory_per_gpu
            )
            self._allocators[device_id] = CachingAllocator(config)
            self._device_states[device_id] = GPUDeviceState(
                device_id=device_id,
                total_memory=memory_per_gpu
            )

        # P2P access management
        self._p2p_manager = P2PTransferManager(num_gpus)
        if enable_p2p:
            self._enable_all_p2p()

        # Allocation tracking
        self._allocations: Dict[int, Tuple[int, MemoryBlock]] = {}  # ptr -> (device_id, block)
        self._round_robin_idx = 0

        # Statistics
        self._stats = {
            'total_allocations': 0,
            'cross_gpu_transfers': 0,
            'load_balance_decisions': 0,
            'allocation_failures': 0,
        }

    def _enable_all_p2p(self):
        """Enable P2P access between all adjacent GPUs."""
        for i in range(self._num_gpus):
            for j in range(self._num_gpus):
                if i != j:
                    self._p2p_manager.enable_peer_access(i, j)

    def allocate(
        self,
        size: int,
        device_id: Optional[int] = None,
        stream: int = 0
    ) -> Optional[MemoryBlock]:
        """
        Allocate memory on a GPU.

        Args:
            size: Size in bytes
            device_id: Target GPU (None for automatic selection)
            stream: CUDA stream for allocation

        Returns:
            MemoryBlock on success, None on failure
        """
        with self._lock:
            # Select device
            if device_id is None:
                device_id = self._select_device(size)
                self._stats['load_balance_decisions'] += 1

            if device_id < 0 or device_id >= self._num_gpus:
                logger.error(f"Invalid device_id: {device_id}")
                return None

            # Check memory availability
            state = self._device_states[device_id]
            if state.free_memory < size:
                # Try other GPUs
                alternate = self._find_device_with_memory(size, exclude=device_id)
                if alternate is not None:
                    device_id = alternate
                    state = self._device_states[device_id]
                else:
                    self._stats['allocation_failures'] += 1
                    return None

            # Allocate on selected device
            allocator = self._allocators[device_id]
            block = allocator.allocate(size, stream)

            if block is None:
                self._stats['allocation_failures'] += 1
                return None

            # Update tracking
            block.device_id = device_id
            self._allocations[block.ptr] = (device_id, block)

            state.allocated_memory += block.size
            state.num_allocations += 1
            state.memory_pressure = state.utilization

            self._stats['total_allocations'] += 1

            return block

    def free(self, block: MemoryBlock):
        """Free memory block."""
        with self._lock:
            if block.ptr not in self._allocations:
                return

            device_id, _ = self._allocations.pop(block.ptr)
            allocator = self._allocators[device_id]
            allocator.free(block)

            state = self._device_states[device_id]
            state.allocated_memory -= block.size
            state.num_allocations -= 1
            state.memory_pressure = state.utilization

    def _select_device(self, size: int) -> int:
        """Select device based on load balancing strategy."""
        if self._strategy == LoadBalanceStrategy.ROUND_ROBIN:
            device = self._round_robin_idx
            self._round_robin_idx = (self._round_robin_idx + 1) % self._num_gpus
            return device

        elif self._strategy == LoadBalanceStrategy.LEAST_UTILIZED:
            return min(
                self._device_states.values(),
                key=lambda s: s.utilization
            ).device_id

        elif self._strategy == LoadBalanceStrategy.MEMORY_PRESSURE:
            # Find device with lowest pressure that has enough memory
            candidates = [
                s for s in self._device_states.values()
                if s.free_memory >= size and s.memory_pressure < 0.9
            ]
            if candidates:
                return min(candidates, key=lambda s: s.memory_pressure).device_id
            return self._select_device_fallback(size)

        elif self._strategy == LoadBalanceStrategy.LOCALITY_AWARE:
            # For now, use least utilized (could be extended with access patterns)
            return min(
                self._device_states.values(),
                key=lambda s: s.utilization
            ).device_id

        # Default: first available GPU
        return 0

    def _select_device_fallback(self, size: int) -> int:
        """Fallback device selection."""
        for state in sorted(self._device_states.values(), key=lambda s: s.utilization):
            if state.free_memory >= size:
                return state.device_id
        return 0

    def _find_device_with_memory(self, size: int, exclude: int) -> Optional[int]:
        """Find a device with enough memory."""
        for state in sorted(self._device_states.values(), key=lambda s: s.free_memory, reverse=True):
            if state.device_id != exclude and state.free_memory >= size:
                return state.device_id
        return None

    def transfer(
        self,
        block: MemoryBlock,
        target_device: int,
        stream: int = 0
    ) -> Optional[MemoryBlock]:
        """
        Transfer a block to another GPU.

        Args:
            block: Source memory block
            target_device: Target GPU ID
            stream: CUDA stream for transfer

        Returns:
            New MemoryBlock on target device
        """
        with self._lock:
            if block.ptr not in self._allocations:
                return None

            source_device, _ = self._allocations[block.ptr]

            if source_device == target_device:
                return block  # Already on target

            # Allocate on target
            target_block = self.allocate(block.size, target_device, stream)
            if target_block is None:
                return None

            # Initiate transfer
            event = self._p2p_manager.transfer(
                block.ptr,
                target_block.ptr,
                block.size,
                source_device,
                target_device,
                stream
            )

            if event is None:
                # Transfer failed, free target and return None
                self.free(target_block)
                return None

            self._stats['cross_gpu_transfers'] += 1

            return target_block

    def replicate(
        self,
        block: MemoryBlock,
        target_devices: List[int],
        stream: int = 0
    ) -> Dict[int, MemoryBlock]:
        """
        Replicate a block to multiple GPUs.

        Args:
            block: Source memory block
            target_devices: List of target GPU IDs
            stream: CUDA stream for transfers

        Returns:
            Dict mapping device_id to replicated block
        """
        with self._lock:
            if block.ptr not in self._allocations:
                return {}

            source_device, _ = self._allocations[block.ptr]
            replicas = {source_device: block}

            for device_id in target_devices:
                if device_id != source_device:
                    replica = self.transfer(block, device_id, stream)
                    if replica:
                        replicas[device_id] = replica

            return replicas

    def set_strategy(self, strategy: LoadBalanceStrategy):
        """Change the load balancing strategy."""
        with self._lock:
            self._strategy = strategy

    def get_device_state(self, device_id: int) -> Optional[GPUDeviceState]:
        """Get state for a specific device."""
        return self._device_states.get(device_id)

    def get_all_device_states(self) -> List[GPUDeviceState]:
        """Get states for all devices."""
        return list(self._device_states.values())

    def get_total_stats(self) -> MemoryStats:
        """Get combined statistics across all GPUs."""
        stats = MemoryStats()

        for device_id, allocator in self._allocators.items():
            device_stats = allocator.get_stats()
            stats.allocated += device_stats.allocated
            stats.reserved += device_stats.reserved
            stats.active += device_stats.active
            stats.inactive += device_stats.inactive
            stats.num_allocs += device_stats.num_allocs
            stats.num_frees += device_stats.num_frees
            stats.num_ooms += device_stats.num_ooms
            stats.peak_allocated = max(stats.peak_allocated, device_stats.peak_allocated)

        return stats

    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics."""
        total_stats = self.get_total_stats()

        device_info = {}
        for device_id, state in self._device_states.items():
            device_info[f'gpu_{device_id}'] = {
                'total_memory': state.total_memory,
                'allocated_memory': state.allocated_memory,
                'free_memory': state.free_memory,
                'utilization': state.utilization,
                'num_allocations': state.num_allocations,
                'memory_pressure': state.memory_pressure,
            }

        return {
            **self._stats,
            'strategy': self._strategy.name,
            'num_gpus': self._num_gpus,
            'total_allocated': total_stats.allocated,
            'total_active': total_stats.active,
            'devices': device_info,
            'p2p': self._p2p_manager.get_statistics(),
        }

    def empty_all_caches(self):
        """Empty caches on all devices."""
        with self._lock:
            for allocator in self._allocators.values():
                allocator.empty_cache()


# =============================================================================
# Distributed Tensor Manager
# =============================================================================

@dataclass
class DistributedTensor:
    """Represents a tensor distributed across multiple GPUs."""
    name: str
    shape: Tuple[int, ...]
    dtype: str
    shards: Dict[int, MemoryBlock]  # device_id -> block
    total_size: int
    shard_dim: int = 0  # Dimension along which tensor is sharded

    @property
    def num_shards(self) -> int:
        return len(self.shards)

    @property
    def devices(self) -> List[int]:
        return list(self.shards.keys())


class DistributedTensorManager:
    """
    Manages tensors distributed across multiple GPUs.

    Features:
    - Automatic sharding
    - Gather/scatter operations
    - All-reduce support
    - Gradient synchronization
    """

    def __init__(self, allocator: MultiGPUAllocator):
        self._lock = threading.Lock()
        self._allocator = allocator
        self._tensors: Dict[str, DistributedTensor] = {}

        # Statistics
        self._stats = {
            'tensors_created': 0,
            'tensors_deleted': 0,
            'gather_ops': 0,
            'scatter_ops': 0,
            'allreduce_ops': 0,
        }

    def create_distributed(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: str = 'float32',
        devices: Optional[List[int]] = None,
        shard_dim: int = 0
    ) -> Optional[DistributedTensor]:
        """
        Create a distributed tensor across GPUs.

        Args:
            name: Tensor name
            shape: Total tensor shape
            dtype: Data type
            devices: GPUs to distribute across (None for all)
            shard_dim: Dimension to shard along

        Returns:
            DistributedTensor on success
        """
        with self._lock:
            return self._create_distributed_locked(
                name, shape, dtype, devices, shard_dim
            )

    def _create_distributed_locked(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: str = 'float32',
        devices: Optional[List[int]] = None,
        shard_dim: int = 0
    ) -> Optional[DistributedTensor]:
        """
        Create a distributed tensor. Assumes ``self._lock`` is already held.

        This no-lock helper exists so that other public methods which already
        hold ``self._lock`` (e.g. ``scatter``) can create tensors without
        re-acquiring the non-reentrant lock, which would deadlock.
        """
        if name in self._tensors:
            logger.warning(f"Tensor {name} already exists")
            return None

        if devices is None:
            devices = list(range(self._allocator._num_gpus))

        # Calculate shard sizes
        dtype_size = {'float32': 4, 'float16': 2, 'float64': 8, 'int32': 4, 'int64': 8}.get(dtype, 4)
        total_elements = 1
        for dim in shape:
            total_elements *= dim
        total_size = total_elements * dtype_size

        # Calculate per-shard size
        shard_count = len(devices)
        elements_per_shard = total_elements // shard_count
        shard_size = elements_per_shard * dtype_size

        # Allocate shards
        shards = {}
        for device_id in devices:
            block = self._allocator.allocate(shard_size, device_id)
            if block is None:
                # Cleanup on failure
                for allocated_block in shards.values():
                    self._allocator.free(allocated_block)
                return None
            shards[device_id] = block

        tensor = DistributedTensor(
            name=name,
            shape=shape,
            dtype=dtype,
            shards=shards,
            total_size=total_size,
            shard_dim=shard_dim
        )

        self._tensors[name] = tensor
        self._stats['tensors_created'] += 1

        return tensor

    def delete_tensor(self, name: str) -> bool:
        """Delete a distributed tensor."""
        with self._lock:
            if name not in self._tensors:
                return False

            tensor = self._tensors.pop(name)

            for block in tensor.shards.values():
                self._allocator.free(block)

            self._stats['tensors_deleted'] += 1
            return True

    def gather(
        self,
        tensor: DistributedTensor,
        target_device: int
    ) -> Optional[MemoryBlock]:
        """
        Gather all shards to a single device.

        Args:
            tensor: Distributed tensor to gather
            target_device: Target GPU for gathered data

        Returns:
            MemoryBlock containing gathered data
        """
        with self._lock:
            # Allocate space for full tensor
            gathered = self._allocator.allocate(tensor.total_size, target_device)
            if gathered is None:
                return None

            # Transfer each shard (simulation)
            for device_id, shard in tensor.shards.items():
                if device_id != target_device:
                    self._allocator._p2p_manager.transfer(
                        shard.ptr,
                        gathered.ptr,  # Would need offset calculation
                        shard.size,
                        device_id,
                        target_device
                    )

            self._stats['gather_ops'] += 1
            return gathered

    def scatter(
        self,
        source_block: MemoryBlock,
        source_device: int,
        target_devices: List[int],
        name: str,
        shape: Tuple[int, ...],
        dtype: str = 'float32'
    ) -> Optional[DistributedTensor]:
        """
        Scatter data from one device to multiple devices.

        Args:
            source_block: Source memory block
            source_device: Source GPU
            target_devices: Target GPUs
            name: Name for resulting distributed tensor
            shape: Tensor shape
            dtype: Data type

        Returns:
            DistributedTensor with scattered data
        """
        with self._lock:
            # Use the no-lock helper: we already hold self._lock, and the
            # non-reentrant threading.Lock would deadlock on re-acquisition.
            tensor = self._create_distributed_locked(name, shape, dtype, target_devices)
            if tensor is None:
                return None

            # Transfer shards (simulation)
            shard_size = source_block.size // len(target_devices)
            for i, device_id in enumerate(target_devices):
                if device_id != source_device:
                    self._allocator._p2p_manager.transfer(
                        source_block.ptr + i * shard_size,
                        tensor.shards[device_id].ptr,
                        shard_size,
                        source_device,
                        device_id
                    )

            self._stats['scatter_ops'] += 1
            return tensor

    def all_reduce(
        self,
        tensor: DistributedTensor,
        operation: str = 'sum'
    ) -> bool:
        """
        Perform all-reduce across tensor shards.

        Args:
            tensor: Distributed tensor
            operation: Reduction operation ('sum', 'mean', 'max', 'min')

        Returns:
            True on success
        """
        with self._lock:
            # In a real implementation, this would perform
            # ring all-reduce or tree all-reduce

            # Simulation: transfer between all pairs
            devices = list(tensor.shards.keys())
            for i, src_device in enumerate(devices):
                dst_device = devices[(i + 1) % len(devices)]
                if src_device != dst_device:
                    self._allocator._p2p_manager.transfer(
                        tensor.shards[src_device].ptr,
                        tensor.shards[dst_device].ptr,
                        tensor.shards[src_device].size,
                        src_device,
                        dst_device
                    )

            self._stats['allreduce_ops'] += 1
            return True

    def get_tensor(self, name: str) -> Optional[DistributedTensor]:
        """Get a distributed tensor by name."""
        return self._tensors.get(name)

    def list_tensors(self) -> List[str]:
        """List all distributed tensor names."""
        return list(self._tensors.keys())

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics."""
        total_memory = sum(
            sum(s.size for s in t.shards.values())
            for t in self._tensors.values()
        )

        return {
            **self._stats,
            'num_tensors': len(self._tensors),
            'total_distributed_memory': total_memory,
        }


# =============================================================================
# Gradient Synchronizer for Distributed Training
# =============================================================================

class GradientSynchronizer:
    """
    Synchronizes gradients across GPUs for distributed training.

    Features:
    - Ring all-reduce
    - Gradient compression (optional)
    - Overlapped communication
    """

    def __init__(self, allocator: MultiGPUAllocator, num_gpus: int = 2):
        self._lock = threading.Lock()
        self._allocator = allocator
        self._num_gpus = num_gpus

        # Gradient buffers
        self._grad_buffers: Dict[int, List[MemoryBlock]] = {
            i: [] for i in range(num_gpus)
        }

        # Statistics
        self._stats = {
            'sync_rounds': 0,
            'bytes_synchronized': 0,
            'compression_ratio': 1.0,
        }

    def register_gradients(
        self,
        device_id: int,
        gradient_blocks: List[MemoryBlock]
    ):
        """Register gradient blocks for a device."""
        with self._lock:
            self._grad_buffers[device_id] = gradient_blocks

    def synchronize(self, use_compression: bool = False) -> bool:
        """
        Synchronize gradients across all GPUs.

        Args:
            use_compression: Whether to use gradient compression

        Returns:
            True on success
        """
        with self._lock:
            # Ring all-reduce pattern
            for round_idx in range(self._num_gpus - 1):
                for device_id in range(self._num_gpus):
                    next_device = (device_id + 1) % self._num_gpus

                    for grad_block in self._grad_buffers[device_id]:
                        # Transfer gradient
                        transfer_size = grad_block.size
                        if use_compression:
                            transfer_size = transfer_size // 4  # 4x compression

                        self._allocator._p2p_manager.transfer(
                            grad_block.ptr,
                            grad_block.ptr,  # Would be receiving buffer
                            transfer_size,
                            device_id,
                            next_device
                        )

                        self._stats['bytes_synchronized'] += transfer_size

            self._stats['sync_rounds'] += 1
            if use_compression:
                self._stats['compression_ratio'] = 4.0

            return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get synchronization statistics."""
        return {
            **self._stats,
            'num_gpus': self._num_gpus,
            'registered_buffers': sum(len(b) for b in self._grad_buffers.values()),
        }
