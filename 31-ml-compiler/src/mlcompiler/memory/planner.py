"""Memory planning and allocation for ML compiler."""

from dataclasses import dataclass, field
from typing import Any
import logging
from enum import Enum

from ..ir import IRModule, Function, Operation, OpCode, Value, TensorType

logger = logging.getLogger(__name__)


class AllocationStrategy(Enum):
    """Memory allocation strategies."""
    GREEDY = "greedy"
    LINEAR_SCAN = "linear_scan"
    GRAPH_COLORING = "graph_coloring"
    BEST_FIT = "best_fit"


@dataclass
class Lifetime:
    """Lifetime of a value."""
    value_id: str
    start: int  # First use
    end: int    # Last use
    size_bytes: int

    @property
    def duration(self) -> int:
        return self.end - self.start


@dataclass
class BufferAllocation:
    """Buffer allocation info."""
    value_id: str
    offset: int
    size: int
    memory_space: str = "global"


@dataclass
class MemoryPlan:
    """Complete memory plan for a function."""
    allocations: dict[str, BufferAllocation]
    total_size: int
    peak_memory: int
    reuse_count: int = 0


class LifetimeAnalyzer:
    """Analyzes value lifetimes in IR."""

    def analyze(self, func: Function) -> dict[str, Lifetime]:
        """Analyze lifetimes of all values.

        Args:
            func: Function to analyze

        Returns:
            Dictionary of value ID to lifetime
        """
        lifetimes = {}

        # Assign indices to operations
        op_index = {}
        idx = 0
        for block in func.body.blocks:
            for op in block.operations:
                op_index[op.id] = idx
                idx += 1

        # Track first and last use of each value
        value_first_use = {}
        value_last_use = {}

        # Process arguments
        for i, arg in enumerate(func.arguments):
            value_first_use[arg.id] = 0
            value_last_use[arg.id] = 0

        # Process operations
        for block in func.body.blocks:
            for op in block.operations:
                op_idx = op_index[op.id]

                # Record definition
                for output in op.outputs:
                    if output.id not in value_first_use:
                        value_first_use[output.id] = op_idx
                    value_last_use[output.id] = op_idx

                # Record uses
                for inp in op.inputs:
                    if inp.id not in value_first_use:
                        value_first_use[inp.id] = op_idx
                    value_last_use[inp.id] = max(
                        value_last_use.get(inp.id, 0),
                        op_idx
                    )

        # Create lifetimes
        for value_id in value_first_use:
            # Find value to get size
            value = self._find_value(func, value_id)
            if value:
                size = value.type.size_bytes
            else:
                size = 0

            lifetimes[value_id] = Lifetime(
                value_id=value_id,
                start=value_first_use[value_id],
                end=value_last_use.get(value_id, value_first_use[value_id]),
                size_bytes=size
            )

        return lifetimes

    def _find_value(self, func: Function, value_id: str) -> Value:
        """Find value by ID."""
        # Check arguments
        for arg in func.arguments:
            if arg.id == value_id:
                return arg

        # Check operation outputs
        for block in func.body.blocks:
            for op in block.operations:
                for output in op.outputs:
                    if output.id == value_id:
                        return output

        return None


class MemoryPlanner:
    """Plans memory allocation for IR functions."""

    def __init__(self, strategy: AllocationStrategy = AllocationStrategy.GREEDY):
        """Initialize memory planner.

        Args:
            strategy: Allocation strategy to use
        """
        self.strategy = strategy
        self.lifetime_analyzer = LifetimeAnalyzer()

    def plan(self, func: Function) -> MemoryPlan:
        """Create memory plan for function.

        Args:
            func: Function to plan memory for

        Returns:
            Memory plan
        """
        # Analyze lifetimes
        lifetimes = self.lifetime_analyzer.analyze(func)

        # Allocate based on strategy
        if self.strategy == AllocationStrategy.GREEDY:
            return self._plan_greedy(lifetimes)
        elif self.strategy == AllocationStrategy.LINEAR_SCAN:
            return self._plan_linear_scan(lifetimes)
        elif self.strategy == AllocationStrategy.BEST_FIT:
            return self._plan_best_fit(lifetimes)
        else:
            return self._plan_greedy(lifetimes)

    def _plan_greedy(self, lifetimes: dict[str, Lifetime]) -> MemoryPlan:
        """Greedy allocation with buffer reuse."""
        allocations = {}
        free_buffers = []  # (offset, size, end_time)
        current_offset = 0
        peak_memory = 0
        reuse_count = 0

        # Sort by start time
        sorted_values = sorted(
            lifetimes.values(),
            key=lambda l: (l.start, -l.size_bytes)
        )

        for lifetime in sorted_values:
            # Free expired buffers
            newly_free = []
            for offset, size, end_time in free_buffers:
                if end_time < lifetime.start:
                    newly_free.append((offset, size, end_time))

            for item in newly_free:
                free_buffers.remove(item)

            # Try to reuse buffer
            best_fit = None
            best_waste = float('inf')

            for i, (offset, size, end_time) in enumerate(free_buffers):
                if size >= lifetime.size_bytes:
                    waste = size - lifetime.size_bytes
                    if waste < best_waste:
                        best_waste = waste
                        best_fit = i

            if best_fit is not None:
                # Reuse buffer
                offset, size, _ = free_buffers[best_fit]
                del free_buffers[best_fit]
                allocations[lifetime.value_id] = BufferAllocation(
                    value_id=lifetime.value_id,
                    offset=offset,
                    size=size
                )
                # Return to free pool when done
                free_buffers.append((offset, size, lifetime.end))
                reuse_count += 1
            else:
                # Allocate new
                allocations[lifetime.value_id] = BufferAllocation(
                    value_id=lifetime.value_id,
                    offset=current_offset,
                    size=lifetime.size_bytes
                )
                free_buffers.append((current_offset, lifetime.size_bytes, lifetime.end))
                current_offset += lifetime.size_bytes

            # Track peak
            active_memory = sum(
                lifetimes[alloc.value_id].size_bytes
                for alloc in allocations.values()
                if lifetimes[alloc.value_id].start <= lifetime.start <= lifetimes[alloc.value_id].end
            )
            peak_memory = max(peak_memory, active_memory)

        return MemoryPlan(
            allocations=allocations,
            total_size=current_offset,
            peak_memory=peak_memory,
            reuse_count=reuse_count
        )

    def _plan_linear_scan(self, lifetimes: dict[str, Lifetime]) -> MemoryPlan:
        """Linear scan allocation."""
        allocations = {}
        active = []  # (end, offset, size)
        current_offset = 0
        peak_memory = 0
        reuse_count = 0

        # Sort by start
        sorted_values = sorted(lifetimes.values(), key=lambda l: l.start)

        for lifetime in sorted_values:
            # Expire old allocations
            expired = [(e, o, s) for e, o, s in active if e < lifetime.start]
            free_space = []
            for end, offset, size in expired:
                active.remove((end, offset, size))
                free_space.append((offset, size))

            # Try to reuse
            allocated = False
            for offset, size in sorted(free_space, key=lambda x: x[1]):
                if size >= lifetime.size_bytes:
                    allocations[lifetime.value_id] = BufferAllocation(
                        value_id=lifetime.value_id,
                        offset=offset,
                        size=size
                    )
                    active.append((lifetime.end, offset, size))
                    allocated = True
                    reuse_count += 1
                    break

            if not allocated:
                allocations[lifetime.value_id] = BufferAllocation(
                    value_id=lifetime.value_id,
                    offset=current_offset,
                    size=lifetime.size_bytes
                )
                active.append((lifetime.end, current_offset, lifetime.size_bytes))
                current_offset += lifetime.size_bytes

            # Track peak
            current_memory = sum(s for _, _, s in active)
            peak_memory = max(peak_memory, current_memory)

        return MemoryPlan(
            allocations=allocations,
            total_size=current_offset,
            peak_memory=peak_memory,
            reuse_count=reuse_count
        )

    def _plan_best_fit(self, lifetimes: dict[str, Lifetime]) -> MemoryPlan:
        """Best-fit allocation strategy."""
        allocations = {}
        holes = []  # (offset, size)
        current_offset = 0
        peak_memory = 0
        reuse_count = 0

        # Sort by size descending
        sorted_values = sorted(
            lifetimes.values(),
            key=lambda l: -l.size_bytes
        )

        for lifetime in sorted_values:
            # Find best fit hole
            best_hole = None
            best_idx = -1

            for i, (offset, size) in enumerate(holes):
                if size >= lifetime.size_bytes:
                    if best_hole is None or size < best_hole[1]:
                        best_hole = (offset, size)
                        best_idx = i

            if best_hole:
                # Use hole
                offset, size = best_hole
                del holes[best_idx]

                allocations[lifetime.value_id] = BufferAllocation(
                    value_id=lifetime.value_id,
                    offset=offset,
                    size=lifetime.size_bytes
                )

                # Create smaller hole if space left
                if size > lifetime.size_bytes:
                    new_offset = offset + lifetime.size_bytes
                    new_size = size - lifetime.size_bytes
                    holes.append((new_offset, new_size))

                reuse_count += 1
            else:
                # Allocate at end
                allocations[lifetime.value_id] = BufferAllocation(
                    value_id=lifetime.value_id,
                    offset=current_offset,
                    size=lifetime.size_bytes
                )
                current_offset += lifetime.size_bytes

            peak_memory = max(peak_memory, current_offset)

        return MemoryPlan(
            allocations=allocations,
            total_size=current_offset,
            peak_memory=peak_memory,
            reuse_count=reuse_count
        )


class InplaceOptimizer:
    """Optimizes operations for in-place execution."""

    def optimize(self, func: Function, lifetimes: dict[str, Lifetime]) -> dict[str, str]:
        """Find in-place opportunities.

        Args:
            func: Function to optimize
            lifetimes: Value lifetimes

        Returns:
            Mapping of output value ID to input value ID for in-place ops
        """
        inplace = {}

        for block in func.body.blocks:
            for op in block.operations:
                # Only elementwise ops can be in-place
                if not op.is_elementwise:
                    continue

                # Check if input's lifetime ends here
                for inp in op.inputs:
                    if inp.id in lifetimes:
                        lifetime = lifetimes[inp.id]
                        # If last use and same shape, can be in-place
                        if len(inp.uses) == 1:
                            if op.outputs and inp.type.shape == op.outputs[0].type.shape:
                                inplace[op.outputs[0].id] = inp.id
                                break

        return inplace


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_allocated: int
    peak_memory: int
    buffer_reuse_rate: float
    fragmentation: float


def analyze_memory_usage(plan: MemoryPlan) -> MemoryStats:
    """Analyze memory plan statistics.

    Args:
        plan: Memory plan

    Returns:
        Memory statistics
    """
    total_requested = sum(alloc.size for alloc in plan.allocations.values())
    reuse_rate = plan.reuse_count / max(len(plan.allocations), 1)

    # Estimate fragmentation
    if plan.total_size > 0:
        fragmentation = 1.0 - (total_requested / plan.total_size)
    else:
        fragmentation = 0.0

    return MemoryStats(
        total_allocated=plan.total_size,
        peak_memory=plan.peak_memory,
        buffer_reuse_rate=reuse_rate,
        fragmentation=max(0, fragmentation)
    )
