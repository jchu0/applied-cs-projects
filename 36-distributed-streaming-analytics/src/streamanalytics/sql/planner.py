"""Physical Planner - Convert logical plans to distributed physical plans.

Handles join strategy selection, data distribution, exchange operators,
and partitioning for distributed query execution.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Any

from .ast import Expression, AggregateExpr, DataType
from .logical import (
    LogicalPlan, LogicalScan, LogicalFilter, LogicalProject, LogicalJoin,
    LogicalAggregate, LogicalSort, LogicalLimit, LogicalUnion, LogicalDistinct,
    Schema, Statistics,
)


class JoinStrategy(Enum):
    """Physical join strategies."""

    HASH = auto()          # Hash join - good for large tables
    BROADCAST = auto()     # Broadcast smaller table to all nodes
    MERGE = auto()         # Sort-merge join - good for sorted inputs
    NESTED_LOOP = auto()   # Nested loop - fallback for complex predicates


class DistributionType(Enum):
    """Data distribution types."""

    SINGLETON = auto()     # Single partition
    HASH = auto()          # Hash partitioned
    BROADCAST = auto()     # Replicated to all nodes
    RANGE = auto()         # Range partitioned
    ROUND_ROBIN = auto()   # Round-robin partitioned
    RANDOM = auto()        # Random distribution


class AggregateStrategy(Enum):
    """Aggregate execution strategies."""

    SINGLE = auto()        # Single-phase aggregation
    TWO_PHASE = auto()     # Partial + final aggregation
    STREAMING = auto()     # Streaming aggregation (for sorted input)


@dataclass
class DataDistribution:
    """Describes how data is distributed."""

    dist_type: DistributionType = DistributionType.SINGLETON
    partition_columns: list[str] = field(default_factory=list)
    partition_count: int = 1

    def requires_exchange(self, required: DataDistribution) -> bool:
        """Check if exchange is needed to satisfy required distribution."""
        if required.dist_type == DistributionType.SINGLETON:
            return self.dist_type != DistributionType.SINGLETON

        if required.dist_type == DistributionType.BROADCAST:
            return self.dist_type != DistributionType.BROADCAST

        if required.dist_type == DistributionType.HASH:
            if self.dist_type != DistributionType.HASH:
                return True
            return set(self.partition_columns) != set(required.partition_columns)

        return False


@dataclass
class PartitionSpec:
    """Partition specification for data."""

    partition_columns: list[str] = field(default_factory=list)
    partition_count: int = 1
    partition_function: str = "hash"  # hash, range, round_robin


# ============================================================================
# Physical Plan Nodes
# ============================================================================

@dataclass
class PhysicalPlan(ABC):
    """Base class for physical plan nodes."""

    schema: Schema = field(default_factory=Schema)
    stats: Optional[Statistics] = None
    distribution: DataDistribution = field(default_factory=DataDistribution)
    cost: float = 0.0

    @abstractmethod
    def children(self) -> list[PhysicalPlan]:
        """Get child plan nodes."""
        pass

    @abstractmethod
    def to_string(self, indent: int = 0) -> str:
        """Convert to readable string representation."""
        pass

    @abstractmethod
    def output_partitioning(self) -> DataDistribution:
        """Get output data distribution."""
        pass


@dataclass
class PhysicalScan(PhysicalPlan):
    """Physical table scan operator."""

    table_name: str = ""
    table_schema: Optional[str] = None
    projection: Optional[list[int]] = None
    filters: list[Expression] = field(default_factory=list)
    partition_filters: list[Expression] = field(default_factory=list)

    def children(self) -> list[PhysicalPlan]:
        return []

    def to_string(self, indent: int = 0) -> str:
        prefix = "  " * indent
        name = f"{self.table_schema}.{self.table_name}" if self.table_schema else self.table_name
        proj_str = f" [cols: {self.projection}]" if self.projection else ""
        filter_str = ""
        if self.filters:
            filter_str = f" (filters: {len(self.filters)})"
        return f"{prefix}PhysicalScan: {name}{proj_str}{filter_str}"

    def output_partitioning(self) -> DataDistribution:
        return self.distribution


@dataclass
class PhysicalFilter(PhysicalPlan):
    """Physical filter operator."""

    predicate: Expression = field(default_factory=lambda: None)
    input: PhysicalPlan = field(default_factory=lambda: PhysicalScan())

    def children(self) -> list[PhysicalPlan]:
        return [self.input]

    def to_string(self, indent: int = 0) -> str:
        prefix = "  " * indent
        lines = [f"{prefix}PhysicalFilter: {self.predicate.to_sql() if self.predicate else 'None'}"]
        lines.append(self.input.to_string(indent + 1))
        return "\n".join(lines)

    def output_partitioning(self) -> DataDistribution:
        return self.input.output_partitioning()


@dataclass
class PhysicalProject(PhysicalPlan):
    """Physical projection operator."""

    expressions: list[Expression] = field(default_factory=list)
    input: PhysicalPlan = field(default_factory=lambda: PhysicalScan())

    def children(self) -> list[PhysicalPlan]:
        return [self.input]

    def to_string(self, indent: int = 0) -> str:
        prefix = "  " * indent
        exprs_str = ", ".join(e.to_sql() for e in self.expressions)
        lines = [f"{prefix}PhysicalProject: {exprs_str}"]
        lines.append(self.input.to_string(indent + 1))
        return "\n".join(lines)

    def output_partitioning(self) -> DataDistribution:
        return self.input.output_partitioning()


@dataclass
class PhysicalHashJoin(PhysicalPlan):
    """Hash join operator."""

    left: PhysicalPlan = field(default_factory=lambda: PhysicalScan())
    right: PhysicalPlan = field(default_factory=lambda: PhysicalScan())
    left_keys: list[Expression] = field(default_factory=list)
    right_keys: list[Expression] = field(default_factory=list)
    join_type: str = "INNER"
    condition: Optional[Expression] = None
    build_side: str = "right"  # Which side to build hash table from

    def children(self) -> list[PhysicalPlan]:
        return [self.left, self.right]

    def to_string(self, indent: int = 0) -> str:
        prefix = "  " * indent
        left_keys_str = ", ".join(k.to_sql() for k in self.left_keys)
        right_keys_str = ", ".join(k.to_sql() for k in self.right_keys)
        lines = [f"{prefix}PhysicalHashJoin: {self.join_type} (build={self.build_side})"]
        lines.append(f"{prefix}  Left Keys: [{left_keys_str}]")
        lines.append(f"{prefix}  Right Keys: [{right_keys_str}]")
        if self.condition:
            lines.append(f"{prefix}  Condition: {self.condition.to_sql()}")
        lines.append(self.left.to_string(indent + 1))
        lines.append(self.right.to_string(indent + 1))
        return "\n".join(lines)

    def output_partitioning(self) -> DataDistribution:
        # Hash join preserves partitioning of probe side (non-build side)
        if self.build_side == "right":
            return self.left.output_partitioning()
        return self.right.output_partitioning()


@dataclass
class PhysicalBroadcastJoin(PhysicalPlan):
    """Broadcast hash join - broadcast smaller table."""

    left: PhysicalPlan = field(default_factory=lambda: PhysicalScan())
    right: PhysicalPlan = field(default_factory=lambda: PhysicalScan())
    broadcast_side: str = "right"  # Which side to broadcast
    left_keys: list[Expression] = field(default_factory=list)
    right_keys: list[Expression] = field(default_factory=list)
    join_type: str = "INNER"
    condition: Optional[Expression] = None

    def children(self) -> list[PhysicalPlan]:
        return [self.left, self.right]

    def to_string(self, indent: int = 0) -> str:
        prefix = "  " * indent
        lines = [f"{prefix}PhysicalBroadcastJoin: {self.join_type} (broadcast={self.broadcast_side})"]
        lines.append(self.left.to_string(indent + 1))
        lines.append(self.right.to_string(indent + 1))
        return "\n".join(lines)

    def output_partitioning(self) -> DataDistribution:
        # Preserves partitioning of non-broadcast side
        if self.broadcast_side == "right":
            return self.left.output_partitioning()
        return self.right.output_partitioning()


@dataclass
class PhysicalMergeJoin(PhysicalPlan):
    """Sort-merge join operator."""

    left: PhysicalPlan = field(default_factory=lambda: PhysicalScan())
    right: PhysicalPlan = field(default_factory=lambda: PhysicalScan())
    left_keys: list[Expression] = field(default_factory=list)
    right_keys: list[Expression] = field(default_factory=list)
    join_type: str = "INNER"

    def children(self) -> list[PhysicalPlan]:
        return [self.left, self.right]

    def to_string(self, indent: int = 0) -> str:
        prefix = "  " * indent
        lines = [f"{prefix}PhysicalMergeJoin: {self.join_type}"]
        lines.append(self.left.to_string(indent + 1))
        lines.append(self.right.to_string(indent + 1))
        return "\n".join(lines)

    def output_partitioning(self) -> DataDistribution:
        return self.left.output_partitioning()


@dataclass
class PhysicalAggregate(PhysicalPlan):
    """Physical aggregation operator."""

    group_by: list[Expression] = field(default_factory=list)
    aggregates: list[AggregateExpr] = field(default_factory=list)
    input: PhysicalPlan = field(default_factory=lambda: PhysicalScan())
    strategy: AggregateStrategy = AggregateStrategy.SINGLE
    is_partial: bool = False  # For two-phase aggregation

    def children(self) -> list[PhysicalPlan]:
        return [self.input]

    def to_string(self, indent: int = 0) -> str:
        prefix = "  " * indent
        phase = "Partial" if self.is_partial else "Final"
        group_str = ", ".join(e.to_sql() for e in self.group_by) if self.group_by else "[]"
        agg_str = ", ".join(a.to_sql() for a in self.aggregates)
        lines = [f"{prefix}PhysicalAggregate ({phase}, {self.strategy.name}):"]
        lines.append(f"{prefix}  Group By: {group_str}")
        lines.append(f"{prefix}  Aggregates: {agg_str}")
        lines.append(self.input.to_string(indent + 1))
        return "\n".join(lines)

    def output_partitioning(self) -> DataDistribution:
        if self.group_by:
            # Grouped aggregation preserves hash partitioning on group keys
            return DataDistribution(
                dist_type=DistributionType.HASH,
                partition_columns=[e.to_sql() for e in self.group_by],
            )
        # Global aggregation produces singleton
        return DataDistribution(dist_type=DistributionType.SINGLETON)


@dataclass
class PhysicalSort(PhysicalPlan):
    """Physical sort operator."""

    sort_expressions: list[Expression] = field(default_factory=list)
    ascending: list[bool] = field(default_factory=list)
    nulls_first: list[bool] = field(default_factory=list)
    input: PhysicalPlan = field(default_factory=lambda: PhysicalScan())
    global_sort: bool = False  # Whether this is a global or local sort

    def children(self) -> list[PhysicalPlan]:
        return [self.input]

    def to_string(self, indent: int = 0) -> str:
        prefix = "  " * indent
        scope = "Global" if self.global_sort else "Local"
        sorts = []
        for expr, asc in zip(self.sort_expressions, self.ascending):
            order = "ASC" if asc else "DESC"
            sorts.append(f"{expr.to_sql()} {order}")
        lines = [f"{prefix}PhysicalSort ({scope}): [{', '.join(sorts)}]"]
        lines.append(self.input.to_string(indent + 1))
        return "\n".join(lines)

    def output_partitioning(self) -> DataDistribution:
        if self.global_sort:
            return DataDistribution(dist_type=DistributionType.SINGLETON)
        return self.input.output_partitioning()


@dataclass
class PhysicalLimit(PhysicalPlan):
    """Physical limit operator."""

    limit: Optional[int] = None
    offset: int = 0
    input: PhysicalPlan = field(default_factory=lambda: PhysicalScan())
    is_local: bool = False  # Local limit per partition vs global

    def children(self) -> list[PhysicalPlan]:
        return [self.input]

    def to_string(self, indent: int = 0) -> str:
        prefix = "  " * indent
        scope = "Local" if self.is_local else "Global"
        lines = [f"{prefix}PhysicalLimit ({scope}): limit={self.limit}, offset={self.offset}"]
        lines.append(self.input.to_string(indent + 1))
        return "\n".join(lines)

    def output_partitioning(self) -> DataDistribution:
        if self.is_local:
            return self.input.output_partitioning()
        return DataDistribution(dist_type=DistributionType.SINGLETON)


@dataclass
class PhysicalExchange(PhysicalPlan):
    """Data exchange/shuffle operator."""

    input: PhysicalPlan = field(default_factory=lambda: PhysicalScan())
    target_distribution: DataDistribution = field(default_factory=DataDistribution)
    exchange_type: str = "hash"  # hash, broadcast, gather, round_robin

    def children(self) -> list[PhysicalPlan]:
        return [self.input]

    def to_string(self, indent: int = 0) -> str:
        prefix = "  " * indent
        dist_info = f"{self.exchange_type}"
        if self.target_distribution.partition_columns:
            cols = ", ".join(self.target_distribution.partition_columns)
            dist_info += f" by [{cols}]"
        lines = [f"{prefix}PhysicalExchange: {dist_info}"]
        lines.append(self.input.to_string(indent + 1))
        return "\n".join(lines)

    def output_partitioning(self) -> DataDistribution:
        return self.target_distribution


@dataclass
class PhysicalUnion(PhysicalPlan):
    """Physical union operator."""

    inputs: list[PhysicalPlan] = field(default_factory=list)
    all: bool = False

    def children(self) -> list[PhysicalPlan]:
        return self.inputs

    def to_string(self, indent: int = 0) -> str:
        prefix = "  " * indent
        all_str = "ALL " if self.all else ""
        lines = [f"{prefix}PhysicalUnion {all_str}"]
        for inp in self.inputs:
            lines.append(inp.to_string(indent + 1))
        return "\n".join(lines)

    def output_partitioning(self) -> DataDistribution:
        # Union produces random distribution
        return DataDistribution(dist_type=DistributionType.RANDOM)


# ============================================================================
# Physical Planner
# ============================================================================

@dataclass
class PlannerConfig:
    """Configuration for physical planner."""

    broadcast_threshold_bytes: int = 10 * 1024 * 1024  # 10MB
    broadcast_threshold_rows: int = 100000
    target_partition_count: int = 200
    prefer_merge_join: bool = False
    enable_adaptive: bool = True


class PhysicalPlanner:
    """Converts logical plans to physical plans."""

    def __init__(self, config: Optional[PlannerConfig] = None):
        """Initialize with configuration."""
        self.config = config or PlannerConfig()

    def plan(self, logical: LogicalPlan) -> PhysicalPlan:
        """Convert logical plan to physical plan."""
        physical = self._plan_node(logical)
        physical = self._ensure_root_distribution(physical)
        return physical

    def _plan_node(self, logical: LogicalPlan) -> PhysicalPlan:
        """Plan a single logical node."""
        if isinstance(logical, LogicalScan):
            return self._plan_scan(logical)
        elif isinstance(logical, LogicalFilter):
            return self._plan_filter(logical)
        elif isinstance(logical, LogicalProject):
            return self._plan_project(logical)
        elif isinstance(logical, LogicalJoin):
            return self._plan_join(logical)
        elif isinstance(logical, LogicalAggregate):
            return self._plan_aggregate(logical)
        elif isinstance(logical, LogicalSort):
            return self._plan_sort(logical)
        elif isinstance(logical, LogicalLimit):
            return self._plan_limit(logical)
        elif isinstance(logical, LogicalUnion):
            return self._plan_union(logical)
        elif isinstance(logical, LogicalDistinct):
            return self._plan_distinct(logical)
        else:
            raise ValueError(f"Unknown logical plan type: {type(logical)}")

    def _plan_scan(self, logical: LogicalScan) -> PhysicalPlan:
        """Plan a table scan."""
        return PhysicalScan(
            table_name=logical.table_name,
            table_schema=logical.table_schema,
            projection=logical.projection,
            schema=logical.schema,
            stats=logical.stats,
            distribution=DataDistribution(
                dist_type=DistributionType.HASH,
                partition_count=self.config.target_partition_count,
            ),
        )

    def _plan_filter(self, logical: LogicalFilter) -> PhysicalPlan:
        """Plan a filter."""
        input_plan = self._plan_node(logical.input)
        return PhysicalFilter(
            predicate=logical.predicate,
            input=input_plan,
            schema=logical.schema,
        )

    def _plan_project(self, logical: LogicalProject) -> PhysicalPlan:
        """Plan a projection."""
        input_plan = self._plan_node(logical.input)
        return PhysicalProject(
            expressions=logical.expressions,
            input=input_plan,
            schema=logical.schema,
        )

    def _plan_join(self, logical: LogicalJoin) -> PhysicalPlan:
        """Plan a join with strategy selection."""
        left_plan = self._plan_node(logical.left)
        right_plan = self._plan_node(logical.right)

        # Extract join keys from condition
        left_keys, right_keys, remaining = self._extract_join_keys(logical.condition)

        # Select join strategy
        strategy = self._select_join_strategy(logical, left_plan, right_plan)

        if strategy == JoinStrategy.BROADCAST:
            broadcast_side = self._select_broadcast_side(left_plan, right_plan)
            return self._plan_broadcast_join(
                left_plan, right_plan, left_keys, right_keys,
                logical.join_type.name, remaining, broadcast_side
            )
        elif strategy == JoinStrategy.MERGE:
            return self._plan_merge_join(
                left_plan, right_plan, left_keys, right_keys,
                logical.join_type.name
            )
        else:  # HASH join
            return self._plan_hash_join(
                left_plan, right_plan, left_keys, right_keys,
                logical.join_type.name, remaining
            )

    def _extract_join_keys(
        self, condition: Optional[Expression]
    ) -> tuple[list[Expression], list[Expression], Optional[Expression]]:
        """Extract equi-join keys from join condition."""
        if condition is None:
            return [], [], None

        from .ast import BinaryOp, ColumnRef

        left_keys = []
        right_keys = []
        remaining_parts = []

        def extract_from_and(expr: Expression):
            if isinstance(expr, BinaryOp) and expr.operator == "AND":
                extract_from_and(expr.left)
                extract_from_and(expr.right)
            elif isinstance(expr, BinaryOp) and expr.operator == "=":
                if isinstance(expr.left, ColumnRef) and isinstance(expr.right, ColumnRef):
                    left_keys.append(expr.left)
                    right_keys.append(expr.right)
                else:
                    remaining_parts.append(expr)
            else:
                remaining_parts.append(expr)

        extract_from_and(condition)

        remaining = None
        if remaining_parts:
            from .ast import Literal, DataType
            remaining = remaining_parts[0]
            for part in remaining_parts[1:]:
                remaining = BinaryOp(operator="AND", left=remaining, right=part)

        return left_keys, right_keys, remaining

    def _select_join_strategy(
        self, logical: LogicalJoin, left: PhysicalPlan, right: PhysicalPlan
    ) -> JoinStrategy:
        """Select optimal join strategy."""
        left_size = self._estimate_size(left)
        right_size = self._estimate_size(right)

        # Check for broadcast opportunity
        if right_size < self.config.broadcast_threshold_bytes:
            return JoinStrategy.BROADCAST
        if left_size < self.config.broadcast_threshold_bytes:
            return JoinStrategy.BROADCAST

        # Check for merge join opportunity
        if self.config.prefer_merge_join:
            return JoinStrategy.MERGE

        # Default to hash join
        return JoinStrategy.HASH

    def _estimate_size(self, plan: PhysicalPlan) -> int:
        """Estimate data size in bytes."""
        if plan.stats and plan.stats.size_bytes:
            return plan.stats.size_bytes
        if plan.stats and plan.stats.row_count:
            # Assume 100 bytes per row as default
            return plan.stats.row_count * 100
        return self.config.broadcast_threshold_bytes + 1  # Assume large

    def _select_broadcast_side(
        self, left: PhysicalPlan, right: PhysicalPlan
    ) -> str:
        """Select which side to broadcast."""
        left_size = self._estimate_size(left)
        right_size = self._estimate_size(right)
        return "right" if right_size <= left_size else "left"

    def _plan_broadcast_join(
        self, left: PhysicalPlan, right: PhysicalPlan,
        left_keys: list[Expression], right_keys: list[Expression],
        join_type: str, condition: Optional[Expression], broadcast_side: str
    ) -> PhysicalPlan:
        """Create broadcast join plan."""
        # Add broadcast exchange to smaller side
        if broadcast_side == "right":
            right = PhysicalExchange(
                input=right,
                target_distribution=DataDistribution(dist_type=DistributionType.BROADCAST),
                exchange_type="broadcast",
            )
        else:
            left = PhysicalExchange(
                input=left,
                target_distribution=DataDistribution(dist_type=DistributionType.BROADCAST),
                exchange_type="broadcast",
            )

        return PhysicalBroadcastJoin(
            left=left,
            right=right,
            broadcast_side=broadcast_side,
            left_keys=left_keys,
            right_keys=right_keys,
            join_type=join_type,
            condition=condition,
        )

    def _plan_hash_join(
        self, left: PhysicalPlan, right: PhysicalPlan,
        left_keys: list[Expression], right_keys: list[Expression],
        join_type: str, condition: Optional[Expression]
    ) -> PhysicalPlan:
        """Create hash join plan with shuffle."""
        # Add hash exchange on join keys
        if left_keys:
            left_cols = [k.to_sql() for k in left_keys]
            left = PhysicalExchange(
                input=left,
                target_distribution=DataDistribution(
                    dist_type=DistributionType.HASH,
                    partition_columns=left_cols,
                    partition_count=self.config.target_partition_count,
                ),
                exchange_type="hash",
            )

            right_cols = [k.to_sql() for k in right_keys]
            right = PhysicalExchange(
                input=right,
                target_distribution=DataDistribution(
                    dist_type=DistributionType.HASH,
                    partition_columns=right_cols,
                    partition_count=self.config.target_partition_count,
                ),
                exchange_type="hash",
            )

        # Build side is smaller table
        build_side = "right" if self._estimate_size(right) <= self._estimate_size(left) else "left"

        return PhysicalHashJoin(
            left=left,
            right=right,
            left_keys=left_keys,
            right_keys=right_keys,
            join_type=join_type,
            condition=condition,
            build_side=build_side,
        )

    def _plan_merge_join(
        self, left: PhysicalPlan, right: PhysicalPlan,
        left_keys: list[Expression], right_keys: list[Expression],
        join_type: str
    ) -> PhysicalPlan:
        """Create sort-merge join plan."""
        # Sort both sides on join keys
        if left_keys:
            left = PhysicalSort(
                sort_expressions=left_keys,
                ascending=[True] * len(left_keys),
                nulls_first=[True] * len(left_keys),
                input=left,
            )
            right = PhysicalSort(
                sort_expressions=right_keys,
                ascending=[True] * len(right_keys),
                nulls_first=[True] * len(right_keys),
                input=right,
            )

        return PhysicalMergeJoin(
            left=left,
            right=right,
            left_keys=left_keys,
            right_keys=right_keys,
            join_type=join_type,
        )

    def _plan_aggregate(self, logical: LogicalAggregate) -> PhysicalPlan:
        """Plan aggregation with two-phase strategy."""
        input_plan = self._plan_node(logical.input)

        # Determine aggregation strategy
        if not logical.group_by:
            # Global aggregation needs two-phase
            strategy = AggregateStrategy.TWO_PHASE
        elif self._is_pre_partitioned(input_plan, logical.group_by):
            # Data already partitioned on group keys
            strategy = AggregateStrategy.SINGLE
        else:
            strategy = AggregateStrategy.TWO_PHASE

        if strategy == AggregateStrategy.TWO_PHASE:
            # Partial aggregation (local)
            partial = PhysicalAggregate(
                group_by=logical.group_by,
                aggregates=logical.aggregates,
                input=input_plan,
                strategy=strategy,
                is_partial=True,
                schema=logical.schema,
            )

            # Exchange on group keys
            if logical.group_by:
                group_cols = [e.to_sql() for e in logical.group_by]
                partial = PhysicalExchange(
                    input=partial,
                    target_distribution=DataDistribution(
                        dist_type=DistributionType.HASH,
                        partition_columns=group_cols,
                        partition_count=self.config.target_partition_count,
                    ),
                    exchange_type="hash",
                )
            else:
                # Global aggregation gathers to single partition
                partial = PhysicalExchange(
                    input=partial,
                    target_distribution=DataDistribution(dist_type=DistributionType.SINGLETON),
                    exchange_type="gather",
                )

            # Final aggregation
            return PhysicalAggregate(
                group_by=logical.group_by,
                aggregates=logical.aggregates,
                input=partial,
                strategy=strategy,
                is_partial=False,
                schema=logical.schema,
            )
        else:
            return PhysicalAggregate(
                group_by=logical.group_by,
                aggregates=logical.aggregates,
                input=input_plan,
                strategy=strategy,
                is_partial=False,
                schema=logical.schema,
            )

    def _is_pre_partitioned(
        self, plan: PhysicalPlan, group_by: list[Expression]
    ) -> bool:
        """Check if plan is already partitioned on group by columns."""
        dist = plan.output_partitioning()
        if dist.dist_type != DistributionType.HASH:
            return False

        group_cols = {e.to_sql() for e in group_by}
        return set(dist.partition_columns) == group_cols

    def _plan_sort(self, logical: LogicalSort) -> PhysicalPlan:
        """Plan sort with exchange for global ordering."""
        input_plan = self._plan_node(logical.input)

        # Local sort first
        local_sort = PhysicalSort(
            sort_expressions=logical.sort_expressions,
            ascending=logical.ascending,
            nulls_first=logical.nulls_first,
            input=input_plan,
            global_sort=False,
            schema=logical.schema,
        )

        # Gather to single partition for global sort
        gathered = PhysicalExchange(
            input=local_sort,
            target_distribution=DataDistribution(dist_type=DistributionType.SINGLETON),
            exchange_type="gather",
        )

        # Final sort
        return PhysicalSort(
            sort_expressions=logical.sort_expressions,
            ascending=logical.ascending,
            nulls_first=logical.nulls_first,
            input=gathered,
            global_sort=True,
            schema=logical.schema,
        )

    def _plan_limit(self, logical: LogicalLimit) -> PhysicalPlan:
        """Plan limit with local + global strategy."""
        input_plan = self._plan_node(logical.input)

        # Local limit first (limit + offset per partition)
        local_limit = PhysicalLimit(
            limit=logical.limit + logical.offset if logical.limit else None,
            offset=0,
            input=input_plan,
            is_local=True,
            schema=logical.schema,
        )

        # Gather to single partition
        gathered = PhysicalExchange(
            input=local_limit,
            target_distribution=DataDistribution(dist_type=DistributionType.SINGLETON),
            exchange_type="gather",
        )

        # Final limit with offset
        return PhysicalLimit(
            limit=logical.limit,
            offset=logical.offset,
            input=gathered,
            is_local=False,
            schema=logical.schema,
        )

    def _plan_union(self, logical: LogicalUnion) -> PhysicalPlan:
        """Plan union operator."""
        left_plan = self._plan_node(logical.left)
        right_plan = self._plan_node(logical.right)

        return PhysicalUnion(
            inputs=[left_plan, right_plan],
            all=logical.all,
            schema=logical.schema,
        )

    def _plan_distinct(self, logical: LogicalDistinct) -> PhysicalPlan:
        """Plan distinct using aggregation."""
        input_plan = self._plan_node(logical.input)

        # Use aggregation with group by all columns
        # Schema should have all output columns
        from .ast import ColumnRef
        group_by = [
            ColumnRef(name=col_name)
            for col_name, _ in logical.schema.columns
        ]

        return PhysicalAggregate(
            group_by=group_by,
            aggregates=[],
            input=input_plan,
            strategy=AggregateStrategy.TWO_PHASE,
            schema=logical.schema,
        )

    def _ensure_root_distribution(self, plan: PhysicalPlan) -> PhysicalPlan:
        """Ensure root produces singleton distribution for final result."""
        if plan.output_partitioning().dist_type == DistributionType.SINGLETON:
            return plan

        return PhysicalExchange(
            input=plan,
            target_distribution=DataDistribution(dist_type=DistributionType.SINGLETON),
            exchange_type="gather",
            schema=plan.schema,
        )

    def explain(self, plan: PhysicalPlan) -> str:
        """Generate explain output for physical plan."""
        lines = ["=== Physical Plan ==="]
        lines.append(plan.to_string())
        lines.append("")
        lines.append("=== Statistics ===")
        lines.append(f"Estimated cost: {plan.cost:.2f}")
        return "\n".join(lines)
