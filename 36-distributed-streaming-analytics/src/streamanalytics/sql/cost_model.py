"""Cost-Based Query Optimizer.

Implements cost models, cardinality estimation, statistics propagation,
and dynamic programming join optimization for distributed query execution.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Any, FrozenSet
import math
import functools

from .ast import Expression, ColumnRef, Literal, BinaryOp, UnaryOp, DataType
from .logical import (
    LogicalPlan, LogicalScan, LogicalFilter, LogicalProject, LogicalJoin,
    LogicalAggregate, LogicalSort, LogicalLimit, LogicalUnion, LogicalDistinct,
    Schema, Statistics,
)
from .optimizer import OptimizationRule


# ============================================================================
# Cost Constants
# ============================================================================

@dataclass
class CostConstants:
    """Cost model constants for different operations."""

    # CPU costs (per row)
    cpu_tuple_cost: float = 0.01      # Cost to process a tuple
    cpu_index_cost: float = 0.005     # Cost of an index lookup
    cpu_comparison_cost: float = 0.0025  # Cost of a comparison
    cpu_hash_cost: float = 0.02       # Cost of hashing a tuple
    cpu_sort_comparison: float = 0.03  # Cost of comparison during sort

    # I/O costs (per page/block)
    seq_page_cost: float = 1.0        # Sequential disk I/O
    random_page_cost: float = 4.0     # Random disk I/O

    # Network costs (for distributed)
    network_transfer_cost: float = 0.1    # Per KB transferred
    network_latency_cost: float = 10.0    # Per message latency

    # Memory costs
    memory_tuple_cost: float = 0.001  # Cost of memory operations

    # Hash join specific
    hash_build_factor: float = 2.0    # Factor for building hash table
    hash_probe_factor: float = 1.0    # Factor for probing hash table

    # Sort-merge specific
    sort_factor: float = 1.5          # log(N) factor for sorting
    merge_factor: float = 1.0         # Factor for merging

    # Default statistics
    default_row_count: int = 1000     # Default if unknown
    default_row_width: int = 100      # Bytes per row
    page_size: int = 8192             # Block size in bytes


# ============================================================================
# Statistics Estimator
# ============================================================================

@dataclass
class ColumnStatistics:
    """Statistics for a single column."""

    distinct_count: int = 0
    null_fraction: float = 0.0
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    histogram: Optional[list[tuple[Any, float]]] = None  # (bucket_bound, frequency)
    most_common_values: Optional[list[tuple[Any, float]]] = None  # (value, frequency)


@dataclass
class TableStatistics:
    """Statistics for a table."""

    row_count: int = 0
    row_width_bytes: int = 100
    column_stats: dict[str, ColumnStatistics] = field(default_factory=dict)

    @property
    def size_bytes(self) -> int:
        return self.row_count * self.row_width_bytes


class StatisticsEstimator:
    """Estimates cardinality and selectivity for query operations."""

    def __init__(self, constants: Optional[CostConstants] = None):
        self.constants = constants or CostConstants()
        self.table_stats: dict[str, TableStatistics] = {}

    def register_table(self, table_name: str, stats: TableStatistics) -> None:
        """Register statistics for a table."""
        self.table_stats[table_name] = stats

    def estimate_scan(self, scan: LogicalScan) -> Statistics:
        """Estimate statistics for a table scan."""
        table_stats = self.table_stats.get(scan.table_name)

        if table_stats:
            row_count = table_stats.row_count
            size_bytes = table_stats.size_bytes
        else:
            row_count = self.constants.default_row_count
            size_bytes = row_count * self.constants.default_row_width

        # Apply projection if present
        if scan.projection and scan.schema.columns:
            proj_ratio = len(scan.projection) / len(scan.schema.columns)
            size_bytes = int(size_bytes * proj_ratio)

        return Statistics(
            row_count=row_count,
            size_bytes=size_bytes,
        )

    def estimate_filter(self, filter_node: LogicalFilter, input_stats: Statistics) -> Statistics:
        """Estimate statistics after a filter operation."""
        selectivity = self.estimate_selectivity(filter_node.predicate)

        row_count = int((input_stats.row_count or self.constants.default_row_count) * selectivity)
        row_count = max(1, row_count)  # At least 1 row

        size_bytes = None
        if input_stats.size_bytes and input_stats.row_count:
            bytes_per_row = input_stats.size_bytes / input_stats.row_count
            size_bytes = int(row_count * bytes_per_row)

        return Statistics(
            row_count=row_count,
            size_bytes=size_bytes,
        )

    def estimate_selectivity(self, predicate: Expression) -> float:
        """Estimate selectivity of a predicate (0.0 to 1.0)."""
        if predicate is None:
            return 1.0

        if isinstance(predicate, Literal):
            if predicate.value is True:
                return 1.0
            elif predicate.value is False:
                return 0.0
            return 0.5

        if isinstance(predicate, BinaryOp):
            return self._estimate_binary_selectivity(predicate)

        if isinstance(predicate, UnaryOp):
            if predicate.operator == "NOT":
                return 1.0 - self.estimate_selectivity(predicate.operand)
            return 0.5

        # Default selectivity for unknown predicates
        return 0.5

    def _estimate_binary_selectivity(self, expr: BinaryOp) -> float:
        """Estimate selectivity of a binary operation."""
        op = expr.operator.upper()

        # Logical operators
        if op == "AND":
            left_sel = self.estimate_selectivity(expr.left)
            right_sel = self.estimate_selectivity(expr.right)
            return left_sel * right_sel  # Independence assumption

        if op == "OR":
            left_sel = self.estimate_selectivity(expr.left)
            right_sel = self.estimate_selectivity(expr.right)
            # P(A OR B) = P(A) + P(B) - P(A AND B)
            return min(1.0, left_sel + right_sel - left_sel * right_sel)

        # Comparison operators
        if op == "=":
            # Try to use column statistics
            col_stats = self._get_column_stats(expr.left) or self._get_column_stats(expr.right)
            if col_stats and col_stats.distinct_count > 0:
                return 1.0 / col_stats.distinct_count
            return 0.1  # Default for equality

        if op == "<>":
            return 1.0 - 0.1  # Complement of equality

        if op in ("<", "<=", ">", ">="):
            # Range predicates - assume uniform distribution
            col_stats = self._get_column_stats(expr.left)
            if col_stats and col_stats.min_value is not None and col_stats.max_value is not None:
                return self._estimate_range_selectivity(expr, col_stats)
            return 0.33  # Default for range

        if op == "LIKE":
            # LIKE selectivity heuristics
            if isinstance(expr.right, Literal) and isinstance(expr.right.value, str):
                pattern = expr.right.value
                if pattern.startswith('%'):
                    return 0.5  # Leading wildcard is expensive
                elif '%' in pattern:
                    return 0.1  # Some wildcard
                else:
                    return 0.05  # Exact match
            return 0.2

        if op == "IN":
            # IN clause - estimate based on number of values
            return 0.25  # Default

        if op == "BETWEEN":
            return 0.25  # Range predicate

        return 0.5  # Unknown operator

    def _get_column_stats(self, expr: Expression) -> Optional[ColumnStatistics]:
        """Get column statistics if expression is a column reference."""
        if not isinstance(expr, ColumnRef):
            return None

        table = expr.table or ""
        col_name = expr.name

        if table in self.table_stats:
            return self.table_stats[table].column_stats.get(col_name)
        return None

    def _estimate_range_selectivity(self, expr: BinaryOp, col_stats: ColumnStatistics) -> float:
        """Estimate selectivity for range predicates using column statistics."""
        if col_stats.min_value is None or col_stats.max_value is None:
            return 0.33

        # Get the comparison value
        value = None
        if isinstance(expr.right, Literal):
            value = expr.right.value
        elif isinstance(expr.left, Literal):
            value = expr.left.value

        if value is None:
            return 0.33

        try:
            min_val = float(col_stats.min_value)
            max_val = float(col_stats.max_value)
            val = float(value)

            if max_val == min_val:
                return 0.5

            position = (val - min_val) / (max_val - min_val)
            position = max(0.0, min(1.0, position))

            op = expr.operator.upper()
            if op in ("<", "<="):
                return position
            elif op in (">", ">="):
                return 1.0 - position
        except (ValueError, TypeError):
            pass

        return 0.33

    def estimate_join(
        self,
        left_stats: Statistics,
        right_stats: Statistics,
        join_type: str,
        condition: Optional[Expression] = None
    ) -> Statistics:
        """Estimate statistics for a join operation."""
        left_rows = left_stats.row_count or self.constants.default_row_count
        right_rows = right_stats.row_count or self.constants.default_row_count

        # Estimate join selectivity
        if condition:
            join_selectivity = self.estimate_selectivity(condition)
        else:
            # Cross join
            join_selectivity = 1.0

        # Calculate output cardinality based on join type
        join_type_upper = join_type.upper()

        if join_type_upper == "INNER":
            # For equi-joins, use FK/PK estimation
            output_rows = int(max(left_rows, right_rows) * join_selectivity)
        elif join_type_upper == "LEFT":
            # At least left_rows, potentially more with multiple matches
            output_rows = int(left_rows * (1 + join_selectivity * (right_rows / left_rows - 1)))
            output_rows = max(left_rows, output_rows)
        elif join_type_upper == "RIGHT":
            output_rows = int(right_rows * (1 + join_selectivity * (left_rows / right_rows - 1)))
            output_rows = max(right_rows, output_rows)
        elif join_type_upper == "FULL":
            output_rows = int(left_rows + right_rows - left_rows * right_rows * join_selectivity)
        elif join_type_upper == "CROSS":
            output_rows = left_rows * right_rows
        else:
            output_rows = int(left_rows * right_rows * join_selectivity)

        output_rows = max(1, output_rows)

        # Estimate output size
        left_width = (left_stats.size_bytes or 0) // max(1, left_rows)
        right_width = (right_stats.size_bytes or 0) // max(1, right_rows)
        output_width = left_width + right_width

        return Statistics(
            row_count=output_rows,
            size_bytes=output_rows * output_width if output_width > 0 else None,
        )

    def estimate_aggregate(
        self,
        input_stats: Statistics,
        group_by: list[Expression]
    ) -> Statistics:
        """Estimate statistics for aggregation."""
        input_rows = input_stats.row_count or self.constants.default_row_count

        if not group_by:
            # Global aggregation - single row output
            return Statistics(row_count=1, size_bytes=100)

        # Estimate distinct groups
        # Use column statistics if available, otherwise use heuristics
        distinct_estimate = input_rows

        for expr in group_by:
            if isinstance(expr, ColumnRef):
                col_stats = self._get_column_stats(expr)
                if col_stats and col_stats.distinct_count > 0:
                    distinct_estimate = min(distinct_estimate, col_stats.distinct_count)

        # Cap at input rows
        output_rows = min(input_rows, distinct_estimate)
        output_rows = max(1, int(output_rows * 0.8))  # Assume some reduction

        return Statistics(
            row_count=output_rows,
            size_bytes=output_rows * self.constants.default_row_width,
        )

    def estimate_sort(self, input_stats: Statistics) -> Statistics:
        """Estimate statistics for sorting (same cardinality)."""
        return Statistics(
            row_count=input_stats.row_count,
            size_bytes=input_stats.size_bytes,
        )

    def estimate_limit(
        self,
        input_stats: Statistics,
        limit: Optional[int],
        offset: int = 0
    ) -> Statistics:
        """Estimate statistics for limit operation."""
        input_rows = input_stats.row_count or self.constants.default_row_count

        if limit is not None:
            output_rows = min(input_rows - offset, limit)
            output_rows = max(0, output_rows)
        else:
            output_rows = max(0, input_rows - offset)

        width = self.constants.default_row_width
        if input_stats.size_bytes and input_stats.row_count:
            width = input_stats.size_bytes // input_stats.row_count

        return Statistics(
            row_count=output_rows,
            size_bytes=output_rows * width,
        )


# ============================================================================
# Cost Model
# ============================================================================

@dataclass
class OperatorCost:
    """Cost breakdown for an operator."""

    cpu_cost: float = 0.0
    io_cost: float = 0.0
    network_cost: float = 0.0
    memory_cost: float = 0.0

    @property
    def total_cost(self) -> float:
        return self.cpu_cost + self.io_cost + self.network_cost + self.memory_cost

    def __add__(self, other: OperatorCost) -> OperatorCost:
        return OperatorCost(
            cpu_cost=self.cpu_cost + other.cpu_cost,
            io_cost=self.io_cost + other.io_cost,
            network_cost=self.network_cost + other.network_cost,
            memory_cost=self.memory_cost + other.memory_cost,
        )


class CostModel:
    """Cost model for query operators."""

    def __init__(
        self,
        constants: Optional[CostConstants] = None,
        stats_estimator: Optional[StatisticsEstimator] = None
    ):
        self.constants = constants or CostConstants()
        self.stats_estimator = stats_estimator or StatisticsEstimator(self.constants)

    def estimate_plan_cost(self, plan: LogicalPlan) -> tuple[float, Statistics]:
        """Estimate total cost and output statistics for a logical plan."""
        cost, stats = self._estimate_recursive(plan)
        return cost.total_cost, stats

    def _estimate_recursive(self, plan: LogicalPlan) -> tuple[OperatorCost, Statistics]:
        """Recursively estimate cost for a plan tree."""
        if isinstance(plan, LogicalScan):
            return self._cost_scan(plan)
        elif isinstance(plan, LogicalFilter):
            return self._cost_filter(plan)
        elif isinstance(plan, LogicalProject):
            return self._cost_project(plan)
        elif isinstance(plan, LogicalJoin):
            return self._cost_join(plan)
        elif isinstance(plan, LogicalAggregate):
            return self._cost_aggregate(plan)
        elif isinstance(plan, LogicalSort):
            return self._cost_sort(plan)
        elif isinstance(plan, LogicalLimit):
            return self._cost_limit(plan)
        elif isinstance(plan, LogicalUnion):
            return self._cost_union(plan)
        elif isinstance(plan, LogicalDistinct):
            return self._cost_distinct(plan)
        else:
            # Unknown operator - use default
            return OperatorCost(), Statistics(row_count=1000)

    def _cost_scan(self, scan: LogicalScan) -> tuple[OperatorCost, Statistics]:
        """Calculate cost for a table scan."""
        stats = self.stats_estimator.estimate_scan(scan)
        rows = stats.row_count or self.constants.default_row_count

        # Calculate pages
        size_bytes = stats.size_bytes or rows * self.constants.default_row_width
        num_pages = max(1, size_bytes // self.constants.page_size)

        cost = OperatorCost(
            cpu_cost=rows * self.constants.cpu_tuple_cost,
            io_cost=num_pages * self.constants.seq_page_cost,
        )

        return cost, stats

    def _cost_filter(self, filter_node: LogicalFilter) -> tuple[OperatorCost, Statistics]:
        """Calculate cost for a filter."""
        input_cost, input_stats = self._estimate_recursive(filter_node.input)
        output_stats = self.stats_estimator.estimate_filter(filter_node, input_stats)

        rows = input_stats.row_count or self.constants.default_row_count

        filter_cost = OperatorCost(
            cpu_cost=rows * self.constants.cpu_comparison_cost,
        )

        return input_cost + filter_cost, output_stats

    def _cost_project(self, project: LogicalProject) -> tuple[OperatorCost, Statistics]:
        """Calculate cost for a projection."""
        input_cost, input_stats = self._estimate_recursive(project.input)

        rows = input_stats.row_count or self.constants.default_row_count

        # Cost is just evaluating expressions
        project_cost = OperatorCost(
            cpu_cost=rows * len(project.expressions) * self.constants.cpu_tuple_cost,
        )

        # Output has same row count but potentially different width
        output_stats = Statistics(
            row_count=input_stats.row_count,
            size_bytes=input_stats.size_bytes,  # Simplified
        )

        return input_cost + project_cost, output_stats

    def _cost_join(self, join: LogicalJoin) -> tuple[OperatorCost, Statistics]:
        """Calculate cost for a join (using hash join model)."""
        left_cost, left_stats = self._estimate_recursive(join.left)
        right_cost, right_stats = self._estimate_recursive(join.right)

        output_stats = self.stats_estimator.estimate_join(
            left_stats, right_stats,
            join.join_type.name,
            join.condition
        )

        left_rows = left_stats.row_count or self.constants.default_row_count
        right_rows = right_stats.row_count or self.constants.default_row_count

        # Hash join cost model
        # Build phase: hash the smaller side
        # Probe phase: probe with the larger side
        build_rows = min(left_rows, right_rows)
        probe_rows = max(left_rows, right_rows)

        join_cost = OperatorCost(
            cpu_cost=(
                build_rows * self.constants.cpu_hash_cost * self.constants.hash_build_factor +
                probe_rows * self.constants.cpu_hash_cost * self.constants.hash_probe_factor
            ),
            memory_cost=build_rows * self.constants.memory_tuple_cost,
        )

        return left_cost + right_cost + join_cost, output_stats

    def _cost_aggregate(self, agg: LogicalAggregate) -> tuple[OperatorCost, Statistics]:
        """Calculate cost for aggregation."""
        input_cost, input_stats = self._estimate_recursive(agg.input)
        output_stats = self.stats_estimator.estimate_aggregate(input_stats, agg.group_by)

        input_rows = input_stats.row_count or self.constants.default_row_count
        output_rows = output_stats.row_count or 1

        # Hash aggregate cost
        agg_cost = OperatorCost(
            cpu_cost=(
                input_rows * self.constants.cpu_hash_cost +  # Hashing
                input_rows * len(agg.aggregates) * self.constants.cpu_tuple_cost  # Aggregate computation
            ),
            memory_cost=output_rows * self.constants.memory_tuple_cost,
        )

        return input_cost + agg_cost, output_stats

    def _cost_sort(self, sort: LogicalSort) -> tuple[OperatorCost, Statistics]:
        """Calculate cost for sorting."""
        input_cost, input_stats = self._estimate_recursive(sort.input)
        output_stats = self.stats_estimator.estimate_sort(input_stats)

        rows = input_stats.row_count or self.constants.default_row_count

        # O(n log n) sort cost
        if rows > 1:
            log_rows = math.log2(rows)
        else:
            log_rows = 1

        sort_cost = OperatorCost(
            cpu_cost=rows * log_rows * self.constants.cpu_sort_comparison * self.constants.sort_factor,
            memory_cost=rows * self.constants.memory_tuple_cost,
        )

        return input_cost + sort_cost, output_stats

    def _cost_limit(self, limit: LogicalLimit) -> tuple[OperatorCost, Statistics]:
        """Calculate cost for limit."""
        input_cost, input_stats = self._estimate_recursive(limit.input)
        output_stats = self.stats_estimator.estimate_limit(
            input_stats, limit.limit, limit.offset
        )

        # Minimal cost - just counting rows
        limit_cost = OperatorCost(
            cpu_cost=(limit.limit or 0) * self.constants.cpu_tuple_cost * 0.1,
        )

        return input_cost + limit_cost, output_stats

    def _cost_union(self, union: LogicalUnion) -> tuple[OperatorCost, Statistics]:
        """Calculate cost for union."""
        left_cost, left_stats = self._estimate_recursive(union.left)
        right_cost, right_stats = self._estimate_recursive(union.right)

        left_rows = left_stats.row_count or self.constants.default_row_count
        right_rows = right_stats.row_count or self.constants.default_row_count

        output_rows = left_rows + right_rows
        if not union.all:
            output_rows = int(output_rows * 0.8)  # Dedup estimate

        output_stats = Statistics(row_count=output_rows)

        union_cost = OperatorCost()
        if not union.all:
            # Dedup cost
            union_cost = OperatorCost(
                cpu_cost=output_rows * self.constants.cpu_hash_cost,
                memory_cost=output_rows * self.constants.memory_tuple_cost,
            )

        return left_cost + right_cost + union_cost, output_stats

    def _cost_distinct(self, distinct: LogicalDistinct) -> tuple[OperatorCost, Statistics]:
        """Calculate cost for distinct."""
        input_cost, input_stats = self._estimate_recursive(distinct.input)

        input_rows = input_stats.row_count or self.constants.default_row_count
        output_rows = int(input_rows * 0.8)  # Estimate

        output_stats = Statistics(row_count=output_rows)

        # Hash-based distinct
        distinct_cost = OperatorCost(
            cpu_cost=input_rows * self.constants.cpu_hash_cost,
            memory_cost=output_rows * self.constants.memory_tuple_cost,
        )

        return input_cost + distinct_cost, output_stats


# ============================================================================
# Join Enumeration (Dynamic Programming)
# ============================================================================

@dataclass
class JoinNode:
    """Represents a node in the join graph."""

    plan: LogicalPlan
    tables: FrozenSet[str]
    cost: float
    stats: Statistics


@dataclass
class JoinEdge:
    """Represents a join condition between tables."""

    left_table: str
    right_table: str
    condition: Expression
    selectivity: float = 0.1


class DPJoinOptimizer:
    """Dynamic programming based join optimizer.

    Implements the classic DP algorithm for join enumeration:
    1. Build join graph from query
    2. Enumerate all join orderings using DP
    3. Select minimum cost ordering
    """

    def __init__(self, cost_model: Optional[CostModel] = None):
        self.cost_model = cost_model or CostModel()

    def optimize_join_order(
        self,
        relations: list[LogicalPlan],
        join_conditions: list[Expression]
    ) -> LogicalPlan:
        """Find optimal join ordering using dynamic programming."""
        if len(relations) <= 1:
            return relations[0] if relations else LogicalScan()

        if len(relations) == 2:
            return self._create_join(relations[0], relations[1], join_conditions)

        # Build initial join nodes
        table_to_plan: dict[str, JoinNode] = {}
        for plan in relations:
            tables = self._get_tables(plan)
            cost, stats = self.cost_model.estimate_plan_cost(plan)
            table_to_plan[frozenset(tables)] = JoinNode(
                plan=plan,
                tables=frozenset(tables),
                cost=cost,
                stats=stats,
            )

        # Parse join conditions into edges
        edges = self._build_join_graph(join_conditions)

        # DP memoization table
        # Key: frozenset of table names
        # Value: best JoinNode for that subset
        dp: dict[FrozenSet[str], JoinNode] = {}

        # Base case: single relations
        for tables, node in table_to_plan.items():
            dp[tables] = node

        # Enumerate subsets of increasing size
        all_tables = frozenset().union(*table_to_plan.keys())

        for size in range(2, len(table_to_plan) + 1):
            for subset in self._subsets_of_size(all_tables, size):
                best_node = None
                best_cost = float('inf')

                # Try all ways to partition this subset into two non-empty parts
                for left_subset in self._subsets_of_size(subset, size - 1):
                    if left_subset not in dp:
                        continue

                    right_subset = subset - left_subset
                    if not right_subset or right_subset not in dp:
                        continue

                    # Check if there's a join edge between left and right
                    condition = self._find_join_condition(
                        left_subset, right_subset, edges
                    )

                    left_node = dp[left_subset]
                    right_node = dp[right_subset]

                    # Estimate cost of joining
                    join_stats = self.cost_model.stats_estimator.estimate_join(
                        left_node.stats, right_node.stats,
                        "INNER", condition
                    )

                    # Cost = cost of children + cost of this join
                    join_cost = self._estimate_join_cost(
                        left_node.stats, right_node.stats
                    )
                    total_cost = left_node.cost + right_node.cost + join_cost

                    if total_cost < best_cost:
                        best_cost = total_cost
                        join_plan = LogicalJoin(
                            left=left_node.plan,
                            right=right_node.plan,
                            condition=condition,
                        )
                        best_node = JoinNode(
                            plan=join_plan,
                            tables=subset,
                            cost=total_cost,
                            stats=join_stats,
                        )

                if best_node:
                    dp[subset] = best_node

        # Return the optimal join tree
        result = dp.get(all_tables)
        if result:
            return result.plan

        # Fallback: left-deep join tree
        return self._create_left_deep_join(relations, join_conditions)

    def _get_tables(self, plan: LogicalPlan) -> set[str]:
        """Get all table names in a plan."""
        tables: set[str] = set()
        if isinstance(plan, LogicalScan):
            tables.add(plan.effective_name)
        for child in plan.children():
            tables.update(self._get_tables(child))
        return tables

    def _build_join_graph(self, conditions: list[Expression]) -> list[JoinEdge]:
        """Build join graph from conditions."""
        edges: list[JoinEdge] = []

        for condition in conditions:
            left_tables, right_tables = self._extract_join_tables(condition)
            if left_tables and right_tables:
                for lt in left_tables:
                    for rt in right_tables:
                        if lt != rt:
                            edges.append(JoinEdge(
                                left_table=lt,
                                right_table=rt,
                                condition=condition,
                            ))
        return edges

    def _extract_join_tables(self, expr: Expression) -> tuple[set[str], set[str]]:
        """Extract tables from a join condition (assumes equi-join)."""
        if isinstance(expr, BinaryOp):
            if expr.operator == "=":
                left_tables = self._get_expr_tables(expr.left)
                right_tables = self._get_expr_tables(expr.right)
                return left_tables, right_tables
            elif expr.operator == "AND":
                lt1, rt1 = self._extract_join_tables(expr.left)
                lt2, rt2 = self._extract_join_tables(expr.right)
                return lt1 | lt2, rt1 | rt2
        return set(), set()

    def _get_expr_tables(self, expr: Expression) -> set[str]:
        """Get tables referenced in an expression."""
        tables: set[str] = set()
        if isinstance(expr, ColumnRef) and expr.table:
            tables.add(expr.table)
        for child in expr.children():
            tables.update(self._get_expr_tables(child))
        return tables

    def _find_join_condition(
        self,
        left_tables: FrozenSet[str],
        right_tables: FrozenSet[str],
        edges: list[JoinEdge]
    ) -> Optional[Expression]:
        """Find join condition between two table sets."""
        matching_conditions: list[Expression] = []

        for edge in edges:
            if (edge.left_table in left_tables and edge.right_table in right_tables) or \
               (edge.left_table in right_tables and edge.right_table in left_tables):
                matching_conditions.append(edge.condition)

        if not matching_conditions:
            return None

        if len(matching_conditions) == 1:
            return matching_conditions[0]

        # Combine with AND
        result = matching_conditions[0]
        for cond in matching_conditions[1:]:
            result = BinaryOp(operator="AND", left=result, right=cond)
        return result

    def _estimate_join_cost(self, left_stats: Statistics, right_stats: Statistics) -> float:
        """Estimate the cost of a join operation."""
        left_rows = left_stats.row_count or 1000
        right_rows = right_stats.row_count or 1000

        constants = self.cost_model.constants

        # Hash join cost
        build_rows = min(left_rows, right_rows)
        probe_rows = max(left_rows, right_rows)

        cost = (
            build_rows * constants.cpu_hash_cost * constants.hash_build_factor +
            probe_rows * constants.cpu_hash_cost * constants.hash_probe_factor
        )

        return cost

    def _subsets_of_size(self, s: FrozenSet[str], size: int) -> list[FrozenSet[str]]:
        """Generate all subsets of given size."""
        if size > len(s):
            return []
        if size == 0:
            return [frozenset()]
        if size == len(s):
            return [s]

        elements = list(s)
        result: list[FrozenSet[str]] = []

        def generate(start: int, current: list[str]):
            if len(current) == size:
                result.append(frozenset(current))
                return
            for i in range(start, len(elements)):
                current.append(elements[i])
                generate(i + 1, current)
                current.pop()

        generate(0, [])
        return result

    def _create_join(
        self,
        left: LogicalPlan,
        right: LogicalPlan,
        conditions: list[Expression]
    ) -> LogicalPlan:
        """Create a join between two plans."""
        left_tables = self._get_tables(left)
        right_tables = self._get_tables(right)

        # Find applicable condition
        edges = self._build_join_graph(conditions)
        condition = self._find_join_condition(
            frozenset(left_tables), frozenset(right_tables), edges
        )

        return LogicalJoin(left=left, right=right, condition=condition)

    def _create_left_deep_join(
        self,
        relations: list[LogicalPlan],
        conditions: list[Expression]
    ) -> LogicalPlan:
        """Create a left-deep join tree as fallback."""
        if not relations:
            return LogicalScan()

        result = relations[0]
        for rel in relations[1:]:
            result = self._create_join(result, rel, conditions)

        return result


# ============================================================================
# Cost-Based Optimizer Rule
# ============================================================================

class CostBasedJoinReordering(OptimizationRule):
    """Cost-based join reordering rule using DP."""

    def __init__(self, cost_model: Optional[CostModel] = None):
        self.cost_model = cost_model or CostModel()
        self.dp_optimizer = DPJoinOptimizer(self.cost_model)

    @property
    def name(self) -> str:
        return "CostBasedJoinReordering"

    def apply(self, plan: LogicalPlan) -> tuple[LogicalPlan, bool]:
        result = self._reorder(plan)
        changed = result is not plan
        return result, changed

    def _reorder(self, plan: LogicalPlan) -> LogicalPlan:
        """Apply cost-based join reordering."""
        if isinstance(plan, LogicalJoin):
            # Flatten joins and collect conditions
            relations, conditions = self._flatten_joins(plan)

            if len(relations) > 2:
                # Use DP optimizer for multi-way joins
                optimized = self.dp_optimizer.optimize_join_order(relations, conditions)
                return optimized

        # Recursively process children
        new_children = [self._reorder(child) for child in plan.children()]
        return plan._with_children(new_children)

    def _flatten_joins(
        self, plan: LogicalPlan
    ) -> tuple[list[LogicalPlan], list[Expression]]:
        """Flatten nested inner joins."""
        relations: list[LogicalPlan] = []
        conditions: list[Expression] = []

        if isinstance(plan, LogicalJoin) and plan.join_type.name == "INNER":
            left_rels, left_conds = self._flatten_joins(plan.left)
            right_rels, right_conds = self._flatten_joins(plan.right)
            relations = left_rels + right_rels
            conditions = left_conds + right_conds
            if plan.condition:
                conditions.append(plan.condition)
        else:
            relations = [plan]

        return relations, conditions


# ============================================================================
# Cost-Based Physical Planning
# ============================================================================

class CostBasedPhysicalPlanner:
    """Physical planner that uses cost estimates to select operators."""

    def __init__(
        self,
        cost_model: Optional[CostModel] = None,
        partition_count: int = 200,
        broadcast_threshold_rows: int = 100000,
        broadcast_threshold_bytes: int = 10 * 1024 * 1024  # 10MB
    ):
        self.cost_model = cost_model or CostModel()
        self.partition_count = partition_count
        self.broadcast_threshold_rows = broadcast_threshold_rows
        self.broadcast_threshold_bytes = broadcast_threshold_bytes

    def select_join_strategy(
        self,
        left_stats: Statistics,
        right_stats: Statistics
    ) -> tuple[str, str, float]:
        """Select optimal join strategy based on cost.

        Returns: (strategy, build_side, cost)
        """
        left_rows = left_stats.row_count or 1000
        right_rows = right_stats.row_count or 1000
        left_bytes = left_stats.size_bytes or left_rows * 100
        right_bytes = right_stats.size_bytes or right_rows * 100

        constants = self.cost_model.constants

        # Calculate costs for each strategy
        costs = {}

        # Broadcast join (broadcast smaller side)
        smaller_rows = min(left_rows, right_rows)
        larger_rows = max(left_rows, right_rows)
        smaller_bytes = min(left_bytes, right_bytes)

        if smaller_rows <= self.broadcast_threshold_rows or smaller_bytes <= self.broadcast_threshold_bytes:
            broadcast_cost = (
                smaller_bytes * constants.network_transfer_cost / 1024 +  # Transfer
                constants.network_latency_cost * self.partition_count +   # Latency
                smaller_rows * constants.cpu_hash_cost * constants.hash_build_factor +
                larger_rows * constants.cpu_hash_cost * constants.hash_probe_factor
            )
            costs['broadcast'] = broadcast_cost

        # Hash join (shuffle both sides)
        hash_cost = (
            (left_bytes + right_bytes) * constants.network_transfer_cost / 1024 +  # Shuffle
            constants.network_latency_cost * 2 * self.partition_count +  # Latency
            min(left_rows, right_rows) * constants.cpu_hash_cost * constants.hash_build_factor +
            max(left_rows, right_rows) * constants.cpu_hash_cost * constants.hash_probe_factor
        )
        costs['hash'] = hash_cost

        # Merge join (sort both sides if not sorted)
        if left_rows > 1 and right_rows > 1:
            sort_cost = (
                left_rows * math.log2(left_rows) * constants.cpu_sort_comparison +
                right_rows * math.log2(right_rows) * constants.cpu_sort_comparison
            )
        else:
            sort_cost = 0

        merge_cost = (
            (left_bytes + right_bytes) * constants.network_transfer_cost / 1024 +
            sort_cost +
            (left_rows + right_rows) * constants.cpu_comparison_cost * constants.merge_factor
        )
        costs['merge'] = merge_cost

        # Select minimum cost strategy
        best_strategy = min(costs, key=costs.get)
        best_cost = costs[best_strategy]

        # Determine build/broadcast side
        if left_rows <= right_rows:
            build_side = 'left'
        else:
            build_side = 'right'

        return best_strategy, build_side, best_cost

    def select_aggregate_strategy(
        self,
        input_stats: Statistics,
        group_by: list[Expression],
        is_distributed: bool = True
    ) -> tuple[str, float]:
        """Select optimal aggregation strategy.

        Returns: (strategy, cost)
        """
        input_rows = input_stats.row_count or 1000
        constants = self.cost_model.constants

        if not is_distributed:
            # Single-phase local aggregation
            cost = input_rows * constants.cpu_hash_cost
            return 'single', cost

        if not group_by:
            # Global aggregation - needs gather
            cost = (
                input_rows * constants.cpu_hash_cost +  # Local aggregation
                constants.network_latency_cost * self.partition_count +  # Gather latency
                self.partition_count * constants.cpu_hash_cost  # Final merge
            )
            return 'two_phase_global', cost

        # Grouped aggregation options

        # Two-phase: partial + final
        # Estimate number of groups
        num_groups = min(input_rows, input_rows // 10)  # Heuristic

        two_phase_cost = (
            input_rows * constants.cpu_hash_cost +  # Partial
            num_groups * self.partition_count * constants.network_transfer_cost / 1024 +  # Shuffle
            constants.network_latency_cost * self.partition_count +
            num_groups * constants.cpu_hash_cost  # Final
        )

        # If data is pre-partitioned on group keys, single phase is cheaper
        single_phase_cost = input_rows * constants.cpu_hash_cost

        if single_phase_cost < two_phase_cost:
            return 'single', single_phase_cost
        else:
            return 'two_phase', two_phase_cost


# ============================================================================
# Integrated Cost-Based Optimizer
# ============================================================================

class CostBasedOptimizer:
    """Full cost-based query optimizer.

    Combines rule-based transformations with cost-based join ordering
    and operator selection.
    """

    def __init__(
        self,
        cost_model: Optional[CostModel] = None,
        stats_estimator: Optional[StatisticsEstimator] = None
    ):
        self.cost_model = cost_model or CostModel(stats_estimator=stats_estimator)
        self.stats_estimator = stats_estimator or self.cost_model.stats_estimator

        # Import rule-based optimizations
        from .optimizer import (
            ConstantFolding, PredicatePushdown, ProjectionPruning,
            CommonSubexpressionElimination
        )

        # Rule-based phase (heuristic)
        self.rule_optimizer_rules = [
            ConstantFolding(),
            PredicatePushdown(),
            ProjectionPruning(),
            CommonSubexpressionElimination(),
        ]

        # Cost-based phase
        self.cost_based_rules = [
            CostBasedJoinReordering(self.cost_model),
        ]

        self.max_iterations = 5

    def register_table_stats(self, table_name: str, stats: TableStatistics) -> None:
        """Register statistics for a table."""
        self.stats_estimator.register_table(table_name, stats)

    def optimize(self, plan: LogicalPlan) -> LogicalPlan:
        """Optimize a logical plan using cost-based optimization."""
        current = plan

        # Phase 1: Apply rule-based optimizations
        for _ in range(self.max_iterations):
            changed = False
            for rule in self.rule_optimizer_rules:
                new_plan, rule_changed = rule.apply(current)
                if rule_changed:
                    changed = True
                    current = new_plan
            if not changed:
                break

        # Phase 2: Apply cost-based optimizations
        for rule in self.cost_based_rules:
            current, _ = rule.apply(current)

        # Phase 3: Propagate statistics through the optimized plan
        current = self._propagate_statistics(current)

        return current

    def _propagate_statistics(self, plan: LogicalPlan) -> LogicalPlan:
        """Propagate statistics through the plan tree."""
        _, stats = self.cost_model.estimate_plan_cost(plan)
        plan.stats = stats

        # Recursively propagate to children
        for child in plan.children():
            self._propagate_statistics(child)

        return plan

    def explain(self, plan: LogicalPlan, verbose: bool = False) -> str:
        """Generate explain output with cost estimates."""
        lines = ["=== Cost-Based Optimizer Explain ==="]
        lines.append("")

        # Original plan
        lines.append("Original Plan:")
        lines.append(plan.to_string())
        lines.append("")

        # Optimized plan
        optimized = self.optimize(plan)
        lines.append("Optimized Plan:")
        lines.append(optimized.to_string())
        lines.append("")

        # Cost estimates
        original_cost, original_stats = self.cost_model.estimate_plan_cost(plan)
        optimized_cost, optimized_stats = self.cost_model.estimate_plan_cost(optimized)

        lines.append("Cost Estimates:")
        lines.append(f"  Original:  {original_cost:.2f}")
        lines.append(f"  Optimized: {optimized_cost:.2f}")
        lines.append(f"  Savings:   {(1 - optimized_cost/original_cost)*100:.1f}%")
        lines.append("")

        if verbose:
            lines.append("Statistics:")
            lines.append(f"  Original rows:  {original_stats.row_count or 'unknown'}")
            lines.append(f"  Optimized rows: {optimized_stats.row_count or 'unknown'}")

        return "\n".join(lines)
