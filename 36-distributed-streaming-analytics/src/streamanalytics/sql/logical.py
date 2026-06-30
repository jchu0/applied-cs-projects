"""Logical Plan - Representation of query plans before physical optimization.

Provides logical plan nodes and a builder that converts parsed SQL
statements into a logical plan tree.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional, Callable

from .ast import (
    Expression, ColumnRef, Literal, BinaryOp, AggregateExpr,
    SelectStatement, JoinType, DataType,
)


class LogicalPlanType(Enum):
    """Types of logical plan nodes."""

    SCAN = auto()
    FILTER = auto()
    PROJECT = auto()
    JOIN = auto()
    AGGREGATE = auto()
    SORT = auto()
    LIMIT = auto()
    UNION = auto()
    DISTINCT = auto()
    SUBQUERY = auto()


@dataclass
class Schema:
    """Schema representing column names and types."""

    columns: list[tuple[str, DataType]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.columns)

    def column_names(self) -> list[str]:
        return [name for name, _ in self.columns]

    def column_types(self) -> list[DataType]:
        return [dtype for _, dtype in self.columns]

    def find_column(self, name: str, table: Optional[str] = None) -> Optional[int]:
        """Find column index by name, optionally qualified by table."""
        for i, (col_name, _) in enumerate(self.columns):
            if table:
                if col_name == f"{table}.{name}" or col_name == name:
                    return i
            elif col_name == name or col_name.endswith(f".{name}"):
                return i
        return None

    def merge(self, other: Schema) -> Schema:
        """Merge two schemas."""
        return Schema(columns=self.columns + other.columns)


@dataclass
class Statistics:
    """Plan node statistics for optimization."""

    row_count: Optional[int] = None
    size_bytes: Optional[int] = None
    distinct_values: dict[str, int] = field(default_factory=dict)
    min_values: dict[str, Any] = field(default_factory=dict)
    max_values: dict[str, Any] = field(default_factory=dict)
    null_counts: dict[str, int] = field(default_factory=dict)


# ============================================================================
# Logical Plan Nodes
# ============================================================================

@dataclass
class LogicalPlan(ABC):
    """Base class for logical plan nodes."""

    schema: Schema = field(default_factory=Schema)
    stats: Optional[Statistics] = None

    @abstractmethod
    def children(self) -> list[LogicalPlan]:
        """Get child plan nodes."""
        pass

    @abstractmethod
    def node_type(self) -> LogicalPlanType:
        """Get the plan node type."""
        pass

    @abstractmethod
    def to_string(self, indent: int = 0) -> str:
        """Convert to readable string representation."""
        pass

    def transform(self, func: Callable[[LogicalPlan], LogicalPlan]) -> LogicalPlan:
        """Transform plan tree with a function."""
        new_children = [child.transform(func) for child in self.children()]
        return func(self._with_children(new_children))

    @abstractmethod
    def _with_children(self, children: list[LogicalPlan]) -> LogicalPlan:
        """Create copy with new children."""
        pass

    def collect_expressions(self) -> list[Expression]:
        """Collect all expressions in this node."""
        return []


@dataclass
class LogicalScan(LogicalPlan):
    """Scan a table."""

    table_name: str = ""
    table_schema: Optional[str] = None
    alias: Optional[str] = None
    projection: Optional[list[int]] = None  # Column indices to project

    def children(self) -> list[LogicalPlan]:
        return []

    def node_type(self) -> LogicalPlanType:
        return LogicalPlanType.SCAN

    def to_string(self, indent: int = 0) -> str:
        prefix = "  " * indent
        name = f"{self.table_schema}.{self.table_name}" if self.table_schema else self.table_name
        alias_str = f" AS {self.alias}" if self.alias else ""
        proj_str = f" [cols: {self.projection}]" if self.projection else ""
        return f"{prefix}Scan: {name}{alias_str}{proj_str}"

    def _with_children(self, children: list[LogicalPlan]) -> LogicalPlan:
        return LogicalScan(
            table_name=self.table_name,
            table_schema=self.table_schema,
            alias=self.alias,
            projection=self.projection,
            schema=self.schema,
            stats=self.stats,
        )

    @property
    def effective_name(self) -> str:
        return self.alias or self.table_name


@dataclass
class LogicalFilter(LogicalPlan):
    """Filter rows based on a predicate."""

    predicate: Expression = field(default_factory=lambda: Literal(value=True))
    input: LogicalPlan = field(default_factory=lambda: LogicalScan())

    def children(self) -> list[LogicalPlan]:
        return [self.input]

    def node_type(self) -> LogicalPlanType:
        return LogicalPlanType.FILTER

    def to_string(self, indent: int = 0) -> str:
        prefix = "  " * indent
        lines = [f"{prefix}Filter: {self.predicate.to_sql()}"]
        lines.append(self.input.to_string(indent + 1))
        return "\n".join(lines)

    def _with_children(self, children: list[LogicalPlan]) -> LogicalPlan:
        return LogicalFilter(
            predicate=self.predicate,
            input=children[0],
            schema=self.schema,
            stats=self.stats,
        )

    def collect_expressions(self) -> list[Expression]:
        return [self.predicate]


@dataclass
class LogicalProject(LogicalPlan):
    """Project (select) specific expressions."""

    expressions: list[Expression] = field(default_factory=list)
    input: LogicalPlan = field(default_factory=lambda: LogicalScan())
    aliases: list[Optional[str]] = field(default_factory=list)

    def children(self) -> list[LogicalPlan]:
        return [self.input]

    def node_type(self) -> LogicalPlanType:
        return LogicalPlanType.PROJECT

    def to_string(self, indent: int = 0) -> str:
        prefix = "  " * indent
        exprs = []
        for expr, alias in zip(self.expressions, self.aliases):
            if alias:
                exprs.append(f"{expr.to_sql()} AS {alias}")
            else:
                exprs.append(expr.to_sql())
        lines = [f"{prefix}Project: {', '.join(exprs)}"]
        lines.append(self.input.to_string(indent + 1))
        return "\n".join(lines)

    def _with_children(self, children: list[LogicalPlan]) -> LogicalPlan:
        return LogicalProject(
            expressions=self.expressions,
            input=children[0],
            aliases=self.aliases,
            schema=self.schema,
            stats=self.stats,
        )

    def collect_expressions(self) -> list[Expression]:
        return list(self.expressions)


@dataclass
class LogicalJoin(LogicalPlan):
    """Join two inputs."""

    join_type: JoinType = JoinType.INNER
    left: LogicalPlan = field(default_factory=lambda: LogicalScan())
    right: LogicalPlan = field(default_factory=lambda: LogicalScan())
    condition: Optional[Expression] = None
    using_columns: list[str] = field(default_factory=list)

    def children(self) -> list[LogicalPlan]:
        return [self.left, self.right]

    def node_type(self) -> LogicalPlanType:
        return LogicalPlanType.JOIN

    def to_string(self, indent: int = 0) -> str:
        prefix = "  " * indent
        cond_str = f" ON {self.condition.to_sql()}" if self.condition else ""
        using_str = f" USING ({', '.join(self.using_columns)})" if self.using_columns else ""
        lines = [f"{prefix}{self.join_type.name} Join{cond_str}{using_str}"]
        lines.append(self.left.to_string(indent + 1))
        lines.append(self.right.to_string(indent + 1))
        return "\n".join(lines)

    def _with_children(self, children: list[LogicalPlan]) -> LogicalPlan:
        return LogicalJoin(
            join_type=self.join_type,
            left=children[0],
            right=children[1],
            condition=self.condition,
            using_columns=self.using_columns,
            schema=self.schema,
            stats=self.stats,
        )

    def collect_expressions(self) -> list[Expression]:
        return [self.condition] if self.condition else []


@dataclass
class LogicalAggregate(LogicalPlan):
    """Aggregate with group by."""

    group_by: list[Expression] = field(default_factory=list)
    aggregates: list[AggregateExpr] = field(default_factory=list)
    input: LogicalPlan = field(default_factory=lambda: LogicalScan())

    def children(self) -> list[LogicalPlan]:
        return [self.input]

    def node_type(self) -> LogicalPlanType:
        return LogicalPlanType.AGGREGATE

    def to_string(self, indent: int = 0) -> str:
        prefix = "  " * indent
        group_str = ", ".join(e.to_sql() for e in self.group_by) if self.group_by else "[]"
        agg_str = ", ".join(a.to_sql() for a in self.aggregates)
        lines = [f"{prefix}Aggregate: group=[{group_str}], agg=[{agg_str}]"]
        lines.append(self.input.to_string(indent + 1))
        return "\n".join(lines)

    def _with_children(self, children: list[LogicalPlan]) -> LogicalPlan:
        return LogicalAggregate(
            group_by=self.group_by,
            aggregates=self.aggregates,
            input=children[0],
            schema=self.schema,
            stats=self.stats,
        )

    def collect_expressions(self) -> list[Expression]:
        return list(self.group_by) + list(self.aggregates)


@dataclass
class LogicalSort(LogicalPlan):
    """Sort by expressions."""

    sort_expressions: list[Expression] = field(default_factory=list)
    ascending: list[bool] = field(default_factory=list)
    nulls_first: list[bool] = field(default_factory=list)
    input: LogicalPlan = field(default_factory=lambda: LogicalScan())

    def children(self) -> list[LogicalPlan]:
        return [self.input]

    def node_type(self) -> LogicalPlanType:
        return LogicalPlanType.SORT

    def to_string(self, indent: int = 0) -> str:
        prefix = "  " * indent
        sorts = []
        for expr, asc in zip(self.sort_expressions, self.ascending):
            order = "ASC" if asc else "DESC"
            sorts.append(f"{expr.to_sql()} {order}")
        lines = [f"{prefix}Sort: [{', '.join(sorts)}]"]
        lines.append(self.input.to_string(indent + 1))
        return "\n".join(lines)

    def _with_children(self, children: list[LogicalPlan]) -> LogicalPlan:
        return LogicalSort(
            sort_expressions=self.sort_expressions,
            ascending=self.ascending,
            nulls_first=self.nulls_first,
            input=children[0],
            schema=self.schema,
            stats=self.stats,
        )

    def collect_expressions(self) -> list[Expression]:
        return list(self.sort_expressions)


@dataclass
class LogicalLimit(LogicalPlan):
    """Limit rows returned."""

    limit: Optional[int] = None
    offset: int = 0
    input: LogicalPlan = field(default_factory=lambda: LogicalScan())

    def children(self) -> list[LogicalPlan]:
        return [self.input]

    def node_type(self) -> LogicalPlanType:
        return LogicalPlanType.LIMIT

    def to_string(self, indent: int = 0) -> str:
        prefix = "  " * indent
        limit_str = f"limit={self.limit}" if self.limit else ""
        offset_str = f", offset={self.offset}" if self.offset else ""
        lines = [f"{prefix}Limit: {limit_str}{offset_str}"]
        lines.append(self.input.to_string(indent + 1))
        return "\n".join(lines)

    def _with_children(self, children: list[LogicalPlan]) -> LogicalPlan:
        return LogicalLimit(
            limit=self.limit,
            offset=self.offset,
            input=children[0],
            schema=self.schema,
            stats=self.stats,
        )


@dataclass
class LogicalUnion(LogicalPlan):
    """Union of two queries."""

    left: LogicalPlan = field(default_factory=lambda: LogicalScan())
    right: LogicalPlan = field(default_factory=lambda: LogicalScan())
    all: bool = False

    def children(self) -> list[LogicalPlan]:
        return [self.left, self.right]

    def node_type(self) -> LogicalPlanType:
        return LogicalPlanType.UNION

    def to_string(self, indent: int = 0) -> str:
        prefix = "  " * indent
        all_str = " ALL" if self.all else ""
        lines = [f"{prefix}Union{all_str}"]
        lines.append(self.left.to_string(indent + 1))
        lines.append(self.right.to_string(indent + 1))
        return "\n".join(lines)

    def _with_children(self, children: list[LogicalPlan]) -> LogicalPlan:
        return LogicalUnion(
            left=children[0],
            right=children[1],
            all=self.all,
            schema=self.schema,
            stats=self.stats,
        )


@dataclass
class LogicalDistinct(LogicalPlan):
    """Remove duplicates."""

    input: LogicalPlan = field(default_factory=lambda: LogicalScan())

    def children(self) -> list[LogicalPlan]:
        return [self.input]

    def node_type(self) -> LogicalPlanType:
        return LogicalPlanType.DISTINCT

    def to_string(self, indent: int = 0) -> str:
        prefix = "  " * indent
        lines = [f"{prefix}Distinct"]
        lines.append(self.input.to_string(indent + 1))
        return "\n".join(lines)

    def _with_children(self, children: list[LogicalPlan]) -> LogicalPlan:
        return LogicalDistinct(
            input=children[0],
            schema=self.schema,
            stats=self.stats,
        )


# ============================================================================
# Logical Plan Builder
# ============================================================================

class LogicalPlanBuilder:
    """Builds logical plans from parsed SQL statements."""

    def __init__(self, catalog: Optional[dict[str, Schema]] = None):
        """Initialize with optional table catalog."""
        self.catalog = catalog or {}

    def build(self, stmt: SelectStatement) -> LogicalPlan:
        """Build a logical plan from a SELECT statement."""
        plan = self._build_from_clause(stmt)
        plan = self._build_where_clause(stmt, plan)
        plan = self._build_aggregation(stmt, plan)
        plan = self._build_having_clause(stmt, plan)
        plan = self._build_projection(stmt, plan)
        plan = self._build_distinct(stmt, plan)
        plan = self._build_order_by(stmt, plan)
        plan = self._build_limit(stmt, plan)
        plan = self._build_set_operation(stmt, plan)
        return plan

    def _build_from_clause(self, stmt: SelectStatement) -> LogicalPlan:
        """Build plan for FROM clause."""
        if not stmt.from_clause:
            # No FROM clause - single row with expressions
            return LogicalScan(table_name="__dual__")

        from_clause = stmt.from_clause
        plan = self._build_table_scan(from_clause.table)

        # Build joins
        for join in from_clause.joins:
            right_plan = self._build_table_scan(join.table)
            plan = LogicalJoin(
                join_type=join.join_type,
                left=plan,
                right=right_plan,
                condition=join.condition,
                using_columns=join.using_columns,
            )

        return plan

    def _build_table_scan(self, table_ref) -> LogicalPlan:
        """Build a table scan node."""
        schema = self.catalog.get(table_ref.name, Schema())
        return LogicalScan(
            table_name=table_ref.name,
            table_schema=table_ref.schema,
            alias=table_ref.alias,
            schema=schema,
        )

    def _build_where_clause(self, stmt: SelectStatement, plan: LogicalPlan) -> LogicalPlan:
        """Build plan for WHERE clause."""
        if not stmt.where_clause:
            return plan
        return LogicalFilter(predicate=stmt.where_clause.condition, input=plan)

    def _build_aggregation(self, stmt: SelectStatement, plan: LogicalPlan) -> LogicalPlan:
        """Build plan for GROUP BY and aggregations."""
        # Check if we have aggregates in select list
        aggregates = self._collect_aggregates(stmt.select_list)
        has_group_by = stmt.group_by is not None

        if not aggregates and not has_group_by:
            return plan

        group_by = []
        if stmt.group_by:
            group_by = stmt.group_by.expressions

        return LogicalAggregate(
            group_by=group_by,
            aggregates=aggregates,
            input=plan,
        )

    def _collect_aggregates(self, expressions: list[Expression]) -> list[AggregateExpr]:
        """Collect aggregate expressions from expression list."""
        aggregates = []
        for expr in expressions:
            self._find_aggregates(expr, aggregates)
        return aggregates

    def _find_aggregates(self, expr: Expression, result: list[AggregateExpr]) -> None:
        """Recursively find aggregate expressions."""
        if isinstance(expr, AggregateExpr):
            result.append(expr)
        for child in expr.children():
            self._find_aggregates(child, result)

    def _build_having_clause(self, stmt: SelectStatement, plan: LogicalPlan) -> LogicalPlan:
        """Build plan for HAVING clause."""
        if not stmt.having:
            return plan
        return LogicalFilter(predicate=stmt.having.condition, input=plan)

    def _build_projection(self, stmt: SelectStatement, plan: LogicalPlan) -> LogicalPlan:
        """Build plan for SELECT list."""
        if not stmt.select_list:
            return plan  # SELECT * - no explicit projection needed

        expressions = stmt.select_list
        aliases = [expr.alias for expr in expressions]

        return LogicalProject(
            expressions=expressions,
            input=plan,
            aliases=aliases,
        )

    def _build_distinct(self, stmt: SelectStatement, plan: LogicalPlan) -> LogicalPlan:
        """Build plan for SELECT DISTINCT."""
        if not stmt.distinct:
            return plan
        return LogicalDistinct(input=plan)

    def _build_order_by(self, stmt: SelectStatement, plan: LogicalPlan) -> LogicalPlan:
        """Build plan for ORDER BY."""
        if not stmt.order_by:
            return plan

        expressions = [item.expression for item in stmt.order_by.items]
        ascending = [item.order.name == "ASC" for item in stmt.order_by.items]
        nulls_first = [
            item.nulls.name == "NULLS_FIRST" if item.nulls else True
            for item in stmt.order_by.items
        ]

        return LogicalSort(
            sort_expressions=expressions,
            ascending=ascending,
            nulls_first=nulls_first,
            input=plan,
        )

    def _build_limit(self, stmt: SelectStatement, plan: LogicalPlan) -> LogicalPlan:
        """Build plan for LIMIT/OFFSET."""
        if not stmt.limit:
            return plan
        return LogicalLimit(
            limit=stmt.limit.limit,
            offset=stmt.limit.offset or 0,
            input=plan,
        )

    def _build_set_operation(self, stmt: SelectStatement, plan: LogicalPlan) -> LogicalPlan:
        """Build plan for UNION/INTERSECT/EXCEPT."""
        if not stmt.set_operation or not stmt.right_query:
            return plan

        right_plan = self.build(stmt.right_query)

        if stmt.set_operation == "UNION":
            return LogicalUnion(left=plan, right=right_plan, all=stmt.set_all)

        # For INTERSECT and EXCEPT, would need additional plan types
        # For now, treat as union
        return LogicalUnion(left=plan, right=right_plan, all=stmt.set_all)
