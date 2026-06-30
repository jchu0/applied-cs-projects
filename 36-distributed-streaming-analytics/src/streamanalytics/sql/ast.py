"""SQL Abstract Syntax Tree nodes.

Defines the AST structure for parsed SQL statements including
expressions, clauses, and complete statements.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional


class DataType(Enum):
    """SQL data types."""

    INTEGER = auto()
    BIGINT = auto()
    FLOAT = auto()
    DOUBLE = auto()
    DECIMAL = auto()
    VARCHAR = auto()
    TEXT = auto()
    BOOLEAN = auto()
    TIMESTAMP = auto()
    DATE = auto()
    TIME = auto()
    INTERVAL = auto()
    ARRAY = auto()
    MAP = auto()
    STRUCT = auto()
    NULL = auto()
    UNKNOWN = auto()


class JoinType(Enum):
    """SQL JOIN types."""

    INNER = auto()
    LEFT = auto()
    RIGHT = auto()
    FULL = auto()
    CROSS = auto()
    LEFT_SEMI = auto()
    LEFT_ANTI = auto()


class AggregateFunction(Enum):
    """SQL aggregate functions."""

    COUNT = auto()
    SUM = auto()
    AVG = auto()
    MIN = auto()
    MAX = auto()
    COUNT_DISTINCT = auto()
    STDDEV = auto()
    VARIANCE = auto()
    FIRST = auto()
    LAST = auto()
    COLLECT_LIST = auto()
    COLLECT_SET = auto()


class SortOrder(Enum):
    """Sort order for ORDER BY."""

    ASC = auto()
    DESC = auto()


class NullOrdering(Enum):
    """NULL ordering for ORDER BY."""

    NULLS_FIRST = auto()
    NULLS_LAST = auto()


# ============================================================================
# Expression Nodes
# ============================================================================

@dataclass
class Expression(ABC):
    """Base class for all SQL expressions."""

    alias: Optional[str] = None
    data_type: DataType = DataType.UNKNOWN

    @abstractmethod
    def children(self) -> list[Expression]:
        """Return child expressions."""
        pass

    @abstractmethod
    def to_sql(self) -> str:
        """Convert back to SQL string."""
        pass

    def transform(self, func) -> Expression:
        """Transform expression tree with a function."""
        new_children = [child.transform(func) for child in self.children()]
        return func(self._with_children(new_children))

    @abstractmethod
    def _with_children(self, children: list[Expression]) -> Expression:
        """Create copy with new children."""
        pass


@dataclass
class Literal(Expression):
    """Literal value expression."""

    value: Any = None

    def children(self) -> list[Expression]:
        return []

    def to_sql(self) -> str:
        if self.value is None:
            return "NULL"
        elif isinstance(self.value, str):
            escaped = self.value.replace("'", "''")
            return f"'{escaped}'"
        elif isinstance(self.value, bool):
            return "TRUE" if self.value else "FALSE"
        else:
            return str(self.value)

    def _with_children(self, children: list[Expression]) -> Expression:
        return Literal(value=self.value, alias=self.alias, data_type=self.data_type)


@dataclass
class ColumnRef(Expression):
    """Column reference expression."""

    name: str = ""
    table: Optional[str] = None
    schema: Optional[str] = None

    def children(self) -> list[Expression]:
        return []

    def to_sql(self) -> str:
        parts = []
        if self.schema:
            parts.append(self.schema)
        if self.table:
            parts.append(self.table)
        parts.append(self.name)
        return ".".join(parts)

    def _with_children(self, children: list[Expression]) -> Expression:
        return ColumnRef(
            name=self.name, table=self.table, schema=self.schema,
            alias=self.alias, data_type=self.data_type
        )

    @property
    def qualified_name(self) -> str:
        """Get fully qualified column name."""
        if self.table:
            return f"{self.table}.{self.name}"
        return self.name


@dataclass
class BinaryOp(Expression):
    """Binary operation expression."""

    operator: str = ""
    left: Expression = field(default_factory=lambda: Literal())
    right: Expression = field(default_factory=lambda: Literal())

    def children(self) -> list[Expression]:
        return [self.left, self.right]

    def to_sql(self) -> str:
        return f"({self.left.to_sql()} {self.operator} {self.right.to_sql()})"

    def _with_children(self, children: list[Expression]) -> Expression:
        return BinaryOp(
            operator=self.operator, left=children[0], right=children[1],
            alias=self.alias, data_type=self.data_type
        )


@dataclass
class UnaryOp(Expression):
    """Unary operation expression."""

    operator: str = ""
    operand: Expression = field(default_factory=lambda: Literal())
    is_prefix: bool = True

    def children(self) -> list[Expression]:
        return [self.operand]

    def to_sql(self) -> str:
        if self.is_prefix:
            return f"({self.operator} {self.operand.to_sql()})"
        return f"({self.operand.to_sql()} {self.operator})"

    def _with_children(self, children: list[Expression]) -> Expression:
        return UnaryOp(
            operator=self.operator, operand=children[0], is_prefix=self.is_prefix,
            alias=self.alias, data_type=self.data_type
        )


@dataclass
class FunctionCall(Expression):
    """Function call expression."""

    name: str = ""
    args: list[Expression] = field(default_factory=list)
    distinct: bool = False

    def children(self) -> list[Expression]:
        return self.args

    def to_sql(self) -> str:
        distinct_str = "DISTINCT " if self.distinct else ""
        args_str = ", ".join(arg.to_sql() for arg in self.args)
        return f"{self.name}({distinct_str}{args_str})"

    def _with_children(self, children: list[Expression]) -> Expression:
        return FunctionCall(
            name=self.name, args=children, distinct=self.distinct,
            alias=self.alias, data_type=self.data_type
        )


@dataclass
class AggregateExpr(Expression):
    """Aggregate function expression."""

    function: AggregateFunction = AggregateFunction.COUNT
    args: list[Expression] = field(default_factory=list)
    distinct: bool = False
    filter_clause: Optional[Expression] = None

    def children(self) -> list[Expression]:
        children = list(self.args)
        if self.filter_clause:
            children.append(self.filter_clause)
        return children

    def to_sql(self) -> str:
        distinct_str = "DISTINCT " if self.distinct else ""
        if self.args:
            args_str = ", ".join(arg.to_sql() for arg in self.args)
        else:
            args_str = "*"
        sql = f"{self.function.name}({distinct_str}{args_str})"
        if self.filter_clause:
            sql += f" FILTER (WHERE {self.filter_clause.to_sql()})"
        return sql

    def _with_children(self, children: list[Expression]) -> Expression:
        args = children[:-1] if self.filter_clause else children
        filter_clause = children[-1] if self.filter_clause else None
        return AggregateExpr(
            function=self.function, args=args, distinct=self.distinct,
            filter_clause=filter_clause, alias=self.alias, data_type=self.data_type
        )


@dataclass
class CaseExpr(Expression):
    """CASE expression."""

    operand: Optional[Expression] = None
    when_clauses: list[tuple[Expression, Expression]] = field(default_factory=list)
    else_clause: Optional[Expression] = None

    def children(self) -> list[Expression]:
        children = []
        if self.operand:
            children.append(self.operand)
        for when_cond, then_expr in self.when_clauses:
            children.extend([when_cond, then_expr])
        if self.else_clause:
            children.append(self.else_clause)
        return children

    def to_sql(self) -> str:
        parts = ["CASE"]
        if self.operand:
            parts.append(self.operand.to_sql())
        for when_cond, then_expr in self.when_clauses:
            parts.append(f"WHEN {when_cond.to_sql()} THEN {then_expr.to_sql()}")
        if self.else_clause:
            parts.append(f"ELSE {self.else_clause.to_sql()}")
        parts.append("END")
        return " ".join(parts)

    def _with_children(self, children: list[Expression]) -> Expression:
        idx = 0
        operand = None
        if self.operand:
            operand = children[idx]
            idx += 1
        when_clauses = []
        for _ in self.when_clauses:
            when_clauses.append((children[idx], children[idx + 1]))
            idx += 2
        else_clause = children[idx] if self.else_clause else None
        return CaseExpr(
            operand=operand, when_clauses=when_clauses, else_clause=else_clause,
            alias=self.alias, data_type=self.data_type
        )


@dataclass
class SubqueryExpr(Expression):
    """Subquery expression."""

    query: SelectStatement = field(default_factory=lambda: SelectStatement())
    subquery_type: str = "scalar"  # scalar, exists, in, any, all

    def children(self) -> list[Expression]:
        return []  # Subquery is a separate statement

    def to_sql(self) -> str:
        prefix = ""
        if self.subquery_type == "exists":
            prefix = "EXISTS "
        return f"{prefix}({self.query.to_sql()})"

    def _with_children(self, children: list[Expression]) -> Expression:
        return SubqueryExpr(
            query=self.query, subquery_type=self.subquery_type,
            alias=self.alias, data_type=self.data_type
        )


@dataclass
class InList(Expression):
    """IN (list) expression."""

    expr: Expression = field(default_factory=lambda: Literal())
    values: list[Expression] = field(default_factory=list)
    negated: bool = False

    def children(self) -> list[Expression]:
        return [self.expr] + self.values

    def to_sql(self) -> str:
        not_str = "NOT " if self.negated else ""
        values_str = ", ".join(v.to_sql() for v in self.values)
        return f"({self.expr.to_sql()} {not_str}IN ({values_str}))"

    def _with_children(self, children: list[Expression]) -> Expression:
        return InList(
            expr=children[0], values=children[1:], negated=self.negated,
            alias=self.alias, data_type=self.data_type
        )


@dataclass
class Between(Expression):
    """BETWEEN expression."""

    expr: Expression = field(default_factory=lambda: Literal())
    low: Expression = field(default_factory=lambda: Literal())
    high: Expression = field(default_factory=lambda: Literal())
    negated: bool = False

    def children(self) -> list[Expression]:
        return [self.expr, self.low, self.high]

    def to_sql(self) -> str:
        not_str = "NOT " if self.negated else ""
        return f"({self.expr.to_sql()} {not_str}BETWEEN {self.low.to_sql()} AND {self.high.to_sql()})"

    def _with_children(self, children: list[Expression]) -> Expression:
        return Between(
            expr=children[0], low=children[1], high=children[2], negated=self.negated,
            alias=self.alias, data_type=self.data_type
        )


@dataclass
class Like(Expression):
    """LIKE expression."""

    expr: Expression = field(default_factory=lambda: Literal())
    pattern: Expression = field(default_factory=lambda: Literal())
    escape: Optional[str] = None
    negated: bool = False

    def children(self) -> list[Expression]:
        return [self.expr, self.pattern]

    def to_sql(self) -> str:
        not_str = "NOT " if self.negated else ""
        sql = f"({self.expr.to_sql()} {not_str}LIKE {self.pattern.to_sql()}"
        if self.escape:
            sql += f" ESCAPE '{self.escape}'"
        return sql + ")"

    def _with_children(self, children: list[Expression]) -> Expression:
        return Like(
            expr=children[0], pattern=children[1], escape=self.escape,
            negated=self.negated, alias=self.alias, data_type=self.data_type
        )


@dataclass
class IsNull(Expression):
    """IS NULL expression."""

    expr: Expression = field(default_factory=lambda: Literal())
    negated: bool = False

    def children(self) -> list[Expression]:
        return [self.expr]

    def to_sql(self) -> str:
        not_str = "NOT " if self.negated else ""
        return f"({self.expr.to_sql()} IS {not_str}NULL)"

    def _with_children(self, children: list[Expression]) -> Expression:
        return IsNull(
            expr=children[0], negated=self.negated,
            alias=self.alias, data_type=self.data_type
        )


@dataclass
class Cast(Expression):
    """CAST expression."""

    expr: Expression = field(default_factory=lambda: Literal())
    target_type: DataType = DataType.UNKNOWN

    def children(self) -> list[Expression]:
        return [self.expr]

    def to_sql(self) -> str:
        return f"CAST({self.expr.to_sql()} AS {self.target_type.name})"

    def _with_children(self, children: list[Expression]) -> Expression:
        return Cast(
            expr=children[0], target_type=self.target_type,
            alias=self.alias, data_type=self.data_type
        )


# ============================================================================
# Clause Nodes
# ============================================================================

@dataclass
class TableRef:
    """Table reference in FROM clause."""

    name: str = ""
    schema: Optional[str] = None
    alias: Optional[str] = None

    def to_sql(self) -> str:
        sql = f"{self.schema}.{self.name}" if self.schema else self.name
        if self.alias:
            sql += f" AS {self.alias}"
        return sql

    @property
    def effective_name(self) -> str:
        """Get effective table name (alias if present)."""
        return self.alias or self.name


@dataclass
class JoinClause:
    """JOIN clause."""

    join_type: JoinType = JoinType.INNER
    table: TableRef = field(default_factory=TableRef)
    condition: Optional[Expression] = None
    using_columns: list[str] = field(default_factory=list)

    def to_sql(self) -> str:
        join_name = self.join_type.name.replace("_", " ")
        sql = f"{join_name} JOIN {self.table.to_sql()}"
        if self.condition:
            sql += f" ON {self.condition.to_sql()}"
        elif self.using_columns:
            cols = ", ".join(self.using_columns)
            sql += f" USING ({cols})"
        return sql


@dataclass
class FromClause:
    """FROM clause with joins."""

    table: TableRef = field(default_factory=TableRef)
    joins: list[JoinClause] = field(default_factory=list)

    def to_sql(self) -> str:
        sql = f"FROM {self.table.to_sql()}"
        for join in self.joins:
            sql += f" {join.to_sql()}"
        return sql


@dataclass
class WhereClause:
    """WHERE clause."""

    condition: Expression = field(default_factory=lambda: Literal(value=True))

    def to_sql(self) -> str:
        return f"WHERE {self.condition.to_sql()}"


@dataclass
class GroupByClause:
    """GROUP BY clause."""

    expressions: list[Expression] = field(default_factory=list)
    with_rollup: bool = False
    with_cube: bool = False
    grouping_sets: Optional[list[list[Expression]]] = None

    def to_sql(self) -> str:
        if self.grouping_sets:
            sets = []
            for gs in self.grouping_sets:
                exprs = ", ".join(e.to_sql() for e in gs)
                sets.append(f"({exprs})")
            return f"GROUP BY GROUPING SETS ({', '.join(sets)})"

        exprs_sql = ", ".join(e.to_sql() for e in self.expressions)
        sql = f"GROUP BY {exprs_sql}"
        if self.with_rollup:
            sql += " WITH ROLLUP"
        elif self.with_cube:
            sql += " WITH CUBE"
        return sql


@dataclass
class HavingClause:
    """HAVING clause."""

    condition: Expression = field(default_factory=lambda: Literal(value=True))

    def to_sql(self) -> str:
        return f"HAVING {self.condition.to_sql()}"


@dataclass
class OrderByItem:
    """Single ORDER BY item."""

    expression: Expression = field(default_factory=lambda: Literal())
    order: SortOrder = SortOrder.ASC
    nulls: Optional[NullOrdering] = None

    def to_sql(self) -> str:
        sql = self.expression.to_sql()
        sql += f" {self.order.name}"
        if self.nulls:
            sql += f" {self.nulls.name.replace('_', ' ')}"
        return sql


@dataclass
class OrderByClause:
    """ORDER BY clause."""

    items: list[OrderByItem] = field(default_factory=list)

    def to_sql(self) -> str:
        items_sql = ", ".join(item.to_sql() for item in self.items)
        return f"ORDER BY {items_sql}"


@dataclass
class LimitClause:
    """LIMIT/OFFSET clause."""

    limit: Optional[int] = None
    offset: Optional[int] = None

    def to_sql(self) -> str:
        parts = []
        if self.limit is not None:
            parts.append(f"LIMIT {self.limit}")
        if self.offset is not None:
            parts.append(f"OFFSET {self.offset}")
        return " ".join(parts)


# ============================================================================
# Statement Nodes
# ============================================================================

@dataclass
class SelectStatement:
    """Complete SELECT statement."""

    select_list: list[Expression] = field(default_factory=list)
    distinct: bool = False
    from_clause: Optional[FromClause] = None
    where_clause: Optional[WhereClause] = None
    group_by: Optional[GroupByClause] = None
    having: Optional[HavingClause] = None
    order_by: Optional[OrderByClause] = None
    limit: Optional[LimitClause] = None

    # For UNION/INTERSECT/EXCEPT
    set_operation: Optional[str] = None  # UNION, INTERSECT, EXCEPT
    set_all: bool = False
    right_query: Optional[SelectStatement] = None

    def to_sql(self) -> str:
        parts = ["SELECT"]

        if self.distinct:
            parts.append("DISTINCT")

        if self.select_list:
            select_items = []
            for expr in self.select_list:
                sql = expr.to_sql()
                if expr.alias:
                    sql += f" AS {expr.alias}"
                select_items.append(sql)
            parts.append(", ".join(select_items))
        else:
            parts.append("*")

        if self.from_clause:
            parts.append(self.from_clause.to_sql())

        if self.where_clause:
            parts.append(self.where_clause.to_sql())

        if self.group_by:
            parts.append(self.group_by.to_sql())

        if self.having:
            parts.append(self.having.to_sql())

        if self.order_by:
            parts.append(self.order_by.to_sql())

        if self.limit:
            parts.append(self.limit.to_sql())

        sql = " ".join(parts)

        if self.set_operation and self.right_query:
            all_str = " ALL" if self.set_all else ""
            sql = f"({sql}) {self.set_operation}{all_str} ({self.right_query.to_sql()})"

        return sql


@dataclass
class CreateTableStatement:
    """CREATE TABLE statement."""

    name: str = ""
    schema: Optional[str] = None
    columns: list[tuple[str, DataType]] = field(default_factory=list)
    if_not_exists: bool = False
    as_query: Optional[SelectStatement] = None

    def to_sql(self) -> str:
        if_not_exists_str = "IF NOT EXISTS " if self.if_not_exists else ""
        table_name = f"{self.schema}.{self.name}" if self.schema else self.name

        if self.as_query:
            return f"CREATE TABLE {if_not_exists_str}{table_name} AS {self.as_query.to_sql()}"

        cols = ", ".join(f"{name} {dtype.name}" for name, dtype in self.columns)
        return f"CREATE TABLE {if_not_exists_str}{table_name} ({cols})"


@dataclass
class InsertStatement:
    """INSERT statement."""

    table: str = ""
    schema: Optional[str] = None
    columns: list[str] = field(default_factory=list)
    values: list[list[Expression]] = field(default_factory=list)
    query: Optional[SelectStatement] = None

    def to_sql(self) -> str:
        table_name = f"{self.schema}.{self.table}" if self.schema else self.table
        cols_str = f"({', '.join(self.columns)})" if self.columns else ""

        if self.query:
            return f"INSERT INTO {table_name} {cols_str} {self.query.to_sql()}"

        values_strs = []
        for row in self.values:
            row_str = ", ".join(expr.to_sql() for expr in row)
            values_strs.append(f"({row_str})")

        return f"INSERT INTO {table_name} {cols_str} VALUES {', '.join(values_strs)}"
