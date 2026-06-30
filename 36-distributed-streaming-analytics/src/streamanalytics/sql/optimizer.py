"""Logical Optimizer - Rule-based query optimization.

Implements optimization rules including predicate pushdown, projection
pruning, join reordering, constant folding, and common subexpression
elimination.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Callable
import copy

from .ast import (
    Expression, ColumnRef, Literal, BinaryOp, UnaryOp, FunctionCall,
    AggregateExpr, CaseExpr, InList, Between, IsNull, DataType,
)
from .logical import (
    LogicalPlan, LogicalScan, LogicalFilter, LogicalProject, LogicalJoin,
    LogicalAggregate, LogicalSort, LogicalLimit, LogicalUnion, LogicalDistinct,
    Schema, Statistics,
)


# ============================================================================
# Optimization Rules
# ============================================================================

class OptimizationRule(ABC):
    """Base class for optimization rules."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Rule name for logging."""
        pass

    @abstractmethod
    def apply(self, plan: LogicalPlan) -> tuple[LogicalPlan, bool]:
        """Apply rule to plan. Returns (new_plan, changed)."""
        pass


class PredicatePushdown(OptimizationRule):
    """Push filter predicates down through the plan tree.

    Benefits:
    - Reduces data early in the pipeline
    - Can enable index scans
    - Reduces join sizes
    """

    @property
    def name(self) -> str:
        return "PredicatePushdown"

    def apply(self, plan: LogicalPlan) -> tuple[LogicalPlan, bool]:
        changed = False
        result = self._push_down(plan)
        if result is not plan:
            changed = True
        return result, changed

    def _push_down(self, plan: LogicalPlan) -> LogicalPlan:
        """Recursively push predicates down."""
        if isinstance(plan, LogicalFilter):
            return self._push_filter(plan)

        # Recursively process children
        new_children = [self._push_down(child) for child in plan.children()]
        return plan._with_children(new_children)

    def _push_filter(self, filter_node: LogicalFilter) -> LogicalPlan:
        """Push a filter node down if possible."""
        predicate = filter_node.predicate
        input_plan = filter_node.input

        # Push through Project
        if isinstance(input_plan, LogicalProject):
            pushed = LogicalFilter(
                predicate=predicate,
                input=input_plan.input,
                schema=input_plan.input.schema,
            )
            return LogicalProject(
                expressions=input_plan.expressions,
                input=self._push_down(pushed),
                aliases=input_plan.aliases,
                schema=input_plan.schema,
            )

        # Push through Join
        if isinstance(input_plan, LogicalJoin):
            return self._push_through_join(predicate, input_plan)

        # Can't push further - return original
        return LogicalFilter(
            predicate=predicate,
            input=self._push_down(input_plan),
            schema=filter_node.schema,
        )

    def _push_through_join(self, predicate: Expression, join: LogicalJoin) -> LogicalPlan:
        """Push predicate through a join."""
        # Split conjunctions
        predicates = self._split_conjunctions(predicate)

        left_predicates = []
        right_predicates = []
        join_predicates = []
        remaining_predicates = []

        left_tables = self._get_table_references(join.left)
        right_tables = self._get_table_references(join.right)

        for pred in predicates:
            tables = self._get_referenced_tables(pred)

            if tables.issubset(left_tables):
                left_predicates.append(pred)
            elif tables.issubset(right_tables):
                right_predicates.append(pred)
            elif tables.issubset(left_tables | right_tables):
                join_predicates.append(pred)
            else:
                remaining_predicates.append(pred)

        # Build new left side
        new_left = join.left
        if left_predicates:
            new_left = LogicalFilter(
                predicate=self._combine_conjunctions(left_predicates),
                input=new_left,
            )
        new_left = self._push_down(new_left)

        # Build new right side
        new_right = join.right
        if right_predicates:
            new_right = LogicalFilter(
                predicate=self._combine_conjunctions(right_predicates),
                input=new_right,
            )
        new_right = self._push_down(new_right)

        # Build new join
        new_condition = join.condition
        if join_predicates:
            all_join_preds = join_predicates
            if join.condition:
                all_join_preds = [join.condition] + join_predicates
            new_condition = self._combine_conjunctions(all_join_preds)

        new_join = LogicalJoin(
            join_type=join.join_type,
            left=new_left,
            right=new_right,
            condition=new_condition,
            using_columns=join.using_columns,
            schema=join.schema,
        )

        # Apply remaining predicates on top
        if remaining_predicates:
            return LogicalFilter(
                predicate=self._combine_conjunctions(remaining_predicates),
                input=new_join,
            )

        return new_join

    def _split_conjunctions(self, expr: Expression) -> list[Expression]:
        """Split AND expressions into list."""
        if isinstance(expr, BinaryOp) and expr.operator == "AND":
            left_parts = self._split_conjunctions(expr.left)
            right_parts = self._split_conjunctions(expr.right)
            return left_parts + right_parts
        return [expr]

    def _combine_conjunctions(self, predicates: list[Expression]) -> Expression:
        """Combine expressions with AND."""
        if not predicates:
            return Literal(value=True, data_type=DataType.BOOLEAN)
        result = predicates[0]
        for pred in predicates[1:]:
            result = BinaryOp(operator="AND", left=result, right=pred)
        return result

    def _get_table_references(self, plan: LogicalPlan) -> set[str]:
        """Get all table names referenced by a plan."""
        tables = set()
        if isinstance(plan, LogicalScan):
            tables.add(plan.effective_name)
        for child in plan.children():
            tables.update(self._get_table_references(child))
        return tables

    def _get_referenced_tables(self, expr: Expression) -> set[str]:
        """Get table names referenced in an expression."""
        tables = set()
        if isinstance(expr, ColumnRef) and expr.table:
            tables.add(expr.table)
        for child in expr.children():
            tables.update(self._get_referenced_tables(child))
        return tables


class ProjectionPruning(OptimizationRule):
    """Remove unused columns from projections.

    Benefits:
    - Reduces memory usage
    - Improves cache efficiency
    - Can enable more efficient scans
    """

    @property
    def name(self) -> str:
        return "ProjectionPruning"

    def apply(self, plan: LogicalPlan) -> tuple[LogicalPlan, bool]:
        # Collect required columns from the root
        required = self._collect_required_columns(plan)
        result = self._prune(plan, required)
        changed = result is not plan
        return result, changed

    def _collect_required_columns(self, plan: LogicalPlan) -> set[str]:
        """Collect all columns required by a plan."""
        required = set()

        for expr in plan.collect_expressions():
            required.update(self._get_columns_from_expr(expr))

        for child in plan.children():
            required.update(self._collect_required_columns(child))

        return required

    def _get_columns_from_expr(self, expr: Expression) -> set[str]:
        """Get column names from an expression."""
        columns = set()
        if isinstance(expr, ColumnRef):
            columns.add(expr.qualified_name)
        for child in expr.children():
            columns.update(self._get_columns_from_expr(child))
        return columns

    def _prune(self, plan: LogicalPlan, required: set[str]) -> LogicalPlan:
        """Prune unused columns from plan."""
        if isinstance(plan, LogicalProject):
            # Keep only required expressions
            new_exprs = []
            new_aliases = []
            for expr, alias in zip(plan.expressions, plan.aliases):
                output_name = alias or (expr.to_sql() if not isinstance(expr, ColumnRef) else expr.name)
                if output_name in required or self._expr_produces_required(expr, required):
                    new_exprs.append(expr)
                    new_aliases.append(alias)

            # Always keep at least one expression
            if not new_exprs:
                new_exprs = [plan.expressions[0]] if plan.expressions else []
                new_aliases = [plan.aliases[0]] if plan.aliases else []

            return LogicalProject(
                expressions=new_exprs,
                input=self._prune(plan.input, required),
                aliases=new_aliases,
                schema=plan.schema,
            )

        if isinstance(plan, LogicalScan):
            # Add projection pushdown to scan
            if plan.schema.columns:
                indices = []
                for i, (col_name, _) in enumerate(plan.schema.columns):
                    full_name = f"{plan.effective_name}.{col_name}"
                    if col_name in required or full_name in required:
                        indices.append(i)
                if indices and len(indices) < len(plan.schema.columns):
                    return LogicalScan(
                        table_name=plan.table_name,
                        table_schema=plan.table_schema,
                        alias=plan.alias,
                        projection=indices,
                        schema=plan.schema,
                    )
            return plan

        # Recursively process children
        new_children = [self._prune(child, required) for child in plan.children()]
        return plan._with_children(new_children)

    def _expr_produces_required(self, expr: Expression, required: set[str]) -> bool:
        """Check if expression produces any required column."""
        if isinstance(expr, ColumnRef):
            return expr.qualified_name in required or expr.name in required
        return any(self._expr_produces_required(child, required) for child in expr.children())


class JoinReordering(OptimizationRule):
    """Reorder joins for better performance.

    Benefits:
    - Smaller intermediate results
    - Better use of indexes
    - Reduced memory usage
    """

    @property
    def name(self) -> str:
        return "JoinReordering"

    def apply(self, plan: LogicalPlan) -> tuple[LogicalPlan, bool]:
        result = self._reorder(plan)
        changed = result is not plan
        return result, changed

    def _reorder(self, plan: LogicalPlan) -> LogicalPlan:
        """Reorder joins in the plan."""
        if isinstance(plan, LogicalJoin):
            # Collect all join inputs and conditions
            inputs, conditions = self._flatten_joins(plan)

            if len(inputs) <= 2:
                # Nothing to reorder
                new_children = [self._reorder(child) for child in plan.children()]
                return plan._with_children(new_children)

            # Simple heuristic: put smaller tables first (based on stats)
            ordered_inputs = self._order_by_size(inputs)

            # Rebuild join tree
            result = ordered_inputs[0]
            for i, input_plan in enumerate(ordered_inputs[1:]):
                # Find applicable condition
                applicable_cond = self._find_condition(
                    result, input_plan, conditions
                )
                result = LogicalJoin(
                    join_type=plan.join_type,
                    left=result,
                    right=input_plan,
                    condition=applicable_cond,
                )

            return result

        # Recursively process children
        new_children = [self._reorder(child) for child in plan.children()]
        return plan._with_children(new_children)

    def _flatten_joins(self, plan: LogicalPlan) -> tuple[list[LogicalPlan], list[Expression]]:
        """Flatten nested joins into list of inputs and conditions."""
        inputs = []
        conditions = []

        if isinstance(plan, LogicalJoin):
            if plan.join_type.name == "INNER":
                left_inputs, left_conds = self._flatten_joins(plan.left)
                right_inputs, right_conds = self._flatten_joins(plan.right)
                inputs = left_inputs + right_inputs
                conditions = left_conds + right_conds
                if plan.condition:
                    conditions.append(plan.condition)
            else:
                # Non-inner joins can't be reordered freely
                inputs = [plan]
        else:
            inputs = [plan]

        return inputs, conditions

    def _order_by_size(self, inputs: list[LogicalPlan]) -> list[LogicalPlan]:
        """Order inputs by estimated size (smaller first)."""
        def get_size(plan: LogicalPlan) -> int:
            if plan.stats and plan.stats.row_count:
                return plan.stats.row_count
            # Default heuristic
            if isinstance(plan, LogicalScan):
                return 1000  # Base table
            if isinstance(plan, LogicalFilter):
                return 100  # Filtered
            return 10000  # Unknown

        return sorted(inputs, key=get_size)

    def _find_condition(
        self, left: LogicalPlan, right: LogicalPlan, conditions: list[Expression]
    ) -> Optional[Expression]:
        """Find condition that applies to left and right."""
        left_tables = self._get_tables(left)
        right_tables = self._get_tables(right)
        all_tables = left_tables | right_tables

        applicable = []
        for cond in conditions:
            cond_tables = self._get_condition_tables(cond)
            if cond_tables.issubset(all_tables) and cond_tables & left_tables and cond_tables & right_tables:
                applicable.append(cond)

        if not applicable:
            return None

        if len(applicable) == 1:
            return applicable[0]

        # Combine with AND
        result = applicable[0]
        for cond in applicable[1:]:
            result = BinaryOp(operator="AND", left=result, right=cond)
        return result

    def _get_tables(self, plan: LogicalPlan) -> set[str]:
        """Get all tables in a plan."""
        tables = set()
        if isinstance(plan, LogicalScan):
            tables.add(plan.effective_name)
        for child in plan.children():
            tables.update(self._get_tables(child))
        return tables

    def _get_condition_tables(self, expr: Expression) -> set[str]:
        """Get tables referenced in a condition."""
        tables = set()
        if isinstance(expr, ColumnRef) and expr.table:
            tables.add(expr.table)
        for child in expr.children():
            tables.update(self._get_condition_tables(child))
        return tables


class ConstantFolding(OptimizationRule):
    """Evaluate constant expressions at compile time.

    Benefits:
    - Reduces runtime computation
    - Simplifies plan structure
    - Can enable other optimizations
    """

    @property
    def name(self) -> str:
        return "ConstantFolding"

    def apply(self, plan: LogicalPlan) -> tuple[LogicalPlan, bool]:
        changed = [False]  # Use list to allow mutation in nested function

        def fold_in_plan(p: LogicalPlan) -> LogicalPlan:
            # Fold expressions in this node
            new_exprs = []
            for expr in p.collect_expressions():
                folded = self._fold_expression(expr)
                if folded is not expr:
                    changed[0] = True
                new_exprs.append(folded)

            # Update node with folded expressions
            result = self._update_expressions(p, new_exprs)

            # Recursively process children
            new_children = [fold_in_plan(child) for child in result.children()]
            return result._with_children(new_children)

        result = fold_in_plan(plan)
        return result, changed[0]

    def _fold_expression(self, expr: Expression) -> Expression:
        """Fold constant expressions."""
        # First fold children
        folded_children = [self._fold_expression(child) for child in expr.children()]
        expr = expr._with_children(folded_children)

        # Check if all children are literals
        if not all(isinstance(child, Literal) for child in expr.children()):
            return expr

        # Fold binary operations
        if isinstance(expr, BinaryOp):
            left_val = expr.left.value if isinstance(expr.left, Literal) else None
            right_val = expr.right.value if isinstance(expr.right, Literal) else None

            if left_val is None or right_val is None:
                return expr

            try:
                result = self._eval_binary_op(expr.operator, left_val, right_val)
                return Literal(value=result, data_type=self._infer_type(result))
            except Exception:
                return expr

        # Fold unary operations
        if isinstance(expr, UnaryOp):
            operand_val = expr.operand.value if isinstance(expr.operand, Literal) else None

            if operand_val is None:
                return expr

            try:
                result = self._eval_unary_op(expr.operator, operand_val)
                return Literal(value=result, data_type=self._infer_type(result))
            except Exception:
                return expr

        return expr

    def _eval_binary_op(self, op: str, left, right):
        """Evaluate binary operation."""
        ops = {
            "+": lambda a, b: a + b,
            "-": lambda a, b: a - b,
            "*": lambda a, b: a * b,
            "/": lambda a, b: a / b if b != 0 else None,
            "%": lambda a, b: a % b if b != 0 else None,
            "=": lambda a, b: a == b,
            "<>": lambda a, b: a != b,
            "<": lambda a, b: a < b,
            "<=": lambda a, b: a <= b,
            ">": lambda a, b: a > b,
            ">=": lambda a, b: a >= b,
            "AND": lambda a, b: a and b,
            "OR": lambda a, b: a or b,
            "||": lambda a, b: str(a) + str(b),
        }
        if op in ops:
            return ops[op](left, right)
        raise ValueError(f"Unknown operator: {op}")

    def _eval_unary_op(self, op: str, operand):
        """Evaluate unary operation."""
        if op == "-":
            return -operand
        if op == "NOT":
            return not operand
        raise ValueError(f"Unknown operator: {op}")

    def _infer_type(self, value) -> DataType:
        """Infer data type from Python value."""
        if value is None:
            return DataType.NULL
        if isinstance(value, bool):
            return DataType.BOOLEAN
        if isinstance(value, int):
            return DataType.INTEGER
        if isinstance(value, float):
            return DataType.DOUBLE
        if isinstance(value, str):
            return DataType.VARCHAR
        return DataType.UNKNOWN

    def _update_expressions(self, plan: LogicalPlan, new_exprs: list[Expression]) -> LogicalPlan:
        """Update plan node with new expressions."""
        if isinstance(plan, LogicalFilter) and new_exprs:
            return LogicalFilter(predicate=new_exprs[0], input=plan.input, schema=plan.schema)
        if isinstance(plan, LogicalProject) and new_exprs:
            return LogicalProject(
                expressions=new_exprs, input=plan.input,
                aliases=plan.aliases, schema=plan.schema
            )
        # For other node types, return unchanged
        return plan


class CommonSubexpressionElimination(OptimizationRule):
    """Identify and eliminate common subexpressions.

    Benefits:
    - Avoids redundant computation
    - Reduces memory for intermediate results
    """

    @property
    def name(self) -> str:
        return "CommonSubexpressionElimination"

    def apply(self, plan: LogicalPlan) -> tuple[LogicalPlan, bool]:
        # Collect all expressions
        expr_counts: dict[str, int] = {}
        self._count_expressions(plan, expr_counts)

        # Find duplicates
        duplicates = {sql for sql, count in expr_counts.items() if count > 1}

        if not duplicates:
            return plan, False

        # Replace duplicates with references
        result = self._eliminate(plan, duplicates)
        return result, True

    def _count_expressions(self, plan: LogicalPlan, counts: dict[str, int]) -> None:
        """Count expression occurrences."""
        for expr in plan.collect_expressions():
            self._count_expr(expr, counts)
        for child in plan.children():
            self._count_expressions(child, counts)

    def _count_expr(self, expr: Expression, counts: dict[str, int]) -> None:
        """Count a single expression and its children."""
        sql = expr.to_sql()
        counts[sql] = counts.get(sql, 0) + 1
        for child in expr.children():
            self._count_expr(child, counts)

    def _eliminate(self, plan: LogicalPlan, duplicates: set[str]) -> LogicalPlan:
        """Eliminate common subexpressions."""
        # For now, just mark duplicates - full implementation would
        # extract to common projection and reference
        new_children = [self._eliminate(child, duplicates) for child in plan.children()]
        return plan._with_children(new_children)


# ============================================================================
# Optimizer
# ============================================================================

class LogicalOptimizer:
    """Query optimizer that applies rules to logical plans."""

    def __init__(self, rules: Optional[list[OptimizationRule]] = None):
        """Initialize with optimization rules."""
        if rules is None:
            rules = [
                ConstantFolding(),
                PredicatePushdown(),
                ProjectionPruning(),
                CommonSubexpressionElimination(),
                JoinReordering(),
            ]
        self.rules = rules
        self.max_iterations = 10

    def optimize(self, plan: LogicalPlan) -> LogicalPlan:
        """Apply optimization rules until fixpoint."""
        current = plan

        for iteration in range(self.max_iterations):
            changed = False

            for rule in self.rules:
                new_plan, rule_changed = rule.apply(current)
                if rule_changed:
                    changed = True
                    current = new_plan

            if not changed:
                break

        return current

    def explain(self, plan: LogicalPlan) -> str:
        """Generate explain output showing optimization steps."""
        lines = ["=== Original Plan ==="]
        lines.append(plan.to_string())
        lines.append("")

        current = plan
        for rule in self.rules:
            new_plan, changed = rule.apply(current)
            if changed:
                lines.append(f"=== After {rule.name} ===")
                lines.append(new_plan.to_string())
                lines.append("")
                current = new_plan

        lines.append("=== Final Plan ===")
        lines.append(current.to_string())

        return "\n".join(lines)
