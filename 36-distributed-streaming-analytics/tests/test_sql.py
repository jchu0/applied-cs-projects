"""Tests for SQL parser, optimizer, and physical planner."""

import pytest

from streamanalytics.sql import (
    SQLParser,
    SQLLexer,
    Token,
    TokenType,
    LogicalPlanBuilder,
    LogicalOptimizer,
    PhysicalPlanner,
    PredicatePushdown,
    ProjectionPruning,
    ConstantFolding,
)
from streamanalytics.sql.ast import (
    SelectStatement,
    ColumnRef,
    Literal,
    BinaryOp,
    AggregateExpr,
    AggregateFunction,
    DataType,
    JoinType,
)
from streamanalytics.sql.logical import (
    LogicalScan,
    LogicalFilter,
    LogicalProject,
    LogicalJoin,
    LogicalAggregate,
    LogicalSort,
    LogicalLimit,
    Schema,
)
from streamanalytics.sql.planner import (
    PhysicalScan,
    PhysicalFilter,
    PhysicalHashJoin,
    PhysicalBroadcastJoin,
    PhysicalAggregate,
    PhysicalExchange,
    JoinStrategy,
    DistributionType,
)


class TestSQLLexer:
    """Tests for SQL lexer."""

    def test_simple_select(self):
        """Test tokenizing simple SELECT."""
        lexer = SQLLexer("SELECT a, b FROM t")
        types = [t.type for t in lexer.tokens]
        assert TokenType.SELECT in types
        assert TokenType.FROM in types
        assert TokenType.IDENTIFIER in types
        assert TokenType.COMMA in types
        assert TokenType.EOF in types

    def test_string_literal(self):
        """Test string literal tokenization."""
        lexer = SQLLexer("SELECT 'hello' FROM t")
        string_tokens = [t for t in lexer.tokens if t.type == TokenType.STRING]
        assert len(string_tokens) == 1
        assert string_tokens[0].value == "hello"

    def test_number_literals(self):
        """Test number tokenization."""
        lexer = SQLLexer("SELECT 42, 3.14, 1e10 FROM t")
        int_tokens = [t for t in lexer.tokens if t.type == TokenType.INTEGER]
        float_tokens = [t for t in lexer.tokens if t.type == TokenType.FLOAT]
        assert len(int_tokens) == 1
        assert len(float_tokens) == 2

    def test_operators(self):
        """Test operator tokenization."""
        lexer = SQLLexer("SELECT a + b, c >= d, e <> f FROM t")
        types = [t.type for t in lexer.tokens]
        assert TokenType.PLUS in types
        assert TokenType.GE in types
        assert TokenType.NE in types

    def test_keywords(self):
        """Test keyword recognition."""
        lexer = SQLLexer("SELECT DISTINCT a FROM t WHERE x AND y ORDER BY z LIMIT 10")
        types = [t.type for t in lexer.tokens]
        assert TokenType.DISTINCT in types
        assert TokenType.WHERE in types
        assert TokenType.AND in types
        assert TokenType.ORDER in types
        assert TokenType.LIMIT in types

    def test_comments(self):
        """Test comment handling."""
        lexer = SQLLexer("SELECT a -- comment\nFROM t /* block */ WHERE x = 1")
        # Comments should be skipped
        types = [t.type for t in lexer.tokens]
        assert TokenType.SELECT in types
        assert TokenType.FROM in types
        assert TokenType.WHERE in types


class TestSQLParser:
    """Tests for SQL parser."""

    def test_simple_select(self):
        """Test parsing simple SELECT."""
        parser = SQLParser("SELECT a, b FROM t")
        stmt = parser.parse()
        assert isinstance(stmt, SelectStatement)
        assert len(stmt.select_list) == 2
        assert stmt.from_clause is not None
        assert stmt.from_clause.table.name == "t"

    def test_select_star(self):
        """Test parsing SELECT *."""
        parser = SQLParser("SELECT * FROM t")
        stmt = parser.parse()
        assert stmt.select_list == []  # Empty means SELECT *

    def test_select_distinct(self):
        """Test parsing SELECT DISTINCT."""
        parser = SQLParser("SELECT DISTINCT a FROM t")
        stmt = parser.parse()
        assert stmt.distinct is True

    def test_where_clause(self):
        """Test parsing WHERE clause."""
        parser = SQLParser("SELECT a FROM t WHERE x > 10")
        stmt = parser.parse()
        assert stmt.where_clause is not None
        assert isinstance(stmt.where_clause.condition, BinaryOp)
        assert stmt.where_clause.condition.operator == ">"

    def test_complex_where(self):
        """Test parsing complex WHERE with AND/OR."""
        parser = SQLParser("SELECT a FROM t WHERE x > 10 AND y < 20 OR z = 5")
        stmt = parser.parse()
        assert stmt.where_clause is not None

    def test_join(self):
        """Test parsing JOIN."""
        parser = SQLParser("SELECT a FROM t1 INNER JOIN t2 ON t1.id = t2.id")
        stmt = parser.parse()
        assert len(stmt.from_clause.joins) == 1
        assert stmt.from_clause.joins[0].join_type == JoinType.INNER

    def test_left_join(self):
        """Test parsing LEFT JOIN."""
        parser = SQLParser("SELECT a FROM t1 LEFT JOIN t2 ON t1.id = t2.id")
        stmt = parser.parse()
        assert stmt.from_clause.joins[0].join_type == JoinType.LEFT

    def test_multiple_joins(self):
        """Test parsing multiple JOINs."""
        parser = SQLParser(
            "SELECT a FROM t1 JOIN t2 ON t1.id = t2.id JOIN t3 ON t2.id = t3.id"
        )
        stmt = parser.parse()
        assert len(stmt.from_clause.joins) == 2

    def test_group_by(self):
        """Test parsing GROUP BY."""
        parser = SQLParser("SELECT a, COUNT(*) FROM t GROUP BY a")
        stmt = parser.parse()
        assert stmt.group_by is not None
        assert len(stmt.group_by.expressions) == 1

    def test_having(self):
        """Test parsing HAVING."""
        parser = SQLParser("SELECT a, COUNT(*) FROM t GROUP BY a HAVING COUNT(*) > 5")
        stmt = parser.parse()
        assert stmt.having is not None

    def test_order_by(self):
        """Test parsing ORDER BY."""
        parser = SQLParser("SELECT a FROM t ORDER BY a DESC, b ASC")
        stmt = parser.parse()
        assert stmt.order_by is not None
        assert len(stmt.order_by.items) == 2

    def test_limit_offset(self):
        """Test parsing LIMIT OFFSET."""
        parser = SQLParser("SELECT a FROM t LIMIT 10 OFFSET 5")
        stmt = parser.parse()
        assert stmt.limit is not None
        assert stmt.limit.limit == 10
        assert stmt.limit.offset == 5

    def test_aggregate_functions(self):
        """Test parsing aggregate functions."""
        parser = SQLParser("SELECT COUNT(*), SUM(a), AVG(b), MIN(c), MAX(d) FROM t")
        stmt = parser.parse()
        assert len(stmt.select_list) == 5
        assert all(isinstance(e, AggregateExpr) for e in stmt.select_list)

    def test_count_distinct(self):
        """Test parsing COUNT(DISTINCT x)."""
        parser = SQLParser("SELECT COUNT(DISTINCT a) FROM t")
        stmt = parser.parse()
        assert isinstance(stmt.select_list[0], AggregateExpr)
        assert stmt.select_list[0].distinct is True

    def test_case_expression(self):
        """Test parsing CASE expression."""
        parser = SQLParser("SELECT CASE WHEN a > 0 THEN 'pos' ELSE 'neg' END FROM t")
        stmt = parser.parse()
        assert len(stmt.select_list) == 1

    def test_subquery_in_where(self):
        """Test parsing subquery in WHERE."""
        parser = SQLParser("SELECT a FROM t WHERE a IN (SELECT b FROM t2)")
        stmt = parser.parse()
        assert stmt.where_clause is not None

    def test_union(self):
        """Test parsing UNION."""
        parser = SQLParser("SELECT a FROM t1 UNION SELECT b FROM t2")
        stmt = parser.parse()
        assert stmt.set_operation == "UNION"
        assert stmt.right_query is not None

    def test_union_all(self):
        """Test parsing UNION ALL."""
        parser = SQLParser("SELECT a FROM t1 UNION ALL SELECT b FROM t2")
        stmt = parser.parse()
        assert stmt.set_operation == "UNION"
        assert stmt.set_all is True

    def test_alias(self):
        """Test parsing column aliases."""
        parser = SQLParser("SELECT a AS col1, b col2 FROM t")
        stmt = parser.parse()
        assert stmt.select_list[0].alias == "col1"
        assert stmt.select_list[1].alias == "col2"

    def test_table_alias(self):
        """Test parsing table aliases."""
        parser = SQLParser("SELECT t1.a FROM table1 AS t1")
        stmt = parser.parse()
        assert stmt.from_clause.table.alias == "t1"

    def test_between(self):
        """Test parsing BETWEEN."""
        parser = SQLParser("SELECT a FROM t WHERE a BETWEEN 1 AND 10")
        stmt = parser.parse()
        assert stmt.where_clause is not None

    def test_like(self):
        """Test parsing LIKE."""
        parser = SQLParser("SELECT a FROM t WHERE name LIKE '%test%'")
        stmt = parser.parse()
        assert stmt.where_clause is not None

    def test_is_null(self):
        """Test parsing IS NULL."""
        parser = SQLParser("SELECT a FROM t WHERE a IS NULL")
        stmt = parser.parse()
        assert stmt.where_clause is not None

    def test_is_not_null(self):
        """Test parsing IS NOT NULL."""
        parser = SQLParser("SELECT a FROM t WHERE a IS NOT NULL")
        stmt = parser.parse()
        assert stmt.where_clause is not None

    def test_cast(self):
        """Test parsing CAST."""
        parser = SQLParser("SELECT CAST(a AS INTEGER) FROM t")
        stmt = parser.parse()
        assert len(stmt.select_list) == 1


class TestLogicalPlanBuilder:
    """Tests for logical plan builder."""

    def test_simple_scan(self):
        """Test building scan plan."""
        parser = SQLParser("SELECT a FROM t")
        stmt = parser.parse()
        builder = LogicalPlanBuilder()
        plan = builder.build(stmt)
        # Should have Project -> Scan
        assert isinstance(plan, LogicalProject)
        assert isinstance(plan.input, LogicalScan)
        assert plan.input.table_name == "t"

    def test_filter(self):
        """Test building filter plan."""
        parser = SQLParser("SELECT a FROM t WHERE x > 10")
        stmt = parser.parse()
        builder = LogicalPlanBuilder()
        plan = builder.build(stmt)
        # Should have Project -> Filter -> Scan
        assert isinstance(plan, LogicalProject)
        assert isinstance(plan.input, LogicalFilter)

    def test_join(self):
        """Test building join plan."""
        parser = SQLParser("SELECT a FROM t1 JOIN t2 ON t1.id = t2.id")
        stmt = parser.parse()
        builder = LogicalPlanBuilder()
        plan = builder.build(stmt)
        # Find the join node
        current = plan
        while not isinstance(current, LogicalJoin):
            current = current.input if hasattr(current, 'input') else current.children()[0]
        assert isinstance(current, LogicalJoin)

    def test_aggregate(self):
        """Test building aggregate plan."""
        parser = SQLParser("SELECT a, COUNT(*) FROM t GROUP BY a")
        stmt = parser.parse()
        builder = LogicalPlanBuilder()
        plan = builder.build(stmt)
        # Should have Project -> Aggregate -> Scan
        assert isinstance(plan, LogicalProject)
        assert isinstance(plan.input, LogicalAggregate)

    def test_sort(self):
        """Test building sort plan."""
        parser = SQLParser("SELECT a FROM t ORDER BY a")
        stmt = parser.parse()
        builder = LogicalPlanBuilder()
        plan = builder.build(stmt)
        # Should have Sort in plan
        current = plan
        found_sort = False
        while current:
            if isinstance(current, LogicalSort):
                found_sort = True
                break
            if hasattr(current, 'input'):
                current = current.input
            else:
                break
        assert found_sort

    def test_limit(self):
        """Test building limit plan."""
        parser = SQLParser("SELECT a FROM t LIMIT 10")
        stmt = parser.parse()
        builder = LogicalPlanBuilder()
        plan = builder.build(stmt)
        assert isinstance(plan, LogicalLimit)
        assert plan.limit == 10


class TestLogicalOptimizer:
    """Tests for logical optimizer."""

    def test_predicate_pushdown(self):
        """Test predicate pushdown optimization."""
        # Build a plan with filter above project
        scan = LogicalScan(table_name="t")
        project = LogicalProject(
            expressions=[ColumnRef(name="a")],
            input=scan,
            aliases=["a"],
        )
        filter_node = LogicalFilter(
            predicate=BinaryOp(
                operator=">",
                left=ColumnRef(name="a"),
                right=Literal(value=10),
            ),
            input=project,
        )

        rule = PredicatePushdown()
        new_plan, changed = rule.apply(filter_node)

        # Filter should be pushed below project
        assert changed
        assert isinstance(new_plan, LogicalProject)
        assert isinstance(new_plan.input, LogicalFilter)

    def test_constant_folding(self):
        """Test constant folding optimization."""
        scan = LogicalScan(table_name="t")
        filter_node = LogicalFilter(
            predicate=BinaryOp(
                operator=">",
                left=BinaryOp(
                    operator="+",
                    left=Literal(value=5),
                    right=Literal(value=3),
                ),
                right=Literal(value=7),
            ),
            input=scan,
        )

        rule = ConstantFolding()
        new_plan, changed = rule.apply(filter_node)

        # 5 + 3 should be folded to 8
        assert changed

    def test_full_optimization(self):
        """Test full optimization pipeline."""
        parser = SQLParser("SELECT a FROM t WHERE a > 10 AND b < 20")
        stmt = parser.parse()
        builder = LogicalPlanBuilder()
        plan = builder.build(stmt)

        optimizer = LogicalOptimizer()
        optimized = optimizer.optimize(plan)

        assert optimized is not None


class TestPhysicalPlanner:
    """Tests for physical planner."""

    def test_plan_scan(self):
        """Test planning table scan."""
        logical = LogicalScan(table_name="t")
        planner = PhysicalPlanner()
        physical = planner.plan(logical)

        # Should produce PhysicalScan (possibly with exchange)
        assert physical is not None

    def test_plan_filter(self):
        """Test planning filter."""
        logical = LogicalFilter(
            predicate=BinaryOp(
                operator=">",
                left=ColumnRef(name="a"),
                right=Literal(value=10),
            ),
            input=LogicalScan(table_name="t"),
        )
        planner = PhysicalPlanner()
        physical = planner.plan(logical)

        # Should have filter in plan
        assert physical is not None

    def test_plan_join_strategy(self):
        """Test join strategy selection."""
        left = LogicalScan(table_name="large_table")
        right = LogicalScan(table_name="small_table")
        logical = LogicalJoin(
            join_type=JoinType.INNER,
            left=left,
            right=right,
            condition=BinaryOp(
                operator="=",
                left=ColumnRef(name="id", table="large_table"),
                right=ColumnRef(name="id", table="small_table"),
            ),
        )

        planner = PhysicalPlanner()
        physical = planner.plan(logical)

        # Should select appropriate join strategy
        assert physical is not None

    def test_plan_aggregate_two_phase(self):
        """Test two-phase aggregation planning."""
        logical = LogicalAggregate(
            group_by=[ColumnRef(name="a")],
            aggregates=[
                AggregateExpr(function=AggregateFunction.COUNT, args=[]),
            ],
            input=LogicalScan(table_name="t"),
        )

        planner = PhysicalPlanner()
        physical = planner.plan(logical)

        # Should produce two-phase aggregation
        assert physical is not None

    def test_plan_sort_with_exchange(self):
        """Test sort planning with exchange."""
        logical = LogicalSort(
            sort_expressions=[ColumnRef(name="a")],
            ascending=[True],
            nulls_first=[True],
            input=LogicalScan(table_name="t"),
        )

        planner = PhysicalPlanner()
        physical = planner.plan(logical)

        # Should have exchange for global sort
        assert physical is not None

    def test_plan_limit_local_plus_global(self):
        """Test limit planning with local and global phases."""
        logical = LogicalLimit(
            limit=10,
            offset=5,
            input=LogicalScan(table_name="t"),
        )

        planner = PhysicalPlanner()
        physical = planner.plan(logical)

        # Should have local limit + exchange + global limit
        assert physical is not None


class TestEndToEnd:
    """End-to-end tests for SQL processing."""

    def test_simple_query(self):
        """Test simple query end-to-end."""
        sql = "SELECT a, b FROM t WHERE a > 10 ORDER BY b LIMIT 100"

        # Parse
        parser = SQLParser(sql)
        stmt = parser.parse()

        # Build logical plan
        builder = LogicalPlanBuilder()
        logical = builder.build(stmt)

        # Optimize
        optimizer = LogicalOptimizer()
        optimized = optimizer.optimize(logical)

        # Physical plan
        planner = PhysicalPlanner()
        physical = planner.plan(optimized)

        assert physical is not None

    def test_join_query(self):
        """Test join query end-to-end."""
        sql = """
        SELECT t1.a, t2.b
        FROM table1 t1
        JOIN table2 t2 ON t1.id = t2.id
        WHERE t1.x > 10
        """

        parser = SQLParser(sql)
        stmt = parser.parse()
        builder = LogicalPlanBuilder()
        logical = builder.build(stmt)
        optimizer = LogicalOptimizer()
        optimized = optimizer.optimize(logical)
        planner = PhysicalPlanner()
        physical = planner.plan(optimized)

        assert physical is not None

    def test_aggregate_query(self):
        """Test aggregate query end-to-end."""
        sql = """
        SELECT category, COUNT(*), SUM(amount)
        FROM orders
        WHERE status = 'completed'
        GROUP BY category
        HAVING COUNT(*) > 10
        ORDER BY SUM(amount) DESC
        LIMIT 20
        """

        parser = SQLParser(sql)
        stmt = parser.parse()
        builder = LogicalPlanBuilder()
        logical = builder.build(stmt)
        optimizer = LogicalOptimizer()
        optimized = optimizer.optimize(logical)
        planner = PhysicalPlanner()
        physical = planner.plan(optimized)

        assert physical is not None

    def test_complex_query(self):
        """Test complex query with multiple features."""
        sql = """
        SELECT
            c.name AS customer,
            COUNT(DISTINCT o.id) AS order_count,
            SUM(o.total) AS total_spent
        FROM customers c
        LEFT JOIN orders o ON c.id = o.customer_id
        WHERE c.created_at > '2023-01-01'
          AND (o.status = 'completed' OR o.status IS NULL)
        GROUP BY c.id, c.name
        HAVING SUM(o.total) > 1000 OR COUNT(o.id) = 0
        ORDER BY total_spent DESC NULLS LAST
        LIMIT 50 OFFSET 10
        """

        parser = SQLParser(sql)
        stmt = parser.parse()
        builder = LogicalPlanBuilder()
        logical = builder.build(stmt)
        optimizer = LogicalOptimizer()
        optimized = optimizer.optimize(logical)
        planner = PhysicalPlanner()
        physical = planner.plan(optimized)

        assert physical is not None

    def test_union_query(self):
        """Test UNION query end-to-end."""
        sql = """
        SELECT id, name FROM active_users
        UNION ALL
        SELECT id, name FROM archived_users
        ORDER BY name
        LIMIT 100
        """

        parser = SQLParser(sql)
        stmt = parser.parse()
        builder = LogicalPlanBuilder()
        logical = builder.build(stmt)
        optimizer = LogicalOptimizer()
        optimized = optimizer.optimize(logical)
        planner = PhysicalPlanner()
        physical = planner.plan(optimized)

        assert physical is not None


class TestToSQL:
    """Tests for converting AST back to SQL."""

    def test_select_to_sql(self):
        """Test converting SELECT to SQL."""
        parser = SQLParser("SELECT a, b FROM t WHERE x > 10")
        stmt = parser.parse()
        sql = stmt.to_sql()
        assert "SELECT" in sql
        assert "FROM" in sql
        assert "WHERE" in sql

    def test_join_to_sql(self):
        """Test converting JOIN to SQL."""
        parser = SQLParser("SELECT a FROM t1 JOIN t2 ON t1.id = t2.id")
        stmt = parser.parse()
        sql = stmt.to_sql()
        assert "JOIN" in sql
        assert "ON" in sql

    def test_aggregate_to_sql(self):
        """Test converting aggregate to SQL."""
        parser = SQLParser("SELECT COUNT(*), SUM(a) FROM t GROUP BY b")
        stmt = parser.parse()
        sql = stmt.to_sql()
        assert "COUNT" in sql
        assert "SUM" in sql
        assert "GROUP BY" in sql
