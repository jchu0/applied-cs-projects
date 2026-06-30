"""Tests for the cost-based query optimizer.

Tests cost model, statistics estimation, DP join optimization,
and cost-based operator selection.
"""

import pytest
import math

from streamanalytics.sql.cost_model import (
    CostConstants,
    ColumnStatistics,
    TableStatistics,
    StatisticsEstimator,
    OperatorCost,
    CostModel,
    JoinNode,
    JoinEdge,
    DPJoinOptimizer,
    CostBasedJoinReordering,
    CostBasedPhysicalPlanner,
    CostBasedOptimizer,
)
from streamanalytics.sql.logical import (
    LogicalPlan,
    LogicalScan,
    LogicalFilter,
    LogicalProject,
    LogicalJoin,
    LogicalAggregate,
    LogicalSort,
    LogicalLimit,
    LogicalDistinct,
    Schema,
    Statistics,
)
from streamanalytics.sql.ast import (
    ColumnRef,
    Literal,
    BinaryOp,
    AggregateExpr,
    AggregateFunction,
    DataType,
    JoinType,
)


# ============================================================================
# Cost Constants Tests
# ============================================================================

class TestCostConstants:
    """Tests for CostConstants."""

    def test_default_values(self):
        """Test default cost constant values."""
        constants = CostConstants()

        assert constants.cpu_tuple_cost == 0.01
        assert constants.seq_page_cost == 1.0
        assert constants.random_page_cost == 4.0
        assert constants.network_transfer_cost == 0.1
        assert constants.default_row_count == 1000
        assert constants.page_size == 8192

    def test_custom_values(self):
        """Test custom cost constant values."""
        constants = CostConstants(
            cpu_tuple_cost=0.02,
            seq_page_cost=0.5,
            default_row_count=5000,
        )

        assert constants.cpu_tuple_cost == 0.02
        assert constants.seq_page_cost == 0.5
        assert constants.default_row_count == 5000


# ============================================================================
# Statistics Estimator Tests
# ============================================================================

class TestStatisticsEstimator:
    """Tests for StatisticsEstimator."""

    @pytest.fixture
    def estimator(self):
        """Create a statistics estimator."""
        est = StatisticsEstimator()

        # Register test tables
        est.register_table("users", TableStatistics(
            row_count=100000,
            row_width_bytes=200,
            column_stats={
                "id": ColumnStatistics(distinct_count=100000),
                "status": ColumnStatistics(distinct_count=5),
                "age": ColumnStatistics(distinct_count=80, min_value=18, max_value=100),
            }
        ))

        est.register_table("orders", TableStatistics(
            row_count=1000000,
            row_width_bytes=150,
            column_stats={
                "id": ColumnStatistics(distinct_count=1000000),
                "user_id": ColumnStatistics(distinct_count=50000),
                "amount": ColumnStatistics(distinct_count=10000, min_value=1, max_value=10000),
            }
        ))

        return est

    def test_estimate_scan_with_stats(self, estimator):
        """Test scan estimation with registered stats."""
        scan = LogicalScan(table_name="users")
        stats = estimator.estimate_scan(scan)

        assert stats.row_count == 100000
        assert stats.size_bytes == 100000 * 200

    def test_estimate_scan_without_stats(self, estimator):
        """Test scan estimation without registered stats."""
        scan = LogicalScan(table_name="unknown_table")
        stats = estimator.estimate_scan(scan)

        # Uses defaults
        assert stats.row_count == 1000
        assert stats.size_bytes == 1000 * 100

    def test_estimate_scan_with_projection(self, estimator):
        """Test scan estimation with column projection."""
        scan = LogicalScan(
            table_name="users",
            schema=Schema(columns=[
                ("id", DataType.INTEGER),
                ("name", DataType.VARCHAR),
                ("email", DataType.VARCHAR),
                ("status", DataType.VARCHAR),
            ]),
            projection=[0, 1]  # Only 2 of 4 columns
        )
        stats = estimator.estimate_scan(scan)

        assert stats.row_count == 100000
        # Size should be reduced
        assert stats.size_bytes == int(100000 * 200 * 0.5)

    def test_estimate_selectivity_equality(self, estimator):
        """Test selectivity estimation for equality predicates."""
        # With column stats
        pred = BinaryOp(
            operator="=",
            left=ColumnRef(name="status", table="users"),
            right=Literal(value="active", data_type=DataType.VARCHAR)
        )
        sel = estimator.estimate_selectivity(pred)
        assert sel == 1.0 / 5  # 5 distinct values

    def test_estimate_selectivity_range(self, estimator):
        """Test selectivity estimation for range predicates."""
        # age > 50 with min=18, max=100
        pred = BinaryOp(
            operator=">",
            left=ColumnRef(name="age", table="users"),
            right=Literal(value=50, data_type=DataType.INTEGER)
        )
        sel = estimator.estimate_selectivity(pred)

        # Position = (50-18)/(100-18) = 32/82 ≈ 0.39
        # > means 1 - position
        expected = 1.0 - (50 - 18) / (100 - 18)
        assert abs(sel - expected) < 0.01

    def test_estimate_selectivity_and(self, estimator):
        """Test selectivity for AND conditions."""
        pred = BinaryOp(
            operator="AND",
            left=BinaryOp(
                operator="=",
                left=ColumnRef(name="status", table="users"),
                right=Literal(value="active", data_type=DataType.VARCHAR)
            ),
            right=BinaryOp(
                operator=">",
                left=ColumnRef(name="age", table="users"),
                right=Literal(value=50, data_type=DataType.INTEGER)
            )
        )
        sel = estimator.estimate_selectivity(pred)

        # Independence assumption: sel1 * sel2
        expected = (1.0 / 5) * (1.0 - (50 - 18) / (100 - 18))
        assert abs(sel - expected) < 0.01

    def test_estimate_selectivity_or(self, estimator):
        """Test selectivity for OR conditions."""
        pred = BinaryOp(
            operator="OR",
            left=Literal(value=True, data_type=DataType.BOOLEAN),
            right=Literal(value=False, data_type=DataType.BOOLEAN)
        )
        sel = estimator.estimate_selectivity(pred)

        # P(A OR B) = P(A) + P(B) - P(A)P(B) = 1 + 0 - 0 = 1
        assert sel == 1.0

    def test_estimate_filter(self, estimator):
        """Test filter statistics estimation."""
        input_stats = Statistics(row_count=100000, size_bytes=20000000)

        filter_node = LogicalFilter(
            predicate=BinaryOp(
                operator="=",
                left=ColumnRef(name="status", table="users"),
                right=Literal(value="active", data_type=DataType.VARCHAR)
            ),
            input=LogicalScan(table_name="users"),
        )

        output_stats = estimator.estimate_filter(filter_node, input_stats)

        # Expected 1/5 of rows
        expected_rows = 100000 // 5
        assert output_stats.row_count == expected_rows

    def test_estimate_join_inner(self, estimator):
        """Test inner join cardinality estimation."""
        left_stats = Statistics(row_count=100000, size_bytes=20000000)
        right_stats = Statistics(row_count=1000000, size_bytes=150000000)

        output_stats = estimator.estimate_join(
            left_stats, right_stats, "INNER", None
        )

        # Without condition, assume some selectivity
        assert output_stats.row_count > 0
        assert output_stats.size_bytes > 0

    def test_estimate_join_left(self, estimator):
        """Test left join cardinality estimation."""
        left_stats = Statistics(row_count=100000, size_bytes=20000000)
        right_stats = Statistics(row_count=1000000, size_bytes=150000000)

        output_stats = estimator.estimate_join(
            left_stats, right_stats, "LEFT", None
        )

        # Left join should preserve at least left rows
        assert output_stats.row_count >= left_stats.row_count

    def test_estimate_aggregate_global(self, estimator):
        """Test global aggregation estimation."""
        input_stats = Statistics(row_count=100000, size_bytes=20000000)

        output_stats = estimator.estimate_aggregate(input_stats, group_by=[])

        # Global aggregation produces 1 row
        assert output_stats.row_count == 1

    def test_estimate_aggregate_grouped(self, estimator):
        """Test grouped aggregation estimation."""
        input_stats = Statistics(row_count=100000, size_bytes=20000000)

        group_by = [ColumnRef(name="status", table="users")]
        output_stats = estimator.estimate_aggregate(input_stats, group_by)

        # Should have fewer rows than input
        assert output_stats.row_count < input_stats.row_count

    def test_estimate_limit(self, estimator):
        """Test limit estimation."""
        input_stats = Statistics(row_count=100000, size_bytes=20000000)

        output_stats = estimator.estimate_limit(input_stats, limit=100, offset=10)

        assert output_stats.row_count == 100


# ============================================================================
# Cost Model Tests
# ============================================================================

class TestCostModel:
    """Tests for CostModel."""

    @pytest.fixture
    def cost_model(self):
        """Create a cost model with test data."""
        stats_estimator = StatisticsEstimator()
        stats_estimator.register_table("users", TableStatistics(
            row_count=100000,
            row_width_bytes=200,
        ))
        stats_estimator.register_table("orders", TableStatistics(
            row_count=1000000,
            row_width_bytes=150,
        ))
        return CostModel(stats_estimator=stats_estimator)

    def test_operator_cost_addition(self):
        """Test OperatorCost addition."""
        cost1 = OperatorCost(cpu_cost=10, io_cost=20, network_cost=5)
        cost2 = OperatorCost(cpu_cost=5, io_cost=10, network_cost=3)

        total = cost1 + cost2

        assert total.cpu_cost == 15
        assert total.io_cost == 30
        assert total.network_cost == 8
        assert total.total_cost == 53

    def test_scan_cost(self, cost_model):
        """Test cost estimation for table scan."""
        scan = LogicalScan(table_name="users")
        cost, stats = cost_model.estimate_plan_cost(scan)

        assert cost > 0
        assert stats.row_count == 100000

    def test_filter_cost(self, cost_model):
        """Test cost estimation for filter."""
        plan = LogicalFilter(
            predicate=BinaryOp(
                operator="=",
                left=ColumnRef(name="id"),
                right=Literal(value=1, data_type=DataType.INTEGER)
            ),
            input=LogicalScan(table_name="users")
        )

        cost, stats = cost_model.estimate_plan_cost(plan)

        assert cost > 0
        # Filter should reduce rows
        assert stats.row_count < 100000

    def test_join_cost(self, cost_model):
        """Test cost estimation for join."""
        plan = LogicalJoin(
            left=LogicalScan(table_name="users"),
            right=LogicalScan(table_name="orders"),
        )

        cost, stats = cost_model.estimate_plan_cost(plan)

        assert cost > 0
        assert stats.row_count > 0

    def test_aggregate_cost(self, cost_model):
        """Test cost estimation for aggregation."""
        plan = LogicalAggregate(
            group_by=[ColumnRef(name="user_id")],
            aggregates=[AggregateExpr(function=AggregateFunction.COUNT, args=[Literal(value=1)])],
            input=LogicalScan(table_name="orders")
        )

        cost, stats = cost_model.estimate_plan_cost(plan)

        assert cost > 0
        # Aggregation should reduce rows
        assert stats.row_count < 1000000

    def test_sort_cost(self, cost_model):
        """Test cost estimation for sorting."""
        plan = LogicalSort(
            sort_expressions=[ColumnRef(name="id")],
            ascending=[True],
            nulls_first=[True],
            input=LogicalScan(table_name="users")
        )

        cost, stats = cost_model.estimate_plan_cost(plan)

        # Sort cost should include O(n log n) component
        assert cost > 0
        # Row count unchanged
        assert stats.row_count == 100000

    def test_complex_plan_cost(self, cost_model):
        """Test cost estimation for complex query."""
        # SELECT users.id, COUNT(orders.id)
        # FROM users JOIN orders ON users.id = orders.user_id
        # WHERE users.status = 'active'
        # GROUP BY users.id
        # ORDER BY COUNT(orders.id) DESC
        # LIMIT 10

        plan = LogicalLimit(
            limit=10,
            input=LogicalSort(
                sort_expressions=[ColumnRef(name="count")],
                ascending=[False],
                nulls_first=[False],
                input=LogicalAggregate(
                    group_by=[ColumnRef(name="id", table="users")],
                    aggregates=[AggregateExpr(function=AggregateFunction.COUNT, args=[ColumnRef(name="id", table="orders")])],
                    input=LogicalJoin(
                        left=LogicalFilter(
                            predicate=BinaryOp(
                                operator="=",
                                left=ColumnRef(name="status", table="users"),
                                right=Literal(value="active", data_type=DataType.VARCHAR)
                            ),
                            input=LogicalScan(table_name="users"),
                        ),
                        right=LogicalScan(table_name="orders"),
                        condition=BinaryOp(
                            operator="=",
                            left=ColumnRef(name="id", table="users"),
                            right=ColumnRef(name="user_id", table="orders")
                        )
                    )
                )
            )
        )

        cost, stats = cost_model.estimate_plan_cost(plan)

        assert cost > 0
        assert stats.row_count == 10  # Limited to 10 rows


# ============================================================================
# DP Join Optimizer Tests
# ============================================================================

class TestDPJoinOptimizer:
    """Tests for DPJoinOptimizer."""

    @pytest.fixture
    def optimizer(self):
        """Create a DP join optimizer."""
        stats_estimator = StatisticsEstimator()
        stats_estimator.register_table("small", TableStatistics(row_count=100))
        stats_estimator.register_table("medium", TableStatistics(row_count=10000))
        stats_estimator.register_table("large", TableStatistics(row_count=1000000))

        cost_model = CostModel(stats_estimator=stats_estimator)
        return DPJoinOptimizer(cost_model)

    def test_two_table_join(self, optimizer):
        """Test join ordering with two tables."""
        relations = [
            LogicalScan(table_name="small"),
            LogicalScan(table_name="large"),
        ]
        conditions = [
            BinaryOp(
                operator="=",
                left=ColumnRef(name="id", table="small"),
                right=ColumnRef(name="small_id", table="large")
            )
        ]

        result = optimizer.optimize_join_order(relations, conditions)

        assert isinstance(result, LogicalJoin)

    def test_three_table_join(self, optimizer):
        """Test join ordering with three tables."""
        relations = [
            LogicalScan(table_name="small"),
            LogicalScan(table_name="medium"),
            LogicalScan(table_name="large"),
        ]
        conditions = [
            BinaryOp(
                operator="=",
                left=ColumnRef(name="id", table="small"),
                right=ColumnRef(name="small_id", table="medium")
            ),
            BinaryOp(
                operator="=",
                left=ColumnRef(name="id", table="medium"),
                right=ColumnRef(name="medium_id", table="large")
            ),
        ]

        result = optimizer.optimize_join_order(relations, conditions)

        # Should create nested joins
        assert isinstance(result, LogicalJoin)

        # Verify all tables are included
        def get_tables(plan):
            tables = set()
            if isinstance(plan, LogicalScan):
                tables.add(plan.table_name)
            for child in plan.children():
                tables.update(get_tables(child))
            return tables

        tables = get_tables(result)
        assert tables == {"small", "medium", "large"}

    def test_single_relation(self, optimizer):
        """Test with single relation (no join)."""
        relations = [LogicalScan(table_name="small")]
        conditions = []

        result = optimizer.optimize_join_order(relations, conditions)

        assert isinstance(result, LogicalScan)

    def test_subsets_generation(self, optimizer):
        """Test subset generation helper."""
        s = frozenset(["a", "b", "c"])

        subsets_1 = optimizer._subsets_of_size(s, 1)
        assert len(subsets_1) == 3

        subsets_2 = optimizer._subsets_of_size(s, 2)
        assert len(subsets_2) == 3

        subsets_3 = optimizer._subsets_of_size(s, 3)
        assert len(subsets_3) == 1


# ============================================================================
# Cost-Based Join Reordering Rule Tests
# ============================================================================

class TestCostBasedJoinReordering:
    """Tests for CostBasedJoinReordering optimization rule."""

    @pytest.fixture
    def rule(self):
        """Create the optimization rule."""
        stats_estimator = StatisticsEstimator()
        stats_estimator.register_table("a", TableStatistics(row_count=100))
        stats_estimator.register_table("b", TableStatistics(row_count=1000))
        stats_estimator.register_table("c", TableStatistics(row_count=10000))

        cost_model = CostModel(stats_estimator=stats_estimator)
        return CostBasedJoinReordering(cost_model)

    def test_rule_name(self, rule):
        """Test rule name."""
        assert rule.name == "CostBasedJoinReordering"

    def test_apply_simple_join(self, rule):
        """Test applying rule to simple join."""
        plan = LogicalJoin(
            left=LogicalScan(table_name="a"),
            right=LogicalScan(table_name="b"),
        )

        result, changed = rule.apply(plan)

        # May or may not change depending on cost
        assert isinstance(result, LogicalJoin)

    def test_apply_nested_joins(self, rule):
        """Test applying rule to nested joins."""
        # (a JOIN b) JOIN c
        plan = LogicalJoin(
            left=LogicalJoin(
                left=LogicalScan(table_name="a"),
                right=LogicalScan(table_name="b"),
            ),
            right=LogicalScan(table_name="c"),
        )

        result, changed = rule.apply(plan)

        assert isinstance(result, (LogicalJoin, LogicalScan))


# ============================================================================
# Cost-Based Physical Planner Tests
# ============================================================================

class TestCostBasedPhysicalPlanner:
    """Tests for CostBasedPhysicalPlanner."""

    @pytest.fixture
    def planner(self):
        """Create the physical planner."""
        return CostBasedPhysicalPlanner()

    def test_select_join_strategy_broadcast(self, planner):
        """Test join strategy selection for broadcast join."""
        small_stats = Statistics(row_count=1000, size_bytes=100000)  # 100KB
        large_stats = Statistics(row_count=10000000, size_bytes=1000000000)  # 1GB

        strategy, build_side, cost = planner.select_join_strategy(small_stats, large_stats)

        # Should choose broadcast for small table
        assert strategy == "broadcast"
        assert build_side == "left"  # Small side
        assert cost > 0

    def test_select_join_strategy_hash(self, planner):
        """Test join strategy selection for hash join."""
        # Both tables large
        stats1 = Statistics(row_count=10000000, size_bytes=1000000000)
        stats2 = Statistics(row_count=10000000, size_bytes=1000000000)

        strategy, build_side, cost = planner.select_join_strategy(stats1, stats2)

        # Should choose hash join for large tables
        assert strategy in ["hash", "merge"]
        assert cost > 0

    def test_select_aggregate_strategy_global(self, planner):
        """Test aggregate strategy for global aggregation."""
        input_stats = Statistics(row_count=1000000)

        strategy, cost = planner.select_aggregate_strategy(
            input_stats, group_by=[], is_distributed=True
        )

        assert strategy == "two_phase_global"
        assert cost > 0

    def test_select_aggregate_strategy_grouped(self, planner):
        """Test aggregate strategy for grouped aggregation."""
        input_stats = Statistics(row_count=1000000)
        group_by = [ColumnRef(name="user_id")]

        strategy, cost = planner.select_aggregate_strategy(
            input_stats, group_by, is_distributed=True
        )

        assert strategy in ["single", "two_phase"]
        assert cost > 0

    def test_select_aggregate_strategy_local(self, planner):
        """Test aggregate strategy for local (non-distributed) execution."""
        input_stats = Statistics(row_count=1000000)

        strategy, cost = planner.select_aggregate_strategy(
            input_stats, group_by=[], is_distributed=False
        )

        assert strategy == "single"
        assert cost > 0


# ============================================================================
# Cost-Based Optimizer Tests
# ============================================================================

class TestCostBasedOptimizer:
    """Tests for the integrated CostBasedOptimizer."""

    @pytest.fixture
    def optimizer(self):
        """Create the optimizer."""
        optimizer = CostBasedOptimizer()
        optimizer.register_table_stats("users", TableStatistics(
            row_count=100000,
            row_width_bytes=200,
            column_stats={
                "id": ColumnStatistics(distinct_count=100000),
                "status": ColumnStatistics(distinct_count=5),
            }
        ))
        optimizer.register_table_stats("orders", TableStatistics(
            row_count=1000000,
            row_width_bytes=150,
            column_stats={
                "id": ColumnStatistics(distinct_count=1000000),
                "user_id": ColumnStatistics(distinct_count=50000),
            }
        ))
        return optimizer

    def test_optimize_simple_query(self, optimizer):
        """Test optimizing a simple query."""
        plan = LogicalFilter(
            predicate=BinaryOp(
                operator="=",
                left=ColumnRef(name="status"),
                right=Literal(value="active", data_type=DataType.VARCHAR)
            ),
            input=LogicalScan(table_name="users")
        )

        result = optimizer.optimize(plan)

        # Should still be a filter (predicate pushdown doesn't change this simple case)
        assert isinstance(result, (LogicalFilter, LogicalScan))

    def test_optimize_join_query(self, optimizer):
        """Test optimizing a join query."""
        plan = LogicalJoin(
            left=LogicalScan(table_name="users"),
            right=LogicalScan(table_name="orders"),
            condition=BinaryOp(
                operator="=",
                left=ColumnRef(name="id", table="users"),
                right=ColumnRef(name="user_id", table="orders")
            )
        )

        result = optimizer.optimize(plan)

        # Should have statistics propagated
        assert result.stats is not None

    def test_explain_output(self, optimizer):
        """Test explain output generation."""
        plan = LogicalJoin(
            left=LogicalScan(table_name="users"),
            right=LogicalScan(table_name="orders"),
        )

        explain = optimizer.explain(plan)

        assert "Cost-Based Optimizer Explain" in explain
        assert "Original Plan" in explain
        assert "Optimized Plan" in explain
        assert "Cost Estimates" in explain

    def test_statistics_propagation(self, optimizer):
        """Test that statistics are propagated through the plan."""
        plan = LogicalFilter(
            predicate=BinaryOp(
                operator="=",
                left=ColumnRef(name="id"),
                right=Literal(value=1, data_type=DataType.INTEGER)
            ),
            input=LogicalScan(table_name="users")
        )

        result = optimizer.optimize(plan)

        # Statistics should be set on the result
        assert result.stats is not None
        assert result.stats.row_count is not None


# ============================================================================
# Integration Tests
# ============================================================================

class TestCostModelIntegration:
    """Integration tests for the complete cost-based optimization pipeline."""

    def test_full_optimization_pipeline(self):
        """Test the full optimization pipeline."""
        # Create optimizer with stats
        optimizer = CostBasedOptimizer()
        optimizer.register_table_stats("customers", TableStatistics(
            row_count=10000,
            column_stats={"id": ColumnStatistics(distinct_count=10000)}
        ))
        optimizer.register_table_stats("orders", TableStatistics(
            row_count=100000,
            column_stats={"customer_id": ColumnStatistics(distinct_count=8000)}
        ))
        optimizer.register_table_stats("products", TableStatistics(
            row_count=1000,
            column_stats={"id": ColumnStatistics(distinct_count=1000)}
        ))

        # Build a complex query plan
        # SELECT c.name, SUM(o.amount)
        # FROM customers c
        # JOIN orders o ON c.id = o.customer_id
        # JOIN products p ON o.product_id = p.id
        # WHERE c.status = 'active'
        # GROUP BY c.name
        # ORDER BY SUM(o.amount) DESC
        # LIMIT 10

        plan = LogicalLimit(
            limit=10,
            input=LogicalSort(
                sort_expressions=[ColumnRef(name="sum_amount")],
                ascending=[False],
                nulls_first=[False],
                input=LogicalAggregate(
                    group_by=[ColumnRef(name="name", table="c")],
                    aggregates=[AggregateExpr(function=AggregateFunction.SUM, args=[ColumnRef(name="amount", table="o")])],
                    input=LogicalJoin(
                        left=LogicalJoin(
                            left=LogicalFilter(
                                predicate=BinaryOp(
                                    operator="=",
                                    left=ColumnRef(name="status", table="c"),
                                    right=Literal(value="active", data_type=DataType.VARCHAR)
                                ),
                                input=LogicalScan(table_name="customers", alias="c"),
                            ),
                            right=LogicalScan(table_name="orders", alias="o"),
                            condition=BinaryOp(
                                operator="=",
                                left=ColumnRef(name="id", table="c"),
                                right=ColumnRef(name="customer_id", table="o")
                            )
                        ),
                        right=LogicalScan(table_name="products", alias="p"),
                        condition=BinaryOp(
                            operator="=",
                            left=ColumnRef(name="product_id", table="o"),
                            right=ColumnRef(name="id", table="p")
                        )
                    )
                )
            )
        )

        # Optimize
        optimized = optimizer.optimize(plan)

        # Verify optimization worked
        assert optimized is not None
        assert optimized.stats is not None

        # Get costs
        original_cost, _ = optimizer.cost_model.estimate_plan_cost(plan)
        optimized_cost, _ = optimizer.cost_model.estimate_plan_cost(optimized)

        # Both should have positive costs
        assert original_cost > 0
        assert optimized_cost > 0

    def test_cost_improves_with_optimization(self):
        """Test that optimization generally improves (or maintains) cost."""
        optimizer = CostBasedOptimizer()
        optimizer.register_table_stats("t1", TableStatistics(row_count=100))
        optimizer.register_table_stats("t2", TableStatistics(row_count=10000))
        optimizer.register_table_stats("t3", TableStatistics(row_count=1000000))

        # Create a suboptimal join order: large with large first, then small
        # ((t2 JOIN t3) JOIN t1)
        plan = LogicalJoin(
            left=LogicalJoin(
                left=LogicalScan(table_name="t2"),
                right=LogicalScan(table_name="t3"),
            ),
            right=LogicalScan(table_name="t1"),
        )

        original_cost, _ = optimizer.cost_model.estimate_plan_cost(plan)
        optimized = optimizer.optimize(plan)
        optimized_cost, _ = optimizer.cost_model.estimate_plan_cost(optimized)

        # Optimized should not be significantly worse
        # (May be same or better depending on heuristics)
        assert optimized_cost <= original_cost * 1.1  # Allow 10% tolerance
