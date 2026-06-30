"""Tests for the optimization module."""

import pytest
import time
import importlib.util
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# Check if redis is available
REDIS_AVAILABLE = importlib.util.find_spec("redis") is not None

from semantic_layer.models import MetricQuery
from semantic_layer.optimization import (
    AggregateGranularity,
    AggregateTableConfig,
    AggregateTableManager,
    CacheEntry,
    CacheStats,
    CostEstimator,
    FilterPushdownRule,
    MaterializedViewAdvisor,
    MaterializedViewRecommendation,
    OptimizationResult,
    OptimizedQueryEngine,
    ProjectionPruningRule,
    QueryCache,
    QueryCost,
    QueryPlan,
    QueryPlanner,
    RedisCache,
    TableStats,
    TimeRangeOptimizationRule,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_query():
    """Create a sample metric query."""
    return MetricQuery(
        metrics=["total_revenue", "order_count"],
        dimensions=["country", "product_category"],
        time_grain="day",
        start_date="2024-01-01",
        end_date="2024-01-31",
        filters=[],
    )


@pytest.fixture
def aggregate_config():
    """Create a sample aggregate table config."""
    return AggregateTableConfig(
        name="daily_revenue",
        source_table="fct_orders",
        metrics=["total_revenue", "order_count"],
        dimensions=["country", "product_category"],
        time_column="order_date",
        granularity=AggregateGranularity.DAILY,
    )


@pytest.fixture
def aggregate_manager(aggregate_config):
    """Create aggregate manager with sample config."""
    manager = AggregateTableManager()
    manager.register_table(aggregate_config)
    return manager


@pytest.fixture
def cost_estimator():
    """Create cost estimator with sample stats."""
    estimator = CostEstimator()
    estimator.update_table_stats(TableStats(
        table_name="fct_orders",
        row_count=10_000_000,
        size_bytes=5_000_000_000,
    ))
    return estimator


# =============================================================================
# AggregateTableConfig Tests
# =============================================================================

class TestAggregateTableConfig:
    """Tests for AggregateTableConfig."""

    def test_creation(self):
        """Test creating aggregate table config."""
        config = AggregateTableConfig(
            name="test_agg",
            source_table="orders",
            metrics=["revenue"],
            dimensions=["country"],
            time_column="created_at",
            granularity=AggregateGranularity.DAILY,
        )
        assert config.name == "test_agg"
        assert config.source_table == "orders"
        assert config.granularity == AggregateGranularity.DAILY

    def test_qualified_name(self):
        """Test qualified name generation."""
        config = AggregateTableConfig(
            name="test_agg",
            source_table="orders",
            metrics=["revenue"],
            dimensions=["country"],
            time_column="created_at",
            granularity=AggregateGranularity.MONTHLY,
        )
        assert config.qualified_name == "agg_orders_monthly"

    def test_default_values(self):
        """Test default values."""
        config = AggregateTableConfig(
            name="test_agg",
            source_table="orders",
            metrics=["revenue"],
            dimensions=[],
            time_column="created_at",
            granularity=AggregateGranularity.DAILY,
        )
        assert config.refresh_schedule == "0 * * * *"
        assert config.retention_days == 365
        assert config.partitioned_by is None
        assert config.clustered_by == []


# =============================================================================
# AggregateTableManager Tests
# =============================================================================

class TestAggregateTableManager:
    """Tests for AggregateTableManager."""

    def test_register_table(self, aggregate_config):
        """Test registering an aggregate table."""
        manager = AggregateTableManager()
        manager.register_table(aggregate_config)
        assert manager.get_table("daily_revenue") is not None

    def test_get_nonexistent_table(self):
        """Test getting a non-existent table."""
        manager = AggregateTableManager()
        assert manager.get_table("nonexistent") is None

    def test_find_matching_table(self, aggregate_manager):
        """Test finding a matching aggregate table."""
        table = aggregate_manager.find_matching_table(
            source_table="fct_orders",
            metrics=["total_revenue"],
            dimensions=["country"],
            granularity=AggregateGranularity.DAILY,
        )
        assert table is not None
        assert table.name == "daily_revenue"

    def test_find_matching_table_wrong_source(self, aggregate_manager):
        """Test not finding table with wrong source."""
        table = aggregate_manager.find_matching_table(
            source_table="wrong_table",
            metrics=["total_revenue"],
            dimensions=["country"],
            granularity=AggregateGranularity.DAILY,
        )
        assert table is None

    def test_find_matching_table_missing_metric(self, aggregate_manager):
        """Test not finding table with missing metric."""
        table = aggregate_manager.find_matching_table(
            source_table="fct_orders",
            metrics=["missing_metric"],
            dimensions=["country"],
            granularity=AggregateGranularity.DAILY,
        )
        assert table is None

    def test_find_matching_table_coarser_granularity(self, aggregate_manager):
        """Test finding table with coarser granularity."""
        table = aggregate_manager.find_matching_table(
            source_table="fct_orders",
            metrics=["total_revenue"],
            dimensions=["country"],
            granularity=AggregateGranularity.WEEKLY,
        )
        # Daily table can satisfy weekly query
        assert table is not None

    def test_generate_create_ddl(self, aggregate_manager, aggregate_config):
        """Test DDL generation."""
        ddl = aggregate_manager.generate_create_ddl(aggregate_config)
        assert "CREATE TABLE" in ddl
        assert "agg_fct_orders_daily" in ddl
        assert "order_date TIMESTAMP" in ddl
        assert "total_revenue_sum" in ddl

    def test_generate_refresh_query(self, aggregate_manager, aggregate_config):
        """Test refresh query generation."""
        sql = aggregate_manager.generate_refresh_query(aggregate_config)
        assert "INSERT INTO" in sql
        assert "SUM(total_revenue)" in sql
        assert "GROUP BY" in sql

    def test_list_tables(self, aggregate_manager, aggregate_config):
        """Test listing tables."""
        tables = aggregate_manager.list_tables()
        assert len(tables) == 1
        assert tables[0].name == "daily_revenue"

    def test_recommend_aggregates(self):
        """Test aggregate recommendations."""
        manager = AggregateTableManager()
        patterns = [
            {
                "source_table": "orders",
                "dimensions": ["country", "product"],
                "metrics": ["revenue"],
                "time_column": "created_at",
            }
            for _ in range(15)
        ]
        recommendations = manager.recommend_aggregates(patterns, min_frequency=10)
        assert len(recommendations) == 1
        assert recommendations[0].source_table == "orders"


# =============================================================================
# CostEstimator Tests
# =============================================================================

class TestCostEstimator:
    """Tests for CostEstimator."""

    def test_update_table_stats(self):
        """Test updating table stats."""
        estimator = CostEstimator()
        stats = TableStats(
            table_name="test_table",
            row_count=1000000,
            size_bytes=100000000,
        )
        estimator.update_table_stats(stats)
        # No direct getter, but estimate should work

    def test_estimate_cost_basic(self, cost_estimator, sample_query):
        """Test basic cost estimation."""
        cost = cost_estimator.estimate_cost(sample_query, "fct_orders")
        assert cost.estimated_rows > 0
        assert cost.estimated_bytes > 0
        assert cost.estimated_time_ms > 0
        assert cost.io_cost > 0
        assert cost.cpu_cost > 0

    def test_estimate_cost_unknown_table(self, sample_query):
        """Test cost estimation for unknown table uses defaults."""
        estimator = CostEstimator()
        cost = estimator.estimate_cost(sample_query, "unknown_table")
        assert cost.estimated_rows > 0

    def test_total_cost(self, cost_estimator, sample_query):
        """Test total cost calculation."""
        cost = cost_estimator.estimate_cost(sample_query, "fct_orders")
        assert cost.total_cost == cost.io_cost + cost.cpu_cost + cost.network_cost

    def test_estimate_with_aggregate(self, cost_estimator, sample_query):
        """Test cost estimation with aggregate table."""
        # Add stats for aggregate table
        cost_estimator.update_table_stats(TableStats(
            table_name="agg_table",
            row_count=100000,  # Much smaller
            size_bytes=50000000,
        ))

        cost = cost_estimator.estimate_cost(
            sample_query,
            "fct_orders",
            use_aggregate=True,
            aggregate_table="agg_table",
        )
        # Should use aggregate table stats
        assert cost.estimated_rows <= 100000


# =============================================================================
# QueryPlanner Tests
# =============================================================================

class TestQueryPlanner:
    """Tests for QueryPlanner."""

    def test_plan_basic(self, aggregate_manager, cost_estimator, sample_query):
        """Test basic query planning."""
        planner = QueryPlanner(aggregate_manager, cost_estimator)
        plan = planner.plan(sample_query, "fct_orders")

        assert isinstance(plan, QueryPlan)
        assert plan.original_query == sample_query
        assert plan.rewritten_sql is not None
        assert plan.estimated_cost is not None

    def test_plan_uses_aggregate(self, aggregate_manager, cost_estimator, sample_query):
        """Test that planner uses aggregate table when beneficial."""
        # Add favorable aggregate table stats
        cost_estimator.update_table_stats(TableStats(
            table_name="agg_fct_orders_daily",
            row_count=1000,  # Very small
            size_bytes=100000,
        ))

        planner = QueryPlanner(aggregate_manager, cost_estimator)
        plan = planner.plan(sample_query, "fct_orders")

        assert plan.use_aggregate_table
        assert plan.aggregate_table_name is not None

    def test_plan_cache_key(self, aggregate_manager, cost_estimator, sample_query):
        """Test that plan has cache key."""
        planner = QueryPlanner(aggregate_manager, cost_estimator)
        plan = planner.plan(sample_query, "fct_orders")

        assert plan.cache_key is not None
        assert len(plan.cache_key) == 32

    def test_plan_cache_ttl(self, aggregate_manager, cost_estimator, sample_query):
        """Test that plan has cache TTL."""
        planner = QueryPlanner(aggregate_manager, cost_estimator)
        plan = planner.plan(sample_query, "fct_orders")

        assert plan.cache_ttl_seconds is not None
        assert plan.cache_ttl_seconds > 0

    def test_explain(self, aggregate_manager, cost_estimator, sample_query):
        """Test query plan explanation."""
        planner = QueryPlanner(aggregate_manager, cost_estimator)
        plan = planner.plan(sample_query, "fct_orders")
        explanation = planner.explain(plan)

        assert "Query Execution Plan" in explanation
        assert "Metrics:" in explanation
        assert "Dimensions:" in explanation
        assert "Generated SQL:" in explanation


# =============================================================================
# QueryCache Tests
# =============================================================================

class TestQueryCache:
    """Tests for QueryCache."""

    def test_set_and_get(self):
        """Test setting and getting values."""
        cache = QueryCache()
        cache.set("key1", {"data": [1, 2, 3]})
        result = cache.get("key1")
        assert result == {"data": [1, 2, 3]}

    def test_get_nonexistent(self):
        """Test getting non-existent key."""
        cache = QueryCache()
        result = cache.get("nonexistent")
        assert result is None

    def test_ttl_expiration(self):
        """Test that entries expire."""
        cache = QueryCache(default_ttl=1)
        cache.set("key1", "value1")

        # Should exist immediately
        assert cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(1.1)
        assert cache.get("key1") is None

    def test_custom_ttl(self):
        """Test custom TTL per entry."""
        cache = QueryCache()
        cache.set("key1", "value1", ttl_seconds=1)
        cache.set("key2", "value2", ttl_seconds=300)

        time.sleep(1.1)
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"

    def test_delete(self):
        """Test deleting entries."""
        cache = QueryCache()
        cache.set("key1", "value1")
        assert cache.delete("key1")
        assert cache.get("key1") is None

    def test_delete_nonexistent(self):
        """Test deleting non-existent entry."""
        cache = QueryCache()
        assert not cache.delete("nonexistent")

    def test_clear(self):
        """Test clearing cache."""
        cache = QueryCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_eviction(self):
        """Test LRU eviction."""
        cache = QueryCache(max_size=2)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Should trigger eviction

        # key1 should be evicted (LRU)
        assert cache.get("key1") is None
        assert cache.get("key2") is not None
        assert cache.get("key3") is not None

    def test_stats(self):
        """Test cache statistics."""
        cache = QueryCache()
        cache.set("key1", "value1")

        cache.get("key1")  # Hit
        cache.get("key2")  # Miss

        stats = cache.stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.size == 1

    def test_hit_rate(self):
        """Test hit rate calculation."""
        cache = QueryCache()
        cache.set("key1", "value1")

        for _ in range(4):
            cache.get("key1")  # 4 hits
        cache.get("key2")  # 1 miss

        stats = cache.stats()
        assert stats.hit_rate == 0.8


# =============================================================================
# CacheStats Tests
# =============================================================================

class TestCacheStats:
    """Tests for CacheStats."""

    def test_default_values(self):
        """Test default values."""
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.size == 0

    def test_hit_rate_empty(self):
        """Test hit rate with no requests."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_all_hits(self):
        """Test hit rate with all hits."""
        stats = CacheStats(hits=10, misses=0)
        assert stats.hit_rate == 1.0

    def test_hit_rate_mixed(self):
        """Test hit rate with mixed results."""
        stats = CacheStats(hits=75, misses=25)
        assert stats.hit_rate == 0.75


# =============================================================================
# CacheEntry Tests
# =============================================================================

class TestCacheEntry:
    """Tests for CacheEntry."""

    def test_is_expired_fresh(self):
        """Test fresh entry is not expired."""
        entry = CacheEntry(
            key="key1",
            value="value1",
            created_at=time.time(),
            ttl_seconds=300,
        )
        assert not entry.is_expired

    def test_is_expired_old(self):
        """Test old entry is expired."""
        entry = CacheEntry(
            key="key1",
            value="value1",
            created_at=time.time() - 400,
            ttl_seconds=300,
        )
        assert entry.is_expired


# =============================================================================
# OptimizationRule Tests
# =============================================================================

class TestTimeRangeOptimizationRule:
    """Tests for TimeRangeOptimizationRule."""

    def test_no_recommendation_short_range(self):
        """Test no recommendation for short time range."""
        rule = TimeRangeOptimizationRule()
        query = MetricQuery(
            metrics=["revenue"],
            dimensions=[],
            time_grain="day",
            start_date="2024-01-01",
            end_date="2024-01-31",
            filters=[],
        )
        result = rule.apply(query, "orders")
        assert "Consider using" not in result.note or result.note == ""

    def test_recommendation_long_range(self):
        """Test recommendation for long time range."""
        rule = TimeRangeOptimizationRule()
        query = MetricQuery(
            metrics=["revenue"],
            dimensions=[],
            time_grain="day",
            start_date="2023-01-01",
            end_date="2024-06-01",
            filters=[],
        )
        result = rule.apply(query, "orders")
        assert "Consider using" in result.note


class TestFilterPushdownRule:
    """Tests for FilterPushdownRule."""

    def test_with_filters(self):
        """Test pushdown note with filters."""
        rule = FilterPushdownRule()
        query = MetricQuery(
            metrics=["revenue"],
            dimensions=[],
            time_grain="day",
            start_date="2024-01-01",
            end_date="2024-01-31",
            filters=[{"field": "country", "operator": "=", "value": "'US'"}],
        )
        result = rule.apply(query, "orders")
        assert "Pushing down" in result.note

    def test_without_filters(self):
        """Test no note without filters."""
        rule = FilterPushdownRule()
        query = MetricQuery(
            metrics=["revenue"],
            dimensions=[],
            time_grain="day",
            start_date="2024-01-01",
            end_date="2024-01-31",
            filters=[],
        )
        result = rule.apply(query, "orders")
        assert result.modified is False


class TestProjectionPruningRule:
    """Tests for ProjectionPruningRule."""

    def test_counts_columns(self):
        """Test column counting."""
        rule = ProjectionPruningRule()
        query = MetricQuery(
            metrics=["revenue", "orders"],
            dimensions=["country", "product"],
            time_grain="day",
            start_date="2024-01-01",
            end_date="2024-01-31",
            filters=[],
        )
        result = rule.apply(query, "orders")
        assert "4 required columns" in result.note


# =============================================================================
# MaterializedViewAdvisor Tests
# =============================================================================

class TestMaterializedViewAdvisor:
    """Tests for MaterializedViewAdvisor."""

    def test_record_query(self, sample_query):
        """Test recording query execution."""
        advisor = MaterializedViewAdvisor()
        advisor.record_query(sample_query, 1500.0, 100000)
        # Should not raise

    def test_recommend_empty(self):
        """Test recommendations with no history."""
        advisor = MaterializedViewAdvisor()
        recommendations = advisor.analyze_and_recommend()
        assert recommendations == []

    def test_recommend_infrequent(self, sample_query):
        """Test no recommendations for infrequent queries."""
        advisor = MaterializedViewAdvisor()
        for _ in range(3):  # Less than threshold
            advisor.record_query(sample_query, 1500.0, 100000)
        recommendations = advisor.analyze_and_recommend(min_occurrences=5)
        assert recommendations == []

    def test_recommend_frequent_slow(self, sample_query):
        """Test recommendations for frequent, slow queries."""
        advisor = MaterializedViewAdvisor()
        for _ in range(10):
            advisor.record_query(sample_query, 2000.0, 100000)
        recommendations = advisor.analyze_and_recommend(min_occurrences=5, min_avg_time_ms=1000)
        assert len(recommendations) == 1
        assert isinstance(recommendations[0], MaterializedViewRecommendation)


# =============================================================================
# OptimizedQueryEngine Tests
# =============================================================================

class TestOptimizedQueryEngine:
    """Tests for OptimizedQueryEngine."""

    def test_creation(self):
        """Test engine creation."""
        engine = OptimizedQueryEngine()
        assert engine.warehouse_type == "snowflake"
        assert engine.use_cache

    def test_creation_without_cache(self):
        """Test engine creation without cache."""
        engine = OptimizedQueryEngine(use_cache=False)
        assert not engine.use_cache

    def test_plan_query(self, sample_query):
        """Test query planning."""
        engine = OptimizedQueryEngine()
        plan = engine.plan_query(sample_query, "fct_orders")
        assert isinstance(plan, QueryPlan)

    def test_explain(self, sample_query):
        """Test query explanation."""
        engine = OptimizedQueryEngine()
        explanation = engine.explain(sample_query, "fct_orders")
        assert "Query Execution Plan" in explanation

    def test_invalidate_cache(self):
        """Test cache invalidation."""
        engine = OptimizedQueryEngine()
        engine._cache.set("key1", "value1")
        engine.invalidate_cache()
        assert engine._cache.get("key1") is None


# =============================================================================
# RedisCache Tests
# =============================================================================

@pytest.mark.skipif(not REDIS_AVAILABLE, reason="redis not installed")
class TestRedisCache:
    """Tests for RedisCache."""

    def test_creation(self):
        """Test cache creation."""
        cache = RedisCache(host="localhost", port=6379)
        assert cache.prefix == "semantic:"
        assert cache.default_ttl == 300

    def test_key_prefix(self):
        """Test key prefixing."""
        cache = RedisCache(prefix="test:")
        assert cache._key("mykey") == "test:mykey"

    @patch('redis.Redis')
    def test_get_with_mock(self, mock_redis_class):
        """Test get with mocked Redis."""
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis.get.return_value = b'{"data": "value"}'
        mock_redis_class.return_value = mock_redis

        cache = RedisCache()
        result = cache.get("key1")
        assert result == {"data": "value"}

    @patch('redis.Redis')
    def test_set_with_mock(self, mock_redis_class):
        """Test set with mocked Redis."""
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis_class.return_value = mock_redis

        cache = RedisCache()
        cache.set("key1", {"data": "value"}, ttl_seconds=60)
        mock_redis.setex.assert_called_once()

    @patch('redis.Redis')
    def test_delete_with_mock(self, mock_redis_class):
        """Test delete with mocked Redis."""
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis.delete.return_value = 1
        mock_redis_class.return_value = mock_redis

        cache = RedisCache()
        result = cache.delete("key1")
        assert result is True


# =============================================================================
# QueryCost Tests
# =============================================================================

class TestQueryCost:
    """Tests for QueryCost."""

    def test_total_cost(self):
        """Test total cost calculation."""
        cost = QueryCost(
            estimated_rows=1000,
            estimated_bytes=10000,
            estimated_time_ms=100.0,
            io_cost=0.01,
            cpu_cost=0.005,
            network_cost=0.001,
        )
        assert cost.total_cost == pytest.approx(0.016)

    def test_total_cost_no_network(self):
        """Test total cost with no network cost."""
        cost = QueryCost(
            estimated_rows=1000,
            estimated_bytes=10000,
            estimated_time_ms=100.0,
            io_cost=0.01,
            cpu_cost=0.005,
        )
        assert cost.total_cost == pytest.approx(0.015)


# =============================================================================
# Integration Tests
# =============================================================================

class TestOptimizationIntegration:
    """Integration tests for optimization module."""

    def test_full_optimization_flow(self, sample_query):
        """Test full query optimization flow."""
        # Create engine
        engine = OptimizedQueryEngine(warehouse_type="snowflake")

        # Register aggregate table
        agg_config = AggregateTableConfig(
            name="daily_revenue",
            source_table="fct_orders",
            metrics=["total_revenue", "order_count"],
            dimensions=["country", "product_category"],
            time_column="order_date",
            granularity=AggregateGranularity.DAILY,
        )
        engine.aggregate_manager.register_table(agg_config)

        # Add table stats
        engine.cost_estimator.update_table_stats(TableStats(
            table_name="fct_orders",
            row_count=100_000_000,
            size_bytes=50_000_000_000,
        ))
        engine.cost_estimator.update_table_stats(TableStats(
            table_name="agg_fct_orders_daily",
            row_count=100_000,
            size_bytes=50_000_000,
        ))

        # Plan query
        plan = engine.plan_query(sample_query, "fct_orders")

        # Verify optimization
        assert plan.use_aggregate_table
        assert plan.aggregate_table_name == "agg_fct_orders_daily"
        assert len(plan.optimization_notes) > 0

    def test_caching_flow(self, sample_query):
        """Test caching flow."""
        engine = OptimizedQueryEngine(use_cache=True, cache_backend="memory")

        # Plan and cache
        plan = engine.plan_query(sample_query, "fct_orders")

        # Cache result
        engine._cache.set(plan.cache_key, [{"revenue": 1000}])

        # Verify cache hit
        cached = engine._cache.get(plan.cache_key)
        assert cached == [{"revenue": 1000}]

    def test_multiple_optimization_rules(self, sample_query):
        """Test that optimization rules can be applied."""
        engine = OptimizedQueryEngine()
        plan = engine.plan_query(sample_query, "fct_orders")

        # Should have a valid plan with or without optimization notes
        assert plan is not None
        assert plan.rewritten_sql is not None

    def test_warehouse_type_sql_generation(self):
        """Test SQL generation for different warehouses."""
        query = MetricQuery(
            metrics=["revenue"],
            dimensions=["country"],
            time_grain="day",
            start_date="2024-01-01",
            end_date="2024-01-31",
            filters=[],
        )

        for warehouse in ["snowflake", "bigquery", "postgres"]:
            engine = OptimizedQueryEngine(warehouse_type=warehouse)
            plan = engine.plan_query(query, "orders")
            assert "SELECT" in plan.rewritten_sql
            assert "GROUP BY" in plan.rewritten_sql
