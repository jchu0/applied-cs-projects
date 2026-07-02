"""Integration tests for the Semantic Layer."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from semantic_layer.models import (
    CalculationMethod,
    TimeGrain,
    MetricDefinition,
    MetricQuery,
    Dimension,
)
from semantic_layer.query_engine import (
    MetricCatalog,
    SemanticQueryEngine,
    QueryExecutor,
)
from semantic_layer.api import (
    SemanticLayerAPI,
    MetricRegistry,
    MetricQueryRequest,
)


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    @pytest.fixture
    def setup_metrics(self):
        """Set up a complete metric ecosystem."""
        registry = MetricRegistry()

        # Define base metrics
        revenue_metric = MetricDefinition(
            name="total_revenue",
            label="Total Revenue",
            description="Sum of all revenue",
            model="fact_orders",
            calculation_method=CalculationMethod.SUM,
            expression="order_amount",
            timestamp="order_date",
            time_grains=[TimeGrain.DAY, TimeGrain.WEEK, TimeGrain.MONTH],
            dimensions=["customer_segment", "product_category", "region"],
            meta={"owner": "finance-team", "tier": 1}
        )

        order_count_metric = MetricDefinition(
            name="order_count",
            label="Order Count",
            description="Number of orders",
            model="fact_orders",
            calculation_method=CalculationMethod.COUNT,
            expression="order_id",
            timestamp="order_date",
            time_grains=[TimeGrain.DAY, TimeGrain.WEEK, TimeGrain.MONTH],
            dimensions=["customer_segment", "product_category", "region"]
        )

        customer_count_metric = MetricDefinition(
            name="unique_customers",
            label="Unique Customers",
            description="Count of unique customers",
            model="fact_orders",
            calculation_method=CalculationMethod.COUNT_DISTINCT,
            expression="customer_id",
            timestamp="order_date",
            time_grains=[TimeGrain.DAY, TimeGrain.WEEK, TimeGrain.MONTH],
            dimensions=["customer_segment", "region"]
        )

        # Define derived metrics
        aov_metric = MetricDefinition(
            name="average_order_value",
            label="Average Order Value",
            description="Average revenue per order",
            model="fact_orders",
            calculation_method=CalculationMethod.DERIVED,
            expression="{{ metric('total_revenue') }} / {{ metric('order_count') }}",
            timestamp="order_date",
            time_grains=[TimeGrain.MONTH],
            dimensions=["customer_segment", "product_category"]
        )

        # Register all metrics
        for metric in [revenue_metric, order_count_metric, customer_count_metric, aov_metric]:
            registry.register_metric(metric)

        return registry

    def test_complete_workflow(self, setup_metrics):
        """Test complete workflow from registration to query."""
        registry = setup_metrics
        catalog = registry.get_catalog()
        engine = SemanticQueryEngine(catalog, warehouse_type="snowflake")
        api = SemanticLayerAPI(catalog, engine)

        # 1. List available metrics
        metrics = api.list_metrics()
        assert len(metrics) == 4
        assert any(m.name == "total_revenue" for m in metrics)
        assert any(m.name == "average_order_value" for m in metrics)

        # 2. Get specific metric details
        revenue_info = api.get_metric("total_revenue")
        assert revenue_info.label == "Total Revenue"
        assert "customer_segment" in revenue_info.dimensions

        # 3. Query single metric
        response = api.query_metric(
            metric_name="total_revenue",
            start_date="2024-01-01",
            end_date="2024-01-31",
            time_grain="week",
            dimensions=["customer_segment"]
        )

        assert response.metadata["metrics"] == ["total_revenue"]
        assert response.metadata["time_grain"] == "week"

        # 4. Query multiple metrics together
        response = api.query_metrics(
            metrics=["total_revenue", "order_count", "unique_customers"],
            start_date="2024-01-01",
            end_date="2024-03-31",
            time_grain="month",
            dimensions=["region", "customer_segment"],
            show_sql=True
        )

        assert len(response.metadata["metrics"]) == 3
        assert response.sql is not None

        # 5. Query derived metric
        response = api.query_metric(
            metric_name="average_order_value",
            start_date="2024-01-01",
            end_date="2024-12-31",
            time_grain="month",
            dimensions=["product_category"]
        )

        assert response.metadata["metrics"] == ["average_order_value"]

    def test_sql_generation_scenarios(self, setup_metrics):
        """Test various SQL generation scenarios."""
        registry = setup_metrics
        catalog = registry.get_catalog()
        engine = SemanticQueryEngine(catalog, warehouse_type="snowflake")

        # Scenario 1: Simple aggregation
        query = MetricQuery(
            metrics=["total_revenue"],
            dimensions=[],
            filters=[],
            time_grain="day",
            start_date="2024-01-01",
            end_date="2024-01-07"
        )

        sql = engine.generate_sql(query)
        assert "SUM(order_amount)" in sql
        assert "DATE_TRUNC('day'," in sql

        # Scenario 2: Multiple dimensions and filters
        query = MetricQuery(
            metrics=["unique_customers"],
            dimensions=["customer_segment", "region"],
            filters=[
                {"field": "region", "operator": "IN", "value": ["NA", "EU"]},
                {"field": "customer_segment", "operator": "!=", "value": "unknown"}
            ],
            time_grain="month",
            start_date="2024-01-01",
            end_date="2024-12-31"
        )

        sql = engine.generate_sql(query)
        assert "COUNT(DISTINCT customer_id)" in sql
        assert "customer_segment" in sql
        assert "region" in sql
        assert "region IN ('NA', 'EU')" in sql
        assert "customer_segment != 'unknown'" in sql

        # Scenario 3: Derived metric
        query = MetricQuery(
            metrics=["average_order_value"],
            dimensions=["product_category"],
            filters=[],
            time_grain="month",
            start_date="2024-01-01",
            end_date="2024-06-30"
        )

        sql = engine.generate_sql(query)
        # Should expand both referenced metrics
        assert "SUM(order_amount)" in sql
        assert "COUNT(order_id)" in sql

    @pytest.mark.asyncio
    async def test_async_query_execution(self, setup_metrics):
        """Test asynchronous query execution."""
        registry = setup_metrics
        catalog = registry.get_catalog()
        engine = SemanticQueryEngine(catalog)

        # Mock warehouse connection
        mock_connection = AsyncMock()
        executor = QueryExecutor(mock_connection)

        # Mock the execute method to return sample data
        sample_data = [
            {"period": "2024-01-01", "region": "NA", "revenue": 100000},
            {"period": "2024-01-01", "region": "EU", "revenue": 80000},
            {"period": "2024-01-02", "region": "NA", "revenue": 110000},
            {"period": "2024-01-02", "region": "EU", "revenue": 85000}
        ]

        with patch.object(executor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = sample_data

            query = MetricQuery(
                metrics=["total_revenue"],
                dimensions=["region"],
                filters=[],
                time_grain="day",
                start_date="2024-01-01",
                end_date="2024-01-02"
            )

            result = await executor.execute_metric_query(engine, query)

            assert result.row_count == 4
            assert len(result.data) == 4
            assert result.data[0]["revenue"] == 100000

    def test_metric_validation_and_errors(self, setup_metrics):
        """Test metric validation and error handling."""
        registry = setup_metrics
        catalog = registry.get_catalog()
        engine = SemanticQueryEngine(catalog)
        api = SemanticLayerAPI(catalog, engine)

        # Test invalid metric query
        with pytest.raises(ValueError, match="Query validation failed"):
            api.query_metric(
                metric_name="nonexistent_metric",
                start_date="2024-01-01",
                end_date="2024-01-31"
            )

        # Test invalid date range
        with pytest.raises(ValueError, match="start_date must be before end_date"):
            api.query_metric(
                metric_name="total_revenue",
                start_date="2024-12-31",
                end_date="2024-01-01"
            )

        # Test invalid time grain
        with pytest.raises(ValueError, match="Invalid time grain"):
            api.query_metrics(
                metrics=["total_revenue"],
                start_date="2024-01-01",
                end_date="2024-01-31",
                time_grain="invalid_grain"
            )

    def test_warehouse_compatibility(self, setup_metrics):
        """Test SQL generation for different warehouse types."""
        registry = setup_metrics
        catalog = registry.get_catalog()

        query = MetricQuery(
            metrics=["total_revenue"],
            dimensions=["region"],
            filters=[],
            time_grain="month",
            start_date="2024-01-01",
            end_date="2024-12-31"
        )

        # Test Snowflake
        snowflake_engine = SemanticQueryEngine(catalog, warehouse_type="snowflake")
        snowflake_sql = snowflake_engine.generate_sql(query)
        assert "DATE_TRUNC('month'," in snowflake_sql

        # Test BigQuery
        bigquery_engine = SemanticQueryEngine(catalog, warehouse_type="bigquery")
        bigquery_sql = bigquery_engine.generate_sql(query)
        assert "DATE_TRUNC(order_date, MONTH)" in bigquery_sql

        # Test PostgreSQL
        postgres_engine = SemanticQueryEngine(catalog, warehouse_type="postgres")
        postgres_sql = postgres_engine.generate_sql(query)
        assert "DATE_TRUNC('month'," in postgres_sql

    def test_complex_derived_metrics(self):
        """Test complex derived metric scenarios."""
        registry = MetricRegistry()

        # Define base metrics
        revenue = MetricDefinition(
            name="revenue",
            label="Revenue",
            description="Total revenue",
            model="sales",
            calculation_method=CalculationMethod.SUM,
            expression="amount",
            timestamp="date",
            time_grains=[TimeGrain.DAY]
        )

        costs = MetricDefinition(
            name="costs",
            label="Costs",
            description="Total costs",
            model="sales",
            calculation_method=CalculationMethod.SUM,
            expression="cost",
            timestamp="date",
            time_grains=[TimeGrain.DAY]
        )

        # Register base metrics
        registry.register_metric(revenue)
        registry.register_metric(costs)

        # Define derived metric
        profit_margin = MetricDefinition(
            name="profit_margin",
            label="Profit Margin",
            description="Profit margin percentage",
            model="sales",
            calculation_method=CalculationMethod.DERIVED,
            expression="({{ metric('revenue') }} - {{ metric('costs') }}) / {{ metric('revenue') }} * 100",
            timestamp="date",
            time_grains=[TimeGrain.MONTH]
        )

        registry.register_metric(profit_margin)

        catalog = registry.get_catalog()
        engine = SemanticQueryEngine(catalog)

        query = MetricQuery(
            metrics=["profit_margin"],
            dimensions=[],
            filters=[],
            time_grain="month",
            start_date="2024-01-01",
            end_date="2024-12-31"
        )

        sql = engine.generate_sql(query)

        # Check that both base metrics are expanded
        assert "SUM(amount)" in sql  # revenue
        assert "SUM(cost)" in sql    # costs
        assert "* 100" in sql         # percentage calculation


class TestPerformanceAndScale:
    """Test performance and scalability."""

    def test_large_metric_catalog(self):
        """Test handling large number of metrics."""
        catalog = MetricCatalog()

        # Add 1000 metrics
        for i in range(1000):
            metric = MetricDefinition(
                name=f"metric_{i}",
                label=f"Metric {i}",
                description=f"Description for metric {i}",
                model="fact_table",
                calculation_method=CalculationMethod.SUM,
                expression=f"column_{i}",
                timestamp="date",
                time_grains=[TimeGrain.DAY],
                meta={"category": f"cat_{i % 10}", "tier": (i % 3) + 1}
            )
            catalog.add_metric(metric)

        # Test searching
        results = catalog.list_metrics(search="metric_500")
        assert len(results) == 1
        assert results[0].name == "metric_500"

        # Test filtering by category
        cat_results = catalog.list_metrics(category="cat_5")
        assert len(cat_results) == 100  # 1000 metrics / 10 categories

    def test_complex_query_generation(self):
        """Test generating complex SQL queries."""
        catalog = MetricCatalog()

        # Add metrics with many dimensions
        metric = MetricDefinition(
            name="complex_metric",
            label="Complex Metric",
            description="A complex metric",
            model="fact_table",
            calculation_method=CalculationMethod.SUM,
            expression="amount",
            timestamp="created_at",
            time_grains=[TimeGrain.DAY],
            dimensions=[f"dim_{i}" for i in range(20)],  # 20 dimensions
            filters=[
                {"field": f"filter_{i}", "operator": "=", "value": f"value_{i}"}
                for i in range(10)  # 10 filters
            ]
        )
        catalog.add_metric(metric)

        engine = SemanticQueryEngine(catalog)

        query = MetricQuery(
            metrics=["complex_metric"],
            dimensions=[f"dim_{i}" for i in range(10)],  # Query 10 dimensions
            filters=[
                {"field": f"user_filter_{i}", "operator": "IN", "value": [i, i + 1, i + 2]}
                for i in range(5)  # Add 5 more filters
            ],
            time_grain="day",
            start_date="2024-01-01",
            end_date="2024-12-31",
            limit=1000,
            offset=0
        )

        sql = engine.generate_sql(query)

        # Verify SQL contains all elements
        assert "SUM(amount)" in sql
        for i in range(10):
            assert f"dim_{i}" in sql  # Dimensions in SELECT and GROUP BY
            assert f"filter_{i} = 'value_{i}'" in sql  # Metric filters (escaped literals)

        for i in range(5):
            assert f"user_filter_{i} IN ({i}, {i+1}, {i+2})" in sql  # Query filters

        assert "LIMIT 1000" in sql
        assert "OFFSET 0" in sql