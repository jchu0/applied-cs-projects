"""Tests for fact table model definitions and builders."""

import pytest
from typing import List, Dict, Any

from semantic_layer.facts import (
    FactColumn,
    FactModel,
    FctOrders,
    FctRevenue,
    FctCosts,
    FactBuilder,
)


class TestFactColumn:
    """Test FactColumn dataclass."""

    def test_create_basic_column(self):
        """Test creating a basic fact column."""
        col = FactColumn(
            name="order_id",
            data_type="string",
            description="Order identifier",
        )

        assert col.name == "order_id"
        assert col.data_type == "string"
        assert col.description == "Order identifier"
        assert col.is_key is False
        assert col.is_foreign_key is False
        assert col.is_measure is False
        assert col.is_degenerate_dimension is False

    def test_create_key_column(self):
        """Test creating a primary key column."""
        col = FactColumn(
            name="fact_key",
            data_type="integer",
            description="Surrogate key",
            is_key=True,
        )

        assert col.is_key is True
        assert col.is_foreign_key is False

    def test_create_foreign_key_column(self):
        """Test creating a foreign key column."""
        col = FactColumn(
            name="customer_id",
            data_type="string",
            description="Foreign key to dim_customers",
            is_foreign_key=True,
        )

        assert col.is_foreign_key is True
        assert col.is_key is False

    def test_create_measure_column(self):
        """Test creating a measure column."""
        col = FactColumn(
            name="order_total",
            data_type="decimal",
            description="Total order amount",
            is_measure=True,
        )

        assert col.is_measure is True

    def test_create_degenerate_dimension_column(self):
        """Test creating a degenerate dimension column."""
        col = FactColumn(
            name="payment_method",
            data_type="string",
            description="Payment method used",
            is_degenerate_dimension=True,
        )

        assert col.is_degenerate_dimension is True


class TestFactModel:
    """Test FactModel dataclass."""

    def test_create_basic_fact(self):
        """Test creating a basic fact model."""
        columns = [
            FactColumn(
                name="order_id",
                data_type="string",
                description="Primary key",
                is_key=True,
            ),
            FactColumn(
                name="amount",
                data_type="decimal",
                description="Order amount",
                is_measure=True,
            ),
        ]

        fact = FactModel(
            name="fct_orders",
            description="Order fact table",
            source_models=["stg_orders"],
            columns=columns,
            grain="one row per order",
        )

        assert fact.name == "fct_orders"
        assert fact.description == "Order fact table"
        assert fact.source_models == ["stg_orders"]
        assert len(fact.columns) == 2
        assert fact.grain == "one row per order"
        assert fact.unique_key == ""
        assert fact.incremental_strategy is None
        assert fact.partition_by is None
        assert fact.config == {}

    def test_fact_with_incremental_config(self):
        """Test fact model with incremental configuration."""
        fact = FactModel(
            name="fct_events",
            description="Event fact table",
            source_models=["stg_events"],
            columns=[],
            grain="one row per event",
            unique_key="event_id",
            incremental_strategy="merge",
        )

        assert fact.unique_key == "event_id"
        assert fact.incremental_strategy == "merge"

    def test_fact_with_partition(self):
        """Test fact model with partitioning."""
        fact = FactModel(
            name="fct_sales",
            description="Sales fact table",
            source_models=["stg_sales"],
            columns=[],
            grain="one row per sale",
            partition_by={
                "field": "sale_date",
                "data_type": "date",
                "granularity": "day",
            },
        )

        assert fact.partition_by["field"] == "sale_date"
        assert fact.partition_by["granularity"] == "day"

    def test_fact_with_config(self):
        """Test fact model with custom config."""
        fact = FactModel(
            name="fct_transactions",
            description="Transaction fact table",
            source_models=["stg_transactions"],
            columns=[],
            grain="one row per transaction",
            config={
                "materialized": "incremental",
                "tags": ["hourly", "finance"],
                "schema": "marts_finance",
            },
        )

        assert fact.config["materialized"] == "incremental"
        assert "hourly" in fact.config["tags"]

    def test_fact_with_multiple_sources(self):
        """Test fact model with multiple source models."""
        fact = FactModel(
            name="fct_order_items",
            description="Order items fact table",
            source_models=["stg_orders", "stg_order_items", "stg_products"],
            columns=[],
            grain="one row per order item",
        )

        assert len(fact.source_models) == 3
        assert "stg_orders" in fact.source_models


class TestFctOrders:
    """Test FctOrders fact model."""

    def test_get_definition(self):
        """Test getting order fact definition."""
        definition = FctOrders.get_definition()

        assert definition.name == "fct_orders"
        assert "one row per order" in definition.grain
        assert definition.unique_key == "order_id"
        assert definition.incremental_strategy == "merge"

    def test_source_models(self):
        """Test source models for order fact."""
        definition = FctOrders.get_definition()

        assert "int_payments_with_customers" in definition.source_models
        assert "stg_shopify__order_items" in definition.source_models

    def test_columns_exist(self):
        """Test order fact has required columns."""
        definition = FctOrders.get_definition()
        column_names = [col.name for col in definition.columns]

        expected_columns = [
            "order_id",
            "customer_id",
            "order_date",
            "payment_method",
            "country_code",
            "order_total",
            "total_items",
            "unique_products",
        ]

        for expected in expected_columns:
            assert expected in column_names, f"Missing column: {expected}"

    def test_primary_key_defined(self):
        """Test order_id is defined as primary key."""
        definition = FctOrders.get_definition()
        key_columns = [col for col in definition.columns if col.is_key]

        assert len(key_columns) == 1
        assert key_columns[0].name == "order_id"

    def test_foreign_keys_defined(self):
        """Test foreign keys are defined."""
        definition = FctOrders.get_definition()
        fk_columns = [col for col in definition.columns if col.is_foreign_key]

        assert len(fk_columns) >= 1
        fk_names = [col.name for col in fk_columns]
        assert "customer_id" in fk_names

    def test_measures_defined(self):
        """Test measures are defined."""
        definition = FctOrders.get_definition()
        measure_columns = [col for col in definition.columns if col.is_measure]

        assert len(measure_columns) >= 1
        measure_names = [col.name for col in measure_columns]
        assert "order_total" in measure_names
        assert "total_items" in measure_names
        assert "unique_products" in measure_names

    def test_degenerate_dimensions_defined(self):
        """Test degenerate dimensions are defined."""
        definition = FctOrders.get_definition()
        degen_columns = [col for col in definition.columns if col.is_degenerate_dimension]

        assert len(degen_columns) >= 1
        degen_names = [col.name for col in degen_columns]
        assert "payment_method" in degen_names
        assert "country_code" in degen_names

    def test_generate_sql(self):
        """Test SQL generation for order fact."""
        sql = FctOrders.generate_sql()

        assert "materialized='incremental'" in sql
        assert "unique_key='order_id'" in sql
        assert "incremental_strategy='merge'" in sql
        assert "int_payments_with_customers" in sql
        assert "stg_shopify__order_items" in sql
        assert "is_incremental()" in sql
        assert "order_total" in sql

    def test_config_settings(self):
        """Test fact config settings."""
        definition = FctOrders.get_definition()

        assert definition.config["materialized"] == "incremental"
        assert "hourly" in definition.config["tags"]


class TestFctRevenue:
    """Test FctRevenue fact model."""

    def test_get_definition(self):
        """Test getting revenue fact definition."""
        definition = FctRevenue.get_definition()

        assert definition.name == "fct_revenue"
        assert "one row per day per customer segment per country" in definition.grain
        assert definition.unique_key == "revenue_id"

    def test_source_models(self):
        """Test source models for revenue fact."""
        definition = FctRevenue.get_definition()

        assert "fct_orders" in definition.source_models

    def test_columns_exist(self):
        """Test revenue fact has required columns."""
        definition = FctRevenue.get_definition()
        column_names = [col.name for col in definition.columns]

        expected_columns = [
            "revenue_id",
            "revenue_date",
            "customer_segment",
            "country_code",
            "order_count",
            "total_revenue",
            "avg_order_value",
            "unique_customers",
        ]

        for expected in expected_columns:
            assert expected in column_names, f"Missing column: {expected}"

    def test_measures_defined(self):
        """Test measures are defined."""
        definition = FctRevenue.get_definition()
        measure_columns = [col for col in definition.columns if col.is_measure]

        assert len(measure_columns) >= 1
        measure_names = [col.name for col in measure_columns]
        assert "order_count" in measure_names
        assert "total_revenue" in measure_names
        assert "avg_order_value" in measure_names
        assert "unique_customers" in measure_names

    def test_generate_sql(self):
        """Test SQL generation for revenue fact."""
        sql = FctRevenue.generate_sql()

        assert "materialized='table'" in sql
        assert "fct_orders" in sql
        assert "dim_customers" in sql
        assert "customer_segment" in sql
        assert "dbt_utils.generate_surrogate_key" in sql

    def test_config_settings(self):
        """Test fact config settings."""
        definition = FctRevenue.get_definition()

        assert definition.config["materialized"] == "table"
        assert "daily" in definition.config["tags"]
        assert "finance" in definition.config["tags"]


class TestFctCosts:
    """Test FctCosts fact model."""

    def test_get_definition(self):
        """Test getting cost fact definition."""
        definition = FctCosts.get_definition()

        assert definition.name == "fct_costs"
        assert "one row per cost entry" in definition.grain
        assert definition.unique_key == "cost_id"

    def test_source_models(self):
        """Test source models for cost fact."""
        definition = FctCosts.get_definition()

        assert "stg_erp__costs" in definition.source_models

    def test_columns_exist(self):
        """Test cost fact has required columns."""
        definition = FctCosts.get_definition()
        column_names = [col.name for col in definition.columns]

        expected_columns = [
            "cost_id",
            "cost_date",
            "cost_type",
            "cost_center",
            "amount",
        ]

        for expected in expected_columns:
            assert expected in column_names, f"Missing column: {expected}"

    def test_degenerate_dimensions_defined(self):
        """Test degenerate dimensions are defined."""
        definition = FctCosts.get_definition()
        degen_columns = [col for col in definition.columns if col.is_degenerate_dimension]

        assert len(degen_columns) >= 1
        degen_names = [col.name for col in degen_columns]
        assert "cost_type" in degen_names
        assert "cost_center" in degen_names

    def test_config_settings(self):
        """Test fact config settings."""
        definition = FctCosts.get_definition()

        assert definition.config["materialized"] == "table"
        assert "daily" in definition.config["tags"]
        assert "finance" in definition.config["tags"]


class TestFactBuilder:
    """Test FactBuilder class."""

    def test_add_fact(self):
        """Test adding a fact to the builder."""
        builder = FactBuilder()

        fact = FactModel(
            name="fct_test",
            description="Test fact table",
            source_models=["stg_test"],
            columns=[],
            grain="one row per record",
        )

        result = builder.add_fact(fact)

        # Should return self for chaining
        assert result is builder
        assert builder.get_fact("fct_test") is not None

    def test_add_multiple_facts(self):
        """Test adding multiple facts."""
        builder = FactBuilder()

        fact1 = FactModel(
            name="fct_one",
            description="First fact",
            source_models=["stg_one"],
            columns=[],
            grain="one row per record",
        )

        fact2 = FactModel(
            name="fct_two",
            description="Second fact",
            source_models=["stg_two"],
            columns=[],
            grain="one row per record",
        )

        builder.add_fact(fact1).add_fact(fact2)

        assert len(builder.list_facts()) == 2
        assert "fct_one" in builder.list_facts()
        assert "fct_two" in builder.list_facts()

    def test_get_fact(self):
        """Test retrieving a fact."""
        builder = FactBuilder()

        fact = FactModel(
            name="fct_test",
            description="Test fact table",
            source_models=["stg_test"],
            columns=[],
            grain="one row per record",
        )

        builder.add_fact(fact)

        retrieved = builder.get_fact("fct_test")
        assert retrieved is not None
        assert retrieved.name == "fct_test"
        assert retrieved.description == "Test fact table"

    def test_get_nonexistent_fact(self):
        """Test getting a fact that doesn't exist."""
        builder = FactBuilder()

        result = builder.get_fact("nonexistent")
        assert result is None

    def test_list_facts(self):
        """Test listing all facts."""
        builder = FactBuilder()

        facts = [
            FactModel(
                name=f"fct_{i}",
                description=f"Fact {i}",
                source_models=[f"stg_{i}"],
                columns=[],
                grain="one row per record",
            )
            for i in range(5)
        ]

        for fact in facts:
            builder.add_fact(fact)

        fact_list = builder.list_facts()

        assert len(fact_list) == 5
        for i in range(5):
            assert f"fct_{i}" in fact_list

    def test_generate_yaml_basic(self):
        """Test generating YAML schema for a fact."""
        builder = FactBuilder()

        fact = FactModel(
            name="fct_test",
            description="Test fact table",
            source_models=["stg_test"],
            columns=[
                FactColumn(
                    name="test_id",
                    data_type="integer",
                    description="Primary key",
                    is_key=True,
                ),
                FactColumn(
                    name="amount",
                    data_type="decimal",
                    description="Amount measure",
                    is_measure=True,
                ),
            ],
            grain="one row per record",
            unique_key="test_id",
        )

        builder.add_fact(fact)

        yaml_output = builder.generate_yaml("fct_test")

        assert "version: 2" in yaml_output
        assert "name: fct_test" in yaml_output
        assert '"Test fact table"' in yaml_output
        assert "name: test_id" in yaml_output
        assert "unique" in yaml_output
        assert "not_null" in yaml_output

    def test_generate_yaml_with_foreign_keys(self):
        """Test YAML includes tests for foreign key columns."""
        builder = FactBuilder()

        fact = FactModel(
            name="fct_test",
            description="Test fact table",
            source_models=["stg_test"],
            columns=[
                FactColumn(
                    name="order_id",
                    data_type="string",
                    description="Primary key",
                    is_key=True,
                ),
                FactColumn(
                    name="customer_id",
                    data_type="string",
                    description="Foreign key to dim_customers",
                    is_foreign_key=True,
                ),
            ],
            grain="one row per order",
            unique_key="order_id",
        )

        builder.add_fact(fact)

        yaml_output = builder.generate_yaml("fct_test")

        # Foreign key columns should have relationships test
        assert "relationships" in yaml_output
        assert "not_null" in yaml_output

    def test_generate_yaml_with_measures(self):
        """Test YAML includes tests for measure columns."""
        builder = FactBuilder()

        fact = FactModel(
            name="fct_test",
            description="Test fact table",
            source_models=["stg_test"],
            columns=[
                FactColumn(
                    name="id",
                    data_type="string",
                    description="Primary key",
                    is_key=True,
                ),
                FactColumn(
                    name="revenue",
                    data_type="decimal",
                    description="Revenue amount",
                    is_measure=True,
                ),
            ],
            grain="one row per record",
            unique_key="id",
        )

        builder.add_fact(fact)

        yaml_output = builder.generate_yaml("fct_test")

        # Measure columns should have expression_is_true test for non-negative
        assert "dbt_utils.expression_is_true" in yaml_output
        assert ">= 0" in yaml_output

    def test_generate_yaml_nonexistent(self):
        """Test generating YAML for nonexistent fact."""
        builder = FactBuilder()

        result = builder.generate_yaml("nonexistent")
        assert result == ""

    def test_get_incremental_facts(self):
        """Test getting list of incremental facts."""
        builder = FactBuilder()

        # Add incremental fact
        incremental_fact = FactModel(
            name="fct_incremental",
            description="Incremental fact",
            source_models=["stg_source"],
            columns=[],
            grain="one row per record",
            config={"materialized": "incremental"},
        )

        # Add table fact
        table_fact = FactModel(
            name="fct_table",
            description="Table fact",
            source_models=["stg_source"],
            columns=[],
            grain="one row per record",
            config={"materialized": "table"},
        )

        # Add view fact
        view_fact = FactModel(
            name="fct_view",
            description="View fact",
            source_models=["stg_source"],
            columns=[],
            grain="one row per record",
            config={"materialized": "view"},
        )

        builder.add_fact(incremental_fact)
        builder.add_fact(table_fact)
        builder.add_fact(view_fact)

        incremental_facts = builder.get_incremental_facts()

        assert len(incremental_facts) == 1
        assert "fct_incremental" in incremental_facts
        assert "fct_table" not in incremental_facts
        assert "fct_view" not in incremental_facts

    def test_add_predefined_facts(self):
        """Test adding predefined fact models."""
        builder = FactBuilder()

        builder.add_fact(FctOrders.get_definition())
        builder.add_fact(FctRevenue.get_definition())
        builder.add_fact(FctCosts.get_definition())

        fact_list = builder.list_facts()

        assert "fct_orders" in fact_list
        assert "fct_revenue" in fact_list
        assert "fct_costs" in fact_list

    def test_chain_operations(self):
        """Test chaining multiple operations."""
        builder = (
            FactBuilder()
            .add_fact(FctOrders.get_definition())
            .add_fact(FctRevenue.get_definition())
            .add_fact(FctCosts.get_definition())
        )

        assert len(builder.list_facts()) == 3

    def test_overwrite_existing_fact(self):
        """Test that adding a fact with the same name overwrites."""
        builder = FactBuilder()

        fact_v1 = FactModel(
            name="fct_test",
            description="Version 1",
            source_models=["stg_test"],
            columns=[],
            grain="v1",
        )

        fact_v2 = FactModel(
            name="fct_test",
            description="Version 2",
            source_models=["stg_test"],
            columns=[],
            grain="v2",
        )

        builder.add_fact(fact_v1)
        builder.add_fact(fact_v2)

        retrieved = builder.get_fact("fct_test")
        assert retrieved.description == "Version 2"
        assert retrieved.grain == "v2"
