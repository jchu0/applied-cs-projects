"""Tests for configuration classes."""

import pytest
from typing import Dict, Any

from semantic_layer.config import (
    WarehouseConfig,
    ModelConfig,
    SourceConfig,
    SemanticLayerConfig,
    MetricFilter,
    MetricMeta,
)


class TestWarehouseConfig:
    """Test WarehouseConfig dataclass."""

    def test_create_snowflake_config(self):
        """Test creating a Snowflake warehouse config."""
        config = WarehouseConfig(
            warehouse_type="snowflake",
            account="my_account",
            user="my_user",
            password="my_password",
            database="analytics",
            schema="marts",
        )

        assert config.warehouse_type == "snowflake"
        assert config.account == "my_account"
        assert config.user == "my_user"
        assert config.database == "analytics"
        assert config.schema == "marts"

    def test_create_bigquery_config(self):
        """Test creating a BigQuery warehouse config."""
        config = WarehouseConfig(
            warehouse_type="bigquery",
            project="my-gcp-project",
            database="analytics",
            schema="marts",
        )

        assert config.warehouse_type == "bigquery"
        assert config.project == "my-gcp-project"

    def test_create_postgres_config(self):
        """Test creating a Postgres warehouse config."""
        config = WarehouseConfig(
            warehouse_type="postgres",
            host="localhost",
            port=5432,
            database="analytics",
            schema="public",
            user="postgres",
            password="password",
        )

        assert config.warehouse_type == "postgres"
        assert config.host == "localhost"
        assert config.port == 5432

    def test_create_redshift_config(self):
        """Test creating a Redshift warehouse config."""
        config = WarehouseConfig(
            warehouse_type="redshift",
            host="my-cluster.xxxxx.region.redshift.amazonaws.com",
            port=5439,
            database="dev",
            schema="public",
            user="admin",
            password="password",
        )

        assert config.warehouse_type == "redshift"
        assert config.port == 5439

    def test_default_values(self):
        """Test default values for warehouse config."""
        config = WarehouseConfig(warehouse_type="postgres")

        assert config.host == ""
        assert config.port == 0
        assert config.database == ""
        assert config.schema == ""
        assert config.user == ""
        assert config.password == ""
        assert config.account == ""
        assert config.project == ""
        assert config.options == {}

    def test_with_options(self):
        """Test warehouse config with additional options."""
        config = WarehouseConfig(
            warehouse_type="snowflake",
            account="my_account",
            options={
                "warehouse": "compute_wh",
                "role": "analyst",
                "authenticator": "externalbrowser",
            },
        )

        assert config.options["warehouse"] == "compute_wh"
        assert config.options["role"] == "analyst"
        assert config.options["authenticator"] == "externalbrowser"


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_create_basic_config(self):
        """Test creating a basic model config."""
        config = ModelConfig(name="my_model")

        assert config.name == "my_model"
        assert config.materialized == "view"
        assert config.schema == ""
        assert config.unique_key is None
        assert config.incremental_strategy is None
        assert config.partition_by is None
        assert config.cluster_by is None
        assert config.tags == []

    def test_table_materialization(self):
        """Test table materialization config."""
        config = ModelConfig(
            name="dim_customers",
            materialized="table",
            schema="marts",
        )

        assert config.materialized == "table"
        assert config.schema == "marts"

    def test_incremental_config(self):
        """Test incremental model config."""
        config = ModelConfig(
            name="fct_events",
            materialized="incremental",
            unique_key="event_id",
            incremental_strategy="merge",
        )

        assert config.materialized == "incremental"
        assert config.unique_key == "event_id"
        assert config.incremental_strategy == "merge"

    def test_partition_and_cluster_config(self):
        """Test BigQuery-style partition and cluster config."""
        config = ModelConfig(
            name="fct_events",
            materialized="table",
            partition_by={
                "field": "event_date",
                "data_type": "date",
                "granularity": "day",
            },
            cluster_by=["event_type", "user_id"],
        )

        assert config.partition_by["field"] == "event_date"
        assert config.partition_by["granularity"] == "day"
        assert "event_type" in config.cluster_by
        assert "user_id" in config.cluster_by

    def test_with_tags(self):
        """Test model config with tags."""
        config = ModelConfig(
            name="fct_orders",
            materialized="table",
            tags=["daily", "core", "finance"],
        )

        assert "daily" in config.tags
        assert "core" in config.tags
        assert "finance" in config.tags


class TestSourceConfig:
    """Test SourceConfig dataclass."""

    def test_create_basic_source(self):
        """Test creating a basic source config."""
        config = SourceConfig(
            name="raw_data",
            database="raw",
            schema="public",
        )

        assert config.name == "raw_data"
        assert config.database == "raw"
        assert config.schema == "public"
        assert config.loader == ""
        assert config.loaded_at_field == ""
        assert config.freshness_warn_after_hours == 12
        assert config.freshness_error_after_hours == 24
        assert config.tables == []

    def test_source_with_loader(self):
        """Test source config with loader information."""
        config = SourceConfig(
            name="shopify_data",
            database="raw",
            schema="shopify",
            loader="fivetran",
            loaded_at_field="_fivetran_synced",
        )

        assert config.loader == "fivetran"
        assert config.loaded_at_field == "_fivetran_synced"

    def test_source_with_freshness(self):
        """Test source config with custom freshness settings."""
        config = SourceConfig(
            name="events_data",
            database="raw",
            schema="events",
            freshness_warn_after_hours=1,
            freshness_error_after_hours=2,
        )

        assert config.freshness_warn_after_hours == 1
        assert config.freshness_error_after_hours == 2

    def test_source_with_tables(self):
        """Test source config with table definitions."""
        config = SourceConfig(
            name="stripe_data",
            database="raw",
            schema="stripe",
            loader="fivetran",
            tables=[
                {
                    "name": "charges",
                    "identifier": "charge",
                    "columns": [
                        {"name": "id", "description": "Charge ID"},
                        {"name": "amount", "description": "Charge amount in cents"},
                    ],
                },
                {
                    "name": "customers",
                    "identifier": "customer",
                    "columns": [
                        {"name": "id", "description": "Customer ID"},
                        {"name": "email", "description": "Customer email"},
                    ],
                },
            ],
        )

        assert len(config.tables) == 2
        assert config.tables[0]["name"] == "charges"
        assert config.tables[1]["name"] == "customers"


class TestSemanticLayerConfig:
    """Test SemanticLayerConfig dataclass."""

    def test_default_config(self):
        """Test default semantic layer config."""
        config = SemanticLayerConfig()

        assert config.project_name == "analytics"
        assert config.version == "1.0.0"
        assert config.warehouse is None
        assert config.model_defaults is not None
        assert config.staging_schema == "staging"
        assert config.marts_schema == "analytics"
        assert config.metric_definitions_path == "semantic_layer/metrics"
        assert config.dimension_definitions_path == "semantic_layer/dimensions"

    def test_custom_project_config(self):
        """Test custom project configuration."""
        config = SemanticLayerConfig(
            project_name="my_analytics",
            version="2.0.0",
            staging_schema="stg",
            marts_schema="marts",
        )

        assert config.project_name == "my_analytics"
        assert config.version == "2.0.0"
        assert config.staging_schema == "stg"
        assert config.marts_schema == "marts"

    def test_config_with_warehouse(self):
        """Test config with warehouse connection."""
        warehouse = WarehouseConfig(
            warehouse_type="snowflake",
            account="my_account",
            database="analytics",
        )

        config = SemanticLayerConfig(
            project_name="analytics",
            warehouse=warehouse,
        )

        assert config.warehouse is not None
        assert config.warehouse.warehouse_type == "snowflake"
        assert config.warehouse.account == "my_account"

    def test_config_with_model_defaults(self):
        """Test config with custom model defaults."""
        model_defaults = ModelConfig(
            name="default",
            materialized="table",
            schema="marts",
            tags=["core"],
        )

        config = SemanticLayerConfig(
            project_name="analytics",
            model_defaults=model_defaults,
        )

        assert config.model_defaults.materialized == "table"
        assert config.model_defaults.schema == "marts"

    def test_custom_paths(self):
        """Test custom metric and dimension paths."""
        config = SemanticLayerConfig(
            project_name="analytics",
            metric_definitions_path="models/metrics",
            dimension_definitions_path="models/dimensions",
        )

        assert config.metric_definitions_path == "models/metrics"
        assert config.dimension_definitions_path == "models/dimensions"


class TestMetricFilter:
    """Test MetricFilter dataclass."""

    def test_create_equals_filter(self):
        """Test creating an equals filter."""
        filter_def = MetricFilter(
            field="status",
            operator="=",
            value="completed",
        )

        assert filter_def.field == "status"
        assert filter_def.operator == "="
        assert filter_def.value == "completed"

    def test_create_in_filter(self):
        """Test creating an IN filter."""
        filter_def = MetricFilter(
            field="country",
            operator="IN",
            value="('US', 'CA', 'MX')",
        )

        assert filter_def.field == "country"
        assert filter_def.operator == "IN"
        assert "US" in filter_def.value

    def test_create_comparison_filter(self):
        """Test creating comparison filters."""
        gt_filter = MetricFilter(
            field="amount",
            operator=">",
            value="100",
        )

        gte_filter = MetricFilter(
            field="quantity",
            operator=">=",
            value="10",
        )

        assert gt_filter.operator == ">"
        assert gte_filter.operator == ">="

    def test_create_like_filter(self):
        """Test creating a LIKE filter."""
        filter_def = MetricFilter(
            field="email",
            operator="LIKE",
            value="'%@company.com'",
        )

        assert filter_def.operator == "LIKE"
        assert "@company.com" in filter_def.value

    def test_create_not_equals_filter(self):
        """Test creating a not equals filter."""
        filter_def = MetricFilter(
            field="status",
            operator="!=",
            value="cancelled",
        )

        assert filter_def.operator == "!="


class TestMetricMeta:
    """Test MetricMeta dataclass."""

    def test_default_values(self):
        """Test default values for metric metadata."""
        meta = MetricMeta()

        assert meta.owner == ""
        assert meta.tier == 1
        assert meta.is_percentage is False
        assert meta.deprecated is False
        assert meta.tags == []

    def test_custom_owner_and_tier(self):
        """Test custom owner and tier."""
        meta = MetricMeta(
            owner="finance-team",
            tier=1,
        )

        assert meta.owner == "finance-team"
        assert meta.tier == 1

    def test_tier_levels(self):
        """Test different tier levels."""
        tier1 = MetricMeta(tier=1)  # Critical metric
        tier2 = MetricMeta(tier=2)  # Important metric
        tier3 = MetricMeta(tier=3)  # Nice to have metric

        assert tier1.tier == 1
        assert tier2.tier == 2
        assert tier3.tier == 3

    def test_percentage_metric(self):
        """Test percentage metric flag."""
        meta = MetricMeta(
            owner="analytics",
            is_percentage=True,
        )

        assert meta.is_percentage is True

    def test_deprecated_metric(self):
        """Test deprecated metric flag."""
        meta = MetricMeta(
            owner="legacy-team",
            deprecated=True,
        )

        assert meta.deprecated is True

    def test_with_tags(self):
        """Test metric metadata with tags."""
        meta = MetricMeta(
            owner="product-team",
            tags=["growth", "acquisition", "weekly"],
        )

        assert "growth" in meta.tags
        assert "acquisition" in meta.tags
        assert "weekly" in meta.tags

    def test_full_metadata(self):
        """Test fully specified metric metadata."""
        meta = MetricMeta(
            owner="data-team",
            tier=1,
            is_percentage=True,
            deprecated=False,
            tags=["conversion", "funnel", "daily"],
        )

        assert meta.owner == "data-team"
        assert meta.tier == 1
        assert meta.is_percentage is True
        assert meta.deprecated is False
        assert len(meta.tags) == 3
