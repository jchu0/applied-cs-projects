"""Tests for metadata collectors."""

import asyncio
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from observability.collectors import (
    BigQueryCollector,
    DatabricksCollector,
    GenericSQLCollector,
    InMemoryCollector,
    MetadataCollector,
    PostgresCollector,
    RedshiftCollector,
    SnowflakeCollector,
    create_collector,
)
from observability.config import CollectorConfig, WarehouseConfig
from observability.models import (
    ColumnMetadata,
    ColumnStats,
    LineageInfo,
    TableMetadata,
)


@pytest.fixture
def collector_config():
    """Create a collector configuration."""
    return CollectorConfig(
        sample_size=100,
        profile_timeout_seconds=60,
        batch_size=50,
        parallel_workers=2,
    )


@pytest.fixture
def warehouse_config():
    """Create a warehouse configuration."""
    return WarehouseConfig(
        warehouse_type="postgres",
        host="localhost",
        port=5432,
        database="testdb",
        user="testuser",
        password="testpass",
    )


@pytest.fixture
def sample_table_metadata():
    """Create sample table metadata."""
    return TableMetadata(
        table_id="db.schema.table",
        database="db",
        schema="schema",
        table_name="table",
        columns=[
            ColumnMetadata(
                name="id",
                data_type="integer",
                nullable=False,
                description="Primary key",
                stats=None,
            ),
            ColumnMetadata(
                name="name",
                data_type="varchar",
                nullable=True,
                description="User name",
                stats=None,
            ),
        ],
        row_count=1000,
        size_bytes=50000,
        last_modified=datetime.now(),
    )


@pytest.fixture
def sample_column_stats():
    """Create sample column statistics."""
    return {
        "id": ColumnStats(
            null_count=0,
            null_ratio=0.0,
            distinct_count=1000,
            min_value=1,
            max_value=1000,
            mean=500.5,
            stddev=288.7,
        ),
        "name": ColumnStats(
            null_count=10,
            null_ratio=0.01,
            distinct_count=950,
        ),
    }


class TestInMemoryCollector:
    """Tests for InMemoryCollector."""

    def test_creation(self, collector_config):
        """Test collector creation."""
        collector = InMemoryCollector(collector_config)
        assert collector.config == collector_config

    @pytest.mark.asyncio
    async def test_add_and_collect_table(
        self, collector_config, sample_table_metadata
    ):
        """Test adding and collecting table metadata."""
        collector = InMemoryCollector(collector_config)
        collector.add_table(sample_table_metadata)

        result = await collector.collect_schema("db.schema.table")
        assert result.table_id == sample_table_metadata.table_id
        assert len(result.columns) == 2

    @pytest.mark.asyncio
    async def test_collect_nonexistent_table(self, collector_config):
        """Test collecting non-existent table raises error."""
        collector = InMemoryCollector(collector_config)

        with pytest.raises(ValueError, match="not found"):
            await collector.collect_schema("nonexistent.table")

    @pytest.mark.asyncio
    async def test_collect_stats(
        self, collector_config, sample_table_metadata, sample_column_stats
    ):
        """Test collecting column statistics."""
        collector = InMemoryCollector(collector_config)
        collector.add_table(sample_table_metadata, stats=sample_column_stats)

        stats = await collector.collect_stats("db.schema.table")
        assert "id" in stats
        assert stats["id"].distinct_count == 1000

    @pytest.mark.asyncio
    async def test_collect_lineage(
        self, collector_config, sample_table_metadata
    ):
        """Test collecting lineage information."""
        collector = InMemoryCollector(collector_config)
        lineage = LineageInfo(
            table_id="db.schema.table",
            upstream=["db.schema.source"],
            downstream=["db.schema.target"],
        )
        collector.add_table(sample_table_metadata, lineage=lineage)

        result = await collector.collect_lineage("db.schema.table")
        assert result.upstream == ["db.schema.source"]

    @pytest.mark.asyncio
    async def test_get_column_sample(self, collector_config):
        """Test getting column sample data."""
        collector = InMemoryCollector(collector_config)
        collector._samples["db.schema.table"] = {
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"]
        }

        sample = await collector.get_column_sample("db.schema.table", "name", 3)
        assert len(sample) == 3
        assert sample == ["Alice", "Bob", "Charlie"]


class TestGenericSQLCollector:
    """Tests for GenericSQLCollector."""

    @pytest.mark.asyncio
    async def test_collect_schema_invalid_ref(
        self, collector_config, warehouse_config
    ):
        """Test collect_schema with invalid table reference."""
        collector = GenericSQLCollector(collector_config, warehouse_config)

        with pytest.raises(ValueError, match="Invalid table reference"):
            await collector.collect_schema("invalid_ref")

    @pytest.mark.asyncio
    async def test_profile_column_numeric(
        self, collector_config, warehouse_config
    ):
        """Test profiling numeric column."""
        collector = GenericSQLCollector(collector_config, warehouse_config)

        # Mock execute to return expected results
        collector._execute = AsyncMock(return_value=[{
            "total_count": 1000,
            "null_count": 10,
            "distinct_count": 500,
            "min_value": 0,
            "max_value": 1000,
            "mean": 500.0,
            "stddev": 250.0,
        }])

        stats = await collector._profile_column(
            "db.schema.table", "amount", "numeric"
        )

        assert stats.null_count == 10
        assert stats.null_ratio == 0.01
        assert stats.mean == 500.0

    @pytest.mark.asyncio
    async def test_profile_column_string(
        self, collector_config, warehouse_config
    ):
        """Test profiling string column."""
        collector = GenericSQLCollector(collector_config, warehouse_config)

        collector._execute = AsyncMock(return_value=[{
            "total_count": 1000,
            "null_count": 50,
            "distinct_count": 200,
            "min_value": "aaa",
            "max_value": "zzz",
            "mean": None,
            "stddev": None,
        }])

        stats = await collector._profile_column(
            "db.schema.table", "name", "varchar"
        )

        assert stats.null_count == 50
        assert stats.null_ratio == 0.05
        assert stats.mean is None

    @pytest.mark.asyncio
    async def test_collect_lineage_default(
        self, collector_config, warehouse_config
    ):
        """Test default lineage collection."""
        collector = GenericSQLCollector(collector_config, warehouse_config)

        lineage = await collector.collect_lineage("db.schema.table")
        assert lineage.table_id == "db.schema.table"


class TestSnowflakeCollector:
    """Tests for SnowflakeCollector."""

    @pytest.mark.asyncio
    async def test_connect_without_library(
        self, collector_config, warehouse_config
    ):
        """Test connection fails without snowflake library."""
        warehouse_config.warehouse_type = "snowflake"
        collector = SnowflakeCollector(collector_config, warehouse_config)

        with patch("observability.collectors.HAS_SNOWFLAKE", False):
            with pytest.raises(ImportError, match="snowflake"):
                await collector.connect()

    @pytest.mark.asyncio
    async def test_collect_schema_valid(
        self, collector_config, warehouse_config
    ):
        """Test schema collection with valid table reference."""
        warehouse_config.warehouse_type = "snowflake"
        collector = SnowflakeCollector(collector_config, warehouse_config)

        # Mock execute to return expected results
        collector._execute = AsyncMock(side_effect=[
            # columns query result
            [
                {"column_name": "id", "data_type": "NUMBER", "is_nullable": "NO", "comment": "Primary key"},
                {"column_name": "name", "data_type": "VARCHAR", "is_nullable": "YES", "comment": None},
            ],
            # stats query result
            [{"row_count": 1000, "bytes": 50000, "last_altered": datetime.now()}],
        ])

        result = await collector.collect_schema("DB.SCHEMA.TABLE")

        assert result.table_id == "DB.SCHEMA.TABLE"
        assert result.database == "DB"
        assert result.schema == "SCHEMA"
        assert result.table_name == "TABLE"
        assert len(result.columns) == 2
        assert result.columns[0].name == "id"

    @pytest.mark.asyncio
    async def test_collect_schema_invalid_ref(
        self, collector_config, warehouse_config
    ):
        """Test schema collection with invalid reference."""
        warehouse_config.warehouse_type = "snowflake"
        collector = SnowflakeCollector(collector_config, warehouse_config)

        with pytest.raises(ValueError, match="Invalid table reference"):
            await collector.collect_schema("INVALID")


class TestBigQueryCollector:
    """Tests for BigQueryCollector."""

    @pytest.mark.asyncio
    async def test_connect_without_library(
        self, collector_config, warehouse_config
    ):
        """Test connection fails without bigquery library."""
        warehouse_config.warehouse_type = "bigquery"
        collector = BigQueryCollector(collector_config, warehouse_config)

        with patch("observability.collectors.HAS_BIGQUERY", False):
            with pytest.raises(ImportError, match="bigquery"):
                await collector.connect()

    @pytest.mark.asyncio
    async def test_collect_schema_valid(
        self, collector_config, warehouse_config
    ):
        """Test schema collection with valid table reference."""
        warehouse_config.warehouse_type = "bigquery"
        collector = BigQueryCollector(collector_config, warehouse_config)

        # Mock execute
        collector._execute = AsyncMock(side_effect=[
            # columns query
            [
                {"column_name": "user_id", "data_type": "INT64", "is_nullable": "NO", "description": "User ID"},
                {"column_name": "email", "data_type": "STRING", "is_nullable": "YES", "description": None},
            ],
            # stats query
            [{"row_count": 5000, "size_bytes": 100000, "last_modified_time": 1704067200000}],
        ])

        result = await collector.collect_schema("project.dataset.table")

        assert result.table_id == "project.dataset.table"
        assert result.database == "project"
        assert result.schema == "dataset"
        assert len(result.columns) == 2


class TestPostgresCollector:
    """Tests for PostgresCollector."""

    @pytest.mark.asyncio
    async def test_connect_without_library(
        self, collector_config, warehouse_config
    ):
        """Test connection fails without asyncpg library."""
        collector = PostgresCollector(collector_config, warehouse_config)

        with patch("observability.collectors.HAS_ASYNCPG", False):
            with pytest.raises(ImportError, match="asyncpg"):
                await collector.connect()

    @pytest.mark.asyncio
    async def test_collect_schema_two_part_ref(
        self, collector_config, warehouse_config
    ):
        """Test schema collection with two-part table reference."""
        collector = PostgresCollector(collector_config, warehouse_config)

        collector._execute = AsyncMock(side_effect=[
            # columns query
            [
                {"column_name": "id", "data_type": "integer", "is_nullable": "NO", "description": None},
            ],
            # stats query
            [{"row_count": 100, "size_bytes": 8192}],
        ])

        result = await collector.collect_schema("public.users")

        assert result.schema == "public"
        assert result.table_name == "users"
        assert result.database == "testdb"

    @pytest.mark.asyncio
    async def test_collect_schema_three_part_ref(
        self, collector_config, warehouse_config
    ):
        """Test schema collection with three-part table reference."""
        collector = PostgresCollector(collector_config, warehouse_config)

        collector._execute = AsyncMock(side_effect=[
            [{"column_name": "id", "data_type": "integer", "is_nullable": "NO", "description": None}],
            [{"row_count": 100, "size_bytes": 8192}],
        ])

        result = await collector.collect_schema("mydb.public.users")

        assert result.database == "mydb"
        assert result.schema == "public"


class TestRedshiftCollector:
    """Tests for RedshiftCollector."""

    @pytest.mark.asyncio
    async def test_collect_schema_uses_svv_tables(
        self, collector_config, warehouse_config
    ):
        """Test that Redshift uses SVV system tables."""
        warehouse_config.warehouse_type = "redshift"
        collector = RedshiftCollector(collector_config, warehouse_config)

        execute_calls = []

        async def mock_execute(query):
            execute_calls.append(query)
            if "SVV_COLUMNS" in query:
                return [{"column_name": "id", "data_type": "int4", "is_nullable": "NO", "description": None}]
            elif "SVV_TABLE_INFO" in query:
                return [{"row_count": 1000, "size_bytes": 102400}]
            return []

        collector._execute = mock_execute

        await collector.collect_schema("public.events")

        assert any("SVV_COLUMNS" in call for call in execute_calls)
        assert any("SVV_TABLE_INFO" in call for call in execute_calls)


class TestDatabricksCollector:
    """Tests for DatabricksCollector."""

    @pytest.mark.asyncio
    async def test_connect_without_library(
        self, collector_config, warehouse_config
    ):
        """Test connection fails without databricks library."""
        warehouse_config.warehouse_type = "databricks"
        warehouse_config.options["http_path"] = "/sql/1.0/warehouses/abc"
        collector = DatabricksCollector(collector_config, warehouse_config)

        with patch("observability.collectors.HAS_DATABRICKS", False):
            with pytest.raises(ImportError, match="databricks"):
                await collector.connect()

    @pytest.mark.asyncio
    async def test_collect_schema_unity_catalog(
        self, collector_config, warehouse_config
    ):
        """Test schema collection with Unity Catalog format."""
        warehouse_config.warehouse_type = "databricks"
        collector = DatabricksCollector(collector_config, warehouse_config)

        collector._execute = AsyncMock(side_effect=[
            # DESCRIBE TABLE result
            [
                {"col_name": "id", "data_type": "bigint", "comment": "Primary key"},
                {"col_name": "created_at", "data_type": "timestamp", "comment": None},
                {"col_name": "# Partition Information", "data_type": "", "comment": ""},
            ],
            # DESCRIBE DETAIL result
            [{"numFiles": 10, "sizeInBytes": 1048576, "lastModified": 1704067200000}],
        ])

        result = await collector.collect_schema("catalog.schema.table")

        assert result.database == "catalog"
        assert result.schema == "schema"
        assert len(result.columns) == 2  # Excludes partition info row
        assert result.columns[0].name == "id"


class TestCreateCollector:
    """Tests for create_collector factory function."""

    def test_create_snowflake(self, collector_config, warehouse_config):
        """Test creating Snowflake collector."""
        warehouse_config.warehouse_type = "snowflake"
        collector = create_collector(collector_config, warehouse_config)
        assert isinstance(collector, SnowflakeCollector)

    def test_create_bigquery(self, collector_config, warehouse_config):
        """Test creating BigQuery collector."""
        warehouse_config.warehouse_type = "bigquery"
        collector = create_collector(collector_config, warehouse_config)
        assert isinstance(collector, BigQueryCollector)

    def test_create_bigquery_alias(self, collector_config, warehouse_config):
        """Test creating BigQuery collector with alias."""
        warehouse_config.warehouse_type = "bq"
        collector = create_collector(collector_config, warehouse_config)
        assert isinstance(collector, BigQueryCollector)

    def test_create_postgres(self, collector_config, warehouse_config):
        """Test creating PostgreSQL collector."""
        warehouse_config.warehouse_type = "postgres"
        collector = create_collector(collector_config, warehouse_config)
        assert isinstance(collector, PostgresCollector)

    def test_create_postgresql(self, collector_config, warehouse_config):
        """Test creating PostgreSQL collector with full name."""
        warehouse_config.warehouse_type = "postgresql"
        collector = create_collector(collector_config, warehouse_config)
        assert isinstance(collector, PostgresCollector)

    def test_create_redshift(self, collector_config, warehouse_config):
        """Test creating Redshift collector."""
        warehouse_config.warehouse_type = "redshift"
        collector = create_collector(collector_config, warehouse_config)
        assert isinstance(collector, RedshiftCollector)

    def test_create_databricks(self, collector_config, warehouse_config):
        """Test creating Databricks collector."""
        warehouse_config.warehouse_type = "databricks"
        collector = create_collector(collector_config, warehouse_config)
        assert isinstance(collector, DatabricksCollector)

    def test_create_unknown_fallback(self, collector_config, warehouse_config):
        """Test fallback to generic collector for unknown types."""
        warehouse_config.warehouse_type = "unknown"
        collector = create_collector(collector_config, warehouse_config)
        assert isinstance(collector, GenericSQLCollector)

    def test_create_case_insensitive(self, collector_config, warehouse_config):
        """Test warehouse type is case insensitive."""
        warehouse_config.warehouse_type = "SNOWFLAKE"
        collector = create_collector(collector_config, warehouse_config)
        assert isinstance(collector, SnowflakeCollector)


class TestCollectorConnectionManagement:
    """Tests for connection management in collectors."""

    @pytest.mark.asyncio
    async def test_disconnect(self, collector_config, warehouse_config):
        """Test disconnection cleanup."""
        collector = GenericSQLCollector(collector_config, warehouse_config)
        collector._connected = True
        collector._connection = MagicMock()
        collector._connection.close = MagicMock()

        await collector.disconnect()

        assert collector._connection is None
        assert collector._connected is False

    @pytest.mark.asyncio
    async def test_context_manager(self, collector_config, warehouse_config):
        """Test connection context manager."""
        collector = PostgresCollector(collector_config, warehouse_config)

        # Mock connect
        collector.connect = AsyncMock()
        collector._connected = True
        collector._connection = MagicMock()

        async with collector.connection() as conn:
            assert conn is not None

    @pytest.mark.asyncio
    async def test_execute_without_connection(
        self, collector_config, warehouse_config
    ):
        """Test execute returns empty without connection."""
        collector = GenericSQLCollector(collector_config, warehouse_config)

        # Mock _get_connection to return None
        collector._get_connection = AsyncMock(return_value=None)

        result = await collector._execute("SELECT 1")
        assert result == []


class TestCollectorEdgeCases:
    """Tests for edge cases in collectors."""

    @pytest.mark.asyncio
    async def test_empty_table(self, collector_config, warehouse_config):
        """Test handling empty table results."""
        collector = GenericSQLCollector(collector_config, warehouse_config)
        collector._execute = AsyncMock(return_value=[])

        stats = await collector._profile_column("db.schema.table", "col", "varchar")

        assert stats.null_count == 0
        assert stats.null_ratio == 0.0
        assert stats.distinct_count == 0

    @pytest.mark.asyncio
    async def test_null_stats(self, collector_config, warehouse_config):
        """Test handling null statistics."""
        collector = GenericSQLCollector(collector_config, warehouse_config)
        collector._execute = AsyncMock(return_value=[{
            "total_count": 0,
            "null_count": 0,
            "distinct_count": 0,
            "min_value": None,
            "max_value": None,
            "mean": None,
            "stddev": None,
        }])

        stats = await collector._profile_column("db.schema.table", "col", "integer")

        assert stats.null_ratio == 0.0
        assert stats.mean is None

    @pytest.mark.asyncio
    async def test_sample_empty_result(self, collector_config, warehouse_config):
        """Test getting sample from empty table."""
        collector = GenericSQLCollector(collector_config, warehouse_config)
        collector._execute = AsyncMock(return_value=[])

        sample = await collector.get_column_sample("db.schema.table", "col", 10)
        assert sample == []
