"""Tests for warehouse adapters."""

import pytest
import sqlite3
from unittest.mock import MagicMock, AsyncMock, patch

from semantic_layer.query_engine import (
    WarehouseAdapter,
    SQLiteAdapter,
    PostgresAdapter,
    SnowflakeAdapter,
    BigQueryAdapter,
    InMemoryAdapter,
    create_adapter,
    QueryExecutor,
)


class TestInMemoryAdapter:
    """Tests for InMemoryAdapter."""

    @pytest.mark.asyncio
    async def test_empty_adapter_returns_empty(self):
        """Test that empty adapter returns empty list."""
        adapter = InMemoryAdapter()
        result = await adapter.execute("SELECT * FROM anything")
        assert result == []

    @pytest.mark.asyncio
    async def test_add_and_query_table(self):
        """Test adding data and querying it."""
        adapter = InMemoryAdapter()
        adapter.add_table("orders", [
            {"id": 1, "amount": 100},
            {"id": 2, "amount": 200},
        ])

        result = await adapter.execute("SELECT * FROM orders")
        assert len(result) == 2
        assert result[0]["id"] == 1

    @pytest.mark.asyncio
    async def test_initialize_with_data(self):
        """Test initializing adapter with pre-loaded data."""
        data = {
            "users": [{"name": "Alice"}, {"name": "Bob"}],
            "orders": [{"total": 100}],
        }
        adapter = InMemoryAdapter(data)

        users_result = await adapter.execute("SELECT * FROM users")
        assert len(users_result) == 2

        orders_result = await adapter.execute("SELECT * FROM orders")
        assert len(orders_result) == 1


class TestSQLiteAdapter:
    """Tests for SQLiteAdapter."""

    @pytest.fixture
    def sqlite_connection(self):
        """Create an in-memory SQLite connection with test data."""
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE test_table (
                id INTEGER PRIMARY KEY,
                name TEXT,
                value REAL
            )
        """)
        cursor.execute("INSERT INTO test_table VALUES (1, 'alpha', 10.5)")
        cursor.execute("INSERT INTO test_table VALUES (2, 'beta', 20.5)")
        conn.commit()
        return conn

    @pytest.mark.asyncio
    async def test_execute_simple_query(self, sqlite_connection):
        """Test executing a simple SELECT query."""
        adapter = SQLiteAdapter(sqlite_connection)
        result = await adapter.execute("SELECT * FROM test_table")

        assert len(result) == 2
        assert result[0]["id"] == 1
        assert result[0]["name"] == "alpha"
        assert result[1]["value"] == 20.5

    @pytest.mark.asyncio
    async def test_execute_with_where(self, sqlite_connection):
        """Test executing query with WHERE clause."""
        adapter = SQLiteAdapter(sqlite_connection)
        result = await adapter.execute("SELECT * FROM test_table WHERE id = 1")

        assert len(result) == 1
        assert result[0]["name"] == "alpha"

    @pytest.mark.asyncio
    async def test_close_connection(self, sqlite_connection):
        """Test closing the connection."""
        adapter = SQLiteAdapter(sqlite_connection)
        await adapter.close()
        # Connection should be closed
        with pytest.raises(sqlite3.ProgrammingError):
            sqlite_connection.cursor()


class TestCreateAdapter:
    """Tests for the create_adapter factory function."""

    def test_create_sqlite_adapter(self):
        """Test creating SQLite adapter."""
        conn = MagicMock()
        adapter = create_adapter("sqlite", conn)
        assert isinstance(adapter, SQLiteAdapter)

    def test_create_postgres_adapter(self):
        """Test creating PostgreSQL adapter."""
        conn = MagicMock()
        adapter = create_adapter("postgres", conn)
        assert isinstance(adapter, PostgresAdapter)

    def test_create_postgresql_adapter(self):
        """Test creating PostgreSQL adapter with full name."""
        conn = MagicMock()
        adapter = create_adapter("postgresql", conn)
        assert isinstance(adapter, PostgresAdapter)

    def test_create_redshift_adapter(self):
        """Test creating Redshift adapter (uses PostgresAdapter)."""
        conn = MagicMock()
        adapter = create_adapter("redshift", conn)
        assert isinstance(adapter, PostgresAdapter)

    def test_create_snowflake_adapter(self):
        """Test creating Snowflake adapter."""
        conn = MagicMock()
        adapter = create_adapter("snowflake", conn)
        assert isinstance(adapter, SnowflakeAdapter)

    def test_create_bigquery_adapter(self):
        """Test creating BigQuery adapter."""
        client = MagicMock()
        adapter = create_adapter("bigquery", client)
        assert isinstance(adapter, BigQueryAdapter)

    def test_create_memory_adapter(self):
        """Test creating in-memory adapter."""
        adapter = create_adapter("memory", None)
        assert isinstance(adapter, InMemoryAdapter)

    def test_unsupported_warehouse_type(self):
        """Test that unsupported warehouse type raises error."""
        with pytest.raises(ValueError, match="Unsupported warehouse type"):
            create_adapter("unsupported_db", MagicMock())


class TestQueryExecutor:
    """Tests for QueryExecutor with real adapters."""

    @pytest.mark.asyncio
    async def test_executor_with_memory_adapter(self):
        """Test QueryExecutor with in-memory adapter."""
        data = {"orders": [{"period": "2024-01", "revenue": 1000}]}
        executor = QueryExecutor(data, warehouse_type="memory")

        result = await executor.execute("SELECT * FROM orders")
        assert len(result) == 1
        assert result[0]["revenue"] == 1000

    @pytest.mark.asyncio
    async def test_executor_with_sqlite(self):
        """Test QueryExecutor with SQLite adapter."""
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE metrics (period TEXT, value REAL)")
        cursor.execute("INSERT INTO metrics VALUES ('2024-01', 100.0)")
        conn.commit()

        executor = QueryExecutor(conn, warehouse_type="sqlite")
        result = await executor.execute("SELECT * FROM metrics")

        assert len(result) == 1
        assert result[0]["period"] == "2024-01"
        assert result[0]["value"] == 100.0

        await executor.close()


class TestAdapterErrorHandling:
    """Tests for error handling in adapters."""

    @pytest.mark.asyncio
    async def test_sqlite_invalid_sql(self):
        """Test SQLite adapter handles invalid SQL."""
        conn = sqlite3.connect(":memory:")
        adapter = SQLiteAdapter(conn)

        with pytest.raises(sqlite3.OperationalError):
            await adapter.execute("SELECT * FROM nonexistent_table")
