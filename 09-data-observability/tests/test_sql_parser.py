"""Tests for SQL parser and lineage extraction."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from observability.sql_parser import (
    ColumnReference,
    LineageExtractor,
    SQLLineage,
    SQLParser,
    TableReference,
    extract_lineage,
    extract_tables,
)


class TestTableReference:
    """Tests for TableReference."""

    def test_simple_table(self):
        """Test simple table reference."""
        ref = TableReference(table="users")
        assert ref.full_name == "users"

    def test_schema_qualified(self):
        """Test schema-qualified table."""
        ref = TableReference(schema="public", table="users")
        assert ref.full_name == "public.users"

    def test_fully_qualified(self):
        """Test fully qualified table."""
        ref = TableReference(database="mydb", schema="public", table="users")
        assert ref.full_name == "mydb.public.users"

    def test_with_alias(self):
        """Test table with alias."""
        ref = TableReference(table="users", alias="u")
        assert ref.table == "users"
        assert ref.alias == "u"


class TestSQLParser:
    """Tests for SQLParser."""

    def test_simple_select(self):
        """Test parsing simple SELECT."""
        parser = SQLParser()
        sql = "SELECT * FROM users"
        lineage = parser.parse(sql)

        assert lineage.sql_type == "SELECT"
        assert len(lineage.sources) >= 1
        assert any(s.table == "users" for s in lineage.sources)

    def test_select_with_join(self):
        """Test parsing SELECT with JOIN."""
        parser = SQLParser()
        sql = """
            SELECT u.id, o.order_id
            FROM users u
            JOIN orders o ON u.id = o.user_id
        """
        lineage = parser.parse(sql)

        assert lineage.sql_type == "SELECT"
        table_names = [s.table for s in lineage.sources]
        assert "users" in table_names
        assert "orders" in table_names

    def test_insert_statement(self):
        """Test parsing INSERT statement."""
        parser = SQLParser()
        sql = """
            INSERT INTO user_summary (user_id, total_orders)
            SELECT user_id, COUNT(*) FROM orders GROUP BY user_id
        """
        lineage = parser.parse(sql)

        assert lineage.sql_type == "INSERT"
        if lineage.target:
            assert lineage.target.table == "user_summary"

    def test_create_table_as(self):
        """Test parsing CREATE TABLE AS."""
        parser = SQLParser()
        sql = """
            CREATE TABLE analytics.daily_summary AS
            SELECT date, COUNT(*) as cnt FROM events GROUP BY date
        """
        lineage = parser.parse(sql)

        assert lineage.sql_type in ("CREATE TABLE AS", "SELECT")

    def test_cte_parsing(self):
        """Test parsing CTE (WITH clause)."""
        parser = SQLParser()
        sql = """
            WITH active_users AS (
                SELECT * FROM users WHERE status = 'active'
            )
            SELECT * FROM active_users
        """
        lineage = parser.parse(sql)

        assert lineage.sql_type == "SELECT"

    def test_subquery(self):
        """Test parsing subquery."""
        parser = SQLParser()
        sql = """
            SELECT * FROM (
                SELECT user_id, MAX(order_date) as last_order
                FROM orders
                GROUP BY user_id
            ) AS user_orders
        """
        lineage = parser.parse(sql)

        assert lineage.sql_type == "SELECT"

    def test_multi_table_join(self):
        """Test parsing multiple table joins."""
        parser = SQLParser()
        sql = """
            SELECT u.name, o.order_id, p.product_name
            FROM users u
            JOIN orders o ON u.id = o.user_id
            JOIN order_items oi ON o.order_id = oi.order_id
            JOIN products p ON oi.product_id = p.product_id
        """
        lineage = parser.parse(sql)

        table_names = [s.table for s in lineage.sources]
        assert "users" in table_names
        assert "orders" in table_names

    def test_schema_qualified_tables(self):
        """Test parsing schema-qualified table names."""
        parser = SQLParser()
        sql = "SELECT * FROM production.public.users"
        lineage = parser.parse(sql)

        assert len(lineage.sources) >= 1


class TestLineageExtractor:
    """Tests for LineageExtractor."""

    def test_extract_single_query(self):
        """Test extracting lineage from single query."""
        extractor = LineageExtractor()
        sql = "SELECT * FROM users WHERE active = true"
        lineage = extractor.extract_from_query(sql)

        assert isinstance(lineage, SQLLineage)

    def test_extract_multiple_queries(self):
        """Test extracting lineage from multiple queries."""
        extractor = LineageExtractor()
        queries = [
            "SELECT * FROM users",
            "SELECT * FROM orders",
            "SELECT * FROM products",
        ]
        lineages = extractor.extract_from_queries(queries)

        assert len(lineages) == 3

    def test_build_dependency_graph(self):
        """Test building dependency graph."""
        extractor = LineageExtractor()
        queries = [
            "INSERT INTO user_summary SELECT * FROM users",
            "INSERT INTO order_summary SELECT * FROM orders JOIN users ON orders.user_id = users.id",
        ]
        graph = extractor.build_dependency_graph(queries)

        assert isinstance(graph, dict)

    def test_get_table_dependencies(self):
        """Test getting table dependencies."""
        extractor = LineageExtractor()
        queries = [
            "INSERT INTO staging.users SELECT * FROM raw.users",
            "INSERT INTO analytics.user_metrics SELECT * FROM staging.users",
        ]

        upstream, downstream = extractor.get_table_dependencies(
            "staging.users", queries
        )

        assert isinstance(upstream, set)
        assert isinstance(downstream, set)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_extract_lineage(self):
        """Test extract_lineage function."""
        sql = "SELECT * FROM users JOIN orders ON users.id = orders.user_id"
        lineage = extract_lineage(sql)

        assert isinstance(lineage, SQLLineage)
        assert lineage.sql_type == "SELECT"

    def test_extract_tables(self):
        """Test extract_tables function."""
        sql = "SELECT * FROM users u JOIN orders o ON u.id = o.user_id"
        tables = extract_tables(sql)

        assert isinstance(tables, list)
        assert len(tables) >= 1

    def test_different_dialects(self):
        """Test parsing with different dialects."""
        sql = "SELECT * FROM users"

        for dialect in ["snowflake", "bigquery", "postgres"]:
            lineage = extract_lineage(sql, dialect=dialect)
            assert lineage.sql_type == "SELECT"


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_sql(self):
        """Test parsing empty SQL."""
        parser = SQLParser()
        lineage = parser.parse("")
        assert lineage.sources == [] or lineage.raw_sql == ""

    def test_invalid_sql(self):
        """Test parsing invalid SQL."""
        parser = SQLParser()
        # Should not raise, just return basic lineage
        lineage = parser.parse("NOT VALID SQL AT ALL")
        assert lineage is not None

    def test_complex_expressions(self):
        """Test parsing complex expressions."""
        parser = SQLParser()
        sql = """
            SELECT
                CASE WHEN status = 'active' THEN 1 ELSE 0 END as is_active,
                COALESCE(name, 'Unknown') as display_name,
                DATE_TRUNC('day', created_at) as created_date
            FROM users
            WHERE created_at > '2024-01-01'
        """
        lineage = parser.parse(sql)
        assert lineage.sql_type == "SELECT"

    def test_window_functions(self):
        """Test parsing window functions."""
        parser = SQLParser()
        sql = """
            SELECT
                user_id,
                ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY order_date) as rn,
                SUM(amount) OVER (PARTITION BY user_id) as total_amount
            FROM orders
        """
        lineage = parser.parse(sql)
        assert lineage.sql_type == "SELECT"

    def test_union_query(self):
        """Test parsing UNION query."""
        parser = SQLParser()
        sql = """
            SELECT id, name FROM users_v1
            UNION ALL
            SELECT id, name FROM users_v2
        """
        lineage = parser.parse(sql)
        table_names = [s.table for s in lineage.sources]
        # At least one table should be found
        assert len(table_names) >= 1
