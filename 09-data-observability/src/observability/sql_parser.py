"""SQL parsing for lineage extraction.

Uses sqlglot for parsing SQL queries to extract table references,
column dependencies, and data flow information.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Try to import sqlglot, fall back to regex-based parsing if not available
try:
    import sqlglot
    from sqlglot import exp
    SQLGLOT_AVAILABLE = True
except ImportError:
    SQLGLOT_AVAILABLE = False
    logger.warning("sqlglot not installed, using basic regex parsing")

import re


@dataclass
class TableReference:
    """Reference to a table in SQL."""

    database: Optional[str] = None
    schema: Optional[str] = None
    table: str = ""
    alias: Optional[str] = None

    @property
    def full_name(self) -> str:
        """Get fully qualified table name."""
        parts = []
        if self.database:
            parts.append(self.database)
        if self.schema:
            parts.append(self.schema)
        parts.append(self.table)
        return ".".join(parts)


@dataclass
class ColumnReference:
    """Reference to a column in SQL."""

    table: Optional[str] = None
    column: str = ""
    alias: Optional[str] = None


@dataclass
class SQLLineage:
    """Lineage information extracted from SQL."""

    sources: List[TableReference] = field(default_factory=list)
    target: Optional[TableReference] = None
    column_mappings: List[Dict[str, str]] = field(default_factory=list)
    sql_type: str = "SELECT"
    raw_sql: str = ""


class SQLParser:
    """Parser for extracting lineage from SQL statements."""

    def __init__(self, dialect: str = "snowflake"):
        """Initialize the parser.

        Args:
            dialect: SQL dialect (snowflake, bigquery, postgres, etc.)
        """
        self.dialect = dialect

    def parse(self, sql: str) -> SQLLineage:
        """Parse SQL and extract lineage information.

        Args:
            sql: SQL statement to parse

        Returns:
            SQLLineage with extracted information
        """
        if SQLGLOT_AVAILABLE:
            return self._parse_with_sqlglot(sql)
        else:
            return self._parse_with_regex(sql)

    def _parse_with_sqlglot(self, sql: str) -> SQLLineage:
        """Parse SQL using sqlglot library."""
        lineage = SQLLineage(raw_sql=sql)

        try:
            parsed = sqlglot.parse_one(sql, read=self.dialect)
        except Exception as e:
            logger.warning(f"Failed to parse SQL: {e}")
            return self._parse_with_regex(sql)

        # Determine SQL type
        if isinstance(parsed, exp.Insert):
            lineage.sql_type = "INSERT"
            lineage.target = self._extract_target_table(parsed)
            lineage.sources = self._extract_source_tables(parsed)
            lineage.column_mappings = self._extract_insert_mappings(parsed)
        elif isinstance(parsed, exp.Create):
            if parsed.find(exp.Table):
                lineage.sql_type = "CREATE TABLE AS"
                lineage.target = self._extract_target_table(parsed)
                # Get sources from the SELECT part
                select = parsed.find(exp.Select)
                if select:
                    lineage.sources = self._extract_source_tables(select)
                    lineage.column_mappings = self._extract_select_mappings(select)
        elif isinstance(parsed, exp.Select):
            lineage.sql_type = "SELECT"
            lineage.sources = self._extract_source_tables(parsed)
            lineage.column_mappings = self._extract_select_mappings(parsed)
        elif isinstance(parsed, exp.Merge):
            lineage.sql_type = "MERGE"
            lineage.target = self._extract_target_table(parsed)
            lineage.sources = self._extract_source_tables(parsed)
        elif isinstance(parsed, exp.Update):
            lineage.sql_type = "UPDATE"
            lineage.target = self._extract_target_table(parsed)
            lineage.sources = self._extract_source_tables(parsed)

        return lineage

    def _extract_target_table(self, parsed) -> Optional[TableReference]:
        """Extract target table from parsed SQL."""
        if not SQLGLOT_AVAILABLE:
            return None

        # Find the first Table expression (target for INSERT/CREATE/UPDATE)
        if isinstance(parsed, exp.Insert):
            table = parsed.find(exp.Table)
        elif isinstance(parsed, exp.Create):
            # For CREATE, get the table being created
            table = parsed.this if isinstance(parsed.this, exp.Table) else parsed.find(exp.Table)
        elif isinstance(parsed, exp.Update):
            table = parsed.this if isinstance(parsed.this, exp.Table) else parsed.find(exp.Table)
        elif isinstance(parsed, exp.Merge):
            table = parsed.this if isinstance(parsed.this, exp.Table) else None
        else:
            table = None

        if table:
            return self._table_to_reference(table)
        return None

    def _extract_source_tables(self, parsed) -> List[TableReference]:
        """Extract source tables from parsed SQL."""
        if not SQLGLOT_AVAILABLE:
            return []

        sources = []

        # Find all FROM and JOIN clauses
        for table in parsed.find_all(exp.Table):
            # Skip if this is the target table in INSERT
            if isinstance(parsed, exp.Insert):
                if table == parsed.find(exp.Table):
                    continue

            ref = self._table_to_reference(table)
            if ref and ref.table:
                sources.append(ref)

        return sources

    def _table_to_reference(self, table) -> TableReference:
        """Convert sqlglot Table to TableReference."""
        ref = TableReference(table=table.name)

        if table.db:
            ref.schema = table.db
        if table.catalog:
            ref.database = table.catalog
        if table.alias:
            ref.alias = table.alias

        return ref

    def _extract_select_mappings(self, parsed) -> List[Dict[str, str]]:
        """Extract column mappings from SELECT clause."""
        if not SQLGLOT_AVAILABLE:
            return []

        mappings = []

        # Get SELECT expressions
        for select_expr in parsed.find_all(exp.Select):
            for expr in select_expr.expressions:
                mapping = self._extract_column_mapping(expr)
                if mapping:
                    mappings.append(mapping)

        return mappings

    def _extract_insert_mappings(self, parsed) -> List[Dict[str, str]]:
        """Extract column mappings from INSERT statement."""
        if not SQLGLOT_AVAILABLE:
            return []

        mappings = []

        # Get INSERT columns
        columns = parsed.find(exp.Schema)
        select = parsed.find(exp.Select)

        if columns and select:
            col_names = [col.name for col in columns.find_all(exp.Column)]
            select_exprs = list(select.expressions)

            for i, (col, expr) in enumerate(zip(col_names, select_exprs)):
                source_col = self._extract_source_column(expr)
                if source_col:
                    mappings.append({
                        "source": source_col,
                        "target": col
                    })

        return mappings

    def _extract_column_mapping(self, expr) -> Optional[Dict[str, str]]:
        """Extract a single column mapping from expression."""
        if not SQLGLOT_AVAILABLE:
            return None

        # Get target name (alias or column name)
        if isinstance(expr, exp.Alias):
            target = expr.alias
            source = self._extract_source_column(expr.this)
        elif isinstance(expr, exp.Column):
            target = expr.name
            source = expr.name
            if expr.table:
                source = f"{expr.table}.{source}"
        else:
            return None

        if source and target:
            return {"source": source, "target": target}
        return None

    def _extract_source_column(self, expr) -> Optional[str]:
        """Extract source column name from expression."""
        if not SQLGLOT_AVAILABLE:
            return None

        if isinstance(expr, exp.Column):
            if expr.table:
                return f"{expr.table}.{expr.name}"
            return expr.name
        elif isinstance(expr, exp.Alias):
            return self._extract_source_column(expr.this)

        # For complex expressions, try to find a column
        col = expr.find(exp.Column) if hasattr(expr, 'find') else None
        if col:
            if col.table:
                return f"{col.table}.{col.name}"
            return col.name

        return None

    def _parse_with_regex(self, sql: str) -> SQLLineage:
        """Fallback regex-based parsing."""
        lineage = SQLLineage(raw_sql=sql)
        sql_upper = sql.upper()

        # Determine SQL type
        if sql_upper.strip().startswith("INSERT"):
            lineage.sql_type = "INSERT"
        elif sql_upper.strip().startswith("CREATE"):
            lineage.sql_type = "CREATE TABLE AS"
        elif sql_upper.strip().startswith("MERGE"):
            lineage.sql_type = "MERGE"
        elif sql_upper.strip().startswith("UPDATE"):
            lineage.sql_type = "UPDATE"
        else:
            lineage.sql_type = "SELECT"

        # Extract tables using regex
        lineage.sources = self._extract_tables_regex(sql)

        # For INSERT/CREATE, first table is typically the target
        if lineage.sql_type in ("INSERT", "CREATE TABLE AS", "MERGE", "UPDATE"):
            if lineage.sources:
                lineage.target = lineage.sources.pop(0)

        return lineage

    def _extract_tables_regex(self, sql: str) -> List[TableReference]:
        """Extract table names using regex patterns."""
        tables = []

        # Pattern for table names (handles schema.table and database.schema.table)
        table_pattern = r'(?:FROM|JOIN|INTO|UPDATE|TABLE)\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*){0,2})'

        for match in re.finditer(table_pattern, sql, re.IGNORECASE):
            table_name = match.group(1)
            parts = table_name.split(".")

            ref = TableReference(table=parts[-1])
            if len(parts) >= 2:
                ref.schema = parts[-2]
            if len(parts) >= 3:
                ref.database = parts[-3]

            tables.append(ref)

        return tables


class LineageExtractor:
    """High-level lineage extraction from SQL workloads."""

    def __init__(self, dialect: str = "snowflake"):
        """Initialize the extractor.

        Args:
            dialect: SQL dialect for parsing
        """
        self.parser = SQLParser(dialect)
        self._table_aliases: Dict[str, str] = {}

    def extract_from_query(self, sql: str) -> SQLLineage:
        """Extract lineage from a single SQL query.

        Args:
            sql: SQL query string

        Returns:
            SQLLineage with extracted dependencies
        """
        return self.parser.parse(sql)

    def extract_from_queries(self, queries: List[str]) -> List[SQLLineage]:
        """Extract lineage from multiple SQL queries.

        Args:
            queries: List of SQL query strings

        Returns:
            List of SQLLineage objects
        """
        return [self.extract_from_query(q) for q in queries]

    def build_dependency_graph(
        self,
        queries: List[str]
    ) -> Dict[str, Set[str]]:
        """Build a dependency graph from multiple queries.

        Args:
            queries: List of SQL queries

        Returns:
            Dictionary mapping target tables to their source tables
        """
        deps: Dict[str, Set[str]] = {}

        for query in queries:
            lineage = self.extract_from_query(query)

            if lineage.target:
                target = lineage.target.full_name
                if target not in deps:
                    deps[target] = set()

                for source in lineage.sources:
                    deps[target].add(source.full_name)

        return deps

    def get_table_dependencies(
        self,
        table_name: str,
        queries: List[str]
    ) -> Tuple[Set[str], Set[str]]:
        """Get upstream and downstream dependencies for a table.

        Args:
            table_name: The table to analyze
            queries: List of SQL queries defining the data flow

        Returns:
            Tuple of (upstream tables, downstream tables)
        """
        upstream: Set[str] = set()
        downstream: Set[str] = set()

        for query in queries:
            lineage = self.extract_from_query(query)

            # Check if this table is the target
            if lineage.target and table_name in lineage.target.full_name:
                for source in lineage.sources:
                    upstream.add(source.full_name)

            # Check if this table is a source
            for source in lineage.sources:
                if table_name in source.full_name:
                    if lineage.target:
                        downstream.add(lineage.target.full_name)

        return upstream, downstream


def extract_lineage(sql: str, dialect: str = "snowflake") -> SQLLineage:
    """Convenience function to extract lineage from SQL.

    Args:
        sql: SQL query string
        dialect: SQL dialect

    Returns:
        SQLLineage with extracted information
    """
    parser = SQLParser(dialect)
    return parser.parse(sql)


def extract_tables(sql: str, dialect: str = "snowflake") -> List[str]:
    """Extract all table names from SQL.

    Args:
        sql: SQL query string
        dialect: SQL dialect

    Returns:
        List of fully qualified table names
    """
    lineage = extract_lineage(sql, dialect)
    tables = [s.full_name for s in lineage.sources]
    if lineage.target:
        tables.append(lineage.target.full_name)
    return tables
