"""Semantic Query Engine for translating metric queries to SQL."""

import datetime as _dt
import logging
import math
import re
from typing import Any, Dict, List, Mapping, Optional

from semantic_layer.models import (
    CalculationMethod,
    MetricDefinition,
    MetricQuery,
    QueryResult,
)

logger = logging.getLogger(__name__)

# --- SQL injection hardening helpers -------------------------------------
#
# The warehouse adapters only accept a fully rendered SQL string (no bind
# parameters), so every identifier and literal that reaches generate_sql()
# must be validated or escaped before interpolation.

#: Valid time grains accepted in generated SQL.
VALID_TIME_GRAINS = ("day", "week", "month", "quarter", "year")

#: Operators allowed in filter specifications.
ALLOWED_FILTER_OPERATORS = frozenset({
    "=", "!=", "<>", "<", "<=", ">", ">=",
    "IN", "NOT IN", "LIKE", "NOT LIKE", "IS", "IS NOT",
})

# Simple (optionally dot-qualified) SQL identifier: column, table.column, etc.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")


def validate_identifier(name: Any, context: str = "identifier") -> str:
    """Validate that *name* is a safe SQL identifier and return it.

    Raises ValueError for anything that is not a plain (optionally
    dot-qualified) identifier, preventing SQL injection through field,
    dimension, or column names.
    """
    if not isinstance(name, str) or not _IDENTIFIER_RE.match(name):
        raise ValueError(f"Invalid SQL {context}: {name!r}")
    return name


def validate_iso_date(value: Any, context: str = "date") -> str:
    """Strictly validate an ISO-8601 date/datetime and return a canonical string.

    The returned string is guaranteed to contain only characters produced by
    ``datetime.isoformat()`` and is therefore safe to embed in single quotes.
    """
    if isinstance(value, _dt.datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, _dt.date):
        return value.isoformat()
    if isinstance(value, str):
        try:
            return _dt.date.fromisoformat(value).isoformat()
        except ValueError:
            pass
        try:
            return _dt.datetime.fromisoformat(value).isoformat(sep=" ")
        except ValueError:
            pass
    raise ValueError(f"Invalid ISO-8601 {context}: {value!r}")


def _render_scalar_literal(value: Any) -> str:
    """Render a single scalar Python value as a safely escaped SQL literal."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite numeric literal not allowed: {value!r}")
        return repr(value)
    if isinstance(value, _dt.datetime):
        return f"'{value.isoformat(sep=' ')}'"
    if isinstance(value, _dt.date):
        return f"'{value.isoformat()}'"
    if isinstance(value, str):
        # Standard SQL string escaping: double any embedded single quotes.
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    raise ValueError(f"Unsupported SQL literal type: {type(value).__name__}")


def render_sql_literal(value: Any) -> str:
    """Render a Python value (scalar or flat sequence) as a safe SQL literal."""
    if isinstance(value, (list, tuple, set, frozenset)):
        if not value:
            raise ValueError("Empty value list is not a valid SQL literal")
        return "(" + ", ".join(_render_scalar_literal(v) for v in value) + ")"
    return _render_scalar_literal(value)


def render_filter_condition(filter_spec: Mapping[str, Any]) -> str:
    """Render a filter spec as a safe SQL condition.

    The field is validated as an identifier, the operator is checked against
    an allow-list, and the value is escaped as a SQL literal (never
    interpolated raw).
    """
    field = validate_identifier(filter_spec.get("field"), "filter field")

    operator = str(filter_spec.get("operator", "")).strip().upper()
    if operator not in ALLOWED_FILTER_OPERATORS:
        raise ValueError(f"Unsupported filter operator: {filter_spec.get('operator')!r}")

    value = filter_spec.get("value")
    if operator in ("IN", "NOT IN"):
        if not isinstance(value, (list, tuple, set, frozenset)) or not value:
            raise ValueError(
                f"{operator} filter on {field!r} requires a non-empty list of values"
            )
        rendered = render_sql_literal(value)
    elif operator in ("IS", "IS NOT"):
        if value is not None and not isinstance(value, bool):
            raise ValueError(f"{operator} filter on {field!r} only supports NULL/TRUE/FALSE")
        rendered = _render_scalar_literal(value)
    else:
        rendered = _render_scalar_literal(value)

    return f"{field} {operator} {rendered}"


class MetricCatalog:
    """Catalog of metric definitions."""

    def __init__(self):
        self._metrics: Dict[str, MetricDefinition] = {}
        self._dimensions: Dict[str, Dict[str, Any]] = {}

    def add_metric(self, metric: MetricDefinition) -> None:
        """Add a metric to the catalog."""
        self._metrics[metric.name] = metric
        logger.info(f"Added metric: {metric.name}")

    def get_metric(self, name: str) -> Optional[MetricDefinition]:
        """Get a metric by name."""
        return self._metrics.get(name)

    def list_metrics(
        self,
        category: Optional[str] = None,
        search: Optional[str] = None,
    ) -> List[MetricDefinition]:
        """List metrics with optional filtering."""
        results = list(self._metrics.values())

        if category:
            results = [
                m for m in results
                if m.meta.get("category") == category
            ]

        if search:
            search_lower = search.lower()
            results = [
                m for m in results
                if search_lower in m.name.lower()
                or search_lower in m.description.lower()
            ]

        return results

    def add_dimension(self, name: str, config: Dict[str, Any]) -> None:
        """Add a dimension to the catalog."""
        self._dimensions[name] = config

    def get_dimension(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a dimension by name."""
        return self._dimensions.get(name)

    def list_dimensions(self) -> List[str]:
        """List names of all registered dimensions."""
        return list(self._dimensions.keys())


class SemanticQueryEngine:
    """Engine to translate semantic queries to SQL."""

    def __init__(self, catalog: MetricCatalog, warehouse_type: str = "snowflake"):
        self.catalog = catalog
        self.warehouse_type = warehouse_type

    def generate_sql(self, query: MetricQuery) -> str:
        """Generate SQL from semantic query."""
        # Resolve metric definitions
        metric_specs = []
        for metric_name in query.metrics:
            metric = self.catalog.get_metric(metric_name)
            if not metric:
                raise ValueError(f"Metric not found: {metric_name}")
            metric_specs.append(metric)

        if not metric_specs:
            raise ValueError("No metrics specified")

        # Validate untrusted inputs before any SQL interpolation
        if query.time_grain not in VALID_TIME_GRAINS:
            raise ValueError(f"Invalid time grain: {query.time_grain}")

        dimensions = self._validate_dimensions(query.dimensions, metric_specs)

        # Build SELECT clause
        select_parts = []

        # Time dimension
        time_col = self._get_time_column(metric_specs[0], query.time_grain)
        select_parts.append(f"{time_col} as period")

        # Regular dimensions
        for dim in dimensions:
            select_parts.append(dim)

        # Metrics
        for metric in metric_specs:
            agg_expr = self._build_aggregation(metric)
            select_parts.append(f"{agg_expr} as {metric.name}")

        # Build FROM clause
        from_clause = self._build_from_clause(metric_specs, dimensions)

        # Build WHERE clause
        where_parts = []
        timestamp_col = validate_identifier(
            metric_specs[0].timestamp, "timestamp column"
        )
        start_date = validate_iso_date(query.start_date, "start_date")
        end_date = validate_iso_date(query.end_date, "end_date")
        where_parts.append(f"{timestamp_col} >= '{start_date}'")
        where_parts.append(f"{timestamp_col} < '{end_date}'")

        for filter_spec in query.filters:
            where_parts.append(render_filter_condition(filter_spec))

        # Add metric-level filters
        for metric in metric_specs:
            for f in metric.filters:
                where_parts.append(render_filter_condition(f))

        # Build GROUP BY clause
        group_parts = ["period"] + dimensions

        # Build ORDER BY clause
        order_by = "period"

        # Assemble SQL
        sql = f"""SELECT
    {', '.join(select_parts)}
FROM {from_clause}
WHERE {' AND '.join(where_parts)}
GROUP BY {', '.join(group_parts)}
ORDER BY {order_by}"""

        if query.limit:
            limit = int(query.limit)
            if limit < 0:
                raise ValueError(f"Invalid limit: {query.limit!r}")
            sql += f"\nLIMIT {limit}"
            if query.offset is not None:
                offset = int(query.offset)
                if offset < 0:
                    raise ValueError(f"Invalid offset: {query.offset!r}")
                sql += f" OFFSET {offset}"

        return sql

    def _validate_dimensions(
        self, dimensions: List[str], metric_specs: List[MetricDefinition]
    ) -> List[str]:
        """Validate query dimensions against the registered schema."""
        allowed = set()
        for metric in metric_specs:
            allowed.update(metric.dimensions)
        allowed.update(self.catalog.list_dimensions())

        validated = []
        for dim in dimensions:
            validate_identifier(dim, "dimension")
            if allowed and dim not in allowed:
                raise ValueError(
                    f"Unknown dimension: {dim!r} (not registered for the queried metrics)"
                )
            validated.append(dim)
        return validated

    def _get_time_column(self, metric: MetricDefinition, grain: str) -> str:
        """Get time column with appropriate truncation."""
        if grain not in VALID_TIME_GRAINS:
            raise ValueError(f"Invalid time grain: {grain}")
        timestamp = validate_identifier(metric.timestamp, "timestamp column")

        if self.warehouse_type == "snowflake":
            return f"DATE_TRUNC('{grain}', {timestamp})"
        elif self.warehouse_type == "bigquery":
            return f"DATE_TRUNC({timestamp}, {grain.upper()})"
        elif self.warehouse_type == "postgres":
            return f"DATE_TRUNC('{grain}', {timestamp})"
        elif self.warehouse_type == "redshift":
            return f"DATE_TRUNC('{grain}', {timestamp})"
        else:
            return f"DATE_TRUNC('{grain}', {timestamp})"

    def _build_aggregation(self, metric: MetricDefinition) -> str:
        """Build aggregation expression."""
        method = metric.calculation_method
        expr = metric.expression

        if method == CalculationMethod.SUM:
            return f"SUM({expr})"
        elif method == CalculationMethod.COUNT:
            return f"COUNT({expr})"
        elif method == CalculationMethod.COUNT_DISTINCT:
            return f"COUNT(DISTINCT {expr})"
        elif method == CalculationMethod.AVERAGE:
            return f"AVG({expr})"
        elif method == CalculationMethod.MIN:
            return f"MIN({expr})"
        elif method == CalculationMethod.MAX:
            return f"MAX({expr})"
        elif method == CalculationMethod.DERIVED:
            return self._expand_derived_metric(expr)
        else:
            return expr

    def _expand_derived_metric(self, expression: str) -> str:
        """Expand derived metric references."""
        # Parse metric references like {{ metric('total_revenue') }}
        def replace_metric(match):
            metric_name = match.group(1)
            metric_def = self.catalog.get_metric(metric_name)
            if metric_def:
                return self._build_aggregation(metric_def)
            return metric_name

        pattern = r"\{\{\s*metric\('(\w+)'\)\s*\}\}"
        return re.sub(pattern, replace_metric, expression)

    def _build_from_clause(
        self, metrics: List[MetricDefinition], dimensions: List[str]
    ) -> str:
        """Build FROM clause with necessary joins."""
        # Get all required tables
        tables = set()
        for metric in metrics:
            tables.add(metric.model)

        # For now, assume single table
        # In production, would need to handle joins
        return validate_identifier(list(tables)[0], "table name")

    def validate_query(self, query: MetricQuery) -> List[str]:
        """Validate a metric query and return errors."""
        errors = []

        # Check metrics exist
        for metric_name in query.metrics:
            if not self.catalog.get_metric(metric_name):
                errors.append(f"Metric not found: {metric_name}")

        # Check dimensions are valid
        for dim in query.dimensions:
            # Could check against catalog dimensions
            pass

        # Check time grain
        valid_grains = ["day", "week", "month", "quarter", "year"]
        if query.time_grain not in valid_grains:
            errors.append(f"Invalid time grain: {query.time_grain}")

        # Check dates (strict ISO-8601)
        try:
            start = validate_iso_date(query.start_date, "start_date")
            end = validate_iso_date(query.end_date, "end_date")
        except ValueError as e:
            errors.append(str(e))
        else:
            if start >= end:
                errors.append("start_date must be before end_date")

        return errors


class WarehouseAdapter:
    """Abstract base for warehouse-specific adapters."""

    async def execute(self, sql: str) -> List[Dict[str, Any]]:
        """Execute SQL and return results as list of dicts."""
        raise NotImplementedError("Subclass must implement execute()")

    async def close(self) -> None:
        """Close the connection."""
        pass


class SQLiteAdapter(WarehouseAdapter):
    """SQLite adapter for testing and development."""

    def __init__(self, connection: Any):
        self._connection = connection

    async def execute(self, sql: str) -> List[Dict[str, Any]]:
        """Execute SQL against SQLite database."""
        import sqlite3

        logger.info(f"Executing SQLite query: {sql[:100]}...")

        cursor = self._connection.cursor()
        try:
            cursor.execute(sql)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            return [dict(zip(columns, row)) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"SQLite execution error: {e}")
            raise
        finally:
            cursor.close()

    async def close(self) -> None:
        """Close SQLite connection."""
        if self._connection:
            self._connection.close()


class PostgresAdapter(WarehouseAdapter):
    """PostgreSQL/Redshift adapter."""

    def __init__(self, connection: Any):
        self._connection = connection

    async def execute(self, sql: str) -> List[Dict[str, Any]]:
        """Execute SQL against PostgreSQL."""
        logger.info(f"Executing PostgreSQL query: {sql[:100]}...")

        cursor = self._connection.cursor()
        try:
            cursor.execute(sql)
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                return [dict(zip(columns, row)) for row in rows]
            return []
        except Exception as e:
            logger.error(f"PostgreSQL execution error: {e}")
            raise
        finally:
            cursor.close()

    async def close(self) -> None:
        """Close PostgreSQL connection."""
        if self._connection:
            self._connection.close()


class SnowflakeAdapter(WarehouseAdapter):
    """Snowflake adapter."""

    def __init__(self, connection: Any):
        self._connection = connection

    async def execute(self, sql: str) -> List[Dict[str, Any]]:
        """Execute SQL against Snowflake."""
        logger.info(f"Executing Snowflake query: {sql[:100]}...")

        cursor = self._connection.cursor()
        try:
            cursor.execute(sql)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.error(f"Snowflake execution error: {e}")
            raise
        finally:
            cursor.close()

    async def close(self) -> None:
        """Close Snowflake connection."""
        if self._connection:
            self._connection.close()


class BigQueryAdapter(WarehouseAdapter):
    """Google BigQuery adapter."""

    def __init__(self, client: Any):
        self._client = client

    async def execute(self, sql: str) -> List[Dict[str, Any]]:
        """Execute SQL against BigQuery."""
        logger.info(f"Executing BigQuery query: {sql[:100]}...")

        try:
            query_job = self._client.query(sql)
            results = query_job.result()
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"BigQuery execution error: {e}")
            raise

    async def close(self) -> None:
        """Close BigQuery client."""
        if self._client:
            self._client.close()


class InMemoryAdapter(WarehouseAdapter):
    """In-memory adapter for testing with pre-loaded data."""

    def __init__(self, data: Optional[Dict[str, List[Dict[str, Any]]]] = None):
        self._data = data or {}

    def add_table(self, table_name: str, rows: List[Dict[str, Any]]) -> None:
        """Add data for a table."""
        self._data[table_name] = rows

    async def execute(self, sql: str) -> List[Dict[str, Any]]:
        """Execute simple queries against in-memory data.

        NOTE: This is a simplified implementation for testing.
        It handles basic SELECT queries but not complex SQL.
        """
        logger.info(f"Executing in-memory query: {sql[:100]}...")

        # Simple table extraction from SQL
        sql_lower = sql.lower()
        if "from" in sql_lower:
            # Extract table name
            from_idx = sql_lower.find("from")
            rest = sql[from_idx + 5:].strip()
            table_name = rest.split()[0].strip()

            if table_name in self._data:
                return self._data[table_name]

        return []


def create_adapter(warehouse_type: str, connection: Any) -> WarehouseAdapter:
    """Factory function to create the appropriate adapter.

    Args:
        warehouse_type: Type of warehouse (sqlite, postgres, redshift, snowflake, bigquery, memory)
        connection: Connection object or client for the warehouse

    Returns:
        WarehouseAdapter instance
    """
    adapters = {
        "sqlite": SQLiteAdapter,
        "postgres": PostgresAdapter,
        "postgresql": PostgresAdapter,
        "redshift": PostgresAdapter,  # Redshift is PostgreSQL-compatible
        "snowflake": SnowflakeAdapter,
        "bigquery": BigQueryAdapter,
        "memory": InMemoryAdapter,
    }

    adapter_class = adapters.get(warehouse_type.lower())
    if not adapter_class:
        raise ValueError(f"Unsupported warehouse type: {warehouse_type}")

    return adapter_class(connection)


class QueryExecutor:
    """Execute queries against the warehouse."""

    def __init__(self, connection: Any, warehouse_type: str = "memory"):
        """Initialize QueryExecutor.

        Args:
            connection: Database connection or client object.
                       For 'memory' type, can be None or Dict[str, List[Dict]].
            warehouse_type: Type of warehouse (sqlite, postgres, snowflake, bigquery, memory)
        """
        self._connection = connection
        self._warehouse_type = warehouse_type

        # Create the appropriate adapter
        if warehouse_type.lower() == "memory":
            self._adapter = InMemoryAdapter(connection if isinstance(connection, dict) else None)
        else:
            self._adapter = create_adapter(warehouse_type, connection)

    async def execute(self, sql: str) -> List[Dict[str, Any]]:
        """Execute SQL and return results."""
        return await self._adapter.execute(sql)

    async def execute_metric_query(
        self, engine: SemanticQueryEngine, query: MetricQuery
    ) -> QueryResult:
        """Execute a metric query."""
        # Validate query
        errors = engine.validate_query(query)
        if errors:
            raise ValueError(f"Query validation failed: {errors}")

        # Generate SQL
        sql = engine.generate_sql(query)

        # Execute
        data = await self.execute(sql)

        return QueryResult(
            data=data,
            metadata={
                "metrics": query.metrics,
                "dimensions": query.dimensions,
                "time_grain": query.time_grain,
            },
            sql=sql,
            row_count=len(data),
        )

    async def close(self) -> None:
        """Close the underlying connection."""
        await self._adapter.close()


# Pre-built metric definitions for common use cases
def create_revenue_metric(model: str, amount_column: str, timestamp_column: str) -> MetricDefinition:
    """Create a standard revenue metric."""
    return MetricDefinition(
        name="total_revenue",
        label="Total Revenue",
        description="Sum of all revenue",
        model=model,
        calculation_method=CalculationMethod.SUM,
        expression=amount_column,
        timestamp=timestamp_column,
        time_grains=[
            grain for grain in [
                "day", "week", "month", "quarter", "year"
            ]
        ],
        dimensions=["customer_segment", "country_code", "payment_method"],
        meta={"owner": "finance-team", "tier": 1},
    )


def create_user_metric(model: str, user_column: str, timestamp_column: str) -> MetricDefinition:
    """Create a standard active users metric."""
    return MetricDefinition(
        name="active_users",
        label="Active Users",
        description="Count of distinct active users",
        model=model,
        calculation_method=CalculationMethod.COUNT_DISTINCT,
        expression=user_column,
        timestamp=timestamp_column,
        time_grains=["day", "week", "month"],
        dimensions=["country_code", "customer_segment"],
        meta={"owner": "product-team", "tier": 1},
    )
