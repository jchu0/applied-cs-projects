"""Metadata collectors for various data sources."""

import asyncio
import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, AsyncIterator

from observability.config import CollectorConfig, WarehouseConfig
from observability.models import (
    ColumnMetadata,
    ColumnStats,
    LineageInfo,
    TableMetadata,
)

logger = logging.getLogger(__name__)

# Optional imports for database connectors
try:
    import asyncpg
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False
    asyncpg = None

try:
    import snowflake.connector
    HAS_SNOWFLAKE = True
except ImportError:
    HAS_SNOWFLAKE = False

try:
    from google.cloud import bigquery
    HAS_BIGQUERY = True
except ImportError:
    HAS_BIGQUERY = False
    bigquery = None

try:
    from databricks import sql as databricks_sql
    HAS_DATABRICKS = True
except ImportError:
    HAS_DATABRICKS = False
    databricks_sql = None


class MetadataCollector(ABC):
    """Base class for metadata collectors."""

    def __init__(self, config: CollectorConfig):
        self.config = config

    @abstractmethod
    async def collect_schema(self, table_ref: str) -> TableMetadata:
        """Collect schema metadata for a table."""
        pass

    @abstractmethod
    async def collect_stats(self, table_ref: str) -> Dict[str, ColumnStats]:
        """Collect column statistics for a table."""
        pass

    @abstractmethod
    async def collect_lineage(self, table_ref: str) -> LineageInfo:
        """Collect lineage information for a table."""
        pass

    @abstractmethod
    async def get_column_sample(
        self, table_ref: str, column_name: str, sample_size: int
    ) -> List[Any]:
        """Get a sample of values from a column."""
        pass


class InMemoryCollector(MetadataCollector):
    """In-memory collector for testing and development."""

    def __init__(self, config: CollectorConfig):
        super().__init__(config)
        self._tables: Dict[str, TableMetadata] = {}
        self._stats: Dict[str, Dict[str, ColumnStats]] = {}
        self._lineage: Dict[str, LineageInfo] = {}
        self._samples: Dict[str, Dict[str, List[Any]]] = {}

    def add_table(
        self,
        metadata: TableMetadata,
        stats: Optional[Dict[str, ColumnStats]] = None,
        lineage: Optional[LineageInfo] = None,
    ) -> None:
        """Add a table to the in-memory store."""
        self._tables[metadata.table_id] = metadata
        if stats:
            self._stats[metadata.table_id] = stats
        if lineage:
            self._lineage[metadata.table_id] = lineage

    def add_sample(
        self, table_ref: str, column_name: str, sample: List[Any]
    ) -> None:
        """Add sample data for a column."""
        if table_ref not in self._samples:
            self._samples[table_ref] = {}
        self._samples[table_ref][column_name] = sample

    async def collect_schema(self, table_ref: str) -> TableMetadata:
        """Collect schema metadata for a table."""
        if table_ref not in self._tables:
            raise ValueError(f"Table {table_ref} not found")
        return self._tables[table_ref]

    async def collect_stats(self, table_ref: str) -> Dict[str, ColumnStats]:
        """Collect column statistics for a table."""
        return self._stats.get(table_ref, {})

    async def collect_lineage(self, table_ref: str) -> LineageInfo:
        """Collect lineage information for a table."""
        return self._lineage.get(
            table_ref,
            LineageInfo(table_id=table_ref),
        )

    async def get_column_sample(
        self, table_ref: str, column_name: str, sample_size: int
    ) -> List[Any]:
        """Get a sample of values from a column."""
        if table_ref in self._samples and column_name in self._samples[table_ref]:
            sample = self._samples[table_ref][column_name]
            return sample[:sample_size]
        return []


class GenericSQLCollector(MetadataCollector):
    """Generic SQL-based collector for relational databases."""

    def __init__(
        self,
        config: CollectorConfig,
        warehouse_config: WarehouseConfig,
    ):
        super().__init__(config)
        self.warehouse_config = warehouse_config
        self._connection = None
        self._pool = None
        self._connected = False

    async def connect(self) -> None:
        """Establish database connection."""
        raise NotImplementedError("Subclasses must implement connect()")

    async def disconnect(self) -> None:
        """Close database connection."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
        if self._connection is not None:
            if hasattr(self._connection, 'close'):
                close_result = self._connection.close()
                if asyncio.iscoroutine(close_result):
                    await close_result
            self._connection = None
        self._connected = False

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[Any]:
        """Context manager for database connection."""
        if not self._connected:
            await self.connect()
        try:
            yield self._connection
        finally:
            pass  # Keep connection open for reuse

    async def _get_connection(self) -> Any:
        """Get database connection."""
        if not self._connected:
            await self.connect()
        return self._connection

    async def _execute(self, query: str) -> List[Dict[str, Any]]:
        """Execute a query and return results."""
        logger.debug(f"Executing query: {query[:100]}...")
        conn = await self._get_connection()
        if conn is None:
            logger.warning("No database connection available")
            return []
        return await self._execute_with_connection(conn, query)

    async def _execute_with_connection(
        self, conn: Any, query: str
    ) -> List[Dict[str, Any]]:
        """Execute query with the given connection. Override in subclasses."""
        return []

    async def collect_schema(self, table_ref: str) -> TableMetadata:
        """Collect schema metadata for a table."""
        parts = table_ref.split(".")
        if len(parts) != 3:
            raise ValueError(f"Invalid table reference: {table_ref}")

        database, schema, table = parts

        # Get columns from information schema
        columns_query = f"""
            SELECT
                column_name,
                data_type,
                is_nullable,
                column_default
            FROM information_schema.columns
            WHERE table_catalog = '{database}'
              AND table_schema = '{schema}'
              AND table_name = '{table}'
            ORDER BY ordinal_position
        """
        columns_result = await self._execute(columns_query)

        # Get table stats
        stats_query = f"""
            SELECT
                COUNT(*) as row_count
            FROM {table_ref}
        """
        stats_result = await self._execute(stats_query)

        columns = [
            ColumnMetadata(
                name=row.get("column_name", ""),
                data_type=row.get("data_type", ""),
                nullable=row.get("is_nullable", "YES") == "YES",
                description=None,
                stats=None,
            )
            for row in columns_result
        ]

        row_count = stats_result[0]["row_count"] if stats_result else 0

        return TableMetadata(
            table_id=table_ref,
            database=database,
            schema=schema,
            table_name=table,
            columns=columns,
            row_count=row_count,
            size_bytes=0,
            last_modified=datetime.now(),
        )

    async def collect_stats(self, table_ref: str) -> Dict[str, ColumnStats]:
        """Collect column statistics for a table."""
        schema = await self.collect_schema(table_ref)
        stats = {}

        for col in schema.columns:
            col_stats = await self._profile_column(
                table_ref, col.name, col.data_type
            )
            stats[col.name] = col_stats

        return stats

    async def _profile_column(
        self, table_ref: str, column_name: str, data_type: str
    ) -> ColumnStats:
        """Profile a single column."""
        is_numeric = data_type.lower() in (
            "integer", "int", "bigint", "smallint",
            "numeric", "decimal", "float", "double", "real", "number",
        )

        if is_numeric:
            query = f"""
                SELECT
                    COUNT(*) as total_count,
                    SUM(CASE WHEN {column_name} IS NULL THEN 1 ELSE 0 END) as null_count,
                    COUNT(DISTINCT {column_name}) as distinct_count,
                    MIN({column_name}) as min_value,
                    MAX({column_name}) as max_value,
                    AVG({column_name}) as mean,
                    STDDEV({column_name}) as stddev
                FROM {table_ref}
            """
        else:
            query = f"""
                SELECT
                    COUNT(*) as total_count,
                    SUM(CASE WHEN {column_name} IS NULL THEN 1 ELSE 0 END) as null_count,
                    COUNT(DISTINCT {column_name}) as distinct_count,
                    MIN({column_name}) as min_value,
                    MAX({column_name}) as max_value,
                    NULL as mean,
                    NULL as stddev
                FROM {table_ref}
            """

        result = await self._execute(query)

        if not result:
            return ColumnStats(
                null_count=0,
                null_ratio=0.0,
                distinct_count=0,
            )

        row = result[0]
        total = row.get("total_count", 1)
        null_count = row.get("null_count", 0)

        return ColumnStats(
            null_count=null_count,
            null_ratio=null_count / total if total > 0 else 0,
            distinct_count=row.get("distinct_count", 0),
            min_value=row.get("min_value"),
            max_value=row.get("max_value"),
            mean=float(row["mean"]) if row.get("mean") else None,
            stddev=float(row["stddev"]) if row.get("stddev") else None,
        )

    async def collect_lineage(self, table_ref: str) -> LineageInfo:
        """Collect lineage information for a table."""
        # Basic implementation - would use query logs in production
        return LineageInfo(table_id=table_ref)

    async def get_column_sample(
        self, table_ref: str, column_name: str, sample_size: int
    ) -> List[Any]:
        """Get a sample of values from a column."""
        query = f"""
            SELECT {column_name}
            FROM {table_ref}
            WHERE {column_name} IS NOT NULL
            LIMIT {sample_size}
        """
        result = await self._execute(query)
        return [row[column_name] for row in result]


class SnowflakeCollector(GenericSQLCollector):
    """Collector for Snowflake data warehouse."""

    def __init__(
        self,
        config: CollectorConfig,
        warehouse_config: WarehouseConfig,
    ):
        super().__init__(config, warehouse_config)
        if not HAS_SNOWFLAKE:
            logger.warning(
                "snowflake-connector-python not installed. "
                "Install with: pip install snowflake-connector-python"
            )

    async def connect(self) -> None:
        """Establish Snowflake connection."""
        if not HAS_SNOWFLAKE:
            raise ImportError("snowflake-connector-python is required")

        loop = asyncio.get_event_loop()
        params = {
            "user": self.warehouse_config.user,
            "password": self.warehouse_config.password,
            "account": self.warehouse_config.host,  # Snowflake account identifier
            "database": self.warehouse_config.database,
            **self.warehouse_config.options,
        }

        # Run synchronous connect in executor
        self._connection = await loop.run_in_executor(
            None,
            lambda: snowflake.connector.connect(**params)
        )
        self._connected = True
        logger.info(f"Connected to Snowflake account: {self.warehouse_config.host}")

    async def _execute_with_connection(
        self, conn: Any, query: str
    ) -> List[Dict[str, Any]]:
        """Execute query with Snowflake connection."""
        loop = asyncio.get_event_loop()

        def _run_query():
            cursor = conn.cursor()
            try:
                cursor.execute(query)
                columns = [desc[0].lower() for desc in cursor.description or []]
                rows = cursor.fetchall()
                return [dict(zip(columns, row)) for row in rows]
            finally:
                cursor.close()

        return await loop.run_in_executor(None, _run_query)

    async def collect_schema(self, table_ref: str) -> TableMetadata:
        """Collect schema metadata with Snowflake-specific optimizations."""
        parts = table_ref.split(".")
        if len(parts) != 3:
            raise ValueError(f"Invalid table reference: {table_ref}")

        database, schema, table = parts

        # Use Snowflake's INFORMATION_SCHEMA
        columns_query = f"""
            SELECT
                column_name,
                data_type,
                is_nullable,
                comment
            FROM {database}.information_schema.columns
            WHERE table_schema = '{schema}'
              AND table_name = '{table}'
            ORDER BY ordinal_position
        """
        columns_result = await self._execute(columns_query)

        # Get table metadata from TABLES view
        stats_query = f"""
            SELECT
                row_count,
                bytes,
                last_altered
            FROM {database}.information_schema.tables
            WHERE table_schema = '{schema}'
              AND table_name = '{table}'
        """
        stats_result = await self._execute(stats_query)

        columns = [
            ColumnMetadata(
                name=row.get("column_name", ""),
                data_type=row.get("data_type", ""),
                nullable=row.get("is_nullable", "YES") == "YES",
                description=row.get("comment"),
                stats=None,
            )
            for row in columns_result
        ]

        row_count = 0
        size_bytes = 0
        last_modified = datetime.now()

        if stats_result:
            row_count = stats_result[0].get("row_count", 0) or 0
            size_bytes = stats_result[0].get("bytes", 0) or 0
            last_modified = stats_result[0].get("last_altered", datetime.now())

        return TableMetadata(
            table_id=table_ref,
            database=database,
            schema=schema,
            table_name=table,
            columns=columns,
            row_count=row_count,
            size_bytes=size_bytes,
            last_modified=last_modified,
        )


class BigQueryCollector(GenericSQLCollector):
    """Collector for Google BigQuery."""

    def __init__(
        self,
        config: CollectorConfig,
        warehouse_config: WarehouseConfig,
    ):
        super().__init__(config, warehouse_config)
        self._client = None
        if not HAS_BIGQUERY:
            logger.warning(
                "google-cloud-bigquery not installed. "
                "Install with: pip install google-cloud-bigquery"
            )

    async def connect(self) -> None:
        """Establish BigQuery connection."""
        if not HAS_BIGQUERY:
            raise ImportError("google-cloud-bigquery is required")

        loop = asyncio.get_event_loop()

        def _create_client():
            # Uses default credentials or explicit project
            project = self.warehouse_config.options.get(
                "project", self.warehouse_config.database
            )
            return bigquery.Client(project=project)

        self._client = await loop.run_in_executor(None, _create_client)
        self._connection = self._client
        self._connected = True
        logger.info(f"Connected to BigQuery project: {self._client.project}")

    async def _execute_with_connection(
        self, conn: Any, query: str
    ) -> List[Dict[str, Any]]:
        """Execute query with BigQuery client."""
        loop = asyncio.get_event_loop()

        def _run_query():
            query_job = conn.query(query)
            results = query_job.result()
            return [dict(row.items()) for row in results]

        return await loop.run_in_executor(None, _run_query)

    async def collect_schema(self, table_ref: str) -> TableMetadata:
        """Collect schema metadata with BigQuery-specific optimizations."""
        parts = table_ref.split(".")
        if len(parts) != 3:
            raise ValueError(f"Invalid table reference: {table_ref}")

        project, dataset, table = parts

        # Use BigQuery's INFORMATION_SCHEMA
        columns_query = f"""
            SELECT
                column_name,
                data_type,
                is_nullable,
                description
            FROM `{project}.{dataset}.INFORMATION_SCHEMA.COLUMNS`
            WHERE table_name = '{table}'
            ORDER BY ordinal_position
        """
        columns_result = await self._execute(columns_query)

        # Get table metadata
        stats_query = f"""
            SELECT
                row_count,
                size_bytes,
                last_modified_time
            FROM `{project}.{dataset}.__TABLES__`
            WHERE table_id = '{table}'
        """
        stats_result = await self._execute(stats_query)

        columns = [
            ColumnMetadata(
                name=row.get("column_name", ""),
                data_type=row.get("data_type", ""),
                nullable=row.get("is_nullable", "YES") == "YES",
                description=row.get("description"),
                stats=None,
            )
            for row in columns_result
        ]

        row_count = 0
        size_bytes = 0
        last_modified = datetime.now()

        if stats_result:
            row_count = stats_result[0].get("row_count", 0) or 0
            size_bytes = stats_result[0].get("size_bytes", 0) or 0
            # BigQuery returns timestamp in milliseconds
            ts = stats_result[0].get("last_modified_time", 0)
            if ts:
                last_modified = datetime.fromtimestamp(ts / 1000)

        return TableMetadata(
            table_id=table_ref,
            database=project,
            schema=dataset,
            table_name=table,
            columns=columns,
            row_count=row_count,
            size_bytes=size_bytes,
            last_modified=last_modified,
        )


class PostgresCollector(GenericSQLCollector):
    """Collector for PostgreSQL databases."""

    def __init__(
        self,
        config: CollectorConfig,
        warehouse_config: WarehouseConfig,
    ):
        super().__init__(config, warehouse_config)
        if not HAS_ASYNCPG:
            logger.warning(
                "asyncpg not installed. "
                "Install with: pip install asyncpg"
            )

    async def connect(self) -> None:
        """Establish PostgreSQL connection pool."""
        if not HAS_ASYNCPG:
            raise ImportError("asyncpg is required for PostgreSQL")

        self._pool = await asyncpg.create_pool(
            host=self.warehouse_config.host,
            port=self.warehouse_config.port,
            database=self.warehouse_config.database,
            user=self.warehouse_config.user,
            password=self.warehouse_config.password,
            min_size=1,
            max_size=self.config.parallel_workers,
            **self.warehouse_config.options,
        )
        self._connected = True
        logger.info(
            f"Connected to PostgreSQL: {self.warehouse_config.host}:"
            f"{self.warehouse_config.port}/{self.warehouse_config.database}"
        )

    async def _get_connection(self) -> Any:
        """Get connection from pool."""
        if not self._connected:
            await self.connect()
        return self._pool

    async def _execute_with_connection(
        self, conn: Any, query: str
    ) -> List[Dict[str, Any]]:
        """Execute query with asyncpg pool."""
        async with conn.acquire() as connection:
            rows = await connection.fetch(query)
            return [dict(row) for row in rows]

    async def collect_schema(self, table_ref: str) -> TableMetadata:
        """Collect schema metadata with PostgreSQL-specific optimizations."""
        parts = table_ref.split(".")
        if len(parts) == 3:
            database, schema, table = parts
        elif len(parts) == 2:
            schema, table = parts
            database = self.warehouse_config.database
        else:
            raise ValueError(f"Invalid table reference: {table_ref}")

        # Get columns from information_schema
        columns_query = f"""
            SELECT
                column_name,
                data_type,
                is_nullable,
                col_description(
                    (quote_ident(table_schema) || '.' || quote_ident(table_name))::regclass,
                    ordinal_position
                ) as description
            FROM information_schema.columns
            WHERE table_schema = '{schema}'
              AND table_name = '{table}'
            ORDER BY ordinal_position
        """
        columns_result = await self._execute(columns_query)

        # Get table stats using pg_stat_user_tables
        stats_query = f"""
            SELECT
                reltuples::bigint as row_count,
                pg_total_relation_size('{schema}.{table}') as size_bytes
            FROM pg_class
            WHERE relname = '{table}'
              AND relnamespace = (SELECT oid FROM pg_namespace WHERE nspname = '{schema}')
        """
        stats_result = await self._execute(stats_query)

        columns = [
            ColumnMetadata(
                name=row.get("column_name", ""),
                data_type=row.get("data_type", ""),
                nullable=row.get("is_nullable", "YES") == "YES",
                description=row.get("description"),
                stats=None,
            )
            for row in columns_result
        ]

        row_count = 0
        size_bytes = 0

        if stats_result:
            row_count = int(stats_result[0].get("row_count", 0) or 0)
            size_bytes = int(stats_result[0].get("size_bytes", 0) or 0)

        return TableMetadata(
            table_id=table_ref,
            database=database,
            schema=schema,
            table_name=table,
            columns=columns,
            row_count=row_count,
            size_bytes=size_bytes,
            last_modified=datetime.now(),
        )


class RedshiftCollector(PostgresCollector):
    """Collector for Amazon Redshift (PostgreSQL-compatible)."""

    async def collect_schema(self, table_ref: str) -> TableMetadata:
        """Collect schema metadata with Redshift-specific optimizations."""
        parts = table_ref.split(".")
        if len(parts) == 3:
            database, schema, table = parts
        elif len(parts) == 2:
            schema, table = parts
            database = self.warehouse_config.database
        else:
            raise ValueError(f"Invalid table reference: {table_ref}")

        # Use Redshift-specific system tables
        columns_query = f"""
            SELECT
                column_name,
                data_type,
                is_nullable,
                remarks as description
            FROM SVV_COLUMNS
            WHERE table_schema = '{schema}'
              AND table_name = '{table}'
            ORDER BY ordinal_position
        """
        columns_result = await self._execute(columns_query)

        # Get row count from Redshift's statistics
        stats_query = f"""
            SELECT
                reltuples as row_count,
                size * 1024 * 1024 as size_bytes
            FROM SVV_TABLE_INFO
            WHERE schema = '{schema}'
              AND "table" = '{table}'
        """
        stats_result = await self._execute(stats_query)

        columns = [
            ColumnMetadata(
                name=row.get("column_name", ""),
                data_type=row.get("data_type", ""),
                nullable=row.get("is_nullable", "YES") == "YES",
                description=row.get("description"),
                stats=None,
            )
            for row in columns_result
        ]

        row_count = 0
        size_bytes = 0

        if stats_result:
            row_count = int(stats_result[0].get("row_count", 0) or 0)
            size_bytes = int(stats_result[0].get("size_bytes", 0) or 0)

        return TableMetadata(
            table_id=table_ref,
            database=database,
            schema=schema,
            table_name=table,
            columns=columns,
            row_count=row_count,
            size_bytes=size_bytes,
            last_modified=datetime.now(),
        )


class DatabricksCollector(GenericSQLCollector):
    """Collector for Databricks Unity Catalog."""

    def __init__(
        self,
        config: CollectorConfig,
        warehouse_config: WarehouseConfig,
    ):
        super().__init__(config, warehouse_config)
        if not HAS_DATABRICKS:
            logger.warning(
                "databricks-sql-connector not installed. "
                "Install with: pip install databricks-sql-connector"
            )

    async def connect(self) -> None:
        """Establish Databricks SQL connection."""
        if not HAS_DATABRICKS:
            raise ImportError("databricks-sql-connector is required")

        loop = asyncio.get_event_loop()

        def _create_connection():
            return databricks_sql.connect(
                server_hostname=self.warehouse_config.host,
                http_path=self.warehouse_config.options.get("http_path", ""),
                access_token=self.warehouse_config.password,  # Use token as password
            )

        self._connection = await loop.run_in_executor(None, _create_connection)
        self._connected = True
        logger.info(f"Connected to Databricks: {self.warehouse_config.host}")

    async def _execute_with_connection(
        self, conn: Any, query: str
    ) -> List[Dict[str, Any]]:
        """Execute query with Databricks connection."""
        loop = asyncio.get_event_loop()

        def _run_query():
            cursor = conn.cursor()
            try:
                cursor.execute(query)
                columns = [desc[0].lower() for desc in cursor.description or []]
                rows = cursor.fetchall()
                return [dict(zip(columns, row)) for row in rows]
            finally:
                cursor.close()

        return await loop.run_in_executor(None, _run_query)

    async def collect_schema(self, table_ref: str) -> TableMetadata:
        """Collect schema metadata with Databricks Unity Catalog."""
        parts = table_ref.split(".")
        if len(parts) != 3:
            raise ValueError(f"Invalid table reference: {table_ref}")

        catalog, schema, table = parts

        # Use DESCRIBE command for Unity Catalog
        describe_query = f"DESCRIBE TABLE {catalog}.{schema}.{table}"
        describe_result = await self._execute(describe_query)

        columns = []
        for row in describe_result:
            col_name = row.get("col_name", "")
            # Skip metadata rows (start with #)
            if col_name and not col_name.startswith("#"):
                columns.append(
                    ColumnMetadata(
                        name=col_name,
                        data_type=row.get("data_type", ""),
                        nullable=True,  # Databricks doesn't report nullable in DESCRIBE
                        description=row.get("comment"),
                        stats=None,
                    )
                )

        # Get table details
        detail_query = f"DESCRIBE DETAIL {catalog}.{schema}.{table}"
        detail_result = await self._execute(detail_query)

        row_count = 0
        size_bytes = 0
        last_modified = datetime.now()

        if detail_result:
            row_count = int(detail_result[0].get("numFiles", 0) or 0)
            size_bytes = int(detail_result[0].get("sizeInBytes", 0) or 0)
            ts = detail_result[0].get("lastModified")
            if ts:
                last_modified = datetime.fromtimestamp(ts / 1000)

        return TableMetadata(
            table_id=table_ref,
            database=catalog,
            schema=schema,
            table_name=table,
            columns=columns,
            row_count=row_count,
            size_bytes=size_bytes,
            last_modified=last_modified,
        )


def create_collector(
    config: CollectorConfig,
    warehouse_config: WarehouseConfig,
) -> MetadataCollector:
    """Factory function to create appropriate collector based on warehouse type."""
    warehouse_type = warehouse_config.warehouse_type.lower()

    collectors = {
        "snowflake": SnowflakeCollector,
        "bigquery": BigQueryCollector,
        "bq": BigQueryCollector,
        "postgres": PostgresCollector,
        "postgresql": PostgresCollector,
        "redshift": RedshiftCollector,
        "databricks": DatabricksCollector,
    }

    collector_class = collectors.get(warehouse_type)
    if collector_class is None:
        logger.warning(
            f"Unknown warehouse type: {warehouse_type}. "
            f"Using GenericSQLCollector."
        )
        return GenericSQLCollector(config, warehouse_config)

    return collector_class(config, warehouse_config)
