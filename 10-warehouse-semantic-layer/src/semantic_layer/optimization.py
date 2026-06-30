"""Query optimization, pre-aggregation, and caching for the semantic layer.

This module provides:
- Query planner with cost-based optimization
- Pre-aggregation table management
- Materialized view recommendations
- Query result caching with TTL policies
- Aggregate detection and rollup optimization
"""

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from semantic_layer.models import MetricQuery, TimeGrain

logger = logging.getLogger(__name__)


# =============================================================================
# Pre-Aggregation Tables
# =============================================================================

class AggregateGranularity(Enum):
    """Time granularity for aggregate tables."""

    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


@dataclass
class AggregateTableConfig:
    """Configuration for a pre-aggregated table."""

    name: str
    source_table: str
    metrics: List[str]
    dimensions: List[str]
    time_column: str
    granularity: AggregateGranularity
    refresh_schedule: str = "0 * * * *"  # Cron expression
    retention_days: int = 365
    partitioned_by: Optional[str] = None
    clustered_by: List[str] = field(default_factory=list)

    @property
    def qualified_name(self) -> str:
        """Get qualified table name."""
        return f"agg_{self.source_table}_{self.granularity.value}"


class AggregateTableManager:
    """Manage pre-aggregated tables."""

    def __init__(self):
        self._tables: Dict[str, AggregateTableConfig] = {}
        self._refresh_history: Dict[str, List[datetime]] = {}

    def register_table(self, config: AggregateTableConfig) -> None:
        """Register a pre-aggregated table."""
        self._tables[config.name] = config
        logger.info(f"Registered aggregate table: {config.name}")

    def get_table(self, name: str) -> Optional[AggregateTableConfig]:
        """Get aggregate table by name."""
        return self._tables.get(name)

    def find_matching_table(
        self,
        source_table: str,
        metrics: List[str],
        dimensions: List[str],
        granularity: AggregateGranularity,
    ) -> Optional[AggregateTableConfig]:
        """Find an aggregate table that satisfies the query requirements."""
        for table in self._tables.values():
            if table.source_table != source_table:
                continue

            # Check if table has required metrics
            if not set(metrics).issubset(set(table.metrics)):
                continue

            # Check if table has required dimensions
            if not set(dimensions).issubset(set(table.dimensions)):
                continue

            # Check granularity compatibility (can use finer granularity)
            if self._is_granularity_compatible(granularity, table.granularity):
                return table

        return None

    def _is_granularity_compatible(
        self,
        required: AggregateGranularity,
        available: AggregateGranularity,
    ) -> bool:
        """Check if available granularity can satisfy required granularity."""
        order = [
            AggregateGranularity.HOURLY,
            AggregateGranularity.DAILY,
            AggregateGranularity.WEEKLY,
            AggregateGranularity.MONTHLY,
            AggregateGranularity.QUARTERLY,
            AggregateGranularity.YEARLY,
        ]
        return order.index(available) <= order.index(required)

    def generate_create_ddl(
        self,
        config: AggregateTableConfig,
        warehouse_type: str = "snowflake",
    ) -> str:
        """Generate DDL to create aggregate table."""
        columns = []

        # Time column
        columns.append(f"{config.time_column} TIMESTAMP")

        # Dimensions
        for dim in config.dimensions:
            columns.append(f"{dim} VARCHAR")

        # Metrics (pre-aggregated)
        for metric in config.metrics:
            columns.append(f"{metric}_sum DECIMAL(38,4)")
            columns.append(f"{metric}_count BIGINT")

        columns_sql = ",\n    ".join(columns)

        ddl = f"""CREATE TABLE IF NOT EXISTS {config.qualified_name} (
    {columns_sql}
)"""

        if config.partitioned_by and warehouse_type in ("snowflake", "bigquery"):
            if warehouse_type == "snowflake":
                ddl += f"\nCLUSTER BY ({config.partitioned_by})"
            else:
                ddl += f"\nPARTITION BY DATE({config.partitioned_by})"

        return ddl

    def generate_refresh_query(
        self,
        config: AggregateTableConfig,
        warehouse_type: str = "snowflake",
    ) -> str:
        """Generate SQL to refresh aggregate table."""
        time_trunc = self._get_time_trunc(config.granularity, warehouse_type)

        select_parts = [f"{time_trunc}({config.time_column}) as {config.time_column}"]
        select_parts.extend(config.dimensions)

        for metric in config.metrics:
            select_parts.append(f"SUM({metric}) as {metric}_sum")
            select_parts.append(f"COUNT({metric}) as {metric}_count")

        group_by = [f"{time_trunc}({config.time_column})"] + config.dimensions

        return f"""INSERT INTO {config.qualified_name}
SELECT
    {', '.join(select_parts)}
FROM {config.source_table}
GROUP BY {', '.join(group_by)}"""

    def _get_time_trunc(
        self,
        granularity: AggregateGranularity,
        warehouse_type: str,
    ) -> str:
        """Get time truncation function."""
        grain_map = {
            AggregateGranularity.HOURLY: "hour",
            AggregateGranularity.DAILY: "day",
            AggregateGranularity.WEEKLY: "week",
            AggregateGranularity.MONTHLY: "month",
            AggregateGranularity.QUARTERLY: "quarter",
            AggregateGranularity.YEARLY: "year",
        }
        grain = grain_map[granularity]

        if warehouse_type == "bigquery":
            return f"DATE_TRUNC"
        return f"DATE_TRUNC('{grain}', "  # For Snowflake/Postgres

    def recommend_aggregates(
        self,
        query_patterns: List[Dict[str, Any]],
        min_frequency: int = 10,
    ) -> List[AggregateTableConfig]:
        """Recommend aggregate tables based on query patterns."""
        recommendations = []

        # Group patterns by source table and dimensions
        pattern_groups: Dict[str, List[Dict]] = {}
        for pattern in query_patterns:
            key = f"{pattern['source_table']}:{':'.join(sorted(pattern['dimensions']))}"
            if key not in pattern_groups:
                pattern_groups[key] = []
            pattern_groups[key].append(pattern)

        # Generate recommendations for frequent patterns
        for key, patterns in pattern_groups.items():
            if len(patterns) >= min_frequency:
                sample = patterns[0]
                recommendations.append(
                    AggregateTableConfig(
                        name=f"agg_{len(recommendations)}",
                        source_table=sample['source_table'],
                        metrics=list(set(m for p in patterns for m in p.get('metrics', []))),
                        dimensions=sample['dimensions'],
                        time_column=sample.get('time_column', 'created_at'),
                        granularity=AggregateGranularity.DAILY,
                    )
                )

        return recommendations

    def list_tables(self) -> List[AggregateTableConfig]:
        """List all registered aggregate tables."""
        return list(self._tables.values())


# =============================================================================
# Query Cost Estimation
# =============================================================================

@dataclass
class TableStats:
    """Statistics for a table used in cost estimation."""

    table_name: str
    row_count: int
    size_bytes: int
    column_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    last_analyzed: Optional[datetime] = None


@dataclass
class QueryCost:
    """Estimated cost of executing a query."""

    estimated_rows: int
    estimated_bytes: int
    estimated_time_ms: float
    io_cost: float
    cpu_cost: float
    network_cost: float = 0.0

    @property
    def total_cost(self) -> float:
        """Total estimated cost."""
        return self.io_cost + self.cpu_cost + self.network_cost


class CostEstimator:
    """Estimate query execution costs."""

    def __init__(self):
        self._table_stats: Dict[str, TableStats] = {}
        self._cost_per_byte_scanned = 0.000001  # $0.001 per GB
        self._cost_per_cpu_second = 0.0001

    def update_table_stats(self, stats: TableStats) -> None:
        """Update statistics for a table."""
        self._table_stats[stats.table_name] = stats

    def estimate_cost(
        self,
        query: MetricQuery,
        table_name: str,
        use_aggregate: bool = False,
        aggregate_table: Optional[str] = None,
    ) -> QueryCost:
        """Estimate cost of executing a query."""
        stats = self._table_stats.get(table_name)

        if stats is None:
            # Use defaults if no stats available
            stats = TableStats(
                table_name=table_name,
                row_count=1_000_000,
                size_bytes=1_000_000_000,
            )

        # Estimate rows based on filters and time range
        selectivity = self._estimate_selectivity(query)
        estimated_rows = int(stats.row_count * selectivity)

        # Use aggregate table stats if available
        if use_aggregate and aggregate_table:
            agg_stats = self._table_stats.get(aggregate_table)
            if agg_stats:
                estimated_rows = min(estimated_rows, agg_stats.row_count)
                stats = agg_stats

        # Estimate bytes scanned
        bytes_per_row = stats.size_bytes / stats.row_count if stats.row_count > 0 else 100
        estimated_bytes = int(estimated_rows * bytes_per_row)

        # Calculate costs
        io_cost = estimated_bytes * self._cost_per_byte_scanned

        # CPU cost based on aggregations
        num_aggregations = len(query.metrics)
        cpu_seconds = (estimated_rows / 1_000_000) * num_aggregations
        cpu_cost = cpu_seconds * self._cost_per_cpu_second

        # Estimate time
        estimated_time_ms = (io_cost + cpu_cost) * 10_000  # Rough conversion

        return QueryCost(
            estimated_rows=estimated_rows,
            estimated_bytes=estimated_bytes,
            estimated_time_ms=estimated_time_ms,
            io_cost=io_cost,
            cpu_cost=cpu_cost,
        )

    def _estimate_selectivity(self, query: MetricQuery) -> float:
        """Estimate query selectivity (0-1)."""
        selectivity = 1.0

        # Time range selectivity
        start = datetime.fromisoformat(query.start_date.replace('Z', '+00:00') if 'Z' in query.start_date else query.start_date)
        end = datetime.fromisoformat(query.end_date.replace('Z', '+00:00') if 'Z' in query.end_date else query.end_date)
        days = (end - start).days
        selectivity *= min(days / 365, 1.0)  # Assume 1 year of data

        # Filter selectivity (simplified)
        for _ in query.filters:
            selectivity *= 0.1  # Each filter reduces by 90%

        return max(selectivity, 0.001)  # Minimum 0.1% selectivity


# =============================================================================
# Query Planner
# =============================================================================

@dataclass
class QueryPlan:
    """Execution plan for a query."""

    original_query: MetricQuery
    rewritten_sql: str
    use_aggregate_table: bool
    aggregate_table_name: Optional[str]
    estimated_cost: QueryCost
    optimization_notes: List[str]
    cache_key: Optional[str] = None
    cache_ttl_seconds: Optional[int] = None


class QueryPlanner:
    """Plan and optimize semantic queries."""

    def __init__(
        self,
        aggregate_manager: AggregateTableManager,
        cost_estimator: CostEstimator,
    ):
        self.aggregate_manager = aggregate_manager
        self.cost_estimator = cost_estimator
        self._optimization_rules: List[OptimizationRule] = []
        self._register_default_rules()

    def _register_default_rules(self) -> None:
        """Register default optimization rules."""
        self._optimization_rules = [
            AggregateTableRule(self.aggregate_manager),
            TimeRangeOptimizationRule(),
            FilterPushdownRule(),
            ProjectionPruningRule(),
        ]

    def plan(
        self,
        query: MetricQuery,
        source_table: str,
        warehouse_type: str = "snowflake",
    ) -> QueryPlan:
        """Create an optimized execution plan for a query."""
        optimization_notes = []
        use_aggregate = False
        aggregate_table = None
        rewritten_query = query

        # Apply optimization rules
        for rule in self._optimization_rules:
            result = rule.apply(rewritten_query, source_table)
            if result.modified:
                rewritten_query = result.query
                optimization_notes.append(result.note)

                if isinstance(rule, AggregateTableRule) and result.aggregate_table:
                    use_aggregate = True
                    aggregate_table = result.aggregate_table

        # Estimate costs
        base_cost = self.cost_estimator.estimate_cost(
            rewritten_query,
            source_table,
            use_aggregate=False,
        )

        agg_cost = None
        if use_aggregate and aggregate_table:
            agg_cost = self.cost_estimator.estimate_cost(
                rewritten_query,
                source_table,
                use_aggregate=True,
                aggregate_table=aggregate_table,
            )

        # Choose cheaper plan
        if agg_cost and agg_cost.total_cost < base_cost.total_cost:
            final_cost = agg_cost
            optimization_notes.append(
                f"Using aggregate table saves {(1 - agg_cost.total_cost/base_cost.total_cost)*100:.1f}%"
            )
        else:
            final_cost = base_cost
            use_aggregate = False
            aggregate_table = None

        # Generate SQL
        sql = self._generate_sql(
            rewritten_query,
            source_table if not use_aggregate else aggregate_table,
            warehouse_type,
        )

        # Compute cache key
        cache_key = self._compute_cache_key(query)
        cache_ttl = self._determine_cache_ttl(query)

        return QueryPlan(
            original_query=query,
            rewritten_sql=sql,
            use_aggregate_table=use_aggregate,
            aggregate_table_name=aggregate_table,
            estimated_cost=final_cost,
            optimization_notes=optimization_notes,
            cache_key=cache_key,
            cache_ttl_seconds=cache_ttl,
        )

    def _generate_sql(
        self,
        query: MetricQuery,
        table_name: str,
        warehouse_type: str,
    ) -> str:
        """Generate SQL for the query."""
        # Time truncation based on warehouse
        if warehouse_type == "bigquery":
            time_trunc = f"DATE_TRUNC({query.time_grain.upper()}, created_at)"
        else:
            time_trunc = f"DATE_TRUNC('{query.time_grain}', created_at)"

        select_parts = [f"{time_trunc} as period"]
        select_parts.extend(query.dimensions)

        for metric in query.metrics:
            select_parts.append(f"SUM({metric}) as {metric}")

        where_parts = [
            f"created_at >= '{query.start_date}'",
            f"created_at < '{query.end_date}'",
        ]

        for f in query.filters:
            where_parts.append(f"{f['field']} {f['operator']} {f['value']}")

        group_by = ["period"] + query.dimensions

        sql = f"""SELECT
    {', '.join(select_parts)}
FROM {table_name}
WHERE {' AND '.join(where_parts)}
GROUP BY {', '.join(group_by)}
ORDER BY period"""

        if query.limit:
            sql += f"\nLIMIT {query.limit}"

        return sql

    def _compute_cache_key(self, query: MetricQuery) -> str:
        """Compute cache key for a query."""
        key_data = {
            "metrics": sorted(query.metrics),
            "dimensions": sorted(query.dimensions),
            "time_grain": query.time_grain,
            "start_date": query.start_date,
            "end_date": query.end_date,
            "filters": sorted(json.dumps(f) for f in query.filters),
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]

    def _determine_cache_ttl(self, query: MetricQuery) -> int:
        """Determine cache TTL based on query characteristics."""
        # Historical data can be cached longer
        end_date = datetime.fromisoformat(
            query.end_date.replace('Z', '+00:00') if 'Z' in query.end_date else query.end_date
        )
        now = datetime.now()

        if end_date < now - timedelta(days=7):
            return 86400  # 24 hours for historical data
        elif end_date < now - timedelta(days=1):
            return 3600  # 1 hour for recent data
        else:
            return 300  # 5 minutes for current data

    def explain(self, plan: QueryPlan) -> str:
        """Generate human-readable explanation of query plan."""
        lines = [
            "Query Execution Plan",
            "=" * 50,
            f"Metrics: {', '.join(plan.original_query.metrics)}",
            f"Dimensions: {', '.join(plan.original_query.dimensions)}",
            f"Time Range: {plan.original_query.start_date} to {plan.original_query.end_date}",
            f"Granularity: {plan.original_query.time_grain}",
            "",
            "Optimization Applied:",
        ]

        for note in plan.optimization_notes:
            lines.append(f"  - {note}")

        if plan.use_aggregate_table:
            lines.append(f"\nUsing aggregate table: {plan.aggregate_table_name}")

        lines.extend([
            "",
            "Cost Estimates:",
            f"  Estimated rows: {plan.estimated_cost.estimated_rows:,}",
            f"  Estimated bytes: {plan.estimated_cost.estimated_bytes:,}",
            f"  Estimated time: {plan.estimated_cost.estimated_time_ms:.2f}ms",
            f"  Total cost: ${plan.estimated_cost.total_cost:.6f}",
            "",
            f"Cache Key: {plan.cache_key}",
            f"Cache TTL: {plan.cache_ttl_seconds}s",
            "",
            "Generated SQL:",
            plan.rewritten_sql,
        ])

        return "\n".join(lines)


# =============================================================================
# Optimization Rules
# =============================================================================

@dataclass
class OptimizationResult:
    """Result of applying an optimization rule."""

    modified: bool
    query: MetricQuery
    note: str = ""
    aggregate_table: Optional[str] = None


class OptimizationRule(ABC):
    """Base class for query optimization rules."""

    @abstractmethod
    def apply(
        self,
        query: MetricQuery,
        source_table: str,
    ) -> OptimizationResult:
        """Apply the optimization rule."""
        pass


class AggregateTableRule(OptimizationRule):
    """Rule to use pre-aggregated tables when available."""

    def __init__(self, aggregate_manager: AggregateTableManager):
        self.aggregate_manager = aggregate_manager

    def apply(
        self,
        query: MetricQuery,
        source_table: str,
    ) -> OptimizationResult:
        """Check if an aggregate table can be used."""
        granularity_map = {
            "day": AggregateGranularity.DAILY,
            "week": AggregateGranularity.WEEKLY,
            "month": AggregateGranularity.MONTHLY,
            "quarter": AggregateGranularity.QUARTERLY,
            "year": AggregateGranularity.YEARLY,
        }

        granularity = granularity_map.get(query.time_grain)
        if not granularity:
            return OptimizationResult(modified=False, query=query)

        agg_table = self.aggregate_manager.find_matching_table(
            source_table,
            query.metrics,
            query.dimensions,
            granularity,
        )

        if agg_table:
            return OptimizationResult(
                modified=True,
                query=query,
                note=f"Can use aggregate table: {agg_table.name}",
                aggregate_table=agg_table.qualified_name,
            )

        return OptimizationResult(modified=False, query=query)


class TimeRangeOptimizationRule(OptimizationRule):
    """Optimize queries with large time ranges."""

    def apply(
        self,
        query: MetricQuery,
        source_table: str,
    ) -> OptimizationResult:
        """Suggest coarser granularity for large time ranges."""
        start = datetime.fromisoformat(
            query.start_date.replace('Z', '+00:00') if 'Z' in query.start_date else query.start_date
        )
        end = datetime.fromisoformat(
            query.end_date.replace('Z', '+00:00') if 'Z' in query.end_date else query.end_date
        )
        days = (end - start).days

        recommendations = []
        if days > 365 and query.time_grain == "day":
            recommendations.append("Consider using 'week' or 'month' granularity for >1 year range")
        elif days > 90 and query.time_grain == "day":
            recommendations.append("Consider using 'week' granularity for >90 day range")

        if recommendations:
            return OptimizationResult(
                modified=False,  # Only advisory
                query=query,
                note="; ".join(recommendations),
            )

        return OptimizationResult(modified=False, query=query)


class FilterPushdownRule(OptimizationRule):
    """Push filters down to reduce data scanned."""

    def apply(
        self,
        query: MetricQuery,
        source_table: str,
    ) -> OptimizationResult:
        """Analyze filter pushdown opportunities."""
        if query.filters:
            return OptimizationResult(
                modified=False,
                query=query,
                note=f"Pushing down {len(query.filters)} filters to base table scan",
            )
        return OptimizationResult(modified=False, query=query)


class ProjectionPruningRule(OptimizationRule):
    """Prune unnecessary columns from scan."""

    def apply(
        self,
        query: MetricQuery,
        source_table: str,
    ) -> OptimizationResult:
        """Identify columns to prune."""
        required_columns = set(query.metrics) | set(query.dimensions)
        return OptimizationResult(
            modified=False,
            query=query,
            note=f"Scanning only {len(required_columns)} required columns",
        )


# =============================================================================
# Query Cache
# =============================================================================

@dataclass
class CacheEntry:
    """Entry in the query cache."""

    key: str
    value: Any
    created_at: float
    ttl_seconds: int
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        return time.time() > self.created_at + self.ttl_seconds


class QueryCache:
    """In-memory query result cache with TTL."""

    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._stats = CacheStats()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        entry = self._cache.get(key)

        if entry is None:
            self._stats.misses += 1
            return None

        if entry.is_expired:
            del self._cache[key]
            self._stats.misses += 1
            return None

        entry.access_count += 1
        entry.last_accessed = time.time()
        self._stats.hits += 1
        return entry.value

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Set value in cache."""
        if len(self._cache) >= self.max_size:
            self._evict()

        self._cache[key] = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            ttl_seconds=ttl_seconds or self.default_ttl,
        )

    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all entries."""
        self._cache.clear()

    def _evict(self) -> None:
        """Evict entries to make room."""
        # Remove expired entries first
        expired = [k for k, v in self._cache.items() if v.is_expired]
        for key in expired:
            del self._cache[key]

        # If still over capacity, remove LRU entries
        if len(self._cache) >= self.max_size:
            entries = sorted(
                self._cache.items(),
                key=lambda x: x[1].last_accessed,
            )
            to_remove = len(self._cache) - self.max_size + 1
            for key, _ in entries[:to_remove]:
                del self._cache[key]

    def stats(self) -> 'CacheStats':
        """Get cache statistics."""
        self._stats.size = len(self._cache)
        return self._stats


@dataclass
class CacheStats:
    """Cache statistics."""

    hits: int = 0
    misses: int = 0
    size: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class RedisCache:
    """Redis-backed query cache."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        prefix: str = "semantic:",
        default_ttl: int = 300,
    ):
        self.prefix = prefix
        self.default_ttl = default_ttl
        self._redis = None
        self._config = {"host": host, "port": port, "db": db}
        self._connected = False

    def _connect(self) -> None:
        """Lazy connection to Redis."""
        if not self._connected:
            try:
                import redis
                self._redis = redis.Redis(**self._config)
                self._redis.ping()
                self._connected = True
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self._redis = None

    def _key(self, key: str) -> str:
        """Get prefixed key."""
        return f"{self.prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis."""
        self._connect()
        if not self._redis:
            return None

        try:
            value = self._redis.get(self._key(key))
            if value:
                return json.loads(value)
        except Exception as e:
            logger.error(f"Redis get error: {e}")
        return None

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Set value in Redis."""
        self._connect()
        if not self._redis:
            return

        try:
            self._redis.setex(
                self._key(key),
                ttl_seconds or self.default_ttl,
                json.dumps(value),
            )
        except Exception as e:
            logger.error(f"Redis set error: {e}")

    def delete(self, key: str) -> bool:
        """Delete key from Redis."""
        self._connect()
        if not self._redis:
            return False

        try:
            return bool(self._redis.delete(self._key(key)))
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False

    def clear(self, pattern: str = "*") -> int:
        """Clear keys matching pattern."""
        self._connect()
        if not self._redis:
            return 0

        try:
            keys = self._redis.keys(self._key(pattern))
            if keys:
                return self._redis.delete(*keys)
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
        return 0


# =============================================================================
# Materialized View Advisor
# =============================================================================

@dataclass
class MaterializedViewRecommendation:
    """Recommendation for a materialized view."""

    name: str
    source_query: str
    refresh_strategy: str  # "on_commit", "on_demand", "scheduled"
    estimated_storage_mb: float
    estimated_query_speedup: float
    recommended_for_queries: List[str]


class MaterializedViewAdvisor:
    """Advise on materialized view creation."""

    def __init__(self):
        self._query_history: List[Dict[str, Any]] = []

    def record_query(
        self,
        query: MetricQuery,
        execution_time_ms: float,
        rows_scanned: int,
    ) -> None:
        """Record a query execution for analysis."""
        self._query_history.append({
            "query": query,
            "execution_time_ms": execution_time_ms,
            "rows_scanned": rows_scanned,
            "timestamp": datetime.now().isoformat(),
        })

    def analyze_and_recommend(
        self,
        min_occurrences: int = 5,
        min_avg_time_ms: float = 1000,
    ) -> List[MaterializedViewRecommendation]:
        """Analyze query history and recommend materialized views."""
        recommendations = []

        # Group queries by signature
        query_groups: Dict[str, List[Dict]] = {}
        for record in self._query_history:
            query = record["query"]
            sig = self._query_signature(query)
            if sig not in query_groups:
                query_groups[sig] = []
            query_groups[sig].append(record)

        # Recommend for frequent, slow queries
        for sig, records in query_groups.items():
            if len(records) < min_occurrences:
                continue

            avg_time = sum(r["execution_time_ms"] for r in records) / len(records)
            if avg_time < min_avg_time_ms:
                continue

            sample_query = records[0]["query"]
            recommendations.append(
                MaterializedViewRecommendation(
                    name=f"mv_{sig[:16]}",
                    source_query=self._generate_mv_query(sample_query),
                    refresh_strategy="scheduled",
                    estimated_storage_mb=sum(r["rows_scanned"] for r in records) / 1000000,
                    estimated_query_speedup=avg_time / 100,  # Assume 100ms with MV
                    recommended_for_queries=[sig],
                )
            )

        return recommendations

    def _query_signature(self, query: MetricQuery) -> str:
        """Generate a signature for a query pattern."""
        sig_data = {
            "metrics": sorted(query.metrics),
            "dimensions": sorted(query.dimensions),
            "time_grain": query.time_grain,
        }
        sig_str = json.dumps(sig_data, sort_keys=True)
        return hashlib.sha256(sig_str.encode()).hexdigest()[:32]

    def _generate_mv_query(self, query: MetricQuery) -> str:
        """Generate materialized view definition query."""
        return f"-- Materialized view for {', '.join(query.metrics)}"


# =============================================================================
# Integration
# =============================================================================

class OptimizedQueryEngine:
    """Query engine with integrated optimization and caching."""

    def __init__(
        self,
        warehouse_type: str = "snowflake",
        use_cache: bool = True,
        cache_backend: str = "memory",  # "memory" or "redis"
        redis_config: Optional[Dict[str, Any]] = None,
    ):
        self.warehouse_type = warehouse_type
        self.aggregate_manager = AggregateTableManager()
        self.cost_estimator = CostEstimator()
        self.planner = QueryPlanner(self.aggregate_manager, self.cost_estimator)
        self.mv_advisor = MaterializedViewAdvisor()

        # Initialize cache
        self.use_cache = use_cache
        if use_cache:
            if cache_backend == "redis":
                self._cache = RedisCache(**(redis_config or {}))
            else:
                self._cache = QueryCache()
        else:
            self._cache = None

    def plan_query(
        self,
        query: MetricQuery,
        source_table: str,
    ) -> QueryPlan:
        """Plan a query execution."""
        return self.planner.plan(query, source_table, self.warehouse_type)

    def execute(
        self,
        query: MetricQuery,
        source_table: str,
        executor: Any,  # QueryExecutor
    ) -> Dict[str, Any]:
        """Execute a query with optimization and caching."""
        plan = self.plan_query(query, source_table)

        # Check cache
        if self._cache and plan.cache_key:
            cached = self._cache.get(plan.cache_key)
            if cached:
                return {
                    "data": cached,
                    "from_cache": True,
                    "plan": plan,
                }

        # Execute query
        import asyncio
        start_time = time.time()

        # Run async executor in sync context
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(executor.execute(plan.rewritten_sql))
        finally:
            loop.close()

        execution_time = (time.time() - start_time) * 1000

        # Record for MV advisor
        self.mv_advisor.record_query(
            query,
            execution_time,
            plan.estimated_cost.estimated_rows,
        )

        # Cache result
        if self._cache and plan.cache_key:
            self._cache.set(plan.cache_key, result, plan.cache_ttl_seconds)

        return {
            "data": result,
            "from_cache": False,
            "plan": plan,
            "execution_time_ms": execution_time,
        }

    def explain(self, query: MetricQuery, source_table: str) -> str:
        """Get query explanation."""
        plan = self.plan_query(query, source_table)
        return self.planner.explain(plan)

    def get_mv_recommendations(self) -> List[MaterializedViewRecommendation]:
        """Get materialized view recommendations."""
        return self.mv_advisor.analyze_and_recommend()

    def invalidate_cache(self, pattern: Optional[str] = None) -> None:
        """Invalidate cache entries."""
        if self._cache:
            if pattern:
                if isinstance(self._cache, RedisCache):
                    self._cache.clear(pattern)
            else:
                self._cache.clear()


#  =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    # Simple test
    from semantic_layer.models import MetricQuery

    # Create test query
    query = MetricQuery(
        metrics=["total_revenue", "order_count"],
        dimensions=["country", "product_category"],
        time_grain="day",
        start_date="2024-01-01",
        end_date="2024-01-31",
        filters=[],
    )

    # Create optimized engine
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

    # Plan query
    plan = engine.plan_query(query, "fct_orders")

    print(engine.planner.explain(plan))
