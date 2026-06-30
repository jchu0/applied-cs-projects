# Data Observability Platform (Monte Carlo-lite) - Technical Blueprint

## Executive Summary

This project implements a comprehensive data observability platform that monitors data health, detects anomalies, tracks metadata, and provides impact analysis across the data ecosystem. Inspired by platforms like Monte Carlo and Datadog for data, it demonstrates mastery of data quality management, anomaly detection, metadata graph construction, and operational data intelligence.

> **Concepts covered:** [§05 Metrics + alerting](../../../05-cross-cutting-concerns/observability/metrics/metrics.md) · [§05 Distributed tracing](../../../05-cross-cutting-concerns/observability/distributed-tracing/distributed-tracing.md) · [§05 Logging](../../../05-cross-cutting-concerns/observability/logging/logging.md) · [§02 Real-time analytics (anomaly detection)](../../../02-data-engineering/04-streaming/real-time-analytics/real-time-analytics.md). Pairs with [Project 49 (AI benchmark suite — observability for ML systems)](../../49-ai-benchmark-suite/) and [Project 05 (SaaS observability stack)](../../05-saas-web-platform/). Map: [`CONCEPT_TO_PROJECT_MAP.md`](../../CONCEPT_TO_PROJECT_MAP.md).

**Primary Goals:**
- Build automated data quality monitoring without manual rule configuration
- Implement ML-based anomaly detection for volume, schema, and distribution
- Create a metadata graph for lineage tracking and impact analysis
- Provide actionable alerts with root cause analysis

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Data Sources & Pipelines                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ Airflow  │  │ Dagster  │  │  dbt     │  │  Spark   │            │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘            │
└───────┴─────────────┴─────────────┴─────────────┴───────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Metadata Collectors                               │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐            │
│  │    Schema     │  │    Query      │  │   Pipeline    │            │
│  │   Crawler     │  │    Logger     │  │    Events     │            │
│  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘            │
└──────────┴──────────────────┴──────────────────┴────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 Observability Platform Core                          │
│                                                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐      │
│  │   Metadata      │  │    Anomaly      │  │   Alerting      │      │
│  │    Store        │  │   Detector      │  │    Engine       │      │
│  │                 │  │                 │  │                 │      │
│  │  - Graph DB     │  │  - Volume       │  │  - Rules        │      │
│  │  - Time Series  │  │  - Schema       │  │  - Routing      │      │
│  │  - Search Index │  │  - Distribution │  │  - Escalation   │      │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘      │
│           │                    │                    │                │
│           └────────────────────┼────────────────────┘                │
│                                │                                     │
│                    ┌───────────▼───────────┐                        │
│                    │     API Gateway       │                        │
│                    └───────────┬───────────┘                        │
└────────────────────────────────┼────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         UI Dashboard                                 │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐                │
│  │Lineage  │  │ Alerts  │  │ Health  │  │ Impact  │                │
│  │ Graph   │  │  Feed   │  │ Scores  │  │Analysis │                │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘                │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Collection Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                    Collection Architecture                        │
│                                                                    │
│  Push-based:                    Pull-based:                       │
│  ┌─────────────┐                ┌─────────────┐                   │
│  │  Webhooks   │                │  Scheduled  │                   │
│  │  (events)   │                │  (crawlers) │                   │
│  └──────┬──────┘                └──────┬──────┘                   │
│         │                              │                          │
│         ▼                              ▼                          │
│  ┌────────────────────────────────────────────┐                   │
│  │            Collection Queue                │                   │
│  │         (Kafka / Redis Streams)            │                   │
│  └────────────────┬───────────────────────────┘                   │
│                   │                                               │
│                   ▼                                               │
│  ┌────────────────────────────────────────────┐                   │
│  │          Metadata Processors               │                   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐   │                   │
│  │  │ Schema   │ │ Stats    │ │ Lineage  │   │                   │
│  │  │ Parser   │ │Aggregator│ │ Builder  │   │                   │
│  │  └──────────┘ └──────────┘ └──────────┘   │                   │
│  └────────────────────────────────────────────┘                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Core Internals

### Metadata Collector

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class TableMetadata:
    """Metadata for a data table"""
    table_id: str
    database: str
    schema: str
    table_name: str
    columns: List['ColumnMetadata']
    row_count: int
    size_bytes: int
    last_modified: datetime
    partitions: List[str]
    owner: str
    tags: Dict[str, str]

@dataclass
class ColumnMetadata:
    """Metadata for a column"""
    name: str
    data_type: str
    nullable: bool
    description: Optional[str]
    stats: 'ColumnStats'

@dataclass
class ColumnStats:
    """Statistical profile for a column"""
    null_count: int
    null_ratio: float
    distinct_count: int
    min_value: Any
    max_value: Any
    mean: Optional[float]
    stddev: Optional[float]
    histogram: Optional[List[int]]

class MetadataCollector(ABC):
    """Base class for metadata collectors"""

    @abstractmethod
    async def collect_schema(self, table_ref: str) -> TableMetadata:
        """Collect schema metadata for a table"""
        pass

    @abstractmethod
    async def collect_stats(self, table_ref: str) -> Dict[str, ColumnStats]:
        """Collect column statistics for a table"""
        pass

    @abstractmethod
    async def collect_lineage(self, table_ref: str) -> 'LineageInfo':
        """Collect lineage information for a table"""
        pass

class SnowflakeCollector(MetadataCollector):
    """Collector for Snowflake data warehouse"""

    def __init__(self, connection_params: dict):
        self.conn = snowflake.connector.connect(**connection_params)

    async def collect_schema(self, table_ref: str) -> TableMetadata:
        database, schema, table = table_ref.split('.')

        # Get columns
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
        columns_df = self._execute(columns_query)

        # Get table stats
        stats_query = f"""
            SELECT
                row_count,
                bytes
            FROM {database}.information_schema.tables
            WHERE table_schema = '{schema}'
              AND table_name = '{table}'
        """
        stats_df = self._execute(stats_query)

        return TableMetadata(
            table_id=f"{database}.{schema}.{table}",
            database=database,
            schema=schema,
            table_name=table,
            columns=[
                ColumnMetadata(
                    name=row['column_name'],
                    data_type=row['data_type'],
                    nullable=row['is_nullable'] == 'YES',
                    description=row['comment'],
                    stats=None  # Populated separately
                )
                for _, row in columns_df.iterrows()
            ],
            row_count=stats_df['row_count'].iloc[0],
            size_bytes=stats_df['bytes'].iloc[0],
            last_modified=datetime.now(),
            partitions=[],
            owner='',
            tags={}
        )

    async def collect_stats(self, table_ref: str) -> Dict[str, ColumnStats]:
        database, schema, table = table_ref.split('.')

        # Profile query for each column
        stats = {}
        schema_info = await self.collect_schema(table_ref)

        for col in schema_info.columns:
            col_stats = await self._profile_column(table_ref, col.name, col.data_type)
            stats[col.name] = col_stats

        return stats

    async def _profile_column(
        self,
        table_ref: str,
        column_name: str,
        data_type: str
    ) -> ColumnStats:
        # Build profiling query based on data type
        if data_type in ('NUMBER', 'FLOAT', 'INTEGER'):
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

        result = self._execute(query)
        row = result.iloc[0]

        return ColumnStats(
            null_count=int(row['null_count']),
            null_ratio=row['null_count'] / row['total_count'] if row['total_count'] > 0 else 0,
            distinct_count=int(row['distinct_count']),
            min_value=row['min_value'],
            max_value=row['max_value'],
            mean=float(row['mean']) if row['mean'] else None,
            stddev=float(row['stddev']) if row['stddev'] else None,
            histogram=None
        )
```

### Anomaly Detection Engine

```python
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from typing import Tuple, List
from enum import Enum

class AnomalyType(Enum):
    VOLUME = "volume"
    FRESHNESS = "freshness"
    SCHEMA = "schema"
    DISTRIBUTION = "distribution"
    NULL_RATE = "null_rate"

@dataclass
class Anomaly:
    """Detected anomaly"""
    anomaly_id: str
    table_id: str
    column_name: Optional[str]
    anomaly_type: AnomalyType
    severity: str  # critical, warning, info
    detected_at: datetime
    metric_value: float
    expected_range: Tuple[float, float]
    description: str
    context: Dict[str, Any]

class AnomalyDetector:
    """ML-based anomaly detection for data quality"""

    def __init__(self, config: 'DetectorConfig'):
        self.config = config
        self.models: Dict[str, Any] = {}
        self.history: Dict[str, List[float]] = {}

    async def detect_volume_anomaly(
        self,
        table_id: str,
        current_count: int
    ) -> Optional[Anomaly]:
        """Detect anomalies in row count"""
        history = await self._get_volume_history(table_id)

        if len(history) < self.config.min_history_points:
            return None

        # Calculate statistics
        mean = np.mean(history)
        std = np.std(history)

        # Z-score based detection
        z_score = (current_count - mean) / std if std > 0 else 0

        if abs(z_score) > self.config.volume_threshold:
            severity = "critical" if abs(z_score) > 5 else "warning"

            return Anomaly(
                anomaly_id=generate_id(),
                table_id=table_id,
                column_name=None,
                anomaly_type=AnomalyType.VOLUME,
                severity=severity,
                detected_at=datetime.now(),
                metric_value=current_count,
                expected_range=(mean - 3*std, mean + 3*std),
                description=f"Row count {current_count} is {abs(z_score):.1f} standard deviations from expected {mean:.0f}",
                context={
                    "z_score": z_score,
                    "historical_mean": mean,
                    "historical_std": std,
                    "history": history[-10:]
                }
            )

        return None

    async def detect_freshness_anomaly(
        self,
        table_id: str,
        last_updated: datetime
    ) -> Optional[Anomaly]:
        """Detect anomalies in data freshness"""
        history = await self._get_freshness_history(table_id)

        if len(history) < self.config.min_history_points:
            return None

        # Calculate typical update interval
        intervals = np.diff(history)
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)

        current_interval = (datetime.now() - last_updated).total_seconds()

        if current_interval > mean_interval + 3 * std_interval:
            delay_hours = (current_interval - mean_interval) / 3600

            return Anomaly(
                anomaly_id=generate_id(),
                table_id=table_id,
                column_name=None,
                anomaly_type=AnomalyType.FRESHNESS,
                severity="critical" if delay_hours > 24 else "warning",
                detected_at=datetime.now(),
                metric_value=current_interval,
                expected_range=(0, mean_interval + 3*std_interval),
                description=f"Table has not been updated for {delay_hours:.1f} hours longer than expected",
                context={
                    "last_updated": last_updated.isoformat(),
                    "expected_interval_hours": mean_interval / 3600,
                    "actual_interval_hours": current_interval / 3600
                }
            )

        return None

    async def detect_schema_anomaly(
        self,
        table_id: str,
        current_schema: TableMetadata
    ) -> Optional[Anomaly]:
        """Detect schema changes"""
        previous_schema = await self._get_previous_schema(table_id)

        if not previous_schema:
            return None

        changes = self._diff_schemas(previous_schema, current_schema)

        if changes:
            severity = "critical" if any(
                c['type'] in ('column_removed', 'type_changed') for c in changes
            ) else "warning"

            return Anomaly(
                anomaly_id=generate_id(),
                table_id=table_id,
                column_name=None,
                anomaly_type=AnomalyType.SCHEMA,
                severity=severity,
                detected_at=datetime.now(),
                metric_value=len(changes),
                expected_range=(0, 0),
                description=f"Schema change detected: {len(changes)} modification(s)",
                context={
                    "changes": changes,
                    "previous_column_count": len(previous_schema.columns),
                    "current_column_count": len(current_schema.columns)
                }
            )

        return None

    async def detect_distribution_anomaly(
        self,
        table_id: str,
        column_name: str,
        current_stats: ColumnStats
    ) -> Optional[Anomaly]:
        """Detect anomalies in column value distribution"""
        history = await self._get_distribution_history(table_id, column_name)

        if not history or len(history) < self.config.min_history_points:
            return None

        # Use Isolation Forest for multivariate anomaly detection
        features = np.array([
            [h['null_ratio'], h['distinct_ratio'], h['mean'] or 0, h['stddev'] or 0]
            for h in history
        ])

        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(features)

        current_features = np.array([[
            current_stats.null_ratio,
            current_stats.distinct_count / (current_stats.null_count + current_stats.distinct_count),
            current_stats.mean or 0,
            current_stats.stddev or 0
        ]])

        prediction = model.predict(current_features)

        if prediction[0] == -1:  # Anomaly
            return Anomaly(
                anomaly_id=generate_id(),
                table_id=table_id,
                column_name=column_name,
                anomaly_type=AnomalyType.DISTRIBUTION,
                severity="warning",
                detected_at=datetime.now(),
                metric_value=model.score_samples(current_features)[0],
                expected_range=(-1, 0),
                description=f"Unusual distribution detected for column {column_name}",
                context={
                    "current_stats": {
                        "null_ratio": current_stats.null_ratio,
                        "distinct_count": current_stats.distinct_count,
                        "mean": current_stats.mean,
                        "stddev": current_stats.stddev
                    },
                    "isolation_score": float(model.score_samples(current_features)[0])
                }
            )

        return None

    async def detect_null_rate_anomaly(
        self,
        table_id: str,
        column_name: str,
        current_null_ratio: float
    ) -> Optional[Anomaly]:
        """Detect anomalies in null rate"""
        history = await self._get_null_rate_history(table_id, column_name)

        if len(history) < self.config.min_history_points:
            return None

        mean = np.mean(history)
        std = np.std(history)

        # Check for significant increase in null rate
        if current_null_ratio > mean + 3 * std:
            return Anomaly(
                anomaly_id=generate_id(),
                table_id=table_id,
                column_name=column_name,
                anomaly_type=AnomalyType.NULL_RATE,
                severity="warning" if current_null_ratio < 0.5 else "critical",
                detected_at=datetime.now(),
                metric_value=current_null_ratio,
                expected_range=(max(0, mean - 3*std), min(1, mean + 3*std)),
                description=f"Null rate for {column_name} increased to {current_null_ratio:.1%} (expected ~{mean:.1%})",
                context={
                    "historical_mean": mean,
                    "historical_std": std,
                    "increase_factor": current_null_ratio / mean if mean > 0 else float('inf')
                }
            )

        return None
```

### Lineage Graph

```python
from neo4j import GraphDatabase
from typing import List, Set

class LineageGraph:
    """Graph-based lineage tracking using Neo4j"""

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def add_table(self, table: TableMetadata):
        """Add or update a table node"""
        with self.driver.session() as session:
            session.run("""
                MERGE (t:Table {id: $id})
                SET t.database = $database,
                    t.schema = $schema,
                    t.name = $name,
                    t.row_count = $row_count,
                    t.updated_at = datetime()
            """, {
                'id': table.table_id,
                'database': table.database,
                'schema': table.schema,
                'name': table.table_name,
                'row_count': table.row_count
            })

            # Add columns
            for col in table.columns:
                session.run("""
                    MATCH (t:Table {id: $table_id})
                    MERGE (c:Column {id: $col_id})
                    SET c.name = $name,
                        c.data_type = $data_type
                    MERGE (t)-[:HAS_COLUMN]->(c)
                """, {
                    'table_id': table.table_id,
                    'col_id': f"{table.table_id}.{col.name}",
                    'name': col.name,
                    'data_type': col.data_type
                })

    def add_lineage(
        self,
        source_table: str,
        target_table: str,
        transformation: str = None,
        column_mappings: List[dict] = None
    ):
        """Add a lineage edge between tables"""
        with self.driver.session() as session:
            # Table-level lineage
            session.run("""
                MATCH (source:Table {id: $source})
                MATCH (target:Table {id: $target})
                MERGE (source)-[r:FEEDS_INTO]->(target)
                SET r.transformation = $transformation,
                    r.created_at = datetime()
            """, {
                'source': source_table,
                'target': target_table,
                'transformation': transformation
            })

            # Column-level lineage
            if column_mappings:
                for mapping in column_mappings:
                    session.run("""
                        MATCH (sc:Column {id: $source_col})
                        MATCH (tc:Column {id: $target_col})
                        MERGE (sc)-[r:MAPS_TO]->(tc)
                        SET r.transformation = $transformation
                    """, {
                        'source_col': f"{source_table}.{mapping['source']}",
                        'target_col': f"{target_table}.{mapping['target']}",
                        'transformation': mapping.get('transformation')
                    })

    def get_upstream(self, table_id: str, depth: int = 3) -> List[dict]:
        """Get upstream dependencies"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH path = (t:Table {id: $id})<-[:FEEDS_INTO*1..%d]-(upstream:Table)
                RETURN upstream.id as table_id,
                       length(path) as distance,
                       [rel in relationships(path) | rel.transformation] as transformations
                ORDER BY distance
            """ % depth, {'id': table_id})

            return [dict(record) for record in result]

    def get_downstream(self, table_id: str, depth: int = 3) -> List[dict]:
        """Get downstream dependencies"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH path = (t:Table {id: $id})-[:FEEDS_INTO*1..%d]->(downstream:Table)
                RETURN downstream.id as table_id,
                       length(path) as distance,
                       [rel in relationships(path) | rel.transformation] as transformations
                ORDER BY distance
            """ % depth, {'id': table_id})

            return [dict(record) for record in result]

    def get_impact_analysis(self, table_id: str) -> 'ImpactAnalysis':
        """Analyze impact of changes to a table"""
        downstream = self.get_downstream(table_id, depth=10)

        # Get downstream tables with their metadata
        affected_tables = []
        affected_pipelines = set()
        affected_dashboards = set()

        for item in downstream:
            table_info = self._get_table_info(item['table_id'])
            affected_tables.append(table_info)

            # Track pipelines
            pipelines = self._get_pipelines_for_table(item['table_id'])
            affected_pipelines.update(pipelines)

            # Track dashboards
            dashboards = self._get_dashboards_for_table(item['table_id'])
            affected_dashboards.update(dashboards)

        return ImpactAnalysis(
            source_table=table_id,
            affected_tables=affected_tables,
            affected_pipelines=list(affected_pipelines),
            affected_dashboards=list(affected_dashboards),
            total_downstream=len(downstream)
        )

    def get_column_lineage(self, column_id: str) -> dict:
        """Get column-level lineage"""
        with self.driver.session() as session:
            # Upstream columns
            upstream = session.run("""
                MATCH (c:Column {id: $id})<-[:MAPS_TO*1..5]-(upstream:Column)
                RETURN upstream.id as column_id,
                       upstream.name as name
            """, {'id': column_id})

            # Downstream columns
            downstream = session.run("""
                MATCH (c:Column {id: $id})-[:MAPS_TO*1..5]->(downstream:Column)
                RETURN downstream.id as column_id,
                       downstream.name as name
            """, {'id': column_id})

            return {
                'upstream': [dict(r) for r in upstream],
                'downstream': [dict(r) for r in downstream]
            }
```

### Alerting Engine

```python
from typing import List, Dict
import asyncio

class AlertingEngine:
    """Alert management and routing"""

    def __init__(self, config: AlertConfig):
        self.config = config
        self.channels: Dict[str, AlertChannel] = {}
        self.rules: List[AlertRule] = []

    def add_channel(self, name: str, channel: 'AlertChannel'):
        """Register an alert channel"""
        self.channels[name] = channel

    def add_rule(self, rule: 'AlertRule'):
        """Add an alerting rule"""
        self.rules.append(rule)

    async def process_anomaly(self, anomaly: Anomaly):
        """Process an anomaly and send alerts"""
        # Find matching rules
        matching_rules = [
            rule for rule in self.rules
            if rule.matches(anomaly)
        ]

        if not matching_rules:
            # Use default rule
            matching_rules = [self.config.default_rule]

        # Create alert
        alert = Alert(
            alert_id=generate_id(),
            anomaly=anomaly,
            created_at=datetime.now(),
            status='active'
        )

        # Route to channels
        for rule in matching_rules:
            channels = rule.get_channels(anomaly)

            for channel_name in channels:
                if channel_name in self.channels:
                    await self.channels[channel_name].send(alert)

        # Store alert
        await self._store_alert(alert)

        # Check escalation
        await self._check_escalation(alert, matching_rules)

    async def _check_escalation(self, alert: Alert, rules: List['AlertRule']):
        """Check if alert should be escalated"""
        for rule in rules:
            if rule.escalation_policy:
                asyncio.create_task(
                    self._escalation_timer(alert, rule.escalation_policy)
                )

    async def _escalation_timer(self, alert: Alert, policy: 'EscalationPolicy'):
        """Wait and escalate if not acknowledged"""
        await asyncio.sleep(policy.escalate_after_minutes * 60)

        # Check if still active
        current_alert = await self._get_alert(alert.alert_id)

        if current_alert.status == 'active':
            # Escalate
            for channel_name in policy.escalate_to:
                if channel_name in self.channels:
                    await self.channels[channel_name].send(alert, escalated=True)

class AlertRule:
    """Rule for alert routing"""

    def __init__(
        self,
        name: str,
        conditions: Dict[str, Any],
        channels: List[str],
        escalation_policy: 'EscalationPolicy' = None
    ):
        self.name = name
        self.conditions = conditions
        self.channels_list = channels
        self.escalation_policy = escalation_policy

    def matches(self, anomaly: Anomaly) -> bool:
        """Check if anomaly matches this rule"""
        # Check anomaly type
        if 'anomaly_types' in self.conditions:
            if anomaly.anomaly_type not in self.conditions['anomaly_types']:
                return False

        # Check severity
        if 'severities' in self.conditions:
            if anomaly.severity not in self.conditions['severities']:
                return False

        # Check table patterns
        if 'table_patterns' in self.conditions:
            if not any(
                pattern in anomaly.table_id
                for pattern in self.conditions['table_patterns']
            ):
                return False

        return True

    def get_channels(self, anomaly: Anomaly) -> List[str]:
        """Get channels for this anomaly"""
        return self.channels_list

class SlackChannel:
    """Slack alert channel"""

    def __init__(self, webhook_url: str, channel: str):
        self.webhook_url = webhook_url
        self.channel = channel

    async def send(self, alert: Alert, escalated: bool = False):
        """Send alert to Slack"""
        emoji = {
            'critical': ':rotating_light:',
            'warning': ':warning:',
            'info': ':information_source:'
        }.get(alert.anomaly.severity, ':grey_question:')

        prefix = '[ESCALATED] ' if escalated else ''

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} {prefix}Data Anomaly Detected"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Table:*\n{alert.anomaly.table_id}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Type:*\n{alert.anomaly.anomaly_type.value}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Severity:*\n{alert.anomaly.severity}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Detected:*\n{alert.anomaly.detected_at.strftime('%Y-%m-%d %H:%M:%S')}"
                    }
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Description:*\n{alert.anomaly.description}"
                }
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "View Details"},
                        "url": f"https://observability.company.com/alerts/{alert.alert_id}"
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Acknowledge"},
                        "action_id": f"ack_{alert.alert_id}"
                    }
                ]
            }
        ]

        async with aiohttp.ClientSession() as session:
            await session.post(
                self.webhook_url,
                json={"channel": self.channel, "blocks": blocks}
            )
```

---

## Data Structures

### Core Entities

```python
@dataclass
class Alert:
    """Alert entity"""
    alert_id: str
    anomaly: Anomaly
    created_at: datetime
    status: str  # active, acknowledged, resolved
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None

@dataclass
class ImpactAnalysis:
    """Impact analysis result"""
    source_table: str
    affected_tables: List[TableMetadata]
    affected_pipelines: List[str]
    affected_dashboards: List[str]
    total_downstream: int

@dataclass
class DataHealthScore:
    """Health score for a data asset"""
    table_id: str
    overall_score: float  # 0-100
    freshness_score: float
    volume_score: float
    schema_score: float
    quality_score: float
    calculated_at: datetime
    factors: List[dict]

@dataclass
class Pipeline:
    """Data pipeline entity"""
    pipeline_id: str
    name: str
    orchestrator: str  # airflow, dagster, etc.
    schedule: str
    source_tables: List[str]
    target_tables: List[str]
    last_run: Optional[datetime]
    last_status: Optional[str]
```

---

## API Design

### REST API

```yaml
openapi: 3.0.0
info:
  title: Data Observability API
  version: 1.0.0

paths:
  /tables:
    get:
      summary: List monitored tables
      parameters:
        - name: database
          in: query
          schema:
            type: string
        - name: search
          in: query
          schema:
            type: string
      responses:
        200:
          description: List of tables

  /tables/{table_id}:
    get:
      summary: Get table details
      responses:
        200:
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/TableDetails'

  /tables/{table_id}/lineage:
    get:
      summary: Get table lineage
      parameters:
        - name: direction
          in: query
          schema:
            type: string
            enum: [upstream, downstream, both]
        - name: depth
          in: query
          schema:
            type: integer
            default: 3
      responses:
        200:
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Lineage'

  /tables/{table_id}/health:
    get:
      summary: Get table health score
      responses:
        200:
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/HealthScore'

  /tables/{table_id}/anomalies:
    get:
      summary: Get anomalies for table
      parameters:
        - name: start_date
          in: query
          schema:
            type: string
            format: date
        - name: end_date
          in: query
          schema:
            type: string
            format: date
      responses:
        200:
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/Anomaly'

  /anomalies:
    get:
      summary: List all anomalies
      parameters:
        - name: status
          in: query
          schema:
            type: string
            enum: [active, acknowledged, resolved]
        - name: severity
          in: query
          schema:
            type: string
            enum: [critical, warning, info]

  /anomalies/{anomaly_id}/acknowledge:
    post:
      summary: Acknowledge an anomaly
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                notes:
                  type: string

  /impact/{table_id}:
    get:
      summary: Get impact analysis
      responses:
        200:
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ImpactAnalysis'

  /pipelines:
    get:
      summary: List pipelines
      responses:
        200:
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/Pipeline'

  /search:
    get:
      summary: Search across all entities
      parameters:
        - name: q
          in: query
          required: true
          schema:
            type: string
```

### Python SDK

```python
class ObservabilityClient:
    """Client for Data Observability Platform"""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key

    # Table operations
    def get_table(self, table_id: str) -> TableDetails:
        """Get table details"""
        pass

    def get_table_lineage(
        self,
        table_id: str,
        direction: str = 'both',
        depth: int = 3
    ) -> Lineage:
        """Get table lineage"""
        pass

    def get_table_health(self, table_id: str) -> HealthScore:
        """Get table health score"""
        pass

    # Anomaly operations
    def get_anomalies(
        self,
        table_id: str = None,
        status: str = None,
        severity: str = None
    ) -> List[Anomaly]:
        """Get anomalies"""
        pass

    def acknowledge_anomaly(
        self,
        anomaly_id: str,
        notes: str = None
    ) -> None:
        """Acknowledge an anomaly"""
        pass

    # Impact analysis
    def get_impact(self, table_id: str) -> ImpactAnalysis:
        """Get impact analysis"""
        pass

    # Metadata management
    def add_table_tags(self, table_id: str, tags: Dict[str, str]) -> None:
        """Add tags to a table"""
        pass

    def add_column_description(
        self,
        table_id: str,
        column_name: str,
        description: str
    ) -> None:
        """Add description to a column"""
        pass
```

---

## Enterprise Features

### 1. PII Detection

```python
import re
from typing import List, Tuple

class PIIDetector:
    """Detect PII in column names and data"""

    def __init__(self):
        self.patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone': r'^\+?1?\d{9,15}$',
            'ssn': r'^\d{3}-\d{2}-\d{4}$',
            'credit_card': r'^\d{13,16}$',
            'ip_address': r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
        }

        self.suspicious_names = [
            'ssn', 'social_security', 'password', 'pwd', 'secret',
            'credit_card', 'card_number', 'cvv', 'pin',
            'email', 'phone', 'mobile', 'address', 'dob', 'birth_date',
            'first_name', 'last_name', 'full_name', 'name',
            'ip_address', 'user_agent', 'location'
        ]

    def detect_in_column_name(self, column_name: str) -> List[str]:
        """Detect PII indicators in column name"""
        detections = []
        name_lower = column_name.lower()

        for suspicious in self.suspicious_names:
            if suspicious in name_lower:
                detections.append(f"Column name contains '{suspicious}'")

        return detections

    def detect_in_sample(self, column_name: str, sample: List[str]) -> List[Tuple[str, float]]:
        """Detect PII patterns in data sample"""
        detections = []

        for pii_type, pattern in self.patterns.items():
            matches = sum(1 for value in sample if re.match(pattern, str(value)))
            ratio = matches / len(sample) if sample else 0

            if ratio > 0.5:  # More than 50% match
                detections.append((pii_type, ratio))

        return detections

    async def scan_table(self, table_id: str, collector: MetadataCollector) -> List[PIIDetection]:
        """Scan a table for PII"""
        schema = await collector.collect_schema(table_id)
        detections = []

        for column in schema.columns:
            # Check column name
            name_indicators = self.detect_in_column_name(column.name)

            # Sample data
            sample = await collector.get_column_sample(table_id, column.name, 1000)
            data_indicators = self.detect_in_sample(column.name, sample)

            if name_indicators or data_indicators:
                detections.append(PIIDetection(
                    table_id=table_id,
                    column_name=column.name,
                    name_indicators=name_indicators,
                    data_indicators=data_indicators,
                    confidence=self._calculate_confidence(name_indicators, data_indicators)
                ))

        return detections
```

### 2. Orchestrator Integration

```python
from airflow.hooks.base import BaseHook
from airflow.models import Variable
from airflow.operators.python import PythonOperator

class AirflowIntegration:
    """Integrate with Apache Airflow"""

    def __init__(self, observability_client: ObservabilityClient):
        self.client = observability_client

    def create_lineage_callback(self, source_tables: List[str], target_table: str):
        """Create callback for lineage tracking"""
        def callback(context):
            dag_id = context['dag'].dag_id
            task_id = context['task'].task_id
            run_id = context['run_id']

            for source in source_tables:
                self.client.add_lineage(
                    source_table=source,
                    target_table=target_table,
                    transformation=f"airflow:{dag_id}.{task_id}",
                    context={
                        'run_id': run_id,
                        'execution_date': str(context['execution_date'])
                    }
                )

        return callback

    def create_quality_check_operator(
        self,
        task_id: str,
        table_id: str,
        expectations: List[dict]
    ) -> PythonOperator:
        """Create operator for quality checks"""
        def check_quality(**context):
            # Run anomaly detection
            anomalies = self.client.check_table(table_id)

            if anomalies:
                for anomaly in anomalies:
                    if anomaly.severity == 'critical':
                        raise Exception(f"Critical anomaly: {anomaly.description}")

        return PythonOperator(
            task_id=task_id,
            python_callable=check_quality,
            provide_context=True
        )

    def create_observability_sensor(
        self,
        task_id: str,
        table_id: str,
        max_staleness_hours: int = 24
    ):
        """Sensor to wait for fresh data"""
        from airflow.sensors.base import BaseSensorOperator

        class FreshnessSensor(BaseSensorOperator):
            def poke(self, context):
                health = self.client.get_table_health(table_id)
                return health.freshness_score > 80

        return FreshnessSensor(
            task_id=task_id,
            poke_interval=300,
            timeout=3600
        )
```

### 3. Time-Series Forecasting

```python
from prophet import Prophet
import pandas as pd

class MetricForecaster:
    """Forecast metrics for proactive alerting"""

    def __init__(self):
        self.models: Dict[str, Prophet] = {}

    def train(self, metric_name: str, history: pd.DataFrame):
        """Train a forecasting model"""
        # Prophet expects 'ds' (date) and 'y' (value) columns
        df = history.rename(columns={'timestamp': 'ds', 'value': 'y'})

        model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            daily_seasonality=True,
            weekly_seasonality=True
        )

        model.fit(df)
        self.models[metric_name] = model

    def forecast(
        self,
        metric_name: str,
        periods: int = 24,
        freq: str = 'H'
    ) -> pd.DataFrame:
        """Generate forecast"""
        if metric_name not in self.models:
            raise ValueError(f"No model trained for {metric_name}")

        model = self.models[metric_name]
        future = model.make_future_dataframe(periods=periods, freq=freq)
        forecast = model.predict(future)

        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    async def detect_future_anomalies(
        self,
        table_id: str,
        metric_name: str,
        hours_ahead: int = 24
    ) -> List[Anomaly]:
        """Detect potential future anomalies"""
        forecast = self.forecast(metric_name, periods=hours_ahead)

        anomalies = []
        current_value = await self._get_current_value(table_id, metric_name)

        # Check if current value is outside forecast bounds
        latest_forecast = forecast.iloc[-1]

        if current_value < latest_forecast['yhat_lower'] or current_value > latest_forecast['yhat_upper']:
            anomalies.append(Anomaly(
                anomaly_id=generate_id(),
                table_id=table_id,
                column_name=None,
                anomaly_type=AnomalyType.VOLUME,
                severity="warning",
                detected_at=datetime.now(),
                metric_value=current_value,
                expected_range=(latest_forecast['yhat_lower'], latest_forecast['yhat_upper']),
                description=f"Forecasted anomaly: {metric_name} trending outside expected range",
                context={
                    'forecast': forecast.tail(24).to_dict(),
                    'current_value': current_value
                }
            ))

        return anomalies
```

---

## Stretch Goals

### 1. Automatic Remediation

```python
class AutoRemediation:
    """Automatic remediation for common issues"""

    def __init__(self, config: RemediationConfig):
        self.config = config
        self.remediation_handlers: Dict[str, Callable] = {}

    def register_handler(
        self,
        anomaly_type: AnomalyType,
        handler: Callable[[Anomaly], bool]
    ):
        """Register a remediation handler"""
        self.remediation_handlers[anomaly_type] = handler

    async def attempt_remediation(self, anomaly: Anomaly) -> RemediationResult:
        """Attempt automatic remediation"""
        handler = self.remediation_handlers.get(anomaly.anomaly_type)

        if not handler:
            return RemediationResult(
                success=False,
                message="No handler registered for this anomaly type"
            )

        try:
            success = await handler(anomaly)

            if success:
                return RemediationResult(
                    success=True,
                    message="Automatic remediation successful"
                )
            else:
                return RemediationResult(
                    success=False,
                    message="Remediation attempted but unsuccessful"
                )
        except Exception as e:
            return RemediationResult(
                success=False,
                message=f"Remediation failed: {str(e)}"
            )

# Example remediation handlers
async def remediate_freshness(anomaly: Anomaly) -> bool:
    """Trigger pipeline re-run for stale data"""
    pipeline = await get_pipeline_for_table(anomaly.table_id)

    if pipeline:
        # Trigger Airflow DAG
        await trigger_dag_run(pipeline.pipeline_id)
        return True

    return False

async def remediate_volume_drop(anomaly: Anomaly) -> bool:
    """Check upstream and retry"""
    # Check upstream tables
    upstream = await get_upstream_tables(anomaly.table_id)

    for table in upstream:
        health = await get_table_health(table)
        if health.overall_score < 80:
            # Upstream issue - trigger that pipeline
            await trigger_pipeline_for_table(table)
            return True

    return False
```

### 2. LLM-Based Issue Triage

```python
class LLMTriage:
    """LLM-powered issue analysis and triage"""

    def __init__(self, llm_client):
        self.llm = llm_client

    async def analyze_anomaly(self, anomaly: Anomaly) -> TriageResult:
        """Use LLM to analyze and triage anomaly"""
        # Gather context
        context = await self._gather_context(anomaly)

        prompt = f"""
        Analyze this data anomaly and provide triage recommendations:

        Anomaly Details:
        - Type: {anomaly.anomaly_type.value}
        - Table: {anomaly.table_id}
        - Description: {anomaly.description}
        - Metric Value: {anomaly.metric_value}
        - Expected Range: {anomaly.expected_range}

        Context:
        - Recent Changes: {context['recent_changes']}
        - Upstream Status: {context['upstream_status']}
        - Historical Patterns: {context['historical_patterns']}
        - Related Anomalies: {context['related_anomalies']}

        Provide:
        1. Root cause analysis (most likely causes)
        2. Impact assessment
        3. Recommended actions
        4. Suggested owner/team
        """

        response = await self.llm.complete(prompt)

        return TriageResult(
            anomaly_id=anomaly.anomaly_id,
            analysis=response.root_cause,
            impact=response.impact,
            recommendations=response.actions,
            suggested_owner=response.owner
        )

    async def generate_runbook(self, anomaly_type: AnomalyType) -> str:
        """Generate runbook for anomaly type"""
        # Get historical resolutions
        resolutions = await self._get_historical_resolutions(anomaly_type)

        prompt = f"""
        Generate a runbook for handling {anomaly_type.value} anomalies.

        Historical resolutions:
        {resolutions}

        Include:
        1. Initial investigation steps
        2. Common root causes
        3. Resolution procedures
        4. Escalation criteria
        """

        return await self.llm.complete(prompt)
```

---

## Testing Strategy

### Unit Tests

```python
import pytest
from unittest.mock import Mock, AsyncMock

class TestAnomalyDetector:
    @pytest.fixture
    def detector(self):
        config = DetectorConfig(
            min_history_points=10,
            volume_threshold=3.0
        )
        return AnomalyDetector(config)

    async def test_volume_anomaly_detection(self, detector):
        # Mock history
        detector._get_volume_history = AsyncMock(return_value=[100, 102, 98, 101, 99, 103, 97, 100, 101, 99])

        # Normal value
        anomaly = await detector.detect_volume_anomaly("test.table", 105)
        assert anomaly is None

        # Anomalous value
        anomaly = await detector.detect_volume_anomaly("test.table", 50)
        assert anomaly is not None
        assert anomaly.anomaly_type == AnomalyType.VOLUME
        assert anomaly.severity in ['warning', 'critical']

    async def test_null_rate_anomaly(self, detector):
        detector._get_null_rate_history = AsyncMock(return_value=[0.01, 0.02, 0.01, 0.015, 0.02, 0.01, 0.02, 0.015, 0.01, 0.02])

        # Significant increase
        anomaly = await detector.detect_null_rate_anomaly("test.table", "column", 0.5)
        assert anomaly is not None
        assert anomaly.anomaly_type == AnomalyType.NULL_RATE

class TestLineageGraph:
    @pytest.fixture
    def graph(self):
        # Use test Neo4j instance
        return LineageGraph("bolt://localhost:7687", "neo4j", "test")

    def test_add_and_query_lineage(self, graph):
        # Add tables
        graph.add_table(create_test_table("source"))
        graph.add_table(create_test_table("target"))

        # Add lineage
        graph.add_lineage("source", "target", "dbt:model")

        # Query
        downstream = graph.get_downstream("source")
        assert len(downstream) == 1
        assert downstream[0]['table_id'] == "target"
```

### Integration Tests

```python
class TestObservabilityPlatform:
    @pytest.fixture
    async def platform(self):
        config = PlatformConfig(...)
        platform = ObservabilityPlatform(config)
        await platform.start()
        yield platform
        await platform.stop()

    async def test_end_to_end_anomaly_detection(self, platform):
        # Add table to monitoring
        await platform.add_table("test.schema.table", SnowflakeCollector(...))

        # Simulate data
        await platform.collect_metadata("test.schema.table")

        # Check for anomalies
        anomalies = await platform.get_anomalies("test.schema.table")
        assert isinstance(anomalies, list)

    async def test_alert_routing(self, platform):
        # Register test channel
        test_channel = TestChannel()
        platform.alerting.add_channel("test", test_channel)

        # Create anomaly
        anomaly = create_test_anomaly(severity="critical")

        # Process
        await platform.alerting.process_anomaly(anomaly)

        # Verify alert sent
        assert len(test_channel.sent_alerts) == 1
```

---

## Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
- Core data model and API
- Basic metadata collector (single warehouse)
- Time-series storage setup
- Simple anomaly detection (volume, freshness)

### Phase 2: Detection Engine (Weeks 3-4)
- ML-based anomaly detection
- Schema change detection
- Distribution analysis
- Historical pattern learning

### Phase 3: Lineage and Graph (Weeks 5-6)
- Neo4j lineage graph
- Column-level lineage
- Impact analysis
- Query lineage extraction

### Phase 4: Alerting and UI (Weeks 7-8)
- Alerting engine with routing
- Slack/PagerDuty integration
- Web dashboard
- Lineage visualization

### Phase 5: Enterprise Features (Weeks 9-10)
- PII detection
- Orchestrator integration
- Time-series forecasting
- Auto-remediation

---

## References

- [Monte Carlo Data](https://www.montecarlodata.com/)
- [Great Expectations](https://greatexpectations.io/)
- [Apache Atlas](https://atlas.apache.org/)
- [Amundsen](https://www.amundsen.io/)
- [DataHub](https://datahubproject.io/)
- [OpenLineage](https://openlineage.io/)
