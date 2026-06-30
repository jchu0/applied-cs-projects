"""
Data Observability Platform

A comprehensive platform for monitoring data health, detecting anomalies,
tracking metadata, and providing impact analysis across the data ecosystem.
"""

from observability.config import (
    AlertConfig,
    CollectorConfig,
    DetectorConfig,
    ObservabilityConfig,
    StorageConfig,
    WarehouseConfig,
)
from observability.models import (
    Alert,
    AlertStatus,
    Anomaly,
    AnomalyType,
    ColumnMetadata,
    ColumnStats,
    DataHealthScore,
    ImpactAnalysis,
    LineageInfo,
    MetricHistory,
    Pipeline,
    SchemaChange,
    TableMetadata,
)
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
from observability.detector import (
    AnomalyDetector,
    generate_id,
)
from observability.lineage import (
    LineageEdge,
    LineageGraph,
    LineageNode,
)
from observability.alerting import (
    AlertChannel,
    AlertingEngine,
    AlertRule,
    EscalationPolicy,
    EmailChannel,
    LogChannel,
    PagerDutyChannel,
    SlackChannel,
    TeamsChannel,
    WebhookChannel,
    CRITICAL_ALERT_RULE,
    FRESHNESS_ALERT_RULE,
    SCHEMA_CHANGE_RULE,
)
from observability.sql_parser import (
    ColumnReference,
    LineageExtractor,
    SQLLineage,
    SQLParser,
    TableReference,
    extract_lineage,
    extract_tables,
)
from observability.health import (
    HealthMonitor,
    HealthScorer,
)
from observability.pii import (
    PIIDetection,
    PIIDetector,
    PIIRegistry,
)
from observability.forecaster import (
    AnomalyForecaster,
    Forecast,
    ForecastPoint,
    MetricForecaster,
)
from observability.integrations import (
    AirflowIntegration,
    AutoRemediation,
    DbtIntegration,
    IntegrationConfig,
    ObservabilityClient,
    RemediationResult,
    SparkIntegration,
    remediate_freshness,
    remediate_volume_drop,
)
from observability.llm_triage import (
    AnthropicClient,
    GeneratedRunbook,
    LLMProvider,
    LLMTriageEngine,
    MockLLMClient,
    OpenAIClient,
    RunbookStep,
    TriageConfig,
    TriageReportGenerator,
    TriageResult,
)
from observability.ml_detector import (
    AnomalyCorrelator,
    DetectionResult,
    EnsembleDetector,
    FeatureVector,
    IsolationForestDetector,
    StatisticalDetector,
)
try:
    from observability.api import app, create_app
except ImportError:
    app = None
    create_app = None

__version__ = "0.1.0"

__all__ = [
    # Config
    "AlertConfig",
    "CollectorConfig",
    "DetectorConfig",
    "ObservabilityConfig",
    "StorageConfig",
    "WarehouseConfig",
    # Models
    "Alert",
    "AlertStatus",
    "Anomaly",
    "AnomalyType",
    "ColumnMetadata",
    "ColumnStats",
    "DataHealthScore",
    "ImpactAnalysis",
    "LineageInfo",
    "MetricHistory",
    "Pipeline",
    "SchemaChange",
    "TableMetadata",
    # Collectors
    "BigQueryCollector",
    "DatabricksCollector",
    "GenericSQLCollector",
    "InMemoryCollector",
    "MetadataCollector",
    "PostgresCollector",
    "RedshiftCollector",
    "SnowflakeCollector",
    "create_collector",
    # Detector
    "AnomalyDetector",
    "generate_id",
    # Lineage
    "LineageEdge",
    "LineageGraph",
    "LineageNode",
    # Alerting
    "AlertChannel",
    "AlertingEngine",
    "AlertRule",
    "EmailChannel",
    "EscalationPolicy",
    "LogChannel",
    "PagerDutyChannel",
    "SlackChannel",
    "TeamsChannel",
    "WebhookChannel",
    "CRITICAL_ALERT_RULE",
    "FRESHNESS_ALERT_RULE",
    "SCHEMA_CHANGE_RULE",
    # SQL Parser
    "ColumnReference",
    "LineageExtractor",
    "SQLLineage",
    "SQLParser",
    "TableReference",
    "extract_lineage",
    "extract_tables",
    # Health
    "HealthMonitor",
    "HealthScorer",
    # PII
    "PIIDetection",
    "PIIDetector",
    "PIIRegistry",
    # Forecaster
    "AnomalyForecaster",
    "Forecast",
    "ForecastPoint",
    "MetricForecaster",
    # Integrations
    "AirflowIntegration",
    "AutoRemediation",
    "DbtIntegration",
    "IntegrationConfig",
    "ObservabilityClient",
    "RemediationResult",
    "SparkIntegration",
    "remediate_freshness",
    "remediate_volume_drop",
    # LLM Triage
    "AnthropicClient",
    "GeneratedRunbook",
    "LLMProvider",
    "LLMTriageEngine",
    "MockLLMClient",
    "OpenAIClient",
    "RunbookStep",
    "TriageConfig",
    "TriageReportGenerator",
    "TriageResult",
    # ML Detector
    "AnomalyCorrelator",
    "DetectionResult",
    "EnsembleDetector",
    "FeatureVector",
    "IsolationForestDetector",
    "StatisticalDetector",
    # API
    "app",
    "create_app",
]
