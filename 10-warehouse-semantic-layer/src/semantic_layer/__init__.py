"""
Warehouse Semantic Layer

A modern analytics engineering stack combining a well-designed data warehouse
with a semantic layer that provides consistent, governed business metrics.
"""

from semantic_layer.config import (
    MetricFilter,
    MetricMeta,
    ModelConfig,
    SemanticLayerConfig,
    SourceConfig,
    WarehouseConfig,
)
from semantic_layer.models import (
    CalculationMethod,
    Column,
    DbtProject,
    Dimension,
    Macro,
    MetricDefinition,
    MetricQuery,
    Model,
    QueryResult,
    Source,
    Test,
    TimeGrain,
)
from semantic_layer.query_engine import (
    BigQueryAdapter,
    InMemoryAdapter,
    MetricCatalog,
    PostgresAdapter,
    QueryExecutor,
    SemanticQueryEngine,
    SnowflakeAdapter,
    SQLiteAdapter,
    WarehouseAdapter,
    create_adapter,
    create_revenue_metric,
    create_user_metric,
)
from semantic_layer.dimensions import (
    DimCustomers,
    DimDates,
    DimProducts,
    DimensionBuilder,
    DimensionColumn,
    DimensionModel,
)
from semantic_layer.facts import (
    FactBuilder,
    FactColumn,
    FactModel,
    FctCosts,
    FctOrders,
    FctRevenue,
)
from semantic_layer.testing import (
    DataQualityChecker,
    TestBuilder,
    TestDefinition,
    TestResult,
    TestType,
    create_dimension_tests,
    create_fact_tests,
)
from semantic_layer.api import (
    MetricInfo,
    MetricQueryRequest,
    MetricQueryResponse,
    MetricRegistry,
    MetricValidator,
    SemanticLayerAPI,
)
from semantic_layer.enterprise import (
    AccessController,
    AccessLevel,
    AccessPolicy,
    CICDConfig,
    CICDIntegration,
    ColumnLineage,
    DocumentationGenerator,
    FreshnessCheck,
    FreshnessMonitor,
    FreshnessSLA,
    FreshnessStatus,
    LineageGraph,
    LineageTracker,
    SemanticLayerGovernance,
)
try:
    from semantic_layer.dbt_integration import (
        DbtColumn,
        DbtIntegration,
        DbtManifest,
        DbtMaterialization,
        DbtMetric,
        DbtModel,
        DbtProjectConfig,
        DbtResourceType,
        DbtRunner,
        DbtSource,
        ManifestParser,
        MetricSyncer,
        YamlParser,
        create_dbt_integration,
    )
except ImportError:
    # yaml (pyyaml) not installed - dbt integration not available
    DbtColumn = None
    DbtIntegration = None
    DbtManifest = None
    DbtMaterialization = None
    DbtMetric = None
    DbtModel = None
    DbtProjectConfig = None
    DbtResourceType = None
    DbtRunner = None
    DbtSource = None
    ManifestParser = None
    MetricSyncer = None
    YamlParser = None
    create_dbt_integration = None
from semantic_layer.optimization import (
    AggregateGranularity,
    AggregateTableConfig,
    AggregateTableManager,
    CacheEntry,
    CacheStats,
    CostEstimator,
    MaterializedViewAdvisor,
    MaterializedViewRecommendation,
    OptimizationResult,
    OptimizationRule,
    OptimizedQueryEngine,
    QueryCache,
    QueryCost,
    QueryPlan,
    QueryPlanner,
    RedisCache,
    TableStats,
)

__version__ = "0.1.0"

__all__ = [
    # Config
    "MetricFilter",
    "MetricMeta",
    "ModelConfig",
    "SemanticLayerConfig",
    "SourceConfig",
    "WarehouseConfig",
    # Models
    "CalculationMethod",
    "Column",
    "DbtProject",
    "Dimension",
    "Macro",
    "MetricDefinition",
    "MetricQuery",
    "Model",
    "QueryResult",
    "Source",
    "Test",
    "TimeGrain",
    # Query Engine
    "BigQueryAdapter",
    "InMemoryAdapter",
    "MetricCatalog",
    "PostgresAdapter",
    "QueryExecutor",
    "SemanticQueryEngine",
    "SnowflakeAdapter",
    "SQLiteAdapter",
    "WarehouseAdapter",
    "create_adapter",
    "create_revenue_metric",
    "create_user_metric",
    # Dimensions
    "DimCustomers",
    "DimDates",
    "DimProducts",
    "DimensionBuilder",
    "DimensionColumn",
    "DimensionModel",
    # Facts
    "FactBuilder",
    "FactColumn",
    "FactModel",
    "FctCosts",
    "FctOrders",
    "FctRevenue",
    # Testing
    "DataQualityChecker",
    "TestBuilder",
    "TestDefinition",
    "TestResult",
    "TestType",
    "create_dimension_tests",
    "create_fact_tests",
    # API
    "MetricInfo",
    "MetricQueryRequest",
    "MetricQueryResponse",
    "MetricRegistry",
    "MetricValidator",
    "SemanticLayerAPI",
    # Enterprise
    "AccessController",
    "AccessLevel",
    "AccessPolicy",
    "CICDConfig",
    "CICDIntegration",
    "ColumnLineage",
    "DocumentationGenerator",
    "FreshnessCheck",
    "FreshnessMonitor",
    "FreshnessSLA",
    "FreshnessStatus",
    "LineageGraph",
    "LineageTracker",
    "SemanticLayerGovernance",
    # dbt Integration
    "DbtColumn",
    "DbtIntegration",
    "DbtManifest",
    "DbtMaterialization",
    "DbtMetric",
    "DbtModel",
    "DbtProjectConfig",
    "DbtResourceType",
    "DbtRunner",
    "DbtSource",
    "ManifestParser",
    "MetricSyncer",
    "YamlParser",
    "create_dbt_integration",
    # Query Optimization
    "AggregateGranularity",
    "AggregateTableConfig",
    "AggregateTableManager",
    "CacheEntry",
    "CacheStats",
    "CostEstimator",
    "MaterializedViewAdvisor",
    "MaterializedViewRecommendation",
    "OptimizationResult",
    "OptimizationRule",
    "OptimizedQueryEngine",
    "QueryCache",
    "QueryCost",
    "QueryPlan",
    "QueryPlanner",
    "RedisCache",
    "TableStats",
]
