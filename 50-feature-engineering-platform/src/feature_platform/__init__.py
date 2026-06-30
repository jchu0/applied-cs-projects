"""
Feature Engineering Platform

A production-grade ML feature engineering platform with feature store,
transformations, pipelines, and monitoring capabilities.
"""

from feature_platform.core.models import (
    DataType,
    Entity,
    Feature,
    FeatureSchema,
    FeatureValue,
    FeatureVector,
    FeatureView,
    FeatureSource,
)
from feature_platform.core.config import (
    FeatureStoreConfig,
    OfflineStoreConfig,
    OnlineStoreConfig,
    PipelineConfig,
    ValidationConfig,
)
from feature_platform.transformers.base import (
    BaseTransformer,
    TransformerMixin,
)
from feature_platform.transformers.numeric import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    LogTransformer,
    PowerTransformer,
    Binner,
    QuantileTransformer,
    Normalizer,
)
from feature_platform.transformers.categorical import (
    OneHotEncoder,
    LabelEncoder,
    OrdinalEncoder,
    TargetEncoder,
    FrequencyEncoder,
    BinaryEncoder,
    HashingEncoder,
)
from feature_platform.transformers.temporal import (
    DatePartsExtractor,
    TimeSinceEvent,
    CyclicalEncoder,
    RollingWindowFeatures,
    LagFeatures,
    DateDiffFeatures,
    HolidayFeatures,
    TimeZoneConverter,
)
from feature_platform.transformers.text import (
    TfidfVectorizer,
    CountVectorizer,
    HashingVectorizer,
    NGramExtractor,
    TextCleaner,
    TextStatistics,
)
from feature_platform.transformers.composite import (
    Pipeline,
    FeatureUnion,
    ColumnTransformer,
    SequentialTransformer,
)
from feature_platform.store.registry import (
    FeatureRegistry,
    FeatureDefinition,
    FeatureVersion,
)
from feature_platform.store.offline import (
    OfflineStore,
    ParquetOfflineStore,
    DuckDBOfflineStore,
)
from feature_platform.store.online import (
    OnlineStore,
    RedisOnlineStore,
    InMemoryOnlineStore,
)
from feature_platform.store.feature_store import FeatureStore
from feature_platform.pipeline.dag import (
    DAG,
    DAGNode,
    DAGEdge,
    DAGExecutor,
)
from feature_platform.pipeline.executor import (
    PipelineExecutor,
    ExecutionResult,
    ExecutionStatus,
)
from feature_platform.validation.schema import (
    SchemaValidator,
    SchemaValidationResult,
)
from feature_platform.validation.statistical import (
    StatisticalValidator,
    StatisticalValidationResult,
)
from feature_platform.validation.drift import (
    DriftDetector,
    DriftResult,
    DriftType,
    DriftMethod,
    DriftReport,
)
from feature_platform.validation.advanced_drift import (
    ConceptDriftMethod,
    ConceptDriftResult,
    MultivariateDriftResult,
    WindowedDriftMonitor,
    DDMDetector,
    ADWINDetector,
    CUSUMDetector,
    MultivariateDriftDetector,
)
from feature_platform.discovery.search import (
    FeatureSearchEngine,
    SearchQuery,
    SearchResult,
    SearchFilters,
    SortOrder,
)
from feature_platform.discovery.similarity import (
    FeatureSimilarityEngine,
    SimilarityMethod,
    SimilarityResult,
    FeatureProfile,
)
from feature_platform.discovery.recommendations import (
    FeatureRecommender,
    RecommendationContext,
    FeatureRecommendation,
    RecommendationType,
)
from feature_platform.monitoring.metrics import (
    FeatureMetrics,
    MetricsCollector,
)
from feature_platform.monitoring.alerts import (
    Alert,
    AlertManager,
    AlertSeverity,
)

__version__ = "0.1.0"

# Note: API module should be imported directly from feature_platform.api
# to avoid circular imports:
#   from feature_platform.api import create_app, get_feature_store

__all__ = [
    # Core Models
    "DataType",
    "Entity",
    "Feature",
    "FeatureSchema",
    "FeatureValue",
    "FeatureVector",
    "FeatureView",
    "FeatureSource",
    # Config
    "FeatureStoreConfig",
    "OfflineStoreConfig",
    "OnlineStoreConfig",
    "PipelineConfig",
    "ValidationConfig",
    # Base Transformers
    "BaseTransformer",
    "TransformerMixin",
    # Numeric Transformers
    "StandardScaler",
    "MinMaxScaler",
    "RobustScaler",
    "LogTransformer",
    "PowerTransformer",
    "Binner",
    "QuantileTransformer",
    "Normalizer",
    # Categorical Transformers
    "OneHotEncoder",
    "LabelEncoder",
    "OrdinalEncoder",
    "TargetEncoder",
    "FrequencyEncoder",
    "BinaryEncoder",
    "HashingEncoder",
    # Temporal Transformers
    "DatePartsExtractor",
    "TimeSinceEvent",
    "CyclicalEncoder",
    "RollingWindowFeatures",
    "LagFeatures",
    "DateDiffFeatures",
    "HolidayFeatures",
    "TimeZoneConverter",
    # Text Transformers
    "TfidfVectorizer",
    "CountVectorizer",
    "HashingVectorizer",
    "NGramExtractor",
    "TextCleaner",
    "TextStatistics",
    # Composite Transformers
    "Pipeline",
    "FeatureUnion",
    "ColumnTransformer",
    "SequentialTransformer",
    # Feature Registry
    "FeatureRegistry",
    "FeatureDefinition",
    "FeatureVersion",
    # Stores
    "OfflineStore",
    "ParquetOfflineStore",
    "DuckDBOfflineStore",
    "OnlineStore",
    "RedisOnlineStore",
    "InMemoryOnlineStore",
    "FeatureStore",
    # Pipeline
    "DAG",
    "DAGNode",
    "DAGEdge",
    "DAGExecutor",
    "PipelineExecutor",
    "ExecutionResult",
    "ExecutionStatus",
    # Validation
    "SchemaValidator",
    "SchemaValidationResult",
    "StatisticalValidator",
    "StatisticalValidationResult",
    "DriftDetector",
    "DriftResult",
    "DriftType",
    "DriftMethod",
    "DriftReport",
    # Advanced Drift Detection
    "ConceptDriftMethod",
    "ConceptDriftResult",
    "MultivariateDriftResult",
    "WindowedDriftMonitor",
    "DDMDetector",
    "ADWINDetector",
    "CUSUMDetector",
    "MultivariateDriftDetector",
    # Feature Discovery
    "FeatureSearchEngine",
    "SearchQuery",
    "SearchResult",
    "SearchFilters",
    "SortOrder",
    "FeatureSimilarityEngine",
    "SimilarityMethod",
    "SimilarityResult",
    "FeatureProfile",
    "FeatureRecommender",
    "RecommendationContext",
    "FeatureRecommendation",
    "RecommendationType",
    # Monitoring
    "FeatureMetrics",
    "MetricsCollector",
    "Alert",
    "AlertManager",
    "AlertSeverity",
]
