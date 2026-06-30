"""
Real-Time Streaming Platform

A production-grade streaming platform built on Kafka and Flink.
Supports exactly-once semantics, schema registry, and complex event processing.
"""

from streaming.config import (
    KafkaConfig,
    ProducerConfig,
    ConsumerConfig,
    FlinkConfig,
    TopicConfig,
    SchemaRegistryConfig,
    StreamingPlatformConfig,
)
from streaming.producer import (
    StreamingProducer,
    TransactionalProducer,
    TopicManager,
    DeliveryReport,
)
from streaming.consumer import (
    StreamingConsumer,
    ConsumerGroup,
    Message,
)
from streaming.serializers import (
    AvroSerializer,
    AvroDeserializer,
    JsonSerializer,
    JsonDeserializer,
    create_event_serializer,
    create_event_deserializer,
)
from streaming.flink_jobs import (
    FlinkJobBuilder,
    SourceConfig,
    SinkConfig,
    WindowConfig,
    create_simple_pipeline,
)
from streaming.windowing import (
    Window,
    WindowedValue,
    TumblingWindowAssigner,
    SlidingWindowAssigner,
    SessionWindowAssigner,
    Watermark,
    WatermarkGenerator,
    BoundedOutOfOrdernessGenerator,
    WindowOperator,
    AggregationResult,
)
from streaming.state import (
    StateBackend,
    InMemoryStateBackend,
    ValueState,
    ListState,
    MapState,
    TimerService,
    Timer,
    CheckpointCoordinator,
    StateContext,
)
from streaming.patterns import (
    Deduplicator,
    DeduplicationConfig,
    Debouncer,
    DebouncerConfig,
    Session,
    SessionProcessor,
    Pattern,
    PatternMatch,
    CEPEngine,
    StreamJoiner,
    JoinConfig,
    IntervalJoiner,
)
from streaming.enterprise import (
    SchemaRegistry,
    SchemaVersion,
    MirrorMaker,
    ClusterConfig,
    ReplicationFlow,
    CompactedTopicManager,
    CompactedTopicConfig,
    MetricsCollector,
    ProducerMetrics,
    ConsumerMetrics,
    JobMetrics,
    AlertManager,
    Alert,
    # Autoscaling
    ScalingPolicy,
    ScalingDecision,
    FlinkAutoscaler,
    SavepointConfig,
    SavepointManager,
    AutoscalingController,
    BACKPRESSURE_SCALING_POLICY,
    THROUGHPUT_SCALING_POLICY,
)
from streaming.anomaly_detection import (
    AnomalyType,
    AlertSeverity,
    AnomalyEvent,
    RollingStatistics,
    StatisticalDetector,
    ExponentialMovingAverageDetector,
    ThresholdDetector,
    AnomalyAlerter,
    StreamingAnomalyDetector,
    create_detector_from_config,
)

__version__ = "0.1.0"

__all__ = [
    # Config
    "KafkaConfig",
    "ProducerConfig",
    "ConsumerConfig",
    "FlinkConfig",
    "TopicConfig",
    "SchemaRegistryConfig",
    "StreamingPlatformConfig",
    # Producer
    "StreamingProducer",
    "TransactionalProducer",
    "TopicManager",
    "DeliveryReport",
    # Consumer
    "StreamingConsumer",
    "ConsumerGroup",
    "Message",
    # Serializers
    "AvroSerializer",
    "AvroDeserializer",
    "JsonSerializer",
    "JsonDeserializer",
    "create_event_serializer",
    "create_event_deserializer",
    # Flink
    "FlinkJobBuilder",
    "SourceConfig",
    "SinkConfig",
    "WindowConfig",
    "create_simple_pipeline",
    # Windowing
    "Window",
    "WindowedValue",
    "TumblingWindowAssigner",
    "SlidingWindowAssigner",
    "SessionWindowAssigner",
    "Watermark",
    "WatermarkGenerator",
    "BoundedOutOfOrdernessGenerator",
    "WindowOperator",
    "AggregationResult",
    # State
    "StateBackend",
    "InMemoryStateBackend",
    "ValueState",
    "ListState",
    "MapState",
    "TimerService",
    "Timer",
    "CheckpointCoordinator",
    "StateContext",
    # Patterns
    "Deduplicator",
    "DeduplicationConfig",
    "Debouncer",
    "DebouncerConfig",
    "Session",
    "SessionProcessor",
    "Pattern",
    "PatternMatch",
    "CEPEngine",
    "StreamJoiner",
    "JoinConfig",
    "IntervalJoiner",
    # Enterprise
    "SchemaRegistry",
    "SchemaVersion",
    "MirrorMaker",
    "ClusterConfig",
    "ReplicationFlow",
    "CompactedTopicManager",
    "CompactedTopicConfig",
    "MetricsCollector",
    "ProducerMetrics",
    "ConsumerMetrics",
    "JobMetrics",
    "AlertManager",
    "Alert",
    # Autoscaling
    "ScalingPolicy",
    "ScalingDecision",
    "FlinkAutoscaler",
    "SavepointConfig",
    "SavepointManager",
    "AutoscalingController",
    "BACKPRESSURE_SCALING_POLICY",
    "THROUGHPUT_SCALING_POLICY",
    # Anomaly Detection
    "AnomalyType",
    "AlertSeverity",
    "AnomalyEvent",
    "RollingStatistics",
    "StatisticalDetector",
    "ExponentialMovingAverageDetector",
    "ThresholdDetector",
    "AnomalyAlerter",
    "StreamingAnomalyDetector",
    "create_detector_from_config",
]
