"""Configuration for the streaming platform."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class KafkaConfig:
    """Base Kafka configuration."""

    bootstrap_servers: str
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: Optional[str] = None
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to confluent-kafka config dict."""
        config = {
            "bootstrap.servers": self.bootstrap_servers,
            "security.protocol": self.security_protocol,
        }

        if self.sasl_mechanism:
            config["sasl.mechanism"] = self.sasl_mechanism
        if self.sasl_username:
            config["sasl.username"] = self.sasl_username
        if self.sasl_password:
            config["sasl.password"] = self.sasl_password

        return config


@dataclass
class ProducerConfig(KafkaConfig):
    """Kafka producer configuration."""

    # Schema Registry
    schema_registry_url: Optional[str] = None

    # Reliability
    acks: str = "all"
    enable_idempotence: bool = True
    retries: int = 2147483647
    max_in_flight_requests: int = 5

    # Batching
    batch_size: int = 16384
    linger_ms: int = 10
    compression_type: str = "lz4"

    # Memory
    buffer_memory: int = 33554432

    # Transactions
    transactional_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to confluent-kafka producer config."""
        config = super().to_dict()

        config.update({
            "acks": self.acks,
            "enable.idempotence": self.enable_idempotence,
            "retries": self.retries,
            "max.in.flight.requests.per.connection": self.max_in_flight_requests,
            "batch.size": self.batch_size,
            "linger.ms": self.linger_ms,
            "compression.type": self.compression_type,
            "buffer.memory": self.buffer_memory,
        })

        if self.transactional_id:
            config["transactional.id"] = self.transactional_id

        return config


@dataclass
class ConsumerConfig(KafkaConfig):
    """Kafka consumer configuration."""

    group_id: str = ""  # Required but with default to fix dataclass inheritance
    schema_registry_url: Optional[str] = None

    # Offset management
    auto_offset_reset: str = "earliest"
    enable_auto_commit: bool = False

    # Performance
    max_poll_records: int = 500
    max_poll_interval_ms: int = 300000
    fetch_min_bytes: int = 1
    fetch_max_wait_ms: int = 500

    # Exactly-once
    isolation_level: str = "read_committed"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to confluent-kafka consumer config."""
        config = super().to_dict()

        config.update({
            "group.id": self.group_id,
            "auto.offset.reset": self.auto_offset_reset,
            "enable.auto.commit": self.enable_auto_commit,
            "max.poll.interval.ms": self.max_poll_interval_ms,
            "isolation.level": self.isolation_level,
        })

        return config


@dataclass
class FlinkConfig:
    """Apache Flink configuration."""

    job_name: str
    parallelism: int = 4
    max_parallelism: int = 128

    # Checkpointing
    checkpoint_interval_ms: int = 60000
    checkpoint_timeout_ms: int = 120000
    checkpoint_mode: str = "EXACTLY_ONCE"
    min_pause_between_checkpoints_ms: int = 30000
    max_concurrent_checkpoints: int = 1

    # State backend
    state_backend: str = "rocksdb"
    checkpoint_storage: Optional[str] = None

    # Restart strategy
    restart_strategy: str = "fixed-delay"
    restart_attempts: int = 3
    restart_delay_ms: int = 10000


@dataclass
class TopicConfig:
    """Kafka topic configuration."""

    name: str
    partitions: int = 3
    replication_factor: int = 3

    # Retention
    retention_ms: int = 604800000  # 7 days
    retention_bytes: int = -1  # No limit

    # Compaction
    cleanup_policy: str = "delete"  # delete, compact, or compact,delete
    min_cleanable_dirty_ratio: float = 0.5

    # Segment
    segment_ms: int = 604800000  # 7 days
    segment_bytes: int = 1073741824  # 1GB

    def to_dict(self) -> Dict[str, str]:
        """Convert to topic config dict."""
        return {
            "retention.ms": str(self.retention_ms),
            "retention.bytes": str(self.retention_bytes),
            "cleanup.policy": self.cleanup_policy,
            "min.cleanable.dirty.ratio": str(self.min_cleanable_dirty_ratio),
            "segment.ms": str(self.segment_ms),
            "segment.bytes": str(self.segment_bytes),
        }


@dataclass
class SchemaRegistryConfig:
    """Schema Registry configuration."""

    url: str
    basic_auth_user: Optional[str] = None
    basic_auth_password: Optional[str] = None

    # Compatibility
    default_compatibility: str = "BACKWARD"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to schema registry client config."""
        config = {"url": self.url}

        if self.basic_auth_user:
            config["basic.auth.user.info"] = f"{self.basic_auth_user}:{self.basic_auth_password}"

        return config


@dataclass
class StreamingPlatformConfig:
    """Complete streaming platform configuration."""

    kafka: KafkaConfig
    schema_registry: Optional[SchemaRegistryConfig] = None
    flink: Optional[FlinkConfig] = None
    topics: List[TopicConfig] = field(default_factory=list)
