"""Flink job builder and utilities."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from streaming.config import FlinkConfig


@dataclass
class SourceConfig:
    """Configuration for a Flink source."""

    name: str
    connector: str
    topic: Optional[str] = None
    bootstrap_servers: Optional[str] = None
    group_id: Optional[str] = None
    format: str = "json"
    properties: Dict[str, str] = field(default_factory=dict)


@dataclass
class SinkConfig:
    """Configuration for a Flink sink."""

    name: str
    connector: str
    topic: Optional[str] = None
    bootstrap_servers: Optional[str] = None
    format: str = "json"
    delivery_guarantee: str = "at-least-once"
    properties: Dict[str, str] = field(default_factory=dict)


@dataclass
class WindowConfig:
    """Window configuration."""

    window_type: str  # tumbling, sliding, session
    size_ms: int
    slide_ms: Optional[int] = None  # For sliding windows
    gap_ms: Optional[int] = None  # For session windows


class FlinkJobBuilder:
    """Builder for Flink streaming jobs."""

    def __init__(self, config: FlinkConfig):
        self.config = config
        self._sources: List[SourceConfig] = []
        self._sinks: List[SinkConfig] = []
        self._transformations: List[Dict[str, Any]] = []

    def add_kafka_source(
        self,
        name: str,
        topic: str,
        bootstrap_servers: str,
        group_id: str,
        format: str = "json",
        watermark_column: Optional[str] = None,
        watermark_delay_ms: int = 10000,
    ) -> "FlinkJobBuilder":
        """
        Add a Kafka source.

        Args:
            name: Source name
            topic: Kafka topic
            bootstrap_servers: Kafka bootstrap servers
            group_id: Consumer group ID
            format: Data format (json, avro)
            watermark_column: Column for watermark generation
            watermark_delay_ms: Maximum out-of-orderness

        Returns:
            Self for chaining
        """
        properties = {
            "topic": topic,
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,
        }

        if watermark_column:
            properties["watermark.column"] = watermark_column
            properties["watermark.delay.ms"] = str(watermark_delay_ms)

        self._sources.append(SourceConfig(
            name=name,
            connector="kafka",
            topic=topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            format=format,
            properties=properties,
        ))

        return self

    def add_kafka_sink(
        self,
        name: str,
        topic: str,
        bootstrap_servers: str,
        format: str = "json",
        delivery_guarantee: str = "exactly-once",
        transactional_id_prefix: Optional[str] = None,
    ) -> "FlinkJobBuilder":
        """
        Add a Kafka sink.

        Args:
            name: Sink name
            topic: Kafka topic
            bootstrap_servers: Kafka bootstrap servers
            format: Data format (json, avro)
            delivery_guarantee: Delivery guarantee level
            transactional_id_prefix: Prefix for transactional IDs

        Returns:
            Self for chaining
        """
        properties = {
            "topic": topic,
            "bootstrap.servers": bootstrap_servers,
        }

        if transactional_id_prefix:
            properties["transactional.id.prefix"] = transactional_id_prefix

        self._sinks.append(SinkConfig(
            name=name,
            connector="kafka",
            topic=topic,
            bootstrap_servers=bootstrap_servers,
            format=format,
            delivery_guarantee=delivery_guarantee,
            properties=properties,
        ))

        return self

    def add_filter(
        self,
        source: str,
        condition: str,
        output: str,
    ) -> "FlinkJobBuilder":
        """Add a filter transformation."""
        self._transformations.append({
            "type": "filter",
            "source": source,
            "condition": condition,
            "output": output,
        })
        return self

    def add_map(
        self,
        source: str,
        expression: str,
        output: str,
    ) -> "FlinkJobBuilder":
        """Add a map transformation."""
        self._transformations.append({
            "type": "map",
            "source": source,
            "expression": expression,
            "output": output,
        })
        return self

    def add_window_aggregation(
        self,
        source: str,
        key_by: List[str],
        window: WindowConfig,
        aggregations: Dict[str, str],
        output: str,
    ) -> "FlinkJobBuilder":
        """
        Add a windowed aggregation.

        Args:
            source: Source stream name
            key_by: Columns to group by
            window: Window configuration
            aggregations: Aggregation expressions {alias: expression}
            output: Output stream name

        Returns:
            Self for chaining
        """
        self._transformations.append({
            "type": "window_aggregation",
            "source": source,
            "key_by": key_by,
            "window": window,
            "aggregations": aggregations,
            "output": output,
        })
        return self

    def add_join(
        self,
        left_source: str,
        right_source: str,
        join_type: str,
        condition: str,
        output: str,
    ) -> "FlinkJobBuilder":
        """Add a stream join."""
        self._transformations.append({
            "type": "join",
            "left_source": left_source,
            "right_source": right_source,
            "join_type": join_type,
            "condition": condition,
            "output": output,
        })
        return self

    def build(self) -> Dict[str, Any]:
        """
        Build the job specification.

        Returns:
            Job specification dictionary
        """
        return {
            "job_name": self.config.job_name,
            "config": {
                "parallelism": self.config.parallelism,
                "max_parallelism": self.config.max_parallelism,
                "checkpointing": {
                    "interval_ms": self.config.checkpoint_interval_ms,
                    "timeout_ms": self.config.checkpoint_timeout_ms,
                    "mode": self.config.checkpoint_mode,
                    "min_pause_ms": self.config.min_pause_between_checkpoints_ms,
                    "max_concurrent": self.config.max_concurrent_checkpoints,
                },
                "state_backend": self.config.state_backend,
                "checkpoint_storage": self.config.checkpoint_storage,
            },
            "sources": [
                {
                    "name": s.name,
                    "connector": s.connector,
                    "format": s.format,
                    "properties": s.properties,
                }
                for s in self._sources
            ],
            "sinks": [
                {
                    "name": s.name,
                    "connector": s.connector,
                    "format": s.format,
                    "delivery_guarantee": s.delivery_guarantee,
                    "properties": s.properties,
                }
                for s in self._sinks
            ],
            "transformations": self._transformations,
        }

    def generate_pyflink_code(self) -> str:
        """
        Generate PyFlink code for the job.

        Returns:
            Python code string
        """
        lines = [
            "from pyflink.datastream import StreamExecutionEnvironment",
            "from pyflink.table import StreamTableEnvironment",
            "",
            f'def run_{self.config.job_name.replace("-", "_")}():',
            "    env = StreamExecutionEnvironment.get_execution_environment()",
            f"    env.set_parallelism({self.config.parallelism})",
            f"    env.enable_checkpointing({self.config.checkpoint_interval_ms})",
            "",
            "    t_env = StreamTableEnvironment.create(env)",
            "",
        ]

        # Generate source DDL
        for source in self._sources:
            if source.connector == "kafka":
                lines.extend(self._generate_kafka_source_ddl(source))
                lines.append("")

        # Generate sink DDL
        for sink in self._sinks:
            if sink.connector == "kafka":
                lines.extend(self._generate_kafka_sink_ddl(sink))
                lines.append("")

        # Execute
        lines.append(f'    env.execute("{self.config.job_name}")')
        lines.append("")
        lines.append('if __name__ == "__main__":')
        lines.append(f'    run_{self.config.job_name.replace("-", "_")}()')

        return "\n".join(lines)

    def _generate_kafka_source_ddl(self, source: SourceConfig) -> List[str]:
        """Generate DDL for Kafka source table."""
        return [
            f'    t_env.execute_sql("""',
            f"        CREATE TABLE {source.name} (",
            "            -- Define schema here",
            "            event_time TIMESTAMP(3),",
            "            WATERMARK FOR event_time AS event_time - INTERVAL '10' SECOND",
            "        ) WITH (",
            "            'connector' = 'kafka',",
            f"            'topic' = '{source.topic}',",
            f"            'properties.bootstrap.servers' = '{source.bootstrap_servers}',",
            f"            'properties.group.id' = '{source.group_id}',",
            "            'scan.startup.mode' = 'latest-offset',",
            f"            'format' = '{source.format}'",
            "        )",
            '    """)',
        ]

    def _generate_kafka_sink_ddl(self, sink: SinkConfig) -> List[str]:
        """Generate DDL for Kafka sink table."""
        return [
            f'    t_env.execute_sql("""',
            f"        CREATE TABLE {sink.name} (",
            "            -- Define schema here",
            "        ) WITH (",
            "            'connector' = 'kafka',",
            f"            'topic' = '{sink.topic}',",
            f"            'properties.bootstrap.servers' = '{sink.bootstrap_servers}',",
            f"            'format' = '{sink.format}'",
            "        )",
            '    """)',
        ]


def create_simple_pipeline(
    job_name: str,
    source_topic: str,
    sink_topic: str,
    bootstrap_servers: str,
    group_id: str,
    parallelism: int = 4,
) -> FlinkJobBuilder:
    """
    Create a simple passthrough pipeline.

    Args:
        job_name: Name of the job
        source_topic: Source Kafka topic
        sink_topic: Sink Kafka topic
        bootstrap_servers: Kafka bootstrap servers
        group_id: Consumer group ID
        parallelism: Job parallelism

    Returns:
        Configured FlinkJobBuilder
    """
    config = FlinkConfig(
        job_name=job_name,
        parallelism=parallelism,
    )

    return (
        FlinkJobBuilder(config)
        .add_kafka_source(
            name="source",
            topic=source_topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
        )
        .add_kafka_sink(
            name="sink",
            topic=sink_topic,
            bootstrap_servers=bootstrap_servers,
        )
    )
