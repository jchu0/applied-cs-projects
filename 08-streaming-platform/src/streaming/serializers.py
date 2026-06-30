"""Serialization support for Avro and JSON."""

import io
import json
import struct
from typing import Any, Dict, Optional

import fastavro
from fastavro.schema import parse_schema


class AvroSerializer:
    """Serialize messages using Avro with optional Schema Registry."""

    MAGIC_BYTE = 0

    def __init__(self, schema_str: str, schema_id: Optional[int] = None):
        """
        Initialize serializer.

        Args:
            schema_str: Avro schema as JSON string
            schema_id: Schema Registry ID (if using registry)
        """
        self._schema = parse_schema(json.loads(schema_str))
        self._schema_id = schema_id

    def serialize(self, data: Dict[str, Any]) -> bytes:
        """
        Serialize data to Avro bytes.

        Args:
            data: Dictionary matching schema

        Returns:
            Serialized bytes
        """
        output = io.BytesIO()

        # Write Schema Registry wire format if we have a schema ID
        if self._schema_id is not None:
            output.write(struct.pack(">bI", self.MAGIC_BYTE, self._schema_id))

        # Write Avro data
        fastavro.schemaless_writer(output, self._schema, data)

        return output.getvalue()

    @property
    def schema(self):
        """Get the parsed schema."""
        return self._schema


class AvroDeserializer:
    """Deserialize Avro messages."""

    MAGIC_BYTE = 0
    HEADER_SIZE = 5  # 1 byte magic + 4 bytes schema ID

    def __init__(self, schema_str: str, expect_registry_format: bool = False):
        """
        Initialize deserializer.

        Args:
            schema_str: Avro schema as JSON string
            expect_registry_format: Whether to expect Schema Registry wire format
        """
        self._schema = parse_schema(json.loads(schema_str))
        self._expect_registry_format = expect_registry_format

    def deserialize(self, data: bytes) -> Dict[str, Any]:
        """
        Deserialize Avro bytes to dictionary.

        Args:
            data: Serialized bytes

        Returns:
            Deserialized dictionary
        """
        input_stream = io.BytesIO(data)

        # Skip Schema Registry header if present
        if self._expect_registry_format:
            magic = struct.unpack(">b", input_stream.read(1))[0]
            if magic != self.MAGIC_BYTE:
                raise ValueError(f"Invalid magic byte: {magic}")

            schema_id = struct.unpack(">I", input_stream.read(4))[0]
            # Could validate schema_id here

        return fastavro.schemaless_reader(input_stream, self._schema)

    def extract_schema_id(self, data: bytes) -> int:
        """Extract schema ID from Schema Registry wire format."""
        magic = struct.unpack(">b", data[:1])[0]
        if magic != self.MAGIC_BYTE:
            raise ValueError(f"Invalid magic byte: {magic}")

        return struct.unpack(">I", data[1:5])[0]


class JsonSerializer:
    """Simple JSON serializer."""

    def serialize(self, data: Dict[str, Any]) -> bytes:
        """Serialize to JSON bytes."""
        return json.dumps(data).encode("utf-8")


class JsonDeserializer:
    """Simple JSON deserializer."""

    def deserialize(self, data: bytes) -> Dict[str, Any]:
        """Deserialize JSON bytes."""
        return json.loads(data.decode("utf-8"))


# Common Avro schemas for events

EVENT_SCHEMA = """
{
  "type": "record",
  "name": "Event",
  "namespace": "streaming.events",
  "fields": [
    {"name": "event_id", "type": "string"},
    {"name": "event_type", "type": "string"},
    {"name": "user_id", "type": ["null", "string"], "default": null},
    {"name": "timestamp", "type": "long", "logicalType": "timestamp-millis"},
    {"name": "payload", "type": {"type": "map", "values": "string"}},
    {"name": "metadata", "type": {
      "type": "record",
      "name": "Metadata",
      "fields": [
        {"name": "source", "type": "string"},
        {"name": "version", "type": "int", "default": 1},
        {"name": "correlation_id", "type": ["null", "string"], "default": null}
      ]
    }}
  ]
}
"""

METRIC_SCHEMA = """
{
  "type": "record",
  "name": "Metric",
  "namespace": "streaming.metrics",
  "fields": [
    {"name": "metric_name", "type": "string"},
    {"name": "value", "type": "double"},
    {"name": "timestamp", "type": "long", "logicalType": "timestamp-millis"},
    {"name": "tags", "type": {"type": "map", "values": "string"}},
    {"name": "unit", "type": ["null", "string"], "default": null}
  ]
}
"""

AGGREGATION_SCHEMA = """
{
  "type": "record",
  "name": "Aggregation",
  "namespace": "streaming.aggregations",
  "fields": [
    {"name": "key", "type": "string"},
    {"name": "window_start", "type": "long", "logicalType": "timestamp-millis"},
    {"name": "window_end", "type": "long", "logicalType": "timestamp-millis"},
    {"name": "count", "type": "long"},
    {"name": "sum", "type": "double"},
    {"name": "min", "type": "double"},
    {"name": "max", "type": "double"},
    {"name": "avg", "type": "double"}
  ]
}
"""


def create_event_serializer(schema_id: Optional[int] = None) -> AvroSerializer:
    """Create a serializer for Event schema."""
    return AvroSerializer(EVENT_SCHEMA, schema_id)


def create_event_deserializer(expect_registry: bool = False) -> AvroDeserializer:
    """Create a deserializer for Event schema."""
    return AvroDeserializer(EVENT_SCHEMA, expect_registry)


def create_metric_serializer(schema_id: Optional[int] = None) -> AvroSerializer:
    """Create a serializer for Metric schema."""
    return AvroSerializer(METRIC_SCHEMA, schema_id)


def create_metric_deserializer(expect_registry: bool = False) -> AvroDeserializer:
    """Create a deserializer for Metric schema."""
    return AvroDeserializer(METRIC_SCHEMA, expect_registry)
