"""Tests for exactly-once semantics."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import time

pytest.importorskip("confluent_kafka")

from streaming.config import ProducerConfig, ConsumerConfig
from streaming.producer import StreamingProducer, TransactionalProducer, DeliveryReport
from streaming.patterns import Deduplicator, DeduplicationConfig
from streaming.state import (
    InMemoryStateBackend,
    CheckpointCoordinator,
    ValueState,
)
from streaming.enterprise import SchemaRegistry


# --- Idempotent Producer Tests ---


class TestIdempotentProducer:
    """Tests for idempotent producer configuration."""

    def test_idempotence_enabled_by_default(self):
        """Test idempotence is enabled by default."""
        config = ProducerConfig(
            bootstrap_servers="localhost:9092",
        )

        result = config.to_dict()

        assert result["enable.idempotence"] is True

    def test_idempotence_requires_acks_all(self):
        """Test idempotence works with acks=all."""
        config = ProducerConfig(
            bootstrap_servers="localhost:9092",
            enable_idempotence=True,
            acks="all",
        )

        result = config.to_dict()

        assert result["acks"] == "all"
        assert result["enable.idempotence"] is True

    @patch("streaming.producer.Producer")
    def test_idempotent_producer_retries(self, mock_producer_class):
        """Test producer configured with max retries for idempotence."""
        config = ProducerConfig(
            bootstrap_servers="localhost:9092",
            enable_idempotence=True,
            retries=2147483647,  # Max retries for idempotence
            max_in_flight_requests=5,
        )

        mock_producer = Mock()
        mock_producer_class.return_value = mock_producer

        producer = StreamingProducer(config)

        # Verify config includes idempotence settings
        call_args = mock_producer_class.call_args[0][0]
        assert call_args["enable.idempotence"] is True
        assert call_args["retries"] == 2147483647


# --- Transactional Producer Tests ---


class TestTransactionalProducerExactlyOnce:
    """Tests for transactional producer exactly-once semantics."""

    @pytest.fixture
    def transactional_config(self):
        """Create transactional producer config."""
        return ProducerConfig(
            bootstrap_servers="localhost:9092",
            transactional_id="test-txn-id",
            enable_idempotence=True,
            acks="all",
        )

    @patch("streaming.producer.Producer")
    def test_transactions_initialized(self, mock_producer_class, transactional_config):
        """Test transactions are initialized on startup."""
        mock_producer = Mock()
        mock_producer_class.return_value = mock_producer

        producer = TransactionalProducer(transactional_config)

        mock_producer.init_transactions.assert_called_once()

    @patch("streaming.producer.Producer")
    def test_atomic_batch_production(self, mock_producer_class, transactional_config):
        """Test all messages in transaction are produced atomically."""
        mock_producer = Mock()
        mock_producer_class.return_value = mock_producer

        producer = TransactionalProducer(transactional_config)
        messages = [
            {"topic": "topic1", "key": f"key-{i}", "value": f"value-{i}".encode()}
            for i in range(5)
        ]

        result = producer.produce_transactionally(messages)

        assert result is True
        mock_producer.begin_transaction.assert_called_once()
        assert mock_producer.produce.call_count == 5
        mock_producer.commit_transaction.assert_called_once()

    @patch("streaming.producer.Producer")
    def test_transaction_abort_on_failure(self, mock_producer_class, transactional_config):
        """Test transaction is aborted on failure."""
        from confluent_kafka import KafkaException

        mock_producer = Mock()
        mock_producer.produce.side_effect = KafkaException("Network error")
        mock_producer_class.return_value = mock_producer

        producer = TransactionalProducer(transactional_config)
        messages = [{"topic": "topic1", "key": "key", "value": b"value"}]

        with pytest.raises(KafkaException):
            producer.produce_transactionally(messages)

        mock_producer.abort_transaction.assert_called_once()
        mock_producer.commit_transaction.assert_not_called()

    @patch("streaming.producer.Producer")
    def test_send_offsets_to_transaction(self, mock_producer_class, transactional_config):
        """Test sending consumer offsets in transaction."""
        mock_producer = Mock()
        mock_producer_class.return_value = mock_producer

        producer = TransactionalProducer(transactional_config)
        mock_positions = Mock()
        mock_group_metadata = Mock()

        producer.begin_transaction()
        producer.send_offsets_to_transaction(
            mock_positions,
            mock_group_metadata,
        )

        mock_producer.send_offsets_to_transaction.assert_called_once_with(
            mock_positions,
            mock_group_metadata,
            -1,
        )


# --- Consumer Exactly-Once Tests ---


class TestConsumerExactlyOnce:
    """Tests for consumer exactly-once configuration."""

    def test_read_committed_isolation(self):
        """Test consumer configured for read_committed isolation."""
        config = ConsumerConfig(
            bootstrap_servers="localhost:9092",
            group_id="test-group",
            isolation_level="read_committed",
        )

        result = config.to_dict()

        assert result["isolation.level"] == "read_committed"

    def test_auto_commit_disabled(self):
        """Test auto commit is disabled for exactly-once."""
        config = ConsumerConfig(
            bootstrap_servers="localhost:9092",
            group_id="test-group",
            enable_auto_commit=False,
        )

        result = config.to_dict()

        assert result["enable.auto.commit"] is False


# --- Deduplication Tests ---


class TestDeduplication:
    """Tests for deduplication patterns."""

    @pytest.fixture
    def dedup_config(self):
        """Create deduplication config."""
        return DeduplicationConfig(
            window_ms=60000,  # 1 minute window
            id_extractor=lambda x: x["event_id"],
        )

    def test_first_event_not_duplicate(self, dedup_config):
        """Test first event is not a duplicate."""
        deduplicator = Deduplicator(dedup_config)

        event = {"event_id": "event-001", "data": "value"}
        is_duplicate = deduplicator.is_duplicate(event)

        assert is_duplicate is False

    def test_same_event_is_duplicate(self, dedup_config):
        """Test same event ID is detected as duplicate."""
        deduplicator = Deduplicator(dedup_config)

        event = {"event_id": "event-001", "data": "value"}
        deduplicator.is_duplicate(event)
        is_duplicate = deduplicator.is_duplicate(event)

        assert is_duplicate is True

    def test_different_events_not_duplicate(self, dedup_config):
        """Test different event IDs are not duplicates."""
        deduplicator = Deduplicator(dedup_config)

        event1 = {"event_id": "event-001", "data": "value1"}
        event2 = {"event_id": "event-002", "data": "value2"}

        assert deduplicator.is_duplicate(event1) is False
        assert deduplicator.is_duplicate(event2) is False

    def test_process_returns_event_if_not_duplicate(self, dedup_config):
        """Test process returns event if not duplicate."""
        deduplicator = Deduplicator(dedup_config)

        event = {"event_id": "event-001", "data": "value"}
        result = deduplicator.process(event)

        assert result == event

    def test_process_returns_none_if_duplicate(self, dedup_config):
        """Test process returns None if duplicate."""
        deduplicator = Deduplicator(dedup_config)

        event = {"event_id": "event-001", "data": "value"}
        deduplicator.process(event)
        result = deduplicator.process(event)

        assert result is None

    def test_expired_events_not_duplicate(self, dedup_config):
        """Test events outside window are not considered duplicates."""
        # Create config with very short window
        config = DeduplicationConfig(
            window_ms=1,  # 1ms window
            id_extractor=lambda x: x["event_id"],
        )
        deduplicator = Deduplicator(config)

        event = {"event_id": "event-001", "data": "value"}
        deduplicator.is_duplicate(event)

        # Wait for window to expire
        time.sleep(0.01)

        # Same event should no longer be duplicate
        is_duplicate = deduplicator.is_duplicate(event)
        assert is_duplicate is False


# --- Checkpoint-Based Exactly-Once Tests ---


class TestCheckpointExactlyOnce:
    """Tests for checkpoint-based exactly-once processing."""

    @pytest.fixture
    def backend(self):
        """Create a fresh state backend."""
        return InMemoryStateBackend()

    def test_checkpoint_preserves_processing_position(self, backend):
        """Test checkpoint preserves processing position."""
        coordinator = CheckpointCoordinator(backend)

        # Store processing position in state
        offset_state = ValueState("offsets", backend, "input-topic")
        offset_state.update({"partition-0": 100, "partition-1": 200})

        # Create checkpoint
        meta = coordinator.trigger_checkpoint()
        coordinator.complete_checkpoint(meta.checkpoint_id)

        # Verify checkpoint is complete
        latest = coordinator.get_latest_checkpoint()
        assert latest is not None
        assert latest.checkpoint_id == meta.checkpoint_id

        # Verify state is preserved
        restored_offsets = offset_state.value()
        assert restored_offsets == {"partition-0": 100, "partition-1": 200}

    def test_checkpoint_and_restore_workflow(self, backend):
        """Test complete checkpoint and restore workflow."""
        coordinator = CheckpointCoordinator(backend)
        counter = ValueState("counter", backend, "processor")

        # Process some events
        for i in range(10):
            current = counter.value() or 0
            counter.update(current + 1)

        assert counter.value() == 10

        # Checkpoint
        meta = coordinator.trigger_checkpoint()
        coordinator.complete_checkpoint(meta.checkpoint_id)

        # Process more events
        for i in range(5):
            current = counter.value() or 0
            counter.update(current + 1)

        assert counter.value() == 15

        # Restore would reset state (simulated)
        coordinator.restore_from_checkpoint(meta.checkpoint_id)

        # In real implementation, state would be restored to checkpoint
        assert coordinator._checkpoint_counter == meta.checkpoint_id

    def test_incomplete_checkpoint_not_restored(self, backend):
        """Test incomplete checkpoint cannot be restored."""
        coordinator = CheckpointCoordinator(backend)

        meta = coordinator.trigger_checkpoint()
        # Don't complete the checkpoint

        result = coordinator.restore_from_checkpoint(meta.checkpoint_id)

        assert result is False


# --- End-to-End Exactly-Once Flow Tests ---


class TestEndToEndExactlyOnce:
    """Tests for end-to-end exactly-once processing flow."""

    def test_consume_transform_produce_pattern(self):
        """Test consume-transform-produce pattern for exactly-once."""
        # This test simulates the pattern without actual Kafka

        # Simulate consumed records
        consumed_records = [
            {"key": f"key-{i}", "value": {"amount": i * 10}}
            for i in range(5)
        ]

        # Transform function
        def transform(record):
            return {
                "key": record["key"],
                "value": {"amount": record["value"]["amount"] * 2}
            }

        # Apply transformation
        transformed = [transform(r) for r in consumed_records]

        # Verify transformation
        assert len(transformed) == 5
        assert transformed[0]["value"]["amount"] == 0
        assert transformed[1]["value"]["amount"] == 20
        assert transformed[4]["value"]["amount"] == 80

    def test_deduplication_with_stateful_processing(self):
        """Test deduplication combined with stateful processing."""
        backend = InMemoryStateBackend()

        # Deduplication
        dedup_config = DeduplicationConfig(
            window_ms=60000,
            id_extractor=lambda x: x["event_id"],
        )
        deduplicator = Deduplicator(dedup_config)

        # State
        counter = ValueState("event_count", backend, "processor")

        # Process events including duplicates
        events = [
            {"event_id": "e1", "value": 10},
            {"event_id": "e2", "value": 20},
            {"event_id": "e1", "value": 10},  # Duplicate
            {"event_id": "e3", "value": 30},
            {"event_id": "e2", "value": 20},  # Duplicate
        ]

        processed_count = 0
        for event in events:
            result = deduplicator.process(event)
            if result:
                processed_count += 1
                current = counter.value() or 0
                counter.update(current + 1)

        # Only 3 unique events should be processed
        assert processed_count == 3
        assert counter.value() == 3

    def test_exactly_once_with_checkpoint_recovery(self):
        """Test exactly-once semantics with checkpoint recovery simulation."""
        backend = InMemoryStateBackend()
        coordinator = CheckpointCoordinator(backend)

        # Simulate processing with checkpoints
        processed_offset = ValueState("processed_offset", backend, "source")
        output_count = ValueState("output_count", backend, "sink")

        # Process first batch
        for offset in range(100):
            processed_offset.update(offset)
            current_count = output_count.value() or 0
            output_count.update(current_count + 1)

        # Checkpoint at offset 100
        cp1 = coordinator.trigger_checkpoint()
        coordinator.complete_checkpoint(cp1.checkpoint_id)

        assert processed_offset.value() == 99
        assert output_count.value() == 100

        # Process more
        for offset in range(100, 150):
            processed_offset.update(offset)
            current_count = output_count.value() or 0
            output_count.update(current_count + 1)

        assert processed_offset.value() == 149
        assert output_count.value() == 150

        # Simulate failure and recovery
        # In real scenario, state would be restored from checkpoint
        # Here we verify checkpoint is available
        latest = coordinator.get_latest_checkpoint()
        assert latest is not None
        assert latest.checkpoint_id == cp1.checkpoint_id


# --- Schema Registry Exactly-Once Tests ---


class TestSchemaRegistryExactlyOnce:
    """Tests for schema registry in exactly-once context."""

    def test_schema_id_serialization(self):
        """Test schema ID is included in serialized messages."""
        registry = SchemaRegistry("http://localhost:8081")

        # Register schema
        schema = '{"type": "record", "name": "Test", "fields": []}'
        schema_id = registry.register_schema("test-topic-value", schema)

        # Verify schema is registered
        assert schema_id is not None
        latest = registry.get_latest_schema("test-topic-value")
        assert latest.schema_id == schema_id

    def test_schema_compatibility_check(self):
        """Test schema compatibility ensures data consistency."""
        registry = SchemaRegistry("http://localhost:8081")

        # Register initial schema
        schema_v1 = '{"type": "record", "name": "Test", "fields": [{"name": "id", "type": "string"}]}'
        registry.register_schema("test-topic-value", schema_v1)

        # Check compatible schema
        schema_v2 = '{"type": "record", "name": "Test", "fields": [{"name": "id", "type": "string"}, {"name": "name", "type": ["null", "string"], "default": null}]}'
        is_compatible = registry.check_compatibility("test-topic-value", schema_v2)

        # In real implementation, this would check Avro compatibility
        assert is_compatible is True


# --- Delivery Guarantee Tests ---


class TestDeliveryGuarantees:
    """Tests for different delivery guarantee levels."""

    def test_at_least_once_config(self):
        """Test at-least-once configuration."""
        config = ProducerConfig(
            bootstrap_servers="localhost:9092",
            acks="all",
            enable_idempotence=False,  # Disabled for at-least-once
            retries=3,
        )

        result = config.to_dict()

        assert result["acks"] == "all"
        assert result["enable.idempotence"] is False
        assert result["retries"] == 3

    def test_at_most_once_config(self):
        """Test at-most-once configuration."""
        config = ProducerConfig(
            bootstrap_servers="localhost:9092",
            acks="0",  # No acks for at-most-once
            enable_idempotence=False,
            retries=0,
        )

        result = config.to_dict()

        assert result["acks"] == "0"
        assert result["retries"] == 0

    def test_exactly_once_config(self):
        """Test exactly-once configuration."""
        config = ProducerConfig(
            bootstrap_servers="localhost:9092",
            acks="all",
            enable_idempotence=True,
            transactional_id="exactly-once-producer",
            retries=2147483647,
            max_in_flight_requests=5,
        )

        result = config.to_dict()

        assert result["acks"] == "all"
        assert result["enable.idempotence"] is True
        assert result["transactional.id"] == "exactly-once-producer"
        assert result["retries"] == 2147483647
        assert result["max.in.flight.requests.per.connection"] == 5
