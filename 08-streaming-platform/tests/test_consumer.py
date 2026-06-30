"""Tests for Kafka consumer APIs."""

import pytest
from unittest.mock import Mock, MagicMock, patch, call

pytest.importorskip("confluent_kafka")

from confluent_kafka import KafkaError, KafkaException, TopicPartition

from streaming.config import ConsumerConfig
from streaming.consumer import (
    Message,
    RebalanceListener,
    StreamingConsumer,
    ConsumerGroup,
)


# --- Fixtures ---


@pytest.fixture
def consumer_config():
    """Create a consumer configuration for testing."""
    return ConsumerConfig(
        bootstrap_servers="localhost:9092",
        group_id="test-group",
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        max_poll_records=500,
        isolation_level="read_committed",
    )


@pytest.fixture
def sample_message():
    """Create a sample message for testing."""
    return Message(
        topic="test-topic",
        partition=0,
        offset=123,
        key="test-key",
        value=b"test-value",
        timestamp=1704067200000,
        headers={"trace-id": "abc123"},
    )


# --- Message Tests ---


class TestMessage:
    """Tests for Message dataclass."""

    def test_message_creation(self, sample_message):
        """Test message creation."""
        assert sample_message.topic == "test-topic"
        assert sample_message.partition == 0
        assert sample_message.offset == 123
        assert sample_message.key == "test-key"
        assert sample_message.value == b"test-value"
        assert sample_message.timestamp == 1704067200000
        assert sample_message.headers == {"trace-id": "abc123"}

    def test_message_to_dict(self, sample_message):
        """Test message conversion to dictionary."""
        result = sample_message.to_dict()

        assert result["topic"] == "test-topic"
        assert result["partition"] == 0
        assert result["offset"] == 123
        assert result["key"] == "test-key"
        assert result["value"] == b"test-value"
        assert result["timestamp"] == 1704067200000
        assert result["headers"] == {"trace-id": "abc123"}

    def test_message_with_null_key(self):
        """Test message with null key."""
        msg = Message(
            topic="test-topic",
            partition=0,
            offset=0,
            key=None,
            value=b"value",
            timestamp=0,
            headers={},
        )

        assert msg.key is None
        result = msg.to_dict()
        assert result["key"] is None


# --- RebalanceListener Tests ---


class TestRebalanceListener:
    """Tests for RebalanceListener class."""

    def test_on_assign_callback(self):
        """Test on_assign callback is invoked."""
        mock_consumer = Mock()
        callback = Mock()
        listener = RebalanceListener(mock_consumer)
        listener.set_on_assign(callback)

        partitions = [TopicPartition("test", 0), TopicPartition("test", 1)]
        listener.on_assign(mock_consumer, partitions)

        callback.assert_called_once_with(partitions)

    def test_on_revoke_commits_offsets(self):
        """Test on_revoke commits offsets before callback."""
        mock_consumer = Mock()
        callback = Mock()
        listener = RebalanceListener(mock_consumer)
        listener.set_on_revoke(callback)

        partitions = [TopicPartition("test", 0)]
        listener.on_revoke(mock_consumer, partitions)

        mock_consumer.commit.assert_called_once_with(asynchronous=False)
        callback.assert_called_once_with(partitions)

    def test_on_revoke_handles_commit_error(self):
        """Test on_revoke handles commit errors gracefully."""
        mock_consumer = Mock()
        mock_consumer.commit.side_effect = KafkaException("Commit failed")
        listener = RebalanceListener(mock_consumer)

        partitions = [TopicPartition("test", 0)]
        # Should not raise exception
        listener.on_revoke(mock_consumer, partitions)

    def test_callbacks_optional(self):
        """Test callbacks are optional."""
        mock_consumer = Mock()
        listener = RebalanceListener(mock_consumer)

        # Should not raise
        partitions = [TopicPartition("test", 0)]
        listener.on_assign(mock_consumer, partitions)
        listener.on_revoke(mock_consumer, partitions)


# --- StreamingConsumer Tests ---


class TestStreamingConsumer:
    """Tests for StreamingConsumer class."""

    @patch("streaming.consumer.Consumer")
    def test_consumer_initialization(self, mock_consumer_class, consumer_config):
        """Test consumer initialization."""
        mock_consumer = Mock()
        mock_consumer_class.return_value = mock_consumer

        consumer = StreamingConsumer(consumer_config)

        mock_consumer_class.assert_called_once()
        assert consumer.config == consumer_config

    @patch("streaming.consumer.Consumer")
    def test_subscribe(self, mock_consumer_class, consumer_config):
        """Test subscribing to topics."""
        mock_consumer = Mock()
        mock_consumer_class.return_value = mock_consumer

        consumer = StreamingConsumer(consumer_config)
        consumer.subscribe(["topic1", "topic2"])

        mock_consumer.subscribe.assert_called_once()
        call_args = mock_consumer.subscribe.call_args
        assert call_args[0][0] == ["topic1", "topic2"]

    @patch("streaming.consumer.Consumer")
    def test_subscribe_with_callbacks(self, mock_consumer_class, consumer_config):
        """Test subscribing with rebalance callbacks."""
        mock_consumer = Mock()
        mock_consumer_class.return_value = mock_consumer

        on_assign = Mock()
        on_revoke = Mock()

        consumer = StreamingConsumer(consumer_config)
        consumer.subscribe(["topic1"], on_assign=on_assign, on_revoke=on_revoke)

        mock_consumer.subscribe.assert_called_once()

    @patch("streaming.consumer.Consumer")
    def test_assign_partitions(self, mock_consumer_class, consumer_config):
        """Test manually assigning partitions."""
        mock_consumer = Mock()
        mock_consumer_class.return_value = mock_consumer

        consumer = StreamingConsumer(consumer_config)
        partitions = [TopicPartition("test", 0), TopicPartition("test", 1)]
        consumer.assign(partitions)

        mock_consumer.assign.assert_called_once_with(partitions)

    @patch("streaming.consumer.Consumer")
    def test_poll_returns_message(self, mock_consumer_class, consumer_config):
        """Test polling returns wrapped message."""
        mock_consumer = Mock()
        mock_msg = Mock()
        mock_msg.error.return_value = None
        mock_msg.topic.return_value = "test-topic"
        mock_msg.partition.return_value = 0
        mock_msg.offset.return_value = 123
        mock_msg.key.return_value = b"key"
        mock_msg.value.return_value = b"value"
        mock_msg.timestamp.return_value = (1, 1704067200000)
        mock_msg.headers.return_value = [("trace-id", b"abc")]
        mock_consumer.poll.return_value = mock_msg
        mock_consumer_class.return_value = mock_consumer

        consumer = StreamingConsumer(consumer_config)
        message = consumer.poll(timeout=1.0)

        assert message is not None
        assert message.topic == "test-topic"
        assert message.partition == 0
        assert message.offset == 123
        assert message.key == "key"
        assert message.value == b"value"
        assert message.headers == {"trace-id": "abc"}

    @patch("streaming.consumer.Consumer")
    def test_poll_returns_none_on_timeout(self, mock_consumer_class, consumer_config):
        """Test poll returns None on timeout."""
        mock_consumer = Mock()
        mock_consumer.poll.return_value = None
        mock_consumer_class.return_value = mock_consumer

        consumer = StreamingConsumer(consumer_config)
        message = consumer.poll(timeout=1.0)

        assert message is None

    @patch("streaming.consumer.Consumer")
    def test_poll_handles_partition_eof(self, mock_consumer_class, consumer_config):
        """Test poll handles partition EOF."""
        mock_consumer = Mock()
        mock_msg = Mock()
        mock_error = Mock()
        mock_error.code.return_value = KafkaError._PARTITION_EOF
        mock_msg.error.return_value = mock_error
        mock_consumer.poll.return_value = mock_msg
        mock_consumer_class.return_value = mock_consumer

        consumer = StreamingConsumer(consumer_config)
        message = consumer.poll(timeout=1.0)

        assert message is None

    @patch("streaming.consumer.Consumer")
    def test_poll_raises_on_error(self, mock_consumer_class, consumer_config):
        """Test poll raises on other errors."""
        mock_consumer = Mock()
        mock_msg = Mock()
        mock_error = Mock()
        mock_error.code.return_value = KafkaError.UNKNOWN_TOPIC_OR_PART
        mock_msg.error.return_value = mock_error
        mock_consumer.poll.return_value = mock_msg
        mock_consumer_class.return_value = mock_consumer

        consumer = StreamingConsumer(consumer_config)

        with pytest.raises(KafkaException):
            consumer.poll(timeout=1.0)

    @patch("streaming.consumer.Consumer")
    def test_consume_batch(self, mock_consumer_class, consumer_config):
        """Test consuming a batch of messages."""
        mock_consumer = Mock()
        mock_msgs = []
        for i in range(3):
            mock_msg = Mock()
            mock_msg.error.return_value = None
            mock_msg.topic.return_value = "test-topic"
            mock_msg.partition.return_value = 0
            mock_msg.offset.return_value = i
            mock_msg.key.return_value = b"key"
            mock_msg.value.return_value = b"value"
            mock_msg.timestamp.return_value = (1, 1704067200000)
            mock_msg.headers.return_value = None
            mock_msgs.append(mock_msg)
        mock_consumer.consume.return_value = mock_msgs
        mock_consumer_class.return_value = mock_consumer

        consumer = StreamingConsumer(consumer_config)
        messages = consumer.consume_batch(num_messages=3, timeout=1.0)

        assert len(messages) == 3
        for i, msg in enumerate(messages):
            assert msg.offset == i

    @patch("streaming.consumer.Consumer")
    def test_commit_sync(self, mock_consumer_class, consumer_config):
        """Test synchronous commit."""
        mock_consumer = Mock()
        mock_consumer_class.return_value = mock_consumer

        consumer = StreamingConsumer(consumer_config)
        consumer.commit()

        mock_consumer.commit.assert_called_once_with(asynchronous=False)

    @patch("streaming.consumer.Consumer")
    def test_commit_async(self, mock_consumer_class, consumer_config):
        """Test asynchronous commit."""
        mock_consumer = Mock()
        mock_consumer_class.return_value = mock_consumer

        consumer = StreamingConsumer(consumer_config)
        consumer.commit(asynchronous=True)

        mock_consumer.commit.assert_called_once_with(asynchronous=True)

    @patch("streaming.consumer.Consumer")
    def test_commit_specific_message(self, mock_consumer_class, consumer_config, sample_message):
        """Test committing specific message offset."""
        mock_consumer = Mock()
        mock_consumer_class.return_value = mock_consumer

        consumer = StreamingConsumer(consumer_config)
        consumer.commit(message=sample_message)

        mock_consumer.commit.assert_called_once()
        call_kwargs = mock_consumer.commit.call_args[1]
        assert len(call_kwargs["offsets"]) == 1
        tp = call_kwargs["offsets"][0]
        assert tp.topic == "test-topic"
        assert tp.partition == 0
        assert tp.offset == 124  # message offset + 1

    @patch("streaming.consumer.Consumer")
    def test_seek(self, mock_consumer_class, consumer_config):
        """Test seeking to specific offset."""
        mock_consumer = Mock()
        mock_consumer_class.return_value = mock_consumer

        consumer = StreamingConsumer(consumer_config)
        tp = TopicPartition("test", 0, 100)
        consumer.seek(tp)

        mock_consumer.seek.assert_called_once_with(tp)

    @patch("streaming.consumer.Consumer")
    def test_seek_to_beginning(self, mock_consumer_class, consumer_config):
        """Test seeking to beginning of partitions."""
        mock_consumer = Mock()
        mock_consumer.assignment.return_value = [
            TopicPartition("test", 0),
            TopicPartition("test", 1),
        ]
        mock_consumer_class.return_value = mock_consumer

        consumer = StreamingConsumer(consumer_config)
        consumer.seek_to_beginning()

        assert mock_consumer.seek.call_count == 2

    @patch("streaming.consumer.Consumer")
    def test_seek_to_end(self, mock_consumer_class, consumer_config):
        """Test seeking to end of partitions."""
        mock_consumer = Mock()
        mock_consumer.assignment.return_value = [
            TopicPartition("test", 0),
            TopicPartition("test", 1),
        ]
        mock_consumer.get_watermark_offsets.return_value = (0, 100)
        mock_consumer_class.return_value = mock_consumer

        consumer = StreamingConsumer(consumer_config)
        consumer.seek_to_end()

        assert mock_consumer.seek.call_count == 2

    @patch("streaming.consumer.Consumer")
    def test_pause_partitions(self, mock_consumer_class, consumer_config):
        """Test pausing partitions."""
        mock_consumer = Mock()
        mock_consumer_class.return_value = mock_consumer

        consumer = StreamingConsumer(consumer_config)
        partitions = [TopicPartition("test", 0)]
        consumer.pause(partitions)

        mock_consumer.pause.assert_called_once_with(partitions)

    @patch("streaming.consumer.Consumer")
    def test_resume_partitions(self, mock_consumer_class, consumer_config):
        """Test resuming partitions."""
        mock_consumer = Mock()
        mock_consumer_class.return_value = mock_consumer

        consumer = StreamingConsumer(consumer_config)
        partitions = [TopicPartition("test", 0)]
        consumer.resume(partitions)

        mock_consumer.resume.assert_called_once_with(partitions)

    @patch("streaming.consumer.Consumer")
    def test_get_assignment(self, mock_consumer_class, consumer_config):
        """Test getting current assignment."""
        mock_consumer = Mock()
        expected = [TopicPartition("test", 0), TopicPartition("test", 1)]
        mock_consumer.assignment.return_value = expected
        mock_consumer_class.return_value = mock_consumer

        consumer = StreamingConsumer(consumer_config)
        result = consumer.get_assignment()

        assert result == expected

    @patch("streaming.consumer.Consumer")
    def test_get_committed(self, mock_consumer_class, consumer_config):
        """Test getting committed offsets."""
        mock_consumer = Mock()
        partitions = [TopicPartition("test", 0)]
        mock_consumer.committed.return_value = partitions
        mock_consumer_class.return_value = mock_consumer

        consumer = StreamingConsumer(consumer_config)
        result = consumer.get_committed(partitions)

        mock_consumer.committed.assert_called_once()

    @patch("streaming.consumer.Consumer")
    def test_get_position(self, mock_consumer_class, consumer_config):
        """Test getting current position."""
        mock_consumer = Mock()
        partitions = [TopicPartition("test", 0)]
        mock_consumer.position.return_value = partitions
        mock_consumer_class.return_value = mock_consumer

        consumer = StreamingConsumer(consumer_config)
        result = consumer.get_position(partitions)

        mock_consumer.position.assert_called_once_with(partitions)

    @patch("streaming.consumer.Consumer")
    def test_shutdown(self, mock_consumer_class, consumer_config):
        """Test shutdown sets running to false."""
        mock_consumer = Mock()
        mock_consumer_class.return_value = mock_consumer

        consumer = StreamingConsumer(consumer_config)
        consumer._running = True
        consumer.shutdown()

        assert consumer._running is False

    @patch("streaming.consumer.Consumer")
    def test_close(self, mock_consumer_class, consumer_config):
        """Test close calls consumer close."""
        mock_consumer = Mock()
        mock_consumer_class.return_value = mock_consumer

        consumer = StreamingConsumer(consumer_config)
        consumer.close()

        mock_consumer.close.assert_called_once()


# --- ConsumerGroup Tests ---


class TestConsumerGroup:
    """Tests for ConsumerGroup class."""

    @patch("streaming.consumer.Consumer")
    def test_create_single_consumer(self, mock_consumer_class, consumer_config):
        """Test creating a single consumer in a group."""
        mock_consumer = Mock()
        mock_consumer_class.return_value = mock_consumer

        group = ConsumerGroup(consumer_config, num_consumers=1)
        consumers = group.create_consumers(["topic1"])

        assert len(consumers) == 1
        mock_consumer.subscribe.assert_called()

    @patch("streaming.consumer.Consumer")
    def test_create_multiple_consumers(self, mock_consumer_class, consumer_config):
        """Test creating multiple consumers in a group."""
        mock_consumer = Mock()
        mock_consumer_class.return_value = mock_consumer

        group = ConsumerGroup(consumer_config, num_consumers=3)
        consumers = group.create_consumers(["topic1", "topic2"])

        assert len(consumers) == 3
        assert mock_consumer.subscribe.call_count == 3

    @patch("streaming.consumer.Consumer")
    def test_shutdown_all(self, mock_consumer_class, consumer_config):
        """Test shutting down all consumers."""
        mock_consumer = Mock()
        mock_consumer_class.return_value = mock_consumer

        group = ConsumerGroup(consumer_config, num_consumers=3)
        group.create_consumers(["topic1"])
        group.shutdown_all()

        # All consumers should have _running set to False
        for consumer in group._consumers:
            assert consumer._running is False

    @patch("streaming.consumer.Consumer")
    def test_close_all(self, mock_consumer_class, consumer_config):
        """Test closing all consumers."""
        mock_consumer = Mock()
        mock_consumer_class.return_value = mock_consumer

        group = ConsumerGroup(consumer_config, num_consumers=2)
        group.create_consumers(["topic1"])
        group.close_all()

        assert mock_consumer.close.call_count == 2


# --- ConsumerConfig Tests ---


class TestConsumerConfig:
    """Tests for ConsumerConfig class."""

    def test_to_dict_basic(self):
        """Test basic config conversion."""
        config = ConsumerConfig(
            bootstrap_servers="localhost:9092",
            group_id="test-group",
        )

        result = config.to_dict()

        assert result["bootstrap.servers"] == "localhost:9092"
        assert result["group.id"] == "test-group"
        assert result["auto.offset.reset"] == "earliest"
        assert result["enable.auto.commit"] is False

    def test_isolation_level(self):
        """Test isolation level configuration."""
        config = ConsumerConfig(
            bootstrap_servers="localhost:9092",
            group_id="test-group",
            isolation_level="read_committed",
        )

        result = config.to_dict()

        assert result["isolation.level"] == "read_committed"

    def test_performance_settings(self):
        """Test performance configuration."""
        config = ConsumerConfig(
            bootstrap_servers="localhost:9092",
            group_id="test-group",
            max_poll_records=1000,
            max_poll_interval_ms=600000,
            fetch_min_bytes=1024,
            fetch_max_wait_ms=1000,
        )

        result = config.to_dict()

        assert result["max.poll.interval.ms"] == 600000
