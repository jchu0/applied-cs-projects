"""Tests for Kafka producer APIs."""

import pytest
from unittest.mock import Mock, MagicMock, patch

pytest.importorskip("confluent_kafka")

from streaming.config import ProducerConfig, TopicConfig
from streaming.producer import (
    DeliveryReport,
    StreamingProducer,
    TransactionalProducer,
    TopicManager,
)


# --- Fixtures ---


@pytest.fixture
def producer_config():
    """Create a producer configuration for testing."""
    return ProducerConfig(
        bootstrap_servers="localhost:9092",
        acks="all",
        enable_idempotence=True,
        retries=3,
        batch_size=16384,
        linger_ms=10,
        compression_type="lz4",
    )


@pytest.fixture
def transactional_config():
    """Create a transactional producer configuration."""
    return ProducerConfig(
        bootstrap_servers="localhost:9092",
        acks="all",
        enable_idempotence=True,
        transactional_id="test-txn-id",
    )


@pytest.fixture
def topic_config():
    """Create a topic configuration for testing."""
    return TopicConfig(
        name="test-topic",
        partitions=3,
        replication_factor=1,
        retention_ms=86400000,
    )


# --- DeliveryReport Tests ---


class TestDeliveryReport:
    """Tests for DeliveryReport class."""

    def test_initial_state(self):
        """Test initial state of delivery report."""
        report = DeliveryReport()

        assert report.success_count == 0
        assert report.error_count == 0
        assert report.errors == []

    def test_callback_success(self):
        """Test callback with successful delivery."""
        report = DeliveryReport()
        mock_msg = Mock()
        mock_msg.topic.return_value = "test-topic"
        mock_msg.partition.return_value = 0
        mock_msg.offset.return_value = 123

        report.callback(None, mock_msg)

        assert report.success_count == 1
        assert report.error_count == 0
        assert len(report.errors) == 0

    def test_callback_error(self):
        """Test callback with delivery error."""
        report = DeliveryReport()
        mock_msg = Mock()
        mock_msg.topic.return_value = "test-topic"
        error = Exception("Connection refused")

        report.callback(error, mock_msg)

        assert report.success_count == 0
        assert report.error_count == 1
        assert len(report.errors) == 1
        assert "Delivery failed" in report.errors[0]

    def test_multiple_callbacks(self):
        """Test multiple callbacks tracking."""
        report = DeliveryReport()
        mock_msg = Mock()
        mock_msg.topic.return_value = "test-topic"
        mock_msg.partition.return_value = 0
        mock_msg.offset.return_value = 0

        # 3 successes
        for _ in range(3):
            report.callback(None, mock_msg)

        # 2 failures
        error = Exception("Error")
        for _ in range(2):
            report.callback(error, mock_msg)

        assert report.success_count == 3
        assert report.error_count == 2


# --- StreamingProducer Tests ---


class TestStreamingProducer:
    """Tests for StreamingProducer class."""

    @patch("streaming.producer.Producer")
    def test_producer_initialization(self, mock_producer_class, producer_config):
        """Test producer initialization with config."""
        mock_producer = Mock()
        mock_producer_class.return_value = mock_producer

        producer = StreamingProducer(producer_config)

        mock_producer_class.assert_called_once()
        assert producer.config == producer_config

    @patch("streaming.producer.Producer")
    def test_produce_message(self, mock_producer_class, producer_config):
        """Test producing a single message."""
        mock_producer = Mock()
        mock_producer_class.return_value = mock_producer

        producer = StreamingProducer(producer_config)
        producer.produce(
            topic="test-topic",
            key="test-key",
            value=b"test-value",
        )

        mock_producer.produce.assert_called_once()
        call_kwargs = mock_producer.produce.call_args[1]
        assert call_kwargs["topic"] == "test-topic"
        assert call_kwargs["key"] == b"test-key"
        assert call_kwargs["value"] == b"test-value"

    @patch("streaming.producer.Producer")
    def test_produce_with_headers(self, mock_producer_class, producer_config):
        """Test producing a message with headers."""
        mock_producer = Mock()
        mock_producer_class.return_value = mock_producer

        producer = StreamingProducer(producer_config)
        producer.produce(
            topic="test-topic",
            key="test-key",
            value=b"test-value",
            headers={"trace-id": "abc123"},
        )

        call_kwargs = mock_producer.produce.call_args[1]
        assert call_kwargs["headers"] == [("trace-id", b"abc123")]

    @patch("streaming.producer.Producer")
    def test_produce_with_timestamp(self, mock_producer_class, producer_config):
        """Test producing a message with timestamp."""
        mock_producer = Mock()
        mock_producer_class.return_value = mock_producer

        producer = StreamingProducer(producer_config)
        producer.produce(
            topic="test-topic",
            key="test-key",
            value=b"test-value",
            timestamp_ms=1704067200000,
        )

        call_kwargs = mock_producer.produce.call_args[1]
        assert call_kwargs["timestamp"] == 1704067200000

    @patch("streaming.producer.Producer")
    def test_produce_with_partition(self, mock_producer_class, producer_config):
        """Test producing to a specific partition."""
        mock_producer = Mock()
        mock_producer_class.return_value = mock_producer

        producer = StreamingProducer(producer_config)
        producer.produce(
            topic="test-topic",
            key="test-key",
            value=b"test-value",
            partition=2,
        )

        call_kwargs = mock_producer.produce.call_args[1]
        assert call_kwargs["partition"] == 2

    @patch("streaming.producer.Producer")
    def test_produce_batch(self, mock_producer_class, producer_config):
        """Test producing a batch of messages."""
        mock_producer = Mock()
        mock_producer_class.return_value = mock_producer

        producer = StreamingProducer(producer_config)
        messages = [
            {"key": f"key-{i}", "value": f"value-{i}".encode()}
            for i in range(10)
        ]

        report = producer.produce_batch("test-topic", messages)

        assert mock_producer.produce.call_count == 10
        assert mock_producer.flush.called

    @patch("streaming.producer.Producer")
    def test_produce_buffer_full_retry(self, mock_producer_class, producer_config):
        """Test retry when buffer is full."""
        mock_producer = Mock()
        mock_producer_class.return_value = mock_producer

        # First call raises BufferError, second succeeds
        mock_producer.produce.side_effect = [BufferError(), None]

        producer = StreamingProducer(producer_config)
        producer.produce(
            topic="test-topic",
            key="test-key",
            value=b"test-value",
        )

        # Should have flushed and retried
        mock_producer.flush.assert_called()
        assert mock_producer.produce.call_count == 2

    @patch("streaming.producer.Producer")
    def test_flush(self, mock_producer_class, producer_config):
        """Test flush method."""
        mock_producer = Mock()
        mock_producer.flush.return_value = 0
        mock_producer_class.return_value = mock_producer

        producer = StreamingProducer(producer_config)
        result = producer.flush(timeout=5.0)

        mock_producer.flush.assert_called_once_with(5.0)
        assert result == 0

    @patch("streaming.producer.Producer")
    def test_poll(self, mock_producer_class, producer_config):
        """Test poll method."""
        mock_producer = Mock()
        mock_producer.poll.return_value = 3
        mock_producer_class.return_value = mock_producer

        producer = StreamingProducer(producer_config)
        result = producer.poll(timeout=0.1)

        mock_producer.poll.assert_called_once_with(0.1)
        assert result == 3

    @patch("streaming.producer.Producer")
    def test_close(self, mock_producer_class, producer_config):
        """Test close method flushes producer."""
        mock_producer = Mock()
        mock_producer_class.return_value = mock_producer

        producer = StreamingProducer(producer_config)
        producer.close()

        mock_producer.flush.assert_called()

    @patch("streaming.producer.Producer")
    def test_get_delivery_report(self, mock_producer_class, producer_config):
        """Test getting delivery report."""
        mock_producer = Mock()
        mock_producer_class.return_value = mock_producer

        producer = StreamingProducer(producer_config)
        report = producer.get_delivery_report()

        assert isinstance(report, DeliveryReport)
        assert report.success_count == 0

    @patch("streaming.producer.Producer")
    def test_custom_delivery_callback(self, mock_producer_class, producer_config):
        """Test using custom delivery callback."""
        mock_producer = Mock()
        mock_producer_class.return_value = mock_producer

        custom_callback = Mock()
        producer = StreamingProducer(producer_config)
        producer.produce(
            topic="test-topic",
            key="test-key",
            value=b"test-value",
            on_delivery=custom_callback,
        )

        call_kwargs = mock_producer.produce.call_args[1]
        assert call_kwargs["on_delivery"] == custom_callback


# --- TransactionalProducer Tests ---


class TestTransactionalProducer:
    """Tests for TransactionalProducer class."""

    def test_requires_transactional_id(self, producer_config):
        """Test that transactional ID is required."""
        with pytest.raises(ValueError, match="transactional_id required"):
            TransactionalProducer(producer_config)

    @patch("streaming.producer.Producer")
    def test_init_transactions(self, mock_producer_class, transactional_config):
        """Test that transactions are initialized."""
        mock_producer = Mock()
        mock_producer_class.return_value = mock_producer

        producer = TransactionalProducer(transactional_config)

        mock_producer.init_transactions.assert_called_once()

    @patch("streaming.producer.Producer")
    def test_begin_transaction(self, mock_producer_class, transactional_config):
        """Test beginning a transaction."""
        mock_producer = Mock()
        mock_producer_class.return_value = mock_producer

        producer = TransactionalProducer(transactional_config)
        producer.begin_transaction()

        mock_producer.begin_transaction.assert_called_once()

    @patch("streaming.producer.Producer")
    def test_commit_transaction(self, mock_producer_class, transactional_config):
        """Test committing a transaction."""
        mock_producer = Mock()
        mock_producer_class.return_value = mock_producer

        producer = TransactionalProducer(transactional_config)
        producer.commit_transaction()

        mock_producer.commit_transaction.assert_called_once()

    @patch("streaming.producer.Producer")
    def test_abort_transaction(self, mock_producer_class, transactional_config):
        """Test aborting a transaction."""
        mock_producer = Mock()
        mock_producer_class.return_value = mock_producer

        producer = TransactionalProducer(transactional_config)
        producer.abort_transaction()

        mock_producer.abort_transaction.assert_called_once()

    @patch("streaming.producer.Producer")
    def test_produce_transactionally_success(
        self, mock_producer_class, transactional_config
    ):
        """Test successful transactional produce."""
        mock_producer = Mock()
        mock_producer_class.return_value = mock_producer

        producer = TransactionalProducer(transactional_config)
        messages = [
            {"topic": "topic1", "key": "k1", "value": b"v1"},
            {"topic": "topic2", "key": "k2", "value": b"v2"},
        ]

        result = producer.produce_transactionally(messages)

        assert result is True
        mock_producer.begin_transaction.assert_called_once()
        mock_producer.commit_transaction.assert_called_once()
        assert mock_producer.produce.call_count == 2

    @patch("streaming.producer.Producer")
    def test_produce_transactionally_abort_on_error(
        self, mock_producer_class, transactional_config
    ):
        """Test transaction abort on error."""
        from confluent_kafka import KafkaException

        mock_producer = Mock()
        mock_producer.produce.side_effect = KafkaException("Network error")
        mock_producer_class.return_value = mock_producer

        producer = TransactionalProducer(transactional_config)
        messages = [{"topic": "topic1", "key": "k1", "value": b"v1"}]

        with pytest.raises(KafkaException):
            producer.produce_transactionally(messages)

        mock_producer.abort_transaction.assert_called_once()

    @patch("streaming.producer.Producer")
    def test_begin_transaction_without_id(self, mock_producer_class, producer_config):
        """Test begin transaction fails without transactional ID."""
        mock_producer = Mock()
        mock_producer_class.return_value = mock_producer

        producer = StreamingProducer(producer_config)

        with pytest.raises(RuntimeError, match="Transactions not configured"):
            producer.begin_transaction()


# --- TopicManager Tests ---


class TestTopicManager:
    """Tests for TopicManager class."""

    @patch("streaming.producer.AdminClient")
    def test_create_topic(self, mock_admin_class, producer_config, topic_config):
        """Test creating a topic."""
        mock_admin = Mock()
        mock_future = Mock()
        mock_future.result.return_value = None
        mock_admin.create_topics.return_value = {"test-topic": mock_future}
        mock_admin_class.return_value = mock_admin

        manager = TopicManager(producer_config)
        result = manager.create_topic(topic_config)

        assert result is True
        mock_admin.create_topics.assert_called_once()

    @patch("streaming.producer.AdminClient")
    def test_create_topic_already_exists(
        self, mock_admin_class, producer_config, topic_config
    ):
        """Test creating a topic that already exists."""
        from confluent_kafka import KafkaError, KafkaException

        mock_admin = Mock()
        mock_future = Mock()
        mock_error = Mock(spec=KafkaError)
        mock_error.code.return_value = KafkaError.TOPIC_ALREADY_EXISTS
        mock_future.result.side_effect = KafkaException(mock_error)
        mock_admin.create_topics.return_value = {"test-topic": mock_future}
        mock_admin_class.return_value = mock_admin

        manager = TopicManager(producer_config)
        result = manager.create_topic(topic_config)

        assert result is True

    @patch("streaming.producer.AdminClient")
    def test_delete_topic(self, mock_admin_class, producer_config):
        """Test deleting a topic."""
        mock_admin = Mock()
        mock_future = Mock()
        mock_future.result.return_value = None
        mock_admin.delete_topics.return_value = {"test-topic": mock_future}
        mock_admin_class.return_value = mock_admin

        manager = TopicManager(producer_config)
        result = manager.delete_topic("test-topic")

        assert result is True
        mock_admin.delete_topics.assert_called_once_with(["test-topic"])

    @patch("streaming.producer.AdminClient")
    def test_list_topics(self, mock_admin_class, producer_config):
        """Test listing topics."""
        mock_admin = Mock()
        mock_metadata = Mock()
        mock_metadata.topics = {"topic1": None, "topic2": None, "topic3": None}
        mock_admin.list_topics.return_value = mock_metadata
        mock_admin_class.return_value = mock_admin

        manager = TopicManager(producer_config)
        topics = manager.list_topics()

        assert len(topics) == 3
        assert "topic1" in topics
        assert "topic2" in topics
        assert "topic3" in topics

    @patch("streaming.producer.AdminClient")
    def test_topic_exists_true(self, mock_admin_class, producer_config):
        """Test checking if topic exists (true case)."""
        mock_admin = Mock()
        mock_metadata = Mock()
        mock_metadata.topics = {"test-topic": None}
        mock_admin.list_topics.return_value = mock_metadata
        mock_admin_class.return_value = mock_admin

        manager = TopicManager(producer_config)
        result = manager.topic_exists("test-topic")

        assert result is True

    @patch("streaming.producer.AdminClient")
    def test_topic_exists_false(self, mock_admin_class, producer_config):
        """Test checking if topic exists (false case)."""
        mock_admin = Mock()
        mock_metadata = Mock()
        mock_metadata.topics = {"other-topic": None}
        mock_admin.list_topics.return_value = mock_metadata
        mock_admin_class.return_value = mock_admin

        manager = TopicManager(producer_config)
        result = manager.topic_exists("test-topic")

        assert result is False


# --- Config Tests ---


class TestProducerConfig:
    """Tests for ProducerConfig class."""

    def test_to_dict_basic(self):
        """Test basic config conversion."""
        config = ProducerConfig(
            bootstrap_servers="localhost:9092",
            acks="all",
        )

        result = config.to_dict()

        assert result["bootstrap.servers"] == "localhost:9092"
        assert result["acks"] == "all"
        assert result["enable.idempotence"] is True

    def test_to_dict_with_transactional_id(self):
        """Test config with transactional ID."""
        config = ProducerConfig(
            bootstrap_servers="localhost:9092",
            transactional_id="my-txn-id",
        )

        result = config.to_dict()

        assert result["transactional.id"] == "my-txn-id"

    def test_to_dict_with_sasl(self):
        """Test config with SASL authentication."""
        config = ProducerConfig(
            bootstrap_servers="localhost:9092",
            security_protocol="SASL_SSL",
            sasl_mechanism="PLAIN",
            sasl_username="user",
            sasl_password="pass",
        )

        result = config.to_dict()

        assert result["security.protocol"] == "SASL_SSL"
        assert result["sasl.mechanism"] == "PLAIN"
        assert result["sasl.username"] == "user"
        assert result["sasl.password"] == "pass"

    def test_compression_settings(self):
        """Test compression configuration."""
        config = ProducerConfig(
            bootstrap_servers="localhost:9092",
            compression_type="snappy",
            batch_size=32768,
            linger_ms=20,
        )

        result = config.to_dict()

        assert result["compression.type"] == "snappy"
        assert result["batch.size"] == 32768
        assert result["linger.ms"] == 20
