"""Kafka producer with schema registry support."""

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from confluent_kafka import Producer, KafkaError, KafkaException
from confluent_kafka.admin import AdminClient, NewTopic

from streaming.config import ProducerConfig, TopicConfig

logger = logging.getLogger(__name__)


class DeliveryReport:
    """Delivery report for produced messages."""

    def __init__(self):
        self.success_count = 0
        self.error_count = 0
        self.errors: List[str] = []

    def callback(self, err, msg):
        """Callback for message delivery."""
        if err:
            self.error_count += 1
            self.errors.append(f"Delivery failed for {msg.topic()}: {err}")
            logger.error(f"Delivery failed: {err}")
        else:
            self.success_count += 1
            logger.debug(
                f"Delivered to {msg.topic()} [{msg.partition()}] @ {msg.offset()}"
            )


class StreamingProducer:
    """High-level Kafka producer with batching and reliability features."""

    def __init__(self, config: ProducerConfig):
        self.config = config
        self._producer = Producer(config.to_dict())
        self._delivery_report = DeliveryReport()

        # Initialize transactions if configured
        if config.transactional_id:
            self._producer.init_transactions()

    def produce(
        self,
        topic: str,
        key: str,
        value: bytes,
        headers: Optional[Dict[str, str]] = None,
        timestamp_ms: Optional[int] = None,
        partition: Optional[int] = None,
        on_delivery: Optional[Callable] = None,
    ) -> None:
        """
        Produce a message to a topic.

        Args:
            topic: Target topic
            key: Message key
            value: Serialized message value
            headers: Optional message headers
            timestamp_ms: Optional message timestamp
            partition: Optional target partition
            on_delivery: Optional delivery callback
        """
        callback = on_delivery or self._delivery_report.callback

        # Convert headers to list of tuples
        kafka_headers = None
        if headers:
            kafka_headers = [(k, v.encode()) for k, v in headers.items()]

        try:
            self._producer.produce(
                topic=topic,
                key=key.encode() if isinstance(key, str) else key,
                value=value,
                headers=kafka_headers,
                timestamp=timestamp_ms,
                partition=partition if partition is not None else -1,
                on_delivery=callback,
            )
        except BufferError:
            # Buffer full - flush and retry
            logger.warning("Producer buffer full, flushing...")
            self._producer.flush(timeout=30)
            self._producer.produce(
                topic=topic,
                key=key.encode() if isinstance(key, str) else key,
                value=value,
                headers=kafka_headers,
                timestamp=timestamp_ms,
                partition=partition if partition is not None else -1,
                on_delivery=callback,
            )

    def produce_batch(
        self,
        topic: str,
        messages: List[Dict[str, Any]],
        poll_interval: int = 100,
    ) -> DeliveryReport:
        """
        Produce a batch of messages.

        Args:
            topic: Target topic
            messages: List of messages with 'key', 'value', optional 'headers'
            poll_interval: Messages between polls

        Returns:
            DeliveryReport with success/error counts
        """
        report = DeliveryReport()

        for i, msg in enumerate(messages):
            self.produce(
                topic=topic,
                key=msg["key"],
                value=msg["value"],
                headers=msg.get("headers"),
                timestamp_ms=msg.get("timestamp"),
                on_delivery=report.callback,
            )

            # Poll to trigger callbacks periodically
            if i % poll_interval == 0:
                self._producer.poll(0)

        # Final flush
        self._producer.flush()

        return report

    def flush(self, timeout: float = 30.0) -> int:
        """
        Flush pending messages.

        Args:
            timeout: Maximum time to wait

        Returns:
            Number of messages still in queue
        """
        return self._producer.flush(timeout)

    def poll(self, timeout: float = 0) -> int:
        """
        Poll for delivery callbacks.

        Args:
            timeout: Time to wait for events

        Returns:
            Number of events processed
        """
        return self._producer.poll(timeout)

    def get_delivery_report(self) -> DeliveryReport:
        """Get the current delivery report."""
        return self._delivery_report

    def close(self) -> None:
        """Close the producer."""
        self._producer.flush()

    # Transaction support

    def begin_transaction(self) -> None:
        """Begin a new transaction."""
        if not self.config.transactional_id:
            raise RuntimeError("Transactions not configured")
        self._producer.begin_transaction()

    def commit_transaction(self, timeout: float = -1) -> None:
        """Commit the current transaction."""
        self._producer.commit_transaction(timeout)

    def abort_transaction(self, timeout: float = -1) -> None:
        """Abort the current transaction."""
        self._producer.abort_transaction(timeout)

    def send_offsets_to_transaction(
        self,
        positions,
        group_metadata,
        timeout: float = -1,
    ) -> None:
        """Send consumer offsets to transaction for exactly-once."""
        self._producer.send_offsets_to_transaction(
            positions, group_metadata, timeout
        )


class TransactionalProducer(StreamingProducer):
    """Producer with transaction support for exactly-once semantics."""

    def __init__(self, config: ProducerConfig):
        if not config.transactional_id:
            raise ValueError("transactional_id required for TransactionalProducer")
        super().__init__(config)

    def produce_transactionally(
        self,
        messages: List[Dict[str, Any]],
    ) -> bool:
        """
        Produce messages in a transaction.

        Args:
            messages: List of messages with 'topic', 'key', 'value'

        Returns:
            True if transaction committed successfully
        """
        try:
            self.begin_transaction()

            for msg in messages:
                self.produce(
                    topic=msg["topic"],
                    key=msg["key"],
                    value=msg["value"],
                    headers=msg.get("headers"),
                )

            self.commit_transaction()
            return True

        except KafkaException as e:
            logger.error(f"Transaction failed: {e}")
            try:
                self.abort_transaction()
            except Exception:
                pass
            raise


class TopicManager:
    """Manage Kafka topics."""

    def __init__(self, config: ProducerConfig):
        self._admin = AdminClient(config.to_dict())

    def create_topic(
        self,
        topic_config: TopicConfig,
        timeout: float = 30.0,
    ) -> bool:
        """
        Create a topic.

        Args:
            topic_config: Topic configuration
            timeout: Operation timeout

        Returns:
            True if created successfully
        """
        new_topic = NewTopic(
            topic_config.name,
            num_partitions=topic_config.partitions,
            replication_factor=topic_config.replication_factor,
            config=topic_config.to_dict(),
        )

        futures = self._admin.create_topics([new_topic])

        for topic, future in futures.items():
            try:
                future.result(timeout=timeout)
                logger.info(f"Topic {topic} created")
                return True
            except KafkaException as e:
                if e.args[0].code() == KafkaError.TOPIC_ALREADY_EXISTS:
                    logger.info(f"Topic {topic} already exists")
                    return True
                raise

        return False

    def delete_topic(self, topic_name: str, timeout: float = 30.0) -> bool:
        """Delete a topic."""
        futures = self._admin.delete_topics([topic_name])

        for topic, future in futures.items():
            try:
                future.result(timeout=timeout)
                logger.info(f"Topic {topic} deleted")
                return True
            except Exception as e:
                logger.error(f"Failed to delete topic {topic}: {e}")
                raise

        return False

    def list_topics(self, timeout: float = 10.0) -> List[str]:
        """List all topics."""
        metadata = self._admin.list_topics(timeout=timeout)
        return list(metadata.topics.keys())

    def topic_exists(self, topic_name: str, timeout: float = 10.0) -> bool:
        """Check if a topic exists."""
        return topic_name in self.list_topics(timeout)
