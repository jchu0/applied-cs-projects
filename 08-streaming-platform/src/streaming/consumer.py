"""Kafka consumer with manual commit and exactly-once support."""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from confluent_kafka import Consumer, KafkaError, KafkaException, TopicPartition

from streaming.config import ConsumerConfig

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Consumed message wrapper."""

    topic: str
    partition: int
    offset: int
    key: Optional[str]
    value: bytes
    timestamp: int
    headers: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "topic": self.topic,
            "partition": self.partition,
            "offset": self.offset,
            "key": self.key,
            "value": self.value,
            "timestamp": self.timestamp,
            "headers": self.headers,
        }


class RebalanceListener:
    """Callback handler for partition rebalances."""

    def __init__(self, consumer: Consumer):
        self._consumer = consumer
        self._on_assign_callback: Optional[Callable] = None
        self._on_revoke_callback: Optional[Callable] = None

    def set_on_assign(self, callback: Callable) -> None:
        """Set callback for partition assignment."""
        self._on_assign_callback = callback

    def set_on_revoke(self, callback: Callable) -> None:
        """Set callback for partition revocation."""
        self._on_revoke_callback = callback

    def on_assign(self, consumer, partitions):
        """Called when partitions are assigned."""
        logger.info(f"Partitions assigned: {partitions}")
        if self._on_assign_callback:
            self._on_assign_callback(partitions)

    def on_revoke(self, consumer, partitions):
        """Called when partitions are revoked."""
        logger.info(f"Partitions revoked: {partitions}")
        # Commit current offsets before rebalance
        try:
            consumer.commit(asynchronous=False)
        except KafkaException as e:
            logger.warning(f"Commit failed during rebalance: {e}")

        if self._on_revoke_callback:
            self._on_revoke_callback(partitions)


class StreamingConsumer:
    """High-level Kafka consumer with manual commit support."""

    def __init__(self, config: ConsumerConfig):
        self.config = config
        self._consumer = Consumer(config.to_dict())
        self._rebalance_listener = RebalanceListener(self._consumer)
        self._running = False
        self._subscribed_topics: List[str] = []

    def subscribe(
        self,
        topics: List[str],
        on_assign: Optional[Callable] = None,
        on_revoke: Optional[Callable] = None,
    ) -> None:
        """
        Subscribe to topics.

        Args:
            topics: List of topics to subscribe to
            on_assign: Callback for partition assignment
            on_revoke: Callback for partition revocation
        """
        if on_assign:
            self._rebalance_listener.set_on_assign(on_assign)
        if on_revoke:
            self._rebalance_listener.set_on_revoke(on_revoke)

        self._consumer.subscribe(
            topics,
            on_assign=self._rebalance_listener.on_assign,
            on_revoke=self._rebalance_listener.on_revoke,
        )
        self._subscribed_topics = topics
        logger.info(f"Subscribed to topics: {topics}")

    def assign(self, partitions: List[TopicPartition]) -> None:
        """Manually assign partitions."""
        self._consumer.assign(partitions)
        logger.info(f"Assigned partitions: {partitions}")

    def poll(self, timeout: float = 1.0) -> Optional[Message]:
        """
        Poll for a single message.

        Args:
            timeout: Maximum time to wait

        Returns:
            Message if available, None otherwise
        """
        msg = self._consumer.poll(timeout)

        if msg is None:
            return None

        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                logger.debug(f"Reached end of partition {msg.partition()}")
                return None
            else:
                raise KafkaException(msg.error())

        return self._wrap_message(msg)

    def consume_batch(
        self,
        num_messages: int = 100,
        timeout: float = 1.0,
    ) -> List[Message]:
        """
        Consume a batch of messages.

        Args:
            num_messages: Maximum messages to consume
            timeout: Maximum time to wait

        Returns:
            List of messages
        """
        messages = self._consumer.consume(num_messages, timeout)

        result = []
        for msg in messages:
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    raise KafkaException(msg.error())
            result.append(self._wrap_message(msg))

        return result

    def consume(
        self,
        handler: Callable[[Message], None],
        batch_size: int = 100,
        poll_timeout: float = 1.0,
        commit_interval: int = 100,
    ) -> None:
        """
        Consume messages continuously.

        Args:
            handler: Function to process each message
            batch_size: Messages per batch
            poll_timeout: Poll timeout in seconds
            commit_interval: Messages between commits
        """
        self._running = True
        messages_since_commit = 0

        try:
            while self._running:
                messages = self.consume_batch(batch_size, poll_timeout)

                for msg in messages:
                    try:
                        handler(msg)
                        messages_since_commit += 1

                        if messages_since_commit >= commit_interval:
                            self.commit()
                            messages_since_commit = 0

                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        # Could implement dead letter queue here
                        raise

                # Commit remaining messages
                if messages_since_commit > 0:
                    self.commit()
                    messages_since_commit = 0

        finally:
            self._consumer.close()

    def commit(
        self,
        message: Optional[Message] = None,
        asynchronous: bool = False,
    ) -> None:
        """
        Commit offsets.

        Args:
            message: Optional specific message to commit
            asynchronous: Whether to commit asynchronously
        """
        if message:
            tp = TopicPartition(
                message.topic, message.partition, message.offset + 1
            )
            self._consumer.commit(offsets=[tp], asynchronous=asynchronous)
        else:
            self._consumer.commit(asynchronous=asynchronous)

    def get_committed(
        self,
        partitions: List[TopicPartition],
        timeout: float = 10.0,
    ) -> List[TopicPartition]:
        """Get committed offsets for partitions."""
        return self._consumer.committed(partitions, timeout=timeout)

    def get_position(
        self,
        partitions: List[TopicPartition],
    ) -> List[TopicPartition]:
        """Get current position for partitions."""
        return self._consumer.position(partitions)

    def seek(self, partition: TopicPartition) -> None:
        """Seek to a specific offset."""
        self._consumer.seek(partition)
        logger.info(f"Seeked to {partition}")

    def seek_to_beginning(self, partitions: Optional[List[TopicPartition]] = None) -> None:
        """Seek to beginning of partitions."""
        if partitions:
            for tp in partitions:
                tp.offset = 0
                self._consumer.seek(tp)
        else:
            assignment = self._consumer.assignment()
            for tp in assignment:
                tp.offset = 0
                self._consumer.seek(tp)

    def seek_to_end(self, partitions: Optional[List[TopicPartition]] = None) -> None:
        """Seek to end of partitions."""
        # Get high watermarks
        if not partitions:
            partitions = self._consumer.assignment()

        for tp in partitions:
            _, high = self._consumer.get_watermark_offsets(tp)
            tp.offset = high
            self._consumer.seek(tp)

    def pause(self, partitions: List[TopicPartition]) -> None:
        """Pause consumption of partitions."""
        self._consumer.pause(partitions)
        logger.info(f"Paused partitions: {partitions}")

    def resume(self, partitions: List[TopicPartition]) -> None:
        """Resume consumption of partitions."""
        self._consumer.resume(partitions)
        logger.info(f"Resumed partitions: {partitions}")

    def get_assignment(self) -> List[TopicPartition]:
        """Get current partition assignment."""
        return self._consumer.assignment()

    def shutdown(self) -> None:
        """Signal consumer to stop."""
        self._running = False

    def close(self) -> None:
        """Close the consumer."""
        self._consumer.close()

    def _wrap_message(self, msg) -> Message:
        """Wrap a Kafka message in our Message class."""
        # Parse headers
        headers = {}
        if msg.headers():
            for key, value in msg.headers():
                if value:
                    headers[key] = value.decode() if isinstance(value, bytes) else value

        return Message(
            topic=msg.topic(),
            partition=msg.partition(),
            offset=msg.offset(),
            key=msg.key().decode() if msg.key() else None,
            value=msg.value(),
            timestamp=msg.timestamp()[1] if msg.timestamp()[0] != 0 else 0,
            headers=headers,
        )


class ConsumerGroup:
    """Manage a group of consumers for parallel processing."""

    def __init__(
        self,
        config: ConsumerConfig,
        num_consumers: int = 1,
    ):
        self.config = config
        self.num_consumers = num_consumers
        self._consumers: List[StreamingConsumer] = []

    def create_consumers(self, topics: List[str]) -> List[StreamingConsumer]:
        """Create and subscribe consumers."""
        for _ in range(self.num_consumers):
            consumer = StreamingConsumer(self.config)
            consumer.subscribe(topics)
            self._consumers.append(consumer)

        return self._consumers

    def shutdown_all(self) -> None:
        """Shutdown all consumers."""
        for consumer in self._consumers:
            consumer.shutdown()

    def close_all(self) -> None:
        """Close all consumers."""
        for consumer in self._consumers:
            consumer.close()
