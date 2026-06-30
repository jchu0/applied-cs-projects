# Real-Time Streaming Platform (Kafka/Flink) - Technical Blueprint

## Executive Summary

This project implements a production-grade real-time streaming platform built on Apache Kafka (or Redpanda) and Apache Flink. It demonstrates mastery of streaming semantics, event-time processing, stateful stream processing, and fault-tolerant distributed systems. The platform supports exactly-once processing, complex event processing patterns, and enterprise-grade features like schema registry and multi-region replication.

> **Concepts covered:** [§02 Kafka streaming](../../../02-data-engineering/04-streaming/kafka/kafka-streaming.md) · [§02 Flink streaming](../../../02-data-engineering/04-streaming/flink/flink-streaming.md) · [§02 Real-time analytics](../../../02-data-engineering/04-streaming/real-time-analytics/real-time-analytics.md). For the Kafka-internals view, see [Project 12 (distributed log)](../../12-distributed-log-system/) and [Project 51 (message queue)](../../51-message-queue/); for the Flink-internals view, [Project 36 (streaming analytics)](../../36-distributed-streaming-analytics/). Map: [`CONCEPT_TO_PROJECT_MAP.md`](../../CONCEPT_TO_PROJECT_MAP.md).

**Primary Goals:**
- Build a complete streaming infrastructure with Kafka/Redpanda as the backbone
- Implement stateful stream processing with Apache Flink
- Achieve exactly-once semantics end-to-end
- Support complex event processing patterns (windowing, sessionization, joins)

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Data Sources                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │   CDC    │  │   APIs   │  │   IoT    │  │   Logs   │            │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘            │
│       │             │             │             │                   │
└───────┴─────────────┴─────────────┴─────────────┴───────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Kafka / Redpanda Cluster                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Topics: raw.events, enriched.events, aggregated.metrics    │   │
│  │  Partitions │ Replication │ Compaction │ Retention          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │Schema Registry│  │ Kafka Connect│  │   Streams   │              │
│  │  (Avro/Proto) │  │  (Sources/   │  │   Metadata  │              │
│  │              │  │   Sinks)     │  │   Store     │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└────────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Apache Flink Cluster                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   Job Manager (HA)                           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐       │
│  │Task Manager│ │Task Manager│ │Task Manager│ │Task Manager│       │
│  │  Slots: 4  │ │  Slots: 4  │ │  Slots: 4  │ │  Slots: 4  │       │
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘       │
│                                                                     │
│  State Backend: RocksDB │ Checkpoints: S3 │ Savepoints            │
└────────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Data Sinks                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │  Kafka   │  │   S3     │  │   JDBC   │  │  Redis   │            │
│  │ (output) │  │ (archive)│  │   (DW)   │  │ (cache)  │            │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │
└─────────────────────────────────────────────────────────────────────┘
```

### Event Processing Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                    Event Time Processing                          │
│                                                                    │
│  Event Time    ──┬── Watermark ──┬── Window ──┬── Output ──┬──   │
│  Assignment      │   Generation   │   Trigger   │  Results   │     │
│                  │                │             │            │     │
│  [10:00:01]      │   W=09:59:55   │             │            │     │
│  [10:00:03]      │   W=09:59:58   │             │            │     │
│  [10:00:02]      │   W=09:59:57   │  [10:00]    │  Results   │     │
│  [10:00:05]      │   W=10:00:00   │  Window     │  for 10:00 │     │
│                  │                │  Fires      │            │     │
└──────────────────────────────────────────────────────────────────┘
```

---

## Core Internals

### Kafka Producer API

```java
import org.apache.kafka.clients.producer.*;
import org.apache.kafka.common.serialization.StringSerializer;
import io.confluent.kafka.serializers.KafkaAvroSerializer;

public class StreamingProducer {
    private final KafkaProducer<String, GenericRecord> producer;
    private final String topic;

    public StreamingProducer(String bootstrapServers, String schemaRegistryUrl, String topic) {
        this.topic = topic;

        Properties props = new Properties();
        props.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, bootstrapServers);
        props.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, StringSerializer.class);
        props.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, KafkaAvroSerializer.class);

        // Schema Registry
        props.put("schema.registry.url", schemaRegistryUrl);

        // Exactly-once semantics
        props.put(ProducerConfig.ENABLE_IDEMPOTENCE_CONFIG, true);
        props.put(ProducerConfig.ACKS_CONFIG, "all");
        props.put(ProducerConfig.RETRIES_CONFIG, Integer.MAX_VALUE);
        props.put(ProducerConfig.MAX_IN_FLIGHT_REQUESTS_PER_CONNECTION, 5);

        // Batching for throughput
        props.put(ProducerConfig.BATCH_SIZE_CONFIG, 16384);
        props.put(ProducerConfig.LINGER_MS_CONFIG, 10);
        props.put(ProducerConfig.COMPRESSION_TYPE_CONFIG, "lz4");

        // Buffer memory
        props.put(ProducerConfig.BUFFER_MEMORY_CONFIG, 33554432);

        this.producer = new KafkaProducer<>(props);
    }

    public CompletableFuture<RecordMetadata> sendAsync(String key, GenericRecord value) {
        CompletableFuture<RecordMetadata> future = new CompletableFuture<>();

        ProducerRecord<String, GenericRecord> record = new ProducerRecord<>(topic, key, value);

        producer.send(record, (metadata, exception) -> {
            if (exception != null) {
                future.completeExceptionally(exception);
            } else {
                future.complete(metadata);
            }
        });

        return future;
    }

    public void sendWithCallback(String key, GenericRecord value, Callback callback) {
        ProducerRecord<String, GenericRecord> record = new ProducerRecord<>(topic, key, value);
        producer.send(record, callback);
    }

    public void close() {
        producer.flush();
        producer.close();
    }
}
```

### Kafka Consumer API

```java
import org.apache.kafka.clients.consumer.*;
import org.apache.kafka.common.TopicPartition;

public class StreamingConsumer {
    private final KafkaConsumer<String, GenericRecord> consumer;
    private final List<String> topics;
    private volatile boolean running = true;

    public StreamingConsumer(String bootstrapServers, String groupId, List<String> topics) {
        this.topics = topics;

        Properties props = new Properties();
        props.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, bootstrapServers);
        props.put(ConsumerConfig.GROUP_ID_CONFIG, groupId);
        props.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class);
        props.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, KafkaAvroDeserializer.class);

        // Schema Registry
        props.put("schema.registry.url", schemaRegistryUrl);
        props.put("specific.avro.reader", true);

        // Consumer configuration
        props.put(ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "earliest");
        props.put(ConsumerConfig.ENABLE_AUTO_COMMIT_CONFIG, false);  // Manual commits
        props.put(ConsumerConfig.MAX_POLL_RECORDS_CONFIG, 500);
        props.put(ConsumerConfig.MAX_POLL_INTERVAL_MS_CONFIG, 300000);

        // Isolation level for exactly-once
        props.put(ConsumerConfig.ISOLATION_LEVEL_CONFIG, "read_committed");

        this.consumer = new KafkaConsumer<>(props);
    }

    public void consume(RecordProcessor processor) {
        consumer.subscribe(topics, new RebalanceListener());

        try {
            while (running) {
                ConsumerRecords<String, GenericRecord> records = consumer.poll(Duration.ofMillis(100));

                for (ConsumerRecord<String, GenericRecord> record : records) {
                    try {
                        processor.process(record);
                    } catch (Exception e) {
                        handleProcessingError(record, e);
                    }
                }

                // Commit after processing batch
                consumer.commitSync();
            }
        } finally {
            consumer.close();
        }
    }

    private class RebalanceListener implements ConsumerRebalanceListener {
        @Override
        public void onPartitionsRevoked(Collection<TopicPartition> partitions) {
            // Commit current offsets before rebalance
            consumer.commitSync();
        }

        @Override
        public void onPartitionsAssigned(Collection<TopicPartition> partitions) {
            // Optionally seek to specific offsets
        }
    }

    public void shutdown() {
        running = false;
        consumer.wakeup();
    }
}
```

### Flink Stream Processing

```java
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.windowing.time.Time;
import org.apache.flink.streaming.api.windowing.assigners.*;
import org.apache.flink.streaming.connectors.kafka.*;

public class StreamProcessor {
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        // Checkpointing for exactly-once
        env.enableCheckpointing(60000);
        env.getCheckpointConfig().setCheckpointingMode(CheckpointingMode.EXACTLY_ONCE);
        env.getCheckpointConfig().setMinPauseBetweenCheckpoints(30000);
        env.getCheckpointConfig().setCheckpointTimeout(120000);
        env.getCheckpointConfig().setMaxConcurrentCheckpoints(1);

        // State backend
        env.setStateBackend(new EmbeddedRocksDBStateBackend());
        env.getCheckpointConfig().setCheckpointStorage("s3://checkpoints/");

        // Kafka source
        KafkaSource<Event> source = KafkaSource.<Event>builder()
            .setBootstrapServers("kafka:9092")
            .setTopics("events")
            .setGroupId("flink-processor")
            .setStartingOffsets(OffsetsInitializer.committedOffsets(OffsetResetStrategy.EARLIEST))
            .setDeserializer(new EventDeserializer())
            .build();

        DataStream<Event> events = env.fromSource(
            source,
            WatermarkStrategy
                .<Event>forBoundedOutOfOrderness(Duration.ofSeconds(10))
                .withTimestampAssigner((event, ts) -> event.getTimestamp()),
            "Kafka Source"
        );

        // Processing pipeline
        DataStream<AggregatedMetric> metrics = events
            .filter(event -> event.getType() != null)
            .keyBy(Event::getUserId)
            .window(TumblingEventTimeWindows.of(Time.minutes(5)))
            .aggregate(new MetricsAggregator());

        // Kafka sink with exactly-once
        KafkaSink<AggregatedMetric> sink = KafkaSink.<AggregatedMetric>builder()
            .setBootstrapServers("kafka:9092")
            .setRecordSerializer(new MetricSerializer())
            .setDeliveryGuarantee(DeliveryGuarantee.EXACTLY_ONCE)
            .setTransactionalIdPrefix("flink-metrics")
            .build();

        metrics.sinkTo(sink);

        env.execute("Stream Processing Job");
    }
}
```

### Watermark Strategy

```java
public class EventWatermarkStrategy implements WatermarkStrategy<Event> {

    @Override
    public WatermarkGenerator<Event> createWatermarkGenerator(WatermarkGeneratorSupplier.Context context) {
        return new BoundedOutOfOrdernessWatermarks<>(Duration.ofSeconds(10));
    }

    @Override
    public TimestampAssigner<Event> createTimestampAssigner(TimestampAssignerSupplier.Context context) {
        return (event, recordTimestamp) -> event.getEventTime();
    }
}

// Custom watermark generator for handling idle partitions
public class IdleAwareWatermarkGenerator implements WatermarkGenerator<Event> {
    private final long maxOutOfOrderness;
    private final long idleTimeout;
    private long currentMaxTimestamp;
    private long lastRecordTime;

    public IdleAwareWatermarkGenerator(long maxOutOfOrderness, long idleTimeout) {
        this.maxOutOfOrderness = maxOutOfOrderness;
        this.idleTimeout = idleTimeout;
        this.currentMaxTimestamp = Long.MIN_VALUE;
        this.lastRecordTime = System.currentTimeMillis();
    }

    @Override
    public void onEvent(Event event, long eventTimestamp, WatermarkOutput output) {
        currentMaxTimestamp = Math.max(currentMaxTimestamp, eventTimestamp);
        lastRecordTime = System.currentTimeMillis();
    }

    @Override
    public void onPeriodicEmit(WatermarkOutput output) {
        long now = System.currentTimeMillis();

        if (now - lastRecordTime > idleTimeout) {
            // Emit watermark at current time for idle sources
            output.emitWatermark(new Watermark(now - maxOutOfOrderness));
        } else if (currentMaxTimestamp != Long.MIN_VALUE) {
            output.emitWatermark(new Watermark(currentMaxTimestamp - maxOutOfOrderness));
        }
    }
}
```

---

## Data Structures

### Event Schema (Avro)

```json
{
  "type": "record",
  "name": "Event",
  "namespace": "com.streaming.events",
  "fields": [
    {"name": "event_id", "type": "string"},
    {"name": "event_type", "type": "string"},
    {"name": "user_id", "type": "string"},
    {"name": "timestamp", "type": "long", "logicalType": "timestamp-millis"},
    {"name": "payload", "type": {
      "type": "map",
      "values": "string"
    }},
    {"name": "metadata", "type": {
      "type": "record",
      "name": "Metadata",
      "fields": [
        {"name": "source", "type": "string"},
        {"name": "version", "type": "int"},
        {"name": "correlation_id", "type": ["null", "string"], "default": null}
      ]
    }}
  ]
}
```

### State Management

```java
// Keyed state for user sessions
public class SessionProcessFunction
    extends KeyedProcessFunction<String, Event, Session> {

    private ValueState<Session> sessionState;
    private ValueState<Long> timerState;
    private final long sessionTimeout;

    @Override
    public void open(Configuration parameters) {
        ValueStateDescriptor<Session> sessionDescriptor =
            new ValueStateDescriptor<>("session", Session.class);
        sessionState = getRuntimeContext().getState(sessionDescriptor);

        ValueStateDescriptor<Long> timerDescriptor =
            new ValueStateDescriptor<>("timer", Long.class);
        timerState = getRuntimeContext().getState(timerDescriptor);
    }

    @Override
    public void processElement(Event event, Context ctx, Collector<Session> out) throws Exception {
        Session session = sessionState.value();

        if (session == null) {
            // New session
            session = new Session(event.getUserId(), event.getTimestamp());
        }

        // Update session
        session.addEvent(event);
        sessionState.update(session);

        // Reset session timeout timer
        Long oldTimer = timerState.value();
        if (oldTimer != null) {
            ctx.timerService().deleteEventTimeTimer(oldTimer);
        }

        long newTimer = event.getTimestamp() + sessionTimeout;
        ctx.timerService().registerEventTimeTimer(newTimer);
        timerState.update(newTimer);
    }

    @Override
    public void onTimer(long timestamp, OnTimerContext ctx, Collector<Session> out) throws Exception {
        // Session timed out - emit and clear
        Session session = sessionState.value();
        if (session != null) {
            session.close(timestamp);
            out.collect(session);
            sessionState.clear();
            timerState.clear();
        }
    }
}
```

### Windowing Types

```java
// Tumbling windows - fixed size, non-overlapping
events.keyBy(Event::getUserId)
    .window(TumblingEventTimeWindows.of(Time.minutes(5)))
    .aggregate(new CountAggregator());

// Sliding windows - overlapping windows
events.keyBy(Event::getUserId)
    .window(SlidingEventTimeWindows.of(Time.minutes(10), Time.minutes(5)))
    .aggregate(new AverageAggregator());

// Session windows - gap-based grouping
events.keyBy(Event::getUserId)
    .window(EventTimeSessionWindows.withGap(Time.minutes(30)))
    .aggregate(new SessionAggregator());

// Global windows with custom trigger
events.keyBy(Event::getUserId)
    .window(GlobalWindows.create())
    .trigger(CountTrigger.of(100))
    .evictor(CountEvictor.of(100))
    .aggregate(new BatchAggregator());
```

---

## API Design

### Producer API

```python
from dataclasses import dataclass
from typing import Dict, Any, Optional
from confluent_kafka import Producer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer

class StreamingProducerClient:
    """High-level producer with schema registry support"""

    def __init__(self, config: ProducerConfig):
        self.config = config
        self.schema_registry = SchemaRegistryClient({
            'url': config.schema_registry_url
        })

        # Create Avro serializers per topic
        self.serializers: Dict[str, AvroSerializer] = {}

        # Producer configuration
        producer_config = {
            'bootstrap.servers': config.bootstrap_servers,
            'enable.idempotence': True,
            'acks': 'all',
            'retries': 10000000,
            'max.in.flight.requests.per.connection': 5,
            'compression.type': 'lz4',
            'linger.ms': 10,
            'batch.size': 16384,
        }

        self.producer = Producer(producer_config)

    def send(
        self,
        topic: str,
        key: str,
        value: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        timestamp_ms: Optional[int] = None
    ) -> None:
        """Send a message to a topic"""

        # Get or create serializer for topic
        if topic not in self.serializers:
            schema = self._get_schema(topic)
            self.serializers[topic] = AvroSerializer(
                self.schema_registry,
                schema
            )

        # Serialize value
        serialized = self.serializers[topic](value, None)

        # Convert headers
        kafka_headers = [(k, v.encode()) for k, v in (headers or {}).items()]

        # Produce
        self.producer.produce(
            topic=topic,
            key=key.encode(),
            value=serialized,
            headers=kafka_headers,
            timestamp=timestamp_ms,
            on_delivery=self._delivery_callback
        )

    def flush(self, timeout: float = 10.0) -> int:
        """Flush pending messages"""
        return self.producer.flush(timeout)

    def _delivery_callback(self, err, msg):
        if err:
            logger.error(f"Delivery failed: {err}")
        else:
            logger.debug(f"Delivered to {msg.topic()} [{msg.partition()}] @ {msg.offset()}")
```

### Consumer API

```python
from confluent_kafka import Consumer, KafkaError
from confluent_kafka.schema_registry.avro import AvroDeserializer

class StreamingConsumerClient:
    """High-level consumer with manual commit support"""

    def __init__(self, config: ConsumerConfig):
        self.config = config
        self.schema_registry = SchemaRegistryClient({
            'url': config.schema_registry_url
        })

        consumer_config = {
            'bootstrap.servers': config.bootstrap_servers,
            'group.id': config.group_id,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': False,
            'isolation.level': 'read_committed',
            'max.poll.interval.ms': 300000,
        }

        self.consumer = Consumer(consumer_config)
        self.deserializers: Dict[str, AvroDeserializer] = {}
        self.running = True

    def subscribe(self, topics: List[str]) -> None:
        """Subscribe to topics"""
        self.consumer.subscribe(topics, on_assign=self._on_assign)

    def consume(self, handler: Callable[[Message], None], batch_size: int = 100) -> None:
        """Consume messages in batches"""
        try:
            while self.running:
                messages = self.consumer.consume(batch_size, timeout=1.0)

                for msg in messages:
                    if msg.error():
                        if msg.error().code() == KafkaError._PARTITION_EOF:
                            continue
                        else:
                            raise KafkaException(msg.error())

                    # Deserialize
                    value = self._deserialize(msg)

                    # Process
                    handler(Message(
                        topic=msg.topic(),
                        partition=msg.partition(),
                        offset=msg.offset(),
                        key=msg.key().decode() if msg.key() else None,
                        value=value,
                        timestamp=msg.timestamp()[1],
                        headers=dict(msg.headers() or [])
                    ))

                # Commit batch
                self.consumer.commit()

        finally:
            self.consumer.close()

    def shutdown(self) -> None:
        """Graceful shutdown"""
        self.running = False
```

### Flink Job API

```python
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment

class FlinkJobBuilder:
    """Builder for Flink streaming jobs"""

    def __init__(self, job_name: str):
        self.job_name = job_name
        self.env = StreamExecutionEnvironment.get_execution_environment()
        self.t_env = StreamTableEnvironment.create(self.env)

        # Default configuration
        self.env.set_parallelism(4)
        self.env.enable_checkpointing(60000)

    def with_checkpoint_config(
        self,
        interval_ms: int,
        mode: str = "EXACTLY_ONCE",
        timeout_ms: int = 120000
    ) -> 'FlinkJobBuilder':
        """Configure checkpointing"""
        self.env.enable_checkpointing(interval_ms)
        config = self.env.get_checkpoint_config()
        config.set_checkpoint_timeout(timeout_ms)
        return self

    def with_state_backend(
        self,
        backend_type: str = "rocksdb",
        checkpoint_path: str = None
    ) -> 'FlinkJobBuilder':
        """Configure state backend"""
        if backend_type == "rocksdb":
            from pyflink.datastream import EmbeddedRocksDBStateBackend
            self.env.set_state_backend(EmbeddedRocksDBStateBackend())
        if checkpoint_path:
            self.env.get_checkpoint_config().set_checkpoint_storage(checkpoint_path)
        return self

    def add_kafka_source(
        self,
        topic: str,
        bootstrap_servers: str,
        group_id: str,
        schema: str
    ) -> 'FlinkJobBuilder':
        """Add Kafka source table"""
        self.t_env.execute_sql(f"""
            CREATE TABLE {topic}_source (
                {schema},
                event_time TIMESTAMP(3),
                WATERMARK FOR event_time AS event_time - INTERVAL '10' SECOND
            ) WITH (
                'connector' = 'kafka',
                'topic' = '{topic}',
                'properties.bootstrap.servers' = '{bootstrap_servers}',
                'properties.group.id' = '{group_id}',
                'scan.startup.mode' = 'latest-offset',
                'format' = 'avro-confluent',
                'avro-confluent.url' = 'http://schema-registry:8081'
            )
        """)
        return self

    def execute(self) -> None:
        """Execute the job"""
        self.env.execute(self.job_name)
```

---

## Enterprise Features

### 1. Schema Registry Integration

```python
from confluent_kafka.schema_registry import SchemaRegistryClient, Schema

class SchemaManager:
    """Manage Avro/Protobuf schemas in Confluent Schema Registry"""

    def __init__(self, registry_url: str):
        self.client = SchemaRegistryClient({'url': registry_url})

    def register_schema(
        self,
        subject: str,
        schema_str: str,
        schema_type: str = "AVRO"
    ) -> int:
        """Register a new schema version"""
        schema = Schema(schema_str, schema_type)
        schema_id = self.client.register_schema(subject, schema)
        return schema_id

    def get_latest_schema(self, subject: str) -> Schema:
        """Get the latest schema for a subject"""
        return self.client.get_latest_version(subject)

    def check_compatibility(
        self,
        subject: str,
        schema_str: str,
        schema_type: str = "AVRO"
    ) -> bool:
        """Check if schema is compatible with existing versions"""
        schema = Schema(schema_str, schema_type)
        return self.client.test_compatibility(subject, schema)

    def set_compatibility(
        self,
        subject: str,
        level: str = "BACKWARD"
    ) -> None:
        """Set compatibility level (BACKWARD, FORWARD, FULL, NONE)"""
        self.client.set_compatibility(subject, level)

    def get_schema_by_id(self, schema_id: int) -> Schema:
        """Get schema by ID"""
        return self.client.get_schema(schema_id)
```

### 2. Exactly-Once Semantics

```java
// Kafka Transactions for exactly-once
public class ExactlyOnceProducer {
    private final KafkaProducer<String, String> producer;

    public ExactlyOnceProducer(String bootstrapServers, String transactionalId) {
        Properties props = new Properties();
        props.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, bootstrapServers);
        props.put(ProducerConfig.TRANSACTIONAL_ID_CONFIG, transactionalId);
        props.put(ProducerConfig.ENABLE_IDEMPOTENCE_CONFIG, true);
        props.put(ProducerConfig.ACKS_CONFIG, "all");

        this.producer = new KafkaProducer<>(props);
        this.producer.initTransactions();
    }

    public void sendInTransaction(List<ProducerRecord<String, String>> records) {
        try {
            producer.beginTransaction();

            for (ProducerRecord<String, String> record : records) {
                producer.send(record);
            }

            producer.commitTransaction();
        } catch (ProducerFencedException | OutOfOrderSequenceException e) {
            // Fatal errors - cannot recover
            producer.close();
            throw e;
        } catch (KafkaException e) {
            // Abort and retry
            producer.abortTransaction();
            throw e;
        }
    }

    // Consume-transform-produce pattern
    public void consumeTransformProduce(
        KafkaConsumer<String, String> consumer,
        String outputTopic,
        Function<String, String> transformer
    ) {
        while (true) {
            ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));

            if (!records.isEmpty()) {
                producer.beginTransaction();

                try {
                    for (ConsumerRecord<String, String> record : records) {
                        String transformed = transformer.apply(record.value());
                        producer.send(new ProducerRecord<>(outputTopic, record.key(), transformed));
                    }

                    // Commit consumer offsets in transaction
                    Map<TopicPartition, OffsetAndMetadata> offsets = getOffsetsToCommit(records);
                    producer.sendOffsetsToTransaction(offsets, consumer.groupMetadata());

                    producer.commitTransaction();
                } catch (Exception e) {
                    producer.abortTransaction();
                }
            }
        }
    }
}
```

### 3. Multi-Region Replication

```python
class MirrorMakerConfig:
    """Configuration for Kafka MirrorMaker 2.0 replication"""

    def __init__(self):
        self.clusters = {}
        self.replications = []

    def add_cluster(
        self,
        name: str,
        bootstrap_servers: str,
        config: Dict[str, str] = None
    ) -> 'MirrorMakerConfig':
        self.clusters[name] = {
            'bootstrap.servers': bootstrap_servers,
            **(config or {})
        }
        return self

    def add_replication(
        self,
        source: str,
        target: str,
        topics: str = ".*",
        config: Dict[str, str] = None
    ) -> 'MirrorMakerConfig':
        self.replications.append({
            'source': source,
            'target': target,
            'topics': topics,
            'config': config or {}
        })
        return self

    def generate_config(self) -> str:
        """Generate MirrorMaker 2.0 configuration file"""
        config_lines = []

        # Cluster configurations
        for name, cluster_config in self.clusters.items():
            for key, value in cluster_config.items():
                config_lines.append(f"{name}.{key} = {value}")

        # Replication configurations
        for repl in self.replications:
            prefix = f"{repl['source']}->{repl['target']}"
            config_lines.append(f"{prefix}.enabled = true")
            config_lines.append(f"{prefix}.topics = {repl['topics']}")

            for key, value in repl['config'].items():
                config_lines.append(f"{prefix}.{key} = {value}")

        return "\n".join(config_lines)
```

### 4. Log Compaction

```python
class CompactedTopicManager:
    """Manage compacted topics for materialized views"""

    def create_compacted_topic(
        self,
        topic: str,
        partitions: int,
        config: Dict[str, str] = None
    ):
        """Create a compacted topic"""
        default_config = {
            'cleanup.policy': 'compact',
            'min.cleanable.dirty.ratio': '0.1',
            'delete.retention.ms': '86400000',  # 1 day
            'segment.ms': '604800000',  # 7 days
            'min.compaction.lag.ms': '0',
        }
        default_config.update(config or {})

        admin_client.create_topics([
            NewTopic(topic, partitions, replication_factor=3, config=default_config)
        ])

    def tombstone(self, topic: str, key: str):
        """Send tombstone (null value) to delete key"""
        producer.send(ProducerRecord(topic, key, None))
```

---

## Performance Considerations

### Kafka Tuning

```python
# High-throughput producer configuration
producer_config = {
    'bootstrap.servers': 'kafka:9092',

    # Batching
    'batch.size': 65536,  # 64KB batches
    'linger.ms': 50,  # Wait up to 50ms for batching

    # Compression (lz4 for speed, zstd for ratio)
    'compression.type': 'lz4',

    # Buffer
    'buffer.memory': 67108864,  # 64MB buffer

    # Reliability
    'acks': 'all',
    'enable.idempotence': True,
    'max.in.flight.requests.per.connection': 5,
    'retries': 2147483647,
    'delivery.timeout.ms': 120000,
}

# Low-latency producer configuration
low_latency_config = {
    'batch.size': 0,  # No batching
    'linger.ms': 0,
    'acks': 1,  # Leader ack only
}

# High-throughput consumer configuration
consumer_config = {
    'fetch.min.bytes': 1048576,  # 1MB min fetch
    'fetch.max.wait.ms': 500,
    'max.partition.fetch.bytes': 10485760,  # 10MB per partition
    'max.poll.records': 1000,
}
```

### Flink Optimization

```java
// Memory configuration
Configuration config = new Configuration();
config.set(TaskManagerOptions.MANAGED_MEMORY_SIZE, MemorySize.ofMebiBytes(1024));
config.set(TaskManagerOptions.NETWORK_MEMORY_MIN, MemorySize.ofMebiBytes(64));
config.set(TaskManagerOptions.NETWORK_MEMORY_MAX, MemorySize.ofMebiBytes(256));

// Parallelism tuning
env.setParallelism(16);  // Match Kafka partitions
env.setMaxParallelism(128);  // Allow scaling

// State backend tuning for RocksDB
EmbeddedRocksDBStateBackend backend = new EmbeddedRocksDBStateBackend();
backend.setNumberOfTransferThreads(4);
backend.setNumberOfTransferingThreads(4);

// Checkpoint optimization
CheckpointConfig checkpointConfig = env.getCheckpointConfig();
checkpointConfig.setCheckpointingMode(CheckpointingMode.EXACTLY_ONCE);
checkpointConfig.setMinPauseBetweenCheckpoints(30000);
checkpointConfig.setCheckpointTimeout(60000);
checkpointConfig.setMaxConcurrentCheckpoints(1);

// Unaligned checkpoints for backpressure tolerance
checkpointConfig.enableUnalignedCheckpoints();
```

### Partitioning Strategy

```python
class PartitioningStrategy:
    """Strategies for distributing messages across partitions"""

    @staticmethod
    def by_key(key: str, num_partitions: int) -> int:
        """Default: hash-based partitioning by key"""
        return hash(key) % num_partitions

    @staticmethod
    def round_robin(counter: int, num_partitions: int) -> int:
        """Round-robin for even distribution (no ordering)"""
        return counter % num_partitions

    @staticmethod
    def by_timestamp(timestamp: int, num_partitions: int, bucket_ms: int) -> int:
        """Time-bucket partitioning"""
        bucket = timestamp // bucket_ms
        return bucket % num_partitions

    @staticmethod
    def sticky(current_partition: int, batch_full: bool, num_partitions: int) -> int:
        """Sticky partitioning for better batching"""
        if batch_full:
            return (current_partition + 1) % num_partitions
        return current_partition
```

---

## Stretch Goals

### 1. Real-Time Anomaly Detection

```python
class AnomalyDetector:
    """Streaming anomaly detection using statistical methods"""

    def __init__(self, window_size: int = 1000, num_std: float = 3.0):
        self.window_size = window_size
        self.num_std = num_std

    def create_detector(self, env: StreamExecutionEnvironment):
        """Create Flink anomaly detection job"""
        events = env.from_source(kafka_source, watermark_strategy, "events")

        # Calculate rolling statistics
        stats = events \
            .key_by(lambda e: e.metric_name) \
            .window(SlidingEventTimeWindows.of(Time.minutes(10), Time.minutes(1))) \
            .aggregate(StatisticsAggregator())

        # Detect anomalies
        anomalies = events \
            .connect(stats.broadcast(StateBroadcastDescriptor())) \
            .process(AnomalyProcessFunction(self.num_std))

        return anomalies

class AnomalyProcessFunction(KeyedBroadcastProcessFunction):
    """Detect anomalies based on broadcast statistics"""

    def process_element(self, value, ctx, out):
        # Get statistics from broadcast state
        stats = ctx.get_broadcast_state(self.stats_descriptor).get(value.metric_name)

        if stats:
            z_score = abs(value.value - stats.mean) / stats.stddev

            if z_score > self.num_std:
                out.collect(Anomaly(
                    metric_name=value.metric_name,
                    value=value.value,
                    expected_range=(
                        stats.mean - self.num_std * stats.stddev,
                        stats.mean + self.num_std * stats.stddev
                    ),
                    z_score=z_score,
                    timestamp=value.timestamp
                ))

    def process_broadcast_element(self, value, ctx, out):
        # Update statistics
        ctx.get_broadcast_state(self.stats_descriptor).put(value.metric_name, value)
```

### 2. Elastic Autoscaling

```python
class FlinkAutoscaler:
    """Autoscale Flink job based on metrics"""

    def __init__(self, flink_client: FlinkRestClient):
        self.client = flink_client
        self.metrics_history = []

    def calculate_desired_parallelism(
        self,
        job_id: str,
        current_parallelism: int
    ) -> int:
        """Calculate desired parallelism based on backpressure and throughput"""
        metrics = self.client.get_job_metrics(job_id)

        # Get backpressure metrics
        backpressure = metrics.get('backpressure', 0)

        # Get throughput metrics
        input_rate = metrics.get('numRecordsInPerSecond', 0)
        output_rate = metrics.get('numRecordsOutPerSecond', 0)

        # Get consumer lag from Kafka
        consumer_lag = self._get_consumer_lag(job_id)

        # Scaling logic
        if backpressure > 0.5 or consumer_lag > 100000:
            # Scale up
            return min(current_parallelism * 2, self.max_parallelism)
        elif backpressure < 0.1 and consumer_lag < 1000:
            # Scale down
            return max(current_parallelism // 2, self.min_parallelism)

        return current_parallelism

    def rescale(self, job_id: str, new_parallelism: int):
        """Rescale job with savepoint"""
        # Take savepoint
        savepoint_path = self.client.trigger_savepoint(job_id)

        # Stop job
        self.client.cancel_job(job_id)

        # Restart with new parallelism
        self.client.run_job(
            job_id,
            savepoint_path=savepoint_path,
            parallelism=new_parallelism
        )
```

---

## Testing Strategy

### Unit Tests

```python
import pytest
from unittest.mock import Mock, patch
from testcontainers.kafka import KafkaContainer

@pytest.fixture(scope="module")
def kafka_container():
    with KafkaContainer() as kafka:
        yield kafka

class TestProducer:
    def test_sends_message_to_kafka(self, kafka_container):
        producer = StreamingProducerClient(ProducerConfig(
            bootstrap_servers=kafka_container.get_bootstrap_server()
        ))

        producer.send("test-topic", "key", {"field": "value"})
        producer.flush()

        # Verify message
        consumer = create_consumer(kafka_container.get_bootstrap_server())
        messages = list(consumer.consume("test-topic", timeout=5.0))
        assert len(messages) == 1
        assert messages[0].key == "key"

class TestFlinkProcessing:
    def test_window_aggregation(self):
        env = StreamExecutionEnvironment.get_execution_environment()
        env.set_parallelism(1)

        # Create test data
        test_events = [
            Event("user1", 100, 1000),
            Event("user1", 200, 2000),
            Event("user2", 50, 1500),
        ]

        # Process
        result = env.from_collection(test_events) \
            .key_by(lambda e: e.user_id) \
            .window(TumblingEventTimeWindows.of(Time.seconds(5))) \
            .aggregate(SumAggregator()) \
            .execute_and_collect()

        # Verify
        results = list(result)
        assert len(results) == 2
```

### Integration Tests

```python
class TestEndToEndPipeline:
    @pytest.fixture
    def streaming_env(self, kafka_container, flink_cluster):
        return StreamingTestEnvironment(kafka_container, flink_cluster)

    def test_exactly_once_delivery(self, streaming_env):
        """Verify exactly-once semantics end-to-end"""
        # Produce messages
        producer = streaming_env.create_producer()
        for i in range(1000):
            producer.send("input", f"key-{i}", {"value": i})
        producer.flush()

        # Run Flink job
        job = streaming_env.submit_job("exactly_once_job.jar")

        # Wait for processing
        streaming_env.wait_for_job_completion(job)

        # Verify output
        consumer = streaming_env.create_consumer()
        output = list(consumer.consume("output", timeout=30.0))

        # Check no duplicates
        keys = [msg.key for msg in output]
        assert len(keys) == len(set(keys)), "Duplicates found"
        assert len(keys) == 1000, f"Expected 1000 messages, got {len(keys)}"

    def test_failure_recovery(self, streaming_env):
        """Verify recovery from failures"""
        # Start job
        job = streaming_env.submit_job("stateful_job.jar")

        # Produce some messages
        producer = streaming_env.create_producer()
        for i in range(500):
            producer.send("input", f"key-{i}", {"value": i})
        producer.flush()

        # Wait for checkpoint
        streaming_env.wait_for_checkpoint(job)

        # Kill task manager
        streaming_env.kill_task_manager()

        # Produce more messages
        for i in range(500, 1000):
            producer.send("input", f"key-{i}", {"value": i})
        producer.flush()

        # Wait for recovery and completion
        streaming_env.wait_for_job_completion(job)

        # Verify all messages processed
        consumer = streaming_env.create_consumer()
        output = list(consumer.consume("output", timeout=30.0))
        assert len(output) == 1000
```

### Performance Tests

```python
class TestPerformance:
    def test_throughput(self, streaming_env):
        """Measure sustained throughput"""
        producer = streaming_env.create_producer()

        # Produce 1M messages
        start = time.time()
        for i in range(1_000_000):
            producer.send("input", f"key-{i}", {"value": i})
        producer.flush()
        produce_time = time.time() - start

        throughput = 1_000_000 / produce_time
        print(f"Producer throughput: {throughput:.0f} msgs/sec")
        assert throughput > 100_000  # At least 100K msgs/sec

    def test_latency_p99(self, streaming_env):
        """Measure end-to-end latency"""
        latencies = []

        for i in range(1000):
            start = time.time()
            producer.send("input", f"key-{i}", {"value": i, "ts": start})
            producer.flush()

            # Consume output
            msg = consumer.consume_one("output", timeout=5.0)
            end = time.time()

            latencies.append(end - start)

        p99 = np.percentile(latencies, 99)
        print(f"P99 latency: {p99*1000:.0f}ms")
        assert p99 < 1.0  # Less than 1 second
```

---

## Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
- Set up Kafka/Redpanda cluster
- Implement producer and consumer APIs
- Basic Flink job structure
- Simple streaming pipeline

### Phase 2: Core Streaming (Weeks 3-4)
- Watermark and event-time handling
- Window operations (tumbling, sliding, session)
- State management
- Exactly-once checkpointing

### Phase 3: Advanced Patterns (Weeks 5-6)
- Debouncing and deduplication
- Sessionization
- Complex event processing
- Stream-stream joins

### Phase 4: Enterprise Features (Weeks 7-8)
- Schema registry integration
- Exactly-once sinks
- Multi-region replication
- Log compaction

### Phase 5: Stretch Goals (Weeks 9-10)
- Real-time anomaly detection
- Elastic autoscaling
- Performance optimization
- Production hardening

---

## References

- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
- [Apache Flink Documentation](https://flink.apache.org/docs/)
- [Confluent Schema Registry](https://docs.confluent.io/platform/current/schema-registry/)
- [Designing Data-Intensive Applications](https://dataintensive.net/)
- [Streaming Systems Book](https://www.oreilly.com/library/view/streaming-systems/9781491983867/)
