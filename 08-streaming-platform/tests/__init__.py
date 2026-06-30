"""Tests for streaming platform.

Test modules:
- test_producer: Tests for Kafka producer APIs (DeliveryReport, StreamingProducer, TransactionalProducer, TopicManager)
- test_consumer: Tests for Kafka consumer APIs (Message, RebalanceListener, StreamingConsumer, ConsumerGroup)
- test_windowing: Tests for windowing operations (Window, WindowAssigner, TumblingWindow, SlidingWindow, SessionWindow, Watermarks, WindowOperator)
- test_state: Tests for state management (StateBackend, ValueState, ListState, MapState, TimerService, CheckpointCoordinator)
- test_exactly_once: Tests for exactly-once semantics (idempotent producers, transactions, deduplication, checkpoints)
- test_patterns: Tests for streaming patterns (Deduplicator, Debouncer, SessionProcessor, CEPEngine, StreamJoiner)
"""
