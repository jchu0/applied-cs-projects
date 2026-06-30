"""Tests for advanced streaming patterns."""

import pytest
import time
from unittest.mock import Mock, patch

pytest.importorskip("confluent_kafka")

from streaming.patterns import (
    DeduplicationConfig,
    Deduplicator,
    DebouncerConfig,
    Debouncer,
    Session,
    SessionProcessor,
    Pattern,
    PatternCondition,
    PatternMatch,
    CEPEngine,
    JoinConfig,
    StreamJoiner,
    IntervalJoiner,
)


# --- Deduplicator Tests ---


class TestDeduplicator:
    """Tests for Deduplicator class."""

    @pytest.fixture
    def dedup_config(self):
        """Create deduplication config."""
        return DeduplicationConfig(
            window_ms=60000,
            id_extractor=lambda x: x.get("id"),
        )

    def test_first_event_passes(self, dedup_config):
        """Test first event passes deduplication."""
        deduplicator = Deduplicator(dedup_config)
        event = {"id": "event-1", "data": "value"}

        result = deduplicator.process(event)

        assert result == event

    def test_duplicate_event_filtered(self, dedup_config):
        """Test duplicate event is filtered."""
        deduplicator = Deduplicator(dedup_config)
        event = {"id": "event-1", "data": "value"}

        deduplicator.process(event)
        result = deduplicator.process(event)

        assert result is None

    def test_different_ids_pass(self, dedup_config):
        """Test different event IDs pass."""
        deduplicator = Deduplicator(dedup_config)
        event1 = {"id": "event-1", "data": "value1"}
        event2 = {"id": "event-2", "data": "value2"}

        result1 = deduplicator.process(event1)
        result2 = deduplicator.process(event2)

        assert result1 == event1
        assert result2 == event2

    def test_is_duplicate_returns_boolean(self, dedup_config):
        """Test is_duplicate returns correct boolean."""
        deduplicator = Deduplicator(dedup_config)
        event = {"id": "event-1", "data": "value"}

        first = deduplicator.is_duplicate(event)
        second = deduplicator.is_duplicate(event)

        assert first is False
        assert second is True

    def test_window_expiry(self):
        """Test events expire after window."""
        config = DeduplicationConfig(
            window_ms=10,  # Very short window
            id_extractor=lambda x: x.get("id"),
        )
        deduplicator = Deduplicator(config)
        event = {"id": "event-1", "data": "value"}

        deduplicator.process(event)
        time.sleep(0.02)  # Wait for window to expire
        result = deduplicator.process(event)

        assert result == event  # Should pass after expiry


# --- Debouncer Tests ---


class TestDebouncer:
    """Tests for Debouncer class."""

    @pytest.fixture
    def debounce_config(self):
        """Create debouncer config."""
        return DebouncerConfig(
            window_ms=100,
            key_extractor=lambda x: x.get("user_id"),
        )

    def test_first_event_returns_none(self, debounce_config):
        """Test first event returns None (waiting for debounce)."""
        debouncer = Debouncer(debounce_config)
        event = {"user_id": "user1", "action": "click"}

        result = debouncer.process(event)

        assert result is None

    def test_subsequent_event_within_window_returns_none(self, debounce_config):
        """Test rapid events return None."""
        debouncer = Debouncer(debounce_config)
        event1 = {"user_id": "user1", "action": "click1"}
        event2 = {"user_id": "user1", "action": "click2"}

        debouncer.process(event1)
        result = debouncer.process(event2)

        assert result is None

    def test_event_after_window_returns_previous(self, debounce_config):
        """Test event after window returns previous event."""
        config = DebouncerConfig(
            window_ms=10,  # Short window
            key_extractor=lambda x: x.get("user_id"),
        )
        debouncer = Debouncer(config)

        event1 = {"user_id": "user1", "action": "click1"}
        event2 = {"user_id": "user1", "action": "click2"}

        debouncer.process(event1)
        time.sleep(0.02)  # Wait for window
        result = debouncer.process(event2)

        assert result == event1

    def test_flush_returns_pending_events(self, debounce_config):
        """Test flush returns all pending events."""
        debouncer = Debouncer(debounce_config)
        debouncer.process({"user_id": "user1", "action": "a"})
        debouncer.process({"user_id": "user2", "action": "b"})
        debouncer.process({"user_id": "user3", "action": "c"})

        results = debouncer.flush()

        assert len(results) == 3

    def test_flush_clears_pending(self, debounce_config):
        """Test flush clears pending events."""
        debouncer = Debouncer(debounce_config)
        debouncer.process({"user_id": "user1", "action": "a"})

        debouncer.flush()
        results = debouncer.flush()

        assert len(results) == 0


# --- Session Tests ---


class TestSession:
    """Tests for Session dataclass."""

    def test_session_creation(self):
        """Test session creation."""
        session = Session(
            session_id="sess-1",
            user_id="user-1",
            start_time=1000,
            end_time=5000,
            events=["e1", "e2"],
            event_count=2,
        )

        assert session.session_id == "sess-1"
        assert session.user_id == "user-1"
        assert session.duration_ms == 4000

    def test_session_duration(self):
        """Test session duration calculation."""
        session = Session(
            session_id="s1",
            user_id="u1",
            start_time=0,
            end_time=60000,
        )

        assert session.duration_ms == 60000


# --- SessionProcessor Tests ---


class TestSessionProcessor:
    """Tests for SessionProcessor class."""

    @pytest.fixture
    def session_processor(self):
        """Create session processor."""
        return SessionProcessor(
            gap_ms=30000,  # 30 second gap
            user_id_extractor=lambda x: x.get("user_id"),
            timestamp_extractor=lambda x: x.get("timestamp"),
        )

    def test_first_event_creates_session(self, session_processor):
        """Test first event creates a session."""
        event = {"user_id": "user1", "timestamp": 1000, "action": "view"}

        result = session_processor.process_event(event)

        assert result is None  # No session closed yet
        assert "user1" in session_processor._sessions

    def test_event_within_gap_extends_session(self, session_processor):
        """Test event within gap extends session."""
        event1 = {"user_id": "user1", "timestamp": 1000, "action": "view"}
        event2 = {"user_id": "user1", "timestamp": 10000, "action": "click"}

        session_processor.process_event(event1)
        result = session_processor.process_event(event2)

        assert result is None
        session = session_processor._sessions["user1"]
        assert session.end_time == 10000
        assert session.event_count == 2

    def test_event_after_gap_closes_session(self, session_processor):
        """Test event after gap closes previous session."""
        event1 = {"user_id": "user1", "timestamp": 1000, "action": "view"}
        event2 = {"user_id": "user1", "timestamp": 100000, "action": "click"}

        session_processor.process_event(event1)
        closed = session_processor.process_event(event2)

        assert closed is not None
        assert closed.user_id == "user1"
        assert closed.event_count == 1

    def test_multiple_users_independent_sessions(self, session_processor):
        """Test multiple users have independent sessions."""
        event1 = {"user_id": "user1", "timestamp": 1000, "action": "view"}
        event2 = {"user_id": "user2", "timestamp": 2000, "action": "view"}

        session_processor.process_event(event1)
        session_processor.process_event(event2)

        assert "user1" in session_processor._sessions
        assert "user2" in session_processor._sessions

    def test_close_idle_sessions(self, session_processor):
        """Test closing idle sessions."""
        event = {"user_id": "user1", "timestamp": 1000, "action": "view"}
        session_processor.process_event(event)

        closed = session_processor.close_idle_sessions(100000)

        assert len(closed) == 1
        assert closed[0].user_id == "user1"
        assert "user1" not in session_processor._sessions


# --- Pattern Tests ---


class TestPattern:
    """Tests for Pattern class."""

    def test_pattern_creation(self):
        """Test pattern creation."""
        pattern = Pattern("test-pattern")

        assert pattern.name == "test-pattern"
        assert len(pattern._conditions) == 0

    def test_pattern_with_condition(self):
        """Test pattern with single condition."""
        pattern = Pattern("test")
        pattern.begin("start").where(lambda e: e.get("type") == "A").one()

        assert len(pattern._conditions) == 1
        assert pattern._conditions[0][0] == "start"

    def test_pattern_with_time_constraint(self):
        """Test pattern with time constraint."""
        pattern = Pattern("test")
        pattern.within(60000)

        assert pattern._within_ms == 60000

    def test_chained_conditions(self):
        """Test chaining pattern conditions."""
        pattern = Pattern("sequence")
        (pattern
            .begin("first")
            .where(lambda e: e.get("type") == "A")
            .next("second")
            .where(lambda e: e.get("type") == "B")
            .one())

        assert len(pattern._conditions) == 2


# --- CEPEngine Tests ---


class TestCEPEngine:
    """Tests for CEPEngine class."""

    @pytest.fixture
    def cep_engine(self):
        """Create CEP engine."""
        return CEPEngine()

    def test_register_pattern(self, cep_engine):
        """Test registering a pattern."""
        pattern = Pattern("test-pattern")
        pattern.begin("start").where(lambda e: True).one()

        cep_engine.register_pattern(pattern)

        assert "test-pattern" in cep_engine._patterns

    def test_process_event_no_match(self, cep_engine):
        """Test processing event with no match."""
        pattern = Pattern("test")
        pattern.begin("start").where(lambda e: e.get("type") == "A").one()
        pattern.within(60000)
        cep_engine.register_pattern(pattern)

        matches = cep_engine.process_event(
            {"type": "B", "value": 1},
            timestamp=1000,
        )

        assert len(matches) == 0

    def test_process_event_with_match(self, cep_engine):
        """Test processing event with match."""
        pattern = Pattern("single-a")
        pattern.begin("start").where(lambda e: e.get("type") == "A").one()
        pattern.within(60000)
        cep_engine.register_pattern(pattern)

        matches = cep_engine.process_event(
            {"type": "A", "value": 1},
            timestamp=1000,
        )

        assert len(matches) == 1
        assert matches[0].pattern_name == "single-a"

    def test_multi_event_pattern(self, cep_engine):
        """Test multi-event pattern matching."""
        pattern = Pattern("a-then-b")
        (pattern
            .begin("a")
            .where(lambda e: e.get("type") == "A")
            .next("b")
            .where(lambda e: e.get("type") == "B")
            .one())
        pattern.within(60000)
        cep_engine.register_pattern(pattern)

        # First event
        matches1 = cep_engine.process_event({"type": "A"}, 1000)
        assert len(matches1) == 0

        # Second event completes pattern
        matches2 = cep_engine.process_event({"type": "B"}, 2000)
        assert len(matches2) == 1

    def test_pattern_timeout(self, cep_engine):
        """Test pattern timeout clears buffer."""
        pattern = Pattern("quick")
        pattern.begin("start").where(lambda e: True).one()
        pattern.within(1000)
        cep_engine.register_pattern(pattern)

        # Add event
        cep_engine.process_event({"type": "A"}, 1000)

        # Add event outside window - should clear old events
        cep_engine.process_event({"type": "B"}, 100000)

        # Buffer should only have recent event
        assert len(cep_engine._buffers["quick"]) <= 1


# --- StreamJoiner Tests ---


class TestStreamJoiner:
    """Tests for StreamJoiner class."""

    @pytest.fixture
    def join_config(self):
        """Create join config."""
        return JoinConfig(
            left_key_extractor=lambda x: x.get("user_id"),
            right_key_extractor=lambda x: x.get("user_id"),
            window_ms=10000,
        )

    @pytest.fixture
    def joiner(self, join_config):
        """Create stream joiner."""
        return StreamJoiner(join_config)

    def test_left_event_no_match(self, joiner):
        """Test left event with no matching right."""
        left = {"user_id": "user1", "action": "click"}

        results = joiner.process_left(left, 1000)

        assert len(results) == 0

    def test_right_event_no_match(self, joiner):
        """Test right event with no matching left."""
        right = {"user_id": "user1", "purchase": 100}

        results = joiner.process_right(right, 1000)

        assert len(results) == 0

    def test_join_matching_events(self, joiner):
        """Test joining matching events."""
        left = {"user_id": "user1", "action": "click"}
        right = {"user_id": "user1", "purchase": 100}

        joiner.process_left(left, 1000)
        results = joiner.process_right(right, 2000)

        assert len(results) == 1
        assert results[0] == (left, right)

    def test_join_within_window(self, joiner):
        """Test join only within window."""
        left = {"user_id": "user1", "action": "click"}
        right = {"user_id": "user1", "purchase": 100}

        joiner.process_left(left, 1000)
        # Right event within window
        results = joiner.process_right(right, 5000)

        assert len(results) == 1

    def test_no_join_outside_window(self, joiner):
        """Test no join outside window."""
        left = {"user_id": "user1", "action": "click"}
        right = {"user_id": "user1", "purchase": 100}

        joiner.process_left(left, 1000)
        # Right event outside window
        results = joiner.process_right(right, 100000)

        assert len(results) == 0

    def test_multiple_matches(self, joiner):
        """Test multiple matching events."""
        left1 = {"user_id": "user1", "action": "click1"}
        left2 = {"user_id": "user1", "action": "click2"}
        right = {"user_id": "user1", "purchase": 100}

        joiner.process_left(left1, 1000)
        joiner.process_left(left2, 2000)
        results = joiner.process_right(right, 3000)

        assert len(results) == 2


# --- IntervalJoiner Tests ---


class TestIntervalJoiner:
    """Tests for IntervalJoiner class."""

    @pytest.fixture
    def interval_joiner(self):
        """Create interval joiner."""
        config = JoinConfig(
            left_key_extractor=lambda x: x.get("user_id"),
            right_key_extractor=lambda x: x.get("user_id"),
            window_ms=60000,
        )
        return IntervalJoiner(
            config,
            lower_bound_ms=-10000,  # Right can be up to 10s before left
            upper_bound_ms=10000,   # Right can be up to 10s after left
        )

    def test_join_within_interval(self, interval_joiner):
        """Test join within interval bounds."""
        left = {"user_id": "user1", "action": "click"}
        right = {"user_id": "user1", "ad": "ad1"}

        # Right event is 5 seconds after left
        interval_joiner.process_right(right, 15000)
        results = interval_joiner.process_left(left, 10000)

        assert len(results) == 1

    def test_no_join_outside_interval(self, interval_joiner):
        """Test no join outside interval bounds."""
        left = {"user_id": "user1", "action": "click"}
        right = {"user_id": "user1", "ad": "ad1"}

        # Right event is 20 seconds after left (outside bound)
        interval_joiner.process_right(right, 30000)
        results = interval_joiner.process_left(left, 10000)

        assert len(results) == 0


# --- Integration Tests ---


class TestPatternsIntegration:
    """Integration tests for streaming patterns."""

    def test_session_to_aggregation(self):
        """Test session processing to aggregation flow."""
        processor = SessionProcessor(
            gap_ms=30000,
            user_id_extractor=lambda x: x["user_id"],
            timestamp_extractor=lambda x: x["timestamp"],
        )

        # Simulate user activity
        events = [
            {"user_id": "user1", "timestamp": 1000, "page": "home"},
            {"user_id": "user1", "timestamp": 5000, "page": "products"},
            {"user_id": "user1", "timestamp": 10000, "page": "cart"},
            {"user_id": "user1", "timestamp": 100000, "page": "checkout"},  # New session
        ]

        sessions = []
        for event in events:
            closed = processor.process_event(event)
            if closed:
                sessions.append(closed)

        # Close remaining
        sessions.extend(processor.close_idle_sessions(200000))

        assert len(sessions) == 2
        assert sessions[0].event_count == 3
        assert sessions[1].event_count == 1

    def test_dedup_then_join(self):
        """Test deduplication before joining."""
        # Deduplicator
        dedup = Deduplicator(DeduplicationConfig(
            window_ms=60000,
            id_extractor=lambda x: x["event_id"],
        ))

        # Joiner
        joiner = StreamJoiner(JoinConfig(
            left_key_extractor=lambda x: x["user_id"],
            right_key_extractor=lambda x: x["user_id"],
            window_ms=10000,
        ))

        # Left stream with duplicates
        left_events = [
            {"event_id": "e1", "user_id": "u1", "action": "click"},
            {"event_id": "e1", "user_id": "u1", "action": "click"},  # Duplicate
            {"event_id": "e2", "user_id": "u1", "action": "view"},
        ]

        # Right stream
        right_events = [
            {"event_id": "r1", "user_id": "u1", "purchase": 100},
        ]

        # Process left with dedup
        for i, event in enumerate(left_events):
            deduped = dedup.process(event)
            if deduped:
                joiner.process_left(deduped, i * 1000)

        # Process right
        all_results = []
        for i, event in enumerate(right_events):
            results = joiner.process_right(event, 5000)
            all_results.extend(results)

        # Should only join with deduplicated events (2 unique)
        assert len(all_results) == 2

    def test_cep_alert_pattern(self):
        """Test CEP for alert detection pattern."""
        engine = CEPEngine()

        # Pattern: 3 failed logins within 1 minute
        pattern = Pattern("brute-force")
        (pattern
            .begin("fail1")
            .where(lambda e: e.get("type") == "login_failed")
            .next("fail2")
            .where(lambda e: e.get("type") == "login_failed")
            .next("fail3")
            .where(lambda e: e.get("type") == "login_failed")
            .one())
        pattern.within(60000)
        engine.register_pattern(pattern)

        # Simulate login attempts
        events = [
            ({"type": "login_failed", "user": "user1"}, 1000),
            ({"type": "login_success", "user": "user2"}, 2000),
            ({"type": "login_failed", "user": "user1"}, 3000),
            ({"type": "login_failed", "user": "user1"}, 4000),
        ]

        alerts = []
        for event, timestamp in events:
            matches = engine.process_event(event, timestamp)
            alerts.extend(matches)

        # Should detect the brute force pattern
        assert len(alerts) == 1
        assert alerts[0].pattern_name == "brute-force"
