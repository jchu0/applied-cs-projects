"""Advanced streaming patterns: deduplication, sessionization, CEP, joins."""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

from streaming.state import InMemoryStateBackend, MapState, ValueState
from streaming.windowing import SessionWindowAssigner, Window

T = TypeVar("T")
K = TypeVar("K")


# --- Deduplication ---

@dataclass
class DeduplicationConfig:
    """Configuration for deduplication."""

    window_ms: int  # Time window to remember IDs
    id_extractor: Callable[[Any], str]


class Deduplicator:
    """Deduplicate events within a time window."""

    def __init__(self, config: DeduplicationConfig):
        self.config = config
        self._backend = InMemoryStateBackend()
        self._seen_ids: Dict[str, int] = {}  # id -> timestamp

    def is_duplicate(self, event: Any) -> bool:
        """
        Check if event is a duplicate.

        Args:
            event: Event to check

        Returns:
            True if duplicate, False if new
        """
        event_id = self.config.id_extractor(event)
        current_time = int(time.time() * 1000)

        # Clean old entries
        self._cleanup(current_time)

        # Check if seen
        if event_id in self._seen_ids:
            return True

        # Remember this ID
        self._seen_ids[event_id] = current_time
        return False

    def process(self, event: Any) -> Optional[Any]:
        """
        Process event and return if not duplicate.

        Args:
            event: Event to process

        Returns:
            Event if not duplicate, None if duplicate
        """
        if self.is_duplicate(event):
            return None
        return event

    def _cleanup(self, current_time: int) -> None:
        """Remove expired entries."""
        cutoff = current_time - self.config.window_ms
        self._seen_ids = {
            k: v for k, v in self._seen_ids.items() if v > cutoff
        }


# --- Debouncing ---

@dataclass
class DebouncerConfig:
    """Configuration for debouncing."""

    window_ms: int
    key_extractor: Callable[[Any], str]


class Debouncer:
    """Debounce events - only emit after quiet period."""

    def __init__(self, config: DebouncerConfig):
        self.config = config
        self._pending: Dict[str, tuple] = {}  # key -> (event, timestamp)

    def process(self, event: Any) -> Optional[Any]:
        """
        Process event and update pending state.

        Args:
            event: Event to process

        Returns:
            Previous event if debounce period passed, else None
        """
        key = self.config.key_extractor(event)
        current_time = int(time.time() * 1000)

        # Check if we should emit previous event
        result = None
        if key in self._pending:
            prev_event, prev_time = self._pending[key]
            if current_time - prev_time >= self.config.window_ms:
                result = prev_event

        # Update pending
        self._pending[key] = (event, current_time)

        return result

    def flush(self) -> List[Any]:
        """
        Flush all pending events.

        Returns:
            List of pending events
        """
        results = [event for event, _ in self._pending.values()]
        self._pending.clear()
        return results


# --- Sessionization ---

@dataclass
class Session:
    """Represents a user session."""

    session_id: str
    user_id: str
    start_time: int
    end_time: int
    events: List[Any] = field(default_factory=list)
    event_count: int = 0

    @property
    def duration_ms(self) -> int:
        """Session duration in milliseconds."""
        return self.end_time - self.start_time


class SessionProcessor:
    """Process events into sessions."""

    def __init__(
        self,
        gap_ms: int,
        user_id_extractor: Callable[[Any], str],
        timestamp_extractor: Callable[[Any], int],
    ):
        self.gap_ms = gap_ms
        self.user_id_extractor = user_id_extractor
        self.timestamp_extractor = timestamp_extractor
        self._sessions: Dict[str, Session] = {}

    def process_event(self, event: Any) -> Optional[Session]:
        """
        Process an event and return closed session if any.

        Args:
            event: Event to process

        Returns:
            Closed session if gap exceeded, None otherwise
        """
        user_id = self.user_id_extractor(event)
        timestamp = self.timestamp_extractor(event)

        closed_session = None

        if user_id in self._sessions:
            session = self._sessions[user_id]

            # Check if session should close
            if timestamp - session.end_time > self.gap_ms:
                closed_session = session
                # Start new session
                self._sessions[user_id] = self._create_session(user_id, timestamp, event)
            else:
                # Update existing session
                session.end_time = max(session.end_time, timestamp)
                session.events.append(event)
                session.event_count += 1
        else:
            # Start new session
            self._sessions[user_id] = self._create_session(user_id, timestamp, event)

        return closed_session

    def close_idle_sessions(self, current_time: int) -> List[Session]:
        """
        Close sessions that have been idle.

        Args:
            current_time: Current timestamp

        Returns:
            List of closed sessions
        """
        closed = []
        for user_id in list(self._sessions.keys()):
            session = self._sessions[user_id]
            if current_time - session.end_time > self.gap_ms:
                closed.append(session)
                del self._sessions[user_id]

        return closed

    def _create_session(self, user_id: str, timestamp: int, event: Any) -> Session:
        """Create a new session."""
        import uuid
        return Session(
            session_id=str(uuid.uuid4()),
            user_id=user_id,
            start_time=timestamp,
            end_time=timestamp,
            events=[event],
            event_count=1,
        )


# --- Complex Event Processing (CEP) ---

@dataclass
class PatternMatch:
    """A matched pattern."""

    pattern_name: str
    events: List[Any]
    start_time: int
    end_time: int


class Pattern:
    """Define a pattern for CEP."""

    def __init__(self, name: str):
        self.name = name
        self._conditions: List[tuple] = []  # (name, condition, quantifier)
        self._within_ms: Optional[int] = None

    def begin(self, name: str) -> "PatternCondition":
        """Start pattern with first condition."""
        return PatternCondition(self, name)

    def within(self, ms: int) -> "Pattern":
        """Set time constraint for pattern."""
        self._within_ms = ms
        return self

    def add_condition(
        self,
        name: str,
        condition: Callable[[Any], bool],
        quantifier: str = "one",
    ) -> None:
        """Add a condition to the pattern."""
        self._conditions.append((name, condition, quantifier))


class PatternCondition:
    """Builder for pattern conditions."""

    def __init__(self, pattern: Pattern, name: str):
        self._pattern = pattern
        self._name = name
        self._condition: Optional[Callable[[Any], bool]] = None

    def where(self, condition: Callable[[Any], bool]) -> "PatternCondition":
        """Set condition for current step."""
        self._condition = condition
        return self

    def one(self) -> "Pattern":
        """Match exactly one event."""
        self._pattern.add_condition(self._name, self._condition, "one")
        return self._pattern

    def one_or_more(self) -> "Pattern":
        """Match one or more events."""
        self._pattern.add_condition(self._name, self._condition, "one_or_more")
        return self._pattern

    def optional(self) -> "Pattern":
        """Optionally match event."""
        self._pattern.add_condition(self._name, self._condition, "optional")
        return self._pattern

    def next(self, name: str) -> "PatternCondition":
        """Add next condition (strict contiguity)."""
        self._pattern.add_condition(self._name, self._condition, "one")
        return PatternCondition(self._pattern, name)

    def followed_by(self, name: str) -> "PatternCondition":
        """Add followed-by condition (relaxed contiguity)."""
        self._pattern.add_condition(self._name, self._condition, "one")
        return PatternCondition(self._pattern, name)


class CEPEngine:
    """Complex Event Processing engine."""

    def __init__(self):
        self._patterns: Dict[str, Pattern] = {}
        self._buffers: Dict[str, List[tuple]] = {}  # pattern -> [(event, timestamp)]

    def register_pattern(self, pattern: Pattern) -> None:
        """Register a pattern."""
        self._patterns[pattern.name] = pattern
        self._buffers[pattern.name] = []

    def process_event(
        self,
        event: Any,
        timestamp: int,
    ) -> List[PatternMatch]:
        """
        Process event and return any pattern matches.

        Args:
            event: Event to process
            timestamp: Event timestamp

        Returns:
            List of pattern matches
        """
        matches = []

        for pattern_name, pattern in self._patterns.items():
            buffer = self._buffers[pattern_name]

            # Add event to buffer
            buffer.append((event, timestamp))

            # Clean old events
            if pattern._within_ms:
                cutoff = timestamp - pattern._within_ms
                buffer[:] = [(e, t) for e, t in buffer if t >= cutoff]

            # Try to match pattern
            match = self._try_match(pattern, buffer)
            if match:
                matches.append(match)
                # Clear matched events
                self._buffers[pattern_name] = []

        return matches

    def _try_match(
        self,
        pattern: Pattern,
        buffer: List[tuple],
    ) -> Optional[PatternMatch]:
        """Try to match pattern against buffer."""
        if not pattern._conditions:
            return None

        matched_events = []
        condition_idx = 0
        events = [e for e, _ in buffer]

        for event in events:
            if condition_idx >= len(pattern._conditions):
                break

            name, condition, quantifier = pattern._conditions[condition_idx]

            if condition and condition(event):
                matched_events.append(event)

                if quantifier == "one":
                    condition_idx += 1
                # For one_or_more, stay on same condition until no match

        # Check if all conditions matched
        if condition_idx >= len(pattern._conditions):
            timestamps = [t for _, t in buffer if _ in matched_events]
            return PatternMatch(
                pattern_name=pattern.name,
                events=matched_events,
                start_time=min(timestamps) if timestamps else 0,
                end_time=max(timestamps) if timestamps else 0,
            )

        return None


# --- Stream Joins ---

@dataclass
class JoinConfig:
    """Configuration for stream joins."""

    left_key_extractor: Callable[[Any], str]
    right_key_extractor: Callable[[Any], str]
    window_ms: int
    join_type: str = "inner"  # inner, left, right, full


class StreamJoiner:
    """Join two streams within a time window."""

    def __init__(self, config: JoinConfig):
        self.config = config
        self._left_buffer: Dict[str, List[tuple]] = {}  # key -> [(event, timestamp)]
        self._right_buffer: Dict[str, List[tuple]] = {}

    def process_left(
        self,
        event: Any,
        timestamp: int,
    ) -> List[tuple]:
        """
        Process event from left stream.

        Args:
            event: Left stream event
            timestamp: Event timestamp

        Returns:
            List of joined pairs (left, right)
        """
        key = self.config.left_key_extractor(event)

        # Add to buffer
        if key not in self._left_buffer:
            self._left_buffer[key] = []
        self._left_buffer[key].append((event, timestamp))

        # Clean old events
        self._cleanup_buffer(self._left_buffer, timestamp)
        self._cleanup_buffer(self._right_buffer, timestamp)

        # Find matches in right buffer
        results = []
        if key in self._right_buffer:
            for right_event, right_ts in self._right_buffer[key]:
                if abs(timestamp - right_ts) <= self.config.window_ms:
                    results.append((event, right_event))

        return results

    def process_right(
        self,
        event: Any,
        timestamp: int,
    ) -> List[tuple]:
        """
        Process event from right stream.

        Args:
            event: Right stream event
            timestamp: Event timestamp

        Returns:
            List of joined pairs (left, right)
        """
        key = self.config.right_key_extractor(event)

        # Add to buffer
        if key not in self._right_buffer:
            self._right_buffer[key] = []
        self._right_buffer[key].append((event, timestamp))

        # Clean old events
        self._cleanup_buffer(self._left_buffer, timestamp)
        self._cleanup_buffer(self._right_buffer, timestamp)

        # Find matches in left buffer
        results = []
        if key in self._left_buffer:
            for left_event, left_ts in self._left_buffer[key]:
                if abs(timestamp - left_ts) <= self.config.window_ms:
                    results.append((left_event, event))

        return results

    def _cleanup_buffer(
        self,
        buffer: Dict[str, List[tuple]],
        current_time: int,
    ) -> None:
        """Remove expired events from buffer."""
        cutoff = current_time - self.config.window_ms * 2

        for key in list(buffer.keys()):
            buffer[key] = [(e, t) for e, t in buffer[key] if t >= cutoff]
            if not buffer[key]:
                del buffer[key]


class IntervalJoiner(StreamJoiner):
    """Join streams with interval bounds."""

    def __init__(
        self,
        config: JoinConfig,
        lower_bound_ms: int,
        upper_bound_ms: int,
    ):
        super().__init__(config)
        self.lower_bound_ms = lower_bound_ms
        self.upper_bound_ms = upper_bound_ms

    def process_left(
        self,
        event: Any,
        timestamp: int,
    ) -> List[tuple]:
        """Process with interval bounds."""
        key = self.config.left_key_extractor(event)

        # Add to buffer
        if key not in self._left_buffer:
            self._left_buffer[key] = []
        self._left_buffer[key].append((event, timestamp))

        # Find matches with interval constraint
        results = []
        if key in self._right_buffer:
            for right_event, right_ts in self._right_buffer[key]:
                diff = right_ts - timestamp
                if self.lower_bound_ms <= diff <= self.upper_bound_ms:
                    results.append((event, right_event))

        return results
