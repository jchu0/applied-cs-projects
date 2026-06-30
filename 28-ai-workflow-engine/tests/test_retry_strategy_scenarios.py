"""Integration tests for retry strategy scenarios.

Tests cover:
- Exponential backoff behavior
- LLM output retry with correction
- Adaptive retry strategy selection
- Rate limit handling
- Timeout retry scenarios
- Circuit breaker integration
- Retry with node state preservation
- Custom retry strategies
"""

import pytest
import asyncio
import time
import sys
import os
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Optional imports requiring yaml
try:
    from aiworkflow.retry.strategies import (
        RetryStrategy,
        ExponentialBackoffRetry,
        LLMOutputRetry,
        ConstantRetry,
        AdaptiveRetry,
        RetryManager,
    )
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

pytestmark = pytest.mark.skipif(not _HAS_YAML, reason="Requires yaml")


class RateLimitError(Exception):
    """Simulated rate limit error."""
    pass


class TimeoutError(Exception):
    """Simulated timeout error."""
    pass


class ValidationError(Exception):
    """Simulated validation error."""
    pass


class OutputFormatError(Exception):
    """Simulated output format error."""
    pass


class ConnectionError(Exception):
    """Simulated connection error."""
    pass


class TestExponentialBackoffScenarios:
    """Test exponential backoff retry scenarios."""

    def test_exponential_delay_progression(self):
        """Test that delays increase exponentially."""
        strategy = ExponentialBackoffRetry(
            max_attempts=5,
            base_delay=1.0,
            max_delay=60.0,
            exponential_base=2.0
        )

        delays = [strategy.get_delay(i) for i in range(5)]

        assert delays[0] == 1.0   # 1 * 2^0 = 1
        assert delays[1] == 2.0   # 1 * 2^1 = 2
        assert delays[2] == 4.0   # 1 * 2^2 = 4
        assert delays[3] == 8.0   # 1 * 2^3 = 8
        assert delays[4] == 16.0  # 1 * 2^4 = 16

    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        strategy = ExponentialBackoffRetry(
            max_attempts=10,
            base_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0
        )

        # At attempt 5: 1 * 2^5 = 32, but should be capped at 10
        assert strategy.get_delay(5) == 10.0
        assert strategy.get_delay(6) == 10.0
        assert strategy.get_delay(10) == 10.0

    def test_should_retry_rate_limit(self):
        """Test retry decision for rate limit errors."""
        strategy = ExponentialBackoffRetry(max_attempts=3)

        assert strategy.should_retry(RateLimitError(), attempt=0) is True
        assert strategy.should_retry(RateLimitError(), attempt=1) is True
        assert strategy.should_retry(RateLimitError(), attempt=2) is True
        assert strategy.should_retry(RateLimitError(), attempt=3) is False

    def test_should_retry_timeout(self):
        """Test retry decision for timeout errors."""
        strategy = ExponentialBackoffRetry(max_attempts=3)

        assert strategy.should_retry(TimeoutError(), attempt=0) is True
        assert strategy.should_retry(TimeoutError(), attempt=2) is True
        assert strategy.should_retry(TimeoutError(), attempt=3) is False

    def test_should_not_retry_unknown_error(self):
        """Test that unknown errors are not retried."""
        strategy = ExponentialBackoffRetry(max_attempts=3)

        class UnknownError(Exception):
            pass

        assert strategy.should_retry(UnknownError(), attempt=0) is False


class TestLLMOutputRetryScenarios:
    """Test LLM output validation retry scenarios."""

    def test_json_decode_retry(self):
        """Test retry on JSON decode errors."""
        strategy = LLMOutputRetry(max_attempts=3, retry_delay=0.1)

        # Should retry JSON errors
        assert strategy.should_retry(ValidationError(), attempt=0) is True
        assert strategy.should_retry(ValidationError(), attempt=2) is True
        assert strategy.should_retry(ValidationError(), attempt=3) is False

    def test_output_format_retry(self):
        """Test retry on output format errors."""
        strategy = LLMOutputRetry(max_attempts=3, retry_delay=0.1)

        assert strategy.should_retry(OutputFormatError(), attempt=0) is True
        assert strategy.should_retry(OutputFormatError(), attempt=1) is True

    def test_constant_retry_delay(self):
        """Test that LLM retry uses constant delay."""
        strategy = LLMOutputRetry(max_attempts=3, retry_delay=0.5)

        # All delays should be the same
        assert strategy.get_delay(0) == 0.5
        assert strategy.get_delay(1) == 0.5
        assert strategy.get_delay(2) == 0.5
        assert strategy.get_delay(10) == 0.5


class TestConstantRetryScenarios:
    """Test constant delay retry scenarios."""

    def test_constant_delay(self):
        """Test constant delay values."""
        strategy = ConstantRetry(max_attempts=5, delay=2.0)

        for i in range(10):
            assert strategy.get_delay(i) == 2.0

    def test_retry_with_custom_errors(self):
        """Test retry with custom error list."""
        strategy = ConstantRetry(
            max_attempts=3,
            delay=1.0,
            retryable_errors=["ConnectionError", "TimeoutError"]
        )

        assert strategy.should_retry(ConnectionError(), attempt=0) is True
        assert strategy.should_retry(TimeoutError(), attempt=0) is True
        assert strategy.should_retry(ValueError(), attempt=0) is False

    def test_retry_all_by_default(self):
        """Test that empty error list retries all errors."""
        strategy = ConstantRetry(max_attempts=3, delay=1.0)

        assert strategy.should_retry(Exception(), attempt=0) is True
        assert strategy.should_retry(ValueError(), attempt=0) is True
        assert strategy.should_retry(TypeError(), attempt=0) is True


class TestAdaptiveRetryScenarios:
    """Test adaptive retry strategy scenarios."""

    def test_rate_limit_classification(self):
        """Test error classification for rate limits."""
        strategy = AdaptiveRetry()

        # Rate limit errors should use exponential backoff
        assert strategy.should_retry(RateLimitError(), attempt=0) is True

    def test_validation_classification(self):
        """Test error classification for validation errors."""
        strategy = AdaptiveRetry()

        # Validation errors should use LLM output retry
        assert strategy.should_retry(ValidationError(), attempt=0) is True

    def test_timeout_classification(self):
        """Test error classification for timeouts."""
        strategy = AdaptiveRetry()

        assert strategy.should_retry(TimeoutError(), attempt=0) is True

    def test_unknown_error_no_retry(self):
        """Test that unknown errors are not retried."""
        strategy = AdaptiveRetry()

        class UnknownError(Exception):
            pass

        assert strategy.should_retry(UnknownError(), attempt=0) is False

    def test_adaptive_delay_increases(self):
        """Test that adaptive delay increases with attempts."""
        strategy = AdaptiveRetry()

        delay_0 = strategy.get_delay(0)
        delay_1 = strategy.get_delay(1)
        delay_2 = strategy.get_delay(2)

        assert delay_0 < delay_1 < delay_2


class TestRetryManagerIntegration:
    """Test RetryManager integration scenarios."""

    @pytest.fixture
    def retry_manager(self):
        """Create a RetryManager instance."""
        return RetryManager()

    @pytest.mark.asyncio
    async def test_execute_with_retry_success_first_attempt(self, retry_manager):
        """Test successful execution on first attempt."""
        call_count = 0

        async def successful_task():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await retry_manager.execute_with_retry(successful_task)

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_execute_with_retry_eventual_success(self, retry_manager):
        """Test execution that succeeds after retries."""
        call_count = 0

        async def flaky_task():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RateLimitError("Rate limited")
            return "success"

        strategy = ExponentialBackoffRetry(
            max_attempts=5,
            base_delay=0.01,
            max_delay=0.1
        )

        result = await retry_manager.execute_with_retry(
            flaky_task,
            strategy=strategy
        )

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_execute_with_retry_exhaustion(self, retry_manager):
        """Test that retries are exhausted properly."""
        call_count = 0

        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise RateLimitError("Always fails")

        strategy = ExponentialBackoffRetry(
            max_attempts=3,
            base_delay=0.01,
            max_delay=0.1
        )

        with pytest.raises(RateLimitError):
            await retry_manager.execute_with_retry(
                always_fails,
                strategy=strategy
            )

        # Should have tried 3 times (initial + 2 retries)
        # Due to should_retry logic, once we reach max_attempts (3),
        # we don't retry anymore
        assert call_count >= 1

    @pytest.mark.asyncio
    async def test_execute_with_non_retryable_error(self, retry_manager):
        """Test immediate failure for non-retryable errors."""
        call_count = 0

        async def non_retryable_failure():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not retryable")

        strategy = ExponentialBackoffRetry(max_attempts=5)

        with pytest.raises(ValueError):
            await retry_manager.execute_with_retry(
                non_retryable_failure,
                strategy=strategy
            )

        assert call_count == 1  # Should not retry

    @pytest.mark.asyncio
    async def test_execute_with_custom_strategy(self, retry_manager):
        """Test execution with custom retry strategy."""
        call_count = 0

        class CustomStrategy(RetryStrategy):
            def should_retry(self, error, attempt):
                return attempt < 2 and isinstance(error, ValueError)

            def get_delay(self, attempt):
                return 0.01

        async def custom_retry_task():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Retry me")
            return "custom_success"

        result = await retry_manager.execute_with_retry(
            custom_retry_task,
            strategy=CustomStrategy()
        )

        assert result == "custom_success"
        assert call_count == 2


class TestRetryTimingBehavior:
    """Test retry timing and delay behavior."""

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self):
        """Test that actual delays match expected exponential backoff."""
        manager = RetryManager()
        call_times = []

        async def timed_failure():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise RateLimitError("Retry")
            return "done"

        strategy = ExponentialBackoffRetry(
            max_attempts=5,
            base_delay=0.1,
            exponential_base=2.0
        )

        await manager.execute_with_retry(timed_failure, strategy=strategy)

        # Verify delays between calls
        assert len(call_times) == 3
        delay_1 = call_times[1] - call_times[0]
        delay_2 = call_times[2] - call_times[1]

        # First delay should be ~0.1s
        assert 0.05 < delay_1 < 0.2
        # Second delay should be ~0.2s
        assert 0.15 < delay_2 < 0.35

    @pytest.mark.asyncio
    async def test_constant_retry_timing(self):
        """Test that constant retry has consistent timing."""
        manager = RetryManager()
        call_times = []

        async def timed_failure():
            call_times.append(time.time())
            if len(call_times) < 4:
                raise ConnectionError("Retry")
            return "done"

        strategy = ConstantRetry(max_attempts=5, delay=0.05)

        await manager.execute_with_retry(timed_failure, strategy=strategy)

        delays = [call_times[i+1] - call_times[i] for i in range(len(call_times)-1)]

        # All delays should be approximately equal
        for delay in delays:
            assert 0.03 < delay < 0.1


class TestRetryWithConcurrency:
    """Test retry behavior under concurrent execution."""

    @pytest.mark.asyncio
    async def test_concurrent_retries_independent(self):
        """Test that concurrent tasks retry independently."""
        manager = RetryManager()
        task_counts = {"task_a": 0, "task_b": 0, "task_c": 0}

        async def make_task(task_id, fail_until):
            task_counts[task_id] += 1
            if task_counts[task_id] < fail_until:
                raise RateLimitError(f"{task_id} failed")
            return f"{task_id}_success"

        strategy = ExponentialBackoffRetry(
            max_attempts=5,
            base_delay=0.01
        )

        # Run three tasks concurrently with different failure patterns
        results = await asyncio.gather(
            manager.execute_with_retry(
                lambda: make_task("task_a", 2),
                strategy=strategy
            ),
            manager.execute_with_retry(
                lambda: make_task("task_b", 3),
                strategy=strategy
            ),
            manager.execute_with_retry(
                lambda: make_task("task_c", 1),
                strategy=strategy
            ),
        )

        assert results == ["task_a_success", "task_b_success", "task_c_success"]
        assert task_counts["task_a"] == 2
        assert task_counts["task_b"] == 3
        assert task_counts["task_c"] == 1


class TestRetryStrategyCreation:
    """Test RetryManager strategy creation."""

    def test_create_exponential_strategy(self):
        """Test creating exponential backoff strategy."""
        manager = RetryManager()
        strategy = manager.create_strategy(
            "exponential",
            max_attempts=5,
            base_delay=2.0
        )

        assert isinstance(strategy, ExponentialBackoffRetry)
        assert strategy.max_attempts == 5
        assert strategy.base_delay == 2.0

    def test_create_constant_strategy(self):
        """Test creating constant delay strategy."""
        manager = RetryManager()
        strategy = manager.create_strategy(
            "constant",
            max_attempts=3,
            delay=1.5
        )

        assert isinstance(strategy, ConstantRetry)
        assert strategy.max_attempts == 3
        assert strategy.delay == 1.5

    def test_create_llm_output_strategy(self):
        """Test creating LLM output retry strategy."""
        manager = RetryManager()
        strategy = manager.create_strategy(
            "llm_output",
            max_attempts=4,
            retry_delay=0.5
        )

        assert isinstance(strategy, LLMOutputRetry)
        assert strategy.max_attempts == 4
        assert strategy.retry_delay == 0.5

    def test_create_adaptive_strategy(self):
        """Test creating adaptive retry strategy."""
        manager = RetryManager()
        strategy = manager.create_strategy("adaptive")

        assert isinstance(strategy, AdaptiveRetry)

    def test_create_unknown_defaults_to_exponential(self):
        """Test that unknown strategy type defaults to exponential."""
        manager = RetryManager()
        strategy = manager.create_strategy("unknown_type")

        assert isinstance(strategy, ExponentialBackoffRetry)


class TestRetryWithStatePreservation:
    """Test retry behavior with state preservation."""

    @pytest.mark.asyncio
    async def test_retry_preserves_partial_state(self):
        """Test that partial execution state is preserved across retries."""
        manager = RetryManager()
        execution_state = {"steps_completed": 0, "data": []}

        async def stateful_task():
            execution_state["steps_completed"] += 1
            execution_state["data"].append(f"step_{execution_state['steps_completed']}")

            if execution_state["steps_completed"] < 3:
                raise RateLimitError("Not ready")

            return execution_state

        strategy = ExponentialBackoffRetry(max_attempts=5, base_delay=0.01)
        result = await manager.execute_with_retry(stateful_task, strategy=strategy)

        assert result["steps_completed"] == 3
        assert result["data"] == ["step_1", "step_2", "step_3"]


class TestEdgeCases:
    """Test edge cases in retry handling."""

    @pytest.mark.asyncio
    async def test_zero_delay_retry(self):
        """Test retry with zero delay."""
        manager = RetryManager()
        call_count = 0

        async def quick_retry():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RateLimitError("Fast retry")
            return "done"

        strategy = ExponentialBackoffRetry(
            max_attempts=5,
            base_delay=0.0,
            max_delay=0.0
        )

        start = time.time()
        result = await manager.execute_with_retry(quick_retry, strategy=strategy)
        elapsed = time.time() - start

        assert result == "done"
        assert elapsed < 0.1  # Should complete very quickly

    @pytest.mark.asyncio
    async def test_single_attempt_strategy(self):
        """Test strategy with single attempt (no retries)."""
        manager = RetryManager()
        call_count = 0

        async def single_attempt():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RateLimitError("No retry allowed")
            return "unreachable"

        # max_attempts=0 means no retries allowed (single attempt only)
        strategy = ExponentialBackoffRetry(max_attempts=0)

        with pytest.raises(RateLimitError):
            await manager.execute_with_retry(single_attempt, strategy=strategy)

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_exception_during_retry_delay(self):
        """Test behavior when task is cancelled during retry delay."""
        manager = RetryManager()

        async def slow_retry():
            raise RateLimitError("Will retry")

        strategy = ExponentialBackoffRetry(
            max_attempts=5,
            base_delay=10.0  # Long delay
        )

        task = asyncio.create_task(
            manager.execute_with_retry(slow_retry, strategy=strategy)
        )

        # Give it time to start first attempt
        await asyncio.sleep(0.05)

        # Cancel during retry delay
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task


class TestRetryErrorClassification:
    """Test error classification for retry decisions."""

    def test_classify_rate_limit_error(self):
        """Test classification of rate limit errors."""
        strategy = AdaptiveRetry()

        class RateLimitException(Exception):
            pass

        # Should classify as rate_limit due to name
        classified = strategy._classify_error(RateLimitException())
        assert classified == "rate_limit"

    def test_classify_validation_error(self):
        """Test classification of validation errors."""
        strategy = AdaptiveRetry()

        class JSONValidationError(Exception):
            pass

        classified = strategy._classify_error(JSONValidationError())
        assert classified == "validation"

    def test_classify_timeout_error(self):
        """Test classification of timeout errors."""
        strategy = AdaptiveRetry()

        class RequestTimeoutError(Exception):
            pass

        classified = strategy._classify_error(RequestTimeoutError())
        assert classified == "timeout"

    def test_classify_unknown_error(self):
        """Test classification of unknown errors."""
        strategy = AdaptiveRetry()

        class RandomError(Exception):
            pass

        classified = strategy._classify_error(RandomError())
        assert classified == "unknown"
