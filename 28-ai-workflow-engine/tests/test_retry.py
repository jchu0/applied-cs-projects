"""Unit tests for retry mechanisms."""

import pytest
import asyncio
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import time

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Optional imports requiring yaml
try:
    from aiworkflow.retry import (
        RetryManager,
        RetryStrategy,
        ExponentialBackoffRetry,
        LinearBackoffRetry,
        FixedDelayRetry,
        RetryConfig,
        RetryableError,
        NonRetryableError
    )
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

pytestmark = pytest.mark.skipif(not _HAS_YAML, reason="Requires yaml")


class TestRetryStrategy:
    """Test suite for retry strategies."""

    def test_exponential_backoff_delays(self):
        """Test exponential backoff delay calculation."""
        strategy = ExponentialBackoffRetry(
            base_delay=1.0,
            max_delay=60.0,
            exponential_factor=2.0
        )

        # Test delay progression
        assert strategy.get_delay(0) == 1.0  # 1 * 2^0
        assert strategy.get_delay(1) == 2.0  # 1 * 2^1
        assert strategy.get_delay(2) == 4.0  # 1 * 2^2
        assert strategy.get_delay(3) == 8.0  # 1 * 2^3

        # Test max delay cap
        assert strategy.get_delay(10) == 60.0  # Should be capped at max_delay

    def test_exponential_backoff_with_jitter(self):
        """Test exponential backoff with jitter."""
        strategy = ExponentialBackoffRetry(
            base_delay=1.0,
            max_delay=60.0,
            jitter=True
        )

        # With jitter, delays should vary
        delays = [strategy.get_delay(2) for _ in range(10)]
        assert len(set(delays)) > 1  # Should have different values
        assert all(0 <= d <= 4.0 for d in delays)  # Should be within expected range

    def test_linear_backoff_delays(self):
        """Test linear backoff delay calculation."""
        strategy = LinearBackoffRetry(
            base_delay=2.0,
            increment=3.0,
            max_delay=20.0
        )

        # Test delay progression
        assert strategy.get_delay(0) == 2.0  # base_delay
        assert strategy.get_delay(1) == 5.0  # 2 + 3
        assert strategy.get_delay(2) == 8.0  # 2 + 2*3
        assert strategy.get_delay(3) == 11.0  # 2 + 3*3

        # Test max delay cap
        assert strategy.get_delay(10) == 20.0  # Should be capped

    def test_fixed_delay_strategy(self):
        """Test fixed delay strategy."""
        strategy = FixedDelayRetry(delay=5.0)

        # Delay should always be the same
        assert strategy.get_delay(0) == 5.0
        assert strategy.get_delay(1) == 5.0
        assert strategy.get_delay(100) == 5.0

    def test_should_retry_logic(self):
        """Test retry decision logic."""
        strategy = ExponentialBackoffRetry(max_retries=3)

        # Should retry for retryable errors within limit
        assert strategy.should_retry(
            RetryableError("Temporary failure"),
            attempt=0
        ) is True

        assert strategy.should_retry(
            RetryableError("Temporary failure"),
            attempt=2
        ) is True

        # Should not retry after max attempts
        assert strategy.should_retry(
            RetryableError("Temporary failure"),
            attempt=3
        ) is False

        # Should not retry non-retryable errors
        assert strategy.should_retry(
            NonRetryableError("Permanent failure"),
            attempt=0
        ) is False


class TestRetryManager:
    """Test suite for RetryManager."""

    @pytest.fixture
    def retry_manager(self):
        """Create a RetryManager instance."""
        return RetryManager()

    def test_register_strategy(self, retry_manager):
        """Test registering retry strategies."""
        strategy = ExponentialBackoffRetry()
        retry_manager.register_strategy("custom", strategy)

        assert retry_manager.get_strategy("custom") == strategy

    def test_default_strategy(self, retry_manager):
        """Test default retry strategy."""
        default = retry_manager.get_strategy("default")
        assert isinstance(default, ExponentialBackoffRetry)

    @pytest.mark.asyncio
    async def test_execute_with_retry_success(self, retry_manager):
        """Test successful execution with retry."""
        attempts = []

        async def task():
            attempts.append(len(attempts))
            if len(attempts) < 3:
                raise RetryableError("Temporary failure")
            return "success"

        result = await retry_manager.execute_with_retry(
            task,
            strategy_name="default"
        )

        assert result == "success"
        assert len(attempts) == 3

    @pytest.mark.asyncio
    async def test_execute_with_retry_exhausted(self, retry_manager):
        """Test retry exhaustion."""
        attempts = []

        async def task():
            attempts.append(len(attempts))
            raise RetryableError("Always fails")

        strategy = FixedDelayRetry(delay=0.01, max_retries=3)
        retry_manager.register_strategy("quick_retry", strategy)

        with pytest.raises(RetryableError):
            await retry_manager.execute_with_retry(
                task,
                strategy_name="quick_retry"
            )

        assert len(attempts) == 4  # Initial + 3 retries

    @pytest.mark.asyncio
    async def test_execute_with_non_retryable_error(self, retry_manager):
        """Test non-retryable error handling."""
        attempts = []

        async def task():
            attempts.append(len(attempts))
            raise NonRetryableError("Permanent failure")

        with pytest.raises(NonRetryableError):
            await retry_manager.execute_with_retry(
                task,
                strategy_name="default"
            )

        assert len(attempts) == 1  # Should not retry

    @pytest.mark.asyncio
    async def test_retry_with_timeout(self, retry_manager):
        """Test retry with timeout."""
        async def slow_task():
            await asyncio.sleep(10)
            return "success"

        strategy = FixedDelayRetry(delay=0.01, max_retries=3)
        retry_manager.register_strategy("timeout_test", strategy)

        with pytest.raises(asyncio.TimeoutError):
            await retry_manager.execute_with_retry(
                slow_task,
                strategy_name="timeout_test",
                timeout=0.1
            )

    @pytest.mark.asyncio
    async def test_retry_with_circuit_breaker(self, retry_manager):
        """Test retry with circuit breaker pattern."""
        retry_manager.enable_circuit_breaker(
            failure_threshold=3,
            recovery_timeout=0.1
        )

        failures = 0

        async def failing_task():
            nonlocal failures
            failures += 1
            raise RetryableError("Service unavailable")

        # First set of failures should trip the breaker
        for _ in range(3):
            with pytest.raises(RetryableError):
                await retry_manager.execute_with_retry(
                    failing_task,
                    strategy_name="default"
                )

        # Circuit should be open now
        assert retry_manager.circuit_breaker.is_open

        # Immediate retry should fail fast
        start_time = time.time()
        with pytest.raises(Exception, match="Circuit breaker is open"):
            await retry_manager.execute_with_retry(
                failing_task,
                strategy_name="default"
            )
        elapsed = time.time() - start_time
        assert elapsed < 0.05  # Should fail immediately

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Circuit should be half-open, allowing retry
        async def success_task():
            return "success"

        result = await retry_manager.execute_with_retry(
            success_task,
            strategy_name="default"
        )
        assert result == "success"

        # Circuit should be closed again
        assert not retry_manager.circuit_breaker.is_open


class TestRetryConfig:
    """Test suite for RetryConfig."""

    def test_config_initialization(self):
        """Test retry configuration initialization."""
        config = RetryConfig(
            max_retries=5,
            base_delay=2.0,
            max_delay=120.0,
            exponential_factor=3.0,
            jitter=True,
            retryable_exceptions=[ValueError, KeyError],
            non_retryable_exceptions=[TypeError]
        )

        assert config.max_retries == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 120.0
        assert config.exponential_factor == 3.0
        assert config.jitter is True
        assert ValueError in config.retryable_exceptions
        assert TypeError in config.non_retryable_exceptions

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "max_retries": 3,
            "base_delay": 1.0,
            "strategy": "exponential"
        }

        config = RetryConfig.from_dict(config_dict)

        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.strategy == "exponential"

    def test_config_validation(self):
        """Test config validation."""
        # Invalid max_retries
        with pytest.raises(ValueError, match="max_retries must be positive"):
            RetryConfig(max_retries=-1)

        # Invalid base_delay
        with pytest.raises(ValueError, match="base_delay must be positive"):
            RetryConfig(base_delay=-1.0)

        # Invalid exponential_factor
        with pytest.raises(ValueError, match="exponential_factor must be >= 1"):
            RetryConfig(exponential_factor=0.5)


class TestRetryableError:
    """Test suite for retryable error types."""

    def test_retryable_error_properties(self):
        """Test RetryableError properties."""
        error = RetryableError(
            "Temporary failure",
            retry_after=5.0,
            attempt=2
        )

        assert str(error) == "Temporary failure"
        assert error.retry_after == 5.0
        assert error.attempt == 2
        assert error.is_retryable is True

    def test_non_retryable_error_properties(self):
        """Test NonRetryableError properties."""
        error = NonRetryableError(
            "Permanent failure",
            reason="Invalid configuration"
        )

        assert str(error) == "Permanent failure"
        assert error.reason == "Invalid configuration"
        assert error.is_retryable is False


class TestRetryMetrics:
    """Test suite for retry metrics tracking."""

    def test_metrics_collection(self):
        """Test retry metrics collection."""
        retry_manager = RetryManager(collect_metrics=True)

        # Simulate some retry attempts
        retry_manager.record_attempt("task1", success=False, duration=1.5)
        retry_manager.record_attempt("task1", success=False, duration=2.0)
        retry_manager.record_attempt("task1", success=True, duration=1.0)

        retry_manager.record_attempt("task2", success=True, duration=0.5)

        metrics = retry_manager.get_metrics()

        assert metrics["task1"]["attempts"] == 3
        assert metrics["task1"]["successes"] == 1
        assert metrics["task1"]["failures"] == 2
        assert metrics["task1"]["avg_duration"] == 1.5

        assert metrics["task2"]["attempts"] == 1
        assert metrics["task2"]["successes"] == 1

    def test_metrics_reset(self):
        """Test resetting retry metrics."""
        retry_manager = RetryManager(collect_metrics=True)

        retry_manager.record_attempt("task1", success=True, duration=1.0)
        assert len(retry_manager.get_metrics()) > 0

        retry_manager.reset_metrics()
        assert len(retry_manager.get_metrics()) == 0