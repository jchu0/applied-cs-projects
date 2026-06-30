"""Retry strategies for AI workflow execution."""

from abc import ABC, abstractmethod
from typing import Callable, Any
import asyncio
import logging
import random
import time

logger = logging.getLogger(__name__)


class RetryableError(Exception):
    """Error that indicates the operation should be retried."""

    def __init__(self, message: str = "", retry_after: float = None, attempt: int = 0):
        super().__init__(message)
        self.retry_after = retry_after
        self.attempt = attempt
        self.is_retryable = True


class NonRetryableError(Exception):
    """Error that indicates the operation should NOT be retried."""

    def __init__(self, message: str = "", reason: str = None):
        super().__init__(message)
        self.reason = reason
        self.is_retryable = False


class RetryStrategy(ABC):
    """Base class for retry strategies."""

    @abstractmethod
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if should retry.

        Args:
            error: The exception that occurred
            attempt: Current attempt number (0-indexed)

        Returns:
            True if should retry
        """
        pass

    @abstractmethod
    def get_delay(self, attempt: int) -> float:
        """Get delay before next retry.

        Args:
            attempt: Current attempt number

        Returns:
            Delay in seconds
        """
        pass


class ExponentialBackoffRetry(RetryStrategy):
    """Exponential backoff retry strategy."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        exponential_factor: float = None,
        max_retries: int = None,
        jitter: bool = False,
    ):
        self.max_attempts = max_retries if max_retries is not None else max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_factor if exponential_factor is not None else exponential_base
        self.jitter = jitter

    def should_retry(self, error: Exception, attempt: int) -> bool:
        if attempt >= self.max_attempts:
            return False

        # Check explicit retryable/non-retryable markers
        if getattr(error, 'is_retryable', None) is True:
            return True
        if getattr(error, 'is_retryable', None) is False:
            return False

        # Retry on rate limits and timeouts
        error_type = type(error).__name__
        return error_type in [
            "RateLimitError",
            "TimeoutError",
            "ConnectionError",
            "APIError",
        ]

    def get_delay(self, attempt: int) -> float:
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        if self.jitter:
            delay = random.uniform(0, delay)
        return delay


class LinearBackoffRetry(RetryStrategy):
    """Linear backoff retry strategy."""

    def __init__(
        self,
        base_delay: float = 1.0,
        increment: float = 1.0,
        max_delay: float = 60.0,
        max_retries: int = 3,
    ):
        self.base_delay = base_delay
        self.increment = increment
        self.max_delay = max_delay
        self.max_retries = max_retries

    def should_retry(self, error: Exception, attempt: int) -> bool:
        if attempt >= self.max_retries:
            return False
        if getattr(error, 'is_retryable', None) is False:
            return False
        return True

    def get_delay(self, attempt: int) -> float:
        delay = self.base_delay + attempt * self.increment
        return min(delay, self.max_delay)


class FixedDelayRetry(RetryStrategy):
    """Fixed delay retry strategy."""

    def __init__(self, delay: float = 1.0, max_retries: int = 3):
        self.delay = delay
        self.max_retries = max_retries

    def should_retry(self, error: Exception, attempt: int) -> bool:
        if attempt >= self.max_retries:
            return False
        if getattr(error, 'is_retryable', None) is False:
            return False
        return True

    def get_delay(self, attempt: int) -> float:
        return self.delay


class LLMOutputRetry(RetryStrategy):
    """Retry strategy for LLM output validation failures."""

    def __init__(
        self,
        max_attempts: int = 3,
        retry_delay: float = 0.5
    ):
        self.max_attempts = max_attempts
        self.retry_delay = retry_delay

    def should_retry(self, error: Exception, attempt: int) -> bool:
        if attempt >= self.max_attempts:
            return False

        # Retry on validation/format errors
        error_type = type(error).__name__
        return error_type in [
            "JSONDecodeError",
            "ValidationError",
            "OutputFormatError",
            "KeyError"
        ]

    def get_delay(self, attempt: int) -> float:
        return self.retry_delay


class ConstantRetry(RetryStrategy):
    """Constant delay retry strategy."""

    def __init__(
        self,
        max_attempts: int = 3,
        delay: float = 1.0,
        retryable_errors: list = None
    ):
        self.max_attempts = max_attempts
        self.delay = delay
        self.retryable_errors = retryable_errors or []

    def should_retry(self, error: Exception, attempt: int) -> bool:
        if attempt >= self.max_attempts:
            return False

        if self.retryable_errors:
            return type(error).__name__ in self.retryable_errors

        return True

    def get_delay(self, attempt: int) -> float:
        return self.delay


class AdaptiveRetry(RetryStrategy):
    """Adapts retry strategy based on error patterns."""

    def __init__(self):
        self.strategies = {
            "rate_limit": ExponentialBackoffRetry(max_attempts=5),
            "validation": LLMOutputRetry(max_attempts=3),
            "timeout": ExponentialBackoffRetry(max_attempts=3, base_delay=2.0)
        }

    def should_retry(self, error: Exception, attempt: int) -> bool:
        error_type = self._classify_error(error)
        strategy = self.strategies.get(error_type)

        if strategy:
            return strategy.should_retry(error, attempt)
        return False

    def get_delay(self, attempt: int) -> float:
        return 1.0 * (2 ** attempt)

    def _classify_error(self, error: Exception) -> str:
        """Classify error type."""
        error_name = type(error).__name__.lower()

        if "rate" in error_name or "limit" in error_name:
            return "rate_limit"
        elif "json" in error_name or "validation" in error_name:
            return "validation"
        elif "timeout" in error_name:
            return "timeout"

        return "unknown"


class RetryConfig:
    """Enhanced retry configuration."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_factor: float = 2.0,
        jitter: bool = False,
        strategy: str = "exponential",
        retryable_exceptions: list = None,
        non_retryable_exceptions: list = None,
    ):
        if max_retries < 0:
            raise ValueError("max_retries must be positive")
        if base_delay < 0:
            raise ValueError("base_delay must be positive")
        if exponential_factor < 1.0:
            raise ValueError("exponential_factor must be >= 1")

        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_factor = exponential_factor
        self.jitter = jitter
        self.strategy = strategy
        self.retryable_exceptions = retryable_exceptions or []
        self.non_retryable_exceptions = non_retryable_exceptions or []

    @classmethod
    def from_dict(cls, data: dict) -> "RetryConfig":
        """Create config from dictionary."""
        return cls(
            max_retries=data.get("max_retries", 3),
            base_delay=data.get("base_delay", 1.0),
            max_delay=data.get("max_delay", 60.0),
            exponential_factor=data.get("exponential_factor", 2.0),
            jitter=data.get("jitter", False),
            strategy=data.get("strategy", "exponential"),
        )


class CircuitBreaker:
    """Circuit breaker for retry management."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.is_open = False
        self._last_failure_time = 0.0

    def record_failure(self):
        self.failure_count += 1
        self._last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.is_open = True

    def record_success(self):
        self.failure_count = 0
        self.is_open = False

    def can_execute(self) -> bool:
        if not self.is_open:
            return True
        # Check if recovery timeout has passed (half-open)
        if time.time() - self._last_failure_time >= self.recovery_timeout:
            return True
        return False


class RetryManager:
    """Manages retry execution for workflow nodes."""

    def __init__(self, default_strategy: RetryStrategy = None, collect_metrics: bool = False):
        """Initialize retry manager.

        Args:
            default_strategy: Default retry strategy
            collect_metrics: Whether to collect retry metrics
        """
        self.default_strategy = default_strategy or ExponentialBackoffRetry(base_delay=0.01)
        self._strategies: dict[str, RetryStrategy] = {"default": self.default_strategy}
        self.circuit_breaker: CircuitBreaker | None = None
        self._collect_metrics = collect_metrics
        self._metrics: dict[str, dict] = {}

    def register_strategy(self, name: str, strategy: RetryStrategy):
        """Register a named retry strategy."""
        self._strategies[name] = strategy

    def get_strategy(self, name: str) -> RetryStrategy:
        """Get a named retry strategy."""
        return self._strategies.get(name, self.default_strategy)

    def enable_circuit_breaker(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        """Enable circuit breaker pattern."""
        self.circuit_breaker = CircuitBreaker(failure_threshold, recovery_timeout)

    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        strategy: RetryStrategy = None,
        strategy_name: str = None,
        timeout: float = None,
        **kwargs
    ) -> Any:
        """Execute function with retry logic.

        Args:
            func: Async function to execute
            *args: Positional arguments
            strategy: Retry strategy to use
            strategy_name: Name of registered strategy
            timeout: Timeout in seconds
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If all retries fail
        """
        if strategy_name:
            strat = self.get_strategy(strategy_name)
        else:
            strat = strategy or self.default_strategy

        # Check circuit breaker at start
        if self.circuit_breaker and self.circuit_breaker.is_open:
            if not self.circuit_breaker.can_execute():
                raise Exception("Circuit breaker is open")

        async def _run():
            last_error = None

            for attempt in range(10):  # Hard limit
                try:
                    result = await func(*args, **kwargs)
                    if self.circuit_breaker:
                        self.circuit_breaker.record_success()
                    return result
                except Exception as e:
                    last_error = e

                    if not strat.should_retry(e, attempt):
                        break

                    delay = strat.get_delay(attempt)
                    logger.warning(
                        f"Retry attempt {attempt + 1}, "
                        f"waiting {delay:.1f}s: {e}"
                    )
                    await asyncio.sleep(delay)

            # Record failure for circuit breaker (once per call)
            if self.circuit_breaker and last_error:
                self.circuit_breaker.record_failure()

            if last_error:
                raise last_error

        if timeout:
            return await asyncio.wait_for(_run(), timeout=timeout)
        return await _run()

    def create_strategy(
        self,
        strategy_type: str,
        **kwargs
    ) -> RetryStrategy:
        """Create a retry strategy.

        Args:
            strategy_type: Type of strategy
            **kwargs: Strategy configuration

        Returns:
            Retry strategy instance
        """
        strategies = {
            "exponential": ExponentialBackoffRetry,
            "constant": ConstantRetry,
            "llm_output": LLMOutputRetry,
            "adaptive": AdaptiveRetry
        }

        strategy_class = strategies.get(strategy_type, ExponentialBackoffRetry)
        return strategy_class(**kwargs)

    def record_attempt(self, task_name: str, success: bool = True, duration: float = 0.0):
        """Record a retry attempt for metrics."""
        if task_name not in self._metrics:
            self._metrics[task_name] = {
                "attempts": 0,
                "successes": 0,
                "failures": 0,
                "total_duration": 0.0,
            }
        m = self._metrics[task_name]
        m["attempts"] += 1
        if success:
            m["successes"] += 1
        else:
            m["failures"] += 1
        m["total_duration"] += duration

    def get_metrics(self) -> dict:
        """Get retry metrics."""
        result = {}
        for task, m in self._metrics.items():
            result[task] = {
                "attempts": m["attempts"],
                "successes": m["successes"],
                "failures": m["failures"],
                "avg_duration": m["total_duration"] / m["attempts"] if m["attempts"] > 0 else 0,
            }
        return result

    def reset_metrics(self):
        """Reset all metrics."""
        self._metrics = {}
