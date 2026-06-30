"""Tests for circuit breaker implementation."""

import asyncio
import pytest

from jobqueue.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitOpenError,
    CircuitState,
)


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    @pytest.fixture
    def breaker(self):
        """Create a circuit breaker for testing."""
        return CircuitBreaker(
            name="test",
            failure_threshold=3,
            success_threshold=2,
            reset_timeout=1.0,
        )

    async def test_initial_state_closed(self, breaker):
        """Test that breaker starts in closed state."""
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed
        assert not breaker.is_open

    async def test_successful_call(self, breaker):
        """Test successful call through breaker."""
        async def success():
            return "ok"

        result = await breaker.call(success)
        assert result == "ok"
        assert breaker.state == CircuitState.CLOSED

    async def test_failed_call(self, breaker):
        """Test failed call through breaker."""
        async def failure():
            raise ValueError("error")

        with pytest.raises(ValueError):
            await breaker.call(failure)

        # Should still be closed after one failure
        assert breaker.state == CircuitState.CLOSED

    async def test_opens_after_threshold(self, breaker):
        """Test that breaker opens after failure threshold."""
        async def failure():
            raise ValueError("error")

        # Fail 3 times (threshold)
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.call(failure)

        # Should now be open
        assert breaker.state == CircuitState.OPEN

    async def test_rejects_when_open(self, breaker):
        """Test that breaker rejects calls when open."""
        async def failure():
            raise ValueError("error")

        # Trip the breaker
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.call(failure)

        # Next call should be rejected
        async def success():
            return "ok"

        with pytest.raises(CircuitOpenError):
            await breaker.call(success)

    async def test_half_open_after_timeout(self, breaker):
        """Test transition to half-open after reset timeout."""
        async def failure():
            raise ValueError("error")

        # Trip the breaker
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.call(failure)

        assert breaker.state == CircuitState.OPEN

        # Wait for reset timeout
        await asyncio.sleep(1.1)

        # Next call should go through (half-open state)
        async def success():
            return "ok"

        result = await breaker.call(success)
        assert result == "ok"

    async def test_closes_after_successes(self, breaker):
        """Test that breaker closes after success threshold in half-open."""
        async def failure():
            raise ValueError("error")

        # Trip the breaker
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.call(failure)

        # Wait for reset timeout
        await asyncio.sleep(1.1)

        async def success():
            return "ok"

        # Succeed twice (success_threshold)
        await breaker.call(success)
        await breaker.call(success)

        assert breaker.state == CircuitState.CLOSED

    async def test_reopens_on_half_open_failure(self, breaker):
        """Test that breaker reopens on failure in half-open state."""
        async def failure():
            raise ValueError("error")

        # Trip the breaker
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.call(failure)

        # Wait for reset timeout
        await asyncio.sleep(1.1)

        # Fail in half-open state
        with pytest.raises(ValueError):
            await breaker.call(failure)

        assert breaker.state == CircuitState.OPEN

    async def test_manual_reset(self, breaker):
        """Test manual reset of breaker."""
        async def failure():
            raise ValueError("error")

        # Trip the breaker
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.call(failure)

        assert breaker.state == CircuitState.OPEN

        # Manual reset
        breaker.reset()
        assert breaker.state == CircuitState.CLOSED

    async def test_get_stats(self, breaker):
        """Test getting breaker statistics."""
        stats = breaker.get_stats()

        assert stats["name"] == "test"
        assert stats["state"] == CircuitState.CLOSED.value
        assert stats["failure_count"] == 0
        assert stats["failure_threshold"] == 3


class TestCircuitBreakerRegistry:
    """Tests for CircuitBreakerRegistry."""

    @pytest.fixture
    def registry(self):
        """Create a registry for testing."""
        return CircuitBreakerRegistry()

    async def test_get_or_create(self, registry):
        """Test getting or creating a breaker."""
        breaker1 = await registry.get_or_create("test")
        breaker2 = await registry.get_or_create("test")

        # Should return same instance
        assert breaker1 is breaker2

    async def test_different_breakers(self, registry):
        """Test creating different breakers."""
        breaker1 = await registry.get_or_create("test1")
        breaker2 = await registry.get_or_create("test2")

        assert breaker1 is not breaker2

    async def test_get_all_stats(self, registry):
        """Test getting stats for all breakers."""
        await registry.get_or_create("test1")
        await registry.get_or_create("test2")

        stats = registry.get_all_stats()
        assert "test1" in stats
        assert "test2" in stats

    async def test_reset_all(self, registry):
        """Test resetting all breakers."""
        breaker1 = await registry.get_or_create("test1", failure_threshold=1)
        breaker2 = await registry.get_or_create("test2", failure_threshold=1)

        async def failure():
            raise ValueError("error")

        # Trip both breakers
        with pytest.raises(ValueError):
            await breaker1.call(failure)
        with pytest.raises(ValueError):
            await breaker2.call(failure)

        assert breaker1.state == CircuitState.OPEN
        assert breaker2.state == CircuitState.OPEN

        # Reset all
        registry.reset_all()

        assert breaker1.state == CircuitState.CLOSED
        assert breaker2.state == CircuitState.CLOSED
