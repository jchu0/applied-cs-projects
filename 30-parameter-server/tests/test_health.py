"""Tests for health monitoring."""

import pytest
import asyncio
import time

from paramserver.fault_tolerance.health import (
    HealthMonitor,
    HealthStatus,
    HealthRecord,
)


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_enum_values(self):
        """Test enum values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"


class TestHealthRecord:
    """Tests for HealthRecord dataclass."""

    def test_create_record(self):
        """Test creating health record."""
        record = HealthRecord(
            component_id="server-0",
            component_type="server",
        )

        assert record.component_id == "server-0"
        assert record.component_type == "server"
        assert record.status == HealthStatus.UNKNOWN
        assert record.consecutive_failures == 0

    def test_record_with_metadata(self):
        """Test record with metadata."""
        record = HealthRecord(
            component_id="worker-0",
            component_type="worker",
            metadata={"address": "localhost:8080"},
        )

        assert record.metadata["address"] == "localhost:8080"


class TestHealthMonitorInit:
    """Tests for HealthMonitor initialization."""

    def test_create_default(self):
        """Test default creation."""
        monitor = HealthMonitor()
        assert monitor.heartbeat_interval == 5.0
        assert monitor.failure_threshold == 3

    def test_create_custom(self):
        """Test custom creation."""
        monitor = HealthMonitor(
            heartbeat_interval=1.0,
            failure_threshold=5,
            degraded_threshold=2,
        )
        assert monitor.heartbeat_interval == 1.0
        assert monitor.failure_threshold == 5
        assert monitor.degraded_threshold == 2


class TestComponentRegistration:
    """Tests for component registration."""

    def test_register_component(self):
        """Test registering a component."""
        monitor = HealthMonitor()
        monitor.register_component("server-0", "server")

        status = monitor.get_status("server-0")
        assert status == HealthStatus.HEALTHY

    def test_register_with_metadata(self):
        """Test registering with metadata."""
        monitor = HealthMonitor()
        monitor.register_component(
            "server-0",
            "server",
            metadata={"shard_id": 0},
        )

        record = monitor.get_record("server-0")
        assert record.metadata["shard_id"] == 0

    def test_unregister_component(self):
        """Test unregistering a component."""
        monitor = HealthMonitor()
        monitor.register_component("server-0", "server")

        removed = monitor.unregister_component("server-0")

        assert removed is not None
        assert monitor.get_status("server-0") is None

    def test_unregister_nonexistent(self):
        """Test unregistering nonexistent component."""
        monitor = HealthMonitor()
        removed = monitor.unregister_component("nonexistent")
        assert removed is None


class TestHeartbeat:
    """Tests for heartbeat recording."""

    @pytest.mark.asyncio
    async def test_record_heartbeat(self):
        """Test recording heartbeat."""
        monitor = HealthMonitor(heartbeat_interval=0.1)
        monitor.register_component("server-0", "server")

        await monitor.record_heartbeat("server-0")

        record = monitor.get_record("server-0")
        assert record.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_heartbeat_updates_timestamp(self):
        """Test heartbeat updates timestamp."""
        monitor = HealthMonitor()
        monitor.register_component("server-0", "server")

        old_time = monitor.get_record("server-0").last_heartbeat
        await asyncio.sleep(0.01)
        await monitor.record_heartbeat("server-0")

        new_time = monitor.get_record("server-0").last_heartbeat
        assert new_time > old_time

    @pytest.mark.asyncio
    async def test_heartbeat_auto_registers(self):
        """Test heartbeat auto-registers unknown component."""
        monitor = HealthMonitor()

        await monitor.record_heartbeat("unknown-component")

        assert monitor.get_status("unknown-component") == HealthStatus.HEALTHY


class TestHealthEvaluation:
    """Tests for health status evaluation."""

    @pytest.mark.asyncio
    async def test_healthy_status(self):
        """Test healthy status."""
        monitor = HealthMonitor(heartbeat_interval=1.0)
        monitor.register_component("server-0", "server")

        await monitor.record_heartbeat("server-0")
        status = await monitor.check_health("server-0")

        assert status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_degraded_status(self):
        """Test degraded status after missed heartbeat."""
        monitor = HealthMonitor(
            heartbeat_interval=0.05,
            degraded_threshold=1,
            failure_threshold=3,
        )
        monitor.register_component("server-0", "server")

        # Wait for degraded threshold
        await asyncio.sleep(0.1)

        status = await monitor.check_health("server-0")
        assert status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_unhealthy_status(self):
        """Test unhealthy status after multiple missed heartbeats."""
        monitor = HealthMonitor(
            heartbeat_interval=0.02,
            failure_threshold=2,
        )
        monitor.register_component("server-0", "server")

        # Wait for failure threshold
        await asyncio.sleep(0.1)

        status = await monitor.check_health("server-0")
        assert status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_recovery(self):
        """Test recovery after heartbeat."""
        monitor = HealthMonitor(
            heartbeat_interval=0.02,
            failure_threshold=2,
        )
        monitor.register_component("server-0", "server")

        # Let it become unhealthy
        await asyncio.sleep(0.1)
        status = await monitor.check_health("server-0")
        assert status == HealthStatus.UNHEALTHY

        # Recover with heartbeat
        await monitor.record_heartbeat("server-0")
        status = await monitor.check_health("server-0")
        assert status == HealthStatus.HEALTHY


class TestCallbacks:
    """Tests for health callbacks."""

    @pytest.mark.asyncio
    async def test_failure_callback(self):
        """Test failure callback is triggered."""
        monitor = HealthMonitor(
            heartbeat_interval=0.02,
            failure_threshold=2,
        )

        failures = []

        def on_failure(component_id, record):
            failures.append(component_id)

        monitor.add_failure_callback(on_failure)
        monitor.register_component("server-0", "server")

        # Start monitoring and let it detect failure
        await monitor.start_monitoring()
        await asyncio.sleep(0.15)
        await monitor.stop_monitoring()

        assert "server-0" in failures

    @pytest.mark.asyncio
    async def test_recovery_callback(self):
        """Test recovery callback is triggered."""
        monitor = HealthMonitor(
            heartbeat_interval=0.02,
            failure_threshold=2,
        )

        recoveries = []

        def on_recovery(component_id, record):
            recoveries.append(component_id)

        monitor.add_recovery_callback(on_recovery)
        monitor.register_component("server-0", "server")

        # Let it become unhealthy
        await asyncio.sleep(0.1)
        record = monitor.get_record("server-0")
        record.status = HealthStatus.UNHEALTHY

        # Recover
        await monitor.record_heartbeat("server-0")

        assert "server-0" in recoveries


class TestComponentQueries:
    """Tests for component query methods."""

    def test_get_unhealthy_components(self):
        """Test getting unhealthy components."""
        monitor = HealthMonitor()
        monitor.register_component("healthy-0", "server")
        monitor.register_component("unhealthy-0", "server")
        monitor.register_component("unhealthy-1", "server")

        monitor.get_record("unhealthy-0").status = HealthStatus.UNHEALTHY
        monitor.get_record("unhealthy-1").status = HealthStatus.UNHEALTHY

        unhealthy = monitor.get_unhealthy_components()

        assert len(unhealthy) == 2
        assert "unhealthy-0" in unhealthy
        assert "unhealthy-1" in unhealthy

    def test_get_healthy_components(self):
        """Test getting healthy components."""
        monitor = HealthMonitor()
        monitor.register_component("healthy-0", "server")
        monitor.register_component("unhealthy-0", "server")

        monitor.get_record("unhealthy-0").status = HealthStatus.UNHEALTHY

        healthy = monitor.get_healthy_components()

        assert len(healthy) == 1
        assert "healthy-0" in healthy

    def test_is_component_healthy(self):
        """Test checking if component is healthy."""
        monitor = HealthMonitor()
        monitor.register_component("server-0", "server")

        assert monitor.is_component_healthy("server-0") is True
        assert monitor.is_component_healthy("nonexistent") is False


class TestMonitoring:
    """Tests for background monitoring."""

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        monitor = HealthMonitor(heartbeat_interval=0.1)

        assert not monitor._monitoring

        await monitor.start_monitoring()
        assert monitor._monitoring

        await monitor.stop_monitoring()
        assert not monitor._monitoring

    @pytest.mark.asyncio
    async def test_monitoring_detects_failures(self):
        """Test monitoring detects failures."""
        monitor = HealthMonitor(
            heartbeat_interval=0.02,
            failure_threshold=2,
        )
        monitor.register_component("server-0", "server")

        await monitor.start_monitoring()
        await asyncio.sleep(0.15)
        await monitor.stop_monitoring()

        status = monitor.get_status("server-0")
        assert status == HealthStatus.UNHEALTHY


class TestWaitForHealthy:
    """Tests for wait_for_healthy method."""

    @pytest.mark.asyncio
    async def test_wait_already_healthy(self):
        """Test wait returns immediately if already healthy."""
        monitor = HealthMonitor()
        monitor.register_component("server-0", "server")

        result = await monitor.wait_for_healthy("server-0", timeout=0.1)
        assert result is True

    @pytest.mark.asyncio
    async def test_wait_timeout(self):
        """Test wait times out."""
        monitor = HealthMonitor()
        monitor.register_component("server-0", "server")
        monitor.get_record("server-0").status = HealthStatus.UNHEALTHY

        result = await monitor.wait_for_healthy("server-0", timeout=0.05)
        assert result is False


class TestHealthStats:
    """Tests for health statistics."""

    def test_get_stats(self):
        """Test getting stats."""
        monitor = HealthMonitor(heartbeat_interval=5.0, failure_threshold=3)

        monitor.register_component("healthy-0", "server")
        monitor.register_component("unhealthy-0", "server")
        monitor.get_record("unhealthy-0").status = HealthStatus.UNHEALTHY

        stats = monitor.get_stats()

        assert stats["total_components"] == 2
        assert stats["heartbeat_interval"] == 5.0
        assert stats["failure_threshold"] == 3
        assert stats["status_counts"]["healthy"] == 1
        assert stats["status_counts"]["unhealthy"] == 1
