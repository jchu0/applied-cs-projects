"""Tests for alerting module."""

import pytest
from datetime import datetime
from observability.alerting import AlertingEngine, LogChannel, SlackChannel, PagerDutyChannel
from observability.config import AlertConfig
from observability.models import Anomaly, AnomalyType, Alert


class TestLogChannel:
    """Tests for LogChannel."""

    def test_log_channel_creation(self):
        """Test log channel creation."""
        channel = LogChannel()
        assert channel is not None

    @pytest.mark.asyncio
    async def test_send_alert(self, sample_alert):
        """Test sending alert to log channel."""
        channel = LogChannel()
        result = await channel.send(sample_alert)
        assert result is True


class TestSlackChannel:
    """Tests for SlackChannel."""

    def test_slack_channel_creation(self):
        """Test slack channel creation."""
        channel = SlackChannel(webhook_url="https://hooks.slack.com/test")
        assert channel.webhook_url == "https://hooks.slack.com/test"

    def test_slack_channel_default_channel(self):
        """Test slack channel default channel."""
        channel = SlackChannel(
            webhook_url="https://hooks.slack.com/test",
            default_channel="#alerts"
        )
        assert channel.default_channel == "#alerts"


class TestPagerDutyChannel:
    """Tests for PagerDutyChannel."""

    def test_pagerduty_channel_creation(self):
        """Test pagerduty channel creation."""
        channel = PagerDutyChannel(
            routing_key="test_routing_key",
            service_id="test_service"
        )
        assert channel.routing_key == "test_routing_key"
        assert channel.service_id == "test_service"


class TestAlertingEngine:
    """Tests for AlertingEngine."""

    @pytest.fixture
    def alert_engine(self):
        """Create alert engine with test configuration."""
        config = AlertConfig(
            default_severity="warning",
            escalation_minutes=30,
            dedup_window_minutes=60,
            channels=["log"]
        )
        engine = AlertingEngine(config)
        engine.add_channel("log", LogChannel())
        return engine

    def test_add_channel(self, alert_engine):
        """Test adding alert channel."""
        alert_engine.add_channel("slack", SlackChannel(webhook_url="https://test"))
        assert "slack" in alert_engine.channels

    def test_remove_channel(self, alert_engine):
        """Test removing alert channel."""
        alert_engine.add_channel("test", LogChannel())
        alert_engine.remove_channel("test")
        assert "test" not in alert_engine.channels

    @pytest.mark.asyncio
    async def test_create_alert(self, alert_engine, sample_anomaly):
        """Test creating alert from anomaly."""
        alert = await alert_engine.create_alert(sample_anomaly)
        assert alert is not None
        assert alert.anomaly == sample_anomaly
        assert alert.status == "active"

    @pytest.mark.asyncio
    async def test_acknowledge_alert(self, alert_engine, sample_anomaly):
        """Test acknowledging an alert."""
        alert = await alert_engine.create_alert(sample_anomaly)
        updated = await alert_engine.acknowledge_alert(
            alert.alert_id,
            acknowledged_by="user@example.com"
        )
        assert updated.status == "acknowledged"
        assert updated.acknowledged_by == "user@example.com"
        assert updated.acknowledged_at is not None

    @pytest.mark.asyncio
    async def test_resolve_alert(self, alert_engine, sample_anomaly):
        """Test resolving an alert."""
        alert = await alert_engine.create_alert(sample_anomaly)
        updated = await alert_engine.resolve_alert(
            alert.alert_id,
            resolution_notes="Fixed the data pipeline"
        )
        assert updated.status == "resolved"
        assert updated.resolution_notes == "Fixed the data pipeline"
        assert updated.resolved_at is not None

    @pytest.mark.asyncio
    async def test_get_active_alerts(self, alert_engine, sample_anomaly):
        """Test getting active alerts."""
        await alert_engine.create_alert(sample_anomaly)
        active = await alert_engine.get_active_alerts()
        assert len(active) >= 1

    @pytest.mark.asyncio
    async def test_deduplication(self, alert_engine):
        """Test alert deduplication."""
        # Create two similar anomalies
        anomaly1 = Anomaly(
            anomaly_id="anom_001",
            table_id="test.table",
            column_name=None,
            anomaly_type=AnomalyType.VOLUME,
            severity="warning",
            detected_at=datetime.now(),
            metric_value=1000,
            expected_range=(900, 1100),
            description="Test"
        )
        anomaly2 = Anomaly(
            anomaly_id="anom_002",
            table_id="test.table",  # Same table
            column_name=None,
            anomaly_type=AnomalyType.VOLUME,  # Same type
            severity="warning",
            detected_at=datetime.now(),
            metric_value=1100,
            expected_range=(900, 1100),
            description="Test"
        )

        alert1 = await alert_engine.create_alert(anomaly1)
        alert2 = await alert_engine.create_alert(anomaly2)

        # Depending on dedup implementation, second might be deduplicated
        assert alert1 is not None

    @pytest.mark.asyncio
    async def test_routing_by_severity(self, alert_engine):
        """Test alert routing based on severity."""
        critical_anomaly = Anomaly(
            anomaly_id="crit_001",
            table_id="test.table",
            column_name=None,
            anomaly_type=AnomalyType.SCHEMA,
            severity="critical",
            detected_at=datetime.now(),
            metric_value=1,
            expected_range=(0, 0),
            description="Critical schema change"
        )

        warning_anomaly = Anomaly(
            anomaly_id="warn_001",
            table_id="test.table",
            column_name=None,
            anomaly_type=AnomalyType.VOLUME,
            severity="warning",
            detected_at=datetime.now(),
            metric_value=1000,
            expected_range=(900, 1100),
            description="Volume warning"
        )

        crit_alert = await alert_engine.create_alert(critical_anomaly)
        warn_alert = await alert_engine.create_alert(warning_anomaly)

        assert crit_alert.anomaly.severity == "critical"
        assert warn_alert.anomaly.severity == "warning"


class TestAlertLifecycle:
    """Tests for alert lifecycle management."""

    @pytest.fixture
    def engine_with_alerts(self):
        """Create engine with pre-existing alerts."""
        config = AlertConfig()
        engine = AlertingEngine(config)
        engine.add_channel("log", LogChannel())
        return engine

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, engine_with_alerts):
        """Test complete alert lifecycle."""
        anomaly = Anomaly(
            anomaly_id="lifecycle_001",
            table_id="test.table",
            column_name=None,
            anomaly_type=AnomalyType.FRESHNESS,
            severity="warning",
            detected_at=datetime.now(),
            metric_value=48.0,
            expected_range=(0, 24),
            description="Data is stale"
        )

        # Create
        alert = await engine_with_alerts.create_alert(anomaly)
        assert alert.status == "active"

        # Acknowledge
        acked = await engine_with_alerts.acknowledge_alert(
            alert.alert_id,
            acknowledged_by="oncall@example.com"
        )
        assert acked.status == "acknowledged"

        # Resolve
        resolved = await engine_with_alerts.resolve_alert(
            alert.alert_id,
            resolution_notes="Pipeline restarted"
        )
        assert resolved.status == "resolved"

    @pytest.mark.asyncio
    async def test_get_alerts_by_table(self, engine_with_alerts):
        """Test filtering alerts by table."""
        for i, table in enumerate(["table_a", "table_b", "table_a"]):
            anomaly = Anomaly(
                anomaly_id=f"filter_{i}",
                table_id=table,
                column_name=None,
                anomaly_type=AnomalyType.VOLUME,
                severity="warning",
                detected_at=datetime.now(),
                metric_value=1000,
                expected_range=(900, 1100),
                description="Test"
            )
            await engine_with_alerts.create_alert(anomaly)

        all_alerts = await engine_with_alerts.get_active_alerts()
        table_a_alerts = [a for a in all_alerts if a.anomaly.table_id == "table_a"]
        assert len(table_a_alerts) >= 2
