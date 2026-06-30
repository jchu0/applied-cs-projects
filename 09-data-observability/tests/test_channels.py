"""Tests for additional alert channels."""

import pytest
import asyncio
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from observability.alerting import (
    EmailChannel,
    TeamsChannel,
    WebhookChannel,
)
from observability.models import Alert, Anomaly, AnomalyType


@pytest.fixture
def sample_anomaly():
    """Create a sample anomaly."""
    return Anomaly(
        anomaly_id="anomaly-001",
        table_id="db.schema.users",
        column_name=None,
        anomaly_type=AnomalyType.VOLUME,
        severity="critical",
        detected_at=datetime.now(),
        metric_value=1000,
        expected_range=(1800, 2200),
        description="Row count dropped by 50%",
    )


@pytest.fixture
def sample_alert(sample_anomaly):
    """Create a sample alert."""
    return Alert(
        alert_id="alert-001",
        anomaly=sample_anomaly,
        created_at=datetime.now(),
        status="active",
    )


class TestEmailChannel:
    """Tests for EmailChannel."""

    def test_channel_creation(self):
        """Test creating email channel."""
        channel = EmailChannel(
            smtp_host="smtp.example.com",
            smtp_port=587,
            from_address="alerts@example.com",
            to_addresses=["team@example.com"],
        )

        assert channel.smtp_host == "smtp.example.com"
        assert channel.smtp_port == 587
        assert "team@example.com" in channel.to_addresses

    def test_send_email(self, sample_alert):
        """Test sending email."""
        channel = EmailChannel(
            smtp_host="localhost",
            from_address="alerts@example.com",
            to_addresses=["team@example.com"],
        )

        asyncio.run(channel.send(sample_alert))

        assert len(channel.sent_emails) == 1
        email = channel.sent_emails[0]
        assert email["from"] == "alerts@example.com"
        assert "team@example.com" in email["to"]
        assert "CRITICAL" in email["subject"]
        assert sample_alert.anomaly.table_id in email["body"]

    def test_send_escalated_email(self, sample_alert):
        """Test sending escalated email."""
        channel = EmailChannel(
            from_address="alerts@example.com",
            to_addresses=["urgent@example.com"],
        )

        asyncio.run(channel.send(sample_alert, escalated=True))

        assert len(channel.sent_emails) == 1
        email = channel.sent_emails[0]
        assert "[ESCALATED]" in email["subject"]

    def test_multiple_recipients(self, sample_alert):
        """Test sending to multiple recipients."""
        channel = EmailChannel(
            from_address="alerts@example.com",
            to_addresses=["team@example.com", "oncall@example.com"],
        )

        asyncio.run(channel.send(sample_alert))

        email = channel.sent_emails[0]
        assert len(email["to"]) == 2


class TestWebhookChannel:
    """Tests for WebhookChannel."""

    def test_channel_creation(self):
        """Test creating webhook channel."""
        channel = WebhookChannel(
            url="https://example.com/webhook",
            headers={"Authorization": "Bearer token123"},
        )

        assert channel.url == "https://example.com/webhook"
        assert "Authorization" in channel.headers

    def test_send_webhook(self, sample_alert):
        """Test sending webhook."""
        channel = WebhookChannel(url="https://example.com/webhook")

        asyncio.run(channel.send(sample_alert))

        assert len(channel.sent_webhooks) == 1
        webhook = channel.sent_webhooks[0]
        assert webhook["url"] == "https://example.com/webhook"
        assert webhook["method"] == "POST"
        assert "anomaly" in webhook["payload"]
        assert webhook["payload"]["event"] == "anomaly_detected"

    def test_webhook_with_signature(self, sample_alert):
        """Test webhook with HMAC signature."""
        channel = WebhookChannel(
            url="https://example.com/webhook",
            secret="my-secret-key",
        )

        asyncio.run(channel.send(sample_alert))

        webhook = channel.sent_webhooks[0]
        assert "X-Signature" in webhook["headers"]
        assert webhook["headers"]["X-Signature"].startswith("sha256=")

    def test_escalated_webhook(self, sample_alert):
        """Test escalated webhook payload."""
        channel = WebhookChannel(url="https://example.com/webhook")

        asyncio.run(channel.send(sample_alert, escalated=True))

        webhook = channel.sent_webhooks[0]
        assert webhook["payload"]["escalated"] is True

    def test_custom_headers(self, sample_alert):
        """Test webhook with custom headers."""
        channel = WebhookChannel(
            url="https://example.com/webhook",
            headers={
                "Content-Type": "application/json",
                "X-Custom-Header": "custom-value",
            },
        )

        asyncio.run(channel.send(sample_alert))

        webhook = channel.sent_webhooks[0]
        assert webhook["headers"]["X-Custom-Header"] == "custom-value"


class TestTeamsChannel:
    """Tests for TeamsChannel."""

    def test_channel_creation(self):
        """Test creating Teams channel."""
        channel = TeamsChannel(
            webhook_url="https://outlook.office.com/webhook/xxx"
        )

        assert "outlook.office.com" in channel.webhook_url

    def test_send_teams_message(self, sample_alert):
        """Test sending Teams message."""
        channel = TeamsChannel(
            webhook_url="https://outlook.office.com/webhook/xxx"
        )

        asyncio.run(channel.send(sample_alert))

        assert len(channel.sent_messages) == 1
        card = channel.sent_messages[0]
        assert card["@type"] == "MessageCard"
        assert card["themeColor"] == "FF0000"  # Critical = red

    def test_teams_severity_colors(self, sample_anomaly):
        """Test Teams color coding for severities."""
        channel = TeamsChannel(
            webhook_url="https://outlook.office.com/webhook/xxx"
        )

        severities = {
            "critical": "FF0000",
            "warning": "FFA500",
            "info": "0078D7",
        }

        for severity, expected_color in severities.items():
            channel.sent_messages.clear()
            sample_anomaly.severity = severity
            alert = Alert(
                alert_id=f"alert-{severity}",
                anomaly=sample_anomaly,
                created_at=datetime.now(),
                status="active",
            )

            asyncio.run(channel.send(alert))

            card = channel.sent_messages[0]
            assert card["themeColor"] == expected_color

    def test_escalated_teams_message(self, sample_alert):
        """Test escalated Teams message."""
        channel = TeamsChannel(
            webhook_url="https://outlook.office.com/webhook/xxx"
        )

        asyncio.run(channel.send(sample_alert, escalated=True))

        card = channel.sent_messages[0]
        assert "[ESCALATED]" in card["sections"][0]["activityTitle"]


class TestChannelIntegration:
    """Integration tests for alert channels."""

    def test_multiple_channels_same_alert(self, sample_alert):
        """Test sending to multiple channels."""
        email_channel = EmailChannel(
            from_address="alerts@example.com",
            to_addresses=["team@example.com"],
        )
        webhook_channel = WebhookChannel(url="https://example.com/webhook")
        teams_channel = TeamsChannel(
            webhook_url="https://outlook.office.com/webhook/xxx"
        )

        async def send_all():
            await email_channel.send(sample_alert)
            await webhook_channel.send(sample_alert)
            await teams_channel.send(sample_alert)

        asyncio.run(send_all())

        assert len(email_channel.sent_emails) == 1
        assert len(webhook_channel.sent_webhooks) == 1
        assert len(teams_channel.sent_messages) == 1

    def test_channel_returns_success(self, sample_alert):
        """Test that channels return success."""
        email_channel = EmailChannel(
            from_address="alerts@example.com",
            to_addresses=["team@example.com"],
        )

        result = asyncio.run(email_channel.send(sample_alert))
        assert result is True
