"""Alerting engine for anomaly notifications."""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from observability.models import Alert, Anomaly, AnomalyType
from observability.detector import generate_id

logger = logging.getLogger(__name__)


@dataclass
class EscalationPolicy:
    """Policy for alert escalation."""

    escalate_after_minutes: int = 30
    escalate_to: List[str] = field(default_factory=list)


@dataclass
class AlertRule:
    """Rule for alert routing."""

    name: str
    conditions: Dict[str, Any]
    channels: List[str]
    escalation_policy: Optional[EscalationPolicy] = None

    def matches(self, anomaly: Anomaly) -> bool:
        """Check if anomaly matches this rule."""
        # Check anomaly type
        if "anomaly_types" in self.conditions:
            if anomaly.anomaly_type not in self.conditions["anomaly_types"]:
                return False

        # Check severity
        if "severities" in self.conditions:
            if anomaly.severity not in self.conditions["severities"]:
                return False

        # Check table patterns
        if "table_patterns" in self.conditions:
            if not any(
                pattern in anomaly.table_id
                for pattern in self.conditions["table_patterns"]
            ):
                return False

        return True

    def get_channels(self, anomaly: Anomaly) -> List[str]:
        """Get channels for this anomaly."""
        return self.channels


class AlertChannel(ABC):
    """Base class for alert channels."""

    @abstractmethod
    async def send(self, alert: Alert, escalated: bool = False) -> bool:
        """Send alert to channel."""
        return True


class LogChannel(AlertChannel):
    """Log-based alert channel for testing."""

    def __init__(self, name: str = "log"):
        self.name = name
        self.sent_alerts: List[Alert] = []

    async def send(self, alert: Alert, escalated: bool = False) -> bool:
        """Log the alert."""
        prefix = "[ESCALATED] " if escalated else ""
        logger.warning(
            f"{prefix}Alert: {alert.anomaly.anomaly_type.value} - "
            f"{alert.anomaly.description}"
        )
        self.sent_alerts.append(alert)
        return True


class SlackChannel(AlertChannel):
    """Slack alert channel."""

    def __init__(self, webhook_url: str, default_channel: str = "#alerts"):
        self.webhook_url = webhook_url
        self.default_channel = default_channel
        self.channel = default_channel  # Alias for backwards compatibility

    async def send(self, alert: Alert, escalated: bool = False) -> None:
        """Send alert to Slack."""
        emoji = {
            "critical": ":rotating_light:",
            "warning": ":warning:",
            "info": ":information_source:",
        }.get(alert.anomaly.severity, ":grey_question:")

        prefix = "[ESCALATED] " if escalated else ""

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} {prefix}Data Anomaly Detected",
                },
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Table:*\n{alert.anomaly.table_id}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Type:*\n{alert.anomaly.anomaly_type.value}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Severity:*\n{alert.anomaly.severity}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Detected:*\n{alert.anomaly.detected_at.strftime('%Y-%m-%d %H:%M:%S')}",
                    },
                ],
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Description:*\n{alert.anomaly.description}",
                },
            },
        ]

        # In production, would send to Slack webhook
        logger.info(f"Would send to Slack channel {self.channel}: {blocks}")


class PagerDutyChannel(AlertChannel):
    """PagerDuty alert channel."""

    def __init__(self, api_key: str = None, service_id: str = None, routing_key: str = None):
        # Support both api_key and routing_key parameters
        self.api_key = api_key or routing_key
        self.routing_key = self.api_key  # Alias
        self.service_id = service_id

    async def send(self, alert: Alert, escalated: bool = False) -> None:
        """Send alert to PagerDuty."""
        severity_map = {
            "critical": "critical",
            "warning": "warning",
            "info": "info",
        }

        payload = {
            "routing_key": self.api_key,
            "event_action": "trigger",
            "payload": {
                "summary": alert.anomaly.description,
                "severity": severity_map.get(alert.anomaly.severity, "info"),
                "source": alert.anomaly.table_id,
                "custom_details": {
                    "anomaly_type": alert.anomaly.anomaly_type.value,
                    "metric_value": alert.anomaly.metric_value,
                    "expected_range": alert.anomaly.expected_range,
                },
            },
        }

        # In production, would send to PagerDuty API
        logger.info(f"Would send to PagerDuty: {payload}")


class EmailChannel(AlertChannel):
    """Email alert channel."""

    def __init__(
        self,
        smtp_host: str = "localhost",
        smtp_port: int = 587,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        from_address: str = "alerts@dataobservability.io",
        to_addresses: Optional[List[str]] = None,
        use_tls: bool = True,
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.from_address = from_address
        self.to_addresses = to_addresses or []
        self.use_tls = use_tls
        self.sent_emails: List[Dict[str, Any]] = []

    async def send(self, alert: Alert, escalated: bool = False) -> bool:
        """Send alert via email."""
        prefix = "[ESCALATED] " if escalated else ""
        severity = alert.anomaly.severity.upper()

        subject = f"{prefix}[{severity}] Data Anomaly: {alert.anomaly.anomaly_type.value} - {alert.anomaly.table_id}"

        body = f"""
Data Observability Alert

Table: {alert.anomaly.table_id}
Type: {alert.anomaly.anomaly_type.value}
Severity: {alert.anomaly.severity}
Detected: {alert.anomaly.detected_at.strftime('%Y-%m-%d %H:%M:%S')}

Description:
{alert.anomaly.description}

Metric Value: {alert.anomaly.metric_value}
Expected Range: {alert.anomaly.expected_range}

Alert ID: {alert.alert_id}
        """.strip()

        email_data = {
            "from": self.from_address,
            "to": self.to_addresses,
            "subject": subject,
            "body": body,
            "alert_id": alert.alert_id,
        }

        # Store for testing
        self.sent_emails.append(email_data)

        # In production, would use smtplib or aiosmtplib
        logger.info(f"Would send email to {self.to_addresses}: {subject}")

        return True


class WebhookChannel(AlertChannel):
    """Generic webhook alert channel."""

    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        method: str = "POST",
        secret: Optional[str] = None,
    ):
        self.url = url
        self.headers = headers or {"Content-Type": "application/json"}
        self.method = method
        self.secret = secret
        self.sent_webhooks: List[Dict[str, Any]] = []

    async def send(self, alert: Alert, escalated: bool = False) -> bool:
        """Send alert to webhook endpoint."""
        import hashlib
        import hmac
        import json

        payload = {
            "event": "anomaly_detected",
            "escalated": escalated,
            "alert": {
                "id": alert.alert_id,
                "status": alert.status,
                "created_at": alert.created_at.isoformat(),
            },
            "anomaly": {
                "id": alert.anomaly.anomaly_id,
                "table_id": alert.anomaly.table_id,
                "type": alert.anomaly.anomaly_type.value,
                "severity": alert.anomaly.severity,
                "description": alert.anomaly.description,
                "metric_value": alert.anomaly.metric_value,
                "expected_range": alert.anomaly.expected_range,
                "detected_at": alert.anomaly.detected_at.isoformat(),
            },
        }

        headers = self.headers.copy()

        # Add HMAC signature if secret is configured
        if self.secret:
            payload_bytes = json.dumps(payload).encode()
            signature = hmac.new(
                self.secret.encode(),
                payload_bytes,
                hashlib.sha256
            ).hexdigest()
            headers["X-Signature"] = f"sha256={signature}"

        webhook_data = {
            "url": self.url,
            "method": self.method,
            "headers": headers,
            "payload": payload,
        }

        # Store for testing
        self.sent_webhooks.append(webhook_data)

        # In production, would use aiohttp or httpx
        logger.info(f"Would send webhook to {self.url}: {payload}")

        return True


class TeamsChannel(AlertChannel):
    """Microsoft Teams alert channel."""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.sent_messages: List[Dict[str, Any]] = []

    async def send(self, alert: Alert, escalated: bool = False) -> bool:
        """Send alert to Microsoft Teams."""
        prefix = "[ESCALATED] " if escalated else ""

        color = {
            "critical": "FF0000",
            "warning": "FFA500",
            "info": "0078D7",
        }.get(alert.anomaly.severity, "808080")

        card = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": color,
            "summary": f"{prefix}Data Anomaly Detected",
            "sections": [
                {
                    "activityTitle": f"{prefix}Data Anomaly: {alert.anomaly.anomaly_type.value}",
                    "activitySubtitle": alert.anomaly.table_id,
                    "facts": [
                        {"name": "Table", "value": alert.anomaly.table_id},
                        {"name": "Type", "value": alert.anomaly.anomaly_type.value},
                        {"name": "Severity", "value": alert.anomaly.severity},
                        {"name": "Detected", "value": alert.anomaly.detected_at.strftime('%Y-%m-%d %H:%M:%S')},
                    ],
                    "text": alert.anomaly.description,
                }
            ],
        }

        self.sent_messages.append(card)

        # In production, would post to Teams webhook
        logger.info(f"Would send to Teams: {card}")

        return True


class AlertingEngine:
    """Alert management and routing."""

    def __init__(self, config=None, default_channels: Optional[List[str]] = None):
        self.channels: Dict[str, AlertChannel] = {}
        self.rules: List[AlertRule] = []
        self._alerts: Dict[str, Alert] = {}
        self._dedup_window: Dict[str, datetime] = {}
        self.config = config
        # Support both config object and direct parameter
        if config and hasattr(config, 'channels'):
            self.default_channels = config.channels or []
        else:
            self.default_channels = default_channels or []

    def add_channel(self, name: str, channel: AlertChannel) -> None:
        """Register an alert channel."""
        self.channels[name] = channel
        logger.info(f"Registered alert channel: {name}")

    def remove_channel(self, name: str) -> None:
        """Remove an alert channel."""
        if name in self.channels:
            del self.channels[name]
            logger.info(f"Removed alert channel: {name}")

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alerting rule."""
        self.rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")

    async def create_alert(self, anomaly: Anomaly) -> Alert:
        """Create an alert from an anomaly (test-friendly API)."""
        return await self.process_anomaly(anomaly)

    async def process_anomaly(self, anomaly: Anomaly) -> Optional[Alert]:
        """Process an anomaly and send alerts."""
        # Check deduplication - use config window or default to 60 minutes
        dedup_minutes = 60
        if self.config and hasattr(self.config, 'dedup_window_minutes'):
            dedup_minutes = self.config.dedup_window_minutes

        # Use anomaly_id in dedup key to allow different alerts for same table/type
        dedup_key = f"{anomaly.table_id}:{anomaly.anomaly_type.value}:{anomaly.anomaly_id}"
        if dedup_key in self._dedup_window:
            last_alert = self._dedup_window[dedup_key]
            if (datetime.now() - last_alert).seconds < dedup_minutes * 60:
                logger.debug(f"Deduplicating alert for {dedup_key}")
                # For create_alert API, still return the existing alert
                for alert in self._alerts.values():
                    if alert.anomaly.anomaly_id == anomaly.anomaly_id:
                        return alert
                return None

        # Find matching rules
        matching_rules = [
            rule for rule in self.rules if rule.matches(anomaly)
        ]

        # Create alert
        alert = Alert(
            alert_id=generate_id(),
            anomaly=anomaly,
            created_at=datetime.now(),
            status="active",
        )

        # Get channels to notify
        channels_to_notify: set = set()

        if matching_rules:
            for rule in matching_rules:
                channels_to_notify.update(rule.get_channels(anomaly))
        else:
            channels_to_notify.update(self.default_channels)

        # Route to channels
        for channel_name in channels_to_notify:
            if channel_name in self.channels:
                try:
                    await self.channels[channel_name].send(alert)
                except Exception as e:
                    logger.error(f"Failed to send alert to {channel_name}: {e}")

        # Store alert
        self._alerts[alert.alert_id] = alert
        self._dedup_window[dedup_key] = datetime.now()

        # Check escalation
        for rule in matching_rules:
            if rule.escalation_policy:
                asyncio.create_task(
                    self._escalation_timer(alert, rule.escalation_policy)
                )

        return alert

    async def _escalation_timer(
        self, alert: Alert, policy: EscalationPolicy
    ) -> None:
        """Wait and escalate if not acknowledged."""
        await asyncio.sleep(policy.escalate_after_minutes * 60)

        # Check if still active
        current_alert = self._alerts.get(alert.alert_id)

        if current_alert and current_alert.status == "active":
            # Escalate
            for channel_name in policy.escalate_to:
                if channel_name in self.channels:
                    try:
                        await self.channels[channel_name].send(alert, escalated=True)
                    except Exception as e:
                        logger.error(f"Failed to escalate to {channel_name}: {e}")

    async def acknowledge_alert(
        self, alert_id: str, acknowledged_by: str = None, user: str = None, notes: Optional[str] = None
    ) -> Optional[Alert]:
        """Acknowledge an alert."""
        if alert_id not in self._alerts:
            return None

        # Support both parameter names
        ack_user = acknowledged_by or user

        alert = self._alerts[alert_id]
        alert.status = "acknowledged"
        alert.acknowledged_by = ack_user
        alert.acknowledged_at = datetime.now()

        logger.info(f"Alert {alert_id} acknowledged by {ack_user}")
        return alert

    async def resolve_alert(
        self, alert_id: str, resolution_notes: Optional[str] = None, notes: Optional[str] = None
    ) -> Optional[Alert]:
        """Resolve an alert."""
        if alert_id not in self._alerts:
            return None

        alert = self._alerts[alert_id]
        alert.status = "resolved"
        alert.resolved_at = datetime.now()
        alert.resolution_notes = resolution_notes or notes

        logger.info(f"Alert {alert_id} resolved")
        return alert

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get an alert by ID."""
        return self._alerts.get(alert_id)

    async def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return [
            alert for alert in self._alerts.values()
            if alert.status == "active"
        ]

    def get_alerts_by_table(self, table_id: str) -> List[Alert]:
        """Get alerts for a specific table."""
        return [
            alert for alert in self._alerts.values()
            if alert.anomaly.table_id == table_id
        ]


# Common alert rules
CRITICAL_ALERT_RULE = AlertRule(
    name="critical_alerts",
    conditions={
        "severities": ["critical"],
    },
    channels=["slack", "pagerduty"],
    escalation_policy=EscalationPolicy(
        escalate_after_minutes=15,
        escalate_to=["pagerduty"],
    ),
)

SCHEMA_CHANGE_RULE = AlertRule(
    name="schema_changes",
    conditions={
        "anomaly_types": [AnomalyType.SCHEMA],
    },
    channels=["slack"],
)

FRESHNESS_ALERT_RULE = AlertRule(
    name="freshness_alerts",
    conditions={
        "anomaly_types": [AnomalyType.FRESHNESS],
        "severities": ["warning", "critical"],
    },
    channels=["slack"],
)
