import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional

import redis.asyncio as redis

from config import settings
from models import (
    Notification, Channel, DeliveryStatus, DeliveryResult,
    NotificationPreferences, EmailContent
)
from providers import SendGridProvider, TwilioProvider, WebhookProvider
from template_engine import TemplateEngine

logger = logging.getLogger(__name__)


class NotificationService:
    """Core notification service handling delivery across all channels."""

    def __init__(self):
        self.providers = {
            Channel.EMAIL: SendGridProvider(),
            Channel.SMS: TwilioProvider(),
            Channel.WEBHOOK: WebhookProvider(),
        }
        self.template_engine = TemplateEngine()
        self.redis: Optional[redis.Redis] = None

    async def initialize(self):
        """Initialize async resources."""
        self.redis = redis.from_url(settings.redis_url)
        logger.info("NotificationService initialized")

    async def close(self):
        """Close async resources."""
        if self.redis:
            await self.redis.close()

    async def send(self, notification: Notification) -> DeliveryResult:
        """Send a notification through the appropriate channel."""
        # Check user preferences
        if not await self._check_preferences(notification):
            return DeliveryResult(
                notification_id=notification.id,
                status=DeliveryStatus.OPTED_OUT
            )

        # Get the template and render content
        template = self.template_engine.get_template(notification.template_id)
        if not template:
            return DeliveryResult(
                notification_id=notification.id,
                status=DeliveryStatus.FAILED,
                error_message=f"Template not found: {notification.template_id}"
            )

        # Get provider
        provider = self.providers.get(notification.channel)
        if not provider:
            return DeliveryResult(
                notification_id=notification.id,
                status=DeliveryStatus.FAILED,
                error_message=f"No provider for channel: {notification.channel}"
            )

        # Get recipient address
        recipient = await self._get_recipient_address(
            notification.recipient_id,
            notification.tenant_id,
            notification.channel
        )
        if not recipient:
            return DeliveryResult(
                notification_id=notification.id,
                status=DeliveryStatus.FAILED,
                error_message="Recipient address not found"
            )

        # Validate recipient
        if not provider.validate_recipient(recipient):
            return DeliveryResult(
                notification_id=notification.id,
                status=DeliveryStatus.FAILED,
                error_message="Invalid recipient address"
            )

        # Render content based on channel
        if notification.channel == Channel.EMAIL:
            content = self.template_engine.render_email(
                notification.template_id,
                notification.variables,
                recipient
            )
        elif notification.channel == Channel.SMS:
            content = self.template_engine.render_sms(
                notification.template_id,
                notification.variables,
                recipient
            )
        else:
            # For other channels, render basic content
            content = self.template_engine.render(
                notification.template_id,
                notification.variables
            )

        if not content:
            return DeliveryResult(
                notification_id=notification.id,
                status=DeliveryStatus.FAILED,
                error_message="Failed to render template"
            )

        # Send through provider
        result = await provider.send(recipient, content)
        result.notification_id = notification.id

        # Track delivery
        await self._track_delivery(notification.id, result)

        return result

    async def send_batch(self, notifications: List[Notification]) -> List[DeliveryResult]:
        """Send multiple notifications concurrently."""
        tasks = [self.send(notification) for notification in notifications]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        delivery_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                delivery_results.append(DeliveryResult(
                    notification_id=notifications[i].id,
                    status=DeliveryStatus.FAILED,
                    error_message=str(result)
                ))
            else:
                delivery_results.append(result)

        return delivery_results

    async def _check_preferences(self, notification: Notification) -> bool:
        """Check if user has opted in for this notification type."""
        # Get preferences from Redis cache or database
        prefs = await self._get_preferences(
            notification.recipient_id,
            notification.tenant_id
        )

        if not prefs:
            return True  # Default to allowed if no preferences set

        # Check channel preference
        if notification.channel == Channel.EMAIL and not prefs.email_enabled:
            return False
        if notification.channel == Channel.SMS and not prefs.sms_enabled:
            return False
        if notification.channel == Channel.PUSH and not prefs.push_enabled:
            return False
        if notification.channel == Channel.IN_APP and not prefs.in_app_enabled:
            return False

        # Check category unsubscription
        category = notification.metadata.get("category", "")
        if category in prefs.unsubscribed_categories:
            return False

        # Check quiet hours
        if prefs.quiet_hours_start and prefs.quiet_hours_end:
            if self._in_quiet_hours(prefs):
                # Queue for later delivery instead of blocking
                notification.metadata["deferred_for_quiet_hours"] = "true"

        return True

    async def _get_preferences(self, user_id: str, tenant_id: str) -> Optional[NotificationPreferences]:
        """Get user notification preferences."""
        if not self.redis:
            return None

        key = f"notification_prefs:{tenant_id}:{user_id}"
        data = await self.redis.get(key)

        if not data:
            return None

        # Parse and return preferences
        import json
        try:
            prefs_dict = json.loads(data)
            return NotificationPreferences(**prefs_dict)
        except Exception:
            return None

    async def _get_recipient_address(self, user_id: str, tenant_id: str, channel: Channel) -> Optional[str]:
        """Get the recipient's address for the given channel."""
        if not self.redis:
            # For testing, return placeholder
            if channel == Channel.EMAIL:
                return f"{user_id}@example.com"
            return None

        # In production, this would look up the user's contact info
        # from the user service or a local cache
        key = f"user_contacts:{tenant_id}:{user_id}"
        data = await self.redis.hget(key, channel.value)

        if data:
            return data.decode() if isinstance(data, bytes) else data

        return None

    async def _track_delivery(self, notification_id: str, result: DeliveryResult):
        """Track notification delivery status."""
        if not self.redis:
            return

        key = f"notification_status:{notification_id}"
        status_data = {
            "status": result.status.value,
            "provider_message_id": result.provider_message_id or "",
            "error_message": result.error_message or "",
            "timestamp": result.timestamp.isoformat()
        }

        import json
        await self.redis.set(key, json.dumps(status_data), ex=86400 * 7)  # 7 days

    def _in_quiet_hours(self, prefs: NotificationPreferences) -> bool:
        """Check if current time is within user's quiet hours."""
        from datetime import datetime
        import pytz

        try:
            tz = pytz.timezone(prefs.timezone)
            now = datetime.now(tz)
            current_time = now.strftime("%H:%M")

            start = prefs.quiet_hours_start
            end = prefs.quiet_hours_end

            if start <= end:
                return start <= current_time <= end
            else:
                # Quiet hours span midnight
                return current_time >= start or current_time <= end

        except Exception:
            return False

    async def update_preferences(
        self,
        user_id: str,
        tenant_id: str,
        preferences: NotificationPreferences
    ):
        """Update user notification preferences."""
        if not self.redis:
            return

        key = f"notification_prefs:{tenant_id}:{user_id}"

        import json
        prefs_dict = {
            "user_id": preferences.user_id,
            "tenant_id": preferences.tenant_id,
            "email_enabled": preferences.email_enabled,
            "sms_enabled": preferences.sms_enabled,
            "push_enabled": preferences.push_enabled,
            "in_app_enabled": preferences.in_app_enabled,
            "unsubscribed_categories": preferences.unsubscribed_categories,
            "quiet_hours_start": preferences.quiet_hours_start,
            "quiet_hours_end": preferences.quiet_hours_end,
            "timezone": preferences.timezone
        }

        await self.redis.set(key, json.dumps(prefs_dict))

    async def get_notification_status(self, notification_id: str) -> Optional[DeliveryResult]:
        """Get the delivery status of a notification."""
        if not self.redis:
            return None

        key = f"notification_status:{notification_id}"
        data = await self.redis.get(key)

        if not data:
            return None

        import json
        try:
            status_dict = json.loads(data)
            return DeliveryResult(
                notification_id=notification_id,
                status=DeliveryStatus(status_dict["status"]),
                provider_message_id=status_dict.get("provider_message_id"),
                error_message=status_dict.get("error_message"),
                timestamp=datetime.fromisoformat(status_dict["timestamp"])
            )
        except Exception:
            return None
