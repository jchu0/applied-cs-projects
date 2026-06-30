import logging
import re
from typing import Optional

from config import settings
from models import DeliveryResult, DeliveryStatus, SMSContent
from providers.base import NotificationProvider

logger = logging.getLogger(__name__)


class TwilioProvider(NotificationProvider):
    """Twilio SMS provider."""

    def __init__(self):
        self.account_sid = settings.twilio_account_sid
        self.auth_token = settings.twilio_auth_token
        self.from_number = settings.twilio_phone_number
        self.client = None

        if self.account_sid and self.auth_token:
            try:
                from twilio.rest import Client
                self.client = Client(self.account_sid, self.auth_token)
            except ImportError:
                logger.warning("Twilio library not installed")

    async def send(self, recipient: str, content: SMSContent) -> DeliveryResult:
        """Send an SMS via Twilio."""
        if not self.client:
            logger.warning("Twilio not configured, skipping SMS send")
            return DeliveryResult(
                status=DeliveryStatus.FAILED,
                error_message="Twilio not configured"
            )

        try:
            message = self.client.messages.create(
                body=content.body,
                from_=content.from_number or self.from_number,
                to=recipient
            )

            return DeliveryResult(
                status=DeliveryStatus.SENT,
                provider_message_id=message.sid
            )

        except Exception as e:
            logger.error(f"Failed to send SMS: {e}")
            return DeliveryResult(
                status=DeliveryStatus.FAILED,
                error_message=str(e)
            )

    async def check_status(self, message_id: str) -> DeliveryResult:
        """Check SMS delivery status via Twilio."""
        if not self.client:
            return DeliveryResult(
                provider_message_id=message_id,
                status=DeliveryStatus.FAILED,
                error_message="Twilio not configured"
            )

        try:
            message = self.client.messages(message_id).fetch()

            status_map = {
                "queued": DeliveryStatus.PENDING,
                "sending": DeliveryStatus.PENDING,
                "sent": DeliveryStatus.SENT,
                "delivered": DeliveryStatus.DELIVERED,
                "failed": DeliveryStatus.FAILED,
                "undelivered": DeliveryStatus.FAILED
            }

            return DeliveryResult(
                provider_message_id=message_id,
                status=status_map.get(message.status, DeliveryStatus.PENDING)
            )

        except Exception as e:
            return DeliveryResult(
                provider_message_id=message_id,
                status=DeliveryStatus.FAILED,
                error_message=str(e)
            )

    def validate_recipient(self, recipient: str) -> bool:
        """Validate phone number format (E.164)."""
        pattern = r'^\+[1-9]\d{1,14}$'
        return bool(re.match(pattern, recipient))
