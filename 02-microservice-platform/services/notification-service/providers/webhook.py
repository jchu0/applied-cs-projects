import logging
import json
import hashlib
import hmac
from typing import Optional
from urllib.parse import urlparse

import aiohttp

from config import settings
from models import DeliveryResult, DeliveryStatus, WebhookContent
from providers.base import NotificationProvider

logger = logging.getLogger(__name__)


class WebhookProvider(NotificationProvider):
    """Webhook delivery provider."""

    def __init__(self):
        self.timeout = aiohttp.ClientTimeout(total=30)

    async def send(self, recipient: str, content: WebhookContent) -> DeliveryResult:
        """Deliver webhook to the specified URL."""
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                # Prepare headers
                headers = {
                    "Content-Type": "application/json",
                    "User-Agent": "Microservices-Notification-Service/1.0",
                    **content.headers
                }

                # Make request
                method = content.method.upper()
                if method == "POST":
                    async with session.post(
                        content.url,
                        json=content.payload,
                        headers=headers
                    ) as response:
                        if response.status in (200, 201, 202, 204):
                            return DeliveryResult(
                                status=DeliveryStatus.DELIVERED,
                                provider_message_id=str(response.headers.get("X-Request-Id", ""))
                            )
                        else:
                            body = await response.text()
                            return DeliveryResult(
                                status=DeliveryStatus.FAILED,
                                error_message=f"Webhook returned {response.status}: {body[:200]}"
                            )
                elif method == "GET":
                    async with session.get(
                        content.url,
                        headers=headers
                    ) as response:
                        if response.status == 200:
                            return DeliveryResult(status=DeliveryStatus.DELIVERED)
                        else:
                            return DeliveryResult(
                                status=DeliveryStatus.FAILED,
                                error_message=f"Webhook returned {response.status}"
                            )
                else:
                    return DeliveryResult(
                        status=DeliveryStatus.FAILED,
                        error_message=f"Unsupported method: {method}"
                    )

        except aiohttp.ClientError as e:
            logger.error(f"Webhook delivery failed: {e}")
            return DeliveryResult(
                status=DeliveryStatus.FAILED,
                error_message=str(e)
            )
        except Exception as e:
            logger.error(f"Unexpected error in webhook delivery: {e}")
            return DeliveryResult(
                status=DeliveryStatus.FAILED,
                error_message=str(e)
            )

    async def check_status(self, message_id: str) -> DeliveryResult:
        """Webhooks don't support status checking."""
        return DeliveryResult(
            provider_message_id=message_id,
            status=DeliveryStatus.DELIVERED
        )

    def validate_recipient(self, recipient: str) -> bool:
        """Validate webhook URL."""
        try:
            result = urlparse(recipient)
            return all([result.scheme in ("http", "https"), result.netloc])
        except Exception:
            return False

    @staticmethod
    def sign_payload(payload: dict, secret: str) -> str:
        """Generate HMAC signature for webhook payload."""
        payload_str = json.dumps(payload, sort_keys=True)
        signature = hmac.new(
            secret.encode(),
            payload_str.encode(),
            hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"
