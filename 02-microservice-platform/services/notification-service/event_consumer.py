import asyncio
import json
import logging
from typing import Dict, Any

import nats
from nats.js.api import ConsumerConfig, DeliverPolicy

from config import settings
from models import Notification, Channel
from service import NotificationService

logger = logging.getLogger(__name__)


class EventConsumer:
    """Consumes events from NATS JetStream and triggers notifications."""

    def __init__(self, notification_service: NotificationService):
        self.notification_service = notification_service
        self.nc = None
        self.js = None
        self.subscriptions = []

    async def connect(self):
        """Connect to NATS and set up JetStream."""
        self.nc = await nats.connect(settings.nats_url)
        self.js = self.nc.jetstream()
        logger.info(f"Connected to NATS at {settings.nats_url}")

    async def subscribe(self):
        """Subscribe to relevant events."""
        # Subscribe to all events we care about
        event_handlers = {
            "events.user.created": self.handle_user_created,
            "events.auth.registered": self.handle_user_registered,
            "events.auth.login": self.handle_user_login,
            "events.billing.subscription.created": self.handle_subscription_created,
            "events.billing.subscription.canceled": self.handle_subscription_canceled,
            "events.billing.payment.succeeded": self.handle_payment_succeeded,
            "events.billing.payment.failed": self.handle_payment_failed,
            "events.billing.invoice.paid": self.handle_invoice_paid,
        }

        for subject, handler in event_handlers.items():
            consumer_name = f"notification-{subject.replace('.', '-')}"

            try:
                sub = await self.js.subscribe(
                    subject,
                    durable=consumer_name,
                    cb=self._create_callback(handler),
                    manual_ack=True,
                )
                self.subscriptions.append(sub)
                logger.info(f"Subscribed to {subject}")
            except Exception as e:
                logger.error(f"Failed to subscribe to {subject}: {e}")

    def _create_callback(self, handler):
        """Create a callback that wraps the handler with error handling."""
        async def callback(msg):
            try:
                event = json.loads(msg.data.decode())
                await handler(event)
                await msg.ack()
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode event: {e}")
                await msg.nak()
            except Exception as e:
                logger.error(f"Failed to handle event: {e}")
                await msg.nak()

        return callback

    async def handle_user_created(self, event: Dict[str, Any]):
        """Handle user.created event - send welcome email."""
        data = event.get("data", {})
        tenant_id = event.get("tenant_id", "")
        user_id = data.get("user_id", "")
        email = data.get("email", "")

        logger.info(f"Handling user.created for {email}")

        notification = Notification(
            tenant_id=tenant_id,
            recipient_id=user_id,
            template_id="welcome_email",
            channel=Channel.EMAIL,
            variables={
                "first_name": email.split("@")[0],
                "email": email,
            },
            metadata={"event_type": "user.created"}
        )

        result = await self.notification_service.send(notification)
        logger.info(f"Welcome email result: {result.status}")

    async def handle_user_registered(self, event: Dict[str, Any]):
        """Handle auth.registered event - send welcome email."""
        # Similar to user.created
        await self.handle_user_created(event)

    async def handle_user_login(self, event: Dict[str, Any]):
        """Handle auth.login event - could send security notification."""
        data = event.get("data", {})
        logger.info(f"User logged in: {data.get('user_id')}")
        # Could send "new login detected" notification for security

    async def handle_subscription_created(self, event: Dict[str, Any]):
        """Handle billing.subscription.created event."""
        data = event.get("data", {})
        tenant_id = event.get("tenant_id", "")

        logger.info(f"Handling subscription.created for tenant {tenant_id}")

        notification = Notification(
            tenant_id=tenant_id,
            recipient_id=tenant_id,  # Would need to look up user
            template_id="subscription_created",
            channel=Channel.EMAIL,
            variables={
                "plan_name": data.get("plan_id", "Unknown"),
                "subscription_id": data.get("subscription_id", ""),
            },
            metadata={"event_type": "billing.subscription.created"}
        )

        result = await self.notification_service.send(notification)
        logger.info(f"Subscription confirmation result: {result.status}")

    async def handle_subscription_canceled(self, event: Dict[str, Any]):
        """Handle billing.subscription.canceled event."""
        data = event.get("data", {})
        tenant_id = event.get("tenant_id", "")

        logger.info(f"Handling subscription.canceled for tenant {tenant_id}")
        # Send cancellation confirmation email

    async def handle_payment_succeeded(self, event: Dict[str, Any]):
        """Handle billing.payment.succeeded event."""
        data = event.get("data", {})
        tenant_id = event.get("tenant_id", "")

        logger.info(f"Handling payment.succeeded for tenant {tenant_id}")

        notification = Notification(
            tenant_id=tenant_id,
            recipient_id=tenant_id,
            template_id="invoice_paid",
            channel=Channel.EMAIL,
            variables={
                "amount": str(data.get("amount", 0) / 100),  # Convert cents to dollars
                "invoice_number": data.get("invoice_id", ""),
            },
            metadata={"event_type": "billing.payment.succeeded"}
        )

        result = await self.notification_service.send(notification)
        logger.info(f"Payment receipt result: {result.status}")

    async def handle_payment_failed(self, event: Dict[str, Any]):
        """Handle billing.payment.failed event."""
        data = event.get("data", {})
        tenant_id = event.get("tenant_id", "")

        logger.info(f"Handling payment.failed for tenant {tenant_id}")
        # Send payment failed notification - high priority

    async def handle_invoice_paid(self, event: Dict[str, Any]):
        """Handle billing.invoice.paid event."""
        # Similar to payment.succeeded
        await self.handle_payment_succeeded(event)

    async def close(self):
        """Close subscriptions and connection."""
        for sub in self.subscriptions:
            await sub.unsubscribe()

        if self.nc:
            await self.nc.close()

        logger.info("Event consumer closed")


async def run_consumer():
    """Run the event consumer."""
    # Initialize notification service
    notification_service = NotificationService()
    await notification_service.initialize()

    # Create and start consumer
    consumer = EventConsumer(notification_service)
    await consumer.connect()
    await consumer.subscribe()

    logger.info("Event consumer running...")

    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        await consumer.close()
        await notification_service.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(run_consumer())
