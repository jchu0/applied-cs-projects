import asyncio
import logging
import signal
import sys
from concurrent import futures

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from config import settings
from service import NotificationService
from models import Notification, Channel, DeliveryStatus, NotificationPreferences

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NotificationServicer:
    """gRPC servicer for the Notification Service."""

    def __init__(self, service: NotificationService):
        self.service = service

    async def Send(self, request, context):
        """Send a single notification."""
        notification = Notification(
            tenant_id=request.tenant_id,
            recipient_id=request.recipient_id,
            template_id=request.template_id,
            channel=Channel(request.channel) if request.channel else Channel.EMAIL,
            variables=dict(request.variables),
            priority=request.priority or 5,
            metadata=dict(request.metadata) if request.metadata else {}
        )

        result = await self.service.send(notification)

        return {
            "notification_id": result.notification_id,
            "status": result.status.value
        }

    async def SendBatch(self, request, context):
        """Send multiple notifications."""
        notifications = []
        for req in request.notifications:
            notifications.append(Notification(
                tenant_id=req.tenant_id,
                recipient_id=req.recipient_id,
                template_id=req.template_id,
                channel=Channel(req.channel) if req.channel else Channel.EMAIL,
                variables=dict(req.variables),
                priority=req.priority or 5,
                metadata=dict(req.metadata) if req.metadata else {}
            ))

        results = await self.service.send_batch(notifications)

        success_count = sum(1 for r in results if r.status == DeliveryStatus.SENT)
        failure_count = len(results) - success_count

        return {
            "results": [
                {"notification_id": r.notification_id, "status": r.status.value}
                for r in results
            ],
            "success_count": success_count,
            "failure_count": failure_count
        }

    async def GetNotification(self, request, context):
        """Get notification status."""
        result = await self.service.get_notification_status(request.notification_id)

        if not result:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Notification not found")
            return {}

        return {
            "notification": {
                "id": request.notification_id,
                "status": result.status.value
            }
        }

    async def UpdatePreferences(self, request, context):
        """Update user notification preferences."""
        preferences = NotificationPreferences(
            user_id=request.user_id,
            tenant_id=request.tenant_id,
            email_enabled=request.preferences.email_enabled,
            sms_enabled=request.preferences.sms_enabled,
            push_enabled=request.preferences.push_enabled,
            in_app_enabled=request.preferences.in_app_enabled,
            unsubscribed_categories=list(request.preferences.unsubscribed_categories),
            quiet_hours_start=request.preferences.quiet_hours_start,
            quiet_hours_end=request.preferences.quiet_hours_end,
            timezone=request.preferences.timezone or "UTC"
        )

        await self.service.update_preferences(
            request.user_id,
            request.tenant_id,
            preferences
        )

        return {"preferences": request.preferences}


async def serve():
    """Start the gRPC server."""
    # Initialize the notification service
    service = NotificationService()
    await service.initialize()

    # Create gRPC server
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ]
    )

    # Register health check
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    health_servicer.set(
        "notification-service",
        health_pb2.HealthCheckResponse.SERVING
    )

    # Note: In production, register the generated protobuf service
    # notification_pb2_grpc.add_NotificationServiceServicer_to_server(
    #     NotificationServicer(service), server
    # )

    servicer = NotificationServicer(service)
    _ = servicer  # Placeholder until proto generation

    # Add insecure port
    listen_addr = f"[::]:{settings.grpc_port}"
    server.add_insecure_port(listen_addr)

    # Start server
    await server.start()
    logger.info(f"Notification service started on {listen_addr}")

    # Handle shutdown
    async def shutdown():
        logger.info("Shutting down notification service...")
        health_servicer.set(
            "notification-service",
            health_pb2.HealthCheckResponse.NOT_SERVING
        )
        await server.stop(5)
        await service.close()
        logger.info("Notification service stopped")

    # Set up signal handlers
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown()))

    # Wait for termination
    await server.wait_for_termination()


def main():
    """Main entry point."""
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
