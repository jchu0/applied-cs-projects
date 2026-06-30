import logging
import re
from typing import Optional

from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Content, Attachment, FileContent, FileName, FileType

from config import settings
from models import DeliveryResult, DeliveryStatus, EmailContent
from providers.base import NotificationProvider

logger = logging.getLogger(__name__)


class SendGridProvider(NotificationProvider):
    """SendGrid email provider."""

    def __init__(self):
        self.client = SendGridAPIClient(settings.sendgrid_api_key) if settings.sendgrid_api_key else None
        self.from_email = settings.sendgrid_from_email
        self.from_name = settings.sendgrid_from_name

    async def send(self, recipient: str, content: EmailContent) -> DeliveryResult:
        """Send an email via SendGrid."""
        if not self.client:
            logger.warning("SendGrid API key not configured, skipping email send")
            return DeliveryResult(
                status=DeliveryStatus.FAILED,
                error_message="SendGrid not configured"
            )

        try:
            message = Mail(
                from_email=Email(content.from_email or self.from_email, content.from_name or self.from_name),
                to_emails=To(recipient),
                subject=content.subject,
                html_content=Content("text/html", content.html_body)
            )

            # Add plain text version
            if content.text_body:
                message.add_content(Content("text/plain", content.text_body))

            # Add attachments
            for attachment in content.attachments:
                att = Attachment(
                    FileContent(attachment.get("content", "")),
                    FileName(attachment.get("filename", "attachment")),
                    FileType(attachment.get("type", "application/octet-stream"))
                )
                message.add_attachment(att)

            response = self.client.send(message)

            if response.status_code in (200, 202):
                # Extract message ID from headers
                message_id = response.headers.get("X-Message-Id", "")
                return DeliveryResult(
                    status=DeliveryStatus.SENT,
                    provider_message_id=message_id
                )
            else:
                return DeliveryResult(
                    status=DeliveryStatus.FAILED,
                    error_message=f"SendGrid returned status {response.status_code}"
                )

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return DeliveryResult(
                status=DeliveryStatus.FAILED,
                error_message=str(e)
            )

    async def check_status(self, message_id: str) -> DeliveryResult:
        """Check email delivery status via SendGrid webhooks (not implemented here)."""
        # SendGrid delivery status is typically tracked via webhooks
        return DeliveryResult(
            provider_message_id=message_id,
            status=DeliveryStatus.SENT
        )

    def validate_recipient(self, recipient: str) -> bool:
        """Validate email address format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, recipient))


class SMTPProvider(NotificationProvider):
    """SMTP email provider (fallback)."""

    def __init__(self, host: str = "localhost", port: int = 587):
        self.host = host
        self.port = port

    async def send(self, recipient: str, content: EmailContent) -> DeliveryResult:
        """Send email via SMTP."""
        # Implementation would use aiosmtplib
        return DeliveryResult(
            status=DeliveryStatus.FAILED,
            error_message="SMTP not implemented"
        )

    async def check_status(self, message_id: str) -> DeliveryResult:
        """SMTP doesn't support status checking."""
        return DeliveryResult(
            provider_message_id=message_id,
            status=DeliveryStatus.SENT
        )

    def validate_recipient(self, recipient: str) -> bool:
        """Validate email address format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, recipient))
