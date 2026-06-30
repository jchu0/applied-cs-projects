from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
import uuid


class Channel(Enum):
    """Notification delivery channels."""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"
    IN_APP = "in_app"


class DeliveryStatus(Enum):
    """Notification delivery status."""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    BOUNCED = "bounced"
    OPTED_OUT = "opted_out"


class Priority(Enum):
    """Notification priority levels."""
    LOW = 1
    NORMAL = 5
    HIGH = 10
    URGENT = 15


@dataclass
class Notification:
    """Represents a notification to be sent."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = ""
    recipient_id: str = ""
    template_id: str = ""
    channel: Channel = Channel.EMAIL
    status: DeliveryStatus = DeliveryStatus.PENDING
    variables: Dict[str, str] = field(default_factory=dict)
    priority: int = Priority.NORMAL.value
    scheduled_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    retry_count: int = 0


@dataclass
class Template:
    """Notification template."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = ""
    name: str = ""
    channel: Channel = Channel.EMAIL
    subject: str = ""
    body: str = ""
    default_variables: Dict[str, str] = field(default_factory=dict)
    active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class NotificationPreferences:
    """User notification preferences."""
    user_id: str = ""
    tenant_id: str = ""
    email_enabled: bool = True
    sms_enabled: bool = True
    push_enabled: bool = True
    in_app_enabled: bool = True
    unsubscribed_categories: List[str] = field(default_factory=list)
    quiet_hours_start: Optional[str] = None  # HH:MM format
    quiet_hours_end: Optional[str] = None
    timezone: str = "UTC"


@dataclass
class DeliveryResult:
    """Result of a notification delivery attempt."""
    notification_id: str = ""
    status: DeliveryStatus = DeliveryStatus.PENDING
    provider_message_id: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class EmailContent:
    """Email content after template rendering."""
    to: str = ""
    subject: str = ""
    html_body: str = ""
    text_body: str = ""
    from_email: str = ""
    from_name: str = ""
    reply_to: Optional[str] = None
    attachments: List[Dict] = field(default_factory=list)


@dataclass
class SMSContent:
    """SMS content after template rendering."""
    to: str = ""
    body: str = ""
    from_number: str = ""


@dataclass
class PushContent:
    """Push notification content."""
    device_token: str = ""
    title: str = ""
    body: str = ""
    data: Dict[str, str] = field(default_factory=dict)
    badge: Optional[int] = None
    sound: str = "default"


@dataclass
class WebhookContent:
    """Webhook delivery content."""
    url: str = ""
    method: str = "POST"
    headers: Dict[str, str] = field(default_factory=dict)
    payload: Dict = field(default_factory=dict)
    timeout_seconds: int = 30
