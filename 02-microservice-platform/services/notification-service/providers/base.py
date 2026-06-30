from abc import ABC, abstractmethod
from typing import Any

from models import DeliveryResult


class NotificationProvider(ABC):
    """Base class for notification providers."""

    @abstractmethod
    async def send(self, recipient: str, content: Any) -> DeliveryResult:
        """Send a notification to the recipient."""
        pass

    @abstractmethod
    async def check_status(self, message_id: str) -> DeliveryResult:
        """Check the delivery status of a sent notification."""
        pass

    @abstractmethod
    def validate_recipient(self, recipient: str) -> bool:
        """Validate the recipient address/number."""
        pass
