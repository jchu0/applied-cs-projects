"""
Audit logging service for tracking user actions.
"""
from django.utils import timezone
from .models import AuditLog
from .middleware import get_current_request, get_current_user, get_current_tenant


class AuditService:
    """Service for creating audit log entries."""

    @staticmethod
    def log(
        action: str,
        resource_type: str,
        resource_id: str,
        changes: dict = None,
        user=None,
        tenant=None,
        ip_address: str = None,
        user_agent: str = None,
    ) -> AuditLog:
        """Create an audit log entry."""
        # Get from context if not provided
        if user is None:
            user = get_current_user()
        if tenant is None:
            tenant = get_current_tenant()

        # Get request info
        request = get_current_request()
        if request:
            if ip_address is None:
                ip_address = get_client_ip(request)
            if user_agent is None:
                user_agent = request.META.get('HTTP_USER_AGENT', '')

        return AuditLog.objects.create(
            tenant=tenant,
            user=user,
            action=action,
            resource_type=resource_type,
            resource_id=str(resource_id),
            changes=changes or {},
            ip_address=ip_address,
            user_agent=user_agent,
        )

    @staticmethod
    def log_create(resource_type: str, resource_id: str, data: dict = None, **kwargs):
        """Log a create action."""
        return AuditService.log(
            AuditLog.Action.CREATE,
            resource_type,
            resource_id,
            changes={'new': data} if data else None,
            **kwargs
        )

    @staticmethod
    def log_update(resource_type: str, resource_id: str, old_data: dict = None, new_data: dict = None, **kwargs):
        """Log an update action."""
        changes = {}
        if old_data:
            changes['old'] = old_data
        if new_data:
            changes['new'] = new_data
        return AuditService.log(
            AuditLog.Action.UPDATE,
            resource_type,
            resource_id,
            changes=changes or None,
            **kwargs
        )

    @staticmethod
    def log_delete(resource_type: str, resource_id: str, data: dict = None, **kwargs):
        """Log a delete action."""
        return AuditService.log(
            AuditLog.Action.DELETE,
            resource_type,
            resource_id,
            changes={'deleted': data} if data else None,
            **kwargs
        )

    @staticmethod
    def log_login(user, **kwargs):
        """Log a login action."""
        return AuditService.log(
            AuditLog.Action.LOGIN,
            'user',
            str(user.id),
            user=user,
            **kwargs
        )

    @staticmethod
    def log_logout(user, **kwargs):
        """Log a logout action."""
        return AuditService.log(
            AuditLog.Action.LOGOUT,
            'user',
            str(user.id),
            user=user,
            **kwargs
        )


def get_client_ip(request) -> str:
    """Get client IP address from request."""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0].strip()
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip


# Decorator for automatic audit logging
def audit_action(action: str, resource_type: str):
    """Decorator to automatically log actions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            # Try to get resource ID from result
            resource_id = ''
            if hasattr(result, 'id'):
                resource_id = str(result.id)
            elif isinstance(result, dict) and 'id' in result:
                resource_id = str(result['id'])

            AuditService.log(action, resource_type, resource_id)
            return result
        return wrapper
    return decorator
