"""
Core middleware for the SaaS platform.
"""
import threading
from django.utils.deprecation import MiddlewareMixin

# Thread-local storage for request context
_request_local = threading.local()


def get_current_request():
    """Get the current request from thread-local storage."""
    return getattr(_request_local, 'request', None)


def get_current_user():
    """Get the current user from thread-local storage."""
    request = get_current_request()
    if request and hasattr(request, 'user') and request.user.is_authenticated:
        return request.user
    return None


def get_current_tenant():
    """Get the current tenant from thread-local storage."""
    return getattr(_request_local, 'tenant', None)


class RequestContextMiddleware(MiddlewareMixin):
    """Store request in thread-local storage for access anywhere."""

    def process_request(self, request):
        _request_local.request = request

    def process_response(self, request, response):
        if hasattr(_request_local, 'request'):
            del _request_local.request
        return response


class TenantMiddleware(MiddlewareMixin):
    """
    Extract tenant from request and store in thread-local storage.
    Tenant can be specified via:
    - X-Tenant-ID header
    - tenant_id query parameter
    - subdomain
    """

    def process_request(self, request):
        tenant_id = None

        # Check header first
        tenant_id = request.headers.get('X-Tenant-ID')

        # Check query parameter
        if not tenant_id:
            tenant_id = request.GET.get('tenant_id')

        # Check subdomain
        if not tenant_id:
            host = request.get_host().split(':')[0]
            parts = host.split('.')
            if len(parts) > 2:
                # Assume subdomain is tenant slug
                from apps.tenants.models import Tenant
                try:
                    tenant = Tenant.objects.get(slug=parts[0])
                    tenant_id = str(tenant.id)
                except Tenant.DoesNotExist:
                    pass

        if tenant_id:
            from apps.tenants.models import Tenant
            try:
                tenant = Tenant.objects.get(id=tenant_id)
                _request_local.tenant = tenant
                request.tenant = tenant
            except (Tenant.DoesNotExist, ValueError):
                _request_local.tenant = None
                request.tenant = None
        else:
            _request_local.tenant = None
            request.tenant = None

    def process_response(self, request, response):
        if hasattr(_request_local, 'tenant'):
            del _request_local.tenant
        return response


class AuditLogMiddleware(MiddlewareMixin):
    """Log user actions for audit trail."""

    def process_response(self, request, response):
        # Only log successful modifications
        if request.method in ['POST', 'PUT', 'PATCH', 'DELETE'] and 200 <= response.status_code < 300:
            if hasattr(request, 'user') and request.user.is_authenticated:
                # Audit logging would be done in views/signals for more control
                pass
        return response
