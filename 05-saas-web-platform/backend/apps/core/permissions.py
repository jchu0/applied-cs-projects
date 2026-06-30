"""
Custom permissions for the SaaS platform.
"""
from rest_framework import permissions
from apps.tenants.models import TenantMembership


class IsTenantMember(permissions.BasePermission):
    """Check if user is a member of the tenant."""

    def has_permission(self, request, view):
        if not request.user.is_authenticated:
            return False

        tenant = getattr(request, 'tenant', None)
        if not tenant:
            return False

        return TenantMembership.objects.filter(
            tenant=tenant,
            user=request.user
        ).exists()


class IsTenantAdmin(permissions.BasePermission):
    """Check if user is an admin of the tenant."""

    def has_permission(self, request, view):
        if not request.user.is_authenticated:
            return False

        tenant = getattr(request, 'tenant', None)
        if not tenant:
            return False

        return TenantMembership.objects.filter(
            tenant=tenant,
            user=request.user,
            role__in=[TenantMembership.Role.OWNER, TenantMembership.Role.ADMIN]
        ).exists()


class IsTenantOwner(permissions.BasePermission):
    """Check if user is the owner of the tenant."""

    def has_permission(self, request, view):
        if not request.user.is_authenticated:
            return False

        tenant = getattr(request, 'tenant', None)
        if not tenant:
            return False

        return TenantMembership.objects.filter(
            tenant=tenant,
            user=request.user,
            role=TenantMembership.Role.OWNER
        ).exists()


class HasAPIKeyScope(permissions.BasePermission):
    """Check if API key has required scope."""

    required_scope = None

    def has_permission(self, request, view):
        if not request.auth:
            return False

        # Check if auth is an API key
        from apps.users.models import APIKey
        if not isinstance(request.auth, APIKey):
            return True  # JWT auth, allow

        # Check scopes
        if not self.required_scope:
            return True

        return self.required_scope in request.auth.scopes
