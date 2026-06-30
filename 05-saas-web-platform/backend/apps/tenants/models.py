"""
Tenant models for multi-tenancy support.
"""
import uuid
from django.db import models
from apps.users.models import User


class Tenant(models.Model):
    """Organization/workspace in the SaaS platform."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    slug = models.SlugField(unique=True)
    logo_url = models.URLField(blank=True)

    # Settings
    settings = models.JSONField(default=dict)

    # Billing
    stripe_customer_id = models.CharField(max_length=100, blank=True)

    # Status
    is_active = models.BooleanField(default=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'tenants'
        ordering = ['name']

    def __str__(self):
        return self.name


class TenantMembership(models.Model):
    """Membership linking users to tenants with roles."""

    class Role(models.TextChoices):
        OWNER = 'owner', 'Owner'
        ADMIN = 'admin', 'Admin'
        MEMBER = 'member', 'Member'
        VIEWER = 'viewer', 'Viewer'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(Tenant, on_delete=models.CASCADE, related_name='memberships')
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='tenant_memberships')
    role = models.CharField(max_length=20, choices=Role.choices, default=Role.MEMBER)

    # Invitation tracking
    invited_by = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True, related_name='invitations_sent'
    )
    invited_at = models.DateTimeField(null=True, blank=True)
    accepted_at = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'tenant_memberships'
        unique_together = ['tenant', 'user']

    def __str__(self):
        return f'{self.user.email} - {self.tenant.name} ({self.role})'


class Invitation(models.Model):
    """Pending invitation to join a tenant."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(Tenant, on_delete=models.CASCADE, related_name='invitations')
    email = models.EmailField()
    role = models.CharField(max_length=20, choices=TenantMembership.Role.choices, default=TenantMembership.Role.MEMBER)
    token = models.CharField(max_length=100, unique=True)
    invited_by = models.ForeignKey(User, on_delete=models.CASCADE)
    expires_at = models.DateTimeField()
    accepted_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'invitations'

    def __str__(self):
        return f'{self.email} -> {self.tenant.name}'
