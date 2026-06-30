"""
Core base models for the SaaS platform.
"""
import uuid
from django.db import models


class BaseModel(models.Model):
    """Abstract base model with common fields."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class TenantScopedModel(BaseModel):
    """Abstract base model for tenant-scoped data."""

    tenant = models.ForeignKey(
        'tenants.Tenant',
        on_delete=models.CASCADE,
        related_name='%(class)ss',
    )

    class Meta:
        abstract = True


class AuditLog(BaseModel):
    """Audit log for tracking user actions."""

    class Action(models.TextChoices):
        CREATE = 'create', 'Create'
        UPDATE = 'update', 'Update'
        DELETE = 'delete', 'Delete'
        LOGIN = 'login', 'Login'
        LOGOUT = 'logout', 'Logout'

    tenant = models.ForeignKey(
        'tenants.Tenant',
        on_delete=models.CASCADE,
        null=True,
        blank=True,
    )
    user = models.ForeignKey(
        'users.User',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )

    action = models.CharField(max_length=20, choices=Action.choices)
    resource_type = models.CharField(max_length=100)
    resource_id = models.CharField(max_length=100)
    changes = models.JSONField(default=dict)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)

    class Meta:
        db_table = 'audit_logs'
        indexes = [
            models.Index(fields=['tenant', 'created_at']),
            models.Index(fields=['user', 'created_at']),
            models.Index(fields=['resource_type', 'resource_id']),
        ]
        ordering = ['-created_at']

    def __str__(self):
        return f'{self.action} {self.resource_type} by {self.user}'
