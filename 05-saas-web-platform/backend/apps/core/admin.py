"""
Admin configuration for core app.
"""
from django.contrib import admin
from .models import AuditLog


@admin.register(AuditLog)
class AuditLogAdmin(admin.ModelAdmin):
    list_display = ['action', 'resource_type', 'user', 'tenant', 'created_at']
    list_filter = ['action', 'resource_type', 'created_at']
    search_fields = ['user__email', 'resource_id']
    readonly_fields = ['id', 'created_at', 'updated_at']
