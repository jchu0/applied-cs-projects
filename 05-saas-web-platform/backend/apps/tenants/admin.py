"""
Admin configuration for tenants app.
"""
from django.contrib import admin
from .models import Tenant, TenantMembership, Invitation


@admin.register(Tenant)
class TenantAdmin(admin.ModelAdmin):
    list_display = ['name', 'slug', 'is_active', 'created_at']
    list_filter = ['is_active', 'created_at']
    search_fields = ['name', 'slug']
    prepopulated_fields = {'slug': ('name',)}


@admin.register(TenantMembership)
class TenantMembershipAdmin(admin.ModelAdmin):
    list_display = ['user', 'tenant', 'role', 'created_at']
    list_filter = ['role', 'created_at']
    search_fields = ['user__email', 'tenant__name']


@admin.register(Invitation)
class InvitationAdmin(admin.ModelAdmin):
    list_display = ['email', 'tenant', 'role', 'invited_by', 'expires_at', 'accepted_at']
    list_filter = ['role', 'created_at']
    search_fields = ['email', 'tenant__name']
