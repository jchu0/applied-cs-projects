"""
Admin configuration for billing app.
"""
from django.contrib import admin
from .models import Plan, Subscription, Invoice, PaymentMethod, UsageRecord


@admin.register(Plan)
class PlanAdmin(admin.ModelAdmin):
    list_display = ['name', 'price_monthly', 'price_yearly', 'max_users', 'is_active', 'sort_order']
    list_filter = ['is_active']
    search_fields = ['name', 'slug']
    prepopulated_fields = {'slug': ('name',)}


@admin.register(Subscription)
class SubscriptionAdmin(admin.ModelAdmin):
    list_display = ['tenant', 'plan', 'status', 'billing_interval', 'current_period_end']
    list_filter = ['status', 'billing_interval', 'plan']
    search_fields = ['tenant__name']


@admin.register(Invoice)
class InvoiceAdmin(admin.ModelAdmin):
    list_display = ['number', 'tenant', 'status', 'total', 'due_date', 'paid_at']
    list_filter = ['status', 'created_at']
    search_fields = ['number', 'tenant__name']


@admin.register(PaymentMethod)
class PaymentMethodAdmin(admin.ModelAdmin):
    list_display = ['tenant', 'type', 'card_brand', 'card_last4', 'is_default']
    list_filter = ['type', 'is_default']
    search_fields = ['tenant__name']


@admin.register(UsageRecord)
class UsageRecordAdmin(admin.ModelAdmin):
    list_display = ['tenant', 'metric', 'quantity', 'timestamp']
    list_filter = ['metric', 'timestamp']
    search_fields = ['tenant__name']
