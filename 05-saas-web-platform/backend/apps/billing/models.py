"""
Billing models for subscription and payment management.
"""
import uuid
from django.db import models
from apps.tenants.models import Tenant


class Plan(models.Model):
    """Subscription plan definition."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    slug = models.SlugField(unique=True)
    description = models.TextField(blank=True)

    # Pricing
    price_monthly = models.DecimalField(max_digits=10, decimal_places=2)
    price_yearly = models.DecimalField(max_digits=10, decimal_places=2)
    stripe_price_id_monthly = models.CharField(max_length=100, blank=True)
    stripe_price_id_yearly = models.CharField(max_length=100, blank=True)

    # Limits
    max_users = models.IntegerField(default=5)
    max_storage_gb = models.IntegerField(default=10)
    features = models.JSONField(default=dict)

    is_active = models.BooleanField(default=True)
    sort_order = models.IntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'plans'
        ordering = ['sort_order']

    def __str__(self):
        return self.name


class Subscription(models.Model):
    """Tenant subscription to a plan."""

    class Status(models.TextChoices):
        ACTIVE = 'active', 'Active'
        PAST_DUE = 'past_due', 'Past Due'
        CANCELED = 'canceled', 'Canceled'
        TRIALING = 'trialing', 'Trialing'
        PAUSED = 'paused', 'Paused'

    class BillingInterval(models.TextChoices):
        MONTHLY = 'monthly', 'Monthly'
        YEARLY = 'yearly', 'Yearly'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.OneToOneField(Tenant, on_delete=models.CASCADE, related_name='subscription')
    plan = models.ForeignKey(Plan, on_delete=models.PROTECT)

    status = models.CharField(max_length=20, choices=Status.choices, default=Status.TRIALING)
    billing_interval = models.CharField(max_length=20, choices=BillingInterval.choices, default=BillingInterval.MONTHLY)

    # Stripe
    stripe_subscription_id = models.CharField(max_length=100, blank=True)

    # Dates
    trial_ends_at = models.DateTimeField(null=True, blank=True)
    current_period_start = models.DateTimeField(null=True, blank=True)
    current_period_end = models.DateTimeField(null=True, blank=True)
    canceled_at = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'subscriptions'

    def __str__(self):
        return f'{self.tenant.name} - {self.plan.name}'


class Invoice(models.Model):
    """Invoice for billing."""

    class Status(models.TextChoices):
        DRAFT = 'draft', 'Draft'
        OPEN = 'open', 'Open'
        PAID = 'paid', 'Paid'
        VOID = 'void', 'Void'
        UNCOLLECTIBLE = 'uncollectible', 'Uncollectible'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(Tenant, on_delete=models.CASCADE, related_name='invoices')
    subscription = models.ForeignKey(Subscription, on_delete=models.SET_NULL, null=True, blank=True)

    stripe_invoice_id = models.CharField(max_length=100, blank=True)
    number = models.CharField(max_length=50)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.DRAFT)

    # Amounts
    subtotal = models.DecimalField(max_digits=10, decimal_places=2)
    tax = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    total = models.DecimalField(max_digits=10, decimal_places=2)

    # URLs
    invoice_pdf_url = models.URLField(blank=True)
    hosted_invoice_url = models.URLField(blank=True)

    due_date = models.DateTimeField(null=True, blank=True)
    paid_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'invoices'
        ordering = ['-created_at']

    def __str__(self):
        return f'{self.number} - {self.tenant.name}'


class PaymentMethod(models.Model):
    """Stored payment method."""

    class Type(models.TextChoices):
        CARD = 'card', 'Card'
        BANK_ACCOUNT = 'bank_account', 'Bank Account'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(Tenant, on_delete=models.CASCADE, related_name='payment_methods')

    stripe_payment_method_id = models.CharField(max_length=100)
    type = models.CharField(max_length=20, choices=Type.choices)
    is_default = models.BooleanField(default=False)

    # Card details (masked)
    card_brand = models.CharField(max_length=20, blank=True)
    card_last4 = models.CharField(max_length=4, blank=True)
    card_exp_month = models.IntegerField(null=True, blank=True)
    card_exp_year = models.IntegerField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'payment_methods'

    def __str__(self):
        if self.type == self.Type.CARD:
            return f'{self.card_brand} **** {self.card_last4}'
        return f'{self.type}'


class UsageRecord(models.Model):
    """Track metered usage for billing."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(Tenant, on_delete=models.CASCADE, related_name='usage_records')
    subscription = models.ForeignKey(Subscription, on_delete=models.CASCADE)

    metric = models.CharField(max_length=50)  # e.g., 'api_calls', 'storage_gb'
    quantity = models.IntegerField()
    timestamp = models.DateTimeField()

    stripe_usage_record_id = models.CharField(max_length=100, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'usage_records'
        indexes = [
            models.Index(fields=['tenant', 'metric', 'timestamp']),
        ]
