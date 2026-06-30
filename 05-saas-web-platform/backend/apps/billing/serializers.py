"""
Billing serializers.
"""
from rest_framework import serializers
from .models import Plan, Subscription, Invoice, PaymentMethod


class PlanSerializer(serializers.ModelSerializer):
    """Serializer for subscription plans."""

    class Meta:
        model = Plan
        fields = [
            'id', 'name', 'slug', 'description',
            'price_monthly', 'price_yearly',
            'max_users', 'max_storage_gb', 'features',
            'is_active', 'sort_order',
        ]


class SubscriptionSerializer(serializers.ModelSerializer):
    """Serializer for subscriptions."""
    plan = PlanSerializer(read_only=True)

    class Meta:
        model = Subscription
        fields = [
            'id', 'plan', 'status', 'billing_interval',
            'trial_ends_at', 'current_period_start', 'current_period_end',
            'canceled_at', 'created_at',
        ]


class CreateSubscriptionSerializer(serializers.Serializer):
    """Serializer for creating subscriptions."""
    plan_id = serializers.UUIDField()
    billing_interval = serializers.ChoiceField(choices=Subscription.BillingInterval.choices)
    payment_method_id = serializers.CharField(required=False)


class InvoiceSerializer(serializers.ModelSerializer):
    """Serializer for invoices."""

    class Meta:
        model = Invoice
        fields = [
            'id', 'number', 'status', 'subtotal', 'tax', 'total',
            'invoice_pdf_url', 'hosted_invoice_url',
            'due_date', 'paid_at', 'created_at',
        ]


class PaymentMethodSerializer(serializers.ModelSerializer):
    """Serializer for payment methods."""

    class Meta:
        model = PaymentMethod
        fields = [
            'id', 'type', 'is_default',
            'card_brand', 'card_last4', 'card_exp_month', 'card_exp_year',
            'created_at',
        ]


class SetupIntentSerializer(serializers.Serializer):
    """Serializer for Stripe setup intent."""
    client_secret = serializers.CharField()
