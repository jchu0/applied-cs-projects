"""
Test suite for subscription and billing functionality.
"""

import pytest
from decimal import Decimal
from datetime import timedelta
from unittest.mock import patch, MagicMock

pytest.importorskip("django")

from django.test import TestCase
from django.utils import timezone
from rest_framework.test import APIClient
from rest_framework import status

from apps.users.models import User
from apps.tenants.models import Tenant, TenantMembership
from apps.billing.models import Plan, Subscription, Invoice, PaymentMethod, UsageRecord


class PlanTestCase(TestCase):
    """Test subscription plan functionality."""

    def setUp(self):
        """Set up test data."""
        self.client = APIClient()

        # Create test plans
        self.free_plan = Plan.objects.create(
            name='Free',
            slug='free',
            price_monthly=Decimal('0.00'),
            price_yearly=Decimal('0.00'),
            max_users=1,
            max_storage_gb=1,
            features={
                'api_calls': 1000,
                'projects': 3
            },
            is_active=True,
            sort_order=0
        )

        self.pro_plan = Plan.objects.create(
            name='Professional',
            slug='pro',
            price_monthly=Decimal('29.99'),
            price_yearly=Decimal('299.99'),
            max_users=5,
            max_storage_gb=10,
            features={
                'api_calls': 10000,
                'projects': 20,
                'custom_domain': True
            },
            is_active=True,
            sort_order=1
        )

        self.enterprise_plan = Plan.objects.create(
            name='Enterprise',
            slug='enterprise',
            price_monthly=Decimal('99.99'),
            price_yearly=Decimal('999.99'),
            max_users=-1,  # Unlimited
            max_storage_gb=100,
            features={
                'api_calls': -1,  # Unlimited
                'projects': -1,  # Unlimited
                'custom_domain': True,
                'sso': True,
                'priority_support': True
            },
            is_active=True,
            sort_order=2
        )

        self.test_user = User.objects.create_user(
            email='test@example.com',
            password='TestPass123!'
        )
        self.client.force_authenticate(user=self.test_user)

    def test_plan_creation(self):
        """Test plan creation and attributes."""
        self.assertEqual(self.free_plan.name, 'Free')
        self.assertEqual(self.free_plan.price_monthly, Decimal('0.00'))
        self.assertEqual(self.free_plan.max_users, 1)
        self.assertTrue(self.free_plan.is_active)

    def test_plan_ordering(self):
        """Test plans are ordered by sort_order."""
        plans = Plan.objects.all()
        self.assertEqual(plans[0], self.free_plan)
        self.assertEqual(plans[1], self.pro_plan)
        self.assertEqual(plans[2], self.enterprise_plan)

    def test_plan_features_json(self):
        """Test plan features JSON field."""
        self.assertEqual(self.pro_plan.features['api_calls'], 10000)
        self.assertTrue(self.pro_plan.features['custom_domain'])

    def test_plan_validation(self):
        """Test plan validation and limits."""
        # Free plan should have limitations
        self.assertLess(self.free_plan.max_users, self.pro_plan.max_users)
        self.assertLess(self.free_plan.max_storage_gb, self.pro_plan.max_storage_gb)


class SubscriptionTestCase(TestCase):
    """Test subscription management functionality."""

    def setUp(self):
        """Set up test environment."""
        self.client = APIClient()

        self.user = User.objects.create_user(
            email='subscriber@example.com',
            password='SubPass123!'
        )

        self.tenant = Tenant.objects.create(
            name='Test Tenant',
            slug='test-tenant'
        )

        TenantMembership.objects.create(
            tenant=self.tenant,
            user=self.user,
            role=TenantMembership.Role.OWNER
        )

        self.plan = Plan.objects.create(
            name='Pro',
            slug='pro',
            price_monthly=Decimal('29.99'),
            price_yearly=Decimal('299.99'),
            max_users=5,
            max_storage_gb=10,
            features={'api_calls': 10000},
            is_active=True
        )

        self.client.force_authenticate(user=self.user)

    def test_create_subscription(self):
        """Test creating a new subscription."""
        subscription = Subscription.objects.create(
            tenant=self.tenant,
            plan=self.plan,
            status=Subscription.Status.ACTIVE,
            billing_interval=Subscription.BillingInterval.MONTHLY
        )

        self.assertEqual(subscription.tenant, self.tenant)
        self.assertEqual(subscription.plan, self.plan)
        self.assertEqual(subscription.status, Subscription.Status.ACTIVE)

    def test_subscription_status_choices(self):
        """Test subscription status choices."""
        subscription = Subscription.objects.create(
            tenant=self.tenant,
            plan=self.plan,
            status=Subscription.Status.TRIALING
        )

        self.assertEqual(subscription.status, 'trialing')

        subscription.status = Subscription.Status.ACTIVE
        subscription.save()
        subscription.refresh_from_db()

        self.assertEqual(subscription.status, 'active')

    def test_subscription_billing_interval(self):
        """Test subscription billing interval."""
        subscription = Subscription.objects.create(
            tenant=self.tenant,
            plan=self.plan,
            billing_interval=Subscription.BillingInterval.YEARLY
        )

        self.assertEqual(subscription.billing_interval, 'yearly')

    def test_cancel_subscription(self):
        """Test canceling a subscription."""
        subscription = Subscription.objects.create(
            tenant=self.tenant,
            plan=self.plan,
            status=Subscription.Status.ACTIVE
        )

        subscription.status = Subscription.Status.CANCELED
        subscription.canceled_at = timezone.now()
        subscription.save()

        subscription.refresh_from_db()
        self.assertEqual(subscription.status, 'canceled')
        self.assertIsNotNone(subscription.canceled_at)


class InvoiceTestCase(TestCase):
    """Test invoice generation and management."""

    def setUp(self):
        """Set up test environment."""
        self.client = APIClient()
        self.user = User.objects.create_user(
            email='invoice@example.com',
            password='InvoicePass123!'
        )

        self.tenant = Tenant.objects.create(
            name='Invoice Tenant',
            slug='invoice-tenant'
        )

        self.plan = Plan.objects.create(
            name='Pro',
            slug='pro',
            price_monthly=Decimal('29.99'),
            price_yearly=Decimal('299.99'),
            is_active=True
        )

        self.subscription = Subscription.objects.create(
            tenant=self.tenant,
            plan=self.plan,
            status=Subscription.Status.ACTIVE
        )

        self.client.force_authenticate(user=self.user)

    def test_create_invoice(self):
        """Test invoice creation."""
        invoice = Invoice.objects.create(
            tenant=self.tenant,
            subscription=self.subscription,
            number='INV-2024-001',
            subtotal=Decimal('29.99'),
            tax=Decimal('2.40'),
            total=Decimal('32.39'),
            status=Invoice.Status.PAID
        )

        self.assertEqual(invoice.total, Decimal('32.39'))
        self.assertTrue(invoice.number.startswith('INV'))
        self.assertEqual(invoice.status, 'paid')

    def test_invoice_status_transitions(self):
        """Test invoice status transitions."""
        invoice = Invoice.objects.create(
            tenant=self.tenant,
            number='INV-2024-002',
            subtotal=Decimal('29.99'),
            total=Decimal('29.99'),
            status=Invoice.Status.DRAFT
        )

        self.assertEqual(invoice.status, 'draft')

        invoice.status = Invoice.Status.OPEN
        invoice.save()
        self.assertEqual(invoice.status, 'open')

        invoice.status = Invoice.Status.PAID
        invoice.paid_at = timezone.now()
        invoice.save()
        self.assertEqual(invoice.status, 'paid')

    def test_invoice_ordering(self):
        """Test invoices are ordered by created_at descending."""
        Invoice.objects.create(
            tenant=self.tenant,
            number='INV-2024-001',
            subtotal=Decimal('29.99'),
            total=Decimal('29.99')
        )
        Invoice.objects.create(
            tenant=self.tenant,
            number='INV-2024-002',
            subtotal=Decimal('29.99'),
            total=Decimal('29.99')
        )

        invoices = Invoice.objects.filter(tenant=self.tenant)
        self.assertEqual(invoices[0].number, 'INV-2024-002')


class PaymentMethodTestCase(TestCase):
    """Test payment method management."""

    def setUp(self):
        """Set up test environment."""
        self.client = APIClient()
        self.user = User.objects.create_user(
            email='payment@example.com',
            password='PaymentPass123!'
        )

        self.tenant = Tenant.objects.create(
            name='Payment Tenant',
            slug='payment-tenant'
        )

        self.client.force_authenticate(user=self.user)

    def test_create_payment_method(self):
        """Test creating a payment method."""
        payment_method = PaymentMethod.objects.create(
            tenant=self.tenant,
            stripe_payment_method_id='pm_test123',
            type=PaymentMethod.Type.CARD,
            card_brand='visa',
            card_last4='4242',
            card_exp_month=12,
            card_exp_year=2025,
            is_default=True
        )

        self.assertEqual(payment_method.card_brand, 'visa')
        self.assertEqual(payment_method.card_last4, '4242')
        self.assertTrue(payment_method.is_default)

    def test_payment_method_string_representation(self):
        """Test payment method string representation."""
        payment_method = PaymentMethod.objects.create(
            tenant=self.tenant,
            stripe_payment_method_id='pm_test123',
            type=PaymentMethod.Type.CARD,
            card_brand='visa',
            card_last4='4242'
        )

        self.assertEqual(str(payment_method), 'visa **** 4242')


class UsageTrackingTestCase(TestCase):
    """Test usage tracking."""

    def setUp(self):
        """Set up test environment."""
        self.client = APIClient()
        self.user = User.objects.create_user(
            email='usage@example.com',
            password='UsagePass123!'
        )

        self.tenant = Tenant.objects.create(
            name='Usage Tenant',
            slug='usage-tenant'
        )

        self.plan = Plan.objects.create(
            name='Pro',
            slug='pro',
            price_monthly=Decimal('29.99'),
            price_yearly=Decimal('299.99'),
            max_users=5,
            max_storage_gb=10,
            features={'api_calls': 10000},
            is_active=True
        )

        self.subscription = Subscription.objects.create(
            tenant=self.tenant,
            plan=self.plan,
            status=Subscription.Status.ACTIVE
        )

        self.client.force_authenticate(user=self.user)

    def test_record_api_usage(self):
        """Test recording API usage."""
        usage = UsageRecord.objects.create(
            tenant=self.tenant,
            subscription=self.subscription,
            metric='api_calls',
            quantity=100,
            timestamp=timezone.now()
        )

        self.assertEqual(usage.metric, 'api_calls')
        self.assertEqual(usage.quantity, 100)

    def test_record_storage_usage(self):
        """Test recording storage usage."""
        usage = UsageRecord.objects.create(
            tenant=self.tenant,
            subscription=self.subscription,
            metric='storage_gb',
            quantity=5,
            timestamp=timezone.now()
        )

        self.assertEqual(usage.metric, 'storage_gb')
        self.assertEqual(usage.quantity, 5)

    def test_aggregate_usage(self):
        """Test aggregating usage records."""
        now = timezone.now()

        # Create multiple usage records
        for i in range(5):
            UsageRecord.objects.create(
                tenant=self.tenant,
                subscription=self.subscription,
                metric='api_calls',
                quantity=100,
                timestamp=now - timedelta(days=i)
            )

        total = UsageRecord.objects.filter(
            tenant=self.tenant,
            metric='api_calls'
        ).aggregate(total=models.Sum('quantity'))['total']

        self.assertEqual(total, 500)


# Import models for aggregate function
from django.db import models


class TenantSubscriptionTestCase(TestCase):
    """Test tenant-subscription relationship."""

    def setUp(self):
        """Set up test environment."""
        self.user = User.objects.create_user(
            email='tenant_owner@example.com',
            password='TenantPass123!'
        )

        self.tenant = Tenant.objects.create(
            name='Subscription Tenant',
            slug='subscription-tenant'
        )

        self.plan = Plan.objects.create(
            name='Pro',
            slug='pro',
            price_monthly=Decimal('29.99'),
            price_yearly=Decimal('299.99'),
            is_active=True
        )

    def test_one_to_one_relationship(self):
        """Test tenant can have only one subscription."""
        Subscription.objects.create(
            tenant=self.tenant,
            plan=self.plan,
            status=Subscription.Status.ACTIVE
        )

        # Trying to create another subscription for same tenant should fail
        with self.assertRaises(Exception):
            Subscription.objects.create(
                tenant=self.tenant,
                plan=self.plan,
                status=Subscription.Status.ACTIVE
            )

    def test_access_subscription_from_tenant(self):
        """Test accessing subscription from tenant."""
        subscription = Subscription.objects.create(
            tenant=self.tenant,
            plan=self.plan,
            status=Subscription.Status.ACTIVE
        )

        # Access subscription via tenant
        self.assertEqual(self.tenant.subscription, subscription)
        self.assertEqual(self.tenant.subscription.plan, self.plan)


if __name__ == '__main__':
    pytest.main([__file__])
