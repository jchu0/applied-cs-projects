"""
Stripe service for billing operations.
"""
import stripe
from datetime import datetime, timedelta
from django.conf import settings
from django.utils import timezone

from .models import Plan, Subscription, Invoice, PaymentMethod, UsageRecord
from apps.tenants.models import Tenant

# Configure Stripe
stripe.api_key = getattr(settings, 'STRIPE_SECRET_KEY', '')


class StripeService:
    """Service for Stripe operations."""

    @staticmethod
    def create_customer(tenant: Tenant, email: str) -> str:
        """Create a Stripe customer for a tenant."""
        customer = stripe.Customer.create(
            email=email,
            name=tenant.name,
            metadata={
                'tenant_id': str(tenant.id),
            }
        )
        tenant.stripe_customer_id = customer.id
        tenant.save(update_fields=['stripe_customer_id'])
        return customer.id

    @staticmethod
    def get_or_create_customer(tenant: Tenant, email: str) -> str:
        """Get existing customer or create new one."""
        if tenant.stripe_customer_id:
            return tenant.stripe_customer_id
        return StripeService.create_customer(tenant, email)

    @staticmethod
    def create_subscription(
        tenant: Tenant,
        plan: Plan,
        billing_interval: str,
        payment_method_id: str = None
    ) -> Subscription:
        """Create a Stripe subscription."""
        customer_id = tenant.stripe_customer_id
        if not customer_id:
            raise ValueError('Tenant has no Stripe customer')

        # Get the price ID based on billing interval
        price_id = (
            plan.stripe_price_id_yearly
            if billing_interval == 'yearly'
            else plan.stripe_price_id_monthly
        )

        # Create subscription in Stripe
        stripe_sub = stripe.Subscription.create(
            customer=customer_id,
            items=[{'price': price_id}],
            default_payment_method=payment_method_id,
            trial_period_days=14,
            metadata={
                'tenant_id': str(tenant.id),
                'plan_id': str(plan.id),
            }
        )

        # Create local subscription record
        subscription = Subscription.objects.create(
            tenant=tenant,
            plan=plan,
            status=Subscription.Status.TRIALING,
            billing_interval=billing_interval,
            stripe_subscription_id=stripe_sub.id,
            trial_ends_at=datetime.fromtimestamp(stripe_sub.trial_end, tz=timezone.utc) if stripe_sub.trial_end else None,
            current_period_start=datetime.fromtimestamp(stripe_sub.current_period_start, tz=timezone.utc),
            current_period_end=datetime.fromtimestamp(stripe_sub.current_period_end, tz=timezone.utc),
        )

        return subscription

    @staticmethod
    def cancel_subscription(subscription: Subscription, at_period_end: bool = True) -> None:
        """Cancel a subscription."""
        if subscription.stripe_subscription_id:
            if at_period_end:
                stripe.Subscription.modify(
                    subscription.stripe_subscription_id,
                    cancel_at_period_end=True
                )
            else:
                stripe.Subscription.delete(subscription.stripe_subscription_id)

        subscription.status = Subscription.Status.CANCELED
        subscription.canceled_at = timezone.now()
        subscription.save()

    @staticmethod
    def update_subscription_plan(
        subscription: Subscription,
        new_plan: Plan,
        billing_interval: str
    ) -> Subscription:
        """Update subscription to a new plan."""
        price_id = (
            new_plan.stripe_price_id_yearly
            if billing_interval == 'yearly'
            else new_plan.stripe_price_id_monthly
        )

        # Get current subscription from Stripe
        stripe_sub = stripe.Subscription.retrieve(subscription.stripe_subscription_id)

        # Update the subscription item
        stripe.Subscription.modify(
            subscription.stripe_subscription_id,
            items=[{
                'id': stripe_sub['items']['data'][0].id,
                'price': price_id,
            }],
            proration_behavior='create_prorations'
        )

        subscription.plan = new_plan
        subscription.billing_interval = billing_interval
        subscription.save()

        return subscription

    @staticmethod
    def create_setup_intent(customer_id: str) -> dict:
        """Create a SetupIntent for adding a payment method."""
        setup_intent = stripe.SetupIntent.create(
            customer=customer_id,
            payment_method_types=['card'],
        )
        return {
            'client_secret': setup_intent.client_secret,
            'id': setup_intent.id,
        }

    @staticmethod
    def attach_payment_method(
        tenant: Tenant,
        payment_method_id: str,
        set_default: bool = True
    ) -> PaymentMethod:
        """Attach a payment method to a customer."""
        # Retrieve payment method from Stripe
        stripe_pm = stripe.PaymentMethod.retrieve(payment_method_id)

        # Attach to customer
        stripe.PaymentMethod.attach(
            payment_method_id,
            customer=tenant.stripe_customer_id
        )

        # Set as default if requested
        if set_default:
            stripe.Customer.modify(
                tenant.stripe_customer_id,
                invoice_settings={'default_payment_method': payment_method_id}
            )
            # Unset other defaults
            tenant.payment_methods.filter(is_default=True).update(is_default=False)

        # Create local record
        payment_method = PaymentMethod.objects.create(
            tenant=tenant,
            stripe_payment_method_id=payment_method_id,
            type=PaymentMethod.Type.CARD,
            is_default=set_default,
            card_brand=stripe_pm.card.brand,
            card_last4=stripe_pm.card.last4,
            card_exp_month=stripe_pm.card.exp_month,
            card_exp_year=stripe_pm.card.exp_year,
        )

        return payment_method

    @staticmethod
    def detach_payment_method(payment_method: PaymentMethod) -> None:
        """Detach a payment method from a customer."""
        stripe.PaymentMethod.detach(payment_method.stripe_payment_method_id)
        payment_method.delete()

    @staticmethod
    def create_billing_portal_session(customer_id: str, return_url: str) -> str:
        """Create a billing portal session."""
        session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=return_url,
        )
        return session.url

    @staticmethod
    def create_checkout_session(
        tenant: Tenant,
        plan: Plan,
        billing_interval: str,
        success_url: str,
        cancel_url: str
    ) -> str:
        """Create a Stripe Checkout session."""
        price_id = (
            plan.stripe_price_id_yearly
            if billing_interval == 'yearly'
            else plan.stripe_price_id_monthly
        )

        session = stripe.checkout.Session.create(
            customer=tenant.stripe_customer_id,
            payment_method_types=['card'],
            line_items=[{
                'price': price_id,
                'quantity': 1,
            }],
            mode='subscription',
            success_url=success_url,
            cancel_url=cancel_url,
            metadata={
                'tenant_id': str(tenant.id),
                'plan_id': str(plan.id),
            }
        )

        return session.url

    @staticmethod
    def record_usage(
        subscription: Subscription,
        metric: str,
        quantity: int,
        timestamp: datetime = None
    ) -> UsageRecord:
        """Record metered usage."""
        if timestamp is None:
            timestamp = timezone.now()

        # Find the subscription item for metered billing
        # This assumes metered pricing is set up in Stripe

        usage_record = UsageRecord.objects.create(
            tenant=subscription.tenant,
            subscription=subscription,
            metric=metric,
            quantity=quantity,
            timestamp=timestamp,
        )

        return usage_record

    @staticmethod
    def sync_invoice(stripe_invoice: dict) -> Invoice:
        """Sync an invoice from Stripe webhook."""
        tenant = Tenant.objects.get(stripe_customer_id=stripe_invoice['customer'])

        invoice, created = Invoice.objects.update_or_create(
            stripe_invoice_id=stripe_invoice['id'],
            defaults={
                'tenant': tenant,
                'number': stripe_invoice.get('number', ''),
                'status': stripe_invoice['status'],
                'subtotal': stripe_invoice['subtotal'] / 100,
                'tax': stripe_invoice.get('tax', 0) / 100,
                'total': stripe_invoice['total'] / 100,
                'invoice_pdf_url': stripe_invoice.get('invoice_pdf', ''),
                'hosted_invoice_url': stripe_invoice.get('hosted_invoice_url', ''),
                'due_date': datetime.fromtimestamp(stripe_invoice['due_date'], tz=timezone.utc) if stripe_invoice.get('due_date') else None,
                'paid_at': datetime.fromtimestamp(stripe_invoice['status_transitions']['paid_at'], tz=timezone.utc) if stripe_invoice.get('status_transitions', {}).get('paid_at') else None,
            }
        )

        return invoice
