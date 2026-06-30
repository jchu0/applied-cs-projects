"""
Stripe webhook handlers.
"""
import stripe
from django.conf import settings
from django.utils import timezone
from datetime import datetime

from .models import Subscription, Invoice
from .services import StripeService
from apps.tenants.models import Tenant


def handle_webhook(payload: dict, sig_header: str) -> dict:
    """Handle incoming Stripe webhook."""
    endpoint_secret = getattr(settings, 'STRIPE_WEBHOOK_SECRET', '')

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, endpoint_secret
        )
    except ValueError:
        raise ValueError('Invalid payload')
    except stripe.error.SignatureVerificationError:
        raise ValueError('Invalid signature')

    # Handle the event
    event_type = event['type']
    data = event['data']['object']

    handlers = {
        'customer.subscription.created': handle_subscription_created,
        'customer.subscription.updated': handle_subscription_updated,
        'customer.subscription.deleted': handle_subscription_deleted,
        'invoice.paid': handle_invoice_paid,
        'invoice.payment_failed': handle_invoice_payment_failed,
        'invoice.created': handle_invoice_created,
        'payment_method.attached': handle_payment_method_attached,
        'checkout.session.completed': handle_checkout_completed,
    }

    handler = handlers.get(event_type)
    if handler:
        handler(data)

    return {'status': 'success', 'type': event_type}


def handle_subscription_created(data: dict) -> None:
    """Handle subscription created event."""
    tenant_id = data.get('metadata', {}).get('tenant_id')
    if not tenant_id:
        return

    try:
        tenant = Tenant.objects.get(id=tenant_id)
        # Subscription might already exist from our create call
        Subscription.objects.filter(
            stripe_subscription_id=data['id']
        ).update(
            status=map_subscription_status(data['status']),
            current_period_start=datetime.fromtimestamp(data['current_period_start'], tz=timezone.utc),
            current_period_end=datetime.fromtimestamp(data['current_period_end'], tz=timezone.utc),
        )
    except Tenant.DoesNotExist:
        pass


def handle_subscription_updated(data: dict) -> None:
    """Handle subscription updated event."""
    try:
        subscription = Subscription.objects.get(stripe_subscription_id=data['id'])
        subscription.status = map_subscription_status(data['status'])
        subscription.current_period_start = datetime.fromtimestamp(data['current_period_start'], tz=timezone.utc)
        subscription.current_period_end = datetime.fromtimestamp(data['current_period_end'], tz=timezone.utc)

        if data.get('cancel_at_period_end'):
            subscription.canceled_at = timezone.now()

        subscription.save()
    except Subscription.DoesNotExist:
        pass


def handle_subscription_deleted(data: dict) -> None:
    """Handle subscription deleted event."""
    try:
        subscription = Subscription.objects.get(stripe_subscription_id=data['id'])
        subscription.status = Subscription.Status.CANCELED
        subscription.canceled_at = timezone.now()
        subscription.save()
    except Subscription.DoesNotExist:
        pass


def handle_invoice_paid(data: dict) -> None:
    """Handle invoice paid event."""
    StripeService.sync_invoice(data)


def handle_invoice_payment_failed(data: dict) -> None:
    """Handle invoice payment failed event."""
    invoice = StripeService.sync_invoice(data)

    # Update subscription status
    if invoice.subscription:
        invoice.subscription.status = Subscription.Status.PAST_DUE
        invoice.subscription.save()


def handle_invoice_created(data: dict) -> None:
    """Handle invoice created event."""
    StripeService.sync_invoice(data)


def handle_payment_method_attached(data: dict) -> None:
    """Handle payment method attached event."""
    # Payment methods are typically attached via our API
    pass


def handle_checkout_completed(data: dict) -> None:
    """Handle checkout session completed event."""
    tenant_id = data.get('metadata', {}).get('tenant_id')
    if not tenant_id:
        return

    # The subscription is created automatically by Stripe
    # We'll sync it via the subscription.created webhook


def map_subscription_status(stripe_status: str) -> str:
    """Map Stripe subscription status to our status."""
    status_map = {
        'active': Subscription.Status.ACTIVE,
        'past_due': Subscription.Status.PAST_DUE,
        'canceled': Subscription.Status.CANCELED,
        'trialing': Subscription.Status.TRIALING,
        'paused': Subscription.Status.PAUSED,
        'unpaid': Subscription.Status.PAST_DUE,
        'incomplete': Subscription.Status.TRIALING,
        'incomplete_expired': Subscription.Status.CANCELED,
    }
    return status_map.get(stripe_status, Subscription.Status.ACTIVE)
