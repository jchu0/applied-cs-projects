"""
Billing views for subscription and payment management.
"""
from rest_framework import status, views
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from django.shortcuts import get_object_or_404
from django.conf import settings

from .models import Plan, Subscription, Invoice, PaymentMethod
from .serializers import (
    PlanSerializer,
    SubscriptionSerializer,
    CreateSubscriptionSerializer,
    InvoiceSerializer,
    PaymentMethodSerializer,
)
from .services import StripeService
from .webhooks import handle_webhook
from apps.tenants.models import Tenant, TenantMembership


class PlanListView(views.APIView):
    """List available subscription plans."""

    def get(self, request):
        plans = Plan.objects.filter(is_active=True)
        return Response(PlanSerializer(plans, many=True).data)


class SubscriptionView(views.APIView):
    """Get or manage tenant subscription."""

    def get_tenant(self, request, tenant_id):
        tenant = get_object_or_404(Tenant, id=tenant_id)
        membership = get_object_or_404(TenantMembership, tenant=tenant, user=request.user)
        return tenant, membership

    def get(self, request, tenant_id):
        tenant, _ = self.get_tenant(request, tenant_id)
        try:
            subscription = tenant.subscription
            return Response(SubscriptionSerializer(subscription).data)
        except Subscription.DoesNotExist:
            return Response({'error': 'No active subscription'}, status=status.HTTP_404_NOT_FOUND)

    def post(self, request, tenant_id):
        """Create or update subscription."""
        tenant, membership = self.get_tenant(request, tenant_id)

        if membership.role not in [TenantMembership.Role.OWNER, TenantMembership.Role.ADMIN]:
            return Response({'error': 'Permission denied'}, status=status.HTTP_403_FORBIDDEN)

        serializer = CreateSubscriptionSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        plan = get_object_or_404(Plan, id=serializer.validated_data['plan_id'])

        # Check for existing subscription
        try:
            existing = tenant.subscription
            # Update existing subscription
            subscription = StripeService.update_subscription_plan(
                existing,
                plan,
                serializer.validated_data['billing_interval']
            )
            return Response(SubscriptionSerializer(subscription).data)
        except Subscription.DoesNotExist:
            # Create new subscription
            payment_method_id = serializer.validated_data.get('payment_method_id')
            subscription = StripeService.create_subscription(
                tenant,
                plan,
                serializer.validated_data['billing_interval'],
                payment_method_id
            )
            return Response(
                SubscriptionSerializer(subscription).data,
                status=status.HTTP_201_CREATED
            )

    def delete(self, request, tenant_id):
        """Cancel subscription."""
        tenant, membership = self.get_tenant(request, tenant_id)

        if membership.role != TenantMembership.Role.OWNER:
            return Response({'error': 'Only owner can cancel subscription'}, status=status.HTTP_403_FORBIDDEN)

        try:
            subscription = tenant.subscription
            StripeService.cancel_subscription(subscription)
            return Response({'message': 'Subscription canceled'})
        except Subscription.DoesNotExist:
            return Response({'error': 'No active subscription'}, status=status.HTTP_404_NOT_FOUND)


class InvoiceListView(views.APIView):
    """List tenant invoices."""

    def get(self, request, tenant_id):
        tenant = get_object_or_404(Tenant, id=tenant_id)
        get_object_or_404(TenantMembership, tenant=tenant, user=request.user)

        invoices = tenant.invoices.all()
        return Response(InvoiceSerializer(invoices, many=True).data)


class PaymentMethodListView(views.APIView):
    """List and manage payment methods."""

    def get_tenant(self, request, tenant_id):
        tenant = get_object_or_404(Tenant, id=tenant_id)
        membership = get_object_or_404(TenantMembership, tenant=tenant, user=request.user)
        return tenant, membership

    def get(self, request, tenant_id):
        tenant, _ = self.get_tenant(request, tenant_id)
        methods = tenant.payment_methods.all()
        return Response(PaymentMethodSerializer(methods, many=True).data)

    def post(self, request, tenant_id):
        """Create setup intent for adding payment method."""
        tenant, membership = self.get_tenant(request, tenant_id)

        if membership.role not in [TenantMembership.Role.OWNER, TenantMembership.Role.ADMIN]:
            return Response({'error': 'Permission denied'}, status=status.HTTP_403_FORBIDDEN)

        # Ensure customer exists
        StripeService.get_or_create_customer(tenant, request.user.email)

        setup_intent = StripeService.create_setup_intent(tenant.stripe_customer_id)
        return Response(setup_intent)

    def delete(self, request, tenant_id, method_id):
        """Delete a payment method."""
        tenant, membership = self.get_tenant(request, tenant_id)

        if membership.role not in [TenantMembership.Role.OWNER, TenantMembership.Role.ADMIN]:
            return Response({'error': 'Permission denied'}, status=status.HTTP_403_FORBIDDEN)

        method = get_object_or_404(PaymentMethod, id=method_id, tenant=tenant)
        StripeService.detach_payment_method(method)
        return Response(status=status.HTTP_204_NO_CONTENT)


class WebhookView(views.APIView):
    """Handle Stripe webhooks."""
    permission_classes = [AllowAny]

    def post(self, request):
        payload = request.body
        sig_header = request.META.get('HTTP_STRIPE_SIGNATURE', '')

        try:
            result = handle_webhook(payload, sig_header)
            return Response(result)
        except ValueError as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)


class CheckoutSessionView(views.APIView):
    """Create Stripe Checkout session."""

    def post(self, request, tenant_id):
        tenant = get_object_or_404(Tenant, id=tenant_id)
        membership = get_object_or_404(TenantMembership, tenant=tenant, user=request.user)

        if membership.role not in [TenantMembership.Role.OWNER, TenantMembership.Role.ADMIN]:
            return Response({'error': 'Permission denied'}, status=status.HTTP_403_FORBIDDEN)

        plan_id = request.data.get('plan_id')
        billing_interval = request.data.get('billing_interval', 'monthly')

        plan = get_object_or_404(Plan, id=plan_id)

        # Ensure customer exists
        StripeService.get_or_create_customer(tenant, request.user.email)

        # Get URLs from request or settings
        success_url = request.data.get('success_url', f"{settings.FRONTEND_URL}/dashboard/billing?success=true")
        cancel_url = request.data.get('cancel_url', f"{settings.FRONTEND_URL}/dashboard/billing?canceled=true")

        checkout_url = StripeService.create_checkout_session(
            tenant,
            plan,
            billing_interval,
            success_url,
            cancel_url
        )

        return Response({'checkout_url': checkout_url})


class BillingPortalView(views.APIView):
    """Create Stripe billing portal session."""

    def post(self, request, tenant_id):
        tenant = get_object_or_404(Tenant, id=tenant_id)
        membership = get_object_or_404(TenantMembership, tenant=tenant, user=request.user)

        if membership.role not in [TenantMembership.Role.OWNER, TenantMembership.Role.ADMIN]:
            return Response({'error': 'Permission denied'}, status=status.HTTP_403_FORBIDDEN)

        if not tenant.stripe_customer_id:
            return Response({'error': 'No billing account'}, status=status.HTTP_400_BAD_REQUEST)

        return_url = request.data.get('return_url', f"{settings.FRONTEND_URL}/dashboard/billing")
        portal_url = StripeService.create_billing_portal_session(
            tenant.stripe_customer_id,
            return_url
        )

        return Response({'portal_url': portal_url})
