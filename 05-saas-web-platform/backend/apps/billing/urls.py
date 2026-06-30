from django.urls import path
from . import views

urlpatterns = [
    path('plans/', views.PlanListView.as_view(), name='plan-list'),
    path('tenants/<uuid:tenant_id>/subscription/', views.SubscriptionView.as_view(), name='subscription'),
    path('tenants/<uuid:tenant_id>/invoices/', views.InvoiceListView.as_view(), name='invoice-list'),
    path('tenants/<uuid:tenant_id>/payment-methods/', views.PaymentMethodListView.as_view(), name='payment-methods'),
    path('tenants/<uuid:tenant_id>/checkout/', views.CheckoutSessionView.as_view(), name='checkout-session'),
    path('tenants/<uuid:tenant_id>/portal/', views.BillingPortalView.as_view(), name='billing-portal'),
    path('webhooks/stripe/', views.WebhookView.as_view(), name='stripe-webhook'),
]
