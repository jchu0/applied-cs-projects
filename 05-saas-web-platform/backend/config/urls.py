"""
URL configuration for SaaS platform.
"""
from django.contrib import admin
from django.urls import path, include

# Django's built-in admin and the platform's admin dashboard API
# (``apps.admin_dashboard``) both use the ``admin`` application namespace.
# Mounting both as-is collides: the contrib admin would shadow the dashboard's
# named routes so ``reverse('admin:dashboard')`` resolves to the Django admin
# (or fails) instead of the API. Rename the contrib admin's instance namespace
# to ``django-admin`` so the ``admin`` namespace stays free for the dashboard
# API's reversible routes (``admin:dashboard``, ``admin:stats``, ...).
admin.site.name = 'django-admin'

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/v1/', include('apps.core.urls')),
    path('api/v1/auth/', include('apps.users.urls')),
    path('api/v1/tenants/', include('apps.tenants.urls')),
    path('api/v1/billing/', include('apps.billing.urls')),
    path('api/v1/admin/', include('apps.admin_dashboard.urls')),
    path('api/v1/scheduler/', include('apps.scheduler.urls')),
    path('api/v1/resources/', include('apps.resources.urls')),
    path('api/v1/training/', include('apps.training.urls')),
]
