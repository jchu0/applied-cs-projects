"""
URL configuration for SaaS platform.
"""
from django.contrib import admin
from django.urls import path, include

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
