"""URL configuration for resources app."""

from django.urls import path, include
from rest_framework.routers import DefaultRouter

from .views import (
    ComputeNodeViewSet, GPUViewSet, ResourceAllocationViewSet,
    ResourceQuotaViewSet, ResourceReservationViewSet,
)

router = DefaultRouter()
router.register(r'nodes', ComputeNodeViewSet, basename='compute-node')
router.register(r'gpus', GPUViewSet, basename='gpu')
router.register(r'allocations', ResourceAllocationViewSet, basename='allocation')
router.register(r'quotas', ResourceQuotaViewSet, basename='quota')
router.register(r'reservations', ResourceReservationViewSet, basename='reservation')

urlpatterns = [
    path('', include(router.urls)),
]
