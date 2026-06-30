"""URL configuration for training app."""

from django.urls import path, include
from rest_framework.routers import DefaultRouter

from .views import (
    TrainingJobViewSet, TrainingRunViewSet, ExperimentViewSet,
    HyperparameterSweepViewSet, ModelArtifactViewSet,
)

router = DefaultRouter()
router.register(r'jobs', TrainingJobViewSet, basename='training-job')
router.register(r'runs', TrainingRunViewSet, basename='training-run')
router.register(r'experiments', ExperimentViewSet, basename='experiment')
router.register(r'sweeps', HyperparameterSweepViewSet, basename='sweep')
router.register(r'artifacts', ModelArtifactViewSet, basename='artifact')

urlpatterns = [
    path('', include(router.urls)),
]
