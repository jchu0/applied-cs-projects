from django.urls import path
from . import views

urlpatterns = [
    path('health/', views.HealthCheckView.as_view(), name='health-check'),
    path('ready/', views.ReadyCheckView.as_view(), name='ready-check'),
]
