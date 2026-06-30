from django.urls import path
from . import views

app_name = 'admin'

urlpatterns = [
    path('', views.AdminDashboardView.as_view(), name='dashboard'),
    path('stats/', views.AdminStatsView.as_view(), name='stats'),
    path('users/', views.AdminUsersView.as_view(), name='users'),
    path('tenants/', views.AdminTenantsView.as_view(), name='tenants'),
    path('audit-logs/', views.AdminAuditLogsView.as_view(), name='audit-logs'),
    path('charts/growth/', views.AdminGrowthChartView.as_view(), name='growth-chart'),
]
