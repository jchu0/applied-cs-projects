from django.urls import path
from . import views

urlpatterns = [
    path('', views.TenantListView.as_view(), name='tenant-list'),
    path('<uuid:tenant_id>/', views.TenantDetailView.as_view(), name='tenant-detail'),
    path('<uuid:tenant_id>/members/', views.TenantMemberListView.as_view(), name='tenant-members'),
    path('invitations/<str:token>/accept/', views.AcceptInvitationView.as_view(), name='accept-invitation'),
]
