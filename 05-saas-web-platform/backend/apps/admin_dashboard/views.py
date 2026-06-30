"""
Admin dashboard views for platform management.
"""
from rest_framework import status, views
from rest_framework.response import Response
from rest_framework.permissions import IsAdminUser
from django.db.models import Count, Sum
from django.db.models.functions import TruncDate
from django.utils import timezone
from datetime import timedelta

from apps.users.models import User
from apps.tenants.models import Tenant, TenantMembership
from apps.billing.models import Subscription, Invoice, Plan
from apps.core.models import AuditLog


class AdminDashboardView(views.APIView):
    """Admin dashboard home."""
    permission_classes = [IsAdminUser]

    def get(self, request):
        # User stats
        total_users = User.objects.count()
        active_users = User.objects.filter(is_active=True).count()

        # Tenant stats
        total_tenants = Tenant.objects.count()

        # Subscription stats
        active_subscriptions = Subscription.objects.filter(
            status=Subscription.Status.ACTIVE
        ).count()

        # Recent activity
        recent_users = User.objects.order_by('-created_at')[:5]
        recent_tenants = Tenant.objects.order_by('-created_at')[:5]

        return Response({
            'summary': {
                'total_users': total_users,
                'active_users': active_users,
                'total_tenants': total_tenants,
                'active_subscriptions': active_subscriptions,
            },
            'recent_users': [
                {'id': str(u.id), 'email': u.email, 'created_at': u.created_at}
                for u in recent_users
            ],
            'recent_tenants': [
                {'id': str(t.id), 'name': t.name, 'created_at': t.created_at}
                for t in recent_tenants
            ],
        })


class AdminStatsView(views.APIView):
    """Get admin dashboard statistics."""
    permission_classes = [IsAdminUser]

    def get(self, request):
        # User stats
        total_users = User.objects.count()
        active_users = User.objects.filter(is_active=True).count()
        users_last_7_days = User.objects.filter(
            created_at__gte=timezone.now() - timedelta(days=7)
        ).count()

        # Tenant stats
        total_tenants = Tenant.objects.count()
        active_tenants = Tenant.objects.filter(is_active=True).count()

        # Subscription stats
        total_subscriptions = Subscription.objects.count()
        active_subscriptions = Subscription.objects.filter(
            status=Subscription.Status.ACTIVE
        ).count()
        trialing = Subscription.objects.filter(
            status=Subscription.Status.TRIALING
        ).count()

        # Revenue stats (from paid invoices)
        total_revenue = Invoice.objects.filter(
            status=Invoice.Status.PAID
        ).aggregate(total=Sum('total'))['total'] or 0

        mrr = Invoice.objects.filter(
            status=Invoice.Status.PAID,
            created_at__gte=timezone.now() - timedelta(days=30)
        ).aggregate(total=Sum('total'))['total'] or 0

        return Response({
            'users': {
                'total': total_users,
                'active': active_users,
                'new_last_7_days': users_last_7_days,
            },
            'tenants': {
                'total': total_tenants,
                'active': active_tenants,
            },
            'subscriptions': {
                'total': total_subscriptions,
                'active': active_subscriptions,
                'trialing': trialing,
            },
            'revenue': {
                'total': float(total_revenue),
                'mrr': float(mrr),
            },
        })


class AdminUsersView(views.APIView):
    """List and manage users (admin only)."""
    permission_classes = [IsAdminUser]

    def get(self, request):
        users = User.objects.all().order_by('-created_at')

        # Simple filtering
        email = request.query_params.get('email')
        if email:
            users = users.filter(email__icontains=email)

        is_active = request.query_params.get('is_active')
        if is_active is not None:
            users = users.filter(is_active=is_active == 'true')

        # Pagination
        page = int(request.query_params.get('page', 1))
        per_page = int(request.query_params.get('per_page', 20))
        start = (page - 1) * per_page
        end = start + per_page

        total = users.count()
        users = users[start:end]

        data = [{
            'id': str(user.id),
            'email': user.email,
            'full_name': user.full_name,
            'is_active': user.is_active,
            'is_staff': user.is_staff,
            'created_at': user.created_at,
            'last_login_at': user.last_login_at,
        } for user in users]

        return Response({
            'users': data,
            'total': total,
            'page': page,
            'per_page': per_page,
        })


class AdminTenantsView(views.APIView):
    """List and manage tenants (admin only)."""
    permission_classes = [IsAdminUser]

    def get(self, request):
        tenants = Tenant.objects.all().order_by('-created_at')

        # Filtering
        name = request.query_params.get('name')
        if name:
            tenants = tenants.filter(name__icontains=name)

        is_active = request.query_params.get('is_active')
        if is_active is not None:
            tenants = tenants.filter(is_active=is_active == 'true')

        # Pagination
        page = int(request.query_params.get('page', 1))
        per_page = int(request.query_params.get('per_page', 20))
        start = (page - 1) * per_page
        end = start + per_page

        total = tenants.count()
        tenants = tenants[start:end]

        data = []
        for tenant in tenants:
            member_count = tenant.memberships.count()
            try:
                subscription = tenant.subscription
                plan_name = subscription.plan.name
                sub_status = subscription.status
            except Subscription.DoesNotExist:
                plan_name = 'None'
                sub_status = 'none'

            data.append({
                'id': str(tenant.id),
                'name': tenant.name,
                'slug': tenant.slug,
                'is_active': tenant.is_active,
                'member_count': member_count,
                'plan': plan_name,
                'subscription_status': sub_status,
                'created_at': tenant.created_at,
            })

        return Response({
            'tenants': data,
            'total': total,
            'page': page,
            'per_page': per_page,
        })


class AdminAuditLogsView(views.APIView):
    """View audit logs (admin only)."""
    permission_classes = [IsAdminUser]

    def get(self, request):
        logs = AuditLog.objects.select_related('user', 'tenant').order_by('-created_at')

        # Filtering
        action = request.query_params.get('action')
        if action:
            logs = logs.filter(action=action)

        resource_type = request.query_params.get('resource_type')
        if resource_type:
            logs = logs.filter(resource_type=resource_type)

        user_id = request.query_params.get('user_id')
        if user_id:
            logs = logs.filter(user_id=user_id)

        tenant_id = request.query_params.get('tenant_id')
        if tenant_id:
            logs = logs.filter(tenant_id=tenant_id)

        # Pagination
        page = int(request.query_params.get('page', 1))
        per_page = int(request.query_params.get('per_page', 50))
        start = (page - 1) * per_page
        end = start + per_page

        total = logs.count()
        logs = logs[start:end]

        data = [{
            'id': str(log.id),
            'action': log.action,
            'resource_type': log.resource_type,
            'resource_id': log.resource_id,
            'user': log.user.email if log.user else None,
            'tenant': log.tenant.name if log.tenant else None,
            'ip_address': log.ip_address,
            'created_at': log.created_at,
        } for log in logs]

        return Response({
            'logs': data,
            'total': total,
            'page': page,
            'per_page': per_page,
        })


class AdminGrowthChartView(views.APIView):
    """Get growth chart data."""
    permission_classes = [IsAdminUser]

    def get(self, request):
        days = int(request.query_params.get('days', 30))
        start_date = timezone.now() - timedelta(days=days)

        # User signups by day
        user_signups = (
            User.objects
            .filter(created_at__gte=start_date)
            .annotate(date=TruncDate('created_at'))
            .values('date')
            .annotate(count=Count('id'))
            .order_by('date')
        )

        # Tenant creations by day
        tenant_creations = (
            Tenant.objects
            .filter(created_at__gte=start_date)
            .annotate(date=TruncDate('created_at'))
            .values('date')
            .annotate(count=Count('id'))
            .order_by('date')
        )

        return Response({
            'user_signups': list(user_signups),
            'tenant_creations': list(tenant_creations),
        })
