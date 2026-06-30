"""
Core views for health checks and common endpoints.
"""
from rest_framework import views
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from django.db import connection


class HealthCheckView(views.APIView):
    """Health check endpoint."""
    permission_classes = [AllowAny]

    def get(self, request):
        # Check database connection
        try:
            with connection.cursor() as cursor:
                cursor.execute('SELECT 1')
            db_status = 'healthy'
        except Exception as e:
            db_status = f'unhealthy: {str(e)}'

        return Response({
            'status': 'ok',
            'database': db_status,
        })


class ReadyCheckView(views.APIView):
    """Readiness check endpoint."""
    permission_classes = [AllowAny]

    def get(self, request):
        return Response({'status': 'ready'})
