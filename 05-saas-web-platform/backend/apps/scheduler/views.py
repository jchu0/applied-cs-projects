"""Views for scheduler app."""

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.db.models import Count, Q
from django.utils import timezone

from .models import ScheduledTask, TaskExecution, TaskQueue, CronSchedule, TaskStatus
from .serializers import (
    ScheduledTaskSerializer,
    ScheduledTaskCreateSerializer,
    TaskExecutionSerializer,
    TaskQueueSerializer,
    CronScheduleSerializer,
)
from .services import TaskScheduler, TaskQueueManager


class ScheduledTaskViewSet(viewsets.ModelViewSet):
    """ViewSet for scheduled tasks."""

    permission_classes = [IsAuthenticated]
    serializer_class = ScheduledTaskSerializer

    def get_queryset(self):
        """Filter tasks by tenant."""
        queryset = ScheduledTask.objects.all()
        tenant = getattr(self.request, 'tenant', None)
        if tenant:
            queryset = queryset.filter(tenant=tenant)
        return queryset.select_related('created_by', 'tenant')

    def get_serializer_class(self):
        if self.action == 'create':
            return ScheduledTaskCreateSerializer
        return ScheduledTaskSerializer

    @action(detail=True, methods=['post'])
    def run(self, request, pk=None):
        """Run a task immediately."""
        task = self.get_object()
        scheduler = TaskScheduler(tenant=task.tenant)
        execution = scheduler.run_task(task)

        return Response({
            'execution_id': str(execution.id),
            'status': execution.status,
        })

    @action(detail=True, methods=['post'])
    def cancel(self, request, pk=None):
        """Cancel a task."""
        task = self.get_object()
        scheduler = TaskScheduler(tenant=task.tenant)
        success = scheduler.cancel_task(task)

        return Response({
            'success': success,
            'status': task.status,
        })

    @action(detail=True, methods=['post'])
    def retry(self, request, pk=None):
        """Retry a failed task."""
        task = self.get_object()
        scheduler = TaskScheduler(tenant=task.tenant)
        execution = scheduler.retry_task(task)

        if execution:
            return Response({
                'execution_id': str(execution.id),
                'status': execution.status,
            })
        return Response(
            {'error': 'Max retries exceeded'},
            status=status.HTTP_400_BAD_REQUEST
        )

    @action(detail=True, methods=['get'])
    def executions(self, request, pk=None):
        """Get execution history for a task."""
        task = self.get_object()
        executions = task.executions.all()[:20]
        serializer = TaskExecutionSerializer(executions, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['get'])
    def status(self, request, pk=None):
        """Get detailed status of a task."""
        task = self.get_object()
        scheduler = TaskScheduler(tenant=task.tenant)
        return Response(scheduler.get_task_status(task))

    @action(detail=False, methods=['get'])
    def stats(self, request):
        """Get task statistics."""
        queryset = self.get_queryset()

        stats = queryset.aggregate(
            total=Count('id'),
            pending=Count('id', filter=Q(status=TaskStatus.PENDING)),
            scheduled=Count('id', filter=Q(status=TaskStatus.SCHEDULED)),
            running=Count('id', filter=Q(status=TaskStatus.RUNNING)),
            success=Count('id', filter=Q(status=TaskStatus.SUCCESS)),
            failed=Count('id', filter=Q(status=TaskStatus.FAILED)),
        )

        total_completed = stats['success'] + stats['failed']
        success_rate = (
            stats['success'] / total_completed * 100
            if total_completed > 0 else 0
        )

        return Response({
            'total_tasks': stats['total'],
            'pending_tasks': stats['pending'] + stats['scheduled'],
            'running_tasks': stats['running'],
            'completed_tasks': stats['success'],
            'failed_tasks': stats['failed'],
            'success_rate': round(success_rate, 2),
        })

    @action(detail=False, methods=['get'])
    def upcoming(self, request):
        """Get upcoming scheduled tasks."""
        queryset = self.get_queryset().filter(
            enabled=True,
            status__in=[TaskStatus.SCHEDULED, TaskStatus.PENDING],
            next_run__gte=timezone.now()
        ).order_by('next_run')[:10]

        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)


class TaskExecutionViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for task executions (read-only)."""

    permission_classes = [IsAuthenticated]
    serializer_class = TaskExecutionSerializer

    def get_queryset(self):
        """Filter executions by tenant."""
        queryset = TaskExecution.objects.all()
        tenant = getattr(self.request, 'tenant', None)
        if tenant:
            queryset = queryset.filter(task__tenant=tenant)
        return queryset.select_related('task')


class TaskQueueViewSet(viewsets.ModelViewSet):
    """ViewSet for task queues."""

    permission_classes = [IsAuthenticated]
    serializer_class = TaskQueueSerializer
    queryset = TaskQueue.objects.all()

    @action(detail=False, methods=['get'])
    def stats(self, request):
        """Get queue statistics."""
        manager = TaskQueueManager()
        return Response(manager.get_queue_stats())

    @action(detail=False, methods=['get'])
    def workers(self, request):
        """Get worker status."""
        manager = TaskQueueManager()
        return Response(manager.get_worker_status())

    @action(detail=True, methods=['post'])
    def purge(self, request, pk=None):
        """Purge a queue."""
        queue = self.get_object()
        manager = TaskQueueManager()
        count = manager.purge_queue(queue.name)
        return Response({'purged': count})


class CronScheduleViewSet(viewsets.ModelViewSet):
    """ViewSet for cron schedules."""

    permission_classes = [IsAuthenticated]
    serializer_class = CronScheduleSerializer

    def get_queryset(self):
        """Filter by tenant or show global schedules."""
        queryset = CronSchedule.objects.all()
        tenant = getattr(self.request, 'tenant', None)
        if tenant:
            queryset = queryset.filter(Q(tenant=tenant) | Q(tenant__isnull=True))
        return queryset

    @action(detail=False, methods=['get'])
    def presets(self, request):
        """Get common cron presets."""
        return Response(CronSchedule.get_presets())
