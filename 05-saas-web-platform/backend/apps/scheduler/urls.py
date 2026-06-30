"""URL configuration for scheduler app."""

from django.urls import path, include
from rest_framework.routers import DefaultRouter

from .views import (
    ScheduledTaskViewSet,
    TaskExecutionViewSet,
    TaskQueueViewSet,
    CronScheduleViewSet,
)

router = DefaultRouter()
router.register(r'tasks', ScheduledTaskViewSet, basename='scheduled-task')
router.register(r'executions', TaskExecutionViewSet, basename='task-execution')
router.register(r'queues', TaskQueueViewSet, basename='task-queue')
router.register(r'cron-schedules', CronScheduleViewSet, basename='cron-schedule')

urlpatterns = [
    path('', include(router.urls)),
]
