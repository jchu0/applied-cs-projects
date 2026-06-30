"""Models for task scheduling."""

from django.db import models
from django.conf import settings
from django.utils import timezone
import uuid
import json


class TaskStatus(models.TextChoices):
    """Task execution status."""
    PENDING = 'pending', 'Pending'
    SCHEDULED = 'scheduled', 'Scheduled'
    RUNNING = 'running', 'Running'
    SUCCESS = 'success', 'Success'
    FAILED = 'failed', 'Failed'
    CANCELLED = 'cancelled', 'Cancelled'
    RETRYING = 'retrying', 'Retrying'


class TaskPriority(models.IntegerChoices):
    """Task priority levels."""
    LOW = 1, 'Low'
    NORMAL = 5, 'Normal'
    HIGH = 10, 'High'
    CRITICAL = 20, 'Critical'


class ScheduledTask(models.Model):
    """A scheduled task for background execution."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)

    # Task configuration
    task_name = models.CharField(max_length=255, help_text="Celery task name")
    args = models.JSONField(default=list, blank=True)
    kwargs = models.JSONField(default=dict, blank=True)

    # Scheduling
    schedule_type = models.CharField(
        max_length=20,
        choices=[
            ('once', 'One-time'),
            ('interval', 'Interval'),
            ('cron', 'Cron Expression'),
        ],
        default='once'
    )
    cron_expression = models.CharField(max_length=100, blank=True, null=True)
    interval_seconds = models.IntegerField(null=True, blank=True)
    next_run = models.DateTimeField(null=True, blank=True)
    last_run = models.DateTimeField(null=True, blank=True)

    # Status
    status = models.CharField(
        max_length=20,
        choices=TaskStatus.choices,
        default=TaskStatus.PENDING
    )
    priority = models.IntegerField(
        choices=TaskPriority.choices,
        default=TaskPriority.NORMAL
    )
    enabled = models.BooleanField(default=True)

    # Retry configuration
    max_retries = models.IntegerField(default=3)
    retry_count = models.IntegerField(default=0)
    retry_delay_seconds = models.IntegerField(default=60)

    # Ownership
    tenant = models.ForeignKey(
        'tenants.Tenant',
        on_delete=models.CASCADE,
        related_name='scheduled_tasks',
        null=True,
        blank=True
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name='created_tasks'
    )

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-priority', 'next_run']
        indexes = [
            models.Index(fields=['status', 'enabled', 'next_run']),
            models.Index(fields=['tenant', 'status']),
            models.Index(fields=['task_name']),
        ]

    def __str__(self):
        return f"{self.name} ({self.task_name})"

    def schedule_next_run(self):
        """Calculate and set the next run time."""
        if self.schedule_type == 'once':
            self.enabled = False
        elif self.schedule_type == 'interval' and self.interval_seconds:
            self.next_run = timezone.now() + timezone.timedelta(seconds=self.interval_seconds)
        elif self.schedule_type == 'cron' and self.cron_expression:
            from croniter import croniter
            cron = croniter(self.cron_expression, timezone.now())
            self.next_run = cron.get_next(timezone.datetime)
        self.save()


class TaskExecution(models.Model):
    """Record of a task execution."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    task = models.ForeignKey(
        ScheduledTask,
        on_delete=models.CASCADE,
        related_name='executions'
    )

    # Celery task tracking
    celery_task_id = models.CharField(max_length=255, blank=True, null=True)

    # Execution details
    status = models.CharField(
        max_length=20,
        choices=TaskStatus.choices,
        default=TaskStatus.RUNNING
    )
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    duration_seconds = models.FloatField(null=True, blank=True)

    # Results
    result = models.JSONField(null=True, blank=True)
    error_message = models.TextField(blank=True)
    traceback = models.TextField(blank=True)

    # Resource usage
    memory_mb = models.FloatField(null=True, blank=True)
    cpu_seconds = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ['-started_at']
        indexes = [
            models.Index(fields=['task', 'status']),
            models.Index(fields=['started_at']),
            models.Index(fields=['celery_task_id']),
        ]

    def __str__(self):
        return f"{self.task.name} - {self.status} at {self.started_at}"

    def complete(self, status: str, result=None, error=None, traceback=None):
        """Mark the execution as complete."""
        self.status = status
        self.completed_at = timezone.now()
        self.duration_seconds = (self.completed_at - self.started_at).total_seconds()

        if result is not None:
            self.result = result
        if error:
            self.error_message = str(error)
        if traceback:
            self.traceback = traceback

        self.save()


class TaskQueue(models.Model):
    """Custom task queue configuration."""

    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)

    # Queue configuration
    max_workers = models.IntegerField(default=4)
    max_memory_mb = models.IntegerField(default=1024)
    priority = models.IntegerField(default=5)

    # Rate limiting
    rate_limit = models.CharField(
        max_length=50,
        blank=True,
        help_text="Rate limit e.g., '100/m' for 100 per minute"
    )

    # Status
    is_active = models.BooleanField(default=True)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-priority', 'name']

    def __str__(self):
        return self.name


class CronSchedule(models.Model):
    """Predefined cron schedule for common patterns."""

    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    cron_expression = models.CharField(max_length=100)

    # Ownership
    tenant = models.ForeignKey(
        'tenants.Tenant',
        on_delete=models.CASCADE,
        related_name='cron_schedules',
        null=True,
        blank=True
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['name']
        unique_together = ['tenant', 'name']

    def __str__(self):
        return f"{self.name}: {self.cron_expression}"

    @classmethod
    def get_presets(cls):
        """Return common schedule presets."""
        return {
            'every_minute': '* * * * *',
            'every_5_minutes': '*/5 * * * *',
            'every_hour': '0 * * * *',
            'every_day_midnight': '0 0 * * *',
            'every_day_noon': '0 12 * * *',
            'every_monday': '0 0 * * 1',
            'first_of_month': '0 0 1 * *',
        }
