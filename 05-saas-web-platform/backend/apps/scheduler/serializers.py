"""Serializers for scheduler app."""

from rest_framework import serializers
from .models import ScheduledTask, TaskExecution, TaskQueue, CronSchedule
from .services import TaskScheduler


class TaskExecutionSerializer(serializers.ModelSerializer):
    """Serializer for task executions."""

    class Meta:
        model = TaskExecution
        fields = [
            'id', 'task', 'celery_task_id', 'status',
            'started_at', 'completed_at', 'duration_seconds',
            'result', 'error_message', 'memory_mb', 'cpu_seconds'
        ]
        read_only_fields = ['id', 'task', 'started_at']


class ScheduledTaskSerializer(serializers.ModelSerializer):
    """Serializer for scheduled tasks."""

    latest_execution = serializers.SerializerMethodField()
    executions_count = serializers.SerializerMethodField()

    class Meta:
        model = ScheduledTask
        fields = [
            'id', 'name', 'description', 'task_name',
            'args', 'kwargs', 'schedule_type',
            'cron_expression', 'interval_seconds',
            'next_run', 'last_run', 'status', 'priority',
            'enabled', 'max_retries', 'retry_count',
            'created_at', 'updated_at',
            'latest_execution', 'executions_count'
        ]
        read_only_fields = [
            'id', 'status', 'retry_count',
            'last_run', 'created_at', 'updated_at'
        ]

    def get_latest_execution(self, obj):
        execution = obj.executions.first()
        if execution:
            return TaskExecutionSerializer(execution).data
        return None

    def get_executions_count(self, obj):
        return obj.executions.count()


class ScheduledTaskCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating scheduled tasks."""

    run_at = serializers.DateTimeField(required=False, write_only=True)

    class Meta:
        model = ScheduledTask
        fields = [
            'name', 'description', 'task_name',
            'args', 'kwargs', 'schedule_type',
            'cron_expression', 'interval_seconds',
            'priority', 'max_retries', 'run_at'
        ]

    def validate(self, data):
        schedule_type = data.get('schedule_type', 'once')

        if schedule_type == 'cron' and not data.get('cron_expression'):
            raise serializers.ValidationError(
                "cron_expression is required for cron schedule type"
            )

        if schedule_type == 'interval' and not data.get('interval_seconds'):
            raise serializers.ValidationError(
                "interval_seconds is required for interval schedule type"
            )

        if data.get('cron_expression'):
            try:
                from croniter import croniter
                croniter(data['cron_expression'])
            except Exception:
                raise serializers.ValidationError(
                    "Invalid cron expression"
                )

        return data

    def create(self, validated_data):
        run_at = validated_data.pop('run_at', None)
        request = self.context.get('request')
        tenant = getattr(request, 'tenant', None) if request else None

        scheduler = TaskScheduler(tenant=tenant)
        return scheduler.create_task(
            **validated_data,
            run_at=run_at,
            created_by=request.user if request else None,
        )


class TaskQueueSerializer(serializers.ModelSerializer):
    """Serializer for task queues."""

    class Meta:
        model = TaskQueue
        fields = [
            'id', 'name', 'description', 'max_workers',
            'max_memory_mb', 'priority', 'rate_limit',
            'is_active', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']


class CronScheduleSerializer(serializers.ModelSerializer):
    """Serializer for cron schedules."""

    class Meta:
        model = CronSchedule
        fields = ['id', 'name', 'description', 'cron_expression', 'created_at']
        read_only_fields = ['id', 'created_at']


class TaskRunSerializer(serializers.Serializer):
    """Serializer for running a task immediately."""

    task_id = serializers.UUIDField()


class TaskStatsSerializer(serializers.Serializer):
    """Serializer for task statistics."""

    total_tasks = serializers.IntegerField()
    pending_tasks = serializers.IntegerField()
    running_tasks = serializers.IntegerField()
    completed_tasks = serializers.IntegerField()
    failed_tasks = serializers.IntegerField()
    success_rate = serializers.FloatField()
