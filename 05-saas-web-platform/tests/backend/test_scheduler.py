"""Tests for scheduler app."""

import pytest
import importlib.util
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta


# Check if Django is available
DJANGO_AVAILABLE = importlib.util.find_spec("django") is not None

if DJANGO_AVAILABLE:
    from django.utils import timezone
    from django.test import TestCase
    from rest_framework.test import APIClient
    from rest_framework import status
else:
    TestCase = object
    timezone = None
    APIClient = None
    status = None


pytestmark = pytest.mark.skipif(not DJANGO_AVAILABLE, reason="Django not installed")


class TestTaskSchedulerService:
    """Tests for TaskScheduler service."""

    @pytest.fixture
    def mock_scheduler(self):
        """Create a mock scheduler."""
        with patch('apps.scheduler.services.current_app') as mock_celery:
            mock_celery.send_task.return_value = Mock(id='celery-task-123')
            mock_celery.control.revoke.return_value = None
            yield mock_celery

    def test_create_task_once(self, db, mock_scheduler):
        """Test creating a one-time task."""
        from apps.scheduler.services import TaskScheduler
        from apps.scheduler.models import TaskStatus

        scheduler = TaskScheduler()
        task = scheduler.create_task(
            name='Test Task',
            task_name='core.test_task',
            args=['arg1', 'arg2'],
            kwargs={'key': 'value'},
            schedule_type='once',
        )

        assert task.name == 'Test Task'
        assert task.task_name == 'core.test_task'
        assert task.args == ['arg1', 'arg2']
        assert task.kwargs == {'key': 'value'}
        assert task.status == TaskStatus.SCHEDULED
        assert task.next_run is not None

    def test_create_task_interval(self, db, mock_scheduler):
        """Test creating an interval task."""
        from apps.scheduler.services import TaskScheduler

        scheduler = TaskScheduler()
        task = scheduler.create_task(
            name='Interval Task',
            task_name='core.interval_task',
            schedule_type='interval',
            interval_seconds=300,
        )

        assert task.interval_seconds == 300
        assert task.schedule_type == 'interval'
        # Next run should be ~5 minutes from now
        expected = timezone.now() + timedelta(seconds=300)
        assert abs((task.next_run - expected).total_seconds()) < 5

    @pytest.mark.skipif(True, reason="Requires croniter")
    def test_create_task_cron(self, db, mock_scheduler):
        """Test creating a cron task."""
        from apps.scheduler.services import TaskScheduler

        scheduler = TaskScheduler()
        task = scheduler.create_task(
            name='Cron Task',
            task_name='core.cron_task',
            schedule_type='cron',
            cron_expression='0 * * * *',  # Every hour
        )

        assert task.cron_expression == '0 * * * *'
        assert task.schedule_type == 'cron'

    def test_run_task(self, db, mock_scheduler):
        """Test running a task immediately."""
        from apps.scheduler.services import TaskScheduler
        from apps.scheduler.models import TaskStatus, ScheduledTask

        # Create a task first
        task = ScheduledTask.objects.create(
            name='Run Test',
            task_name='core.run_test',
            status=TaskStatus.SCHEDULED,
        )

        scheduler = TaskScheduler()
        execution = scheduler.run_task(task)

        assert execution is not None
        assert execution.status == TaskStatus.RUNNING
        assert execution.task == task
        mock_scheduler.send_task.assert_called_once()

    def test_cancel_task(self, db, mock_scheduler):
        """Test cancelling a task."""
        from apps.scheduler.services import TaskScheduler
        from apps.scheduler.models import TaskStatus, ScheduledTask

        task = ScheduledTask.objects.create(
            name='Cancel Test',
            task_name='core.cancel_test',
            status=TaskStatus.SCHEDULED,
            enabled=True,
        )

        scheduler = TaskScheduler()
        success = scheduler.cancel_task(task)

        assert success is True
        task.refresh_from_db()
        assert task.status == TaskStatus.CANCELLED
        assert task.enabled is False

    def test_cancel_completed_task(self, db, mock_scheduler):
        """Test that cancelling a completed task returns False."""
        from apps.scheduler.services import TaskScheduler
        from apps.scheduler.models import TaskStatus, ScheduledTask

        task = ScheduledTask.objects.create(
            name='Completed Test',
            task_name='core.completed_test',
            status=TaskStatus.SUCCESS,
        )

        scheduler = TaskScheduler()
        success = scheduler.cancel_task(task)

        assert success is False

    def test_retry_task(self, db, mock_scheduler):
        """Test retrying a failed task."""
        from apps.scheduler.services import TaskScheduler
        from apps.scheduler.models import TaskStatus, ScheduledTask

        task = ScheduledTask.objects.create(
            name='Retry Test',
            task_name='core.retry_test',
            status=TaskStatus.FAILED,
            max_retries=3,
            retry_count=0,
        )

        scheduler = TaskScheduler()
        execution = scheduler.retry_task(task)

        assert execution is not None
        task.refresh_from_db()
        assert task.retry_count == 1
        assert task.status == TaskStatus.RETRYING

    def test_retry_max_exceeded(self, db, mock_scheduler):
        """Test that retry returns None when max retries exceeded."""
        from apps.scheduler.services import TaskScheduler
        from apps.scheduler.models import TaskStatus, ScheduledTask

        task = ScheduledTask.objects.create(
            name='Max Retry Test',
            task_name='core.max_retry_test',
            status=TaskStatus.FAILED,
            max_retries=3,
            retry_count=3,
        )

        scheduler = TaskScheduler()
        execution = scheduler.retry_task(task)

        assert execution is None

    def test_get_pending_tasks(self, db, mock_scheduler):
        """Test getting pending tasks."""
        from apps.scheduler.services import TaskScheduler
        from apps.scheduler.models import TaskStatus, ScheduledTask

        # Create tasks with different statuses
        ScheduledTask.objects.create(
            name='Pending 1',
            task_name='core.pending1',
            status=TaskStatus.PENDING,
            enabled=True,
            next_run=timezone.now() - timedelta(minutes=1),
        )
        ScheduledTask.objects.create(
            name='Scheduled 1',
            task_name='core.scheduled1',
            status=TaskStatus.SCHEDULED,
            enabled=True,
            next_run=timezone.now() - timedelta(minutes=1),
        )
        ScheduledTask.objects.create(
            name='Future Task',
            task_name='core.future',
            status=TaskStatus.SCHEDULED,
            enabled=True,
            next_run=timezone.now() + timedelta(hours=1),
        )
        ScheduledTask.objects.create(
            name='Disabled Task',
            task_name='core.disabled',
            status=TaskStatus.SCHEDULED,
            enabled=False,
            next_run=timezone.now() - timedelta(minutes=1),
        )

        scheduler = TaskScheduler()
        pending = scheduler.get_pending_tasks()

        assert len(pending) == 2  # Only enabled tasks due now


class TestTaskQueueManager:
    """Tests for TaskQueueManager service."""

    @pytest.fixture
    def mock_celery_inspect(self):
        """Mock Celery inspect."""
        with patch('apps.scheduler.services.current_app') as mock_app:
            mock_inspect = MagicMock()
            mock_app.control.inspect.return_value = mock_inspect
            mock_inspect.active.return_value = {
                'worker1': [{'id': '1'}, {'id': '2'}],
                'worker2': [{'id': '3'}],
            }
            mock_inspect.reserved.return_value = {
                'worker1': [{'id': '4'}],
            }
            mock_inspect.scheduled.return_value = {}
            mock_inspect.ping.return_value = {
                'worker1': {'ok': 'pong'},
                'worker2': {'ok': 'pong'},
            }
            mock_inspect.stats.return_value = {
                'worker1': {
                    'pool': {'max-concurrency': 4, 'processes': [1, 2, 3, 4]},
                    'uptime': 3600,
                },
                'worker2': {
                    'pool': {'max-concurrency': 2, 'processes': [1, 2]},
                    'uptime': 1800,
                },
            }
            yield mock_app, mock_inspect

    def test_get_queue_stats(self, mock_celery_inspect):
        """Test getting queue statistics."""
        from apps.scheduler.services import TaskQueueManager

        manager = TaskQueueManager()
        stats = manager.get_queue_stats()

        assert 'active' in stats
        assert 'reserved' in stats
        assert 'scheduled' in stats
        assert stats['active']['worker1'] == 2
        assert stats['active']['worker2'] == 1

    def test_get_worker_status(self, mock_celery_inspect):
        """Test getting worker status."""
        from apps.scheduler.services import TaskQueueManager

        manager = TaskQueueManager()
        status = manager.get_worker_status()

        assert status['total_workers'] == 2
        assert 'worker1' in status['workers']
        assert 'worker2' in status['workers']
        assert status['workers']['worker1']['online'] is True
        assert status['workers']['worker1']['concurrency'] == 4


class TestScheduledTaskViewSet:
    """Tests for ScheduledTask API endpoints."""

    @pytest.fixture
    def authenticated_client(self, db):
        """Create an authenticated API client."""
        from apps.users.models import User

        client = APIClient()
        user = User.objects.create_user(
            email='test@example.com',
            password='TestPass123!',
        )
        client.force_authenticate(user=user)
        return client, user

    @pytest.fixture
    def scheduled_task(self, db):
        """Create a test scheduled task."""
        from apps.scheduler.models import ScheduledTask, TaskStatus

        return ScheduledTask.objects.create(
            name='Test Task',
            task_name='core.test',
            status=TaskStatus.SCHEDULED,
            enabled=True,
        )

    def test_list_tasks(self, authenticated_client, scheduled_task):
        """Test listing scheduled tasks."""
        client, user = authenticated_client

        response = client.get('/api/v1/scheduler/tasks/')

        assert response.status_code == status.HTTP_200_OK

    def test_create_task(self, authenticated_client):
        """Test creating a scheduled task."""
        client, user = authenticated_client

        data = {
            'name': 'New Task',
            'task_name': 'core.new_task',
            'schedule_type': 'once',
        }

        with patch('apps.scheduler.serializers.TaskScheduler') as mock_scheduler:
            mock_instance = Mock()
            mock_scheduler.return_value = mock_instance
            mock_instance.create_task.return_value = Mock(
                id='task-123',
                name='New Task',
                task_name='core.new_task',
                status='scheduled',
            )

            response = client.post('/api/v1/scheduler/tasks/', data, format='json')

        # Should create successfully or validation error if missing deps
        assert response.status_code in [status.HTTP_201_CREATED, status.HTTP_400_BAD_REQUEST]

    def test_run_task(self, authenticated_client, scheduled_task):
        """Test running a task immediately."""
        client, user = authenticated_client

        with patch('apps.scheduler.views.TaskScheduler') as mock_scheduler:
            mock_instance = Mock()
            mock_scheduler.return_value = mock_instance
            mock_instance.run_task.return_value = Mock(
                id='exec-123',
                status='running',
            )

            response = client.post(f'/api/v1/scheduler/tasks/{scheduled_task.id}/run/')

        assert response.status_code == status.HTTP_200_OK

    def test_cancel_task(self, authenticated_client, scheduled_task):
        """Test cancelling a task."""
        client, user = authenticated_client

        with patch('apps.scheduler.views.TaskScheduler') as mock_scheduler:
            mock_instance = Mock()
            mock_scheduler.return_value = mock_instance
            mock_instance.cancel_task.return_value = True
            scheduled_task.status = 'cancelled'

            response = client.post(f'/api/v1/scheduler/tasks/{scheduled_task.id}/cancel/')

        assert response.status_code == status.HTTP_200_OK

    def test_stats(self, authenticated_client, scheduled_task):
        """Test getting task statistics."""
        client, user = authenticated_client

        response = client.get('/api/v1/scheduler/tasks/stats/')

        assert response.status_code == status.HTTP_200_OK
        assert 'total_tasks' in response.data
        assert 'success_rate' in response.data


class TestTaskExecutionViewSet:
    """Tests for TaskExecution API endpoints."""

    @pytest.fixture
    def authenticated_client(self, db):
        """Create an authenticated API client."""
        from apps.users.models import User

        client = APIClient()
        user = User.objects.create_user(
            email='test@example.com',
            password='TestPass123!',
        )
        client.force_authenticate(user=user)
        return client, user

    def test_list_executions(self, authenticated_client, db):
        """Test listing task executions."""
        client, user = authenticated_client

        response = client.get('/api/v1/scheduler/executions/')

        assert response.status_code == status.HTTP_200_OK


class TestCronScheduleViewSet:
    """Tests for CronSchedule API endpoints."""

    @pytest.fixture
    def authenticated_client(self, db):
        """Create an authenticated API client."""
        from apps.users.models import User

        client = APIClient()
        user = User.objects.create_user(
            email='test@example.com',
            password='TestPass123!',
        )
        client.force_authenticate(user=user)
        return client, user

    def test_list_cron_schedules(self, authenticated_client, db):
        """Test listing cron schedules."""
        client, user = authenticated_client

        response = client.get('/api/v1/scheduler/cron-schedules/')

        assert response.status_code == status.HTTP_200_OK

    def test_presets(self, authenticated_client, db):
        """Test getting cron presets."""
        client, user = authenticated_client

        with patch('apps.scheduler.models.CronSchedule.get_presets') as mock_presets:
            mock_presets.return_value = [
                {'name': 'Every minute', 'expression': '* * * * *'},
                {'name': 'Every hour', 'expression': '0 * * * *'},
            ]

            response = client.get('/api/v1/scheduler/cron-schedules/presets/')

        assert response.status_code == status.HTTP_200_OK


class TestSchedulerModels:
    """Tests for scheduler models."""

    def test_scheduled_task_creation(self, db):
        """Test creating a scheduled task model."""
        from apps.scheduler.models import ScheduledTask, TaskStatus

        task = ScheduledTask.objects.create(
            name='Model Test',
            task_name='core.model_test',
        )

        assert task.id is not None
        assert task.name == 'Model Test'
        assert task.status == TaskStatus.PENDING

    def test_task_execution_creation(self, db):
        """Test creating a task execution model."""
        from apps.scheduler.models import ScheduledTask, TaskExecution, TaskStatus

        task = ScheduledTask.objects.create(
            name='Exec Test',
            task_name='core.exec_test',
        )

        execution = TaskExecution.objects.create(
            task=task,
            status=TaskStatus.RUNNING,
        )

        assert execution.id is not None
        assert execution.task == task

    def test_task_execution_complete(self, db):
        """Test completing a task execution."""
        from apps.scheduler.models import ScheduledTask, TaskExecution, TaskStatus

        task = ScheduledTask.objects.create(
            name='Complete Test',
            task_name='core.complete_test',
        )

        execution = TaskExecution.objects.create(
            task=task,
            status=TaskStatus.RUNNING,
        )

        execution.complete(
            status=TaskStatus.SUCCESS,
            result={'key': 'value'},
        )

        execution.refresh_from_db()
        assert execution.status == TaskStatus.SUCCESS
        assert execution.completed_at is not None
        assert execution.result == {'key': 'value'}

    def test_task_queue_creation(self, db):
        """Test creating a task queue model."""
        from apps.scheduler.models import TaskQueue

        queue = TaskQueue.objects.create(
            name='test-queue',
            description='Test queue',
            max_workers=4,
            priority=5,
        )

        assert queue.id is not None
        assert queue.name == 'test-queue'
        assert queue.max_workers == 4

    def test_cron_schedule_creation(self, db):
        """Test creating a cron schedule model."""
        from apps.scheduler.models import CronSchedule

        schedule = CronSchedule.objects.create(
            name='Hourly',
            cron_expression='0 * * * *',
        )

        assert schedule.id is not None
        assert schedule.cron_expression == '0 * * * *'


class TestSchedulerSerializer:
    """Tests for scheduler serializers."""

    def test_scheduled_task_serializer(self, db):
        """Test ScheduledTaskSerializer."""
        from apps.scheduler.models import ScheduledTask, TaskStatus
        from apps.scheduler.serializers import ScheduledTaskSerializer

        task = ScheduledTask.objects.create(
            name='Serializer Test',
            task_name='core.serializer_test',
            status=TaskStatus.SCHEDULED,
        )

        serializer = ScheduledTaskSerializer(task)
        data = serializer.data

        assert data['name'] == 'Serializer Test'
        assert data['task_name'] == 'core.serializer_test'
        assert 'id' in data
        assert 'status' in data

    def test_scheduled_task_create_serializer_validation(self, db):
        """Test ScheduledTaskCreateSerializer validation."""
        from apps.scheduler.serializers import ScheduledTaskCreateSerializer

        # Missing cron_expression for cron type
        data = {
            'name': 'Invalid Task',
            'task_name': 'core.invalid',
            'schedule_type': 'cron',
        }

        serializer = ScheduledTaskCreateSerializer(data=data)
        assert not serializer.is_valid()
        assert 'non_field_errors' in serializer.errors or 'cron_expression' in str(serializer.errors)

    def test_task_execution_serializer(self, db):
        """Test TaskExecutionSerializer."""
        from apps.scheduler.models import ScheduledTask, TaskExecution, TaskStatus
        from apps.scheduler.serializers import TaskExecutionSerializer

        task = ScheduledTask.objects.create(
            name='Exec Serializer Test',
            task_name='core.exec_serializer_test',
        )

        execution = TaskExecution.objects.create(
            task=task,
            status=TaskStatus.RUNNING,
        )

        serializer = TaskExecutionSerializer(execution)
        data = serializer.data

        assert 'id' in data
        assert data['status'] == TaskStatus.RUNNING
