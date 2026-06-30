"""Services for task scheduling."""

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from django.utils import timezone
from django.db import transaction
from celery import current_app
from celery.result import AsyncResult
import logging

from .models import ScheduledTask, TaskExecution, TaskStatus, TaskPriority, TaskQueue

logger = logging.getLogger(__name__)


class TaskScheduler:
    """Service for managing scheduled tasks."""

    def __init__(self, tenant=None):
        self.tenant = tenant

    def create_task(
        self,
        name: str,
        task_name: str,
        args: List = None,
        kwargs: Dict = None,
        schedule_type: str = 'once',
        cron_expression: str = None,
        interval_seconds: int = None,
        run_at: datetime = None,
        priority: int = TaskPriority.NORMAL,
        max_retries: int = 3,
        created_by=None,
    ) -> ScheduledTask:
        """Create a new scheduled task."""
        task = ScheduledTask(
            name=name,
            task_name=task_name,
            args=args or [],
            kwargs=kwargs or {},
            schedule_type=schedule_type,
            cron_expression=cron_expression,
            interval_seconds=interval_seconds,
            priority=priority,
            max_retries=max_retries,
            tenant=self.tenant,
            created_by=created_by,
        )

        # Set initial next_run
        if run_at:
            task.next_run = run_at
        elif schedule_type == 'once':
            task.next_run = timezone.now()
        elif schedule_type == 'interval' and interval_seconds:
            task.next_run = timezone.now() + timedelta(seconds=interval_seconds)
        elif schedule_type == 'cron' and cron_expression:
            try:
                from croniter import croniter
                cron = croniter(cron_expression, timezone.now())
                task.next_run = cron.get_next(datetime)
            except ImportError:
                task.next_run = timezone.now()

        task.status = TaskStatus.SCHEDULED
        task.save()

        logger.info(f"Created scheduled task: {task.name} ({task.id})")
        return task

    def run_task(self, task: ScheduledTask) -> TaskExecution:
        """Execute a scheduled task immediately."""
        # Create execution record
        execution = TaskExecution.objects.create(
            task=task,
            status=TaskStatus.RUNNING
        )

        # Update task status
        task.status = TaskStatus.RUNNING
        task.last_run = timezone.now()
        task.save()

        try:
            # Submit to Celery
            result = current_app.send_task(
                task.task_name,
                args=task.args,
                kwargs=task.kwargs,
                task_id=str(execution.id),
            )
            execution.celery_task_id = result.id
            execution.save()

            logger.info(f"Submitted task {task.name} to Celery: {result.id}")

        except Exception as e:
            execution.complete(
                status=TaskStatus.FAILED,
                error=str(e),
            )
            task.status = TaskStatus.FAILED
            task.save()
            logger.error(f"Failed to submit task {task.name}: {e}")

        return execution

    def cancel_task(self, task: ScheduledTask) -> bool:
        """Cancel a scheduled task."""
        if task.status in [TaskStatus.SUCCESS, TaskStatus.CANCELLED]:
            return False

        # Cancel any running executions
        running = task.executions.filter(status=TaskStatus.RUNNING)
        for execution in running:
            if execution.celery_task_id:
                try:
                    current_app.control.revoke(
                        execution.celery_task_id,
                        terminate=True
                    )
                except Exception as e:
                    logger.warning(f"Failed to revoke task: {e}")

            execution.complete(status=TaskStatus.CANCELLED)

        task.status = TaskStatus.CANCELLED
        task.enabled = False
        task.save()

        logger.info(f"Cancelled task: {task.name}")
        return True

    def retry_task(self, task: ScheduledTask) -> Optional[TaskExecution]:
        """Retry a failed task."""
        if task.retry_count >= task.max_retries:
            logger.warning(f"Task {task.name} exceeded max retries")
            return None

        task.retry_count += 1
        task.status = TaskStatus.RETRYING
        task.save()

        return self.run_task(task)

    def get_pending_tasks(self, limit: int = 100) -> List[ScheduledTask]:
        """Get tasks that are due to run."""
        now = timezone.now()
        return list(
            ScheduledTask.objects.filter(
                enabled=True,
                status__in=[TaskStatus.SCHEDULED, TaskStatus.PENDING],
                next_run__lte=now
            ).order_by('-priority', 'next_run')[:limit]
        )

    def process_pending_tasks(self) -> int:
        """Process all pending tasks. Returns count of tasks processed."""
        tasks = self.get_pending_tasks()
        count = 0

        for task in tasks:
            try:
                self.run_task(task)
                count += 1
            except Exception as e:
                logger.error(f"Error processing task {task.name}: {e}")

        return count

    def get_task_status(self, task: ScheduledTask) -> Dict[str, Any]:
        """Get detailed status of a task."""
        latest_execution = task.executions.first()

        return {
            'id': str(task.id),
            'name': task.name,
            'status': task.status,
            'enabled': task.enabled,
            'next_run': task.next_run,
            'last_run': task.last_run,
            'retry_count': task.retry_count,
            'latest_execution': {
                'id': str(latest_execution.id) if latest_execution else None,
                'status': latest_execution.status if latest_execution else None,
                'started_at': latest_execution.started_at if latest_execution else None,
                'completed_at': latest_execution.completed_at if latest_execution else None,
            } if latest_execution else None
        }

    def get_celery_task_result(self, celery_task_id: str) -> Dict[str, Any]:
        """Get the result of a Celery task."""
        result = AsyncResult(celery_task_id)
        return {
            'task_id': celery_task_id,
            'status': result.status,
            'result': result.result if result.successful() else None,
            'traceback': result.traceback if result.failed() else None,
        }


class TaskQueueManager:
    """Service for managing task queues."""

    def get_queue_stats(self, queue_name: str = None) -> Dict[str, Any]:
        """Get statistics for task queues."""
        inspect = current_app.control.inspect()

        stats = {
            'active': {},
            'reserved': {},
            'scheduled': {},
        }

        try:
            active = inspect.active() or {}
            reserved = inspect.reserved() or {}
            scheduled = inspect.scheduled() or {}

            for worker, tasks in active.items():
                if queue_name is None or any(t.get('delivery_info', {}).get('routing_key') == queue_name for t in tasks):
                    stats['active'][worker] = len(tasks)

            for worker, tasks in reserved.items():
                stats['reserved'][worker] = len(tasks)

            for worker, tasks in scheduled.items():
                stats['scheduled'][worker] = len(tasks)

        except Exception as e:
            logger.error(f"Error getting queue stats: {e}")

        return stats

    def purge_queue(self, queue_name: str) -> int:
        """Purge all tasks from a queue."""
        try:
            purged = current_app.control.purge()
            logger.info(f"Purged {purged} tasks from queue {queue_name}")
            return purged
        except Exception as e:
            logger.error(f"Error purging queue: {e}")
            return 0

    def get_worker_status(self) -> Dict[str, Any]:
        """Get status of all Celery workers."""
        inspect = current_app.control.inspect()

        try:
            ping = inspect.ping() or {}
            stats = inspect.stats() or {}

            workers = {}
            for worker_name in ping.keys():
                worker_stats = stats.get(worker_name, {})
                workers[worker_name] = {
                    'online': True,
                    'concurrency': worker_stats.get('pool', {}).get('max-concurrency', 0),
                    'processes': worker_stats.get('pool', {}).get('processes', []),
                    'uptime': worker_stats.get('uptime', 0),
                }

            return {
                'workers': workers,
                'total_workers': len(workers),
            }

        except Exception as e:
            logger.error(f"Error getting worker status: {e}")
            return {'workers': {}, 'total_workers': 0}


def complete_task_execution(
    execution_id: str,
    status: str,
    result: Any = None,
    error: str = None,
    traceback: str = None,
) -> None:
    """Complete a task execution (called by Celery task callbacks)."""
    try:
        execution = TaskExecution.objects.get(id=execution_id)
        execution.complete(
            status=status,
            result=result,
            error=error,
            traceback=traceback,
        )

        # Update parent task
        task = execution.task
        task.status = status if status in [TaskStatus.SUCCESS, TaskStatus.FAILED] else task.status

        # Schedule next run for recurring tasks
        if status == TaskStatus.SUCCESS and task.schedule_type != 'once':
            task.schedule_next_run()
        elif status == TaskStatus.FAILED:
            task.retry_count += 1

        task.save()

    except TaskExecution.DoesNotExist:
        logger.error(f"Task execution not found: {execution_id}")
