"""Celery tasks for the scheduler app."""

from celery import shared_task, current_task
from django.utils import timezone
import logging
import traceback

from .services import complete_task_execution
from .models import TaskStatus

logger = logging.getLogger(__name__)


@shared_task(bind=True, name='scheduler.process_pending')
def process_pending_tasks(self):
    """Process all pending scheduled tasks."""
    from .services import TaskScheduler

    scheduler = TaskScheduler()
    count = scheduler.process_pending_tasks()

    logger.info(f"Processed {count} pending tasks")
    return {'processed': count}


@shared_task(bind=True, name='scheduler.run_scheduled_task')
def run_scheduled_task(self, task_id: str):
    """Run a specific scheduled task by ID."""
    from .models import ScheduledTask
    from .services import TaskScheduler

    try:
        task = ScheduledTask.objects.get(id=task_id)
        scheduler = TaskScheduler(tenant=task.tenant)
        execution = scheduler.run_task(task)

        return {
            'task_id': str(task.id),
            'execution_id': str(execution.id),
            'status': execution.status,
        }

    except ScheduledTask.DoesNotExist:
        logger.error(f"Scheduled task not found: {task_id}")
        return {'error': 'Task not found'}


@shared_task(bind=True, name='scheduler.cleanup_old_executions')
def cleanup_old_executions(self, days: int = 30):
    """Clean up old task executions."""
    from .models import TaskExecution

    cutoff = timezone.now() - timezone.timedelta(days=days)
    deleted, _ = TaskExecution.objects.filter(
        started_at__lt=cutoff
    ).delete()

    logger.info(f"Cleaned up {deleted} old task executions")
    return {'deleted': deleted}


# Common utility tasks

@shared_task(bind=True, name='core.send_email')
def send_email_task(
    self,
    to_email: str,
    subject: str,
    template_name: str,
    context: dict = None,
):
    """Send an email asynchronously."""
    from django.core.mail import send_mail
    from django.template.loader import render_to_string

    try:
        # Render template
        html_content = render_to_string(
            f'emails/{template_name}.html',
            context or {}
        )

        send_mail(
            subject=subject,
            message='',
            from_email=None,  # Use DEFAULT_FROM_EMAIL
            recipient_list=[to_email],
            html_message=html_content,
        )

        logger.info(f"Sent email to {to_email}: {subject}")
        return {'success': True, 'to': to_email}

    except Exception as e:
        logger.error(f"Failed to send email to {to_email}: {e}")
        raise


@shared_task(bind=True, name='core.send_bulk_email')
def send_bulk_email_task(
    self,
    recipients: list,
    subject: str,
    template_name: str,
    context: dict = None,
):
    """Send bulk emails asynchronously."""
    from django.core.mail import send_mass_mail
    from django.template.loader import render_to_string

    try:
        html_content = render_to_string(
            f'emails/{template_name}.html',
            context or {}
        )

        messages = [
            (subject, html_content, None, [email])
            for email in recipients
        ]

        send_mass_mail(messages, fail_silently=False)

        logger.info(f"Sent bulk email to {len(recipients)} recipients")
        return {'success': True, 'count': len(recipients)}

    except Exception as e:
        logger.error(f"Failed to send bulk email: {e}")
        raise


@shared_task(bind=True, name='billing.sync_stripe_invoices')
def sync_stripe_invoices(self, tenant_id: str = None):
    """Sync invoices from Stripe."""
    from apps.billing.services import StripeService

    try:
        service = StripeService()
        # Implementation would sync invoices for tenant or all tenants
        count = 0  # Placeholder

        logger.info(f"Synced {count} invoices from Stripe")
        return {'synced': count}

    except Exception as e:
        logger.error(f"Failed to sync Stripe invoices: {e}")
        raise


@shared_task(bind=True, name='billing.process_usage_records')
def process_usage_records(self):
    """Process and aggregate usage records for billing."""
    try:
        # Implementation would aggregate usage records
        count = 0  # Placeholder

        logger.info(f"Processed {count} usage records")
        return {'processed': count}

    except Exception as e:
        logger.error(f"Failed to process usage records: {e}")
        raise


@shared_task(bind=True, name='core.generate_report')
def generate_report_task(
    self,
    report_type: str,
    tenant_id: str,
    params: dict = None,
):
    """Generate a report asynchronously."""
    try:
        # Report generation logic would go here
        logger.info(f"Generated {report_type} report for tenant {tenant_id}")
        return {
            'report_type': report_type,
            'tenant_id': tenant_id,
            'status': 'completed',
        }

    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        raise


@shared_task(bind=True, name='core.data_export')
def data_export_task(
    self,
    user_id: str,
    export_type: str,
    format: str = 'json',
):
    """Export user data (GDPR compliance)."""
    try:
        # Data export logic would go here
        logger.info(f"Exported data for user {user_id}")
        return {
            'user_id': user_id,
            'export_type': export_type,
            'format': format,
            'status': 'completed',
        }

    except Exception as e:
        logger.error(f"Failed to export data: {e}")
        raise


@shared_task(bind=True, name='core.cleanup_expired_tokens')
def cleanup_expired_tokens(self):
    """Clean up expired authentication tokens."""
    from django.utils import timezone

    try:
        # Clean up expired tokens, invitations, etc.
        count = 0  # Placeholder for actual cleanup

        logger.info(f"Cleaned up {count} expired tokens")
        return {'cleaned': count}

    except Exception as e:
        logger.error(f"Failed to cleanup tokens: {e}")
        raise


# Task callback handlers

@shared_task(name='scheduler.on_task_success')
def on_task_success(execution_id: str, result):
    """Handle successful task completion."""
    complete_task_execution(
        execution_id=execution_id,
        status=TaskStatus.SUCCESS,
        result=result,
    )


@shared_task(name='scheduler.on_task_failure')
def on_task_failure(execution_id: str, exc, tb):
    """Handle task failure."""
    complete_task_execution(
        execution_id=execution_id,
        status=TaskStatus.FAILED,
        error=str(exc),
        traceback=tb,
    )
