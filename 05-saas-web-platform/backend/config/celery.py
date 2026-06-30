"""Celery configuration for SaaS platform."""

import os
from celery import Celery
from celery.schedules import crontab

# Set default Django settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.development')

app = Celery('saas_platform')

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Load task modules from all registered Django apps.
app.autodiscover_tasks()

# Celery Beat schedule for periodic tasks
app.conf.beat_schedule = {
    'process-pending-tasks': {
        'task': 'scheduler.process_pending',
        'schedule': 60.0,  # Every minute
    },
    'cleanup-old-executions': {
        'task': 'scheduler.cleanup_old_executions',
        'schedule': crontab(hour=2, minute=0),  # Daily at 2am
        'kwargs': {'days': 30},
    },
    'sync-stripe-invoices': {
        'task': 'billing.sync_stripe_invoices',
        'schedule': crontab(hour='*/6', minute=0),  # Every 6 hours
    },
    'process-usage-records': {
        'task': 'billing.process_usage_records',
        'schedule': crontab(hour=0, minute=30),  # Daily at 12:30am
    },
    'cleanup-expired-tokens': {
        'task': 'core.cleanup_expired_tokens',
        'schedule': crontab(hour=3, minute=0),  # Daily at 3am
    },
}

# Task routing for different queues
app.conf.task_routes = {
    'core.send_email': {'queue': 'email'},
    'core.send_bulk_email': {'queue': 'email'},
    'billing.*': {'queue': 'billing'},
    'scheduler.*': {'queue': 'default'},
    'core.generate_report': {'queue': 'reports'},
    'core.data_export': {'queue': 'reports'},
}

# Default queue
app.conf.task_default_queue = 'default'

# Task time limits
app.conf.task_time_limit = 3600  # 1 hour hard limit
app.conf.task_soft_time_limit = 3300  # 55 min soft limit

# Task retry settings
app.conf.task_acks_late = True
app.conf.worker_prefetch_multiplier = 4

# Result backend settings
app.conf.result_expires = 86400  # 24 hours


@app.task(bind=True, ignore_result=True)
def debug_task(self):
    """Debug task for testing Celery connection."""
    print(f'Request: {self.request!r}')
