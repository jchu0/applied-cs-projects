"""Django app configuration for scheduler app."""

from django.apps import AppConfig


class SchedulerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.scheduler'
    verbose_name = 'Task Scheduler'

    def ready(self):
        """Import signals and task autodiscovery on app ready."""
        # Import tasks to ensure they're registered with Celery
        from . import tasks  # noqa: F401
