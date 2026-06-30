"""Training app configuration."""

from django.apps import AppConfig


class TrainingConfig(AppConfig):
    """Configuration for training app."""

    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.training'
    verbose_name = 'Distributed Training'
