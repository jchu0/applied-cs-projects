"""Resources app configuration."""

from django.apps import AppConfig


class ResourcesConfig(AppConfig):
    """Configuration for resources app."""

    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.resources'
    verbose_name = 'Resource Management'
