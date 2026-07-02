"""
Pytest configuration for Django tests.
"""
import os
import sys
import importlib.util
import pytest

# Check if Django is available
DJANGO_AVAILABLE = importlib.util.find_spec("django") is not None

if DJANGO_AVAILABLE:
    import django

    # Add the backend directory to the path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../backend'))

    # Configure Django settings
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.development')


def pytest_configure():
    """Configure Django for testing."""
    if DJANGO_AVAILABLE:
        django.setup()


def pytest_collection_modifyitems(config, items):
    """Skip tests that require Django if it's not installed."""
    if not DJANGO_AVAILABLE:
        skip_django = pytest.mark.skip(reason="Django not installed")
        for item in items:
            item.add_marker(skip_django)


@pytest.fixture(scope='session')
def django_db_setup(django_db_setup, django_db_blocker):
    """Set up the test database.

    Defer to pytest-django's own ``django_db_setup`` (which creates the test
    database and applies all migrations). This wrapper exists so a checked-in
    set of app migrations is exercised end-to-end by the suite; without
    migrations the custom-model tables (e.g. ``users``) never get created.
    """
    return


@pytest.fixture
def api_client():
    """Return an API client for testing."""
    from rest_framework.test import APIClient
    return APIClient()


@pytest.fixture
def user_factory(db):
    """Factory for creating test users."""
    from apps.users.models import User

    def create_user(
        email='test@example.com',
        password='TestPass123!',
        **kwargs
    ):
        user = User.objects.create_user(
            email=email,
            password=password,
            **kwargs
        )
        return user

    return create_user


@pytest.fixture
def authenticated_client(api_client, user_factory):
    """Return an authenticated API client."""
    user = user_factory()
    api_client.force_authenticate(user=user)
    return api_client, user


@pytest.fixture
def admin_user(db):
    """Create an admin user."""
    from apps.users.models import User

    user = User.objects.create_user(
        email='admin@example.com',
        password='AdminPass123!',
        is_staff=True,
        is_superuser=True
    )
    return user


@pytest.fixture
def tenant_factory(db):
    """Factory for creating test tenants."""
    from apps.tenants.models import Tenant

    def create_tenant(name='Test Tenant', slug='test-tenant', **kwargs):
        tenant = Tenant.objects.create(
            name=name,
            slug=slug,
            **kwargs
        )
        return tenant

    return create_tenant
