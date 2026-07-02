"""Regression tests for JWT secret configuration.

JWT_SECRET_KEY must never fall back to a hardcoded value in base settings:
- base.py reads it from the environment with no default,
- development.py supplies a clearly-marked dev-only fallback,
- production.py raises ImproperlyConfigured when it is missing.

These tests import the settings modules directly (no database needed).
"""
import importlib
import sys
from pathlib import Path

import pytest

# config/__init__.py imports the Celery app, so importing any settings module
# requires celery to be installed.
pytest.importorskip('celery')

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def _fresh_import(module_name):
    """Import a settings module with a clean module cache."""
    for name in list(sys.modules):
        if name == 'config' or name.startswith('config.'):
            del sys.modules[name]
    return importlib.import_module(module_name)


def test_base_has_no_hardcoded_jwt_fallback(monkeypatch):
    monkeypatch.delenv('JWT_SECRET_KEY', raising=False)
    base = _fresh_import('config.settings.base')
    assert base.JWT_SECRET_KEY is None


def test_base_reads_jwt_secret_from_env(monkeypatch):
    monkeypatch.setenv('JWT_SECRET_KEY', 'env-provided-secret')
    base = _fresh_import('config.settings.base')
    assert base.JWT_SECRET_KEY == 'env-provided-secret'


def test_development_supplies_dev_only_default(monkeypatch):
    monkeypatch.delenv('JWT_SECRET_KEY', raising=False)
    dev = _fresh_import('config.settings.development')
    assert dev.JWT_SECRET_KEY
    assert dev.JWT_SECRET_KEY != 'your-secret-key'
    assert dev.JWT_SECRET_KEY.startswith('django-insecure-')
    assert dev.DEBUG is True


def test_development_prefers_env_secret(monkeypatch):
    monkeypatch.setenv('JWT_SECRET_KEY', 'env-provided-secret')
    dev = _fresh_import('config.settings.development')
    assert dev.JWT_SECRET_KEY == 'env-provided-secret'


def test_production_requires_jwt_secret(monkeypatch):
    pytest.importorskip('django')
    pytest.importorskip('dj_database_url')
    pytest.importorskip('sentry_sdk')
    from django.core.exceptions import ImproperlyConfigured

    monkeypatch.setenv('DJANGO_SECRET_KEY', 'prod-django-secret')
    monkeypatch.delenv('JWT_SECRET_KEY', raising=False)

    with pytest.raises(ImproperlyConfigured, match='JWT_SECRET_KEY'):
        _fresh_import('config.settings.production')


def test_production_accepts_env_jwt_secret(monkeypatch):
    pytest.importorskip('django')
    pytest.importorskip('dj_database_url')
    pytest.importorskip('sentry_sdk')

    monkeypatch.setenv('DJANGO_SECRET_KEY', 'prod-django-secret')
    monkeypatch.setenv('JWT_SECRET_KEY', 'prod-jwt-secret')

    prod = _fresh_import('config.settings.production')
    assert prod.JWT_SECRET_KEY == 'prod-jwt-secret'
    assert prod.DEBUG is False
