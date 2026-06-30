"""Tests for utils: settings and logging configuration."""

import logging

import structlog

from ml_orchestrator.utils.config import Settings, get_settings
from ml_orchestrator.utils.logging import setup_logging


def test_settings_defaults():
    s = Settings()
    assert s.host == "0.0.0.0"
    assert s.port == 8000
    assert s.debug is False
    assert s.log_level == "INFO"
    assert s.scheduling_policy == "priority"
    assert s.checkpoint_keep_last_n == 3
    assert s.redis_url is None


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_PORT", "9999")
    monkeypatch.setenv("ORCHESTRATOR_DEBUG", "true")
    monkeypatch.setenv("ORCHESTRATOR_SCHEDULING_POLICY", "fair_share")
    s = Settings()
    assert s.port == 9999
    assert s.debug is True
    assert s.scheduling_policy == "fair_share"


def test_get_settings_is_cached():
    a = get_settings()
    b = get_settings()
    assert a is b  # lru_cache returns the same instance


def test_setup_logging_console():
    # basicConfig is a no-op once handlers exist, so we don't assert the root
    # level; we verify the configuration path runs and structlog is usable.
    setup_logging(level="DEBUG", json_format=False)
    structlog.get_logger(__name__).info("hello")


def test_setup_logging_json():
    setup_logging(level="WARNING", json_format=True)
    structlog.get_logger(__name__).warning("warned")
