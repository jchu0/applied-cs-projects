"""Tests for API hardening: auth, rate limiting, and request timeouts."""

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from feature_platform.api.main import create_app, set_feature_store
from feature_platform.store.feature_store import FeatureStore


@pytest.fixture
def mock_store():
    """Create a mock feature store."""
    store = Mock(spec=FeatureStore)
    store.list_feature_views.return_value = []
    return store


def _client(mock_store) -> TestClient:
    """Build a TestClient after env has been configured for the test."""
    set_feature_store(mock_store)
    return TestClient(create_app())


class TestAuthEnabled:
    """Auth enforced when API_KEYS is set."""

    def test_401_when_key_missing(self, mock_store, monkeypatch):
        monkeypatch.setenv("API_KEYS", "secret-key")
        client = _client(mock_store)

        response = client.get("/feature-views")

        assert response.status_code == 401
        assert response.headers.get("www-authenticate") == "Bearer"

    def test_401_when_key_bad(self, mock_store, monkeypatch):
        monkeypatch.setenv("API_KEYS", "secret-key")
        client = _client(mock_store)

        response = client.get(
            "/feature-views", headers={"X-API-Key": "wrong-key"}
        )

        assert response.status_code == 401

    def test_200_with_valid_bearer_key(self, mock_store, monkeypatch):
        monkeypatch.setenv("API_KEYS", "secret-key")
        client = _client(mock_store)

        response = client.get(
            "/feature-views",
            headers={"Authorization": "Bearer secret-key"},
        )

        assert response.status_code == 200

    def test_200_with_valid_header_key(self, mock_store, monkeypatch):
        monkeypatch.setenv("API_KEYS", "k1,k2")
        client = _client(mock_store)

        response = client.get(
            "/feature-views", headers={"X-API-Key": "k2"}
        )

        assert response.status_code == 200

    def test_health_and_openapi_open_without_key(self, mock_store, monkeypatch):
        monkeypatch.setenv("API_KEYS", "secret-key")
        client = _client(mock_store)

        assert client.get("/health").status_code == 200
        assert client.get("/openapi.json").status_code == 200


class TestAuthDisabled:
    """No auth required when API_KEYS is unset."""

    def test_open_when_unset(self, mock_store, monkeypatch):
        monkeypatch.delenv("API_KEYS", raising=False)
        client = _client(mock_store)

        assert client.get("/feature-views").status_code == 200


class TestRateLimiting:
    """429 returned when the per-minute budget is exceeded."""

    def test_429_over_low_limit(self, mock_store, monkeypatch):
        monkeypatch.delenv("API_KEYS", raising=False)
        monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "3")
        client = _client(mock_store)

        statuses = [client.get("/feature-views").status_code for _ in range(5)]

        assert statuses[:3] == [200, 200, 200]
        assert 429 in statuses
        # Retry-After present on the throttled response.
        response = client.get("/feature-views")
        assert response.status_code == 429
        assert "retry-after" in {k.lower() for k in response.headers}

    def test_health_exempt_from_rate_limit(self, mock_store, monkeypatch):
        monkeypatch.delenv("API_KEYS", raising=False)
        monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1")
        client = _client(mock_store)

        for _ in range(5):
            assert client.get("/health").status_code == 200
