"""Tests for Feature Platform REST API."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from fastapi.testclient import TestClient

from feature_platform.api.main import create_app, set_feature_store, get_feature_store
from feature_platform.api.models import (
    HealthStatus,
    FeatureViewCreateRequest,
    EntityModel,
    FeatureDefinitionModel,
    OnlineFeaturesRequest,
    HistoricalFeaturesRequest,
    SearchRequest,
    WriteOnlineFeaturesRequest,
    WriteOfflineFeaturesRequest,
)
from feature_platform.store.feature_store import FeatureStore
from feature_platform.core.models import (
    Entity,
    Feature,
    FeatureView,
    FeatureVector,
    FeatureValue,
    DataType,
)
from feature_platform.store.offline import FeatureData


@pytest.fixture
def mock_store():
    """Create a mock feature store."""
    store = Mock(spec=FeatureStore)
    store.list_feature_views.return_value = []
    return store


@pytest.fixture
def sample_entity():
    """Create a sample entity."""
    return Entity(name="user", join_keys=["user_id"], description="User entity")


@pytest.fixture
def sample_feature_view(sample_entity):
    """Create a sample feature view."""
    return FeatureView(
        name="user_features",
        entities=[sample_entity],
        schema=[
            Feature(name="age", dtype=DataType.INT64, description="User age"),
            Feature(name="income", dtype=DataType.FLOAT64, description="Annual income"),
        ],
        ttl=timedelta(hours=24),
        description="User feature view",
        tags=["user", "demo"],
        owner="test@example.com",
    )


@pytest.fixture
def client(mock_store):
    """Create a test client with mock store."""
    set_feature_store(mock_store)
    app = create_app()
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check_healthy(self, client, mock_store):
        """Test health check returns healthy status."""
        mock_store.list_feature_views.return_value = []

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data
        assert "components" in data

    def test_health_check_with_components(self, client, mock_store):
        """Test health check includes component statuses."""
        response = client.get("/health")

        data = response.json()
        assert "api" in data["components"]
        assert data["components"]["api"] == "healthy"


class TestFeatureViewEndpoints:
    """Tests for feature view CRUD endpoints."""

    def test_list_feature_views_empty(self, client, mock_store):
        """Test listing feature views when empty."""
        mock_store.list_feature_views.return_value = []

        response = client.get("/feature-views")

        assert response.status_code == 200
        data = response.json()
        assert data["feature_views"] == []
        assert data["total"] == 0

    def test_list_feature_views(self, client, mock_store, sample_feature_view):
        """Test listing feature views."""
        mock_store.list_feature_views.return_value = [sample_feature_view]

        response = client.get("/feature-views")

        assert response.status_code == 200
        data = response.json()
        assert len(data["feature_views"]) == 1
        assert data["feature_views"][0]["name"] == "user_features"
        assert data["total"] == 1

    def test_get_feature_view(self, client, mock_store, sample_feature_view):
        """Test getting a single feature view."""
        mock_store.get_feature_view.return_value = sample_feature_view

        response = client.get("/feature-views/user_features")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "user_features"
        assert len(data["entities"]) == 1
        assert len(data["features"]) == 2

    def test_get_feature_view_not_found(self, client, mock_store):
        """Test getting non-existent feature view."""
        mock_store.get_feature_view.return_value = None

        response = client.get("/feature-views/nonexistent")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_create_feature_view(self, client, mock_store):
        """Test creating a feature view."""
        request_data = {
            "name": "test_features",
            "entities": [
                {"name": "user", "join_keys": ["user_id"], "description": "User"}
            ],
            "features": [
                {"name": "score", "dtype": "float64", "description": "User score"}
            ],
            "ttl_seconds": 3600,
            "description": "Test feature view",
        }

        response = client.post("/feature-views", json=request_data)

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "test_features"
        mock_store.apply.assert_called_once()

    def test_delete_feature_view(self, client, mock_store, sample_feature_view):
        """Test deleting a feature view."""
        mock_store.get_feature_view.return_value = sample_feature_view
        mock_store.delete_feature_view.return_value = True

        response = client.delete("/feature-views/user_features")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["deleted"] == "user_features"

    def test_delete_feature_view_not_found(self, client, mock_store):
        """Test deleting non-existent feature view."""
        mock_store.get_feature_view.return_value = None

        response = client.delete("/feature-views/nonexistent")

        assert response.status_code == 404


class TestOnlineFeaturesEndpoint:
    """Tests for online features endpoint."""

    def test_get_online_features(self, client, mock_store):
        """Test getting online features."""
        mock_store.get_online_features.return_value = {
            "user_features:age": [25, 30, 35],
            "user_features:income": [50000.0, 75000.0, 100000.0],
        }

        request_data = {
            "feature_refs": ["user_features:age", "user_features:income"],
            "entity_ids": {"user_id": [1, 2, 3]},
        }

        response = client.post("/features/online", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "features" in data
        assert len(data["features"]["user_features:age"]) == 3
        assert "metadata" in data
        assert "duration_ms" in data["metadata"]


class TestHistoricalFeaturesEndpoint:
    """Tests for historical features endpoint."""

    def test_get_historical_features(self, client, mock_store):
        """Test getting historical features."""
        mock_data = FeatureData(
            entity_ids={"user_id": [1, 2, 3]},
            features={
                "age": [25, 30, 35],
                "income": [50000.0, 75000.0, 100000.0],
            },
            timestamps=None,
            feature_view="user_features",
        )
        mock_store.get_historical_features.return_value = mock_data

        request_data = {
            "entity_df": {
                "user_id": [1, 2, 3],
                "_timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
            },
            "feature_refs": ["user_features:age", "user_features:income"],
        }

        response = client.post("/features/historical", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "entity_ids" in data
        assert "features" in data


class TestMaterializeEndpoint:
    """Tests for materialization endpoint."""

    def test_materialize_features(self, client, mock_store, sample_feature_view):
        """Test materializing features."""
        mock_store.get_feature_view.return_value = sample_feature_view
        mock_store.materialize.return_value = 100

        response = client.post("/features/materialize/user_features")

        assert response.status_code == 200
        data = response.json()
        assert data["feature_view"] == "user_features"
        assert data["rows_materialized"] == 100
        assert "duration_ms" in data

    def test_materialize_not_found(self, client, mock_store):
        """Test materializing non-existent feature view."""
        mock_store.get_feature_view.return_value = None

        response = client.post("/features/materialize/nonexistent")

        assert response.status_code == 404


class TestStatisticsEndpoint:
    """Tests for feature statistics endpoint."""

    def test_get_statistics(self, client, mock_store, sample_feature_view):
        """Test getting feature statistics."""
        mock_store.get_feature_view.return_value = sample_feature_view
        mock_store.get_feature_statistics.return_value = {
            "age": {"count": 100, "null_count": 0, "mean": 30.0, "std": 10.0, "min": 18.0, "max": 65.0},
            "income": {"count": 100, "null_count": 5, "mean": 75000.0, "std": 25000.0, "min": 30000.0, "max": 150000.0},
        }

        response = client.get("/features/user_features/statistics")

        assert response.status_code == 200
        data = response.json()
        assert data["feature_view"] == "user_features"
        assert "age" in data["statistics"]
        assert "income" in data["statistics"]
        assert data["statistics"]["age"]["mean"] == 30.0


class TestSearchEndpoint:
    """Tests for search endpoint."""

    def test_search_features(self, client, mock_store):
        """Test searching for features."""
        mock_store.search_features.return_value = [
            {
                "feature_view": "user_features",
                "feature_name": "age",
                "dtype": "int64",
                "description": "User age",
                "score": 0.95,
                "tags": ["user"],
            }
        ]

        request_data = {"query": "age", "limit": 10}

        response = client.post("/search", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "age"
        assert len(data["results"]) == 1
        assert data["results"][0]["feature_name"] == "age"


class TestWriteFeaturesEndpoints:
    """Tests for write features endpoints."""

    def test_write_online_features(self, client, mock_store, sample_feature_view):
        """Test writing online features."""
        mock_store.get_feature_view.return_value = sample_feature_view

        request_data = {
            "feature_view": "user_features",
            "entity_id": {"user_id": 1},
            "features": {"age": 25, "income": 50000.0},
        }

        response = client.post("/features/online/write", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["rows_written"] == 1

    def test_write_offline_features(self, client, mock_store, sample_feature_view):
        """Test writing offline features."""
        mock_store.get_feature_view.return_value = sample_feature_view

        request_data = {
            "feature_view": "user_features",
            "data": {
                "user_id": [1, 2, 3],
                "age": [25, 30, 35],
                "income": [50000.0, 75000.0, 100000.0],
            },
        }

        response = client.post("/features/offline/write", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["rows_written"] == 3


class TestFeatureVectorEndpoint:
    """Tests for feature vector endpoint."""

    def test_get_feature_vector(self, client, mock_store, sample_feature_view):
        """Test getting a feature vector."""
        mock_store.get_feature_view.return_value = sample_feature_view
        mock_store.get_feature_vector.return_value = FeatureVector(
            entity_id={"user_id": 1},
            features=[
                FeatureValue(name="age", value=25, timestamp=datetime.utcnow()),
                FeatureValue(name="income", value=50000.0, timestamp=datetime.utcnow()),
            ],
            timestamp=datetime.utcnow(),
        )

        response = client.get(
            "/features/user_features/vector",
            params={"entity_id": '{"user_id": 1}'},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["entity_id"]["user_id"] == 1
        assert len(data["features"]) == 2


class TestValidationEndpoint:
    """Tests for validation endpoint."""

    def test_validate_feature_view_success(self, client, mock_store, sample_feature_view):
        """Test validating a valid feature view."""
        mock_store.validate_feature_view.return_value = []

        response = client.post("/feature-views/user_features/validate")

        assert response.status_code == 200
        data = response.json()
        assert data["feature_view"] == "user_features"
        assert data["is_valid"] is True
        assert data["errors"] == []

    def test_validate_feature_view_errors(self, client, mock_store):
        """Test validating an invalid feature view."""
        mock_store.validate_feature_view.return_value = [
            "Entity not registered: user"
        ]

        response = client.post("/feature-views/user_features/validate")

        assert response.status_code == 200
        data = response.json()
        assert data["is_valid"] is False
        assert len(data["errors"]) == 1


class TestAPIModels:
    """Tests for API Pydantic models."""

    def test_health_response_model(self):
        """Test HealthResponse model."""
        from feature_platform.api.models import HealthResponse

        response = HealthResponse(
            status=HealthStatus.HEALTHY,
            version="0.1.0",
            components={"api": "healthy"},
        )

        assert response.status == HealthStatus.HEALTHY
        assert response.version == "0.1.0"
        assert response.timestamp is not None

    def test_feature_view_create_request(self):
        """Test FeatureViewCreateRequest model."""
        request = FeatureViewCreateRequest(
            name="test_view",
            entities=[EntityModel(name="user", join_keys=["user_id"])],
            features=[FeatureDefinitionModel(name="score", dtype="float64")],
        )

        assert request.name == "test_view"
        assert len(request.entities) == 1
        assert len(request.features) == 1

    def test_online_features_request(self):
        """Test OnlineFeaturesRequest model."""
        request = OnlineFeaturesRequest(
            feature_refs=["view:feature1", "view:feature2"],
            entity_ids={"id": [1, 2, 3]},
        )

        assert len(request.feature_refs) == 2
        assert len(request.entity_ids["id"]) == 3


class TestAppConfiguration:
    """Tests for app configuration."""

    def test_create_app_default(self):
        """Test creating app with defaults."""
        app = create_app()

        assert app.title == "Feature Platform API"
        assert app.version == "0.1.0"

    def test_create_app_custom_title(self):
        """Test creating app with custom title."""
        app = create_app(title="Custom API", description="Custom description")

        assert app.title == "Custom API"
        assert app.description == "Custom description"

    def test_cors_middleware(self):
        """Test CORS middleware is configured."""
        app = create_app(cors_origins=["http://localhost:3000"])

        # Check middleware is registered
        assert len(app.user_middleware) > 0


class TestFeatureStoreManagement:
    """Tests for feature store management functions."""

    def test_set_and_get_feature_store(self):
        """Test setting and getting feature store."""
        store = Mock(spec=FeatureStore)
        set_feature_store(store)

        retrieved = get_feature_store()
        assert retrieved is store

    def test_get_feature_store_creates_default(self):
        """Test getting feature store creates default if none set."""
        set_feature_store(None)

        store = get_feature_store()
        assert store is not None
        assert isinstance(store, FeatureStore)
