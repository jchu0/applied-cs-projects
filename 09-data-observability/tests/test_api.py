"""Tests for the REST API."""

import pytest
from datetime import datetime

# Try to import fastapi test client, skip if not available
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient
from observability.api import app, _tables, _anomalies, _alerts, _lineage_graph


@pytest.fixture(autouse=True)
def clear_state():
    """Clear global state before each test."""
    _tables.clear()
    _anomalies.clear()
    _alerts.clear()
    # Reset lineage graph
    _lineage_graph._nodes.clear()
    _lineage_graph._edges.clear()
    yield


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthCheck:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """Test health check returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data


class TestTableEndpoints:
    """Tests for table management endpoints."""

    def test_list_tables_empty(self, client):
        """Test listing tables when none exist."""
        response = client.get("/tables")
        assert response.status_code == 200
        data = response.json()
        assert data["tables"] == []
        assert data["total"] == 0

    def test_register_table(self, client):
        """Test registering a new table."""
        response = client.post("/tables", json={
            "name": "users",
            "database": "prod",
            "schema_name": "public",
            "columns": [
                {"name": "id", "data_type": "integer", "nullable": False},
                {"name": "email", "data_type": "string", "nullable": False},
            ]
        })
        assert response.status_code == 200
        data = response.json()
        assert data["table_id"] == "prod.public.users"
        assert data["status"] == "registered"

    def test_get_table(self, client):
        """Test getting table details."""
        # First register a table
        client.post("/tables", json={
            "name": "users",
            "database": "prod",
            "schema_name": "public",
            "columns": [
                {"name": "id", "data_type": "integer"},
            ]
        })

        # Then get it
        response = client.get("/tables/prod.public.users")
        assert response.status_code == 200
        data = response.json()
        assert data["table_id"] == "prod.public.users"
        assert data["name"] == "users"
        assert data["database"] == "prod"
        assert len(data["columns"]) == 1

    def test_get_table_not_found(self, client):
        """Test getting non-existent table."""
        response = client.get("/tables/nonexistent")
        assert response.status_code == 404

    def test_list_tables_with_filter(self, client):
        """Test listing tables with database filter."""
        # Register tables in different databases
        client.post("/tables", json={
            "name": "users",
            "database": "prod",
            "schema_name": "public",
            "columns": [{"name": "id", "data_type": "integer"}]
        })
        client.post("/tables", json={
            "name": "orders",
            "database": "staging",
            "schema_name": "public",
            "columns": [{"name": "id", "data_type": "integer"}]
        })

        # Filter by database
        response = client.get("/tables?database=prod")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["tables"][0]["database"] == "prod"


class TestLineageEndpoints:
    """Tests for lineage endpoints."""

    def test_get_table_lineage(self, client):
        """Test getting table lineage."""
        # Register table
        client.post("/tables", json={
            "name": "users",
            "database": "prod",
            "schema_name": "public",
            "columns": [{"name": "id", "data_type": "integer"}]
        })

        response = client.get("/tables/prod.public.users/lineage")
        assert response.status_code == 200
        data = response.json()
        assert data["table_id"] == "prod.public.users"
        assert data["upstream"] == []
        assert data["downstream"] == []

    def test_add_lineage(self, client):
        """Test adding lineage relationship."""
        # Register tables
        client.post("/tables", json={
            "name": "raw_users",
            "database": "prod",
            "schema_name": "staging",
            "columns": [{"name": "id", "data_type": "integer"}]
        })
        client.post("/tables", json={
            "name": "users",
            "database": "prod",
            "schema_name": "public",
            "columns": [{"name": "id", "data_type": "integer"}]
        })

        # Add lineage
        response = client.post(
            "/tables/prod.public.users/lineage",
            params={"upstream_table": "prod.staging.raw_users"}
        )
        assert response.status_code == 200

        # Verify lineage
        response = client.get("/tables/prod.public.users/lineage")
        data = response.json()
        assert "prod.staging.raw_users" in data["upstream"]


class TestHealthEndpoints:
    """Tests for health score endpoints."""

    def test_get_table_health(self, client):
        """Test getting table health score."""
        # Register table
        client.post("/tables", json={
            "name": "users",
            "database": "prod",
            "schema_name": "public",
            "columns": [{"name": "id", "data_type": "integer"}]
        })

        response = client.get("/tables/prod.public.users/health")
        assert response.status_code == 200
        data = response.json()
        assert data["table_id"] == "prod.public.users"
        assert "overall_score" in data
        assert "freshness_score" in data
        assert "volume_score" in data
        assert "schema_score" in data
        assert "quality_score" in data
        assert "trend" in data


class TestAnomalyEndpoints:
    """Tests for anomaly endpoints."""

    def test_list_anomalies_empty(self, client):
        """Test listing anomalies when none exist."""
        response = client.get("/anomalies")
        assert response.status_code == 200
        data = response.json()
        assert data["anomalies"] == []
        assert data["total"] == 0

    def test_run_detection(self, client):
        """Test running anomaly detection."""
        # Register table
        client.post("/tables", json={
            "name": "users",
            "database": "prod",
            "schema_name": "public",
            "columns": [{"name": "id", "data_type": "integer"}]
        })

        response = client.post("/tables/prod.public.users/detect")
        assert response.status_code == 200
        data = response.json()
        assert data["table_id"] == "prod.public.users"
        assert "anomalies_detected" in data


class TestPIIEndpoints:
    """Tests for PII detection endpoints."""

    def test_scan_pii(self, client):
        """Test scanning for PII."""
        # Register table with PII column
        client.post("/tables", json={
            "name": "users",
            "database": "prod",
            "schema_name": "public",
            "columns": [
                {"name": "id", "data_type": "integer"},
                {"name": "email", "data_type": "string"},
                {"name": "ssn", "data_type": "string"},
            ]
        })

        response = client.get("/tables/prod.public.users/pii")
        assert response.status_code == 200
        data = response.json()
        assert data["table_id"] == "prod.public.users"
        assert "pii_columns" in data
        assert "recommendations" in data


class TestImpactAnalysis:
    """Tests for impact analysis."""

    def test_get_impact_analysis(self, client):
        """Test getting impact analysis."""
        # Register tables
        client.post("/tables", json={
            "name": "users",
            "database": "prod",
            "schema_name": "public",
            "columns": [{"name": "id", "data_type": "integer"}]
        })

        response = client.get("/impact/prod.public.users")
        assert response.status_code == 200
        data = response.json()
        assert data["table_id"] == "prod.public.users"
        assert "affected_tables" in data
        assert "affected_pipelines" in data
        assert "total_downstream" in data

    def test_impact_analysis_not_found(self, client):
        """Test impact analysis for non-existent table."""
        response = client.get("/impact/nonexistent")
        assert response.status_code == 404
