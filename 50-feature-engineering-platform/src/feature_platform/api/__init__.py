"""
Feature Platform REST API.

Provides production-ready REST endpoints for feature serving,
feature view management, and feature discovery.
"""

from feature_platform.api.main import (
    create_app,
    get_feature_store,
    set_feature_store,
)
from feature_platform.api.models import (
    HealthResponse,
    FeatureViewResponse,
    FeatureViewCreateRequest,
    OnlineFeaturesRequest,
    OnlineFeaturesResponse,
    HistoricalFeaturesRequest,
    HistoricalFeaturesResponse,
    MaterializeRequest,
    MaterializeResponse,
    FeatureStatisticsResponse,
    SearchRequest,
    SearchResponse,
    ErrorResponse,
)

__all__ = [
    # App
    "create_app",
    "get_feature_store",
    "set_feature_store",
    # Models
    "HealthResponse",
    "FeatureViewResponse",
    "FeatureViewCreateRequest",
    "OnlineFeaturesRequest",
    "OnlineFeaturesResponse",
    "HistoricalFeaturesRequest",
    "HistoricalFeaturesResponse",
    "MaterializeRequest",
    "MaterializeResponse",
    "FeatureStatisticsResponse",
    "SearchRequest",
    "SearchResponse",
    "ErrorResponse",
]
