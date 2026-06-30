"""Pydantic models for API request/response schemas."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field


class HealthStatus(str, Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthResponse(BaseModel):
    """Health check response."""
    status: HealthStatus = Field(..., description="Service health status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    components: Dict[str, str] = Field(default_factory=dict, description="Component statuses")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Feature View Models

class FeatureDefinitionModel(BaseModel):
    """Feature definition in a feature view."""
    name: str = Field(..., description="Feature name")
    dtype: str = Field(..., description="Data type")
    description: Optional[str] = Field(None, description="Feature description")
    tags: List[str] = Field(default_factory=list, description="Feature tags")


class EntityModel(BaseModel):
    """Entity definition."""
    name: str = Field(..., description="Entity name")
    join_keys: List[str] = Field(..., description="Join key columns")
    description: Optional[str] = Field(None, description="Entity description")


class FeatureSourceModel(BaseModel):
    """Feature source definition."""
    type: str = Field(..., description="Source type (file, table, stream)")
    path: Optional[str] = Field(None, description="Source path/URI")
    table: Optional[str] = Field(None, description="Table name")
    query: Optional[str] = Field(None, description="SQL query")


class FeatureViewResponse(BaseModel):
    """Feature view response model."""
    name: str = Field(..., description="Feature view name")
    entities: List[EntityModel] = Field(..., description="Associated entities")
    features: List[FeatureDefinitionModel] = Field(..., description="Feature definitions")
    source: Optional[FeatureSourceModel] = Field(None, description="Feature source")
    ttl_seconds: Optional[int] = Field(None, description="Time-to-live in seconds")
    description: Optional[str] = Field(None, description="Feature view description")
    tags: List[str] = Field(default_factory=list, description="Tags")
    owner: Optional[str] = Field(None, description="Owner")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


class FeatureViewCreateRequest(BaseModel):
    """Request to create a feature view."""
    name: str = Field(..., description="Feature view name")
    entities: List[EntityModel] = Field(..., description="Associated entities")
    features: List[FeatureDefinitionModel] = Field(..., description="Feature definitions")
    source: Optional[FeatureSourceModel] = Field(None, description="Feature source")
    ttl_seconds: Optional[int] = Field(None, description="Time-to-live in seconds")
    description: Optional[str] = Field(None, description="Feature view description")
    tags: List[str] = Field(default_factory=list, description="Tags")
    owner: Optional[str] = Field(None, description="Owner")


class FeatureViewListResponse(BaseModel):
    """List of feature views."""
    feature_views: List[FeatureViewResponse] = Field(..., description="Feature views")
    total: int = Field(..., description="Total count")


# Online Features Models

class OnlineFeaturesRequest(BaseModel):
    """Request for online features."""
    feature_refs: List[str] = Field(
        ...,
        description="Feature references (feature_view:feature_name)"
    )
    entity_ids: Dict[str, List[Any]] = Field(
        ...,
        description="Entity column -> list of values"
    )


class OnlineFeaturesResponse(BaseModel):
    """Response with online features."""
    features: Dict[str, List[Any]] = Field(
        ...,
        description="Feature ref -> list of values"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Response metadata"
    )


# Historical Features Models

class HistoricalFeaturesRequest(BaseModel):
    """Request for historical features."""
    entity_df: Dict[str, List[Any]] = Field(
        ...,
        description="Entity DataFrame with timestamps"
    )
    feature_refs: List[str] = Field(
        ...,
        description="Feature references"
    )
    timestamp_column: str = Field(
        default="_timestamp",
        description="Timestamp column name"
    )


class HistoricalFeaturesResponse(BaseModel):
    """Response with historical features."""
    entity_ids: Dict[str, List[Any]] = Field(
        ...,
        description="Entity ID columns"
    )
    features: Dict[str, List[Any]] = Field(
        ...,
        description="Feature columns"
    )
    timestamps: Optional[List[datetime]] = Field(
        None,
        description="Timestamps"
    )


# Materialization Models

class MaterializeRequest(BaseModel):
    """Request to materialize features."""
    start_time: Optional[datetime] = Field(None, description="Start of time range")
    end_time: Optional[datetime] = Field(None, description="End of time range")


class MaterializeResponse(BaseModel):
    """Response from materialization."""
    feature_view: str = Field(..., description="Feature view name")
    rows_materialized: int = Field(..., description="Number of rows materialized")
    duration_ms: float = Field(..., description="Duration in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Feature Statistics Models

class FeatureStatisticsModel(BaseModel):
    """Statistics for a single feature."""
    count: int = Field(..., description="Total count")
    null_count: int = Field(0, description="Null value count")
    mean: Optional[float] = Field(None, description="Mean value (numeric)")
    std: Optional[float] = Field(None, description="Standard deviation (numeric)")
    min: Optional[float] = Field(None, description="Minimum value (numeric)")
    max: Optional[float] = Field(None, description="Maximum value (numeric)")
    unique_count: Optional[int] = Field(None, description="Unique value count")


class FeatureStatisticsResponse(BaseModel):
    """Response with feature statistics."""
    feature_view: str = Field(..., description="Feature view name")
    statistics: Dict[str, FeatureStatisticsModel] = Field(
        ...,
        description="Feature name -> statistics"
    )
    computed_at: datetime = Field(default_factory=datetime.utcnow)


# Search Models

class SearchRequest(BaseModel):
    """Search request."""
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, ge=1, le=100, description="Max results")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional filters")


class SearchResultModel(BaseModel):
    """Single search result."""
    feature_view: str = Field(..., description="Feature view name")
    feature_name: str = Field(..., description="Feature name")
    dtype: str = Field(..., description="Data type")
    description: Optional[str] = Field(None, description="Description")
    score: float = Field(..., description="Relevance score")
    tags: List[str] = Field(default_factory=list, description="Tags")


class SearchResponse(BaseModel):
    """Search response."""
    results: List[SearchResultModel] = Field(..., description="Search results")
    total: int = Field(..., description="Total matching results")
    query: str = Field(..., description="Original query")


# Write Features Models

class WriteOnlineFeaturesRequest(BaseModel):
    """Request to write features to online store."""
    feature_view: str = Field(..., description="Feature view name")
    entity_id: Dict[str, Any] = Field(..., description="Entity identifier")
    features: Dict[str, Any] = Field(..., description="Feature values")
    timestamp: Optional[datetime] = Field(None, description="Feature timestamp")


class WriteOfflineFeaturesRequest(BaseModel):
    """Request to write features to offline store."""
    feature_view: str = Field(..., description="Feature view name")
    data: Dict[str, List[Any]] = Field(..., description="Column data")
    timestamp_column: str = Field(default="_timestamp", description="Timestamp column")
    mode: str = Field(default="append", description="Write mode (append/overwrite)")


class WriteResponse(BaseModel):
    """Response from write operation."""
    success: bool = Field(..., description="Whether write succeeded")
    rows_written: Optional[int] = Field(None, description="Number of rows written")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Feature Vector Model

class FeatureValueModel(BaseModel):
    """Single feature value."""
    name: str = Field(..., description="Feature name")
    value: Any = Field(..., description="Feature value")
    timestamp: Optional[datetime] = Field(None, description="Value timestamp")


class FeatureVectorResponse(BaseModel):
    """Feature vector response."""
    entity_id: Dict[str, Any] = Field(..., description="Entity identifier")
    features: List[FeatureValueModel] = Field(..., description="Feature values")
    timestamp: Optional[datetime] = Field(None, description="Vector timestamp")


# Validation Models

class ValidationRequest(BaseModel):
    """Request to validate a feature view."""
    feature_view: str = Field(..., description="Feature view to validate")


class ValidationResponse(BaseModel):
    """Validation response."""
    feature_view: str = Field(..., description="Feature view name")
    is_valid: bool = Field(..., description="Whether validation passed")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
