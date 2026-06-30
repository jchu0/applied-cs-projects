"""
FastAPI Production API Server for Feature Platform.

Provides REST endpoints for:
- Feature view management (CRUD)
- Online feature serving
- Historical feature retrieval
- Feature materialization
- Feature statistics
- Feature search
"""

import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Path, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from feature_platform import __version__
from feature_platform.core.models import Entity, Feature, FeatureView, DataType
from feature_platform.store.feature_store import FeatureStore
from feature_platform.api.models import (
    HealthResponse,
    HealthStatus,
    ErrorResponse,
    FeatureViewResponse,
    FeatureViewCreateRequest,
    FeatureViewListResponse,
    FeatureDefinitionModel,
    EntityModel,
    FeatureSourceModel,
    OnlineFeaturesRequest,
    OnlineFeaturesResponse,
    HistoricalFeaturesRequest,
    HistoricalFeaturesResponse,
    MaterializeRequest,
    MaterializeResponse,
    FeatureStatisticsResponse,
    FeatureStatisticsModel,
    SearchRequest,
    SearchResponse,
    SearchResultModel,
    WriteOnlineFeaturesRequest,
    WriteOfflineFeaturesRequest,
    WriteResponse,
    FeatureVectorResponse,
    FeatureValueModel,
    ValidationResponse,
)

# Global feature store instance
_feature_store: Optional[FeatureStore] = None


def get_feature_store() -> FeatureStore:
    """Get the global feature store instance."""
    global _feature_store
    if _feature_store is None:
        _feature_store = FeatureStore()
    return _feature_store


def set_feature_store(store: FeatureStore) -> None:
    """Set the global feature store instance."""
    global _feature_store
    _feature_store = store


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    get_feature_store()
    yield
    # Shutdown
    pass


def create_app(
    title: str = "Feature Platform API",
    description: str = "Production REST API for ML Feature Engineering Platform",
    cors_origins: Optional[List[str]] = None,
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        title: API title
        description: API description
        cors_origins: Allowed CORS origins (default: ["*"])

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title=title,
        description=description,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    _register_routes(app)

    return app


def _register_routes(app: FastAPI) -> None:
    """Register all API routes."""

    # Health check endpoint
    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["Health"],
        summary="Health check",
        description="Check the health status of the Feature Platform API",
    )
    async def health_check() -> HealthResponse:
        """Check API health status."""
        components = {
            "api": "healthy",
            "feature_store": "healthy",
        }

        try:
            store = get_feature_store()
            # Basic connectivity check
            store.list_feature_views()
            components["registry"] = "healthy"
        except Exception as e:
            components["registry"] = f"unhealthy: {str(e)}"

        all_healthy = all(v == "healthy" for v in components.values())

        return HealthResponse(
            status=HealthStatus.HEALTHY if all_healthy else HealthStatus.DEGRADED,
            version=__version__,
            timestamp=datetime.utcnow(),
            components=components,
        )

    # Feature View endpoints
    @app.get(
        "/feature-views",
        response_model=FeatureViewListResponse,
        tags=["Feature Views"],
        summary="List feature views",
        description="List all registered feature views",
    )
    async def list_feature_views(
        store: FeatureStore = Depends(get_feature_store),
    ) -> FeatureViewListResponse:
        """List all feature views."""
        views = store.list_feature_views()
        return FeatureViewListResponse(
            feature_views=[_feature_view_to_response(v) for v in views],
            total=len(views),
        )

    @app.get(
        "/feature-views/{name}",
        response_model=FeatureViewResponse,
        tags=["Feature Views"],
        summary="Get feature view",
        description="Get a feature view by name",
        responses={404: {"model": ErrorResponse}},
    )
    async def get_feature_view(
        name: str = Path(..., description="Feature view name"),
        store: FeatureStore = Depends(get_feature_store),
    ) -> FeatureViewResponse:
        """Get a feature view by name."""
        view = store.get_feature_view(name)
        if not view:
            raise HTTPException(
                status_code=404,
                detail=f"Feature view not found: {name}",
            )
        return _feature_view_to_response(view)

    @app.post(
        "/feature-views",
        response_model=FeatureViewResponse,
        tags=["Feature Views"],
        summary="Create feature view",
        description="Create a new feature view",
        status_code=201,
        responses={400: {"model": ErrorResponse}},
    )
    async def create_feature_view(
        request: FeatureViewCreateRequest,
        store: FeatureStore = Depends(get_feature_store),
    ) -> FeatureViewResponse:
        """Create a new feature view."""
        try:
            # Create entities
            entities = [
                Entity(
                    name=e.name,
                    join_keys=e.join_keys,
                    description=e.description,
                )
                for e in request.entities
            ]

            # Create features
            features = [
                Feature(
                    name=f.name,
                    dtype=DataType(f.dtype) if f.dtype in [dt.value for dt in DataType] else DataType.FLOAT64,
                    description=f.description,
                    tags=f.tags,
                )
                for f in request.features
            ]

            # Create feature view
            ttl = timedelta(seconds=request.ttl_seconds) if request.ttl_seconds else None
            view = FeatureView(
                name=request.name,
                entities=entities,
                schema=features,
                ttl=ttl,
                description=request.description,
                tags=request.tags,
                owner=request.owner,
            )

            # Register with store
            store.apply([*entities, view])

            return _feature_view_to_response(view)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.delete(
        "/feature-views/{name}",
        tags=["Feature Views"],
        summary="Delete feature view",
        description="Delete a feature view and its data",
        responses={404: {"model": ErrorResponse}},
    )
    async def delete_feature_view(
        name: str = Path(..., description="Feature view name"),
        store: FeatureStore = Depends(get_feature_store),
    ) -> Dict[str, Any]:
        """Delete a feature view."""
        view = store.get_feature_view(name)
        if not view:
            raise HTTPException(
                status_code=404,
                detail=f"Feature view not found: {name}",
            )

        success = store.delete_feature_view(name)
        return {"success": success, "deleted": name}

    # Online Features endpoints
    @app.post(
        "/features/online",
        response_model=OnlineFeaturesResponse,
        tags=["Features"],
        summary="Get online features",
        description="Get features for online serving with low latency",
    )
    async def get_online_features(
        request: OnlineFeaturesRequest,
        store: FeatureStore = Depends(get_feature_store),
    ) -> OnlineFeaturesResponse:
        """Get online features for serving."""
        start_time = time.time()

        features = store.get_online_features(
            feature_refs=request.feature_refs,
            entity_ids=request.entity_ids,
        )

        duration_ms = (time.time() - start_time) * 1000

        return OnlineFeaturesResponse(
            features=features,
            metadata={
                "duration_ms": duration_ms,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    # Historical Features endpoint
    @app.post(
        "/features/historical",
        response_model=HistoricalFeaturesResponse,
        tags=["Features"],
        summary="Get historical features",
        description="Get historical features with point-in-time correctness",
    )
    async def get_historical_features(
        request: HistoricalFeaturesRequest,
        store: FeatureStore = Depends(get_feature_store),
    ) -> HistoricalFeaturesResponse:
        """Get historical features for training."""
        data = store.get_historical_features(
            entity_df=request.entity_df,
            feature_refs=request.feature_refs,
            timestamp_column=request.timestamp_column,
        )

        # Convert to response format
        timestamps = None
        if data.timestamps is not None:
            timestamps = [
                datetime.fromtimestamp(ts) if isinstance(ts, (int, float)) else ts
                for ts in data.timestamps
            ]

        return HistoricalFeaturesResponse(
            entity_ids=data.entity_ids,
            features={k: list(v) for k, v in data.features.items()},
            timestamps=timestamps,
        )

    # Materialize endpoint
    @app.post(
        "/features/materialize/{feature_view}",
        response_model=MaterializeResponse,
        tags=["Features"],
        summary="Materialize features",
        description="Materialize features from offline to online store",
        responses={404: {"model": ErrorResponse}},
    )
    async def materialize_features(
        feature_view: str = Path(..., description="Feature view name"),
        request: Optional[MaterializeRequest] = None,
        store: FeatureStore = Depends(get_feature_store),
    ) -> MaterializeResponse:
        """Materialize features to online store."""
        view = store.get_feature_view(feature_view)
        if not view:
            raise HTTPException(
                status_code=404,
                detail=f"Feature view not found: {feature_view}",
            )

        start_time = time.time()

        rows = store.materialize(
            feature_view=feature_view,
            start_time=request.start_time if request else None,
            end_time=request.end_time if request else None,
        )

        duration_ms = (time.time() - start_time) * 1000

        return MaterializeResponse(
            feature_view=feature_view,
            rows_materialized=rows,
            duration_ms=duration_ms,
        )

    # Feature Statistics endpoint
    @app.get(
        "/features/{feature_view}/statistics",
        response_model=FeatureStatisticsResponse,
        tags=["Features"],
        summary="Get feature statistics",
        description="Get statistics for features in a feature view",
        responses={404: {"model": ErrorResponse}},
    )
    async def get_feature_statistics(
        feature_view: str = Path(..., description="Feature view name"),
        feature_names: Optional[str] = Query(
            None,
            description="Comma-separated list of feature names"
        ),
        store: FeatureStore = Depends(get_feature_store),
    ) -> FeatureStatisticsResponse:
        """Get feature statistics."""
        view = store.get_feature_view(feature_view)
        if not view:
            raise HTTPException(
                status_code=404,
                detail=f"Feature view not found: {feature_view}",
            )

        names = feature_names.split(",") if feature_names else None
        stats = store.get_feature_statistics(feature_view, names)

        return FeatureStatisticsResponse(
            feature_view=feature_view,
            statistics={
                name: FeatureStatisticsModel(**s)
                for name, s in stats.items()
            },
        )

    # Search endpoint
    @app.post(
        "/search",
        response_model=SearchResponse,
        tags=["Discovery"],
        summary="Search features",
        description="Search for features by name or description",
    )
    async def search_features(
        request: SearchRequest,
        store: FeatureStore = Depends(get_feature_store),
    ) -> SearchResponse:
        """Search for features."""
        results = store.search_features(request.query, request.limit)

        return SearchResponse(
            results=[
                SearchResultModel(
                    feature_view=r.get("feature_view", ""),
                    feature_name=r.get("feature_name", r.get("name", "")),
                    dtype=r.get("dtype", "unknown"),
                    description=r.get("description"),
                    score=r.get("score", 1.0),
                    tags=r.get("tags", []),
                )
                for r in results
            ],
            total=len(results),
            query=request.query,
        )

    # Write Features endpoints
    @app.post(
        "/features/online/write",
        response_model=WriteResponse,
        tags=["Features"],
        summary="Write online features",
        description="Write features to the online store",
        responses={404: {"model": ErrorResponse}},
    )
    async def write_online_features(
        request: WriteOnlineFeaturesRequest,
        store: FeatureStore = Depends(get_feature_store),
    ) -> WriteResponse:
        """Write features to online store."""
        view = store.get_feature_view(request.feature_view)
        if not view:
            raise HTTPException(
                status_code=404,
                detail=f"Feature view not found: {request.feature_view}",
            )

        store.write_to_online_store(
            feature_view=request.feature_view,
            entity_id=request.entity_id,
            features=request.features,
            timestamp=request.timestamp,
        )

        return WriteResponse(success=True, rows_written=1)

    @app.post(
        "/features/offline/write",
        response_model=WriteResponse,
        tags=["Features"],
        summary="Write offline features",
        description="Write features to the offline store",
        responses={404: {"model": ErrorResponse}},
    )
    async def write_offline_features(
        request: WriteOfflineFeaturesRequest,
        store: FeatureStore = Depends(get_feature_store),
    ) -> WriteResponse:
        """Write features to offline store."""
        view = store.get_feature_view(request.feature_view)
        if not view:
            raise HTTPException(
                status_code=404,
                detail=f"Feature view not found: {request.feature_view}",
            )

        store.write_to_offline_store(
            feature_view=request.feature_view,
            data=request.data,
            timestamp_column=request.timestamp_column,
            mode=request.mode,
        )

        # Calculate rows written
        rows = 0
        if request.data:
            first_col = next(iter(request.data.values()))
            rows = len(first_col) if isinstance(first_col, list) else 1

        return WriteResponse(success=True, rows_written=rows)

    # Feature Vector endpoint
    @app.get(
        "/features/{feature_view}/vector",
        response_model=FeatureVectorResponse,
        tags=["Features"],
        summary="Get feature vector",
        description="Get a feature vector for a single entity",
        responses={404: {"model": ErrorResponse}},
    )
    async def get_feature_vector(
        feature_view: str = Path(..., description="Feature view name"),
        entity_id: str = Query(..., description="Entity ID as JSON object"),
        feature_names: Optional[str] = Query(
            None,
            description="Comma-separated list of feature names"
        ),
        store: FeatureStore = Depends(get_feature_store),
    ) -> FeatureVectorResponse:
        """Get feature vector for a single entity."""
        import json

        view = store.get_feature_view(feature_view)
        if not view:
            raise HTTPException(
                status_code=404,
                detail=f"Feature view not found: {feature_view}",
            )

        try:
            entity = json.loads(entity_id)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400,
                detail="Invalid entity_id JSON format",
            )

        names = feature_names.split(",") if feature_names else None
        vector = store.get_feature_vector(feature_view, entity, names)

        return FeatureVectorResponse(
            entity_id=vector.entity_id,
            features=[
                FeatureValueModel(
                    name=fv.name,
                    value=fv.value,
                    timestamp=fv.timestamp,
                )
                for fv in vector.features
            ],
            timestamp=vector.timestamp,
        )

    # Validation endpoint
    @app.post(
        "/feature-views/{name}/validate",
        response_model=ValidationResponse,
        tags=["Feature Views"],
        summary="Validate feature view",
        description="Validate a feature view configuration",
        responses={404: {"model": ErrorResponse}},
    )
    async def validate_feature_view(
        name: str = Path(..., description="Feature view name"),
        store: FeatureStore = Depends(get_feature_store),
    ) -> ValidationResponse:
        """Validate a feature view configuration."""
        errors = store.validate_feature_view(name)

        return ValidationResponse(
            feature_view=name,
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=[],
        )


def _feature_view_to_response(view: FeatureView) -> FeatureViewResponse:
    """Convert FeatureView to response model."""
    return FeatureViewResponse(
        name=view.name,
        entities=[
            EntityModel(
                name=e.name,
                join_keys=e.join_keys,
                description=e.description,
            )
            for e in view.entities
        ],
        features=[
            FeatureDefinitionModel(
                name=f.name,
                dtype=f.dtype.value if hasattr(f.dtype, 'value') else str(f.dtype),
                description=f.description,
                tags=f.tags or [],
            )
            for f in view.schema
        ],
        source=FeatureSourceModel(
            type=view.source.type if view.source else None,
            path=getattr(view.source, 'path', None) if view.source else None,
            table=getattr(view.source, 'table', None) if view.source else None,
            query=getattr(view.source, 'query', None) if view.source else None,
        ) if view.source else None,
        ttl_seconds=int(view.ttl.total_seconds()) if view.ttl else None,
        description=view.description,
        tags=view.tags or [],
        owner=view.owner,
        created_at=getattr(view, 'created_at', None),
        updated_at=getattr(view, 'updated_at', None),
    )


# Create default app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
