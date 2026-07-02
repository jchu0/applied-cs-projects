"""REST API for Data Observability Platform.

Provides endpoints for:
- Table monitoring and metadata
- Anomaly detection and management
- Health scoring
- Data lineage
- PII detection
"""

from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from .models import (
    TableMetadata, ColumnMetadata, Anomaly, AnomalyType,
    DataHealthScore, LineageInfo, Alert, ImpactAnalysis
)
from .detector import AnomalyDetector
from .config import DetectorConfig, CollectorConfig
from .health import HealthScorer, HealthMonitor
from .lineage import LineageGraph
from .pii import PIIDetector, PIIRegistry
from .alerting import AlertingEngine
from .collectors import InMemoryCollector
from .security import install_hardening

# Create FastAPI app
app = FastAPI(
    title="Data Observability API",
    description="Monitor data quality, detect anomalies, and track lineage",
    version="1.0.0",
)

# Wire opt-in production hardening (auth / rate limiting / request timeout).
# All three are disabled by default so tests and the quick-start keep working.
install_hardening(app)

# In-memory storage for demo/testing
_tables: dict[str, TableMetadata] = {}
_anomalies: dict[str, Anomaly] = {}
_alerts: dict[str, Alert] = {}
_lineage_graph = LineageGraph()
_scorer = HealthScorer()
_health_monitor = HealthMonitor(_scorer)
_pii_registry = PIIRegistry()
_detector = AnomalyDetector(DetectorConfig())
_collector = InMemoryCollector(CollectorConfig())


# Request/Response Models
class TableListResponse(BaseModel):
    """Response for listing tables."""
    tables: list[dict]
    total: int


class TableResponse(BaseModel):
    """Response for single table details."""
    table_id: str
    name: str
    database: str
    schema_name: str
    columns: list[dict]
    row_count: Optional[int] = None
    last_updated: Optional[datetime] = None


class AnomalyListResponse(BaseModel):
    """Response for listing anomalies."""
    anomalies: list[dict]
    total: int


class HealthResponse(BaseModel):
    """Response for health score."""
    table_id: str
    overall_score: float
    freshness_score: float
    volume_score: float
    schema_score: float
    quality_score: float
    trend: str


class LineageResponse(BaseModel):
    """Response for lineage info."""
    table_id: str
    upstream: list[str]
    downstream: list[str]
    column_lineage: dict


class AcknowledgeRequest(BaseModel):
    """Request to acknowledge an anomaly."""
    acknowledged_by: str
    notes: Optional[str] = None


class RegisterTableRequest(BaseModel):
    """Request to register a new table for monitoring."""
    name: str
    database: str
    schema_name: str
    columns: list[dict]


class PIIScanResponse(BaseModel):
    """Response for PII scan results."""
    table_id: str
    pii_columns: list[dict]
    recommendations: list[str]


# API Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/tables", response_model=TableListResponse)
async def list_tables(
    database: Optional[str] = Query(None, description="Filter by database"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """List all monitored tables."""
    tables = list(_tables.values())

    if database:
        tables = [t for t in tables if t.database == database]

    total = len(tables)
    tables = tables[offset:offset + limit]

    return TableListResponse(
        tables=[{
            "table_id": t.table_id,
            "name": t.table_name,
            "database": t.database,
            "schema": t.schema,
            "row_count": t.row_count,
        } for t in tables],
        total=total,
    )


@app.post("/tables")
async def register_table(request: RegisterTableRequest):
    """Register a new table for monitoring."""
    table_id = f"{request.database}.{request.schema_name}.{request.name}"

    columns = [
        ColumnMetadata(
            name=col["name"],
            data_type=col.get("data_type", "string"),
            nullable=col.get("nullable", True),
        )
        for col in request.columns
    ]

    metadata = TableMetadata(
        table_id=table_id,
        table_name=request.name,
        database=request.database,
        schema=request.schema_name,
        columns=columns,
        row_count=0,
        size_bytes=0,
        last_modified=datetime.utcnow(),
    )

    _tables[table_id] = metadata
    _lineage_graph.add_table(table_id)

    return {"table_id": table_id, "status": "registered"}


@app.get("/tables/{table_id}")
async def get_table(table_id: str):
    """Get details for a specific table."""
    if table_id not in _tables:
        raise HTTPException(status_code=404, detail=f"Table {table_id} not found")

    table = _tables[table_id]
    return {
        "table_id": table.table_id,
        "name": table.table_name,
        "database": table.database,
        "schema": table.schema,
        "columns": [
            {
                "name": c.name,
                "data_type": c.data_type,
                "nullable": c.nullable,
            }
            for c in table.columns
        ],
        "row_count": table.row_count,
        "last_modified": table.last_modified.isoformat() if table.last_modified else None,
    }


@app.get("/tables/{table_id}/lineage", response_model=LineageResponse)
async def get_table_lineage(table_id: str):
    """Get lineage information for a table."""
    if table_id not in _tables:
        raise HTTPException(status_code=404, detail=f"Table {table_id} not found")

    upstream = _lineage_graph.get_upstream(table_id)
    downstream = _lineage_graph.get_downstream(table_id)

    return LineageResponse(
        table_id=table_id,
        upstream=upstream,
        downstream=downstream,
        column_lineage={},
    )


@app.post("/tables/{table_id}/lineage")
async def add_lineage(table_id: str, upstream_table: str):
    """Add lineage relationship (upstream_table -> table_id)."""
    if table_id not in _tables:
        raise HTTPException(status_code=404, detail=f"Table {table_id} not found")

    _lineage_graph.add_lineage(upstream_table, table_id)
    return {"status": "lineage added", "source": upstream_table, "target": table_id}


@app.get("/tables/{table_id}/health", response_model=HealthResponse)
async def get_table_health(table_id: str):
    """Get health score for a table."""
    if table_id not in _tables:
        raise HTTPException(status_code=404, detail=f"Table {table_id} not found")

    table = _tables[table_id]

    # Get recent anomalies for this table
    table_anomalies = [a for a in _anomalies.values() if a.table_id == table_id]

    # Build column stats from table metadata
    stats = {}
    for col in table.columns:
        if col.stats:
            stats[col.name] = col.stats

    # Calculate health score
    score = _scorer.calculate_health_score(
        table_id=table_id,
        metadata=table,
        stats=stats,
        recent_anomalies=table_anomalies,
    )

    # Get trend from history
    history = _health_monitor.get_health_trend(table_id)
    trend_info = _scorer.calculate_trend(history) if history else {"direction": "stable"}

    return HealthResponse(
        table_id=table_id,
        overall_score=score.overall_score,
        freshness_score=score.freshness_score,
        volume_score=score.volume_score,
        schema_score=score.schema_score,
        quality_score=score.quality_score,
        trend=trend_info.get("direction", "stable"),
    )


@app.get("/tables/{table_id}/anomalies")
async def get_table_anomalies(
    table_id: str,
    severity: Optional[str] = Query(None, description="Filter by severity"),
    anomaly_type: Optional[str] = Query(None, description="Filter by type"),
    limit: int = Query(100, ge=1, le=1000),
):
    """Get anomalies for a specific table."""
    if table_id not in _tables:
        raise HTTPException(status_code=404, detail=f"Table {table_id} not found")

    anomalies = [a for a in _anomalies.values() if a.table_id == table_id]

    if severity:
        anomalies = [a for a in anomalies if a.severity == severity]

    if anomaly_type:
        try:
            atype = AnomalyType[anomaly_type.upper()]
            anomalies = [a for a in anomalies if a.anomaly_type == atype]
        except KeyError:
            pass

    anomalies = anomalies[:limit]

    # Check if there's an alert for this anomaly to get acknowledged status
    def get_alert_for_anomaly(anomaly_id: str) -> Optional[Alert]:
        for alert in _alerts.values():
            if alert.anomaly.anomaly_id == anomaly_id:
                return alert
        return None

    return {
        "table_id": table_id,
        "anomalies": [
            {
                "anomaly_id": a.anomaly_id,
                "type": a.anomaly_type.name,
                "severity": a.severity,
                "description": a.description,
                "detected_at": a.detected_at.isoformat(),
                "acknowledged": get_alert_for_anomaly(a.anomaly_id) is not None
                    and get_alert_for_anomaly(a.anomaly_id).status == "acknowledged",
            }
            for a in anomalies
        ],
        "total": len(anomalies),
    }


@app.post("/tables/{table_id}/detect")
async def run_anomaly_detection(table_id: str):
    """Run anomaly detection for a table."""
    if table_id not in _tables:
        raise HTTPException(status_code=404, detail=f"Table {table_id} not found")

    table = _tables[table_id]

    # Build column stats from table metadata
    stats = {}
    for col in table.columns:
        if col.stats:
            stats[col.name] = col.stats

    # Run async anomaly detection
    anomalies = await _detector.detect_all_anomalies(table_id, table, stats)

    for anomaly in anomalies:
        _anomalies[anomaly.anomaly_id] = anomaly

    return {
        "table_id": table_id,
        "anomalies_detected": len(anomalies),
        "anomaly_ids": [a.anomaly_id for a in anomalies],
    }


@app.get("/anomalies", response_model=AnomalyListResponse)
async def list_anomalies(
    severity: Optional[str] = Query(None, description="Filter by severity"),
    anomaly_type: Optional[str] = Query(None, description="Filter by type"),
    acknowledged: Optional[bool] = Query(None, description="Filter by acknowledged status"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """List all anomalies across all tables."""
    anomalies = list(_anomalies.values())

    if severity:
        anomalies = [a for a in anomalies if a.severity == severity]

    if anomaly_type:
        try:
            atype = AnomalyType[anomaly_type.upper()]
            anomalies = [a for a in anomalies if a.anomaly_type == atype]
        except KeyError:
            pass

    # Helper to check if anomaly is acknowledged via its alert
    def is_acknowledged(anomaly_id: str) -> bool:
        for alert in _alerts.values():
            if alert.anomaly.anomaly_id == anomaly_id:
                return alert.status == "acknowledged"
        return False

    if acknowledged is not None:
        anomalies = [a for a in anomalies if is_acknowledged(a.anomaly_id) == acknowledged]

    total = len(anomalies)
    anomalies = anomalies[offset:offset + limit]

    return AnomalyListResponse(
        anomalies=[
            {
                "anomaly_id": a.anomaly_id,
                "table_id": a.table_id,
                "type": a.anomaly_type.name,
                "severity": a.severity,
                "description": a.description,
                "detected_at": a.detected_at.isoformat(),
                "acknowledged": is_acknowledged(a.anomaly_id),
            }
            for a in anomalies
        ],
        total=total,
    )


@app.get("/anomalies/{anomaly_id}")
async def get_anomaly(anomaly_id: str):
    """Get details for a specific anomaly."""
    if anomaly_id not in _anomalies:
        raise HTTPException(status_code=404, detail=f"Anomaly {anomaly_id} not found")

    a = _anomalies[anomaly_id]

    # Find associated alert if any
    alert = None
    for al in _alerts.values():
        if al.anomaly.anomaly_id == anomaly_id:
            alert = al
            break

    return {
        "anomaly_id": a.anomaly_id,
        "table_id": a.table_id,
        "column_name": a.column_name,
        "type": a.anomaly_type.name,
        "severity": a.severity,
        "description": a.description,
        "detected_at": a.detected_at.isoformat(),
        "metric_value": a.metric_value,
        "expected_range": a.expected_range,
        "acknowledged": alert.status == "acknowledged" if alert else False,
        "acknowledged_by": alert.acknowledged_by if alert else None,
        "acknowledged_at": alert.acknowledged_at.isoformat() if alert and alert.acknowledged_at else None,
        "context": a.context,
    }


@app.post("/anomalies/{anomaly_id}/acknowledge")
async def acknowledge_anomaly(anomaly_id: str, request: AcknowledgeRequest):
    """Acknowledge an anomaly."""
    if anomaly_id not in _anomalies:
        raise HTTPException(status_code=404, detail=f"Anomaly {anomaly_id} not found")

    anomaly = _anomalies[anomaly_id]

    # Find or create an alert for this anomaly
    alert = None
    for al in _alerts.values():
        if al.anomaly.anomaly_id == anomaly_id:
            alert = al
            break

    if alert is None:
        # Create an alert for this anomaly
        import uuid
        alert_id = str(uuid.uuid4())
        alert = Alert(
            alert_id=alert_id,
            anomaly=anomaly,
            created_at=datetime.utcnow(),
            status="active",
        )
        _alerts[alert_id] = alert

    # Acknowledge the alert
    alert.status = "acknowledged"
    alert.acknowledged_by = request.acknowledged_by
    alert.acknowledged_at = datetime.utcnow()
    if request.notes:
        alert.resolution_notes = request.notes

    return {
        "anomaly_id": anomaly_id,
        "alert_id": alert.alert_id,
        "status": "acknowledged",
        "acknowledged_by": request.acknowledged_by,
    }


@app.get("/tables/{table_id}/pii", response_model=PIIScanResponse)
async def scan_table_pii(table_id: str):
    """Scan a table for PII columns."""
    if table_id not in _tables:
        raise HTTPException(status_code=404, detail=f"Table {table_id} not found")

    table = _tables[table_id]
    detector = PIIDetector()

    # Scan the entire table schema
    detections = detector.scan_schema_only(table)

    pii_columns = [
        {
            "column": d.column_name,
            "pii_types": d.pii_types,
            "confidence": d.confidence,
        }
        for d in detections
    ]

    # Generate recommendations for each detection
    recommendations = []
    for d in detections:
        for pii_type in d.pii_types:
            scan_result = {"pii_detected": True, "pii_type": pii_type}
            recs = detector.get_recommendations(scan_result)
            recommendations.extend(recs)
    # Dedupe recommendations
    recommendations = list(set(recommendations))

    return PIIScanResponse(
        table_id=table_id,
        pii_columns=pii_columns,
        recommendations=recommendations,
    )


@app.get("/impact/{table_id}")
async def get_impact_analysis(table_id: str):
    """Get downstream impact analysis for a table."""
    if table_id not in _tables:
        raise HTTPException(status_code=404, detail=f"Table {table_id} not found")

    impact = _lineage_graph.get_impact_analysis(table_id)

    return {
        "table_id": table_id,
        "affected_tables": [t.table_id for t in impact.affected_tables] if impact else [],
        "affected_pipelines": impact.affected_pipelines if impact else [],
        "affected_dashboards": impact.affected_dashboards if impact else [],
        "total_downstream": impact.total_downstream if impact else 0,
    }


# Main entry point for running with uvicorn
def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
