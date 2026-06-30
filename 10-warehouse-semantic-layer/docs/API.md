# Semantic Layer API Documentation

## Overview

The Semantic Layer provides a RESTful API for defining, managing, and querying business metrics across data warehouses. This document covers all available endpoints, request/response formats, and usage examples.

## Base URL

```
https://api.semantic-layer.example.com/v1
```

## Authentication

All API requests require authentication using an API key:

```http
Authorization: Bearer YOUR_API_KEY
```

## API Endpoints

### Metrics Management

#### List Metrics

Returns a list of available metrics with optional filtering.

**Endpoint:** `GET /metrics`

**Query Parameters:**
- `category` (string, optional): Filter by metric category
- `search` (string, optional): Search metrics by name or description
- `tier` (integer, optional): Filter by metric tier (1, 2, or 3)
- `owner` (string, optional): Filter by metric owner

**Response:**
```json
{
  "metrics": [
    {
      "name": "total_revenue",
      "label": "Total Revenue",
      "description": "Sum of all revenue",
      "dimensions": ["customer_segment", "region", "product_category"],
      "time_grains": ["day", "week", "month", "quarter", "year"],
      "calculation_method": "sum",
      "owner": "finance-team",
      "tier": 1
    }
  ],
  "count": 1
}
```

**Example:**
```python
import requests

response = requests.get(
    "https://api.semantic-layer.example.com/v1/metrics",
    params={"category": "financial", "tier": 1},
    headers={"Authorization": "Bearer YOUR_API_KEY"}
)
metrics = response.json()
```

#### Get Metric Details

Returns detailed information about a specific metric.

**Endpoint:** `GET /metrics/{metric_name}`

**Response:**
```json
{
  "name": "total_revenue",
  "label": "Total Revenue",
  "description": "Sum of all revenue from completed orders",
  "model": "fact_orders",
  "calculation_method": "sum",
  "expression": "order_amount",
  "timestamp": "order_date",
  "time_grains": ["day", "week", "month", "quarter", "year"],
  "dimensions": ["customer_segment", "region", "product_category"],
  "filters": [
    {
      "field": "order_status",
      "operator": "=",
      "value": "completed"
    }
  ],
  "meta": {
    "owner": "finance-team",
    "tier": 1,
    "refresh_frequency": "daily",
    "data_quality_score": 0.98
  }
}
```

#### Create Metric

Creates a new metric definition.

**Endpoint:** `POST /metrics`

**Request Body:**
```json
{
  "name": "customer_acquisition_cost",
  "label": "Customer Acquisition Cost",
  "description": "Average cost to acquire a new customer",
  "model": "fact_marketing",
  "calculation_method": "derived",
  "expression": "{{ metric('marketing_spend') }} / {{ metric('new_customers') }}",
  "timestamp": "date",
  "time_grains": ["month", "quarter", "year"],
  "dimensions": ["channel", "campaign"],
  "meta": {
    "owner": "marketing-team",
    "tier": 2
  }
}
```

**Response:**
```json
{
  "status": "created",
  "metric": {
    "name": "customer_acquisition_cost",
    "created_at": "2024-01-15T10:30:00Z"
  }
}
```

#### Update Metric

Updates an existing metric definition.

**Endpoint:** `PUT /metrics/{metric_name}`

**Request Body:** Same as Create Metric

**Response:**
```json
{
  "status": "updated",
  "metric": {
    "name": "customer_acquisition_cost",
    "updated_at": "2024-01-15T11:00:00Z"
  }
}
```

#### Delete Metric

Deletes a metric definition.

**Endpoint:** `DELETE /metrics/{metric_name}`

**Response:**
```json
{
  "status": "deleted",
  "metric": "customer_acquisition_cost"
}
```

### Query Execution

#### Query Metrics

Executes a query for one or more metrics.

**Endpoint:** `POST /query`

**Request Body:**
```json
{
  "metrics": ["total_revenue", "order_count", "average_order_value"],
  "dimensions": ["customer_segment", "region"],
  "filters": [
    {
      "field": "region",
      "operator": "in",
      "value": ["NA", "EU"]
    },
    {
      "field": "customer_segment",
      "operator": "!=",
      "value": "unknown"
    }
  ],
  "time_grain": "month",
  "start_date": "2024-01-01",
  "end_date": "2024-12-31",
  "limit": 1000,
  "offset": 0,
  "show_sql": false
}
```

**Response:**
```json
{
  "data": [
    {
      "period": "2024-01-01",
      "customer_segment": "Enterprise",
      "region": "NA",
      "total_revenue": 1500000,
      "order_count": 450,
      "average_order_value": 3333.33
    },
    {
      "period": "2024-01-01",
      "customer_segment": "Enterprise",
      "region": "EU",
      "total_revenue": 1200000,
      "order_count": 380,
      "average_order_value": 3157.89
    }
  ],
  "metadata": {
    "metrics": ["total_revenue", "order_count", "average_order_value"],
    "dimensions": ["customer_segment", "region"],
    "time_grain": "month",
    "row_count": 2,
    "query_time_ms": 245
  },
  "sql": "SELECT ..."  // Only included if show_sql=true
}
```

**Example:**
```python
import requests
from datetime import datetime

query = {
    "metrics": ["total_revenue"],
    "dimensions": ["region"],
    "time_grain": "week",
    "start_date": "2024-01-01",
    "end_date": "2024-03-31",
    "filters": [
        {"field": "region", "operator": "in", "value": ["NA", "EU"]}
    ]
}

response = requests.post(
    "https://api.semantic-layer.example.com/v1/query",
    json=query,
    headers={"Authorization": "Bearer YOUR_API_KEY"}
)
result = response.json()
```

#### Validate Query

Validates a query without executing it.

**Endpoint:** `POST /query/validate`

**Request Body:** Same as Query Metrics

**Response:**
```json
{
  "valid": true,
  "errors": [],
  "warnings": [
    "Large date range may result in slow query"
  ]
}
```

#### Query Preview

Generates SQL without executing the query.

**Endpoint:** `POST /query/preview`

**Request Body:** Same as Query Metrics

**Response:**
```json
{
  "sql": "SELECT\n    DATE_TRUNC('month', order_date) as period,\n    customer_segment,\n    region,\n    SUM(order_amount) as total_revenue,\n    COUNT(order_id) as order_count\nFROM fact_orders\nWHERE order_date >= '2024-01-01'\n    AND order_date < '2024-12-31'\n    AND region IN ('NA', 'EU')\n    AND customer_segment != 'unknown'\nGROUP BY period, customer_segment, region\nORDER BY period",
  "estimated_cost": 0.05,
  "estimated_rows": 24
}
```

### Dimensions

#### List Dimensions

Returns available dimensions for metrics.

**Endpoint:** `GET /dimensions`

**Query Parameters:**
- `metric` (string, optional): Filter dimensions available for a specific metric

**Response:**
```json
{
  "dimensions": [
    {
      "name": "customer_segment",
      "label": "Customer Segment",
      "description": "Customer segmentation category",
      "data_type": "string",
      "values": ["Enterprise", "Mid-Market", "SMB"],
      "hierarchy": null
    },
    {
      "name": "region",
      "label": "Region",
      "description": "Geographic region",
      "data_type": "string",
      "hierarchy": ["region", "country", "state", "city"]
    }
  ]
}
```

#### Get Dimension Values

Returns possible values for a dimension.

**Endpoint:** `GET /dimensions/{dimension_name}/values`

**Query Parameters:**
- `search` (string, optional): Filter values by search term
- `limit` (integer, optional): Maximum number of values to return

**Response:**
```json
{
  "dimension": "customer_segment",
  "values": [
    {"value": "Enterprise", "label": "Enterprise", "count": 1250},
    {"value": "Mid-Market", "label": "Mid-Market", "count": 3500},
    {"value": "SMB", "label": "Small & Medium Business", "count": 8900}
  ]
}
```

### Import/Export

#### Export Metrics

Exports metric definitions in YAML format.

**Endpoint:** `GET /export/metrics`

**Query Parameters:**
- `format` (string): Export format (`yaml` or `json`)

**Response:**
```yaml
version: 2

metrics:
  - name: total_revenue
    label: "Total Revenue"
    description: "Sum of all revenue"
    model: ref('fact_orders')
    calculation_method: sum
    expression: order_amount
    timestamp: order_date
    time_grains: [day, week, month, quarter, year]
    dimensions:
      - customer_segment
      - region
    meta:
      owner: finance-team
      tier: 1
```

#### Import Metrics

Imports metric definitions from YAML/JSON.

**Endpoint:** `POST /import/metrics`

**Request Body:**
```json
{
  "format": "yaml",
  "content": "version: 2\n\nmetrics:\n  - name: new_metric\n    ..."
}
```

**Response:**
```json
{
  "imported": 5,
  "updated": 2,
  "errors": []
}
```

### Health & Status

#### Health Check

Returns API health status.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "version": "1.2.0",
  "warehouse": {
    "type": "snowflake",
    "connected": true,
    "latency_ms": 45
  },
  "cache": {
    "enabled": true,
    "hit_rate": 0.87
  }
}
```

#### Metric Statistics

Returns usage statistics for metrics.

**Endpoint:** `GET /stats/metrics`

**Response:**
```json
{
  "total_metrics": 45,
  "metrics_by_tier": {
    "1": 12,
    "2": 20,
    "3": 13
  },
  "most_used": [
    {"metric": "total_revenue", "query_count": 1523},
    {"metric": "active_users", "query_count": 987}
  ],
  "recently_updated": [
    {"metric": "conversion_rate", "updated_at": "2024-01-15T10:30:00Z"}
  ]
}
```

## Error Handling

The API uses standard HTTP status codes and returns detailed error messages:

### Error Response Format

```json
{
  "error": {
    "code": "METRIC_NOT_FOUND",
    "message": "Metric 'invalid_metric' not found",
    "details": {
      "metric": "invalid_metric",
      "available_metrics": ["total_revenue", "order_count"]
    }
  },
  "request_id": "req_123abc"
}
```

### Common Error Codes

| Status Code | Error Code | Description |
|------------|------------|-------------|
| 400 | INVALID_REQUEST | Invalid request parameters |
| 401 | UNAUTHORIZED | Invalid or missing API key |
| 403 | FORBIDDEN | Insufficient permissions |
| 404 | METRIC_NOT_FOUND | Requested metric doesn't exist |
| 409 | METRIC_EXISTS | Metric already exists |
| 422 | VALIDATION_ERROR | Query validation failed |
| 429 | RATE_LIMITED | Too many requests |
| 500 | INTERNAL_ERROR | Internal server error |
| 503 | WAREHOUSE_UNAVAILABLE | Data warehouse connection failed |

## Rate Limiting

API requests are rate-limited per API key:

- **Standard tier**: 100 requests per minute
- **Professional tier**: 1,000 requests per minute
- **Enterprise tier**: Unlimited

Rate limit information is included in response headers:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1705320000
```

## Pagination

List endpoints support pagination using `limit` and `offset` parameters:

```http
GET /metrics?limit=20&offset=40
```

Paginated responses include pagination metadata:

```json
{
  "data": [...],
  "pagination": {
    "total": 150,
    "limit": 20,
    "offset": 40,
    "has_more": true
  }
}
```

## Webhooks

Configure webhooks to receive notifications about metric changes:

```json
{
  "url": "https://your-app.com/webhook",
  "events": ["metric.created", "metric.updated", "metric.deleted"],
  "secret": "webhook_secret_key"
}
```

## SDK Examples

### Python SDK

```python
from semantic_layer import Client

client = Client(api_key="YOUR_API_KEY")

# List metrics
metrics = client.metrics.list(category="financial")

# Query metrics
result = client.query(
    metrics=["total_revenue", "order_count"],
    dimensions=["region"],
    start_date="2024-01-01",
    end_date="2024-12-31",
    time_grain="month"
)

# Create metric
client.metrics.create(
    name="new_metric",
    label="New Metric",
    calculation_method="sum",
    expression="amount"
)
```

### JavaScript SDK

```javascript
const SemanticLayer = require('@semantic-layer/client');

const client = new SemanticLayer({
  apiKey: 'YOUR_API_KEY'
});

// Query metrics
const result = await client.query({
  metrics: ['total_revenue'],
  dimensions: ['region'],
  timeGrain: 'month',
  startDate: '2024-01-01',
  endDate: '2024-12-31'
});

// Get metric details
const metric = await client.metrics.get('total_revenue');
```

## Best Practices

1. **Use appropriate time grains**: Choose the coarsest time grain that meets your needs for better performance
2. **Limit dimensions**: Query only necessary dimensions to reduce result size
3. **Cache results**: Implement client-side caching for frequently accessed metrics
4. **Batch requests**: Combine multiple metrics in a single query when possible
5. **Use filters effectively**: Apply filters to reduce data scanned
6. **Monitor usage**: Track API usage and optimize frequently-used queries