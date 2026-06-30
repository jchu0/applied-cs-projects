# Warehouse + Semantic Layer (dbt + Metrics) - Technical Blueprint

## Executive Summary

This project implements a modern analytics engineering stack combining a well-designed data warehouse with a semantic layer that provides consistent, governed business metrics. Built on dbt for transformations and a custom semantic layer for metric definitions, it demonstrates mastery of dimensional modeling, transformation reproducibility, dependency management, and metrics-as-code principles.

> **Concepts covered:** [§02 dbt transformations](../../../02-data-engineering/03-data-warehousing/dbt/dbt-transformations.md) · [§02 Dimensional modeling](../../../02-data-engineering/03-data-warehousing/dimensional-modeling/dimensional-modeling.md) · [§02 SQL optimization](../../../02-data-engineering/03-data-warehousing/sql-optimization/sql-optimization.md). Pairs with [Project 07 (data lakehouse — the substrate)](../../07-data-lakehouse/) and [Project 17 (query engine that executes the SQL)](../../17-columnar-query-engine/). Map: [`CONCEPT_TO_PROJECT_MAP.md`](../../CONCEPT_TO_PROJECT_MAP.md).

**Primary Goals:**
- Build a well-structured data warehouse using dimensional modeling best practices
- Implement a semantic layer with reusable, governed metric definitions
- Create a reproducible transformation pipeline with comprehensive testing
- Enable self-service analytics with consistent business definitions

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Raw Data Sources                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │   ERP    │  │   CRM    │  │  Events  │  │   APIs   │            │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘            │
└───────┴─────────────┴─────────────┴─────────────┴───────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    dbt Transformation Layer                          │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Staging Layer                              │   │
│  │  stg_erp__*, stg_crm__*, stg_events__*                       │   │
│  │  (source-conformed, renamed, typed)                           │   │
│  └──────────────────────────────┬───────────────────────────────┘   │
│                                  │                                   │
│  ┌──────────────────────────────▼───────────────────────────────┐   │
│  │                  Intermediate Layer                           │   │
│  │  int_*  (business logic, joins, calculations)                │   │
│  └──────────────────────────────┬───────────────────────────────┘   │
│                                  │                                   │
│  ┌──────────────────────────────▼───────────────────────────────┐   │
│  │                     Marts Layer                               │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │   │
│  │  │   core/     │  │  finance/   │  │  marketing/ │           │   │
│  │  │ (facts,dims)│  │  (rev,cost) │  │  (campaign) │           │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘           │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Semantic Layer                                    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                  Metric Definitions                          │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │    │
│  │  │   Revenue   │  │    Users    │  │   Orders    │          │    │
│  │  │   Metrics   │  │   Metrics   │  │   Metrics   │          │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘          │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │               Semantic Query Engine                          │    │
│  │  (dimension joins, filter translation, aggregation)          │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└────────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Consumption Layer                                 │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐                │
│  │   BI    │  │  APIs   │  │Notebooks│  │  Apps   │                │
│  │  Tools  │  │         │  │         │  │         │                │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘                │
└─────────────────────────────────────────────────────────────────────┘
```

### dbt Model Flow

```
Sources → Staging → Intermediate → Marts → Metrics

┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  source('stripe', 'payments')                                   │
│         │                                                        │
│         ▼                                                        │
│  stg_stripe__payments (rename, cast, basic cleaning)            │
│         │                                                        │
│         ▼                                                        │
│  int_payments_with_customers (join customer data)               │
│         │                                                        │
│         ▼                                                        │
│  fct_payments (grain: one row per payment)                      │
│         │                                                        │
│         ▼                                                        │
│  metric: total_revenue (SUM(amount))                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Internals

### dbt Project Structure

```
dbt_project/
├── dbt_project.yml
├── packages.yml
├── profiles.yml
├── models/
│   ├── staging/
│   │   ├── stripe/
│   │   │   ├── _stripe__sources.yml
│   │   │   ├── _stripe__models.yml
│   │   │   ├── stg_stripe__payments.sql
│   │   │   └── stg_stripe__customers.sql
│   │   └── salesforce/
│   │       ├── _salesforce__sources.yml
│   │       └── stg_salesforce__accounts.sql
│   ├── intermediate/
│   │   ├── finance/
│   │   │   ├── _int_finance__models.yml
│   │   │   └── int_payments_with_customers.sql
│   │   └── marketing/
│   │       └── int_campaign_performance.sql
│   └── marts/
│       ├── core/
│       │   ├── _core__models.yml
│       │   ├── dim_customers.sql
│       │   ├── dim_products.sql
│       │   ├── dim_dates.sql
│       │   └── fct_orders.sql
│       └── finance/
│           ├── _finance__models.yml
│           ├── fct_revenue.sql
│           └── fct_costs.sql
├── macros/
│   ├── generate_schema_name.sql
│   ├── cents_to_dollars.sql
│   └── date_spine.sql
├── tests/
│   └── generic/
│       └── test_positive_value.sql
├── seeds/
│   └── country_codes.csv
├── snapshots/
│   └── scd_customers.sql
└── semantic_layer/
    ├── metrics/
    │   ├── revenue.yml
    │   ├── users.yml
    │   └── orders.yml
    └── dimensions/
        ├── time.yml
        └── geography.yml
```

### Staging Models

```sql
-- models/staging/stripe/stg_stripe__payments.sql
with source as (
    select * from {{ source('stripe', 'payments') }}
),

renamed as (
    select
        -- ids
        id as payment_id,
        customer_id,
        invoice_id,

        -- properties
        amount as amount_cents,
        currency,
        status as payment_status,
        payment_method_types[0] as payment_method,

        -- timestamps
        created as created_at,
        {{ dbt_utils.safe_cast('metadata:order_id', 'string') }} as order_id

    from source
    where status = 'succeeded'  -- only successful payments
)

select * from renamed
```

```yaml
# models/staging/stripe/_stripe__sources.yml
version: 2

sources:
  - name: stripe
    database: raw
    schema: stripe
    description: Stripe payment data
    loader: fivetran
    loaded_at_field: _fivetran_synced

    freshness:
      warn_after: {count: 12, period: hour}
      error_after: {count: 24, period: hour}

    tables:
      - name: payments
        description: Stripe payment transactions
        columns:
          - name: id
            description: Primary key
            tests:
              - unique
              - not_null
          - name: customer_id
            description: Foreign key to customers
          - name: amount
            description: Payment amount in cents
```

### Intermediate Models

```sql
-- models/intermediate/finance/int_payments_with_customers.sql
with payments as (
    select * from {{ ref('stg_stripe__payments') }}
),

customers as (
    select * from {{ ref('stg_stripe__customers') }}
),

joined as (
    select
        payments.payment_id,
        payments.order_id,
        payments.amount_cents,
        payments.currency,
        payments.payment_method,
        payments.created_at,

        customers.customer_id,
        customers.email,
        customers.country_code,
        customers.created_at as customer_created_at,

        -- derived
        datediff('day', customers.created_at, payments.created_at) as days_since_signup

    from payments
    left join customers
        on payments.customer_id = customers.customer_id
)

select * from joined
```

### Mart Models

```sql
-- models/marts/core/dim_customers.sql
{{
    config(
        materialized='table',
        unique_key='customer_id',
        tags=['daily']
    )
}}

with customers as (
    select * from {{ ref('stg_stripe__customers') }}
),

orders as (
    select * from {{ ref('fct_orders') }}
),

customer_orders as (
    select
        customer_id,
        count(*) as total_orders,
        sum(order_total) as lifetime_value,
        min(order_date) as first_order_date,
        max(order_date) as last_order_date,
        avg(order_total) as average_order_value
    from orders
    group by 1
),

final as (
    select
        customers.customer_id,
        customers.email,
        customers.name,
        customers.country_code,
        customers.created_at,

        -- order metrics
        coalesce(customer_orders.total_orders, 0) as total_orders,
        coalesce(customer_orders.lifetime_value, 0) as lifetime_value,
        customer_orders.first_order_date,
        customer_orders.last_order_date,
        customer_orders.average_order_value,

        -- customer segments
        case
            when customer_orders.total_orders is null then 'prospect'
            when customer_orders.total_orders = 1 then 'new'
            when customer_orders.total_orders < 5 then 'active'
            else 'loyal'
        end as customer_segment,

        -- churn indicator
        case
            when datediff('day', customer_orders.last_order_date, current_date()) > 90
            then true
            else false
        end as is_churned

    from customers
    left join customer_orders
        on customers.customer_id = customer_orders.customer_id
)

select * from final
```

```sql
-- models/marts/core/fct_orders.sql
{{
    config(
        materialized='incremental',
        unique_key='order_id',
        incremental_strategy='merge',
        tags=['hourly']
    )
}}

with payments as (
    select * from {{ ref('int_payments_with_customers') }}
),

order_items as (
    select * from {{ ref('stg_shopify__order_items') }}
),

orders as (
    select
        payments.order_id,
        payments.customer_id,
        payments.created_at as order_date,
        payments.payment_method,
        payments.country_code,

        -- amounts
        {{ cents_to_dollars('payments.amount_cents') }} as order_total,

        -- item aggregations
        sum(order_items.quantity) as total_items,
        count(distinct order_items.product_id) as unique_products

    from payments
    left join order_items
        on payments.order_id = order_items.order_id
    group by 1, 2, 3, 4, 5, 6
)

select * from orders

{% if is_incremental() %}
where order_date > (select max(order_date) from {{ this }})
{% endif %}
```

---

## Semantic Layer

### Metric Definitions

```yaml
# semantic_layer/metrics/revenue.yml
version: 2

metrics:
  - name: total_revenue
    label: "Total Revenue"
    description: "Sum of all successful payment amounts"
    model: ref('fct_orders')
    calculation_method: sum
    expression: order_total

    timestamp: order_date
    time_grains: [day, week, month, quarter, year]

    dimensions:
      - customer_segment
      - country_code
      - payment_method

    filters:
      - field: order_status
        operator: '='
        value: "'completed'"

    meta:
      owner: finance-team
      tier: 1

  - name: average_order_value
    label: "Average Order Value"
    description: "Average revenue per order"
    model: ref('fct_orders')
    calculation_method: average
    expression: order_total

    timestamp: order_date
    time_grains: [day, week, month]

    dimensions:
      - customer_segment
      - country_code

  - name: revenue_per_user
    label: "Revenue per User"
    description: "Total revenue divided by active users"
    calculation_method: derived
    expression: "{{ metric('total_revenue') }} / {{ metric('active_users') }}"

    timestamp: order_date
    time_grains: [month, quarter]

  - name: revenue_growth_rate
    label: "Revenue Growth Rate"
    description: "Month-over-month revenue growth"
    calculation_method: derived
    expression: |
      ({{ metric('total_revenue') }} -
       {{ metric('total_revenue', offset_window=1) }}) /
      {{ metric('total_revenue', offset_window=1) }}

    timestamp: order_date
    time_grains: [month]
    meta:
      is_percentage: true
```

```yaml
# semantic_layer/metrics/users.yml
version: 2

metrics:
  - name: active_users
    label: "Active Users"
    description: "Users who placed at least one order in the period"
    model: ref('fct_orders')
    calculation_method: count_distinct
    expression: customer_id

    timestamp: order_date
    time_grains: [day, week, month]

    dimensions:
      - country_code
      - customer_segment

  - name: new_users
    label: "New Users"
    description: "Users who placed their first order in the period"
    model: ref('fct_orders')
    calculation_method: count_distinct
    expression: customer_id

    timestamp: order_date
    time_grains: [day, week, month]

    filters:
      - field: is_first_order
        operator: '='
        value: 'true'

  - name: user_retention_rate
    label: "User Retention Rate"
    description: "Percentage of users who returned after first purchase"
    calculation_method: derived
    expression: |
      {{ metric('returning_users') }} / {{ metric('new_users', offset_window=1) }}

    timestamp: order_date
    time_grains: [month]
```

### Semantic Query Engine

```python
from typing import List, Dict, Optional
from dataclasses import dataclass
import sqlglot

@dataclass
class MetricQuery:
    """Query specification for metrics"""
    metrics: List[str]
    dimensions: List[str]
    filters: List[Dict]
    time_grain: str
    start_date: str
    end_date: str

class SemanticQueryEngine:
    """Engine to translate semantic queries to SQL"""

    def __init__(self, metric_definitions: Dict, warehouse_type: str):
        self.metrics = metric_definitions
        self.warehouse_type = warehouse_type

    def generate_sql(self, query: MetricQuery) -> str:
        """Generate SQL from semantic query"""
        # Resolve metric definitions
        metric_specs = [self.metrics[m] for m in query.metrics]

        # Build SELECT clause
        select_parts = []

        # Time dimension
        time_col = self._get_time_column(metric_specs[0], query.time_grain)
        select_parts.append(f"{time_col} as period")

        # Regular dimensions
        for dim in query.dimensions:
            select_parts.append(dim)

        # Metrics
        for metric in metric_specs:
            agg_expr = self._build_aggregation(metric)
            select_parts.append(f"{agg_expr} as {metric['name']}")

        # Build FROM clause
        from_clause = self._build_from_clause(metric_specs, query.dimensions)

        # Build WHERE clause
        where_parts = []
        where_parts.append(f"{metric_specs[0]['timestamp']} >= '{query.start_date}'")
        where_parts.append(f"{metric_specs[0]['timestamp']} < '{query.end_date}'")

        for filter_spec in query.filters:
            where_parts.append(
                f"{filter_spec['field']} {filter_spec['operator']} {filter_spec['value']}"
            )

        # Add metric-level filters
        for metric in metric_specs:
            for f in metric.get('filters', []):
                where_parts.append(f"{f['field']} {f['operator']} {f['value']}")

        # Build GROUP BY clause
        group_parts = ['period'] + query.dimensions

        # Assemble SQL
        sql = f"""
        SELECT
            {', '.join(select_parts)}
        FROM {from_clause}
        WHERE {' AND '.join(where_parts)}
        GROUP BY {', '.join(group_parts)}
        ORDER BY period
        """

        return sql

    def _get_time_column(self, metric: Dict, grain: str) -> str:
        """Get time column with appropriate truncation"""
        timestamp = metric['timestamp']

        if self.warehouse_type == 'snowflake':
            return f"DATE_TRUNC('{grain}', {timestamp})"
        elif self.warehouse_type == 'bigquery':
            return f"DATE_TRUNC({timestamp}, {grain.upper()})"
        elif self.warehouse_type == 'postgres':
            return f"DATE_TRUNC('{grain}', {timestamp})"

    def _build_aggregation(self, metric: Dict) -> str:
        """Build aggregation expression"""
        method = metric['calculation_method']
        expr = metric['expression']

        if method == 'sum':
            return f"SUM({expr})"
        elif method == 'count':
            return f"COUNT({expr})"
        elif method == 'count_distinct':
            return f"COUNT(DISTINCT {expr})"
        elif method == 'average':
            return f"AVG({expr})"
        elif method == 'min':
            return f"MIN({expr})"
        elif method == 'max':
            return f"MAX({expr})"
        elif method == 'derived':
            return self._expand_derived_metric(expr)

    def _expand_derived_metric(self, expression: str) -> str:
        """Expand derived metric references"""
        # Parse metric references like {{ metric('total_revenue') }}
        import re

        def replace_metric(match):
            metric_name = match.group(1)
            metric_def = self.metrics[metric_name]
            return self._build_aggregation(metric_def)

        pattern = r"\{\{\s*metric\('(\w+)'\)\s*\}\}"
        return re.sub(pattern, replace_metric, expression)

    def _build_from_clause(self, metrics: List[Dict], dimensions: List[str]) -> str:
        """Build FROM clause with necessary joins"""
        # Get all required tables
        tables = set()
        for metric in metrics:
            tables.add(metric['model'])

        # For now, assume single table
        # In production, would need to handle joins
        return list(tables)[0]
```

### Metric API

```python
from fastapi import FastAPI, Query
from typing import List, Optional
from datetime import date

app = FastAPI()

@app.get("/metrics/{metric_name}")
async def get_metric(
    metric_name: str,
    start_date: date = Query(...),
    end_date: date = Query(...),
    time_grain: str = Query("day"),
    dimensions: List[str] = Query([]),
    filters: List[str] = Query([])
):
    """Query a single metric"""
    query = MetricQuery(
        metrics=[metric_name],
        dimensions=dimensions,
        filters=parse_filters(filters),
        time_grain=time_grain,
        start_date=str(start_date),
        end_date=str(end_date)
    )

    sql = engine.generate_sql(query)
    result = warehouse.execute(sql)

    return {
        "metric": metric_name,
        "data": result,
        "sql": sql if request.query_params.get("show_sql") else None
    }

@app.get("/metrics")
async def list_metrics(
    category: Optional[str] = None,
    search: Optional[str] = None
):
    """List available metrics"""
    metrics = metric_catalog.list_metrics(
        category=category,
        search=search
    )

    return {
        "metrics": [
            {
                "name": m.name,
                "label": m.label,
                "description": m.description,
                "dimensions": m.dimensions,
                "time_grains": m.time_grains
            }
            for m in metrics
        ]
    }

@app.post("/metrics/query")
async def query_metrics(request: MetricQueryRequest):
    """Query multiple metrics together"""
    query = MetricQuery(
        metrics=request.metrics,
        dimensions=request.dimensions,
        filters=request.filters,
        time_grain=request.time_grain,
        start_date=request.start_date,
        end_date=request.end_date
    )

    sql = engine.generate_sql(query)
    result = warehouse.execute(sql)

    return {
        "data": result,
        "metadata": {
            "metrics": request.metrics,
            "row_count": len(result)
        }
    }
```

---

## Data Structures

### Model Configuration

```yaml
# dbt_project.yml
name: 'analytics'
version: '1.0.0'

config-version: 2

profile: 'analytics'

model-paths: ["models"]
analysis-paths: ["analyses"]
test-paths: ["tests"]
seed-paths: ["seeds"]
macro-paths: ["macros"]
snapshot-paths: ["snapshots"]

target-path: "target"
clean-targets:
  - "target"
  - "dbt_packages"

vars:
  start_date: '2020-01-01'

models:
  analytics:
    +materialized: view
    +persist_docs:
      relation: true
      columns: true

    staging:
      +materialized: view
      +schema: staging

    intermediate:
      +materialized: ephemeral

    marts:
      +materialized: table
      +schema: analytics
      core:
        +tags: ['daily', 'core']
      finance:
        +tags: ['daily', 'finance']
```

### Test Definitions

```yaml
# models/marts/core/_core__models.yml
version: 2

models:
  - name: dim_customers
    description: "Customer dimension table"
    columns:
      - name: customer_id
        description: "Primary key"
        tests:
          - unique
          - not_null

      - name: email
        tests:
          - not_null

      - name: customer_segment
        tests:
          - accepted_values:
              values: ['prospect', 'new', 'active', 'loyal']

      - name: lifetime_value
        tests:
          - dbt_utils.expression_is_true:
              expression: ">= 0"

  - name: fct_orders
    description: "Order fact table"
    tests:
      - dbt_utils.unique_combination_of_columns:
          combination_of_columns:
            - order_id

    columns:
      - name: order_id
        description: "Primary key"
        tests:
          - unique
          - not_null

      - name: customer_id
        description: "Foreign key to dim_customers"
        tests:
          - not_null
          - relationships:
              to: ref('dim_customers')
              field: customer_id

      - name: order_total
        tests:
          - not_null
          - dbt_utils.expression_is_true:
              expression: ">= 0"
```

### Freshness Configuration

```yaml
# models/staging/stripe/_stripe__sources.yml
sources:
  - name: stripe
    freshness:
      warn_after: {count: 12, period: hour}
      error_after: {count: 24, period: hour}

    tables:
      - name: payments
        loaded_at_field: _fivetran_synced
        freshness:
          warn_after: {count: 6, period: hour}
          error_after: {count: 12, period: hour}
```

---

## API Design

### dbt Commands

```bash
# Development workflow
dbt debug              # Test warehouse connection
dbt deps               # Install packages
dbt seed               # Load seed data
dbt run                # Run all models
dbt test               # Run all tests
dbt docs generate      # Generate documentation
dbt docs serve         # Serve documentation

# Selective execution
dbt run --select staging.stripe    # Run all Stripe staging models
dbt run --select +fct_orders       # Run fct_orders and all upstream
dbt run --select fct_orders+       # Run fct_orders and all downstream
dbt run --select tag:daily         # Run models tagged 'daily'

# Incremental runs
dbt run --select state:modified    # Only modified models
dbt run --full-refresh             # Rebuild incremental models

# Testing
dbt test --select fct_orders       # Test specific model
dbt source freshness               # Check source freshness
```

### Semantic Layer API

```yaml
openapi: 3.0.0
info:
  title: Semantic Layer API
  version: 1.0.0

paths:
  /metrics:
    get:
      summary: List available metrics
      parameters:
        - name: category
          in: query
          schema:
            type: string
        - name: search
          in: query
          schema:
            type: string
      responses:
        200:
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/MetricList'

  /metrics/{metric_name}:
    get:
      summary: Get metric definition
      responses:
        200:
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/MetricDefinition'

  /metrics/{metric_name}/query:
    get:
      summary: Query a metric
      parameters:
        - name: metric_name
          in: path
          required: true
          schema:
            type: string
        - name: start_date
          in: query
          required: true
          schema:
            type: string
            format: date
        - name: end_date
          in: query
          required: true
          schema:
            type: string
            format: date
        - name: time_grain
          in: query
          schema:
            type: string
            enum: [day, week, month, quarter, year]
        - name: dimensions
          in: query
          schema:
            type: array
            items:
              type: string
      responses:
        200:
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/MetricResult'

  /query:
    post:
      summary: Query multiple metrics
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/QueryRequest'
      responses:
        200:
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/QueryResult'

components:
  schemas:
    QueryRequest:
      type: object
      required:
        - metrics
        - start_date
        - end_date
      properties:
        metrics:
          type: array
          items:
            type: string
        dimensions:
          type: array
          items:
            type: string
        filters:
          type: array
          items:
            $ref: '#/components/schemas/Filter'
        time_grain:
          type: string
        start_date:
          type: string
          format: date
        end_date:
          type: string
          format: date
```

---

## Enterprise Features

### 1. Comprehensive Test Suite

```yaml
# tests/generic/test_positive_value.sql
{% test positive_value(model, column_name) %}

select
    {{ column_name }} as invalid_value,
    count(*) as occurrences
from {{ model }}
where {{ column_name }} < 0
group by 1
having count(*) > 0

{% endtest %}
```

```yaml
# tests/singular/test_revenue_reconciliation.sql
-- Ensure dbt revenue matches source system
with dbt_revenue as (
    select sum(order_total) as total
    from {{ ref('fct_orders') }}
    where order_date between '{{ var("start_date") }}' and '{{ var("end_date") }}'
),

source_revenue as (
    select sum(amount) / 100 as total
    from {{ source('stripe', 'payments') }}
    where created between '{{ var("start_date") }}' and '{{ var("end_date") }}'
)

select
    dbt_revenue.total as dbt_total,
    source_revenue.total as source_total,
    abs(dbt_revenue.total - source_revenue.total) as difference
from dbt_revenue, source_revenue
where abs(dbt_revenue.total - source_revenue.total) > 1  -- Allow $1 variance
```

### 2. Column-Level Lineage

```python
class ColumnLineageExtractor:
    """Extract column-level lineage from dbt models"""

    def __init__(self, project_path: str):
        self.project_path = project_path
        self.parser = sqlglot

    def extract_lineage(self, model_name: str) -> Dict:
        """Extract column lineage for a model"""
        model_sql = self._read_model(model_name)

        # Parse SQL
        ast = self.parser.parse(model_sql)[0]

        # Extract column expressions
        columns = {}
        for select in ast.find_all(sqlglot.exp.Select):
            for expr in select.expressions:
                col_name = expr.alias or str(expr)
                source_columns = self._extract_source_columns(expr)
                columns[col_name] = source_columns

        return columns

    def _extract_source_columns(self, expression) -> List[str]:
        """Extract source columns from an expression"""
        sources = []

        for col in expression.find_all(sqlglot.exp.Column):
            table = col.table or 'unknown'
            column = col.name
            sources.append(f"{table}.{column}")

        return sources

    def generate_lineage_graph(self) -> Dict:
        """Generate full column lineage graph"""
        graph = {}

        for model in self._get_all_models():
            lineage = self.extract_lineage(model)
            graph[model] = lineage

        return graph
```

### 3. CI/CD Integration

```yaml
# .github/workflows/dbt.yml
name: dbt CI

on:
  pull_request:
    paths:
      - 'models/**'
      - 'macros/**'
      - 'tests/**'

jobs:
  dbt-test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dbt
        run: pip install dbt-snowflake

      - name: Install packages
        run: dbt deps

      - name: Check SQL compilation
        run: dbt compile

      - name: Run modified models
        run: dbt run --select state:modified+

      - name: Test modified models
        run: dbt test --select state:modified+

      - name: Generate docs
        run: dbt docs generate

      - name: Check documentation coverage
        run: python scripts/check_docs_coverage.py

      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: dbt-artifacts
          path: target/
```

### 4. Freshness Monitoring

```python
class FreshnessMonitor:
    """Monitor dbt source freshness"""

    def __init__(self, dbt_project_path: str):
        self.project_path = dbt_project_path

    async def check_freshness(self) -> List[FreshnessResult]:
        """Check freshness of all sources"""
        # Run dbt source freshness
        result = subprocess.run(
            ['dbt', 'source', 'freshness', '--output', 'json'],
            cwd=self.project_path,
            capture_output=True
        )

        freshness_data = json.loads(result.stdout)
        results = []

        for source in freshness_data['results']:
            status = 'pass'
            if source['status'] == 'warn':
                status = 'warn'
            elif source['status'] == 'error':
                status = 'error'

            results.append(FreshnessResult(
                source_name=source['unique_id'],
                status=status,
                max_loaded_at=source['max_loaded_at'],
                snapshotted_at=source['snapshotted_at'],
                age_hours=source['age']
            ))

        return results

    async def alert_on_stale(self, results: List[FreshnessResult]):
        """Send alerts for stale sources"""
        stale_sources = [r for r in results if r.status in ('warn', 'error')]

        if stale_sources:
            await self._send_alert(stale_sources)
```

---

## Performance Considerations

### Incremental Models

```sql
-- Efficient incremental model
{{
    config(
        materialized='incremental',
        unique_key='event_id',
        incremental_strategy='merge',
        partition_by={
            'field': 'event_date',
            'data_type': 'date',
            'granularity': 'day'
        },
        cluster_by=['customer_id']
    )
}}

with new_events as (
    select * from {{ ref('stg_events') }}

    {% if is_incremental() %}
    where event_timestamp > (select max(event_timestamp) from {{ this }})
    {% endif %}
)

select
    event_id,
    customer_id,
    event_type,
    event_timestamp,
    date(event_timestamp) as event_date
from new_events
```

### Query Optimization

```sql
-- Optimized aggregation with pre-filtering
{{
    config(
        materialized='table',
        sort=['order_date'],
        dist='customer_id'
    )
}}

with filtered_orders as (
    -- Filter early to reduce data volume
    select *
    from {{ ref('stg_orders') }}
    where order_status = 'completed'
      and order_date >= dateadd('month', -12, current_date())
),

aggregated as (
    select
        customer_id,
        date_trunc('month', order_date) as order_month,
        count(*) as order_count,
        sum(order_total) as total_revenue
    from filtered_orders
    group by 1, 2
)

select * from aggregated
```

### Materialization Strategy

```python
# Materialization decision tree
def get_materialization(model_properties: dict) -> str:
    """Determine optimal materialization"""
    row_count = model_properties.get('estimated_rows', 0)
    is_frequently_queried = model_properties.get('query_frequency', 0) > 100
    has_complex_logic = model_properties.get('complexity', 'low') == 'high'
    is_incremental_candidate = model_properties.get('has_timestamp', False)

    # Staging: always views (fast compilation, low storage)
    if model_properties['layer'] == 'staging':
        return 'view'

    # Intermediate: ephemeral (reduces warehouse tables)
    if model_properties['layer'] == 'intermediate':
        return 'ephemeral'

    # Marts with incremental support
    if is_incremental_candidate and row_count > 1_000_000:
        return 'incremental'

    # Frequently queried or complex: table
    if is_frequently_queried or has_complex_logic:
        return 'table'

    # Default: view
    return 'view'
```

---

## Stretch Goals

### 1. Custom Query Planner

```python
class QueryPlanner:
    """Optimize semantic queries"""

    def __init__(self, catalog: MetricCatalog):
        self.catalog = catalog

    def plan_query(self, query: MetricQuery) -> QueryPlan:
        """Generate optimized query plan"""
        # Analyze metrics for common base
        metric_specs = [self.catalog.get_metric(m) for m in query.metrics]
        common_model = self._find_common_model(metric_specs)

        # Check for pre-computed aggregates
        agg_table = self._find_aggregate_table(
            metrics=query.metrics,
            dimensions=query.dimensions,
            time_grain=query.time_grain
        )

        if agg_table:
            # Use pre-computed aggregate
            return QueryPlan(
                type='aggregate_lookup',
                source=agg_table,
                metrics=query.metrics,
                dimensions=query.dimensions
            )

        # Check for rollup opportunity
        if self._can_rollup(query):
            finer_grain = self._get_finer_grain(query.time_grain)
            return QueryPlan(
                type='rollup',
                source=common_model,
                base_grain=finer_grain,
                target_grain=query.time_grain,
                metrics=query.metrics,
                dimensions=query.dimensions
            )

        # Standard query
        return QueryPlan(
            type='direct',
            source=common_model,
            metrics=query.metrics,
            dimensions=query.dimensions
        )

    def _find_aggregate_table(
        self,
        metrics: List[str],
        dimensions: List[str],
        time_grain: str
    ) -> Optional[str]:
        """Find pre-computed aggregate table"""
        # Look for matching aggregate
        for agg in self.catalog.get_aggregates():
            if (set(metrics).issubset(set(agg.metrics)) and
                set(dimensions).issubset(set(agg.dimensions)) and
                time_grain in agg.time_grains):
                return agg.table_name

        return None
```

### 2. Real-Time Pull-Through

```python
class RealTimePullThrough:
    """Pull-through cache for real-time metrics"""

    def __init__(self, cache: Redis, warehouse: Warehouse):
        self.cache = cache
        self.warehouse = warehouse

    async def get_metric(
        self,
        metric_name: str,
        dimensions: Dict[str, str],
        cache_ttl: int = 60
    ) -> float:
        """Get metric with pull-through caching"""
        cache_key = self._build_cache_key(metric_name, dimensions)

        # Check cache
        cached = await self.cache.get(cache_key)
        if cached:
            return float(cached)

        # Query warehouse
        query = self._build_query(metric_name, dimensions)
        result = await self.warehouse.execute_async(query)
        value = result[0][0]

        # Cache result
        await self.cache.setex(cache_key, cache_ttl, str(value))

        return value

    async def get_metric_stream(
        self,
        metric_name: str,
        dimensions: Dict[str, str]
    ) -> AsyncIterator[float]:
        """Stream metric updates"""
        while True:
            value = await self.get_metric(metric_name, dimensions, cache_ttl=0)
            yield value
            await asyncio.sleep(1)
```

---

## Testing Strategy

### Unit Tests

```python
import pytest
from dbt.tests.util import run_dbt, check_result_nodes_by_name

class TestStagingModels:
    @pytest.fixture(scope="class")
    def project_config_update(self):
        return {
            "name": "test_project",
        }

    def test_stg_stripe_payments(self, project):
        """Test staging model transforms correctly"""
        # Seed test data
        run_dbt(["seed"])

        # Run model
        results = run_dbt(["run", "--select", "stg_stripe__payments"])
        assert len(results) == 1
        assert results[0].status == "success"

        # Test model
        test_results = run_dbt(["test", "--select", "stg_stripe__payments"])
        assert all(r.status == "pass" for r in test_results)

class TestMartModels:
    def test_dim_customers_segments(self, project):
        """Test customer segmentation logic"""
        run_dbt(["run", "--select", "dim_customers"])

        # Query results
        results = project.run_sql("""
            select customer_segment, count(*)
            from {{ ref('dim_customers') }}
            group by 1
        """)

        # Verify segments
        segments = {r[0] for r in results}
        assert segments == {'prospect', 'new', 'active', 'loyal'}
```

### Integration Tests

```python
class TestDataReconciliation:
    """Test data reconciliation between layers"""

    def test_revenue_matches_source(self, project):
        """Ensure mart revenue matches staging"""
        run_dbt(["run"])

        result = project.run_sql("""
            with staging as (
                select sum(amount_cents) / 100 as revenue
                from {{ ref('stg_stripe__payments') }}
            ),
            mart as (
                select sum(order_total) as revenue
                from {{ ref('fct_orders') }}
            )
            select
                staging.revenue as staging_revenue,
                mart.revenue as mart_revenue,
                abs(staging.revenue - mart.revenue) as diff
            from staging, mart
        """)

        assert result[0][2] < 1  # Less than $1 difference

    def test_row_count_consistency(self, project):
        """Test row counts match expected"""
        run_dbt(["run"])

        result = project.run_sql("""
            select count(*) from {{ ref('fct_orders') }}
        """)

        expected_count = project.run_sql("""
            select count(distinct order_id)
            from {{ ref('stg_stripe__payments') }}
        """)

        assert result[0][0] == expected_count[0][0]
```

### Semantic Layer Tests

```python
class TestSemanticLayer:
    """Test semantic layer queries"""

    def test_metric_query_generation(self):
        """Test SQL generation"""
        engine = SemanticQueryEngine(metrics, 'snowflake')

        query = MetricQuery(
            metrics=['total_revenue'],
            dimensions=['country_code'],
            filters=[],
            time_grain='month',
            start_date='2024-01-01',
            end_date='2024-03-01'
        )

        sql = engine.generate_sql(query)

        assert 'SUM(order_total)' in sql
        assert "DATE_TRUNC('month'" in sql
        assert 'GROUP BY' in sql

    def test_derived_metric(self):
        """Test derived metric calculation"""
        query = MetricQuery(
            metrics=['revenue_per_user'],
            dimensions=[],
            filters=[],
            time_grain='month',
            start_date='2024-01-01',
            end_date='2024-03-01'
        )

        sql = engine.generate_sql(query)

        assert 'SUM(order_total)' in sql
        assert 'COUNT(DISTINCT customer_id)' in sql
```

---

## Implementation Phases

### Phase 1: dbt Foundation (Weeks 1-2)
- Project setup and structure
- Staging models for primary sources
- Basic macros and tests
- Documentation setup

### Phase 2: Core Marts (Weeks 3-4)
- Dimensional models (customers, products, dates)
- Fact tables (orders, revenue)
- Incremental models
- Comprehensive testing

### Phase 3: Semantic Layer (Weeks 5-6)
- Metric definitions (YAML)
- Query engine
- API implementation
- Metric validation

### Phase 4: Enterprise Features (Weeks 7-8)
- CI/CD pipeline
- Column-level lineage
- Freshness monitoring
- dbt tests in production

### Phase 5: Advanced Features (Weeks 9-10)
- Custom query planner
- Real-time pull-through
- Performance optimization
- Documentation completion

---

## References

- [dbt Documentation](https://docs.getdbt.com/)
- [dbt Best Practices](https://docs.getdbt.com/best-practices)
- [Semantic Layer Guide](https://docs.getdbt.com/docs/build/metrics)
- [The Data Warehouse Toolkit](https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/books/)
- [Metrics Layer Spec](https://github.com/dbt-labs/semantic-layer-spec)
