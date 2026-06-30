"""Fact table model definitions and builders."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FactColumn:
    """Column in a fact table."""

    name: str
    data_type: str
    description: str
    is_key: bool = False
    is_foreign_key: bool = False
    is_measure: bool = False
    is_degenerate_dimension: bool = False


@dataclass
class FactModel:
    """Definition of a fact table model."""

    name: str
    description: str
    source_models: List[str]
    columns: List[FactColumn]
    grain: str
    unique_key: str = ""
    incremental_strategy: Optional[str] = None
    partition_by: Optional[Dict[str, Any]] = None
    config: Dict[str, Any] = field(default_factory=dict)


class FctOrders:
    """Order fact table model definition."""

    @staticmethod
    def get_definition() -> FactModel:
        """Get the fct_orders model definition."""
        return FactModel(
            name="fct_orders",
            description="Order fact table - one row per order",
            source_models=["int_payments_with_customers", "stg_shopify__order_items"],
            grain="one row per order",
            unique_key="order_id",
            incremental_strategy="merge",
            columns=[
                FactColumn(
                    name="order_id",
                    data_type="string",
                    description="Primary key - unique order identifier",
                    is_key=True,
                ),
                FactColumn(
                    name="customer_id",
                    data_type="string",
                    description="Foreign key to dim_customers",
                    is_foreign_key=True,
                ),
                FactColumn(
                    name="order_date",
                    data_type="timestamp",
                    description="Order timestamp",
                ),
                FactColumn(
                    name="payment_method",
                    data_type="string",
                    description="Payment method used",
                    is_degenerate_dimension=True,
                ),
                FactColumn(
                    name="country_code",
                    data_type="string",
                    description="Country of order",
                    is_degenerate_dimension=True,
                ),
                FactColumn(
                    name="order_total",
                    data_type="decimal",
                    description="Total order amount in dollars",
                    is_measure=True,
                ),
                FactColumn(
                    name="total_items",
                    data_type="integer",
                    description="Total quantity of items",
                    is_measure=True,
                ),
                FactColumn(
                    name="unique_products",
                    data_type="integer",
                    description="Count of unique products in order",
                    is_measure=True,
                ),
            ],
            config={
                "materialized": "incremental",
                "tags": ["hourly"],
            },
        )

    @staticmethod
    def generate_sql() -> str:
        """Generate the SQL for fct_orders."""
        return """{{
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

        {{ cents_to_dollars('payments.amount_cents') }} as order_total,

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
{% endif %}"""


class FctRevenue:
    """Revenue fact table model definition."""

    @staticmethod
    def get_definition() -> FactModel:
        """Get the fct_revenue model definition."""
        return FactModel(
            name="fct_revenue",
            description="Daily revenue aggregation fact table",
            source_models=["fct_orders"],
            grain="one row per day per customer segment per country",
            unique_key="revenue_id",
            columns=[
                FactColumn(
                    name="revenue_id",
                    data_type="string",
                    description="Surrogate key",
                    is_key=True,
                ),
                FactColumn(
                    name="revenue_date",
                    data_type="date",
                    description="Date of revenue",
                ),
                FactColumn(
                    name="customer_segment",
                    data_type="string",
                    description="Customer segment",
                    is_degenerate_dimension=True,
                ),
                FactColumn(
                    name="country_code",
                    data_type="string",
                    description="Country code",
                    is_degenerate_dimension=True,
                ),
                FactColumn(
                    name="order_count",
                    data_type="integer",
                    description="Number of orders",
                    is_measure=True,
                ),
                FactColumn(
                    name="total_revenue",
                    data_type="decimal",
                    description="Sum of order totals",
                    is_measure=True,
                ),
                FactColumn(
                    name="avg_order_value",
                    data_type="decimal",
                    description="Average order value",
                    is_measure=True,
                ),
                FactColumn(
                    name="unique_customers",
                    data_type="integer",
                    description="Distinct customer count",
                    is_measure=True,
                ),
            ],
            config={
                "materialized": "table",
                "tags": ["daily", "finance"],
            },
        )

    @staticmethod
    def generate_sql() -> str:
        """Generate the SQL for fct_revenue."""
        return """{{
    config(
        materialized='table',
        tags=['daily', 'finance']
    )
}}

with orders as (
    select * from {{ ref('fct_orders') }}
),

customers as (
    select * from {{ ref('dim_customers') }}
),

revenue as (
    select
        {{ dbt_utils.generate_surrogate_key([
            'date(orders.order_date)',
            'customers.customer_segment',
            'orders.country_code'
        ]) }} as revenue_id,

        date(orders.order_date) as revenue_date,
        customers.customer_segment,
        orders.country_code,

        count(*) as order_count,
        sum(orders.order_total) as total_revenue,
        avg(orders.order_total) as avg_order_value,
        count(distinct orders.customer_id) as unique_customers

    from orders
    join customers
        on orders.customer_id = customers.customer_id
    group by 1, 2, 3, 4
)

select * from revenue"""


class FctCosts:
    """Cost fact table model definition."""

    @staticmethod
    def get_definition() -> FactModel:
        """Get the fct_costs model definition."""
        return FactModel(
            name="fct_costs",
            description="Cost tracking fact table",
            source_models=["stg_erp__costs"],
            grain="one row per cost entry",
            unique_key="cost_id",
            columns=[
                FactColumn(
                    name="cost_id",
                    data_type="string",
                    description="Primary key",
                    is_key=True,
                ),
                FactColumn(
                    name="cost_date",
                    data_type="date",
                    description="Date of cost",
                ),
                FactColumn(
                    name="cost_type",
                    data_type="string",
                    description="Type of cost (COGS, marketing, etc.)",
                    is_degenerate_dimension=True,
                ),
                FactColumn(
                    name="cost_center",
                    data_type="string",
                    description="Cost center",
                    is_degenerate_dimension=True,
                ),
                FactColumn(
                    name="amount",
                    data_type="decimal",
                    description="Cost amount",
                    is_measure=True,
                ),
            ],
            config={
                "materialized": "table",
                "tags": ["daily", "finance"],
            },
        )


class FactBuilder:
    """Builder for creating fact table models."""

    def __init__(self):
        self._facts: Dict[str, FactModel] = {}

    def add_fact(self, fact: FactModel) -> "FactBuilder":
        """Add a fact table to the builder."""
        self._facts[fact.name] = fact
        return self

    def get_fact(self, name: str) -> Optional[FactModel]:
        """Get a fact table by name."""
        return self._facts.get(name)

    def list_facts(self) -> List[str]:
        """List all fact table names."""
        return list(self._facts.keys())

    def generate_yaml(self, fact_name: str) -> str:
        """Generate YAML schema for a fact table."""
        fact = self._facts.get(fact_name)
        if not fact:
            return ""

        columns_yaml = []
        for col in fact.columns:
            col_yaml = f"""      - name: {col.name}
        description: "{col.description}"
        tests:"""
            if col.is_key:
                col_yaml += """
          - unique
          - not_null"""
            elif col.is_foreign_key:
                col_yaml += """
          - not_null
          - relationships:
              to: ref('dim_customers')
              field: customer_id"""
            elif col.is_measure:
                col_yaml += """
          - not_null
          - dbt_utils.expression_is_true:
              expression: ">= 0\""""
            columns_yaml.append(col_yaml)

        return f"""version: 2

models:
  - name: {fact.name}
    description: "{fact.description}"
    tests:
      - dbt_utils.unique_combination_of_columns:
          combination_of_columns:
            - {fact.unique_key}
    columns:
{chr(10).join(columns_yaml)}"""

    def get_incremental_facts(self) -> List[str]:
        """Get list of incremental fact tables."""
        return [
            name for name, fact in self._facts.items()
            if fact.config.get("materialized") == "incremental"
        ]
