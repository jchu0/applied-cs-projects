"""Dimension model definitions and builders."""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DimensionColumn:
    """Column in a dimension table."""

    name: str
    data_type: str
    description: str
    is_key: bool = False
    is_natural_key: bool = False
    is_scd_valid: bool = False


@dataclass
class DimensionModel:
    """Definition of a dimension model."""

    name: str
    description: str
    source_model: str
    columns: List[DimensionColumn]
    grain: str
    scd_type: int = 1  # 1 for Type 1, 2 for Type 2
    unique_key: str = ""
    config: Dict[str, Any] = field(default_factory=dict)


class DimCustomers:
    """Customer dimension model definition."""

    @staticmethod
    def get_definition() -> DimensionModel:
        """Get the dim_customers model definition."""
        return DimensionModel(
            name="dim_customers",
            description="Customer dimension table with segmentation",
            source_model="stg_stripe__customers",
            grain="one row per customer",
            unique_key="customer_id",
            columns=[
                DimensionColumn(
                    name="customer_id",
                    data_type="string",
                    description="Primary key",
                    is_key=True,
                ),
                DimensionColumn(
                    name="email",
                    data_type="string",
                    description="Customer email address",
                    is_natural_key=True,
                ),
                DimensionColumn(
                    name="name",
                    data_type="string",
                    description="Customer full name",
                ),
                DimensionColumn(
                    name="country_code",
                    data_type="string",
                    description="ISO country code",
                ),
                DimensionColumn(
                    name="created_at",
                    data_type="timestamp",
                    description="Account creation timestamp",
                ),
                DimensionColumn(
                    name="total_orders",
                    data_type="integer",
                    description="Lifetime order count",
                ),
                DimensionColumn(
                    name="lifetime_value",
                    data_type="decimal",
                    description="Lifetime revenue from customer",
                ),
                DimensionColumn(
                    name="first_order_date",
                    data_type="date",
                    description="Date of first order",
                ),
                DimensionColumn(
                    name="last_order_date",
                    data_type="date",
                    description="Date of most recent order",
                ),
                DimensionColumn(
                    name="average_order_value",
                    data_type="decimal",
                    description="Average order value",
                ),
                DimensionColumn(
                    name="customer_segment",
                    data_type="string",
                    description="Customer segment (prospect/new/active/loyal)",
                ),
                DimensionColumn(
                    name="is_churned",
                    data_type="boolean",
                    description="Whether customer has churned (no orders in 90 days)",
                ),
            ],
            config={
                "materialized": "table",
                "tags": ["daily"],
            },
        )

    @staticmethod
    def generate_sql() -> str:
        """Generate the SQL for dim_customers."""
        return """{{
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

        coalesce(customer_orders.total_orders, 0) as total_orders,
        coalesce(customer_orders.lifetime_value, 0) as lifetime_value,
        customer_orders.first_order_date,
        customer_orders.last_order_date,
        customer_orders.average_order_value,

        case
            when customer_orders.total_orders is null then 'prospect'
            when customer_orders.total_orders = 1 then 'new'
            when customer_orders.total_orders < 5 then 'active'
            else 'loyal'
        end as customer_segment,

        case
            when datediff('day', customer_orders.last_order_date, current_date()) > 90
            then true
            else false
        end as is_churned

    from customers
    left join customer_orders
        on customers.customer_id = customer_orders.customer_id
)

select * from final"""


class DimProducts:
    """Product dimension model definition."""

    @staticmethod
    def get_definition() -> DimensionModel:
        """Get the dim_products model definition."""
        return DimensionModel(
            name="dim_products",
            description="Product dimension table",
            source_model="stg_shopify__products",
            grain="one row per product",
            unique_key="product_id",
            columns=[
                DimensionColumn(
                    name="product_id",
                    data_type="string",
                    description="Primary key",
                    is_key=True,
                ),
                DimensionColumn(
                    name="product_name",
                    data_type="string",
                    description="Product name",
                ),
                DimensionColumn(
                    name="category",
                    data_type="string",
                    description="Product category",
                ),
                DimensionColumn(
                    name="subcategory",
                    data_type="string",
                    description="Product subcategory",
                ),
                DimensionColumn(
                    name="brand",
                    data_type="string",
                    description="Product brand",
                ),
                DimensionColumn(
                    name="price",
                    data_type="decimal",
                    description="Current price",
                ),
                DimensionColumn(
                    name="cost",
                    data_type="decimal",
                    description="Product cost",
                ),
                DimensionColumn(
                    name="is_active",
                    data_type="boolean",
                    description="Whether product is currently available",
                ),
            ],
            config={
                "materialized": "table",
                "tags": ["daily"],
            },
        )


class DimDates:
    """Date dimension model definition."""

    @staticmethod
    def get_definition() -> DimensionModel:
        """Get the dim_dates model definition."""
        return DimensionModel(
            name="dim_dates",
            description="Date dimension table",
            source_model="date_spine",
            grain="one row per date",
            unique_key="date_key",
            columns=[
                DimensionColumn(
                    name="date_key",
                    data_type="integer",
                    description="Primary key (YYYYMMDD)",
                    is_key=True,
                ),
                DimensionColumn(
                    name="date_actual",
                    data_type="date",
                    description="The actual date",
                ),
                DimensionColumn(
                    name="day_of_week",
                    data_type="integer",
                    description="Day of week (1-7)",
                ),
                DimensionColumn(
                    name="day_name",
                    data_type="string",
                    description="Day name (Monday, etc.)",
                ),
                DimensionColumn(
                    name="is_weekend",
                    data_type="boolean",
                    description="Whether date is weekend",
                ),
                DimensionColumn(
                    name="week_of_year",
                    data_type="integer",
                    description="Week number in year",
                ),
                DimensionColumn(
                    name="month_number",
                    data_type="integer",
                    description="Month number (1-12)",
                ),
                DimensionColumn(
                    name="month_name",
                    data_type="string",
                    description="Month name",
                ),
                DimensionColumn(
                    name="quarter",
                    data_type="integer",
                    description="Quarter (1-4)",
                ),
                DimensionColumn(
                    name="year",
                    data_type="integer",
                    description="Year",
                ),
                DimensionColumn(
                    name="is_month_end",
                    data_type="boolean",
                    description="Whether last day of month",
                ),
                DimensionColumn(
                    name="is_quarter_end",
                    data_type="boolean",
                    description="Whether last day of quarter",
                ),
                DimensionColumn(
                    name="is_year_end",
                    data_type="boolean",
                    description="Whether last day of year",
                ),
            ],
            config={
                "materialized": "table",
                "tags": ["static"],
            },
        )

    @staticmethod
    def generate_date_spine(start_date: date, end_date: date) -> List[Dict[str, Any]]:
        """Generate date spine data."""
        dates = []
        current = start_date
        day_names = [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday",
        ]
        month_names = [
            "", "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ]

        while current <= end_date:
            date_key = int(current.strftime("%Y%m%d"))
            day_of_week = current.weekday() + 1
            month = current.month
            year = current.year

            # Calculate end of month
            if month == 12:
                next_month = date(year + 1, 1, 1)
            else:
                next_month = date(year, month + 1, 1)
            month_end = next_month - timedelta(days=1)

            # Calculate end of quarter
            quarter = (month - 1) // 3 + 1
            quarter_end_month = quarter * 3
            if quarter_end_month == 12:
                quarter_end = date(year, 12, 31)
            else:
                quarter_end = date(year, quarter_end_month + 1, 1) - timedelta(days=1)

            dates.append({
                "date_key": date_key,
                "date_actual": current,
                "day_of_week": day_of_week,
                "day_name": day_names[current.weekday()],
                "is_weekend": day_of_week >= 6,
                "week_of_year": current.isocalendar()[1],
                "month_number": month,
                "month_name": month_names[month],
                "quarter": quarter,
                "year": year,
                "is_month_end": current == month_end,
                "is_quarter_end": current == quarter_end,
                "is_year_end": current == date(year, 12, 31),
            })

            current += timedelta(days=1)

        return dates


class DimensionBuilder:
    """Builder for creating dimension models."""

    def __init__(self):
        self._dimensions: Dict[str, DimensionModel] = {}

    def add_dimension(self, dimension: DimensionModel) -> "DimensionBuilder":
        """Add a dimension to the builder."""
        self._dimensions[dimension.name] = dimension
        return self

    def get_dimension(self, name: str) -> Optional[DimensionModel]:
        """Get a dimension by name."""
        return self._dimensions.get(name)

    def list_dimensions(self) -> List[str]:
        """List all dimension names."""
        return list(self._dimensions.keys())

    def generate_yaml(self, dimension_name: str) -> str:
        """Generate YAML schema for a dimension."""
        dimension = self._dimensions.get(dimension_name)
        if not dimension:
            return ""

        columns_yaml = []
        for col in dimension.columns:
            col_yaml = f"""      - name: {col.name}
        description: "{col.description}"
        tests:"""
            if col.is_key:
                col_yaml += """
          - unique
          - not_null"""
            columns_yaml.append(col_yaml)

        return f"""version: 2

models:
  - name: {dimension.name}
    description: "{dimension.description}"
    columns:
{chr(10).join(columns_yaml)}"""
