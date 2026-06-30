"""Core data models for Semantic Layer."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class CalculationMethod(Enum):
    """Calculation methods for metrics."""

    SUM = "sum"
    COUNT = "count"
    COUNT_DISTINCT = "count_distinct"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    DERIVED = "derived"


class TimeGrain(Enum):
    """Time granularities for metrics."""

    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


@dataclass
class Dimension:
    """Dimension definition for slicing metrics."""

    name: str
    label: str
    description: str
    data_type: str
    model: str
    column: str
    is_time: bool = False
    hierarchy: Optional[List[str]] = None


@dataclass
class MetricDefinition:
    """Definition of a business metric."""

    name: str
    label: str
    description: str
    model: str
    calculation_method: CalculationMethod
    expression: str
    timestamp: str
    time_grains: List[TimeGrain]
    dimensions: List[str] = field(default_factory=list)
    filters: List[Dict[str, str]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricQuery:
    """Query specification for metrics."""

    metrics: List[str]
    dimensions: List[str]
    filters: List[Dict[str, Any]]
    time_grain: str
    start_date: str
    end_date: str
    limit: Optional[int] = None
    offset: Optional[int] = None


@dataclass
class QueryResult:
    """Result of a metric query."""

    data: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    sql: str
    row_count: int


@dataclass
class Column:
    """Column definition for a model."""

    name: str
    description: str
    data_type: str
    tests: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Model:
    """dbt model definition."""

    name: str
    description: str
    columns: List[Column]
    config: Dict[str, Any] = field(default_factory=dict)
    tests: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Source:
    """Source table definition."""

    name: str
    database: str
    schema: str
    description: str
    loader: str
    tables: List[Dict[str, Any]]
    freshness: Optional[Dict[str, Any]] = None


@dataclass
class Test:
    """Test definition."""

    name: str
    model: str
    column: Optional[str] = None
    test_type: str = "generic"
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Macro:
    """Macro definition."""

    name: str
    description: str
    arguments: List[Dict[str, str]] = field(default_factory=list)
    sql: str = ""


@dataclass
class DbtProject:
    """dbt project structure."""

    name: str
    version: str
    models: List[Model] = field(default_factory=list)
    sources: List[Source] = field(default_factory=list)
    macros: List[Macro] = field(default_factory=list)
    metrics: List[MetricDefinition] = field(default_factory=list)
    dimensions: List[Dimension] = field(default_factory=list)
