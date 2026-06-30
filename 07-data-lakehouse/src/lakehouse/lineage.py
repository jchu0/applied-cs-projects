"""Data lineage tracking for the lakehouse.

This module provides:
- Column-level lineage tracking
- Table dependency graphs
- Impact analysis queries
- Lineage visualization data structures
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from .config import Layer, TableLineage


class LineageType(Enum):
    """Type of lineage relationship."""
    DIRECT = "direct"  # Column A directly maps to column B
    DERIVED = "derived"  # Column B is computed from column A (and possibly others)
    FILTERED = "filtered"  # Column used in filter condition
    JOINED = "joined"  # Column used in join condition
    AGGREGATED = "aggregated"  # Column is result of aggregation


@dataclass
class ColumnLineage:
    """Column-level lineage information."""

    # Source column
    source_table: str
    source_column: str

    # Target column
    target_table: str
    target_column: str

    # Relationship type
    lineage_type: LineageType

    # Transformation details
    transformation: Optional[str] = None  # e.g., "UPPER(name)", "SUM(amount)"

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    created_by: Optional[str] = None


@dataclass
class TableNode:
    """Represents a table in the lineage graph."""

    name: str
    layer: Layer
    path: Optional[str] = None
    description: Optional[str] = None
    columns: List[str] = field(default_factory=list)
    partition_columns: List[str] = field(default_factory=list)

    # Metadata
    owner: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    last_updated: Optional[datetime] = None


@dataclass
class LineageEdge:
    """Represents a dependency edge between tables."""

    source_table: str
    target_table: str
    column_lineages: List[ColumnLineage] = field(default_factory=list)

    # Transformation metadata
    transformation_type: str = "unknown"  # etl, aggregation, join, filter
    query: Optional[str] = None
    job_name: Optional[str] = None
    schedule: Optional[str] = None


class LineageGraph:
    """Graph structure for data lineage."""

    def __init__(self):
        self.nodes: Dict[str, TableNode] = {}
        self.edges: List[LineageEdge] = []
        self._adjacency: Dict[str, Set[str]] = {}  # source -> targets
        self._reverse_adjacency: Dict[str, Set[str]] = {}  # target -> sources

    def add_table(self, node: TableNode) -> None:
        """Add a table node to the graph."""
        self.nodes[node.name] = node
        if node.name not in self._adjacency:
            self._adjacency[node.name] = set()
        if node.name not in self._reverse_adjacency:
            self._reverse_adjacency[node.name] = set()

    def add_edge(self, edge: LineageEdge) -> None:
        """Add a lineage edge between tables."""
        self.edges.append(edge)

        # Update adjacency lists
        if edge.source_table not in self._adjacency:
            self._adjacency[edge.source_table] = set()
        self._adjacency[edge.source_table].add(edge.target_table)

        if edge.target_table not in self._reverse_adjacency:
            self._reverse_adjacency[edge.target_table] = set()
        self._reverse_adjacency[edge.target_table].add(edge.source_table)

    def get_upstream(self, table_name: str, depth: int = -1) -> Set[str]:
        """Get all upstream tables (sources) for a table.

        Args:
            table_name: The table to analyze
            depth: Maximum depth to traverse (-1 for unlimited)

        Returns:
            Set of upstream table names
        """
        upstream = set()
        visited = set()

        def traverse(name: str, current_depth: int):
            if name in visited:
                return
            if depth >= 0 and current_depth > depth:
                return

            visited.add(name)
            sources = self._reverse_adjacency.get(name, set())

            for source in sources:
                upstream.add(source)
                traverse(source, current_depth + 1)

        traverse(table_name, 0)
        return upstream

    def get_downstream(self, table_name: str, depth: int = -1) -> Set[str]:
        """Get all downstream tables (targets) for a table.

        Args:
            table_name: The table to analyze
            depth: Maximum depth to traverse (-1 for unlimited)

        Returns:
            Set of downstream table names
        """
        downstream = set()
        visited = set()

        def traverse(name: str, current_depth: int):
            if name in visited:
                return
            if depth >= 0 and current_depth > depth:
                return

            visited.add(name)
            targets = self._adjacency.get(name, set())

            for target in targets:
                downstream.add(target)
                traverse(target, current_depth + 1)

        traverse(table_name, 0)
        return downstream

    def get_column_lineage(
        self,
        table_name: str,
        column_name: str
    ) -> List[ColumnLineage]:
        """Get column-level lineage for a specific column.

        Returns lineage for both upstream (sources) and downstream (targets).
        """
        lineages = []

        for edge in self.edges:
            for col_lineage in edge.column_lineages:
                # Check if this column is the target
                if (col_lineage.target_table == table_name and
                    col_lineage.target_column == column_name):
                    lineages.append(col_lineage)

                # Check if this column is the source
                if (col_lineage.source_table == table_name and
                    col_lineage.source_column == column_name):
                    lineages.append(col_lineage)

        return lineages

    def impact_analysis(self, table_name: str, column_name: Optional[str] = None) -> Dict[str, Any]:
        """Analyze the impact of changes to a table or column.

        Args:
            table_name: The table being modified
            column_name: Optional specific column being modified

        Returns:
            Impact analysis report
        """
        downstream = self.get_downstream(table_name)

        impact = {
            "source_table": table_name,
            "source_column": column_name,
            "directly_affected_tables": list(self._adjacency.get(table_name, set())),
            "all_affected_tables": list(downstream),
            "affected_count": len(downstream),
            "affected_by_layer": self._group_by_layer(downstream),
            "column_impacts": []
        }

        if column_name:
            # Find all columns that depend on this column
            affected_columns = []
            for edge in self.edges:
                for col_lineage in edge.column_lineages:
                    if (col_lineage.source_table == table_name and
                        col_lineage.source_column == column_name):
                        affected_columns.append({
                            "table": col_lineage.target_table,
                            "column": col_lineage.target_column,
                            "lineage_type": col_lineage.lineage_type.value,
                            "transformation": col_lineage.transformation
                        })
            impact["column_impacts"] = affected_columns

        return impact

    def _group_by_layer(self, tables: Set[str]) -> Dict[str, List[str]]:
        """Group tables by their layer."""
        grouped = {layer.value: [] for layer in Layer}

        for table_name in tables:
            if table_name in self.nodes:
                layer = self.nodes[table_name].layer.value
                grouped[layer].append(table_name)

        return grouped

    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary for serialization."""
        return {
            "nodes": [
                {
                    "name": node.name,
                    "layer": node.layer.value,
                    "path": node.path,
                    "description": node.description,
                    "columns": node.columns,
                    "partition_columns": node.partition_columns,
                    "owner": node.owner,
                    "tags": node.tags,
                }
                for node in self.nodes.values()
            ],
            "edges": [
                {
                    "source": edge.source_table,
                    "target": edge.target_table,
                    "transformation_type": edge.transformation_type,
                    "query": edge.query,
                    "job_name": edge.job_name,
                    "column_lineages": [
                        {
                            "source_table": cl.source_table,
                            "source_column": cl.source_column,
                            "target_table": cl.target_table,
                            "target_column": cl.target_column,
                            "lineage_type": cl.lineage_type.value,
                            "transformation": cl.transformation,
                        }
                        for cl in edge.column_lineages
                    ]
                }
                for edge in self.edges
            ]
        }


class LineageTracker:
    """Main class for tracking data lineage."""

    def __init__(self, storage_path: Optional[str] = None):
        """Initialize the lineage tracker.

        Args:
            storage_path: Optional path to persist lineage data
        """
        self.graph = LineageGraph()
        self.storage_path = storage_path
        self._table_lineages: Dict[str, TableLineage] = {}

    def register_table(
        self,
        name: str,
        layer: Layer,
        columns: List[str],
        path: Optional[str] = None,
        partition_columns: Optional[List[str]] = None,
        description: Optional[str] = None,
        owner: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> None:
        """Register a table in the lineage graph.

        Args:
            name: Table name
            layer: Medallion layer (bronze/silver/gold)
            columns: List of column names
            path: Storage path
            partition_columns: Columns used for partitioning
            description: Table description
            owner: Table owner
            tags: Table tags
        """
        node = TableNode(
            name=name,
            layer=layer,
            columns=columns,
            path=path,
            partition_columns=partition_columns or [],
            description=description,
            owner=owner,
            tags=tags or [],
            last_updated=datetime.now()
        )
        self.graph.add_table(node)

    def add_transformation(
        self,
        source_tables: List[str],
        target_table: str,
        column_mappings: List[Tuple[str, str, str, str, LineageType, Optional[str]]],
        transformation_type: str = "etl",
        query: Optional[str] = None,
        job_name: Optional[str] = None,
        schedule: Optional[str] = None
    ) -> None:
        """Add a transformation that creates lineage.

        Args:
            source_tables: List of source table names
            target_table: Target table name
            column_mappings: List of tuples:
                (source_table, source_column, target_table, target_column, lineage_type, transformation)
            transformation_type: Type of transformation (etl, aggregation, join, etc.)
            query: The transformation query
            job_name: Name of the job performing the transformation
            schedule: Schedule for the transformation
        """
        # Create column lineages
        column_lineages = []
        for mapping in column_mappings:
            src_table, src_col, tgt_table, tgt_col, lineage_type, transform = mapping
            col_lineage = ColumnLineage(
                source_table=src_table,
                source_column=src_col,
                target_table=tgt_table,
                target_column=tgt_col,
                lineage_type=lineage_type,
                transformation=transform
            )
            column_lineages.append(col_lineage)

        # Create edges for each source table
        for source_table in source_tables:
            edge = LineageEdge(
                source_table=source_table,
                target_table=target_table,
                column_lineages=[
                    cl for cl in column_lineages
                    if cl.source_table == source_table
                ],
                transformation_type=transformation_type,
                query=query,
                job_name=job_name,
                schedule=schedule
            )
            self.graph.add_edge(edge)

        # Update table lineage
        upstream = list(set(source_tables))
        if target_table in self._table_lineages:
            existing = self._table_lineages[target_table]
            upstream = list(set(existing.upstream_tables + upstream))

        self._table_lineages[target_table] = TableLineage(
            table_name=target_table,
            upstream_tables=upstream,
            downstream_tables=[],
            transformation_query=query or "",
            refresh_schedule=schedule or ""
        )

        # Update downstream references
        for source in source_tables:
            if source in self._table_lineages:
                if target_table not in self._table_lineages[source].downstream_tables:
                    self._table_lineages[source].downstream_tables.append(target_table)

    def get_table_lineage(self, table_name: str) -> Optional[TableLineage]:
        """Get lineage information for a table."""
        if table_name in self._table_lineages:
            lineage = self._table_lineages[table_name]
            # Update upstream and downstream from graph
            lineage.upstream_tables = list(self.graph.get_upstream(table_name, depth=1))
            lineage.downstream_tables = list(self.graph.get_downstream(table_name, depth=1))
            return lineage
        return None

    def get_column_lineage(self, table_name: str, column_name: str) -> List[ColumnLineage]:
        """Get column-level lineage."""
        return self.graph.get_column_lineage(table_name, column_name)

    def impact_analysis(
        self,
        table_name: str,
        column_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform impact analysis for a table or column change."""
        return self.graph.impact_analysis(table_name, column_name)

    def get_full_upstream_path(
        self,
        table_name: str,
        column_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get the full path from sources to the given table/column.

        Returns a list of transformation steps.
        """
        path = []
        visited = set()

        def trace_upstream(current_table: str, current_column: Optional[str]):
            if current_table in visited:
                return
            visited.add(current_table)

            sources = self.graph._reverse_adjacency.get(current_table, set())

            for source_table in sources:
                # Find the edge
                for edge in self.graph.edges:
                    if edge.source_table == source_table and edge.target_table == current_table:
                        step = {
                            "source_table": source_table,
                            "target_table": current_table,
                            "transformation_type": edge.transformation_type,
                            "job_name": edge.job_name,
                        }

                        if current_column:
                            # Find relevant column lineage
                            for cl in edge.column_lineages:
                                if cl.target_column == current_column:
                                    step["source_column"] = cl.source_column
                                    step["target_column"] = cl.target_column
                                    step["transformation"] = cl.transformation

                        path.append(step)
                        trace_upstream(source_table, step.get("source_column"))

        trace_upstream(table_name, column_name)
        return list(reversed(path))

    def export_lineage(self) -> Dict[str, Any]:
        """Export full lineage data as dictionary."""
        return {
            "graph": self.graph.to_dict(),
            "table_lineages": {
                name: {
                    "table_name": tl.table_name,
                    "upstream_tables": tl.upstream_tables,
                    "downstream_tables": tl.downstream_tables,
                    "transformation_query": tl.transformation_query,
                    "refresh_schedule": tl.refresh_schedule,
                }
                for name, tl in self._table_lineages.items()
            },
            "exported_at": datetime.now().isoformat()
        }


def create_medallion_lineage(
    bronze_tables: List[str],
    silver_tables: List[str],
    gold_tables: List[str],
    bronze_to_silver_mappings: Dict[str, List[str]],  # silver_table -> [bronze_tables]
    silver_to_gold_mappings: Dict[str, List[str]]  # gold_table -> [silver_tables]
) -> LineageTracker:
    """Helper to create lineage for a standard medallion architecture.

    Args:
        bronze_tables: List of bronze table names
        silver_tables: List of silver table names
        gold_tables: List of gold table names
        bronze_to_silver_mappings: Map of silver tables to their bronze sources
        silver_to_gold_mappings: Map of gold tables to their silver sources

    Returns:
        Configured LineageTracker
    """
    tracker = LineageTracker()

    # Register bronze tables
    for table in bronze_tables:
        tracker.register_table(
            name=table,
            layer=Layer.BRONZE,
            columns=[],  # Would be filled from schema
            description=f"Raw data table: {table}"
        )

    # Register silver tables
    for table in silver_tables:
        tracker.register_table(
            name=table,
            layer=Layer.SILVER,
            columns=[],
            description=f"Cleaned data table: {table}"
        )

    # Register gold tables
    for table in gold_tables:
        tracker.register_table(
            name=table,
            layer=Layer.GOLD,
            columns=[],
            description=f"Business-ready table: {table}"
        )

    # Add bronze -> silver lineage
    for silver_table, bronze_sources in bronze_to_silver_mappings.items():
        tracker.add_transformation(
            source_tables=bronze_sources,
            target_table=silver_table,
            column_mappings=[],  # Would be filled with actual mappings
            transformation_type="etl",
            job_name=f"bronze_to_{silver_table}"
        )

    # Add silver -> gold lineage
    for gold_table, silver_sources in silver_to_gold_mappings.items():
        tracker.add_transformation(
            source_tables=silver_sources,
            target_table=gold_table,
            column_mappings=[],
            transformation_type="aggregation",
            job_name=f"silver_to_{gold_table}"
        )

    return tracker
