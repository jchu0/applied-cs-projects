"""Lineage graph for tracking data dependencies."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from observability.models import ImpactAnalysis, LineageInfo, TableMetadata

logger = logging.getLogger(__name__)


@dataclass
class LineageEdge:
    """Edge in the lineage graph."""

    source: str
    target: str
    transformation: Optional[str] = None
    column_mappings: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class LineageNode:
    """Node in the lineage graph."""

    table_id: str
    metadata: Optional[TableMetadata] = None
    upstream: Set[str] = field(default_factory=set)
    downstream: Set[str] = field(default_factory=set)


class LineageGraph:
    """Graph-based lineage tracking."""

    def __init__(self):
        self._nodes: Dict[str, LineageNode] = {}
        self._edges: List[LineageEdge] = []
        self._column_lineage: Dict[str, Dict[str, List[str]]] = {}  # col_id -> {up: [], down: []}

    def add_table(self, table) -> None:
        """Add or update a table node. Accepts either a table_id string or TableMetadata."""
        # Support both string table_id and TableMetadata object
        if isinstance(table, str):
            table_id = table
            metadata = None
        else:
            table_id = table.table_id
            metadata = table

        if table_id not in self._nodes:
            self._nodes[table_id] = LineageNode(table_id=table_id)

        if metadata:
            self._nodes[table_id].metadata = metadata
        logger.info(f"Added table {table_id} to lineage graph")

    def add_lineage(
        self,
        source_table: str,
        target_table: str,
        transformation: Optional[str] = None,
        column_mappings: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        """Add a lineage edge between tables."""
        # Ensure nodes exist
        if source_table not in self._nodes:
            self._nodes[source_table] = LineageNode(table_id=source_table)
        if target_table not in self._nodes:
            self._nodes[target_table] = LineageNode(table_id=target_table)

        # Add edge
        edge = LineageEdge(
            source=source_table,
            target=target_table,
            transformation=transformation,
            column_mappings=column_mappings or [],
        )
        self._edges.append(edge)

        # Update node relationships
        self._nodes[source_table].downstream.add(target_table)
        self._nodes[target_table].upstream.add(source_table)

        # Track column-level lineage
        if column_mappings:
            for mapping in column_mappings:
                source_col = f"{source_table}.{mapping['source']}"
                target_col = f"{target_table}.{mapping['target']}"

                if source_col not in self._column_lineage:
                    self._column_lineage[source_col] = {"upstream": [], "downstream": []}
                if target_col not in self._column_lineage:
                    self._column_lineage[target_col] = {"upstream": [], "downstream": []}

                self._column_lineage[source_col]["downstream"].append(target_col)
                self._column_lineage[target_col]["upstream"].append(source_col)

        logger.info(f"Added lineage: {source_table} -> {target_table}")

    def add_edge(
        self,
        source: str,
        target: str,
        transformation: Optional[str] = None,
        column_mappings: Optional[List[Dict[str, str]]] = None,
        column_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        """Add an edge between tables (alias for add_lineage)."""
        # Convert column_mapping dict to column_mappings list format
        if column_mapping and not column_mappings:
            column_mappings = [{"source": k, "target": v} for k, v in column_mapping.items()]
            # Also store raw column_mapping for direct lookup
            edge_key = f"{source}:{target}"
            if not hasattr(self, '_column_mappings'):
                self._column_mappings = {}
            self._column_mappings[edge_key] = column_mapping
        self.add_lineage(source, target, transformation, column_mappings)

    def get_column_mapping(self, source: str, target: str) -> Optional[Dict[str, str]]:
        """Get column mapping between two tables."""
        edge_key = f"{source}:{target}"
        if hasattr(self, '_column_mappings') and edge_key in self._column_mappings:
            return self._column_mappings[edge_key]
        return None

    def trace_column(self, table_id: str, column_name: str) -> List[str]:
        """Trace the lineage of a specific column back to its sources."""
        result = []
        col_id = f"{table_id}.{column_name}"

        if col_id in self._column_lineage:
            result.extend(self._column_lineage[col_id].get("upstream", []))

        # Also check column mappings
        if hasattr(self, '_column_mappings'):
            for edge_key, mapping in self._column_mappings.items():
                source_table, target_table = edge_key.split(":", 1)
                if target_table == table_id:
                    # Find source column that maps to this column
                    for src_col, tgt_col in mapping.items():
                        if tgt_col == column_name:
                            result.append(f"{source_table}.{src_col}")

        return result

    def remove_table(self, table_id: str) -> bool:
        """Remove a table from the graph."""
        if table_id not in self._nodes:
            return False

        node = self._nodes[table_id]

        # Remove edges involving this table
        self._edges = [e for e in self._edges if e.source != table_id and e.target != table_id]

        # Remove from other nodes' relationships
        for other_id in node.upstream:
            if other_id in self._nodes:
                self._nodes[other_id].downstream.discard(table_id)
        for other_id in node.downstream:
            if other_id in self._nodes:
                self._nodes[other_id].upstream.discard(table_id)

        # Remove node
        del self._nodes[table_id]
        logger.info(f"Removed table {table_id} from lineage graph")
        return True

    def remove_edge(self, source: str, target: str) -> bool:
        """Remove an edge between tables."""
        # Remove from edges list
        original_len = len(self._edges)
        self._edges = [e for e in self._edges if not (e.source == source and e.target == target)]

        if len(self._edges) == original_len:
            return False

        # Update node relationships
        if source in self._nodes:
            self._nodes[source].downstream.discard(target)
        if target in self._nodes:
            self._nodes[target].upstream.discard(source)

        logger.info(f"Removed edge: {source} -> {target}")
        return True

    def get_upstream(self, table_id: str) -> List[str]:
        """Get immediate upstream dependencies."""
        if table_id not in self._nodes:
            return []
        return list(self._nodes[table_id].upstream)

    def get_all_upstream(self, table_id: str, depth: int = 10) -> List[str]:
        """Get all upstream dependencies recursively."""
        if table_id not in self._nodes:
            return []

        result = []
        visited = set()
        queue = [table_id]

        while queue:
            current = queue.pop(0)

            if current in visited:
                continue
            visited.add(current)

            if current != table_id:
                result.append(current)

            node = self._nodes.get(current)
            if node:
                for upstream in node.upstream:
                    if upstream not in visited:
                        queue.append(upstream)

        return result

    def get_downstream(self, table_id: str) -> List[str]:
        """Get immediate downstream dependencies."""
        if table_id not in self._nodes:
            return []
        return list(self._nodes[table_id].downstream)

    def get_all_downstream(self, table_id: str, depth: int = 10) -> List[str]:
        """Get all downstream dependencies recursively."""
        if table_id not in self._nodes:
            return []

        result = []
        visited = set()
        queue = [table_id]

        while queue:
            current = queue.pop(0)

            if current in visited:
                continue
            visited.add(current)

            if current != table_id:
                result.append(current)

            node = self._nodes.get(current)
            if node:
                for downstream in node.downstream:
                    if downstream not in visited:
                        queue.append(downstream)

        return result

    def get_lineage_info(self, table_id: str) -> LineageInfo:
        """Get complete lineage info for a table."""
        upstream = self.get_all_upstream(table_id)
        downstream = self.get_all_downstream(table_id)

        return LineageInfo(
            table_id=table_id,
            upstream=upstream,
            downstream=downstream,
        )

    def _get_transformation(self, source: str, target: str) -> Optional[str]:
        """Get transformation between two tables."""
        for edge in self._edges:
            if edge.source == source and edge.target == target:
                return edge.transformation
        return None

    def get_column_lineage(self, column_id: str) -> Dict[str, List[str]]:
        """Get column-level lineage."""
        return self._column_lineage.get(column_id, {"upstream": [], "downstream": []})

    def get_impact_analysis(self, table_id: str) -> ImpactAnalysis:
        """Analyze impact of changes to a table."""
        downstream = self.get_all_downstream(table_id)

        affected_tables = []
        affected_pipelines: Set[str] = set()
        affected_dashboards: Set[str] = set()

        for downstream_id in downstream:
            node = self._nodes.get(downstream_id)
            if node and node.metadata:
                affected_tables.append(node.metadata)

        return ImpactAnalysis(
            source_table=table_id,
            affected_tables=affected_tables,
            affected_pipelines=list(affected_pipelines),
            affected_dashboards=list(affected_dashboards),
            total_downstream=len(downstream),
        )

    def get_all_tables(self) -> List[str]:
        """Get all table IDs in the graph."""
        return list(self._nodes.keys())

    def get_table_metadata(self, table_id: str) -> Optional[TableMetadata]:
        """Get metadata for a table."""
        node = self._nodes.get(table_id)
        return node.metadata if node else None

    def to_dict(self) -> Dict[str, Any]:
        """Export graph to dictionary format."""
        return {
            "nodes": [
                {
                    "id": node.table_id,
                    "upstream": list(node.upstream),
                    "downstream": list(node.downstream),
                }
                for node in self._nodes.values()
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "transformation": edge.transformation,
                }
                for edge in self._edges
            ],
        }
