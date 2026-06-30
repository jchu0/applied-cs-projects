"""Tests for lineage module."""

import pytest
from observability.lineage import LineageGraph
from observability.models import LineageInfo, ImpactAnalysis


class TestLineageGraph:
    """Tests for LineageGraph class."""

    @pytest.fixture
    def lineage_graph(self):
        """Create empty lineage graph."""
        return LineageGraph()

    @pytest.fixture
    def populated_graph(self):
        """Create lineage graph with sample data."""
        graph = LineageGraph()

        # Add nodes
        graph.add_table("raw.events")
        graph.add_table("raw.users")
        graph.add_table("staging.events_clean")
        graph.add_table("staging.users_clean")
        graph.add_table("analytics.user_events")
        graph.add_table("dashboard.daily_metrics")

        # Add edges (upstream -> downstream)
        graph.add_edge("raw.events", "staging.events_clean")
        graph.add_edge("raw.users", "staging.users_clean")
        graph.add_edge("staging.events_clean", "analytics.user_events")
        graph.add_edge("staging.users_clean", "analytics.user_events")
        graph.add_edge("analytics.user_events", "dashboard.daily_metrics")

        return graph

    def test_add_table(self, lineage_graph):
        """Test adding a table to the graph."""
        lineage_graph.add_table("test.table")
        assert "test.table" in lineage_graph.get_all_tables()

    def test_add_edge(self, lineage_graph):
        """Test adding an edge between tables."""
        lineage_graph.add_table("source")
        lineage_graph.add_table("target")
        lineage_graph.add_edge("source", "target")

        downstream = lineage_graph.get_downstream("source")
        assert "target" in downstream

    def test_get_upstream(self, populated_graph):
        """Test getting upstream tables."""
        upstream = populated_graph.get_upstream("analytics.user_events")
        assert "staging.events_clean" in upstream
        assert "staging.users_clean" in upstream

    def test_get_downstream(self, populated_graph):
        """Test getting downstream tables."""
        downstream = populated_graph.get_downstream("staging.events_clean")
        assert "analytics.user_events" in downstream

    def test_get_all_upstream(self, populated_graph):
        """Test getting all upstream tables recursively."""
        all_upstream = populated_graph.get_all_upstream("dashboard.daily_metrics")
        assert "analytics.user_events" in all_upstream
        assert "staging.events_clean" in all_upstream
        assert "raw.events" in all_upstream

    def test_get_all_downstream(self, populated_graph):
        """Test getting all downstream tables recursively."""
        all_downstream = populated_graph.get_all_downstream("raw.events")
        assert "staging.events_clean" in all_downstream
        assert "analytics.user_events" in all_downstream
        assert "dashboard.daily_metrics" in all_downstream

    def test_get_lineage_info(self, populated_graph):
        """Test getting lineage info for a table."""
        info = populated_graph.get_lineage_info("analytics.user_events")
        assert isinstance(info, LineageInfo)
        assert len(info.upstream) >= 2
        assert len(info.downstream) >= 1

    def test_impact_analysis(self, populated_graph):
        """Test impact analysis for a table."""
        impact = populated_graph.get_impact_analysis("raw.events")
        assert isinstance(impact, ImpactAnalysis)
        assert impact.total_downstream >= 3  # events_clean, user_events, daily_metrics

    def test_circular_dependency_handling(self, lineage_graph):
        """Test handling of circular dependencies."""
        lineage_graph.add_table("a")
        lineage_graph.add_table("b")
        lineage_graph.add_table("c")

        lineage_graph.add_edge("a", "b")
        lineage_graph.add_edge("b", "c")
        lineage_graph.add_edge("c", "a")  # Circular

        # Should not infinite loop
        all_downstream = lineage_graph.get_all_downstream("a")
        assert "b" in all_downstream
        assert "c" in all_downstream

    def test_remove_table(self, populated_graph):
        """Test removing a table from the graph."""
        populated_graph.remove_table("staging.events_clean")
        assert "staging.events_clean" not in populated_graph.get_all_tables()

    def test_remove_edge(self, populated_graph):
        """Test removing an edge from the graph."""
        populated_graph.remove_edge("raw.events", "staging.events_clean")
        downstream = populated_graph.get_downstream("raw.events")
        assert "staging.events_clean" not in downstream

    def test_get_all_tables(self, populated_graph):
        """Test getting all tables."""
        tables = populated_graph.get_all_tables()
        assert len(tables) == 6
        assert "raw.events" in tables
        assert "dashboard.daily_metrics" in tables


class TestLineageInfoModel:
    """Tests for LineageInfo model."""

    def test_lineage_info_creation(self):
        """Test creating lineage info."""
        info = LineageInfo(
            table_id="analytics.users",
            upstream=["raw.users", "raw.profiles"],
            downstream=["dashboard.users"],
            transformations=["join", "filter", "aggregate"]
        )
        assert info.table_id == "analytics.users"
        assert len(info.upstream) == 2
        assert len(info.downstream) == 1
        assert len(info.transformations) == 3

    def test_lineage_info_empty(self):
        """Test lineage info with no connections."""
        info = LineageInfo(table_id="isolated.table")
        assert info.upstream == []
        assert info.downstream == []
        assert info.transformations == []


class TestImpactAnalysisModel:
    """Tests for ImpactAnalysis model."""

    def test_impact_analysis_creation(self, sample_table_metadata):
        """Test creating impact analysis."""
        impact = ImpactAnalysis(
            source_table="raw.events",
            affected_tables=[sample_table_metadata],
            affected_pipelines=["etl_daily", "etl_hourly"],
            affected_dashboards=["user_dashboard", "revenue_dashboard"],
            total_downstream=5
        )
        assert impact.source_table == "raw.events"
        assert len(impact.affected_tables) == 1
        assert len(impact.affected_pipelines) == 2
        assert len(impact.affected_dashboards) == 2
        assert impact.total_downstream == 5


class TestColumnLineage:
    """Tests for column-level lineage."""

    @pytest.fixture
    def graph_with_columns(self):
        """Create graph with column-level lineage."""
        graph = LineageGraph()

        # Add tables
        graph.add_table("raw.events")
        graph.add_table("staging.events")

        # Add edge with column mapping
        graph.add_edge(
            "raw.events",
            "staging.events",
            column_mapping={
                "event_id": "id",
                "event_timestamp": "timestamp",
                "event_type": "type"
            }
        )

        return graph

    def test_column_mapping_stored(self, graph_with_columns):
        """Test that column mapping is stored."""
        mapping = graph_with_columns.get_column_mapping(
            "raw.events",
            "staging.events"
        )
        assert mapping is not None
        assert mapping["event_id"] == "id"

    def test_trace_column_lineage(self, graph_with_columns):
        """Test tracing column lineage."""
        sources = graph_with_columns.trace_column(
            "staging.events",
            "id"
        )
        # Should trace back to raw.events.event_id
        assert len(sources) >= 1
