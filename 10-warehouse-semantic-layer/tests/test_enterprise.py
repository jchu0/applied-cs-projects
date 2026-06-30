"""Tests for enterprise governance and operations features."""

import pytest
from datetime import datetime, timedelta
from typing import List, Dict, Any

from semantic_layer.models import CalculationMethod, TimeGrain, MetricDefinition
from semantic_layer.query_engine import MetricCatalog
from semantic_layer.enterprise import (
    AccessLevel,
    FreshnessStatus,
    ColumnLineage,
    LineageGraph,
    LineageTracker,
    FreshnessSLA,
    FreshnessCheck,
    FreshnessMonitor,
    AccessPolicy,
    AccessController,
    CICDConfig,
    CICDIntegration,
    DocumentationGenerator,
    SemanticLayerGovernance,
)


class TestAccessLevel:
    """Test AccessLevel enum."""

    def test_access_levels_exist(self):
        """Test all access levels are defined."""
        assert AccessLevel.PUBLIC.value == "public"
        assert AccessLevel.INTERNAL.value == "internal"
        assert AccessLevel.RESTRICTED.value == "restricted"
        assert AccessLevel.CONFIDENTIAL.value == "confidential"

    def test_access_level_from_string(self):
        """Test creating access level from string."""
        assert AccessLevel("public") == AccessLevel.PUBLIC
        assert AccessLevel("confidential") == AccessLevel.CONFIDENTIAL


class TestFreshnessStatus:
    """Test FreshnessStatus enum."""

    def test_freshness_statuses_exist(self):
        """Test all freshness statuses are defined."""
        assert FreshnessStatus.FRESH.value == "fresh"
        assert FreshnessStatus.STALE.value == "stale"
        assert FreshnessStatus.CRITICAL.value == "critical"
        assert FreshnessStatus.UNKNOWN.value == "unknown"


class TestColumnLineage:
    """Test ColumnLineage dataclass."""

    def test_create_basic_lineage(self):
        """Test creating basic column lineage."""
        lineage = ColumnLineage(
            source_model="stg_orders",
            source_column="amount",
            target_model="fct_orders",
            target_column="order_total",
        )

        assert lineage.source_model == "stg_orders"
        assert lineage.source_column == "amount"
        assert lineage.target_model == "fct_orders"
        assert lineage.target_column == "order_total"
        assert lineage.transformation == ""
        assert lineage.confidence == 1.0

    def test_lineage_with_transformation(self):
        """Test lineage with transformation details."""
        lineage = ColumnLineage(
            source_model="stg_orders",
            source_column="amount_cents",
            target_model="fct_orders",
            target_column="amount_dollars",
            transformation="amount_cents / 100",
            confidence=0.95,
        )

        assert lineage.transformation == "amount_cents / 100"
        assert lineage.confidence == 0.95


class TestLineageGraph:
    """Test LineageGraph dataclass."""

    def test_add_column(self):
        """Test adding a column to the graph."""
        graph = LineageGraph()
        graph.add_column("orders", "order_id")

        assert "orders" in graph.nodes
        assert "order_id" in graph.nodes["orders"]

    def test_add_multiple_columns(self):
        """Test adding multiple columns to the same model."""
        graph = LineageGraph()
        graph.add_column("orders", "order_id")
        graph.add_column("orders", "customer_id")
        graph.add_column("orders", "amount")

        assert len(graph.nodes["orders"]) == 3
        assert "order_id" in graph.nodes["orders"]
        assert "customer_id" in graph.nodes["orders"]
        assert "amount" in graph.nodes["orders"]

    def test_add_edge(self):
        """Test adding a lineage edge."""
        graph = LineageGraph()
        lineage = ColumnLineage(
            source_model="stg_orders",
            source_column="amount",
            target_model="fct_orders",
            target_column="order_total",
        )

        graph.add_edge(lineage)

        assert len(graph.edges) == 1
        assert "stg_orders" in graph.nodes
        assert "fct_orders" in graph.nodes

    def test_get_upstream(self):
        """Test getting upstream lineage."""
        graph = LineageGraph()

        # Add lineage: stg_orders.amount -> fct_orders.order_total
        graph.add_edge(ColumnLineage(
            source_model="stg_orders",
            source_column="amount",
            target_model="fct_orders",
            target_column="order_total",
        ))

        upstream = graph.get_upstream("fct_orders", "order_total")

        assert len(upstream) == 1
        assert upstream[0].source_model == "stg_orders"
        assert upstream[0].source_column == "amount"

    def test_get_downstream(self):
        """Test getting downstream lineage."""
        graph = LineageGraph()

        # Add lineage: fct_orders.order_total -> mart_revenue.total_revenue
        graph.add_edge(ColumnLineage(
            source_model="fct_orders",
            source_column="order_total",
            target_model="mart_revenue",
            target_column="total_revenue",
        ))

        downstream = graph.get_downstream("fct_orders", "order_total")

        assert len(downstream) == 1
        assert downstream[0].target_model == "mart_revenue"
        assert downstream[0].target_column == "total_revenue"


class TestLineageTracker:
    """Test LineageTracker class."""

    def test_track_transformation(self):
        """Test tracking a column transformation."""
        tracker = LineageTracker()

        tracker.track_transformation(
            source_model="stg_orders",
            source_columns=["amount_cents"],
            target_model="fct_orders",
            target_column="amount_dollars",
            transformation="amount_cents / 100",
        )

        graph = tracker.get_lineage_graph()
        assert len(graph.edges) == 1

    def test_track_multiple_source_columns(self):
        """Test tracking transformation with multiple source columns."""
        tracker = LineageTracker()

        tracker.track_transformation(
            source_model="stg_orders",
            source_columns=["quantity", "unit_price"],
            target_model="fct_orders",
            target_column="total_amount",
            transformation="quantity * unit_price",
        )

        graph = tracker.get_lineage_graph()
        assert len(graph.edges) == 2

    def test_track_metric_lineage(self):
        """Test tracking metric lineage to source columns."""
        tracker = LineageTracker()

        tracker.track_metric_lineage(
            metric_name="total_revenue",
            source_columns=["fct_orders.order_total", "dim_dates.date_key"],
        )

        deps = tracker.get_metric_dependencies("total_revenue")
        assert len(deps) == 2
        assert "fct_orders.order_total" in deps

    def test_get_impact_analysis(self):
        """Test impact analysis for column changes."""
        tracker = LineageTracker()

        # Build lineage chain
        tracker.track_transformation(
            source_model="raw_orders",
            source_columns=["amount"],
            target_model="stg_orders",
            target_column="amount",
        )

        tracker.track_transformation(
            source_model="stg_orders",
            source_columns=["amount"],
            target_model="fct_orders",
            target_column="order_total",
        )

        tracker.track_metric_lineage(
            metric_name="total_revenue",
            source_columns=["fct_orders.order_total"],
        )

        impact = tracker.get_impact_analysis("raw_orders", "amount")

        assert "stg_orders" in impact["models"]
        assert "fct_orders" in impact["models"]
        assert "total_revenue" in impact["metrics"]


class TestFreshnessSLA:
    """Test FreshnessSLA dataclass."""

    def test_create_sla(self):
        """Test creating a freshness SLA."""
        sla = FreshnessSLA(
            metric_name="daily_revenue",
            max_delay_minutes=60,
            warning_threshold_minutes=30,
            owner="finance-team",
        )

        assert sla.metric_name == "daily_revenue"
        assert sla.max_delay_minutes == 60
        assert sla.warning_threshold_minutes == 30
        assert sla.owner == "finance-team"
        assert sla.escalation_contacts == []

    def test_sla_with_escalation(self):
        """Test SLA with escalation contacts."""
        sla = FreshnessSLA(
            metric_name="critical_metric",
            max_delay_minutes=15,
            warning_threshold_minutes=5,
            owner="ops-team",
            escalation_contacts=["oncall@company.com", "manager@company.com"],
        )

        assert len(sla.escalation_contacts) == 2


class TestFreshnessMonitor:
    """Test FreshnessMonitor class."""

    @pytest.fixture
    def catalog(self):
        """Create a metric catalog."""
        catalog = MetricCatalog()
        metric = MetricDefinition(
            name="test_metric",
            label="Test Metric",
            description="Test metric",
            model="orders",
            calculation_method=CalculationMethod.SUM,
            expression="amount",
            timestamp="date",
            time_grains=[TimeGrain.DAY],
        )
        catalog.add_metric(metric)
        return catalog

    @pytest.fixture
    def monitor(self, catalog):
        """Create a freshness monitor."""
        return FreshnessMonitor(catalog)

    def test_set_sla(self, monitor):
        """Test setting an SLA."""
        sla = FreshnessSLA(
            metric_name="test_metric",
            max_delay_minutes=60,
            warning_threshold_minutes=30,
            owner="test-team",
        )

        monitor.set_sla(sla)

        # Check freshness (should be unknown since no update recorded)
        check = monitor.check_freshness("test_metric")
        assert check.status == FreshnessStatus.UNKNOWN

    def test_record_update(self, monitor):
        """Test recording a metric update."""
        now = datetime.now()
        monitor.record_update("test_metric", now)

        check = monitor.check_freshness("test_metric")
        assert check.last_updated == now

    def test_check_freshness_fresh(self, monitor):
        """Test freshness check when metric is fresh."""
        sla = FreshnessSLA(
            metric_name="test_metric",
            max_delay_minutes=60,
            warning_threshold_minutes=30,
            owner="test-team",
        )
        monitor.set_sla(sla)
        monitor.record_update("test_metric", datetime.now())

        check = monitor.check_freshness("test_metric")

        assert check.status == FreshnessStatus.FRESH
        assert check.sla_violated is False

    def test_check_freshness_warning(self, monitor):
        """Test freshness check when metric is stale (warning)."""
        sla = FreshnessSLA(
            metric_name="test_metric",
            max_delay_minutes=60,
            warning_threshold_minutes=30,
            owner="test-team",
        )
        monitor.set_sla(sla)

        # Update was 45 minutes ago (past warning but not error threshold)
        update_time = datetime.now() - timedelta(minutes=45)
        monitor.record_update("test_metric", update_time)

        check = monitor.check_freshness("test_metric")

        assert check.status == FreshnessStatus.STALE
        assert check.sla_violated is False
        assert "WARNING" in check.message

    def test_check_freshness_critical(self, monitor):
        """Test freshness check when SLA is violated."""
        sla = FreshnessSLA(
            metric_name="test_metric",
            max_delay_minutes=60,
            warning_threshold_minutes=30,
            owner="test-team",
        )
        monitor.set_sla(sla)

        # Update was 90 minutes ago (past SLA)
        update_time = datetime.now() - timedelta(minutes=90)
        monitor.record_update("test_metric", update_time)

        check = monitor.check_freshness("test_metric")

        assert check.status == FreshnessStatus.CRITICAL
        assert check.sla_violated is True
        assert "VIOLATED" in check.message

    def test_check_freshness_no_sla(self, monitor):
        """Test freshness check without SLA."""
        monitor.record_update("test_metric", datetime.now())

        check = monitor.check_freshness("test_metric")

        assert check.sla_violated is False
        assert "no SLA defined" in check.message

    def test_check_all(self, monitor):
        """Test checking all metrics with SLAs."""
        sla1 = FreshnessSLA(
            metric_name="metric_1",
            max_delay_minutes=60,
            warning_threshold_minutes=30,
            owner="team-a",
        )
        sla2 = FreshnessSLA(
            metric_name="metric_2",
            max_delay_minutes=120,
            warning_threshold_minutes=60,
            owner="team-b",
        )

        monitor.set_sla(sla1)
        monitor.set_sla(sla2)

        results = monitor.check_all()

        assert len(results) == 2


class TestAccessPolicy:
    """Test AccessPolicy dataclass."""

    def test_create_basic_policy(self):
        """Test creating a basic access policy."""
        policy = AccessPolicy(
            name="revenue_metric",
            level=AccessLevel.INTERNAL,
        )

        assert policy.name == "revenue_metric"
        assert policy.level == AccessLevel.INTERNAL
        assert policy.allowed_roles == []
        assert policy.denied_roles == []
        assert policy.row_level_filter is None
        assert policy.column_mask is None

    def test_policy_with_roles(self):
        """Test policy with role restrictions."""
        policy = AccessPolicy(
            name="salary_data",
            level=AccessLevel.RESTRICTED,
            allowed_roles=["hr-admin", "payroll"],
            denied_roles=["intern"],
        )

        assert "hr-admin" in policy.allowed_roles
        assert "intern" in policy.denied_roles

    def test_policy_with_row_level_filter(self):
        """Test policy with row-level filtering."""
        policy = AccessPolicy(
            name="department_metrics",
            level=AccessLevel.INTERNAL,
            row_level_filter="department_id = {user_department}",
        )

        assert policy.row_level_filter is not None
        assert "department_id" in policy.row_level_filter

    def test_policy_with_column_mask(self):
        """Test policy with column masking."""
        policy = AccessPolicy(
            name="customer_pii",
            level=AccessLevel.CONFIDENTIAL,
            column_mask="XXXX-XXXX-{last_4}",
        )

        assert policy.column_mask is not None


class TestAccessController:
    """Test AccessController class."""

    @pytest.fixture
    def controller(self):
        """Create an access controller."""
        return AccessController()

    def test_set_policy(self, controller):
        """Test setting an access policy."""
        policy = AccessPolicy(
            name="revenue",
            level=AccessLevel.INTERNAL,
        )

        controller.set_policy("revenue_metric", policy)

        # Check access should work
        access = controller.check_access("user1", "revenue_metric")
        # No roles = internal access denied
        assert access is False

    def test_set_user_roles(self, controller):
        """Test setting user roles."""
        controller.set_user_roles("user1", ["analyst", "finance"])

        # Now check with a policy
        policy = AccessPolicy(
            name="revenue",
            level=AccessLevel.INTERNAL,
        )
        controller.set_policy("revenue_metric", policy)

        access = controller.check_access("user1", "revenue_metric")
        assert access is True  # Has roles, so internal access granted

    def test_check_access_public(self, controller):
        """Test access check for public resources."""
        policy = AccessPolicy(
            name="public_metric",
            level=AccessLevel.PUBLIC,
        )
        controller.set_policy("public_metric", policy)

        # Public resources are accessible to everyone
        assert controller.check_access("anonymous", "public_metric") is True

    def test_check_access_with_allowed_roles(self, controller):
        """Test access check with allowed roles."""
        policy = AccessPolicy(
            name="finance_metric",
            level=AccessLevel.RESTRICTED,
            allowed_roles=["finance", "executive"],
        )
        controller.set_policy("finance_metric", policy)

        controller.set_user_roles("finance_user", ["finance"])
        controller.set_user_roles("marketing_user", ["marketing"])

        assert controller.check_access("finance_user", "finance_metric") is True
        assert controller.check_access("marketing_user", "finance_metric") is False

    def test_check_access_with_denied_roles(self, controller):
        """Test access check with denied roles."""
        policy = AccessPolicy(
            name="internal_metric",
            level=AccessLevel.INTERNAL,
            denied_roles=["contractor"],
        )
        controller.set_policy("internal_metric", policy)

        controller.set_user_roles("employee", ["employee"])
        controller.set_user_roles("contractor", ["contractor"])

        assert controller.check_access("employee", "internal_metric") is True
        assert controller.check_access("contractor", "internal_metric") is False

    def test_check_access_no_policy(self, controller):
        """Test access check when no policy exists."""
        # No policy = public access
        assert controller.check_access("anyone", "unprotected_metric") is True

    def test_get_row_filter(self, controller):
        """Test getting row-level filter."""
        policy = AccessPolicy(
            name="department_data",
            level=AccessLevel.INTERNAL,
            row_level_filter="region = '{user_role}'",
        )
        controller.set_policy("department_data", policy)
        controller.set_user_roles("user1", ["NA"])

        filter_sql = controller.get_row_filter("user1", "department_data")

        assert filter_sql == "region = 'NA'"

    def test_get_column_mask(self, controller):
        """Test getting column mask."""
        policy = AccessPolicy(
            name="customer_ssn",
            level=AccessLevel.CONFIDENTIAL,
            column_mask="XXX-XX-{last_4}",
        )
        controller.set_policy("customer_ssn", policy)

        mask = controller.get_column_mask("user1", "customer_ssn")

        assert mask == "XXX-XX-{last_4}"


class TestCICDConfig:
    """Test CICDConfig dataclass."""

    def test_default_config(self):
        """Test default CI/CD config."""
        config = CICDConfig(
            project_name="analytics",
            repository_url="https://github.com/org/repo",
        )

        assert config.project_name == "analytics"
        assert config.repository_url == "https://github.com/org/repo"
        assert config.branch == "main"
        assert config.dbt_project_dir == "."
        assert config.run_tests is True
        assert config.run_freshness is True
        assert config.fail_on_test_failure is True
        assert config.slack_webhook is None

    def test_custom_config(self):
        """Test custom CI/CD config."""
        config = CICDConfig(
            project_name="my_project",
            repository_url="https://github.com/org/repo",
            branch="develop",
            dbt_project_dir="analytics",
            run_tests=True,
            run_freshness=False,
            fail_on_test_failure=False,
            slack_webhook="https://hooks.slack.com/xxx",
        )

        assert config.branch == "develop"
        assert config.dbt_project_dir == "analytics"
        assert config.run_freshness is False
        assert config.slack_webhook is not None


class TestCICDIntegration:
    """Test CICDIntegration class."""

    @pytest.fixture
    def cicd(self):
        """Create a CI/CD integration."""
        config = CICDConfig(
            project_name="analytics",
            repository_url="https://github.com/org/repo",
            branch="main",
        )
        return CICDIntegration(config)

    def test_generate_github_actions(self, cicd):
        """Test generating GitHub Actions workflow."""
        workflow = cicd.generate_github_actions()

        assert "name: Semantic Layer CI/CD" in workflow
        assert "push:" in workflow
        assert "branches: [main]" in workflow
        assert "dbt deps" in workflow
        assert "dbt compile" in workflow
        assert "dbt test" in workflow
        assert "dbt source freshness" in workflow

    def test_generate_pre_commit_hooks(self, cicd):
        """Test generating pre-commit hooks."""
        hooks = cicd.generate_pre_commit_hooks()

        assert "repos:" in hooks
        assert "validate-metrics" in hooks
        assert "check-metric-docs" in hooks
        assert "lint-sql" in hooks
        assert "sqlfluff" in hooks

    def test_validate_changes(self, cicd):
        """Test validating changes."""
        catalog = MetricCatalog()

        # Add metric without owner
        metric = MetricDefinition(
            name="test_metric",
            label="Test",
            description="",  # No description
            model="test",
            calculation_method=CalculationMethod.SUM,
            expression="amount",
            timestamp="date",
            time_grains=[TimeGrain.DAY],
            meta={},  # No owner
        )
        catalog.add_metric(metric)

        changed_files = [
            "semantic_layer/metrics/revenue.yml",
            "models/fct_orders.sql",
        ]

        results = cicd.validate_changes(changed_files, catalog)

        assert results["valid"] is True
        assert "semantic_layer/metrics/revenue.yml" in results["metrics_changed"]
        assert "models/fct_orders.sql" in results["models_changed"]
        assert len(results["warnings"]) >= 1  # Missing description or owner


class TestDocumentationGenerator:
    """Test DocumentationGenerator class."""

    @pytest.fixture
    def catalog(self):
        """Create a catalog with metrics."""
        catalog = MetricCatalog()

        metrics = [
            MetricDefinition(
                name="revenue",
                label="Total Revenue",
                description="Sum of all revenue",
                model="orders",
                calculation_method=CalculationMethod.SUM,
                expression="amount",
                timestamp="date",
                time_grains=[TimeGrain.DAY, TimeGrain.MONTH],
                dimensions=["region", "product"],
                meta={"category": "Finance", "owner": "finance-team"},
            ),
            MetricDefinition(
                name="users",
                label="Active Users",
                description="Count of unique users",
                model="users",
                calculation_method=CalculationMethod.COUNT_DISTINCT,
                expression="user_id",
                timestamp="date",
                time_grains=[TimeGrain.DAY],
                dimensions=["platform"],
                meta={"category": "Product", "owner": "product-team"},
            ),
        ]

        for metric in metrics:
            catalog.add_metric(metric)

        return catalog

    @pytest.fixture
    def doc_gen(self, catalog):
        """Create a documentation generator."""
        return DocumentationGenerator(catalog)

    def test_generate_metric_docs(self, doc_gen):
        """Test generating metric documentation."""
        docs = doc_gen.generate_metric_docs()

        assert "# Metric Catalog" in docs
        assert "## Finance" in docs
        assert "## Product" in docs
        assert "### Total Revenue" in docs
        assert "### Active Users" in docs
        assert "`revenue`" in docs
        assert "Sum of all revenue" in docs
        assert "**Dimensions:**" in docs
        assert "**Time Grains:**" in docs

    def test_generate_data_dictionary(self, doc_gen):
        """Test generating data dictionary."""
        docs = doc_gen.generate_data_dictionary()

        assert "# Data Dictionary" in docs
        assert "## Fact Tables" in docs
        assert "## Dimension Tables" in docs

    def test_generate_lineage_diagram(self, doc_gen):
        """Test generating lineage diagram."""
        tracker = LineageTracker()

        tracker.track_transformation(
            source_model="stg_orders",
            source_columns=["amount"],
            target_model="fct_orders",
            target_column="order_total",
        )

        diagram = doc_gen.generate_lineage_diagram(tracker)

        assert "```mermaid" in diagram
        assert "graph LR" in diagram
        assert "stg_orders_amount" in diagram
        assert "fct_orders_order_total" in diagram


class TestSemanticLayerGovernance:
    """Test SemanticLayerGovernance class."""

    @pytest.fixture
    def catalog(self):
        """Create a catalog with metrics."""
        catalog = MetricCatalog()

        metrics = [
            MetricDefinition(
                name="revenue",
                label="Revenue",
                description="Total revenue",
                model="orders",
                calculation_method=CalculationMethod.SUM,
                expression="amount",
                timestamp="date",
                time_grains=[TimeGrain.DAY],
                meta={"owner": "finance"},
            ),
            MetricDefinition(
                name="users",
                label="Users",
                description="",  # No description
                model="users",
                calculation_method=CalculationMethod.COUNT,
                expression="user_id",
                timestamp="date",
                time_grains=[TimeGrain.DAY],
                meta={},  # No owner
            ),
        ]

        for metric in metrics:
            catalog.add_metric(metric)

        return catalog

    @pytest.fixture
    def governance(self, catalog):
        """Create a governance instance."""
        return SemanticLayerGovernance(catalog)

    def test_governance_components(self, governance):
        """Test governance has all required components."""
        assert governance.catalog is not None
        assert governance.lineage is not None
        assert governance.freshness is not None
        assert governance.access is not None
        assert governance.docs is not None

    def test_get_governance_report(self, governance):
        """Test generating governance report."""
        report = governance.get_governance_report()

        assert report["total_metrics"] == 2
        assert report["metrics_with_owners"] == 1
        assert report["metrics_with_docs"] == 1
        assert report["coverage"]["owner_coverage"] == 50.0
        assert report["coverage"]["doc_coverage"] == 50.0

    def test_governance_report_empty_catalog(self):
        """Test governance report with empty catalog."""
        catalog = MetricCatalog()
        governance = SemanticLayerGovernance(catalog)

        report = governance.get_governance_report()

        assert report["total_metrics"] == 0
        assert report["coverage"]["owner_coverage"] == 0.0
        assert report["coverage"]["doc_coverage"] == 0.0
