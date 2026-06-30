"""Tests for LLM-based issue triage."""

import json
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from observability.models import Anomaly, AnomalyType, DataHealthScore, TableMetadata, ColumnMetadata
from observability.llm_triage import (
    LLMProvider,
    LLMTriageEngine,
    MockLLMClient,
    OpenAIClient,
    AnthropicClient,
    TriageConfig,
    TriageResult,
    GeneratedRunbook,
    RunbookStep,
    TriageReportGenerator,
)


# Fixtures
@pytest.fixture
def sample_anomaly():
    """Create a sample anomaly for testing."""
    return Anomaly(
        anomaly_id="test-anomaly-001",
        table_id="db.schema.test_table",
        column_name="user_id",
        anomaly_type=AnomalyType.NULL_RATE,
        severity="warning",
        detected_at=datetime.now(),
        metric_value=0.35,
        expected_range=(0.0, 0.1),
        description="Null rate increased from 5% to 35%",
        context={"z_score": 4.2, "historical_mean": 0.05},
    )


@pytest.fixture
def sample_table_metadata():
    """Create sample table metadata."""
    return TableMetadata(
        table_id="db.schema.test_table",
        database="db",
        schema="schema",
        table_name="test_table",
        columns=[
            ColumnMetadata(name="user_id", data_type="bigint", nullable=True),
            ColumnMetadata(name="email", data_type="varchar", nullable=True),
        ],
        row_count=1000000,
        size_bytes=500000000,
        last_modified=datetime.now(),
    )


@pytest.fixture
def sample_health_score():
    """Create sample health score."""
    return DataHealthScore(
        table_id="db.schema.test_table",
        overall_score=75.0,
        freshness_score=90.0,
        volume_score=85.0,
        schema_score=100.0,
        quality_score=60.0,
        calculated_at=datetime.now(),
    )


@pytest.fixture
def triage_config():
    """Create triage configuration for mock client."""
    return TriageConfig(
        provider=LLMProvider.LOCAL,
        cache_enabled=False,
    )


@pytest.fixture
def triage_engine(triage_config):
    """Create triage engine with mock client."""
    return LLMTriageEngine(triage_config)


# Mock LLM Client Tests
class TestMockLLMClient:
    """Tests for MockLLMClient."""

    @pytest.mark.asyncio
    async def test_generate_root_cause(self):
        """Test mock client generates root cause analysis."""
        client = MockLLMClient()
        response = await client.generate("Analyze root cause for data issue")

        data = json.loads(response)
        assert "root_cause_analysis" in data
        assert "probable_causes" in data
        assert "recommended_actions" in data
        assert "confidence_score" in data

    @pytest.mark.asyncio
    async def test_generate_runbook(self):
        """Test mock client generates runbook."""
        client = MockLLMClient()
        response = await client.generate("Generate runbook for freshness issue")

        data = json.loads(response)
        assert "title" in data
        assert "steps" in data
        assert len(data["steps"]) > 0

    @pytest.mark.asyncio
    async def test_generate_default_response(self):
        """Test mock client handles unknown prompts."""
        client = MockLLMClient()
        response = await client.generate("Random unmatched prompt")

        data = json.loads(response)
        assert "response" in data


# Triage Engine Tests
class TestLLMTriageEngine:
    """Tests for LLMTriageEngine."""

    @pytest.mark.asyncio
    async def test_triage_anomaly_basic(self, triage_engine, sample_anomaly):
        """Test basic anomaly triage."""
        result = await triage_engine.triage_anomaly(sample_anomaly)

        assert isinstance(result, TriageResult)
        assert result.anomaly_id == sample_anomaly.anomaly_id
        assert len(result.probable_causes) > 0
        assert len(result.recommended_actions) > 0
        assert 0 <= result.confidence_score <= 1

    @pytest.mark.asyncio
    async def test_triage_with_metadata(
        self, triage_engine, sample_anomaly, sample_table_metadata
    ):
        """Test triage with table metadata context."""
        result = await triage_engine.triage_anomaly(
            sample_anomaly,
            table_metadata=sample_table_metadata,
        )

        assert isinstance(result, TriageResult)
        assert result.root_cause_analysis is not None

    @pytest.mark.asyncio
    async def test_triage_with_health_score(
        self, triage_engine, sample_anomaly, sample_health_score
    ):
        """Test triage with health score context."""
        result = await triage_engine.triage_anomaly(
            sample_anomaly,
            health_score=sample_health_score,
        )

        assert isinstance(result, TriageResult)

    @pytest.mark.asyncio
    async def test_triage_with_related_anomalies(self, triage_engine, sample_anomaly):
        """Test triage with related anomalies."""
        related = [
            Anomaly(
                anomaly_id="related-001",
                table_id="db.schema.upstream_table",
                column_name=None,
                anomaly_type=AnomalyType.FRESHNESS,
                severity="warning",
                detected_at=datetime.now(),
                metric_value=3600,
                expected_range=(0, 1800),
                description="Table is stale",
            )
        ]

        result = await triage_engine.triage_anomaly(
            sample_anomaly,
            related_anomalies=related,
        )

        assert isinstance(result, TriageResult)

    @pytest.mark.asyncio
    async def test_triage_caching(self, sample_anomaly):
        """Test triage result caching."""
        config = TriageConfig(
            provider=LLMProvider.LOCAL,
            cache_enabled=True,
            cache_ttl_seconds=3600,
        )
        engine = LLMTriageEngine(config)

        # First call
        result1 = await engine.triage_anomaly(sample_anomaly)

        # Second call should return cached result
        result2 = await engine.triage_anomaly(sample_anomaly)

        assert result1.anomaly_id == result2.anomaly_id
        assert result1.root_cause_analysis == result2.root_cause_analysis

    def test_cache_clear(self, triage_engine):
        """Test cache clearing."""
        triage_engine._cache["test_key"] = TriageResult(
            anomaly_id="test",
            root_cause_analysis="test",
            probable_causes=[],
            recommended_actions=[],
            severity_assessment="",
            estimated_impact="",
            related_anomalies=[],
            confidence_score=0.5,
        )
        triage_engine._cache_timestamps["test_key"] = datetime.now()

        triage_engine.clear_cache()

        assert len(triage_engine._cache) == 0
        assert len(triage_engine._cache_timestamps) == 0


# Runbook Generation Tests
class TestRunbookGeneration:
    """Tests for runbook generation."""

    @pytest.mark.asyncio
    async def test_generate_freshness_runbook(self, triage_engine):
        """Test generating runbook for freshness issues."""
        runbook = await triage_engine.generate_runbook(AnomalyType.FRESHNESS)

        assert isinstance(runbook, GeneratedRunbook)
        assert runbook.anomaly_type == AnomalyType.FRESHNESS
        assert len(runbook.steps) > 0
        assert runbook.title != ""

    @pytest.mark.asyncio
    async def test_generate_volume_runbook(self, triage_engine):
        """Test generating runbook for volume issues."""
        runbook = await triage_engine.generate_runbook(AnomalyType.VOLUME)

        assert isinstance(runbook, GeneratedRunbook)

    @pytest.mark.asyncio
    async def test_runbook_step_structure(self, triage_engine):
        """Test runbook step has proper structure."""
        runbook = await triage_engine.generate_runbook(AnomalyType.FRESHNESS)

        for step in runbook.steps:
            assert isinstance(step, RunbookStep)
            assert step.step_number > 0
            assert step.action != ""
            assert isinstance(step.automated, bool)


# Batch Triage Tests
class TestBatchTriage:
    """Tests for batch triage operations."""

    @pytest.mark.asyncio
    async def test_batch_triage_multiple_anomalies(self, triage_engine):
        """Test triaging multiple anomalies in batch."""
        anomalies = [
            Anomaly(
                anomaly_id=f"anomaly-{i}",
                table_id=f"db.schema.table_{i}",
                column_name=None,
                anomaly_type=AnomalyType.VOLUME,
                severity="warning",
                detected_at=datetime.now(),
                metric_value=float(i * 100),
                expected_range=(0, 50),
                description=f"Volume anomaly {i}",
            )
            for i in range(5)
        ]

        results = await triage_engine.batch_triage(anomalies, max_concurrent=3)

        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.anomaly_id == f"anomaly-{i}"

    @pytest.mark.asyncio
    async def test_batch_triage_with_metadata(self, triage_engine, sample_table_metadata):
        """Test batch triage with table metadata."""
        anomalies = [
            Anomaly(
                anomaly_id="anomaly-1",
                table_id=sample_table_metadata.table_id,
                column_name=None,
                anomaly_type=AnomalyType.VOLUME,
                severity="warning",
                detected_at=datetime.now(),
                metric_value=100,
                expected_range=(0, 50),
                description="Volume anomaly",
            )
        ]

        metadata_map = {sample_table_metadata.table_id: sample_table_metadata}
        results = await triage_engine.batch_triage(anomalies, table_metadata=metadata_map)

        assert len(results) == 1


# Report Generation Tests
class TestTriageReportGenerator:
    """Tests for report generation."""

    @pytest.mark.asyncio
    async def test_generate_incident_report(self, triage_engine, sample_anomaly):
        """Test generating incident report."""
        generator = TriageReportGenerator(triage_engine)

        report = await generator.generate_incident_report([sample_anomaly])

        assert "# Data Quality Incident Report" in report
        assert "NULL_RATE" in report
        assert sample_anomaly.table_id in report

    @pytest.mark.asyncio
    async def test_report_with_multiple_severities(self, triage_engine):
        """Test report with critical and warning anomalies."""
        anomalies = [
            Anomaly(
                anomaly_id="critical-1",
                table_id="db.schema.table1",
                column_name=None,
                anomaly_type=AnomalyType.FRESHNESS,
                severity="critical",
                detected_at=datetime.now(),
                metric_value=7200,
                expected_range=(0, 3600),
                description="Critical freshness issue",
            ),
            Anomaly(
                anomaly_id="warning-1",
                table_id="db.schema.table2",
                column_name=None,
                anomaly_type=AnomalyType.VOLUME,
                severity="warning",
                detected_at=datetime.now(),
                metric_value=500,
                expected_range=(1000, 2000),
                description="Warning volume issue",
            ),
        ]

        generator = TriageReportGenerator(triage_engine)
        report = await generator.generate_incident_report(anomalies)

        assert "## Critical Issues" in report
        assert "## Warnings" in report

    @pytest.mark.asyncio
    async def test_report_empty_anomalies(self, triage_engine):
        """Test report with no anomalies."""
        generator = TriageReportGenerator(triage_engine)
        report = await generator.generate_incident_report([])

        assert "No anomalies to report" in report


# Provider Configuration Tests
class TestProviderConfiguration:
    """Tests for different LLM provider configurations."""

    def test_local_provider_config(self):
        """Test local provider creates mock client."""
        config = TriageConfig(provider=LLMProvider.LOCAL)
        engine = LLMTriageEngine(config)

        client = engine._get_client()
        assert isinstance(client, MockLLMClient)

    def test_openai_provider_config(self):
        """Test OpenAI provider configuration."""
        config = TriageConfig(
            provider=LLMProvider.OPENAI,
            api_key="test-key",
            model="gpt-4",
        )
        engine = LLMTriageEngine(config)

        client = engine._get_client()
        assert isinstance(client, OpenAIClient)
        assert client.api_key == "test-key"
        assert client.model == "gpt-4"

    def test_anthropic_provider_config(self):
        """Test Anthropic provider configuration."""
        config = TriageConfig(
            provider=LLMProvider.ANTHROPIC,
            api_key="test-key",
            model="claude-3-sonnet-20240229",
        )
        engine = LLMTriageEngine(config)

        client = engine._get_client()
        assert isinstance(client, AnthropicClient)
        assert client.api_key == "test-key"


# Context Building Tests
class TestContextBuilding:
    """Tests for triage context building."""

    def test_build_context_basic(self, triage_engine, sample_anomaly):
        """Test basic context building."""
        context = triage_engine._build_context(
            sample_anomaly, None, None, None, None
        )

        assert context["anomaly_type"] == sample_anomaly.anomaly_type.value
        assert context["severity"] == sample_anomaly.severity
        assert context["description"] == sample_anomaly.description

    def test_build_context_with_all_params(
        self,
        triage_engine,
        sample_anomaly,
        sample_table_metadata,
        sample_health_score,
    ):
        """Test context building with all parameters."""
        related = [
            Anomaly(
                anomaly_id="related-1",
                table_id="other.table",
                column_name=None,
                anomaly_type=AnomalyType.VOLUME,
                severity="warning",
                detected_at=datetime.now(),
                metric_value=100,
                expected_range=(0, 50),
                description="Related issue",
            )
        ]
        lineage = {"upstream": ["upstream.table"], "downstream": ["downstream.table"]}

        context = triage_engine._build_context(
            sample_anomaly,
            sample_table_metadata,
            sample_health_score,
            related,
            lineage,
        )

        assert "table" in context
        assert "health" in context
        assert "related_anomalies" in context
        assert "lineage" in context
        assert context["table"]["row_count"] == sample_table_metadata.row_count
        assert context["health"]["overall_score"] == sample_health_score.overall_score


# Prompt Building Tests
class TestPromptBuilding:
    """Tests for prompt construction."""

    def test_triage_prompt_contains_anomaly_details(
        self, triage_engine, sample_anomaly
    ):
        """Test triage prompt includes anomaly details."""
        context = triage_engine._build_context(sample_anomaly, None, None, None, None)
        prompt = triage_engine._build_triage_prompt(sample_anomaly, context)

        assert sample_anomaly.anomaly_type.value in prompt
        assert sample_anomaly.table_id in prompt
        assert sample_anomaly.description in prompt

    def test_runbook_prompt_contains_type(self, triage_engine):
        """Test runbook prompt includes anomaly type."""
        prompt = triage_engine._build_runbook_prompt(AnomalyType.FRESHNESS, {})

        assert "freshness" in prompt.lower()
        assert "staleness" in prompt.lower()


# Response Parsing Tests
class TestResponseParsing:
    """Tests for LLM response parsing."""

    def test_parse_valid_triage_response(self, triage_engine, sample_anomaly):
        """Test parsing valid JSON response."""
        response = json.dumps({
            "root_cause_analysis": "Test analysis",
            "probable_causes": ["cause1", "cause2"],
            "recommended_actions": ["action1"],
            "severity_assessment": "High",
            "estimated_impact": "3 tables affected",
            "confidence_score": 0.85,
        })

        result = triage_engine._parse_triage_response(sample_anomaly, response)

        assert result.root_cause_analysis == "Test analysis"
        assert len(result.probable_causes) == 2
        assert result.confidence_score == 0.85

    def test_parse_invalid_json_response(self, triage_engine, sample_anomaly):
        """Test parsing invalid JSON gracefully."""
        response = "This is not valid JSON"

        result = triage_engine._parse_triage_response(sample_anomaly, response)

        assert result.anomaly_id == sample_anomaly.anomaly_id
        assert result.confidence_score == 0.5  # Default

    def test_parse_runbook_response(self, triage_engine):
        """Test parsing runbook response."""
        response = json.dumps({
            "title": "Test Runbook",
            "summary": "Test summary",
            "steps": [
                {
                    "step_number": 1,
                    "action": "Check logs",
                    "description": "Review logs",
                    "expected_outcome": "Find issue",
                    "automated": False,
                }
            ],
            "prerequisites": ["Access to logs"],
            "estimated_duration": "10 minutes",
        })

        runbook = triage_engine._parse_runbook_response(AnomalyType.FRESHNESS, response)

        assert runbook.title == "Test Runbook"
        assert len(runbook.steps) == 1
        assert runbook.steps[0].action == "Check logs"
