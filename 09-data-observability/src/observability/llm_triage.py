"""LLM-based issue triage and root cause analysis."""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol

from observability.models import Anomaly, AnomalyType, DataHealthScore, TableMetadata

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    LOCAL = "local"  # For testing/local models


class LLMClient(Protocol):
    """Protocol for LLM clients."""

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.3,
    ) -> str:
        """Generate completion from prompt."""
        ...


@dataclass
class TriageResult:
    """Result of LLM-based triage."""

    anomaly_id: str
    root_cause_analysis: str
    probable_causes: List[str]
    recommended_actions: List[str]
    severity_assessment: str
    estimated_impact: str
    related_anomalies: List[str]
    confidence_score: float
    generated_at: datetime = field(default_factory=datetime.now)
    model_used: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class RunbookStep:
    """A step in a generated runbook."""

    step_number: int
    action: str
    description: str
    expected_outcome: str
    rollback_action: Optional[str] = None
    automated: bool = False


@dataclass
class GeneratedRunbook:
    """Auto-generated runbook for issue remediation."""

    anomaly_type: AnomalyType
    title: str
    summary: str
    steps: List[RunbookStep]
    prerequisites: List[str]
    estimated_duration: str
    generated_at: datetime = field(default_factory=datetime.now)


class MockLLMClient:
    """Mock LLM client for testing without API calls."""

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.3,
    ) -> str:
        """Generate mock response based on prompt patterns."""
        # Simulate some latency
        await asyncio.sleep(0.1)

        if "root cause" in prompt.lower():
            return json.dumps({
                "root_cause_analysis": "Based on the anomaly pattern, the issue appears to be caused by an upstream data pipeline failure or schema change.",
                "probable_causes": [
                    "Upstream ETL job failure",
                    "Source system schema change",
                    "Data quality issue in source",
                    "Network connectivity issues",
                ],
                "recommended_actions": [
                    "Check upstream pipeline status",
                    "Verify source system health",
                    "Review recent schema changes",
                    "Check network connectivity logs",
                ],
                "severity_assessment": "High - This issue affects downstream consumers",
                "estimated_impact": "3 downstream tables and 2 dashboards affected",
                "confidence_score": 0.85,
            })
        elif "runbook" in prompt.lower():
            return json.dumps({
                "title": "Data Freshness Issue Remediation",
                "summary": "Steps to investigate and resolve data freshness anomalies",
                "steps": [
                    {
                        "step_number": 1,
                        "action": "Check pipeline status",
                        "description": "Verify the upstream ETL pipeline status in Airflow",
                        "expected_outcome": "Identify if pipeline has failed or is delayed",
                        "automated": True,
                    },
                    {
                        "step_number": 2,
                        "action": "Review logs",
                        "description": "Examine pipeline logs for errors or warnings",
                        "expected_outcome": "Identify root cause of delay",
                        "automated": False,
                    },
                    {
                        "step_number": 3,
                        "action": "Trigger rerun",
                        "description": "If safe, trigger a pipeline rerun",
                        "expected_outcome": "Data refreshed within expected timeframe",
                        "rollback_action": "Cancel rerun if issues persist",
                        "automated": True,
                    },
                ],
                "prerequisites": [
                    "Access to Airflow UI",
                    "Pipeline admin permissions",
                ],
                "estimated_duration": "15-30 minutes",
            })
        else:
            return json.dumps({
                "response": "Analysis complete",
                "details": "No specific pattern matched",
            })


class OpenAIClient:
    """OpenAI API client."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url or "https://api.openai.com/v1"
        self._client = None

    async def _get_client(self):
        """Get or create HTTP client."""
        if self._client is None:
            try:
                import httpx
                self._client = httpx.AsyncClient(
                    base_url=self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=60.0,
                )
            except ImportError:
                raise ImportError("httpx is required for OpenAI client")
        return self._client

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.3,
    ) -> str:
        """Generate completion using OpenAI API."""
        client = await self._get_client()

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a data engineering expert specializing in data quality and observability. Provide detailed, actionable analysis."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        response = await client.post("/chat/completions", json=payload)
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]


class AnthropicClient:
    """Anthropic Claude API client."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-sonnet-20240229",
    ):
        self.api_key = api_key
        self.model = model
        self._client = None

    async def _get_client(self):
        """Get or create HTTP client."""
        if self._client is None:
            try:
                import httpx
                self._client = httpx.AsyncClient(
                    base_url="https://api.anthropic.com/v1",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                    timeout=60.0,
                )
            except ImportError:
                raise ImportError("httpx is required for Anthropic client")
        return self._client

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.3,
    ) -> str:
        """Generate completion using Claude API."""
        client = await self._get_client()

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "system": "You are a data engineering expert specializing in data quality and observability. Provide detailed, actionable analysis in JSON format when requested.",
        }

        response = await client.post("/messages", json=payload)
        response.raise_for_status()

        data = response.json()
        return data["content"][0]["text"]


@dataclass
class TriageConfig:
    """Configuration for LLM triage."""

    provider: LLMProvider = LLMProvider.LOCAL
    api_key: str = ""
    model: str = ""
    max_tokens: int = 1000
    temperature: float = 0.3
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600


class LLMTriageEngine:
    """LLM-powered issue triage and root cause analysis."""

    def __init__(self, config: TriageConfig):
        self.config = config
        self._client: Optional[LLMClient] = None
        self._cache: Dict[str, TriageResult] = {}
        self._cache_timestamps: Dict[str, datetime] = {}

    def _get_client(self) -> LLMClient:
        """Get or create LLM client based on config."""
        if self._client is not None:
            return self._client

        if self.config.provider == LLMProvider.OPENAI:
            self._client = OpenAIClient(
                api_key=self.config.api_key,
                model=self.config.model or "gpt-4",
            )
        elif self.config.provider == LLMProvider.ANTHROPIC:
            self._client = AnthropicClient(
                api_key=self.config.api_key,
                model=self.config.model or "claude-3-sonnet-20240229",
            )
        elif self.config.provider == LLMProvider.AZURE_OPENAI:
            # Azure OpenAI uses different endpoint
            self._client = OpenAIClient(
                api_key=self.config.api_key,
                model=self.config.model,
                base_url=f"https://{self.config.model}.openai.azure.com/openai/deployments/{self.config.model}",
            )
        else:
            self._client = MockLLMClient()

        return self._client

    def _cache_key(self, anomaly: Anomaly, context: Dict[str, Any]) -> str:
        """Generate cache key for triage request."""
        content = f"{anomaly.anomaly_id}:{anomaly.anomaly_type.value}:{json.dumps(context, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached result is still valid."""
        if key not in self._cache_timestamps:
            return False
        age = (datetime.now() - self._cache_timestamps[key]).total_seconds()
        return age < self.config.cache_ttl_seconds

    async def triage_anomaly(
        self,
        anomaly: Anomaly,
        table_metadata: Optional[TableMetadata] = None,
        health_score: Optional[DataHealthScore] = None,
        related_anomalies: Optional[List[Anomaly]] = None,
        lineage_context: Optional[Dict[str, Any]] = None,
    ) -> TriageResult:
        """Perform LLM-based triage on an anomaly."""
        context = self._build_context(
            anomaly, table_metadata, health_score, related_anomalies, lineage_context
        )

        # Check cache
        if self.config.cache_enabled:
            cache_key = self._cache_key(anomaly, context)
            if cache_key in self._cache and self._is_cache_valid(cache_key):
                logger.info(f"Returning cached triage for anomaly {anomaly.anomaly_id}")
                return self._cache[cache_key]

        # Build prompt
        prompt = self._build_triage_prompt(anomaly, context)

        # Get LLM response
        client = self._get_client()
        response = await client.generate(
            prompt=prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        # Parse response
        result = self._parse_triage_response(anomaly, response)

        # Cache result
        if self.config.cache_enabled:
            self._cache[cache_key] = result
            self._cache_timestamps[cache_key] = datetime.now()

        return result

    def _build_context(
        self,
        anomaly: Anomaly,
        table_metadata: Optional[TableMetadata],
        health_score: Optional[DataHealthScore],
        related_anomalies: Optional[List[Anomaly]],
        lineage_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build context for triage prompt."""
        context = {
            "anomaly_type": anomaly.anomaly_type.value,
            "severity": anomaly.severity,
            "description": anomaly.description,
            "metric_value": anomaly.metric_value,
            "expected_range": anomaly.expected_range,
            "context": anomaly.context,
        }

        if table_metadata:
            context["table"] = {
                "table_id": table_metadata.table_id,
                "database": table_metadata.database,
                "schema": table_metadata.schema,
                "row_count": table_metadata.row_count,
                "column_count": len(table_metadata.columns),
                "last_modified": table_metadata.last_modified.isoformat(),
            }

        if health_score:
            context["health"] = {
                "overall_score": health_score.overall_score,
                "freshness_score": health_score.freshness_score,
                "volume_score": health_score.volume_score,
                "schema_score": health_score.schema_score,
                "quality_score": health_score.quality_score,
            }

        if related_anomalies:
            context["related_anomalies"] = [
                {
                    "type": a.anomaly_type.value,
                    "severity": a.severity,
                    "description": a.description,
                }
                for a in related_anomalies[:5]  # Limit to 5
            ]

        if lineage_context:
            context["lineage"] = lineage_context

        return context

    def _build_triage_prompt(
        self, anomaly: Anomaly, context: Dict[str, Any]
    ) -> str:
        """Build prompt for root cause analysis."""
        return f"""Analyze the following data quality anomaly and provide root cause analysis.

## Anomaly Details
- Type: {anomaly.anomaly_type.value}
- Severity: {anomaly.severity}
- Table: {anomaly.table_id}
- Column: {anomaly.column_name or 'N/A'}
- Description: {anomaly.description}
- Metric Value: {anomaly.metric_value}
- Expected Range: {anomaly.expected_range}
- Detected At: {anomaly.detected_at.isoformat()}

## Additional Context
{json.dumps(context, indent=2, default=str)}

## Instructions
Provide a JSON response with the following structure:
{{
    "root_cause_analysis": "Detailed analysis of the root cause",
    "probable_causes": ["cause1", "cause2", ...],
    "recommended_actions": ["action1", "action2", ...],
    "severity_assessment": "Assessment of actual severity",
    "estimated_impact": "Description of downstream impact",
    "confidence_score": 0.0-1.0
}}

Focus on:
1. Identifying the most likely root cause based on the anomaly pattern
2. Considering upstream dependencies and data lineage
3. Providing actionable remediation steps
4. Assessing impact on downstream consumers
"""

    def _parse_triage_response(
        self, anomaly: Anomaly, response: str
    ) -> TriageResult:
        """Parse LLM response into TriageResult."""
        try:
            # Try to extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
            else:
                data = {}
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM response as JSON: {response[:200]}")
            data = {}

        return TriageResult(
            anomaly_id=anomaly.anomaly_id,
            root_cause_analysis=data.get(
                "root_cause_analysis",
                "Unable to determine root cause from available information.",
            ),
            probable_causes=data.get("probable_causes", []),
            recommended_actions=data.get("recommended_actions", []),
            severity_assessment=data.get(
                "severity_assessment", f"Severity: {anomaly.severity}"
            ),
            estimated_impact=data.get(
                "estimated_impact", "Impact assessment not available"
            ),
            related_anomalies=[],
            confidence_score=data.get("confidence_score", 0.5),
            model_used=self.config.model or str(self.config.provider.value),
        )

    async def generate_runbook(
        self,
        anomaly_type: AnomalyType,
        context: Optional[Dict[str, Any]] = None,
    ) -> GeneratedRunbook:
        """Generate a remediation runbook for an anomaly type."""
        prompt = self._build_runbook_prompt(anomaly_type, context or {})

        client = self._get_client()
        response = await client.generate(
            prompt=prompt,
            max_tokens=1500,
            temperature=0.2,
        )

        return self._parse_runbook_response(anomaly_type, response)

    def _build_runbook_prompt(
        self, anomaly_type: AnomalyType, context: Dict[str, Any]
    ) -> str:
        """Build prompt for runbook generation."""
        type_descriptions = {
            AnomalyType.VOLUME: "Row count deviation - unexpected increase or decrease in table size",
            AnomalyType.FRESHNESS: "Data staleness - table not updated within expected timeframe",
            AnomalyType.SCHEMA: "Schema change - columns added, removed, or type changed",
            AnomalyType.NULL_RATE: "Null rate increase - higher than expected null values in column",
            AnomalyType.DISTRIBUTION: "Distribution shift - statistical properties of column changed",
            AnomalyType.UNIQUENESS: "Uniqueness violation - duplicate values in expected unique column",
            AnomalyType.CUSTOM: "Custom rule violation - user-defined quality check failed",
        }

        return f"""Generate a detailed runbook for resolving the following data quality issue.

## Issue Type
{anomaly_type.value}: {type_descriptions.get(anomaly_type, 'Unknown issue type')}

## Context
{json.dumps(context, indent=2, default=str)}

## Instructions
Provide a JSON response with the following structure:
{{
    "title": "Runbook title",
    "summary": "Brief summary of the remediation process",
    "steps": [
        {{
            "step_number": 1,
            "action": "Short action name",
            "description": "Detailed description of what to do",
            "expected_outcome": "What should happen after this step",
            "rollback_action": "How to undo if needed (optional)",
            "automated": true/false
        }}
    ],
    "prerequisites": ["prerequisite1", "prerequisite2"],
    "estimated_duration": "X-Y minutes"
}}

Focus on:
1. Clear, actionable steps that can be followed by an on-call engineer
2. Including both diagnostic and remediation steps
3. Identifying which steps can be automated
4. Providing rollback procedures where applicable
"""

    def _parse_runbook_response(
        self, anomaly_type: AnomalyType, response: str
    ) -> GeneratedRunbook:
        """Parse LLM response into GeneratedRunbook."""
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
            else:
                data = {}
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse runbook response: {response[:200]}")
            data = {}

        steps = []
        for step_data in data.get("steps", []):
            steps.append(RunbookStep(
                step_number=step_data.get("step_number", len(steps) + 1),
                action=step_data.get("action", "Unknown action"),
                description=step_data.get("description", ""),
                expected_outcome=step_data.get("expected_outcome", ""),
                rollback_action=step_data.get("rollback_action"),
                automated=step_data.get("automated", False),
            ))

        return GeneratedRunbook(
            anomaly_type=anomaly_type,
            title=data.get("title", f"{anomaly_type.value} Remediation Runbook"),
            summary=data.get("summary", ""),
            steps=steps,
            prerequisites=data.get("prerequisites", []),
            estimated_duration=data.get("estimated_duration", "Unknown"),
        )

    async def batch_triage(
        self,
        anomalies: List[Anomaly],
        table_metadata: Optional[Dict[str, TableMetadata]] = None,
        max_concurrent: int = 5,
    ) -> List[TriageResult]:
        """Perform triage on multiple anomalies concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def triage_with_limit(anomaly: Anomaly) -> TriageResult:
            async with semaphore:
                metadata = (
                    table_metadata.get(anomaly.table_id)
                    if table_metadata
                    else None
                )
                return await self.triage_anomaly(anomaly, table_metadata=metadata)

        tasks = [triage_with_limit(anomaly) for anomaly in anomalies]
        return await asyncio.gather(*tasks)

    def clear_cache(self) -> None:
        """Clear the triage cache."""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.info("Triage cache cleared")


class TriageReportGenerator:
    """Generate formatted triage reports."""

    def __init__(self, triage_engine: LLMTriageEngine):
        self.engine = triage_engine

    async def generate_incident_report(
        self,
        anomalies: List[Anomaly],
        table_metadata: Optional[Dict[str, TableMetadata]] = None,
    ) -> str:
        """Generate a comprehensive incident report."""
        if not anomalies:
            return "No anomalies to report."

        # Triage all anomalies
        results = await self.engine.batch_triage(anomalies, table_metadata)

        # Group by severity
        critical = []
        warning = []
        for anomaly, result in zip(anomalies, results):
            if anomaly.severity == "critical":
                critical.append((anomaly, result))
            else:
                warning.append((anomaly, result))

        # Build report
        lines = [
            "# Data Quality Incident Report",
            f"Generated: {datetime.now().isoformat()}",
            f"Total Anomalies: {len(anomalies)}",
            f"Critical: {len(critical)} | Warning: {len(warning)}",
            "",
        ]

        if critical:
            lines.append("## Critical Issues")
            for anomaly, result in critical:
                lines.extend(self._format_anomaly_section(anomaly, result))

        if warning:
            lines.append("## Warnings")
            for anomaly, result in warning:
                lines.extend(self._format_anomaly_section(anomaly, result))

        lines.append("## Summary of Recommended Actions")
        all_actions = set()
        for result in results:
            all_actions.update(result.recommended_actions)
        for action in sorted(all_actions):
            lines.append(f"- {action}")

        return "\n".join(lines)

    def _format_anomaly_section(
        self, anomaly: Anomaly, result: TriageResult
    ) -> List[str]:
        """Format a single anomaly section."""
        return [
            f"### {anomaly.anomaly_type.value.upper()}: {anomaly.table_id}",
            f"**Description:** {anomaly.description}",
            f"**Root Cause:** {result.root_cause_analysis}",
            f"**Impact:** {result.estimated_impact}",
            f"**Confidence:** {result.confidence_score:.0%}",
            "",
            "**Probable Causes:**",
            *[f"- {cause}" for cause in result.probable_causes],
            "",
            "**Recommended Actions:**",
            *[f"- {action}" for action in result.recommended_actions],
            "",
        ]
