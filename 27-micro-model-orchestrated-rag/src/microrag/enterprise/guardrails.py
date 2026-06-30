"""Guardrail engine for pipeline safety."""

from abc import ABC, abstractmethod
from typing import Any

from ..schemas import GuardrailResult, RuleResult


class GuardrailRule(ABC):
    """Base class for guardrail rules."""

    @abstractmethod
    def applies_to(self, stage: str) -> bool:
        """Check if rule applies to stage."""
        pass

    @abstractmethod
    async def evaluate(self, input_data: Any, output_data: Any) -> RuleResult:
        """Evaluate the rule."""
        pass


class GuardrailEngine:
    """Applies guardrails at each pipeline stage."""

    def __init__(self, rules: list[GuardrailRule] = None):
        """Initialize engine.

        Args:
            rules: List of guardrail rules
        """
        self.rules = rules or []
        self.violations = []

    def add_rule(self, rule: GuardrailRule):
        """Add a guardrail rule."""
        self.rules.append(rule)

    async def check(
        self,
        stage: str,
        input_data: Any,
        output_data: Any
    ) -> GuardrailResult:
        """Check guardrails for a pipeline stage.

        Args:
            stage: Pipeline stage name
            input_data: Stage input
            output_data: Stage output

        Returns:
            Guardrail check result
        """
        violations = []

        for rule in self.rules:
            if rule.applies_to(stage):
                result = await rule.evaluate(input_data, output_data)
                if not result.passed:
                    violations.append(result)

        if violations:
            self.violations.extend(violations)

            # Determine action based on severity
            if any(v.severity == "critical" for v in violations):
                return GuardrailResult(
                    passed=False,
                    action="block",
                    violations=violations
                )
            elif any(v.severity == "warning" for v in violations):
                return GuardrailResult(
                    passed=True,
                    action="warn",
                    violations=violations
                )

        return GuardrailResult(passed=True, action="pass", violations=[])

    def get_violations(self) -> list:
        """Get all recorded violations."""
        return self.violations

    def clear_violations(self):
        """Clear violation history."""
        self.violations.clear()


class RelevanceGuardrail(GuardrailRule):
    """Ensures retrieved content is relevant to query."""

    def __init__(self, min_relevance: float = 0.3):
        self.min_relevance = min_relevance

    def applies_to(self, stage: str) -> bool:
        return stage in ["retrieval", "reranking"]

    async def evaluate(self, input_data, output_data) -> RuleResult:
        if not output_data:
            return RuleResult(
                passed=False,
                severity="warning",
                message="No results returned"
            )

        # Check if any results meet minimum relevance
        if hasattr(output_data, '__iter__'):
            try:
                scores = [r.score if hasattr(r, 'score') else r.relevance_score
                         for r in output_data]
                max_relevance = max(scores) if scores else 0
            except (AttributeError, TypeError):
                return RuleResult(passed=True)

            if max_relevance < self.min_relevance:
                return RuleResult(
                    passed=False,
                    severity="warning",
                    message=f"Max relevance {max_relevance:.2f} below threshold {self.min_relevance}"
                )

        return RuleResult(passed=True)


class HallucinationGuardrail(GuardrailRule):
    """Checks for hallucinations in generated content."""

    def __init__(self, max_hallucination_ratio: float = 0.1):
        self.max_ratio = max_hallucination_ratio

    def applies_to(self, stage: str) -> bool:
        return stage in ["summarization", "answer_generation", "stabilize"]

    async def evaluate(self, input_data, output_data) -> RuleResult:
        if not output_data:
            return RuleResult(passed=True)

        # Simple heuristic: check if output mentions things not in input
        # Real implementation would use NLI or fact verification
        if isinstance(output_data, str):
            output_text = output_data
        elif hasattr(output_data, 'answer'):
            output_text = output_data.answer
        else:
            return RuleResult(passed=True)

        # Check for suspicious patterns
        suspicious_patterns = [
            "I think",
            "probably",
            "might be",
            "I believe",
        ]

        found_patterns = [p for p in suspicious_patterns if p.lower() in output_text.lower()]

        if len(found_patterns) > 2:
            return RuleResult(
                passed=False,
                severity="warning",
                message=f"Potential hallucination indicators: {', '.join(found_patterns)}"
            )

        return RuleResult(passed=True)


class LengthGuardrail(GuardrailRule):
    """Ensures output meets length requirements."""

    def __init__(self, min_length: int = 10, max_length: int = 10000):
        self.min_length = min_length
        self.max_length = max_length

    def applies_to(self, stage: str) -> bool:
        return True

    async def evaluate(self, input_data, output_data) -> RuleResult:
        if output_data is None:
            return RuleResult(
                passed=False,
                severity="warning",
                message="Output is None"
            )

        # Get length
        if isinstance(output_data, str):
            length = len(output_data)
        elif hasattr(output_data, '__len__'):
            length = len(output_data)
        else:
            return RuleResult(passed=True)

        if length < self.min_length:
            return RuleResult(
                passed=False,
                severity="warning",
                message=f"Output length {length} below minimum {self.min_length}"
            )

        if length > self.max_length:
            return RuleResult(
                passed=False,
                severity="warning",
                message=f"Output length {length} exceeds maximum {self.max_length}"
            )

        return RuleResult(passed=True)


class ConfidenceGuardrail(GuardrailRule):
    """Ensures confidence scores meet threshold."""

    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence

    def applies_to(self, stage: str) -> bool:
        return stage in ["stabilize", "answer_generation"]

    async def evaluate(self, input_data, output_data) -> RuleResult:
        if hasattr(output_data, 'confidence'):
            if output_data.confidence < self.min_confidence:
                return RuleResult(
                    passed=False,
                    severity="warning",
                    message=f"Confidence {output_data.confidence:.2f} below threshold {self.min_confidence}"
                )

        return RuleResult(passed=True)
