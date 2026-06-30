"""A/B testing framework for RAG configurations."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import hashlib
import numpy as np
from scipy import stats


@dataclass
class ABTest:
    """A/B test definition."""

    id: str
    name: str
    control_config: dict
    treatment_config: dict
    traffic_split: float  # Fraction to treatment
    metrics: list[str]
    status: str  # active, paused, completed
    start_time: datetime
    end_time: Optional[datetime] = None
    results: dict = field(default_factory=lambda: {"control": [], "treatment": []})


@dataclass
class ABTestAnalysis:
    """Analysis results for an A/B test."""

    test_id: str
    metric_results: dict[str, dict]
    sample_sizes: dict[str, int]
    recommendation: str
    analysis_time: datetime


class ABTestManager:
    """Manages A/B tests for RAG configurations."""

    def __init__(self, storage=None):
        """Initialize manager.

        Args:
            storage: Storage backend for tests
        """
        self.storage = storage or InMemoryTestStore()
        self.active_tests: dict[str, ABTest] = {}

    async def create_test(
        self,
        name: str,
        control_config: dict,
        treatment_config: dict,
        traffic_split: float = 0.5,
        metrics: Optional[list[str]] = None,
    ) -> str:
        """Create a new A/B test.

        Args:
            name: Test name
            control_config: Control configuration
            treatment_config: Treatment configuration
            traffic_split: Fraction of traffic to treatment (0-1)
            metrics: Metrics to track

        Returns:
            Test ID
        """
        test_id = self._generate_id()

        test = ABTest(
            id=test_id,
            name=name,
            control_config=control_config,
            treatment_config=treatment_config,
            traffic_split=traffic_split,
            metrics=metrics or ["latency", "confidence", "citation_count"],
            status="active",
            start_time=datetime.utcnow(),
        )

        await self.storage.save(test)
        self.active_tests[test_id] = test

        return test_id

    async def get_variant(self, test_id: str, user_id: str) -> str:
        """Get variant assignment for a user.

        Uses consistent hashing for deterministic assignment.

        Args:
            test_id: Test identifier
            user_id: User identifier

        Returns:
            "control" or "treatment"
        """
        test = self.active_tests.get(test_id)
        if not test or test.status != "active":
            return "control"

        # Consistent hashing for user assignment
        hash_input = f"{test_id}:{user_id}"
        hash_val = int(hashlib.md5(hash_input.encode()).hexdigest(), 16) % 100

        return "treatment" if hash_val < test.traffic_split * 100 else "control"

    async def get_config(self, test_id: str, user_id: str) -> dict:
        """Get configuration for a user based on variant.

        Args:
            test_id: Test identifier
            user_id: User identifier

        Returns:
            Configuration dict
        """
        variant = await self.get_variant(test_id, user_id)
        test = self.active_tests.get(test_id)

        if not test:
            return {}

        if variant == "treatment":
            return test.treatment_config
        return test.control_config

    async def record_result(
        self,
        test_id: str,
        user_id: str,
        metrics: dict,
    ):
        """Record metrics for a request.

        Args:
            test_id: Test identifier
            user_id: User identifier
            metrics: Metric values to record
        """
        test = self.active_tests.get(test_id)
        if not test or test.status != "active":
            return

        variant = await self.get_variant(test_id, user_id)
        test.results[variant].append({
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            **metrics,
        })

        # Persist periodically
        if len(test.results[variant]) % 100 == 0:
            await self.storage.save(test)

    async def analyze_test(self, test_id: str) -> ABTestAnalysis:
        """Analyze A/B test results.

        Args:
            test_id: Test identifier

        Returns:
            Analysis results with statistical significance
        """
        test = await self.storage.get(test_id)
        if not test:
            raise ValueError(f"Test not found: {test_id}")

        analysis = {}

        for metric in test.metrics:
            control_vals = [
                r.get(metric, 0) for r in test.results["control"]
                if metric in r
            ]
            treatment_vals = [
                r.get(metric, 0) for r in test.results["treatment"]
                if metric in r
            ]

            if not control_vals or not treatment_vals:
                analysis[metric] = {
                    "control_mean": 0,
                    "treatment_mean": 0,
                    "lift": 0,
                    "p_value": 1.0,
                    "significant": False,
                    "error": "Insufficient data",
                }
                continue

            # T-test for significance
            t_stat, p_value = stats.ttest_ind(control_vals, treatment_vals)

            control_mean = np.mean(control_vals)
            treatment_mean = np.mean(treatment_vals)

            lift = (
                (treatment_mean - control_mean) / control_mean
                if control_mean != 0 else 0
            )

            analysis[metric] = {
                "control_mean": float(control_mean),
                "treatment_mean": float(treatment_mean),
                "control_std": float(np.std(control_vals)),
                "treatment_std": float(np.std(treatment_vals)),
                "lift": float(lift),
                "lift_percent": float(lift * 100),
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
            }

        # Generate recommendation
        recommendation = self._generate_recommendation(analysis)

        return ABTestAnalysis(
            test_id=test_id,
            metric_results=analysis,
            sample_sizes={
                "control": len(test.results["control"]),
                "treatment": len(test.results["treatment"]),
            },
            recommendation=recommendation,
            analysis_time=datetime.utcnow(),
        )

    async def stop_test(self, test_id: str):
        """Stop an active test.

        Args:
            test_id: Test identifier
        """
        test = self.active_tests.get(test_id)
        if test:
            test.status = "completed"
            test.end_time = datetime.utcnow()
            await self.storage.save(test)
            del self.active_tests[test_id]

    async def pause_test(self, test_id: str):
        """Pause an active test.

        Args:
            test_id: Test identifier
        """
        test = self.active_tests.get(test_id)
        if test:
            test.status = "paused"
            await self.storage.save(test)

    async def resume_test(self, test_id: str):
        """Resume a paused test.

        Args:
            test_id: Test identifier
        """
        test = await self.storage.get(test_id)
        if test and test.status == "paused":
            test.status = "active"
            self.active_tests[test_id] = test
            await self.storage.save(test)

    async def list_tests(self, status: Optional[str] = None) -> list[ABTest]:
        """List all tests.

        Args:
            status: Filter by status (active, paused, completed)

        Returns:
            List of tests
        """
        tests = await self.storage.list_all()
        if status:
            tests = [t for t in tests if t.status == status]
        return tests

    def _generate_id(self) -> str:
        """Generate unique test ID."""
        import uuid
        return f"test_{uuid.uuid4().hex[:8]}"

    def _generate_recommendation(self, analysis: dict) -> str:
        """Generate recommendation based on analysis."""
        significant_improvements = []
        significant_regressions = []

        for metric, results in analysis.items():
            if "error" in results:
                continue

            if results["significant"]:
                if results["lift"] > 0:
                    significant_improvements.append(
                        f"{metric} (+{results['lift_percent']:.1f}%)"
                    )
                else:
                    significant_regressions.append(
                        f"{metric} ({results['lift_percent']:.1f}%)"
                    )

        if significant_regressions:
            return (
                f"DO NOT SHIP: Significant regressions in {', '.join(significant_regressions)}. "
                "Investigate before proceeding."
            )
        elif significant_improvements:
            return (
                f"SHIP IT: Significant improvements in {', '.join(significant_improvements)}. "
                "Treatment is better."
            )
        else:
            return (
                "INCONCLUSIVE: No statistically significant differences found. "
                "Consider running longer or increasing sample size."
            )


class InMemoryTestStore:
    """Simple in-memory storage for A/B tests."""

    def __init__(self):
        self._tests: dict[str, ABTest] = {}

    async def save(self, test: ABTest):
        """Save a test."""
        self._tests[test.id] = test

    async def get(self, test_id: str) -> Optional[ABTest]:
        """Get a test by ID."""
        return self._tests.get(test_id)

    async def delete(self, test_id: str):
        """Delete a test."""
        self._tests.pop(test_id, None)

    async def list_all(self) -> list[ABTest]:
        """List all tests."""
        return list(self._tests.values())
