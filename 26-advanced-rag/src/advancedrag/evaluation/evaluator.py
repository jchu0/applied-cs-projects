"""Comprehensive evaluation framework for RAG systems."""

from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
from datetime import datetime

from ..schemas import RAGResult


@dataclass
class EvaluationResult:
    """Result of a single metric evaluation."""

    metric_name: str
    value: float
    confidence_interval: tuple[float, float]
    metadata: dict


@dataclass
class EvaluationReport:
    """Complete evaluation report."""

    eval_id: str
    timestamp: datetime
    metrics: dict[str, EvaluationResult]
    config: dict
    sample_count: int


class RAGEvaluator:
    """Comprehensive evaluation framework for RAG systems."""

    def __init__(self, rag_pipeline=None):
        """Initialize evaluator.

        Args:
            rag_pipeline: Optional RAG pipeline to evaluate
        """
        self.pipeline = rag_pipeline
        self.metrics: dict[str, Callable] = {}
        self._register_default_metrics()

    def register_metric(self, name: str, metric_fn: Callable):
        """Register a custom metric.

        Args:
            name: Metric name
            metric_fn: Function that takes predictions and returns EvaluationResult
        """
        self.metrics[name] = metric_fn

    def _register_default_metrics(self):
        """Register default evaluation metrics."""
        self.metrics["recall@5"] = lambda p: retrieval_recall_at_k(p, k=5)
        self.metrics["recall@10"] = lambda p: retrieval_recall_at_k(p, k=10)
        self.metrics["mrr"] = retrieval_mrr
        self.metrics["ndcg@10"] = lambda p: retrieval_ndcg(p, k=10)
        self.metrics["precision@5"] = lambda p: retrieval_precision_at_k(p, k=5)
        self.metrics["answer_length"] = answer_length_metric
        self.metrics["citation_count"] = citation_count_metric
        self.metrics["confidence"] = confidence_metric
        self.metrics["faithfulness"] = answer_faithfulness_metric
        self.metrics["relevance"] = answer_relevance_metric
        self.metrics["context_precision"] = context_precision_metric
        self.metrics["context_recall"] = context_recall_metric
        self.metrics["answer_correctness"] = answer_correctness_metric

    async def evaluate(
        self,
        test_set: list[dict],
        metrics: Optional[list[str]] = None,
        config: Optional[dict] = None,
    ) -> EvaluationReport:
        """Run evaluation on test set.

        Args:
            test_set: List of test cases with query, expected_answer, relevant_docs
            metrics: Specific metrics to compute (None for all)
            config: Optional configuration overrides

        Returns:
            Complete evaluation report
        """
        metrics = metrics or list(self.metrics.keys())

        # Run pipeline on test set
        predictions = []
        for item in test_set:
            if self.pipeline:
                result = await self.pipeline.query(item["query"])
            else:
                result = item.get("predicted")

            predictions.append({
                "query": item["query"],
                "expected": item.get("expected_answer"),
                "expected_docs": item.get("relevant_docs", []),
                "predicted": result,
            })

        # Compute metrics
        results = {}
        for metric_name in metrics:
            if metric_name in self.metrics:
                try:
                    result = self.metrics[metric_name](predictions)
                    results[metric_name] = result
                except Exception as e:
                    results[metric_name] = EvaluationResult(
                        metric_name=metric_name,
                        value=0.0,
                        confidence_interval=(0.0, 0.0),
                        metadata={"error": str(e)},
                    )

        return EvaluationReport(
            eval_id=f"eval_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.utcnow(),
            metrics=results,
            config=config or {},
            sample_count=len(test_set),
        )

    async def compare_configs(
        self,
        test_set: list[dict],
        configs: list[dict],
        metrics: Optional[list[str]] = None,
    ) -> dict[str, EvaluationReport]:
        """Compare multiple configurations.

        Args:
            test_set: Test cases
            configs: List of configurations to compare
            metrics: Metrics to compute

        Returns:
            Dictionary of config name to evaluation report
        """
        results = {}
        for i, config in enumerate(configs):
            config_name = config.get("name", f"config_{i}")
            report = await self.evaluate(test_set, metrics, config)
            results[config_name] = report
        return results


# Retrieval Metrics

def retrieval_recall_at_k(predictions: list[dict], k: int = 10) -> EvaluationResult:
    """Compute recall at k for retrieval results."""
    recalls = []

    for pred in predictions:
        expected = set(pred.get("expected_docs", []))
        if not expected:
            continue

        result = pred.get("predicted")
        if result and hasattr(result, "retrieval_results"):
            retrieved = set(
                r.document.id for r in result.retrieval_results[:k]
            )
        else:
            retrieved = set()

        recall = len(expected & retrieved) / len(expected) if expected else 1.0
        recalls.append(recall)

    if not recalls:
        return EvaluationResult(
            metric_name=f"recall@{k}",
            value=0.0,
            confidence_interval=(0.0, 0.0),
            metadata={"n_samples": 0},
        )

    return EvaluationResult(
        metric_name=f"recall@{k}",
        value=float(np.mean(recalls)),
        confidence_interval=_bootstrap_ci(recalls),
        metadata={"k": k, "n_samples": len(recalls)},
    )


def retrieval_precision_at_k(predictions: list[dict], k: int = 10) -> EvaluationResult:
    """Compute precision at k for retrieval results."""
    precisions = []

    for pred in predictions:
        expected = set(pred.get("expected_docs", []))
        if not expected:
            continue

        result = pred.get("predicted")
        if result and hasattr(result, "retrieval_results"):
            retrieved = [r.document.id for r in result.retrieval_results[:k]]
        else:
            retrieved = []

        if not retrieved:
            precisions.append(0.0)
            continue

        precision = len(set(retrieved) & expected) / len(retrieved)
        precisions.append(precision)

    if not precisions:
        return EvaluationResult(
            metric_name=f"precision@{k}",
            value=0.0,
            confidence_interval=(0.0, 0.0),
            metadata={"n_samples": 0},
        )

    return EvaluationResult(
        metric_name=f"precision@{k}",
        value=float(np.mean(precisions)),
        confidence_interval=_bootstrap_ci(precisions),
        metadata={"k": k, "n_samples": len(precisions)},
    )


def retrieval_mrr(predictions: list[dict]) -> EvaluationResult:
    """Compute Mean Reciprocal Rank for retrieval results."""
    mrrs = []

    for pred in predictions:
        expected = set(pred.get("expected_docs", []))
        if not expected:
            continue

        result = pred.get("predicted")
        if not result or not hasattr(result, "retrieval_results"):
            mrrs.append(0.0)
            continue

        for rank, r in enumerate(result.retrieval_results):
            if r.document.id in expected:
                mrrs.append(1 / (rank + 1))
                break
        else:
            mrrs.append(0.0)

    if not mrrs:
        return EvaluationResult(
            metric_name="mrr",
            value=0.0,
            confidence_interval=(0.0, 0.0),
            metadata={"n_samples": 0},
        )

    return EvaluationResult(
        metric_name="mrr",
        value=float(np.mean(mrrs)),
        confidence_interval=_bootstrap_ci(mrrs),
        metadata={"n_samples": len(mrrs)},
    )


def retrieval_ndcg(predictions: list[dict], k: int = 10) -> EvaluationResult:
    """Compute Normalized Discounted Cumulative Gain at k."""
    ndcgs = []

    for pred in predictions:
        expected = set(pred.get("expected_docs", []))
        if not expected:
            continue

        result = pred.get("predicted")
        if not result or not hasattr(result, "retrieval_results"):
            ndcgs.append(0.0)
            continue

        # Compute DCG
        dcg = 0.0
        for i, r in enumerate(result.retrieval_results[:k]):
            if r.document.id in expected:
                dcg += 1 / np.log2(i + 2)

        # Compute ideal DCG
        ideal_dcg = sum(1 / np.log2(i + 2) for i in range(min(len(expected), k)))

        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
        ndcgs.append(ndcg)

    if not ndcgs:
        return EvaluationResult(
            metric_name=f"ndcg@{k}",
            value=0.0,
            confidence_interval=(0.0, 0.0),
            metadata={"n_samples": 0},
        )

    return EvaluationResult(
        metric_name=f"ndcg@{k}",
        value=float(np.mean(ndcgs)),
        confidence_interval=_bootstrap_ci(ndcgs),
        metadata={"k": k, "n_samples": len(ndcgs)},
    )


# Answer Quality Metrics

def answer_length_metric(predictions: list[dict]) -> EvaluationResult:
    """Compute average answer length."""
    lengths = []

    for pred in predictions:
        result = pred.get("predicted")
        if result and hasattr(result, "answer"):
            lengths.append(len(result.answer.answer.split()))

    if not lengths:
        return EvaluationResult(
            metric_name="answer_length",
            value=0.0,
            confidence_interval=(0.0, 0.0),
            metadata={"n_samples": 0},
        )

    return EvaluationResult(
        metric_name="answer_length",
        value=float(np.mean(lengths)),
        confidence_interval=_bootstrap_ci(lengths),
        metadata={"n_samples": len(lengths), "unit": "words"},
    )


def citation_count_metric(predictions: list[dict]) -> EvaluationResult:
    """Compute average citation count."""
    counts = []

    for pred in predictions:
        result = pred.get("predicted")
        if result and hasattr(result, "answer"):
            counts.append(len(result.answer.citations))

    if not counts:
        return EvaluationResult(
            metric_name="citation_count",
            value=0.0,
            confidence_interval=(0.0, 0.0),
            metadata={"n_samples": 0},
        )

    return EvaluationResult(
        metric_name="citation_count",
        value=float(np.mean(counts)),
        confidence_interval=_bootstrap_ci(counts),
        metadata={"n_samples": len(counts)},
    )


def confidence_metric(predictions: list[dict]) -> EvaluationResult:
    """Compute average confidence score."""
    scores = []

    for pred in predictions:
        result = pred.get("predicted")
        if result and hasattr(result, "answer"):
            scores.append(result.answer.confidence)

    if not scores:
        return EvaluationResult(
            metric_name="confidence",
            value=0.0,
            confidence_interval=(0.0, 0.0),
            metadata={"n_samples": 0},
        )

    return EvaluationResult(
        metric_name="confidence",
        value=float(np.mean(scores)),
        confidence_interval=_bootstrap_ci(scores),
        metadata={"n_samples": len(scores)},
    )


# LLM-judged / proxy evaluation metrics

def answer_faithfulness_metric(predictions: list[dict]) -> EvaluationResult:
    """Measure faithfulness: word overlap between answer and context as proxy."""
    scores = []

    for pred in predictions:
        result = pred.get("predicted")
        if not result or not hasattr(result, "answer") or not hasattr(result, "context"):
            continue

        answer_words = set(result.answer.answer.lower().split())
        context_words = set(result.context.content.lower().split())

        if not answer_words:
            scores.append(0.0)
            continue

        overlap = len(answer_words & context_words) / len(answer_words)
        scores.append(overlap)

    if not scores:
        return EvaluationResult(
            metric_name="faithfulness",
            value=0.0,
            confidence_interval=(0.0, 0.0),
            metadata={"n_samples": 0},
        )

    return EvaluationResult(
        metric_name="faithfulness",
        value=float(np.mean(scores)),
        confidence_interval=_bootstrap_ci(scores),
        metadata={"n_samples": len(scores)},
    )


def answer_relevance_metric(predictions: list[dict]) -> EvaluationResult:
    """Measure relevance: word overlap between answer and query."""
    scores = []

    for pred in predictions:
        query = pred.get("query", "")
        result = pred.get("predicted")
        if not result or not hasattr(result, "answer") or not query:
            continue

        answer_words = set(result.answer.answer.lower().split())
        query_words = set(query.lower().split())

        if not answer_words or not query_words:
            scores.append(0.0)
            continue

        overlap = len(answer_words & query_words) / len(query_words)
        scores.append(min(1.0, overlap))

    if not scores:
        return EvaluationResult(
            metric_name="relevance",
            value=0.0,
            confidence_interval=(0.0, 0.0),
            metadata={"n_samples": 0},
        )

    return EvaluationResult(
        metric_name="relevance",
        value=float(np.mean(scores)),
        confidence_interval=_bootstrap_ci(scores),
        metadata={"n_samples": len(scores)},
    )


def context_precision_metric(predictions: list[dict]) -> EvaluationResult:
    """Fraction of retrieved docs that are in the expected relevant set."""
    scores = []

    for pred in predictions:
        expected = set(pred.get("expected_docs", []))
        if not expected:
            continue

        result = pred.get("predicted")
        if not result or not hasattr(result, "retrieval_results"):
            scores.append(0.0)
            continue

        retrieved = [r.document.id for r in result.retrieval_results]
        if not retrieved:
            scores.append(0.0)
            continue

        precision = len(set(retrieved) & expected) / len(retrieved)
        scores.append(precision)

    if not scores:
        return EvaluationResult(
            metric_name="context_precision",
            value=0.0,
            confidence_interval=(0.0, 0.0),
            metadata={"n_samples": 0},
        )

    return EvaluationResult(
        metric_name="context_precision",
        value=float(np.mean(scores)),
        confidence_interval=_bootstrap_ci(scores),
        metadata={"n_samples": len(scores)},
    )


def context_recall_metric(predictions: list[dict]) -> EvaluationResult:
    """Fraction of expected relevant docs found in retrieved set."""
    scores = []

    for pred in predictions:
        expected = set(pred.get("expected_docs", []))
        if not expected:
            continue

        result = pred.get("predicted")
        if not result or not hasattr(result, "retrieval_results"):
            scores.append(0.0)
            continue

        retrieved = set(r.document.id for r in result.retrieval_results)
        recall = len(expected & retrieved) / len(expected)
        scores.append(recall)

    if not scores:
        return EvaluationResult(
            metric_name="context_recall",
            value=0.0,
            confidence_interval=(0.0, 0.0),
            metadata={"n_samples": 0},
        )

    return EvaluationResult(
        metric_name="context_recall",
        value=float(np.mean(scores)),
        confidence_interval=_bootstrap_ci(scores),
        metadata={"n_samples": len(scores)},
    )


def answer_correctness_metric(predictions: list[dict]) -> EvaluationResult:
    """Word overlap between predicted answer and expected answer."""
    scores = []

    for pred in predictions:
        expected = pred.get("expected", "")
        result = pred.get("predicted")
        if not result or not hasattr(result, "answer") or not expected:
            continue

        answer_words = set(result.answer.answer.lower().split())
        expected_words = set(expected.lower().split())

        if not expected_words:
            continue

        intersection = len(answer_words & expected_words)
        union = len(answer_words | expected_words)
        f1 = intersection / union if union > 0 else 0.0
        scores.append(f1)

    if not scores:
        return EvaluationResult(
            metric_name="answer_correctness",
            value=0.0,
            confidence_interval=(0.0, 0.0),
            metadata={"n_samples": 0},
        )

    return EvaluationResult(
        metric_name="answer_correctness",
        value=float(np.mean(scores)),
        confidence_interval=_bootstrap_ci(scores),
        metadata={"n_samples": len(scores)},
    )


# Helper Functions

def _bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> tuple[float, float]:
    """Compute bootstrap confidence interval."""
    if not values:
        return (0.0, 0.0)

    values = np.array(values)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(sample))

    lower = np.percentile(bootstrap_means, (1 - confidence) / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 + confidence) / 2 * 100)

    return (float(lower), float(upper))
