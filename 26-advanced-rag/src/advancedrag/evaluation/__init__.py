"""Evaluation framework for Advanced RAG system."""

from .evaluator import (
    EvaluationResult,
    EvaluationReport,
    RAGEvaluator,
    retrieval_recall_at_k,
    retrieval_precision_at_k,
    retrieval_mrr,
    retrieval_ndcg,
    answer_faithfulness_metric,
    answer_relevance_metric,
    context_precision_metric,
    context_recall_metric,
    answer_correctness_metric,
)
from .ab_testing import (
    ABTest,
    ABTestAnalysis,
    ABTestManager,
    InMemoryTestStore,
)

__all__ = [
    # Evaluator
    "EvaluationResult",
    "EvaluationReport",
    "RAGEvaluator",
    # Metrics
    "retrieval_recall_at_k",
    "retrieval_precision_at_k",
    "retrieval_mrr",
    "retrieval_ndcg",
    # LLM-judged / proxy metrics
    "answer_faithfulness_metric",
    "answer_relevance_metric",
    "context_precision_metric",
    "context_recall_metric",
    "answer_correctness_metric",
    # A/B Testing
    "ABTest",
    "ABTestAnalysis",
    "ABTestManager",
    "InMemoryTestStore",
]
