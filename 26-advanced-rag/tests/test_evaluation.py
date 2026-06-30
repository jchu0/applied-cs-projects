"""Tests for evaluation metrics module."""

import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock
from datetime import datetime

from advancedrag.schemas import (
    Document,
    RetrievalResult,
    RerankResult,
    GeneratedAnswer,
    Citation,
    RAGResult,
    ConstructedContext,
)
from advancedrag.evaluation.evaluator import (
    RAGEvaluator,
    EvaluationResult,
    EvaluationReport,
    retrieval_recall_at_k,
    retrieval_precision_at_k,
    retrieval_mrr,
    retrieval_ndcg,
    answer_length_metric,
    citation_count_metric,
    confidence_metric,
    answer_faithfulness_metric,
    answer_relevance_metric,
    context_precision_metric,
    context_recall_metric,
    answer_correctness_metric,
    _bootstrap_ci,
)


class TestRetrievalRecallAtK:
    """Test retrieval recall@k metric."""

    @pytest.fixture
    def mock_rag_result(self):
        """Create mock RAG result with retrieval results."""
        def create_result(retrieved_ids: list[str]) -> Mock:
            result = Mock()
            result.retrieval_results = [
                RetrievalResult(
                    document=Document(id=doc_id, content="", metadata={}),
                    score=0.9 - i * 0.1,
                    retriever_type="test",
                    rank=i
                )
                for i, doc_id in enumerate(retrieved_ids)
            ]
            return result
        return create_result

    def test_perfect_recall(self, mock_rag_result):
        """Test perfect recall when all relevant docs are retrieved."""
        predictions = [
            {
                "query": "test query",
                "expected_docs": ["doc1", "doc2"],
                "predicted": mock_rag_result(["doc1", "doc2", "doc3"])
            }
        ]

        result = retrieval_recall_at_k(predictions, k=5)

        assert result.value == 1.0
        assert result.metric_name == "recall@5"

    def test_partial_recall(self, mock_rag_result):
        """Test partial recall when some relevant docs are retrieved."""
        predictions = [
            {
                "query": "test query",
                "expected_docs": ["doc1", "doc2"],
                "predicted": mock_rag_result(["doc1", "doc3", "doc4"])
            }
        ]

        result = retrieval_recall_at_k(predictions, k=5)

        assert result.value == 0.5  # 1 out of 2 retrieved

    def test_zero_recall(self, mock_rag_result):
        """Test zero recall when no relevant docs are retrieved."""
        predictions = [
            {
                "query": "test query",
                "expected_docs": ["doc1", "doc2"],
                "predicted": mock_rag_result(["doc3", "doc4", "doc5"])
            }
        ]

        result = retrieval_recall_at_k(predictions, k=5)

        assert result.value == 0.0

    def test_recall_respects_k(self, mock_rag_result):
        """Test recall only considers top-k results."""
        predictions = [
            {
                "query": "test query",
                "expected_docs": ["doc3"],  # doc3 is at position 3 (0-indexed: 2)
                "predicted": mock_rag_result(["doc1", "doc2", "doc3"])
            }
        ]

        # With k=2, doc3 should not be considered
        result_k2 = retrieval_recall_at_k(predictions, k=2)
        assert result_k2.value == 0.0

        # With k=5, doc3 should be considered
        result_k5 = retrieval_recall_at_k(predictions, k=5)
        assert result_k5.value == 1.0

    def test_empty_predictions(self):
        """Test with empty predictions."""
        result = retrieval_recall_at_k([], k=5)

        assert result.value == 0.0
        assert result.metadata["n_samples"] == 0

    def test_multiple_predictions(self, mock_rag_result):
        """Test recall averaged over multiple predictions."""
        predictions = [
            {
                "query": "query 1",
                "expected_docs": ["doc1"],
                "predicted": mock_rag_result(["doc1"])  # Recall = 1.0
            },
            {
                "query": "query 2",
                "expected_docs": ["doc2"],
                "predicted": mock_rag_result(["doc3"])  # Recall = 0.0
            }
        ]

        result = retrieval_recall_at_k(predictions, k=5)

        assert result.value == 0.5  # Average of 1.0 and 0.0

    def test_confidence_interval(self, mock_rag_result):
        """Test confidence interval is computed."""
        predictions = [
            {
                "query": f"query {i}",
                "expected_docs": ["doc1"],
                "predicted": mock_rag_result(["doc1"] if i % 2 == 0 else ["doc2"])
            }
            for i in range(20)
        ]

        result = retrieval_recall_at_k(predictions, k=5)

        assert result.confidence_interval[0] <= result.value
        assert result.confidence_interval[1] >= result.value


class TestRetrievalPrecisionAtK:
    """Test retrieval precision@k metric."""

    @pytest.fixture
    def mock_rag_result(self):
        """Create mock RAG result."""
        def create_result(retrieved_ids: list[str]) -> Mock:
            result = Mock()
            result.retrieval_results = [
                RetrievalResult(
                    document=Document(id=doc_id, content="", metadata={}),
                    score=0.9,
                    retriever_type="test",
                    rank=i
                )
                for i, doc_id in enumerate(retrieved_ids)
            ]
            return result
        return create_result

    def test_perfect_precision(self, mock_rag_result):
        """Test perfect precision when all retrieved are relevant."""
        predictions = [
            {
                "query": "test query",
                "expected_docs": ["doc1", "doc2", "doc3"],
                "predicted": mock_rag_result(["doc1", "doc2"])
            }
        ]

        result = retrieval_precision_at_k(predictions, k=5)

        assert result.value == 1.0

    def test_partial_precision(self, mock_rag_result):
        """Test partial precision."""
        predictions = [
            {
                "query": "test query",
                "expected_docs": ["doc1"],
                "predicted": mock_rag_result(["doc1", "doc2"])
            }
        ]

        result = retrieval_precision_at_k(predictions, k=5)

        assert result.value == 0.5  # 1 relevant out of 2 retrieved

    def test_zero_precision(self, mock_rag_result):
        """Test zero precision when no retrieved are relevant."""
        predictions = [
            {
                "query": "test query",
                "expected_docs": ["doc1"],
                "predicted": mock_rag_result(["doc2", "doc3"])
            }
        ]

        result = retrieval_precision_at_k(predictions, k=5)

        assert result.value == 0.0


class TestRetrievalMRR:
    """Test Mean Reciprocal Rank metric."""

    @pytest.fixture
    def mock_rag_result(self):
        """Create mock RAG result."""
        def create_result(retrieved_ids: list[str]) -> Mock:
            result = Mock()
            result.retrieval_results = [
                RetrievalResult(
                    document=Document(id=doc_id, content="", metadata={}),
                    score=0.9,
                    retriever_type="test",
                    rank=i
                )
                for i, doc_id in enumerate(retrieved_ids)
            ]
            return result
        return create_result

    def test_mrr_first_position(self, mock_rag_result):
        """Test MRR when relevant doc is first."""
        predictions = [
            {
                "query": "test query",
                "expected_docs": ["doc1"],
                "predicted": mock_rag_result(["doc1", "doc2", "doc3"])
            }
        ]

        result = retrieval_mrr(predictions)

        assert result.value == 1.0  # 1/1

    def test_mrr_second_position(self, mock_rag_result):
        """Test MRR when relevant doc is second."""
        predictions = [
            {
                "query": "test query",
                "expected_docs": ["doc2"],
                "predicted": mock_rag_result(["doc1", "doc2", "doc3"])
            }
        ]

        result = retrieval_mrr(predictions)

        assert result.value == 0.5  # 1/2

    def test_mrr_third_position(self, mock_rag_result):
        """Test MRR when relevant doc is third."""
        predictions = [
            {
                "query": "test query",
                "expected_docs": ["doc3"],
                "predicted": mock_rag_result(["doc1", "doc2", "doc3"])
            }
        ]

        result = retrieval_mrr(predictions)

        assert abs(result.value - 1/3) < 0.01

    def test_mrr_not_found(self, mock_rag_result):
        """Test MRR when relevant doc is not found."""
        predictions = [
            {
                "query": "test query",
                "expected_docs": ["doc4"],
                "predicted": mock_rag_result(["doc1", "doc2", "doc3"])
            }
        ]

        result = retrieval_mrr(predictions)

        assert result.value == 0.0

    def test_mrr_multiple_relevant_uses_first(self, mock_rag_result):
        """Test MRR uses rank of first relevant doc."""
        predictions = [
            {
                "query": "test query",
                "expected_docs": ["doc2", "doc3"],  # Both relevant
                "predicted": mock_rag_result(["doc1", "doc2", "doc3"])
            }
        ]

        result = retrieval_mrr(predictions)

        assert result.value == 0.5  # 1/2 (doc2 at position 2)


class TestRetrievalNDCG:
    """Test Normalized Discounted Cumulative Gain metric."""

    @pytest.fixture
    def mock_rag_result(self):
        """Create mock RAG result."""
        def create_result(retrieved_ids: list[str]) -> Mock:
            result = Mock()
            result.retrieval_results = [
                RetrievalResult(
                    document=Document(id=doc_id, content="", metadata={}),
                    score=0.9,
                    retriever_type="test",
                    rank=i
                )
                for i, doc_id in enumerate(retrieved_ids)
            ]
            return result
        return create_result

    def test_perfect_ndcg(self, mock_rag_result):
        """Test perfect NDCG when all relevant docs are at top."""
        predictions = [
            {
                "query": "test query",
                "expected_docs": ["doc1", "doc2"],
                "predicted": mock_rag_result(["doc1", "doc2", "doc3"])
            }
        ]

        result = retrieval_ndcg(predictions, k=5)

        assert result.value == 1.0

    def test_ndcg_respects_k(self, mock_rag_result):
        """Test NDCG respects k parameter."""
        predictions = [
            {
                "query": "test query",
                "expected_docs": ["doc3"],  # Only at position 3
                "predicted": mock_rag_result(["doc1", "doc2", "doc3"])
            }
        ]

        result_k2 = retrieval_ndcg(predictions, k=2)
        result_k5 = retrieval_ndcg(predictions, k=5)

        assert result_k2.value == 0.0
        assert result_k5.value > 0.0

    def test_ndcg_formula(self, mock_rag_result):
        """Test NDCG formula is correctly applied."""
        predictions = [
            {
                "query": "test query",
                "expected_docs": ["doc1"],
                "predicted": mock_rag_result(["doc2", "doc1", "doc3"])  # doc1 at position 2
            }
        ]

        result = retrieval_ndcg(predictions, k=5)

        # DCG = 1/log2(2+1) = 1/log2(3) = 0.631
        # IDCG = 1/log2(0+2) = 1/log2(2) = 1.0
        # NDCG = DCG/IDCG = 0.631
        expected_ndcg = (1 / np.log2(3)) / (1 / np.log2(2))
        assert abs(result.value - expected_ndcg) < 0.01

    def test_zero_ndcg(self, mock_rag_result):
        """Test zero NDCG when no relevant docs retrieved."""
        predictions = [
            {
                "query": "test query",
                "expected_docs": ["doc4"],
                "predicted": mock_rag_result(["doc1", "doc2", "doc3"])
            }
        ]

        result = retrieval_ndcg(predictions, k=5)

        assert result.value == 0.0


class TestAnswerQualityMetrics:
    """Test answer quality metrics."""

    @pytest.fixture
    def mock_rag_result_with_answer(self):
        """Create mock RAG result with answer."""
        def create_result(answer_text: str, citations: list[str], confidence: float) -> Mock:
            result = Mock()
            result.answer = Mock()
            result.answer.answer = answer_text
            result.answer.citations = citations
            result.answer.confidence = confidence
            return result
        return create_result

    def test_answer_length_metric(self, mock_rag_result_with_answer):
        """Test answer length metric."""
        predictions = [
            {
                "query": "test query",
                "predicted": mock_rag_result_with_answer(
                    "This is a five word answer.",
                    [], 0.9
                )
            }
        ]

        result = answer_length_metric(predictions)

        assert result.value == 6  # 6 words
        assert result.metadata["unit"] == "words"

    def test_answer_length_multiple(self, mock_rag_result_with_answer):
        """Test answer length averaged over multiple answers."""
        predictions = [
            {
                "query": "q1",
                "predicted": mock_rag_result_with_answer("Two words.", [], 0.9)
            },
            {
                "query": "q2",
                "predicted": mock_rag_result_with_answer("This has four words.", [], 0.9)
            }
        ]

        result = answer_length_metric(predictions)

        assert result.value == 3.0  # Average of 2 and 4

    def test_citation_count_metric(self, mock_rag_result_with_answer):
        """Test citation count metric."""
        citations = [
            Citation(source_id="s1", source_title="T1", quoted_text="Q1", relevance_score=0.9),
            Citation(source_id="s2", source_title="T2", quoted_text="Q2", relevance_score=0.8),
        ]

        result_mock = Mock()
        result_mock.answer = Mock()
        result_mock.answer.citations = citations

        predictions = [{"query": "q", "predicted": result_mock}]

        result = citation_count_metric(predictions)

        assert result.value == 2.0

    def test_confidence_metric(self, mock_rag_result_with_answer):
        """Test confidence metric."""
        predictions = [
            {
                "query": "q1",
                "predicted": mock_rag_result_with_answer("Answer", [], 0.9)
            },
            {
                "query": "q2",
                "predicted": mock_rag_result_with_answer("Answer", [], 0.7)
            }
        ]

        result = confidence_metric(predictions)

        assert result.value == 0.8  # Average of 0.9 and 0.7


class TestBootstrapCI:
    """Test bootstrap confidence interval computation."""

    def test_bootstrap_empty_values(self):
        """Test bootstrap with empty values."""
        ci = _bootstrap_ci([])

        assert ci == (0.0, 0.0)

    def test_bootstrap_single_value(self):
        """Test bootstrap with single value."""
        ci = _bootstrap_ci([0.5])

        assert ci[0] == 0.5
        assert ci[1] == 0.5

    def test_bootstrap_interval_contains_mean(self):
        """Test CI contains the mean."""
        values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        mean = np.mean(values)

        ci = _bootstrap_ci(values)

        assert ci[0] <= mean
        assert ci[1] >= mean

    def test_bootstrap_interval_bounds(self):
        """Test CI bounds are reasonable."""
        values = [0.5] * 100

        ci = _bootstrap_ci(values)

        # With identical values, CI should be tight
        assert ci[0] == 0.5
        assert ci[1] == 0.5


class TestRAGEvaluator:
    """Test RAGEvaluator class."""

    @pytest.fixture
    def evaluator(self):
        """Create RAG evaluator."""
        return RAGEvaluator()

    def test_default_metrics_registered(self, evaluator):
        """Test default metrics are registered."""
        assert "recall@5" in evaluator.metrics
        assert "recall@10" in evaluator.metrics
        assert "mrr" in evaluator.metrics
        assert "ndcg@10" in evaluator.metrics
        assert "precision@5" in evaluator.metrics
        assert "answer_length" in evaluator.metrics
        assert "citation_count" in evaluator.metrics
        assert "confidence" in evaluator.metrics

    def test_register_custom_metric(self, evaluator):
        """Test registering custom metric."""
        def custom_metric(predictions):
            return EvaluationResult(
                metric_name="custom",
                value=1.0,
                confidence_interval=(1.0, 1.0),
                metadata={}
            )

        evaluator.register_metric("custom", custom_metric)

        assert "custom" in evaluator.metrics

    @pytest.mark.asyncio
    async def test_evaluate_with_predictions(self, evaluator):
        """Test evaluation with pre-computed predictions."""
        # Create mock RAG result
        mock_result = Mock()
        mock_result.retrieval_results = [
            RetrievalResult(
                document=Document(id="doc1", content="", metadata={}),
                score=0.9,
                retriever_type="test",
                rank=0
            )
        ]
        mock_result.answer = Mock()
        mock_result.answer.answer = "Test answer"
        mock_result.answer.citations = []
        mock_result.answer.confidence = 0.9

        test_set = [
            {
                "query": "test query",
                "expected_answer": "expected",
                "relevant_docs": ["doc1"],
                "predicted": mock_result
            }
        ]

        report = await evaluator.evaluate(test_set, metrics=["recall@5", "mrr"])

        assert isinstance(report, EvaluationReport)
        assert "recall@5" in report.metrics
        assert "mrr" in report.metrics
        assert report.sample_count == 1

    @pytest.mark.asyncio
    async def test_evaluate_with_pipeline(self):
        """Test evaluation with RAG pipeline."""
        # Create mock pipeline
        mock_pipeline = Mock()
        mock_result = Mock()
        mock_result.retrieval_results = []
        mock_result.answer = Mock()
        mock_result.answer.answer = "Answer"
        mock_result.answer.citations = []
        mock_result.answer.confidence = 0.8

        mock_pipeline.query = AsyncMock(return_value=mock_result)

        evaluator = RAGEvaluator(rag_pipeline=mock_pipeline)

        test_set = [
            {
                "query": "test query",
                "expected_answer": "expected",
                "relevant_docs": []
            }
        ]

        report = await evaluator.evaluate(test_set, metrics=["confidence"])

        assert report.metrics["confidence"].value == 0.8

    @pytest.mark.asyncio
    async def test_evaluate_handles_metric_errors(self, evaluator):
        """Test evaluation handles metric computation errors."""
        def failing_metric(predictions):
            raise ValueError("Metric computation failed")

        evaluator.register_metric("failing", failing_metric)

        test_set = [{"query": "test", "predicted": None}]

        report = await evaluator.evaluate(test_set, metrics=["failing"])

        assert report.metrics["failing"].value == 0.0
        assert "error" in report.metrics["failing"].metadata

    @pytest.mark.asyncio
    async def test_compare_configs(self, evaluator):
        """Test comparing multiple configurations."""
        mock_result = Mock()
        mock_result.retrieval_results = []
        mock_result.answer = Mock()
        mock_result.answer.answer = "Answer"
        mock_result.answer.citations = []
        mock_result.answer.confidence = 0.8

        test_set = [
            {
                "query": "test",
                "relevant_docs": [],
                "predicted": mock_result
            }
        ]

        configs = [
            {"name": "config_a"},
            {"name": "config_b"}
        ]

        results = await evaluator.compare_configs(test_set, configs, metrics=["confidence"])

        assert "config_a" in results
        assert "config_b" in results


class TestEvaluationResult:
    """Test EvaluationResult data structure."""

    def test_evaluation_result_creation(self):
        """Test creating EvaluationResult."""
        result = EvaluationResult(
            metric_name="recall@10",
            value=0.85,
            confidence_interval=(0.80, 0.90),
            metadata={"k": 10, "n_samples": 100}
        )

        assert result.metric_name == "recall@10"
        assert result.value == 0.85
        assert result.confidence_interval == (0.80, 0.90)
        assert result.metadata["k"] == 10


class TestEvaluationReport:
    """Test EvaluationReport data structure."""

    def test_evaluation_report_creation(self):
        """Test creating EvaluationReport."""
        metrics = {
            "recall@10": EvaluationResult(
                metric_name="recall@10",
                value=0.85,
                confidence_interval=(0.80, 0.90),
                metadata={}
            )
        }

        report = EvaluationReport(
            eval_id="eval_123",
            timestamp=datetime.utcnow(),
            metrics=metrics,
            config={"model": "test"},
            sample_count=100
        )

        assert report.eval_id == "eval_123"
        assert report.sample_count == 100
        assert "recall@10" in report.metrics


class TestMetricsWithNoResults:
    """Test metrics when no results are available."""

    def test_recall_no_expected_docs(self):
        """Test recall skips samples without expected docs."""
        predictions = [
            {
                "query": "test",
                "expected_docs": [],  # No expected docs
                "predicted": Mock(retrieval_results=[])
            }
        ]

        result = retrieval_recall_at_k(predictions, k=5)

        assert result.metadata["n_samples"] == 0

    def test_precision_empty_retrieved(self):
        """Test precision with empty retrieval."""
        predictions = [
            {
                "query": "test",
                "expected_docs": ["doc1"],
                "predicted": Mock(retrieval_results=[])
            }
        ]

        result = retrieval_precision_at_k(predictions, k=5)

        assert result.value == 0.0

    def test_mrr_no_retrieval_results(self):
        """Test MRR with no retrieval results."""
        mock_result = Mock()
        mock_result.retrieval_results = []

        predictions = [
            {
                "query": "test",
                "expected_docs": ["doc1"],
                "predicted": mock_result
            }
        ]

        result = retrieval_mrr(predictions)

        assert result.value == 0.0


class TestFaithfulnessMetric:
    """Test answer faithfulness metric."""

    def _make_pred(self, answer_text, context_text):
        result = Mock()
        result.answer = Mock()
        result.answer.answer = answer_text
        result.context = Mock()
        result.context.content = context_text
        return {"query": "test", "predicted": result}

    def test_high_overlap(self):
        """High word overlap between answer and context should give high faithfulness."""
        preds = [self._make_pred(
            "Machine learning processes large datasets efficiently",
            "Machine learning processes large datasets efficiently and accurately",
        )]
        result = answer_faithfulness_metric(preds)
        assert result.value > 0.7

    def test_low_overlap(self):
        """Low word overlap should give low faithfulness."""
        preds = [self._make_pred(
            "Quantum computing revolutionizes cryptography",
            "Dogs and cats are popular household pets",
        )]
        result = answer_faithfulness_metric(preds)
        assert result.value < 0.3

    def test_no_predictions(self):
        """Empty predictions should return 0.0."""
        result = answer_faithfulness_metric([])
        assert result.value == 0.0
        assert result.metadata["n_samples"] == 0


class TestRelevanceMetric:
    """Test answer relevance metric."""

    def _make_pred(self, query, answer_text):
        result = Mock()
        result.answer = Mock()
        result.answer.answer = answer_text
        return {"query": query, "predicted": result}

    def test_relevant_answer(self):
        """Answer containing query terms should have high relevance."""
        preds = [self._make_pred(
            "machine learning",
            "Machine learning is a powerful approach for data analysis and learning from patterns",
        )]
        result = answer_relevance_metric(preds)
        assert result.value > 0.5

    def test_irrelevant_answer(self):
        """Answer without query terms should have low relevance."""
        preds = [self._make_pred(
            "quantum physics",
            "Dogs and cats are popular pets in many households",
        )]
        result = answer_relevance_metric(preds)
        assert result.value == 0.0

    def test_empty_predictions(self):
        """Empty predictions should return 0.0."""
        result = answer_relevance_metric([])
        assert result.value == 0.0


class TestContextPrecisionMetric:
    """Test context precision metric."""

    def _make_pred(self, expected_docs, retrieved_ids):
        result = Mock()
        result.retrieval_results = [
            RetrievalResult(
                document=Document(id=doc_id, content="", metadata={}),
                score=0.9, retriever_type="test", rank=i,
            )
            for i, doc_id in enumerate(retrieved_ids)
        ]
        return {"query": "test", "expected_docs": expected_docs, "predicted": result}

    def test_all_relevant(self):
        """All retrieved docs in expected set should give 1.0."""
        preds = [self._make_pred(["d1", "d2", "d3"], ["d1", "d2"])]
        result = context_precision_metric(preds)
        assert result.value == 1.0

    def test_none_relevant(self):
        """No retrieved docs in expected set should give 0.0."""
        preds = [self._make_pred(["d1"], ["d2", "d3"])]
        result = context_precision_metric(preds)
        assert result.value == 0.0

    def test_partial_relevant(self):
        """Some retrieved docs in expected set should give partial score."""
        preds = [self._make_pred(["d1", "d2"], ["d1", "d3"])]
        result = context_precision_metric(preds)
        assert result.value == 0.5


class TestContextRecallMetric:
    """Test context recall metric."""

    def _make_pred(self, expected_docs, retrieved_ids):
        result = Mock()
        result.retrieval_results = [
            RetrievalResult(
                document=Document(id=doc_id, content="", metadata={}),
                score=0.9, retriever_type="test", rank=i,
            )
            for i, doc_id in enumerate(retrieved_ids)
        ]
        return {"query": "test", "expected_docs": expected_docs, "predicted": result}

    def test_full_recall(self):
        """All expected docs retrieved should give 1.0."""
        preds = [self._make_pred(["d1", "d2"], ["d1", "d2", "d3"])]
        result = context_recall_metric(preds)
        assert result.value == 1.0

    def test_zero_recall(self):
        """None of expected docs retrieved should give 0.0."""
        preds = [self._make_pred(["d1", "d2"], ["d3", "d4"])]
        result = context_recall_metric(preds)
        assert result.value == 0.0

    def test_partial_recall(self):
        """Some expected docs retrieved should give partial score."""
        preds = [self._make_pred(["d1", "d2"], ["d1", "d3"])]
        result = context_recall_metric(preds)
        assert result.value == 0.5


class TestAnswerCorrectnessMetric:
    """Test answer correctness metric."""

    def _make_pred(self, expected, answer_text):
        result = Mock()
        result.answer = Mock()
        result.answer.answer = answer_text
        return {"query": "test", "expected": expected, "predicted": result}

    def test_exact_match(self):
        """Identical answer and expected should give 1.0."""
        preds = [self._make_pred("the answer is correct", "the answer is correct")]
        result = answer_correctness_metric(preds)
        assert result.value == 1.0

    def test_no_match(self):
        """Completely different answer should give 0.0."""
        preds = [self._make_pred("alpha beta gamma", "delta epsilon zeta")]
        result = answer_correctness_metric(preds)
        assert result.value == 0.0

    def test_partial_match(self):
        """Partially overlapping answer should give partial score."""
        preds = [self._make_pred("machine learning is great", "machine learning is terrible")]
        result = answer_correctness_metric(preds)
        assert 0.0 < result.value < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
