"""Monitoring and metrics for RAG system."""

from typing import Optional
import time
from contextlib import contextmanager

# Try to import prometheus_client, provide mocks if not available
try:
    from prometheus_client import Counter, Histogram, Gauge, Summary
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

    # Mock implementations
    class Counter:
        def __init__(self, *args, **kwargs):
            self._value = 0
            self._labels = {}

        def labels(self, **kwargs):
            return self

        def inc(self, value=1):
            self._value += value

    class Histogram:
        def __init__(self, *args, **kwargs):
            self._values = []

        def labels(self, **kwargs):
            return self

        def observe(self, value):
            self._values.append(value)

        @contextmanager
        def time(self):
            start = time.time()
            yield
            self.observe(time.time() - start)

    class Gauge:
        def __init__(self, *args, **kwargs):
            self._value = 0

        def labels(self, **kwargs):
            return self

        def set(self, value):
            self._value = value

        def inc(self, value=1):
            self._value += value

        def dec(self, value=1):
            self._value -= value

    class Summary:
        def __init__(self, *args, **kwargs):
            pass

        def labels(self, **kwargs):
            return self

        def observe(self, value):
            pass


# Request metrics
REQUEST_COUNT = Counter(
    'rag_requests_total',
    'Total RAG requests',
    ['tenant_id', 'status']
)

REQUEST_LATENCY = Histogram(
    'rag_request_latency_seconds',
    'RAG request latency by stage',
    ['stage'],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# Pipeline stage metrics
RETRIEVAL_LATENCY = Histogram(
    'rag_retrieval_latency_seconds',
    'Retrieval latency',
    ['retriever_type'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

RERANKING_LATENCY = Histogram(
    'rag_reranking_latency_seconds',
    'Reranking latency',
    ['reranker_type'],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

GENERATION_LATENCY = Histogram(
    'rag_generation_latency_seconds',
    'Answer generation latency',
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0]
)

# Result metrics
RETRIEVAL_RESULTS = Histogram(
    'rag_retrieval_results',
    'Number of retrieval results',
    buckets=[5, 10, 25, 50, 100, 200]
)

CONFIDENCE_SCORE = Histogram(
    'rag_confidence_score',
    'Answer confidence scores',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

CITATION_COUNT = Histogram(
    'rag_citation_count',
    'Number of citations per answer',
    buckets=[0, 1, 2, 3, 4, 5, 10]
)

# Quality metrics
HALLUCINATION_COUNT = Counter(
    'rag_hallucinations_total',
    'Detected hallucinations',
    ['severity']
)

# Cache metrics
CACHE_HIT_RATE = Gauge(
    'rag_cache_hit_rate',
    'Cache hit rate',
    ['cache_type']
)

CACHE_SIZE = Gauge(
    'rag_cache_size',
    'Current cache size',
    ['cache_type']
)

# System metrics
ACTIVE_REQUESTS = Gauge(
    'rag_active_requests',
    'Currently processing requests',
    ['tenant_id']
)

ERROR_COUNT = Counter(
    'rag_errors_total',
    'Total errors',
    ['error_type', 'stage']
)


class MetricsCollector:
    """Centralized metrics collection for RAG pipeline."""

    def __init__(self, tenant_id: str = "default"):
        self.tenant_id = tenant_id

    @contextmanager
    def track_request(self):
        """Context manager to track full request lifecycle."""
        ACTIVE_REQUESTS.labels(tenant_id=self.tenant_id).inc()
        start_time = time.time()
        status = "success"

        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            duration = time.time() - start_time
            REQUEST_LATENCY.labels(stage="total").observe(duration)
            REQUEST_COUNT.labels(tenant_id=self.tenant_id, status=status).inc()
            ACTIVE_REQUESTS.labels(tenant_id=self.tenant_id).dec()

    @contextmanager
    def track_retrieval(self, retriever_type: str = "hybrid"):
        """Track retrieval stage."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            RETRIEVAL_LATENCY.labels(retriever_type=retriever_type).observe(duration)
            REQUEST_LATENCY.labels(stage="retrieval").observe(duration)

    @contextmanager
    def track_reranking(self, reranker_type: str = "cross-encoder"):
        """Track reranking stage."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            RERANKING_LATENCY.labels(reranker_type=reranker_type).observe(duration)
            REQUEST_LATENCY.labels(stage="reranking").observe(duration)

    @contextmanager
    def track_generation(self):
        """Track generation stage."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            GENERATION_LATENCY.observe(duration)
            REQUEST_LATENCY.labels(stage="generation").observe(duration)

    def record_retrieval_count(self, count: int):
        """Record number of retrieval results."""
        RETRIEVAL_RESULTS.observe(count)

    def record_confidence(self, score: float):
        """Record answer confidence score."""
        CONFIDENCE_SCORE.observe(score)

    def record_citations(self, count: int):
        """Record citation count."""
        CITATION_COUNT.observe(count)

    def record_hallucination(self, severity: str = "low"):
        """Record detected hallucination."""
        HALLUCINATION_COUNT.labels(severity=severity).inc()

    def record_error(self, error_type: str, stage: str):
        """Record an error."""
        ERROR_COUNT.labels(error_type=error_type, stage=stage).inc()

    def update_cache_metrics(self, cache_type: str, hit_rate: float, size: int):
        """Update cache metrics."""
        CACHE_HIT_RATE.labels(cache_type=cache_type).set(hit_rate)
        CACHE_SIZE.labels(cache_type=cache_type).set(size)


def get_collector(tenant_id: str = "default") -> MetricsCollector:
    """Get a metrics collector for a tenant.

    Args:
        tenant_id: Tenant identifier

    Returns:
        MetricsCollector instance
    """
    return MetricsCollector(tenant_id)
