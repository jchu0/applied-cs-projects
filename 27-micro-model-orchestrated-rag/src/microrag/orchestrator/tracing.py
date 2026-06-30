"""Distributed tracing for pipeline execution."""

from contextlib import contextmanager
from typing import Any
import time
import json

from ..schemas import Span, generate_id


class TraceExporter:
    """Base class for trace exporters."""

    def export(self, span: Span):
        """Export a span."""
        raise NotImplementedError


class ConsoleExporter(TraceExporter):
    """Export traces to console."""

    def export(self, span: Span):
        print(f"[TRACE] {span.name}: {span.duration_ms:.2f}ms")


class JSONExporter(TraceExporter):
    """Export traces to JSON file."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.spans = []

    def export(self, span: Span):
        self.spans.append({
            "trace_id": span.trace_id,
            "span_id": span.span_id,
            "parent_id": span.parent_id,
            "name": span.name,
            "start_time": span.start_time,
            "end_time": span.end_time,
            "duration_ms": span.duration_ms,
            "attributes": span.attributes
        })

    def flush(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.spans, f, indent=2)


class InMemoryExporter(TraceExporter):
    """Export traces to memory for testing."""

    def __init__(self):
        self.spans: list[Span] = []

    def export(self, span: Span):
        self.spans.append(span)

    def get_traces(self) -> list[Span]:
        return self.spans

    def clear(self):
        self.spans.clear()


class PipelineTracer:
    """Distributed tracing for pipeline execution."""

    def __init__(self, exporter: TraceExporter = None):
        """Initialize tracer.

        Args:
            exporter: Trace exporter (defaults to console)
        """
        self.exporter = exporter or ConsoleExporter()
        self._active_spans: dict[str, Span] = {}

    def start_trace(self) -> str:
        """Start a new trace.

        Returns:
            Trace ID
        """
        return generate_id()

    @contextmanager
    def trace_step(
        self,
        step_name: str,
        trace_id: str,
        parent_id: str = None
    ):
        """Context manager for tracing a step.

        Args:
            step_name: Name of the step
            trace_id: Trace ID
            parent_id: Optional parent span ID

        Yields:
            Span for the step
        """
        span_id = generate_id()
        start_time = time.time()

        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_id=parent_id,
            name=step_name,
            start_time=start_time,
            attributes={}
        )

        self._active_spans[span_id] = span

        try:
            yield span
        finally:
            span.end_time = time.time()
            span.duration_ms = (span.end_time - span.start_time) * 1000
            self.exporter.export(span)
            del self._active_spans[span_id]

    def add_attribute(self, span: Span, key: str, value: Any):
        """Add attribute to span.

        Args:
            span: Span to modify
            key: Attribute key
            value: Attribute value
        """
        span.attributes[key] = value

    def record_error(self, span: Span, error: Exception):
        """Record an error on a span.

        Args:
            span: Span to modify
            error: Exception that occurred
        """
        span.attributes["error"] = True
        span.attributes["error_type"] = type(error).__name__
        span.attributes["error_message"] = str(error)

    def get_active_spans(self) -> list[Span]:
        """Get currently active spans."""
        return list(self._active_spans.values())


class TracingContext:
    """Context manager for full pipeline tracing."""

    def __init__(self, tracer: PipelineTracer):
        self.tracer = tracer
        self.trace_id = None
        self.root_span = None

    async def __aenter__(self):
        self.trace_id = self.tracer.start_trace()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    @contextmanager
    def step(self, name: str, parent_id: str = None):
        """Trace a step within this context.

        Args:
            name: Step name
            parent_id: Parent span ID

        Yields:
            Span for the step
        """
        with self.tracer.trace_step(name, self.trace_id, parent_id) as span:
            yield span


# Global tracer instance
_default_tracer = None


def get_tracer() -> PipelineTracer:
    """Get default tracer instance."""
    global _default_tracer
    if _default_tracer is None:
        _default_tracer = PipelineTracer(InMemoryExporter())
    return _default_tracer


def set_tracer(tracer: PipelineTracer):
    """Set default tracer instance."""
    global _default_tracer
    _default_tracer = tracer
