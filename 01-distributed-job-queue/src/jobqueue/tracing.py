"""OpenTelemetry tracing integration for the job queue system."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes
from opentelemetry.trace import Status, StatusCode, Span
import structlog

from jobqueue.models import Task, TaskResult, TaskStatus

logger = structlog.get_logger()

# Global tracer
_tracer: trace.Tracer | None = None


def setup_tracing(
    service_name: str = "jobqueue",
    exporter: Any = None,
) -> trace.Tracer:
    """
    Set up OpenTelemetry tracing.

    Args:
        service_name: Name of the service for tracing
        exporter: Optional span exporter (defaults to console)

    Returns:
        Configured tracer
    """
    global _tracer

    resource = Resource.create({
        ResourceAttributes.SERVICE_NAME: service_name,
        ResourceAttributes.SERVICE_VERSION: "0.1.0",
    })

    provider = TracerProvider(resource=resource)

    # Add exporter
    if exporter is None:
        exporter = ConsoleSpanExporter()

    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(__name__)

    logger.info("Tracing initialized", service_name=service_name)

    return _tracer


def get_tracer() -> trace.Tracer:
    """Get the global tracer, initializing if needed."""
    global _tracer
    if _tracer is None:
        _tracer = setup_tracing()
    return _tracer


@asynccontextmanager
async def trace_task_enqueue(task: Task) -> AsyncGenerator[Span, None]:
    """Trace task enqueue operation."""
    tracer = get_tracer()

    with tracer.start_as_current_span("task.enqueue") as span:
        span.set_attribute("task.id", task.id)
        span.set_attribute("task.name", task.name)
        span.set_attribute("task.queue", task.queue)
        span.set_attribute("task.priority", task.priority)

        if task.idempotency_key:
            span.set_attribute("task.idempotency_key", task.idempotency_key)

        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


@asynccontextmanager
async def trace_task_process(task: Task, worker_id: str) -> AsyncGenerator[Span, None]:
    """Trace task processing operation."""
    tracer = get_tracer()

    with tracer.start_as_current_span("task.process") as span:
        span.set_attribute("task.id", task.id)
        span.set_attribute("task.name", task.name)
        span.set_attribute("task.queue", task.queue)
        span.set_attribute("task.retries", task.retries)
        span.set_attribute("worker.id", worker_id)

        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


def trace_task_complete(span: Span, result: TaskResult) -> None:
    """Add completion information to a span."""
    span.set_attribute("task.status", result.status)

    if result.duration_ms:
        span.set_attribute("task.duration_ms", result.duration_ms)

    if result.error:
        span.set_attribute("task.error", result.error)

    if result.status == TaskStatus.SUCCESS:
        span.set_status(Status(StatusCode.OK))
    else:
        span.set_status(Status(StatusCode.ERROR, result.error or "Task failed"))


@asynccontextmanager
async def trace_broker_operation(operation: str, **attributes: Any) -> AsyncGenerator[Span, None]:
    """Trace a broker operation."""
    tracer = get_tracer()

    with tracer.start_as_current_span(f"broker.{operation}") as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)

        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
