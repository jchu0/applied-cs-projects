package tracing

import (
	"context"
	"fmt"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/jaeger"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.21.0"
	"go.opentelemetry.io/otel/trace"
)

// Config holds tracing configuration
type Config struct {
	ServiceName    string
	Environment    string
	JaegerEndpoint string
	SamplingRate   float64
	Enabled        bool
}

// Tracer wraps OpenTelemetry tracer
type Tracer struct {
	provider *sdktrace.TracerProvider
	tracer   trace.Tracer
	config   Config
}

// NewTracer creates a new tracer instance
func NewTracer(cfg Config) (*Tracer, error) {
	if !cfg.Enabled {
		return &Tracer{
			config: cfg,
			tracer: otel.Tracer(cfg.ServiceName),
		}, nil
	}

	// Create Jaeger exporter
	exporter, err := jaeger.New(
		jaeger.WithCollectorEndpoint(jaeger.WithEndpoint(cfg.JaegerEndpoint)),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create jaeger exporter: %w", err)
	}

	// Create resource with service info
	res, err := resource.Merge(
		resource.Default(),
		resource.NewWithAttributes(
			semconv.SchemaURL,
			semconv.ServiceName(cfg.ServiceName),
			semconv.DeploymentEnvironment(cfg.Environment),
		),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create resource: %w", err)
	}

	// Configure sampler
	var sampler sdktrace.Sampler
	if cfg.SamplingRate >= 1.0 {
		sampler = sdktrace.AlwaysSample()
	} else if cfg.SamplingRate <= 0 {
		sampler = sdktrace.NeverSample()
	} else {
		sampler = sdktrace.TraceIDRatioBased(cfg.SamplingRate)
	}

	// Create trace provider
	provider := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exporter),
		sdktrace.WithResource(res),
		sdktrace.WithSampler(sampler),
	)

	// Set global provider and propagator
	otel.SetTracerProvider(provider)
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
		propagation.TraceContext{},
		propagation.Baggage{},
	))

	return &Tracer{
		provider: provider,
		tracer:   provider.Tracer(cfg.ServiceName),
		config:   cfg,
	}, nil
}

// Shutdown shuts down the tracer
func (t *Tracer) Shutdown(ctx context.Context) error {
	if t.provider != nil {
		return t.provider.Shutdown(ctx)
	}
	return nil
}

// StartSpan starts a new span
func (t *Tracer) StartSpan(ctx context.Context, name string, opts ...trace.SpanStartOption) (context.Context, trace.Span) {
	return t.tracer.Start(ctx, name, opts...)
}

// SpanFromContext returns the span from context
func SpanFromContext(ctx context.Context) trace.Span {
	return trace.SpanFromContext(ctx)
}

// AddEvent adds an event to the current span
func AddEvent(ctx context.Context, name string, attrs ...attribute.KeyValue) {
	span := trace.SpanFromContext(ctx)
	span.AddEvent(name, trace.WithAttributes(attrs...))
}

// SetAttributes sets attributes on the current span
func SetAttributes(ctx context.Context, attrs ...attribute.KeyValue) {
	span := trace.SpanFromContext(ctx)
	span.SetAttributes(attrs...)
}

// RecordError records an error on the current span
func RecordError(ctx context.Context, err error, opts ...trace.EventOption) {
	span := trace.SpanFromContext(ctx)
	span.RecordError(err, opts...)
}

// TraceID returns the trace ID from context
func TraceID(ctx context.Context) string {
	span := trace.SpanFromContext(ctx)
	if span.SpanContext().HasTraceID() {
		return span.SpanContext().TraceID().String()
	}
	return ""
}

// SpanID returns the span ID from context
func SpanID(ctx context.Context) string {
	span := trace.SpanFromContext(ctx)
	if span.SpanContext().HasSpanID() {
		return span.SpanContext().SpanID().String()
	}
	return ""
}

// Common attribute helpers
func ServiceAttr(name string) attribute.KeyValue {
	return attribute.String("service.name", name)
}

func OperationAttr(op string) attribute.KeyValue {
	return attribute.String("operation", op)
}

func UserIDAttr(id string) attribute.KeyValue {
	return attribute.String("user.id", id)
}

func TenantIDAttr(id string) attribute.KeyValue {
	return attribute.String("tenant.id", id)
}

func ErrorAttr(err error) attribute.KeyValue {
	return attribute.String("error.message", err.Error())
}

func StatusAttr(status string) attribute.KeyValue {
	return attribute.String("status", status)
}
