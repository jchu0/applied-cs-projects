package tracing

import (
	"net/http"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/trace"
)

// HTTPMiddleware returns HTTP middleware for tracing
func HTTPMiddleware(serviceName string) func(http.Handler) http.Handler {
	tracer := otel.Tracer(serviceName)
	propagator := otel.GetTextMapPropagator()

	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Extract trace context from headers
			ctx := propagator.Extract(r.Context(), propagation.HeaderCarrier(r.Header))

			// Start span
			ctx, span := tracer.Start(
				ctx,
				r.Method+" "+r.URL.Path,
				trace.WithSpanKind(trace.SpanKindServer),
				trace.WithAttributes(
					attribute.String("http.method", r.Method),
					attribute.String("http.url", r.URL.String()),
					attribute.String("http.target", r.URL.Path),
					attribute.String("http.host", r.Host),
					attribute.String("http.scheme", getScheme(r)),
					attribute.String("http.user_agent", r.UserAgent()),
					attribute.String("net.peer.ip", r.RemoteAddr),
				),
			)
			defer span.End()

			// Create wrapped response writer to capture status code
			wrapped := &responseWriter{
				ResponseWriter: w,
				statusCode:     http.StatusOK,
			}

			// Update request with traced context
			r = r.WithContext(ctx)

			// Handle request
			next.ServeHTTP(wrapped, r)

			// Record response attributes
			span.SetAttributes(
				attribute.Int("http.status_code", wrapped.statusCode),
				attribute.Int64("http.response_content_length", int64(wrapped.bytesWritten)),
			)

			// Set span status based on HTTP status code
			if wrapped.statusCode >= 400 {
				span.SetStatus(codes.Error, http.StatusText(wrapped.statusCode))
			} else {
				span.SetStatus(codes.Ok, "")
			}
		})
	}
}

// InjectTraceHeaders injects trace context into HTTP headers
func InjectTraceHeaders(ctx context.Context, headers http.Header) {
	otel.GetTextMapPropagator().Inject(ctx, propagation.HeaderCarrier(headers))
}

// ExtractTraceContext extracts trace context from HTTP headers
func ExtractTraceContext(ctx context.Context, headers http.Header) context.Context {
	return otel.GetTextMapPropagator().Extract(ctx, propagation.HeaderCarrier(headers))
}

// responseWriter wraps http.ResponseWriter to capture status code
type responseWriter struct {
	http.ResponseWriter
	statusCode   int
	bytesWritten int
}

func (w *responseWriter) WriteHeader(code int) {
	w.statusCode = code
	w.ResponseWriter.WriteHeader(code)
}

func (w *responseWriter) Write(b []byte) (int, error) {
	n, err := w.ResponseWriter.Write(b)
	w.bytesWritten += n
	return n, err
}

// Unwrap returns the underlying ResponseWriter
func (w *responseWriter) Unwrap() http.ResponseWriter {
	return w.ResponseWriter
}

// getScheme determines the request scheme
func getScheme(r *http.Request) string {
	if r.TLS != nil {
		return "https"
	}
	if scheme := r.Header.Get("X-Forwarded-Proto"); scheme != "" {
		return scheme
	}
	return "http"
}

// RoundTripper wraps http.RoundTripper with tracing
type RoundTripper struct {
	base        http.RoundTripper
	tracer      trace.Tracer
	serviceName string
}

// NewRoundTripper creates a new tracing round tripper
func NewRoundTripper(serviceName string, base http.RoundTripper) *RoundTripper {
	if base == nil {
		base = http.DefaultTransport
	}
	return &RoundTripper{
		base:        base,
		tracer:      otel.Tracer(serviceName),
		serviceName: serviceName,
	}
}

// RoundTrip implements http.RoundTripper
func (rt *RoundTripper) RoundTrip(r *http.Request) (*http.Response, error) {
	ctx, span := rt.tracer.Start(
		r.Context(),
		r.Method+" "+r.URL.Path,
		trace.WithSpanKind(trace.SpanKindClient),
		trace.WithAttributes(
			attribute.String("http.method", r.Method),
			attribute.String("http.url", r.URL.String()),
			attribute.String("http.target", r.URL.Path),
			attribute.String("net.peer.name", r.URL.Host),
		),
	)
	defer span.End()

	// Inject trace context into headers
	InjectTraceHeaders(ctx, r.Header)

	// Make request
	resp, err := rt.base.RoundTrip(r.WithContext(ctx))
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, err.Error())
		return nil, err
	}

	// Record response
	span.SetAttributes(
		attribute.Int("http.status_code", resp.StatusCode),
	)

	if resp.StatusCode >= 400 {
		span.SetStatus(codes.Error, resp.Status)
	} else {
		span.SetStatus(codes.Ok, "")
	}

	return resp, nil
}
