package metrics

import (
	"net/http"
	"regexp"
	"strconv"
	"time"
)

// HTTPMiddleware returns HTTP middleware for metrics
func (m *Metrics) HTTPMiddleware(pathNormalizer func(string) string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			start := time.Now()
			m.HTTPActiveRequests.Inc()
			defer m.HTTPActiveRequests.Dec()

			// Wrap response writer
			wrapped := &metricsResponseWriter{
				ResponseWriter: w,
				statusCode:     http.StatusOK,
			}

			// Handle request
			next.ServeHTTP(wrapped, r)

			// Calculate duration
			duration := time.Since(start)

			// Normalize path for metrics (replace IDs with placeholders)
			path := r.URL.Path
			if pathNormalizer != nil {
				path = pathNormalizer(path)
			}

			// Record metrics
			m.RecordHTTPRequest(
				r.Method,
				path,
				wrapped.statusCode,
				duration,
				wrapped.bytesWritten,
			)
		})
	}
}

// metricsResponseWriter wraps http.ResponseWriter to capture metrics
type metricsResponseWriter struct {
	http.ResponseWriter
	statusCode   int
	bytesWritten int
}

func (w *metricsResponseWriter) WriteHeader(code int) {
	w.statusCode = code
	w.ResponseWriter.WriteHeader(code)
}

func (w *metricsResponseWriter) Write(b []byte) (int, error) {
	n, err := w.ResponseWriter.Write(b)
	w.bytesWritten += n
	return n, err
}

// Unwrap returns the underlying ResponseWriter
func (w *metricsResponseWriter) Unwrap() http.ResponseWriter {
	return w.ResponseWriter
}

// DefaultPathNormalizer creates a default path normalizer that replaces UUIDs and numeric IDs
func DefaultPathNormalizer() func(string) string {
	uuidPattern := regexp.MustCompile(`[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}`)
	numericPattern := regexp.MustCompile(`/\d+`)

	return func(path string) string {
		// Replace UUIDs
		path = uuidPattern.ReplaceAllString(path, ":id")
		// Replace numeric IDs
		path = numericPattern.ReplaceAllString(path, "/:id")
		return path
	}
}

// Timer is a helper for timing operations
type Timer struct {
	start time.Time
}

// NewTimer creates a new timer
func NewTimer() *Timer {
	return &Timer{start: time.Now()}
}

// ObserveDuration observes the duration
func (t *Timer) ObserveDuration() time.Duration {
	return time.Since(t.start)
}

// ObserveSeconds returns duration in seconds
func (t *Timer) ObserveSeconds() float64 {
	return time.Since(t.start).Seconds()
}

// ResponseSizeMiddleware returns middleware that tracks response sizes
func ResponseSizeMiddleware(histogram interface{ Observe(float64) }) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			wrapped := &metricsResponseWriter{
				ResponseWriter: w,
				statusCode:     http.StatusOK,
			}

			next.ServeHTTP(wrapped, r)

			histogram.Observe(float64(wrapped.bytesWritten))
		})
	}
}

// LatencyBucket returns a string representing the latency bucket
func LatencyBucket(duration time.Duration) string {
	ms := duration.Milliseconds()
	switch {
	case ms < 10:
		return "<10ms"
	case ms < 50:
		return "<50ms"
	case ms < 100:
		return "<100ms"
	case ms < 500:
		return "<500ms"
	case ms < 1000:
		return "<1s"
	case ms < 5000:
		return "<5s"
	default:
		return ">5s"
	}
}

// StatusCodeClass returns the HTTP status code class
func StatusCodeClass(code int) string {
	switch {
	case code >= 500:
		return "5xx"
	case code >= 400:
		return "4xx"
	case code >= 300:
		return "3xx"
	case code >= 200:
		return "2xx"
	default:
		return strconv.Itoa(code)
	}
}
