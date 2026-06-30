package metrics

import (
	"net/http"
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// Metrics holds all Prometheus metrics
type Metrics struct {
	// HTTP metrics
	HTTPRequestsTotal    *prometheus.CounterVec
	HTTPRequestDuration  *prometheus.HistogramVec
	HTTPResponseSize     *prometheus.HistogramVec
	HTTPActiveRequests   prometheus.Gauge

	// gRPC metrics
	GRPCRequestsTotal   *prometheus.CounterVec
	GRPCRequestDuration *prometheus.HistogramVec

	// Database metrics
	DBQueryTotal    *prometheus.CounterVec
	DBQueryDuration *prometheus.HistogramVec
	DBConnections   *prometheus.GaugeVec

	// Redis metrics
	RedisOpsTotal    *prometheus.CounterVec
	RedisOpsDuration *prometheus.HistogramVec

	// Business metrics
	EventsPublished *prometheus.CounterVec
	EventsConsumed  *prometheus.CounterVec

	// Circuit breaker metrics
	CircuitBreakerState   *prometheus.GaugeVec
	CircuitBreakerTrips   *prometheus.CounterVec

	// Custom metrics registry
	registry *prometheus.Registry
}

// Config holds metrics configuration
type Config struct {
	ServiceName string
	Namespace   string
	Subsystem   string
}

// New creates a new metrics instance
func New(cfg Config) *Metrics {
	registry := prometheus.NewRegistry()

	// Register default collectors
	registry.MustRegister(prometheus.NewGoCollector())
	registry.MustRegister(prometheus.NewProcessCollector(prometheus.ProcessCollectorOpts{}))

	factory := promauto.With(registry)

	m := &Metrics{
		registry: registry,

		// HTTP metrics
		HTTPRequestsTotal: factory.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: cfg.Namespace,
				Subsystem: cfg.Subsystem,
				Name:      "http_requests_total",
				Help:      "Total number of HTTP requests",
			},
			[]string{"method", "path", "status"},
		),
		HTTPRequestDuration: factory.NewHistogramVec(
			prometheus.HistogramOpts{
				Namespace: cfg.Namespace,
				Subsystem: cfg.Subsystem,
				Name:      "http_request_duration_seconds",
				Help:      "HTTP request duration in seconds",
				Buckets:   prometheus.DefBuckets,
			},
			[]string{"method", "path", "status"},
		),
		HTTPResponseSize: factory.NewHistogramVec(
			prometheus.HistogramOpts{
				Namespace: cfg.Namespace,
				Subsystem: cfg.Subsystem,
				Name:      "http_response_size_bytes",
				Help:      "HTTP response size in bytes",
				Buckets:   prometheus.ExponentialBuckets(100, 10, 8),
			},
			[]string{"method", "path"},
		),
		HTTPActiveRequests: factory.NewGauge(
			prometheus.GaugeOpts{
				Namespace: cfg.Namespace,
				Subsystem: cfg.Subsystem,
				Name:      "http_active_requests",
				Help:      "Number of active HTTP requests",
			},
		),

		// gRPC metrics
		GRPCRequestsTotal: factory.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: cfg.Namespace,
				Subsystem: cfg.Subsystem,
				Name:      "grpc_requests_total",
				Help:      "Total number of gRPC requests",
			},
			[]string{"method", "code"},
		),
		GRPCRequestDuration: factory.NewHistogramVec(
			prometheus.HistogramOpts{
				Namespace: cfg.Namespace,
				Subsystem: cfg.Subsystem,
				Name:      "grpc_request_duration_seconds",
				Help:      "gRPC request duration in seconds",
				Buckets:   prometheus.DefBuckets,
			},
			[]string{"method", "code"},
		),

		// Database metrics
		DBQueryTotal: factory.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: cfg.Namespace,
				Subsystem: cfg.Subsystem,
				Name:      "db_queries_total",
				Help:      "Total number of database queries",
			},
			[]string{"operation", "table", "status"},
		),
		DBQueryDuration: factory.NewHistogramVec(
			prometheus.HistogramOpts{
				Namespace: cfg.Namespace,
				Subsystem: cfg.Subsystem,
				Name:      "db_query_duration_seconds",
				Help:      "Database query duration in seconds",
				Buckets:   prometheus.DefBuckets,
			},
			[]string{"operation", "table"},
		),
		DBConnections: factory.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: cfg.Namespace,
				Subsystem: cfg.Subsystem,
				Name:      "db_connections",
				Help:      "Number of database connections",
			},
			[]string{"state"},
		),

		// Redis metrics
		RedisOpsTotal: factory.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: cfg.Namespace,
				Subsystem: cfg.Subsystem,
				Name:      "redis_operations_total",
				Help:      "Total number of Redis operations",
			},
			[]string{"operation", "status"},
		),
		RedisOpsDuration: factory.NewHistogramVec(
			prometheus.HistogramOpts{
				Namespace: cfg.Namespace,
				Subsystem: cfg.Subsystem,
				Name:      "redis_operation_duration_seconds",
				Help:      "Redis operation duration in seconds",
				Buckets:   prometheus.DefBuckets,
			},
			[]string{"operation"},
		),

		// Business metrics
		EventsPublished: factory.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: cfg.Namespace,
				Subsystem: cfg.Subsystem,
				Name:      "events_published_total",
				Help:      "Total number of events published",
			},
			[]string{"event_type", "status"},
		),
		EventsConsumed: factory.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: cfg.Namespace,
				Subsystem: cfg.Subsystem,
				Name:      "events_consumed_total",
				Help:      "Total number of events consumed",
			},
			[]string{"event_type", "status"},
		),

		// Circuit breaker metrics
		CircuitBreakerState: factory.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: cfg.Namespace,
				Subsystem: cfg.Subsystem,
				Name:      "circuit_breaker_state",
				Help:      "Circuit breaker state (0=closed, 1=open, 2=half-open)",
			},
			[]string{"name"},
		),
		CircuitBreakerTrips: factory.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: cfg.Namespace,
				Subsystem: cfg.Subsystem,
				Name:      "circuit_breaker_trips_total",
				Help:      "Total number of circuit breaker trips",
			},
			[]string{"name"},
		),
	}

	return m
}

// Handler returns the Prometheus HTTP handler
func (m *Metrics) Handler() http.Handler {
	return promhttp.HandlerFor(m.registry, promhttp.HandlerOpts{
		EnableOpenMetrics: true,
	})
}

// Registry returns the Prometheus registry
func (m *Metrics) Registry() *prometheus.Registry {
	return m.registry
}

// RecordHTTPRequest records an HTTP request
func (m *Metrics) RecordHTTPRequest(method, path string, status int, duration time.Duration, responseSize int) {
	statusStr := strconv.Itoa(status)
	m.HTTPRequestsTotal.WithLabelValues(method, path, statusStr).Inc()
	m.HTTPRequestDuration.WithLabelValues(method, path, statusStr).Observe(duration.Seconds())
	m.HTTPResponseSize.WithLabelValues(method, path).Observe(float64(responseSize))
}

// RecordGRPCRequest records a gRPC request
func (m *Metrics) RecordGRPCRequest(method, code string, duration time.Duration) {
	m.GRPCRequestsTotal.WithLabelValues(method, code).Inc()
	m.GRPCRequestDuration.WithLabelValues(method, code).Observe(duration.Seconds())
}

// RecordDBQuery records a database query
func (m *Metrics) RecordDBQuery(operation, table, status string, duration time.Duration) {
	m.DBQueryTotal.WithLabelValues(operation, table, status).Inc()
	m.DBQueryDuration.WithLabelValues(operation, table).Observe(duration.Seconds())
}

// RecordRedisOp records a Redis operation
func (m *Metrics) RecordRedisOp(operation, status string, duration time.Duration) {
	m.RedisOpsTotal.WithLabelValues(operation, status).Inc()
	m.RedisOpsDuration.WithLabelValues(operation).Observe(duration.Seconds())
}

// RecordEventPublished records a published event
func (m *Metrics) RecordEventPublished(eventType, status string) {
	m.EventsPublished.WithLabelValues(eventType, status).Inc()
}

// RecordEventConsumed records a consumed event
func (m *Metrics) RecordEventConsumed(eventType, status string) {
	m.EventsConsumed.WithLabelValues(eventType, status).Inc()
}

// SetCircuitBreakerState sets the circuit breaker state
func (m *Metrics) SetCircuitBreakerState(name string, state int) {
	m.CircuitBreakerState.WithLabelValues(name).Set(float64(state))
}

// RecordCircuitBreakerTrip records a circuit breaker trip
func (m *Metrics) RecordCircuitBreakerTrip(name string) {
	m.CircuitBreakerTrips.WithLabelValues(name).Inc()
}
