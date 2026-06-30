package health

import (
	"context"
	"encoding/json"
	"net/http"
	"sync"
	"time"
)

// Status represents health status
type Status string

const (
	StatusUp       Status = "UP"
	StatusDown     Status = "DOWN"
	StatusDegraded Status = "DEGRADED"
	StatusUnknown  Status = "UNKNOWN"
)

// CheckFunc is a health check function
type CheckFunc func(ctx context.Context) error

// Check represents a single health check
type Check struct {
	Name    string    `json:"name"`
	Status  Status    `json:"status"`
	Message string    `json:"message,omitempty"`
	Time    time.Time `json:"time"`
}

// Report represents the overall health report
type Report struct {
	Status    Status           `json:"status"`
	Checks    map[string]Check `json:"checks"`
	Timestamp time.Time        `json:"timestamp"`
	Version   string           `json:"version,omitempty"`
}

// Checker manages health checks
type Checker struct {
	mu       sync.RWMutex
	checks   map[string]CheckFunc
	timeout  time.Duration
	version  string
}

// Config holds checker configuration
type Config struct {
	Timeout time.Duration
	Version string
}

// NewChecker creates a new health checker
func NewChecker(cfg Config) *Checker {
	if cfg.Timeout <= 0 {
		cfg.Timeout = 5 * time.Second
	}

	return &Checker{
		checks:  make(map[string]CheckFunc),
		timeout: cfg.Timeout,
		version: cfg.Version,
	}
}

// Register registers a health check
func (c *Checker) Register(name string, check CheckFunc) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.checks[name] = check
}

// Unregister removes a health check
func (c *Checker) Unregister(name string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.checks, name)
}

// Check runs all health checks and returns a report
func (c *Checker) Check(ctx context.Context) *Report {
	c.mu.RLock()
	checks := make(map[string]CheckFunc, len(c.checks))
	for k, v := range c.checks {
		checks[k] = v
	}
	c.mu.RUnlock()

	report := &Report{
		Status:    StatusUp,
		Checks:    make(map[string]Check),
		Timestamp: time.Now().UTC(),
		Version:   c.version,
	}

	if len(checks) == 0 {
		return report
	}

	// Run checks concurrently
	var wg sync.WaitGroup
	var mu sync.Mutex

	for name, checkFn := range checks {
		wg.Add(1)
		go func(name string, fn CheckFunc) {
			defer wg.Done()

			checkCtx, cancel := context.WithTimeout(ctx, c.timeout)
			defer cancel()

			check := Check{
				Name:   name,
				Status: StatusUp,
				Time:   time.Now().UTC(),
			}

			if err := fn(checkCtx); err != nil {
				check.Status = StatusDown
				check.Message = err.Error()
			}

			mu.Lock()
			report.Checks[name] = check
			mu.Unlock()
		}(name, checkFn)
	}

	wg.Wait()

	// Determine overall status
	for _, check := range report.Checks {
		if check.Status == StatusDown {
			report.Status = StatusDown
			break
		}
		if check.Status == StatusDegraded && report.Status == StatusUp {
			report.Status = StatusDegraded
		}
	}

	return report
}

// Handler returns an HTTP handler for health checks
func (c *Checker) Handler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		report := c.Check(r.Context())

		w.Header().Set("Content-Type", "application/json")

		if report.Status != StatusUp {
			w.WriteHeader(http.StatusServiceUnavailable)
		}

		json.NewEncoder(w).Encode(report)
	})
}

// LivenessHandler returns a simple liveness handler
func LivenessHandler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{
			"status": "UP",
		})
	})
}

// ReadinessHandler returns a readiness handler using the checker
func (c *Checker) ReadinessHandler() http.Handler {
	return c.Handler()
}

// Common check functions

// PostgresCheck creates a PostgreSQL health check
func PostgresCheck(pingFn func(ctx context.Context) error) CheckFunc {
	return func(ctx context.Context) error {
		return pingFn(ctx)
	}
}

// RedisCheck creates a Redis health check
func RedisCheck(pingFn func(ctx context.Context) error) CheckFunc {
	return func(ctx context.Context) error {
		return pingFn(ctx)
	}
}

// NATSCheck creates a NATS health check
func NATSCheck(isConnected func() bool) CheckFunc {
	return func(ctx context.Context) error {
		if !isConnected() {
			return &HealthError{Message: "NATS not connected"}
		}
		return nil
	}
}

// HTTPCheck creates an HTTP endpoint health check
func HTTPCheck(url string, expectedStatus int) CheckFunc {
	return func(ctx context.Context) error {
		req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
		if err != nil {
			return err
		}

		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return err
		}
		defer resp.Body.Close()

		if resp.StatusCode != expectedStatus {
			return &HealthError{
				Message: "unexpected status code",
				Code:    resp.StatusCode,
			}
		}

		return nil
	}
}

// HealthError represents a health check error
type HealthError struct {
	Message string
	Code    int
}

func (e *HealthError) Error() string {
	return e.Message
}
