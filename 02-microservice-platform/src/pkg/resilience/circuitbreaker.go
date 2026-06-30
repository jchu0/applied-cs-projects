package resilience

import (
	"errors"
	"sync"
	"time"
)

// State represents circuit breaker state
type State int

const (
	StateClosed State = iota
	StateOpen
	StateHalfOpen
)

func (s State) String() string {
	switch s {
	case StateClosed:
		return "closed"
	case StateOpen:
		return "open"
	case StateHalfOpen:
		return "half-open"
	default:
		return "unknown"
	}
}

// Errors
var (
	ErrCircuitOpen    = errors.New("circuit breaker is open")
	ErrTooManyRequests = errors.New("too many requests in half-open state")
)

// CircuitBreaker implements the circuit breaker pattern
type CircuitBreaker struct {
	name          string
	maxFailures   int
	timeout       time.Duration
	halfOpenMax   int

	mu            sync.Mutex
	state         State
	failures      int
	successes     int
	lastFailure   time.Time
	halfOpenCount int

	// Callbacks
	onStateChange func(name string, from, to State)
}

// Config holds circuit breaker configuration
type Config struct {
	Name        string
	MaxFailures int
	Timeout     time.Duration
	HalfOpenMax int
}

// NewCircuitBreaker creates a new circuit breaker
func NewCircuitBreaker(cfg Config) *CircuitBreaker {
	if cfg.MaxFailures <= 0 {
		cfg.MaxFailures = 5
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 30 * time.Second
	}
	if cfg.HalfOpenMax <= 0 {
		cfg.HalfOpenMax = 3
	}

	return &CircuitBreaker{
		name:        cfg.Name,
		maxFailures: cfg.MaxFailures,
		timeout:     cfg.Timeout,
		halfOpenMax: cfg.HalfOpenMax,
		state:       StateClosed,
	}
}

// Execute executes the given function with circuit breaker protection
func (cb *CircuitBreaker) Execute(fn func() error) error {
	if !cb.allowRequest() {
		return ErrCircuitOpen
	}

	err := fn()

	cb.recordResult(err)
	return err
}

// allowRequest checks if a request should be allowed
func (cb *CircuitBreaker) allowRequest() bool {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	switch cb.state {
	case StateClosed:
		return true

	case StateOpen:
		// Check if timeout has passed
		if time.Since(cb.lastFailure) > cb.timeout {
			cb.setState(StateHalfOpen)
			cb.halfOpenCount = 0
			return true
		}
		return false

	case StateHalfOpen:
		// Allow limited requests in half-open state
		if cb.halfOpenCount < cb.halfOpenMax {
			cb.halfOpenCount++
			return true
		}
		return false
	}

	return false
}

// recordResult records the result of a request
func (cb *CircuitBreaker) recordResult(err error) {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	if err != nil {
		cb.onFailure()
	} else {
		cb.onSuccess()
	}
}

// onSuccess handles a successful request
func (cb *CircuitBreaker) onSuccess() {
	switch cb.state {
	case StateClosed:
		cb.failures = 0

	case StateHalfOpen:
		cb.successes++
		if cb.successes >= cb.halfOpenMax {
			cb.setState(StateClosed)
			cb.failures = 0
			cb.successes = 0
		}
	}
}

// onFailure handles a failed request
func (cb *CircuitBreaker) onFailure() {
	cb.failures++
	cb.lastFailure = time.Now()

	switch cb.state {
	case StateClosed:
		if cb.failures >= cb.maxFailures {
			cb.setState(StateOpen)
		}

	case StateHalfOpen:
		cb.setState(StateOpen)
		cb.successes = 0
	}
}

// setState changes the circuit breaker state
func (cb *CircuitBreaker) setState(state State) {
	if cb.state == state {
		return
	}

	from := cb.state
	cb.state = state

	if cb.onStateChange != nil {
		cb.onStateChange(cb.name, from, state)
	}
}

// State returns the current state
func (cb *CircuitBreaker) State() State {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	return cb.state
}

// OnStateChange sets the callback for state changes
func (cb *CircuitBreaker) OnStateChange(fn func(name string, from, to State)) {
	cb.onStateChange = fn
}

// Reset resets the circuit breaker to closed state
func (cb *CircuitBreaker) Reset() {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.state = StateClosed
	cb.failures = 0
	cb.successes = 0
}

// Metrics returns current metrics
func (cb *CircuitBreaker) Metrics() map[string]interface{} {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	return map[string]interface{}{
		"name":        cb.name,
		"state":       cb.state.String(),
		"failures":    cb.failures,
		"successes":   cb.successes,
		"lastFailure": cb.lastFailure,
	}
}

// CircuitBreakerRegistry manages multiple circuit breakers
type CircuitBreakerRegistry struct {
	mu       sync.RWMutex
	breakers map[string]*CircuitBreaker
	config   Config
}

// NewRegistry creates a new circuit breaker registry
func NewRegistry(defaultConfig Config) *CircuitBreakerRegistry {
	return &CircuitBreakerRegistry{
		breakers: make(map[string]*CircuitBreaker),
		config:   defaultConfig,
	}
}

// Get gets or creates a circuit breaker for the given name
func (r *CircuitBreakerRegistry) Get(name string) *CircuitBreaker {
	r.mu.RLock()
	cb, ok := r.breakers[name]
	r.mu.RUnlock()

	if ok {
		return cb
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	// Double-check
	if cb, ok := r.breakers[name]; ok {
		return cb
	}

	cfg := r.config
	cfg.Name = name
	cb = NewCircuitBreaker(cfg)
	r.breakers[name] = cb

	return cb
}

// All returns all circuit breakers
func (r *CircuitBreakerRegistry) All() map[string]*CircuitBreaker {
	r.mu.RLock()
	defer r.mu.RUnlock()

	result := make(map[string]*CircuitBreaker, len(r.breakers))
	for k, v := range r.breakers {
		result[k] = v
	}
	return result
}
