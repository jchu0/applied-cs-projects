package resilience

import (
	"context"
	"errors"
	"math"
	"math/rand"
	"time"
)

// RetryConfig holds retry configuration
type RetryConfig struct {
	MaxAttempts     int
	InitialDelay    time.Duration
	MaxDelay        time.Duration
	Multiplier      float64
	RandomizeFactor float64
}

// DefaultRetryConfig returns default retry configuration
func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxAttempts:     3,
		InitialDelay:    100 * time.Millisecond,
		MaxDelay:        10 * time.Second,
		Multiplier:      2.0,
		RandomizeFactor: 0.5,
	}
}

// RetryableFunc is a function that can be retried
type RetryableFunc func() error

// RetryableContextFunc is a context-aware function that can be retried
type RetryableContextFunc func(ctx context.Context) error

// IsRetryableFunc determines if an error is retryable
type IsRetryableFunc func(error) bool

// Retry retries a function with exponential backoff
func Retry(fn RetryableFunc, cfg RetryConfig) error {
	return RetryWithContext(context.Background(), func(ctx context.Context) error {
		return fn()
	}, cfg)
}

// RetryWithContext retries a function with context support
func RetryWithContext(ctx context.Context, fn RetryableContextFunc, cfg RetryConfig) error {
	return RetryWithContextAndCheck(ctx, fn, cfg, nil)
}

// RetryWithContextAndCheck retries with a custom retryable check
func RetryWithContextAndCheck(ctx context.Context, fn RetryableContextFunc, cfg RetryConfig, isRetryable IsRetryableFunc) error {
	if cfg.MaxAttempts <= 0 {
		cfg.MaxAttempts = 1
	}

	var lastErr error
	delay := cfg.InitialDelay

	for attempt := 1; attempt <= cfg.MaxAttempts; attempt++ {
		// Check context before attempting
		if err := ctx.Err(); err != nil {
			return err
		}

		// Execute the function
		err := fn(ctx)
		if err == nil {
			return nil
		}

		lastErr = err

		// Check if we should retry
		if isRetryable != nil && !isRetryable(err) {
			return err
		}

		// Don't sleep after the last attempt
		if attempt == cfg.MaxAttempts {
			break
		}

		// Calculate sleep duration with jitter
		jitter := delay * time.Duration(cfg.RandomizeFactor*rand.Float64())
		sleepDuration := delay + jitter

		// Sleep or wait for context cancellation
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(sleepDuration):
		}

		// Calculate next delay
		delay = time.Duration(float64(delay) * cfg.Multiplier)
		if delay > cfg.MaxDelay {
			delay = cfg.MaxDelay
		}
	}

	return lastErr
}

// Backoff calculates backoff duration
type Backoff struct {
	initial    time.Duration
	max        time.Duration
	multiplier float64
	jitter     float64
	attempt    int
}

// NewBackoff creates a new backoff calculator
func NewBackoff(initial, max time.Duration, multiplier, jitter float64) *Backoff {
	return &Backoff{
		initial:    initial,
		max:        max,
		multiplier: multiplier,
		jitter:     jitter,
		attempt:    0,
	}
}

// Next returns the next backoff duration
func (b *Backoff) Next() time.Duration {
	b.attempt++

	// Calculate base delay
	delay := float64(b.initial) * math.Pow(b.multiplier, float64(b.attempt-1))
	if delay > float64(b.max) {
		delay = float64(b.max)
	}

	// Add jitter
	if b.jitter > 0 {
		jitterAmount := delay * b.jitter * rand.Float64()
		delay = delay + jitterAmount
	}

	return time.Duration(delay)
}

// Reset resets the backoff
func (b *Backoff) Reset() {
	b.attempt = 0
}

// Attempt returns the current attempt number
func (b *Backoff) Attempt() int {
	return b.attempt
}

// Common retry errors
var (
	ErrMaxRetriesExceeded = errors.New("max retries exceeded")
)

// DefaultIsRetryable returns a default retryable check function
func DefaultIsRetryable() IsRetryableFunc {
	return func(err error) bool {
		// Don't retry context errors
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return false
		}
		// Retry by default
		return true
	}
}

// RetryWithNotify retries and notifies on each retry
func RetryWithNotify(ctx context.Context, fn RetryableContextFunc, cfg RetryConfig, notify func(error, int, time.Duration)) error {
	if cfg.MaxAttempts <= 0 {
		cfg.MaxAttempts = 1
	}

	var lastErr error
	delay := cfg.InitialDelay

	for attempt := 1; attempt <= cfg.MaxAttempts; attempt++ {
		if err := ctx.Err(); err != nil {
			return err
		}

		err := fn(ctx)
		if err == nil {
			return nil
		}

		lastErr = err

		if attempt == cfg.MaxAttempts {
			break
		}

		// Notify about retry
		if notify != nil {
			notify(err, attempt, delay)
		}

		// Sleep with jitter
		jitter := delay * time.Duration(cfg.RandomizeFactor*rand.Float64())
		sleepDuration := delay + jitter

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(sleepDuration):
		}

		delay = time.Duration(float64(delay) * cfg.Multiplier)
		if delay > cfg.MaxDelay {
			delay = cfg.MaxDelay
		}
	}

	return lastErr
}
