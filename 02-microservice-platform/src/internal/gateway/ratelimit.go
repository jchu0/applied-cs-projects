package gateway

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/redis/go-redis/v9"
)

// RateLimiter implements sliding window rate limiting
type RateLimiter struct {
	client  *redis.Client
	config  RateLimitConfig
	enabled bool
}

// NewRateLimiter creates a new rate limiter
func NewRateLimiter(client *redis.Client, config RateLimitConfig) *RateLimiter {
	return &RateLimiter{
		client:  client,
		config:  config,
		enabled: config.Enabled,
	}
}

// Allow checks if a request should be allowed
func (r *RateLimiter) Allow(ctx context.Context, key string) (bool, int, error) {
	if !r.enabled {
		return true, r.config.RequestsPerMin, nil
	}

	now := time.Now().UnixMicro()
	windowStart := now - r.config.WindowDuration.Microseconds()

	pipe := r.client.Pipeline()

	// Remove old entries
	pipe.ZRemRangeByScore(ctx, key, "0", fmt.Sprintf("%d", windowStart))

	// Count current entries
	countCmd := pipe.ZCard(ctx, key)

	// Add current request
	pipe.ZAdd(ctx, key, redis.Z{Score: float64(now), Member: now})

	// Set expiry
	pipe.Expire(ctx, key, r.config.WindowDuration)

	_, err := pipe.Exec(ctx)
	if err != nil {
		return false, 0, err
	}

	count := countCmd.Val()
	remaining := r.config.RequestsPerMin - int(count)
	if remaining < 0 {
		remaining = 0
	}

	return count < int64(r.config.RequestsPerMin), remaining, nil
}

// Middleware returns HTTP middleware for rate limiting
func (r *RateLimiter) Middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if !r.enabled {
			next.ServeHTTP(w, req)
			return
		}

		// Build rate limit key
		key := r.buildKey(req)

		allowed, remaining, err := r.Allow(req.Context(), key)
		if err != nil {
			// On error, allow the request but log
			next.ServeHTTP(w, req)
			return
		}

		// Set rate limit headers
		w.Header().Set("X-RateLimit-Limit", fmt.Sprintf("%d", r.config.RequestsPerMin))
		w.Header().Set("X-RateLimit-Remaining", fmt.Sprintf("%d", remaining))
		w.Header().Set("X-RateLimit-Reset", fmt.Sprintf("%d", time.Now().Add(r.config.WindowDuration).Unix()))

		if !allowed {
			w.Header().Set("Retry-After", fmt.Sprintf("%d", int(r.config.WindowDuration.Seconds())))
			http.Error(w, "Rate limit exceeded", http.StatusTooManyRequests)
			return
		}

		next.ServeHTTP(w, req)
	})
}

// buildKey builds a rate limit key from the request
func (r *RateLimiter) buildKey(req *http.Request) string {
	// Use tenant + user if available, otherwise IP
	tenantID := req.Header.Get("X-Tenant-ID")
	userID := req.Header.Get("X-User-ID")

	if tenantID != "" && userID != "" {
		return fmt.Sprintf("ratelimit:%s:%s:%s", tenantID, userID, req.URL.Path)
	}

	// Fall back to IP-based limiting
	ip := req.RemoteAddr
	if forwarded := req.Header.Get("X-Forwarded-For"); forwarded != "" {
		ip = forwarded
	}

	return fmt.Sprintf("ratelimit:ip:%s:%s", ip, req.URL.Path)
}
