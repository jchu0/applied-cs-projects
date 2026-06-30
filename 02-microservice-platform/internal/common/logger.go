package common

import (
	"context"
	"log/slog"
	"os"
)

// Logger wraps structured logging
type Logger struct {
	*slog.Logger
}

// NewLogger creates a new structured logger
func NewLogger(serviceName string) *Logger {
	var handler slog.Handler

	// JSON logging for production, text for development
	if os.Getenv("ENVIRONMENT") == "production" {
		handler = slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
			Level: slog.LevelInfo,
		})
	} else {
		handler = slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
			Level: slog.LevelDebug,
		})
	}

	logger := slog.New(handler).With(
		slog.String("service", serviceName),
	)

	return &Logger{Logger: logger}
}

// WithContext returns logger with context values
func (l *Logger) WithContext(ctx context.Context) *Logger {
	// Extract trace ID, request ID, etc. from context
	logger := l.Logger

	if traceID, ok := ctx.Value("trace_id").(string); ok {
		logger = logger.With(slog.String("trace_id", traceID))
	}

	if requestID, ok := ctx.Value("request_id").(string); ok {
		logger = logger.With(slog.String("request_id", requestID))
	}

	if tenantID, ok := ctx.Value("tenant_id").(string); ok {
		logger = logger.With(slog.String("tenant_id", tenantID))
	}

	return &Logger{Logger: logger}
}

// WithTenant returns logger with tenant context
func (l *Logger) WithTenant(tenantID string) *Logger {
	return &Logger{
		Logger: l.Logger.With(slog.String("tenant_id", tenantID)),
	}
}

// WithUser returns logger with user context
func (l *Logger) WithUser(userID string) *Logger {
	return &Logger{
		Logger: l.Logger.With(slog.String("user_id", userID)),
	}
}
