package logging

import (
	"context"
	"os"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

// contextKey is a custom type for context keys
type contextKey string

const (
	// LoggerKey is the context key for the logger
	LoggerKey contextKey = "logger"
	// RequestIDKey is the context key for request ID
	RequestIDKey contextKey = "request_id"
	// TraceIDKey is the context key for trace ID
	TraceIDKey contextKey = "trace_id"
	// TenantIDKey is the context key for tenant ID
	TenantIDKey contextKey = "tenant_id"
	// UserIDKey is the context key for user ID
	UserIDKey contextKey = "user_id"
)

// Logger wraps zap.Logger with additional context-aware methods
type Logger struct {
	*zap.Logger
}

// NewLogger creates a new logger with the specified level and service name
func NewLogger(level string, serviceName string) (*Logger, error) {
	// Parse log level
	var zapLevel zapcore.Level
	if err := zapLevel.UnmarshalText([]byte(level)); err != nil {
		zapLevel = zapcore.InfoLevel
	}

	// Encoder config
	encoderConfig := zapcore.EncoderConfig{
		TimeKey:        "timestamp",
		LevelKey:       "level",
		NameKey:        "logger",
		CallerKey:      "caller",
		FunctionKey:    zapcore.OmitKey,
		MessageKey:     "message",
		StacktraceKey:  "stacktrace",
		LineEnding:     zapcore.DefaultLineEnding,
		EncodeLevel:    zapcore.LowercaseLevelEncoder,
		EncodeTime:     zapcore.ISO8601TimeEncoder,
		EncodeDuration: zapcore.MillisDurationEncoder,
		EncodeCaller:   zapcore.ShortCallerEncoder,
	}

	// Create core
	core := zapcore.NewCore(
		zapcore.NewJSONEncoder(encoderConfig),
		zapcore.AddSync(os.Stdout),
		zapLevel,
	)

	// Create logger
	logger := zap.New(core,
		zap.AddCaller(),
		zap.AddStacktrace(zapcore.ErrorLevel),
	).With(zap.String("service", serviceName))

	return &Logger{Logger: logger}, nil
}

// WithContext returns a logger with fields from context
func (l *Logger) WithContext(ctx context.Context) *Logger {
	fields := []zap.Field{}

	if requestID := ctx.Value(RequestIDKey); requestID != nil {
		fields = append(fields, zap.String("request_id", requestID.(string)))
	}

	if traceID := ctx.Value(TraceIDKey); traceID != nil {
		fields = append(fields, zap.String("trace_id", traceID.(string)))
	}

	if tenantID := ctx.Value(TenantIDKey); tenantID != nil {
		fields = append(fields, zap.String("tenant_id", tenantID.(string)))
	}

	if userID := ctx.Value(UserIDKey); userID != nil {
		fields = append(fields, zap.String("user_id", userID.(string)))
	}

	if len(fields) == 0 {
		return l
	}

	return &Logger{Logger: l.With(fields...)}
}

// WithField returns a logger with an additional field
func (l *Logger) WithField(key string, value interface{}) *Logger {
	return &Logger{Logger: l.With(zap.Any(key, value))}
}

// WithFields returns a logger with additional fields
func (l *Logger) WithFields(fields map[string]interface{}) *Logger {
	zapFields := make([]zap.Field, 0, len(fields))
	for k, v := range fields {
		zapFields = append(zapFields, zap.Any(k, v))
	}
	return &Logger{Logger: l.With(zapFields...)}
}

// WithError returns a logger with an error field
func (l *Logger) WithError(err error) *Logger {
	return &Logger{Logger: l.With(zap.Error(err))}
}

// ToContext adds the logger to the context
func (l *Logger) ToContext(ctx context.Context) context.Context {
	return context.WithValue(ctx, LoggerKey, l)
}

// FromContext extracts the logger from context
func FromContext(ctx context.Context) *Logger {
	if logger, ok := ctx.Value(LoggerKey).(*Logger); ok {
		return logger
	}
	// Return a default logger if not found
	logger, _ := NewLogger("info", "default")
	return logger
}

// Global logger instance
var globalLogger *Logger

// Init initializes the global logger
func Init(level string, serviceName string) error {
	logger, err := NewLogger(level, serviceName)
	if err != nil {
		return err
	}
	globalLogger = logger
	return nil
}

// L returns the global logger
func L() *Logger {
	if globalLogger == nil {
		globalLogger, _ = NewLogger("info", "default")
	}
	return globalLogger
}

// Sync flushes any buffered log entries
func Sync() error {
	if globalLogger != nil {
		return globalLogger.Sync()
	}
	return nil
}
