package common

import (
	"os"
	"strconv"
	"time"
)

// Config holds common service configuration
type Config struct {
	// Server settings
	GRPCPort string
	HTTPPort string

	// Database settings
	DatabaseURL string

	// Redis settings
	RedisURL string

	// JWT settings
	JWTSecret          string
	JWTPrivateKeyPath  string
	JWTPublicKeyPath   string
	AccessTokenExpiry  time.Duration
	RefreshTokenExpiry time.Duration

	// Service settings
	ServiceName    string
	ServiceVersion string
	Environment    string

	// Observability
	JaegerEndpoint string
	MetricsPort    string
}

// LoadConfig loads configuration from environment variables
func LoadConfig(serviceName string) *Config {
	return &Config{
		GRPCPort:           getEnv("GRPC_PORT", "9090"),
		HTTPPort:           getEnv("HTTP_PORT", "8080"),
		DatabaseURL:        getEnv("DATABASE_URL", "postgres://postgres:postgres@localhost:5432/"+serviceName+"?sslmode=disable"),
		RedisURL:           getEnv("REDIS_URL", "redis://localhost:6379"),
		JWTSecret:          getEnv("JWT_SECRET", "development-secret-key-change-in-production"),
		JWTPrivateKeyPath:  getEnv("JWT_PRIVATE_KEY_PATH", ""),
		JWTPublicKeyPath:   getEnv("JWT_PUBLIC_KEY_PATH", ""),
		AccessTokenExpiry:  getDurationEnv("ACCESS_TOKEN_EXPIRY", 15*time.Minute),
		RefreshTokenExpiry: getDurationEnv("REFRESH_TOKEN_EXPIRY", 7*24*time.Hour),
		ServiceName:        serviceName,
		ServiceVersion:     getEnv("SERVICE_VERSION", "1.0.0"),
		Environment:        getEnv("ENVIRONMENT", "development"),
		JaegerEndpoint:     getEnv("JAEGER_ENDPOINT", "http://localhost:14268/api/traces"),
		MetricsPort:        getEnv("METRICS_PORT", "9091"),
	}
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getDurationEnv(key string, defaultValue time.Duration) time.Duration {
	if value := os.Getenv(key); value != "" {
		if minutes, err := strconv.Atoi(value); err == nil {
			return time.Duration(minutes) * time.Minute
		}
	}
	return defaultValue
}
