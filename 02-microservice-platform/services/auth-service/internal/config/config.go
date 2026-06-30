package config

import (
	"os"
	"time"
)

// Config holds all configuration for the auth service
type Config struct {
	ServiceName     string
	GRPCPort        string
	HTTPPort        string
	DatabaseURL     string
	RedisURL        string
	NatsURL         string
	JaegerEndpoint  string
	UserServiceAddr string

	// JWT configuration
	JWTPrivateKeyPath string
	JWTPublicKeyPath  string
	JWTIssuer         string
	JWTAudience       string
	AccessTokenTTL    time.Duration
	RefreshTokenTTL   time.Duration
}

// Load loads configuration from environment variables
func Load() *Config {
	return &Config{
		ServiceName:       getEnv("SERVICE_NAME", "auth-service"),
		GRPCPort:          getEnv("GRPC_PORT", "9090"),
		HTTPPort:          getEnv("HTTP_PORT", "8080"),
		DatabaseURL:       getEnv("DATABASE_URL", "postgres://authservice:authservice_pass@localhost:5433/auth?sslmode=disable"),
		RedisURL:          getEnv("REDIS_URL", "redis://localhost:6379"),
		NatsURL:           getEnv("NATS_URL", "nats://localhost:4222"),
		JaegerEndpoint:    getEnv("JAEGER_ENDPOINT", "http://localhost:14268/api/traces"),
		UserServiceAddr:   getEnv("USER_SERVICE_ADDR", "localhost:9091"),
		JWTPrivateKeyPath: getEnv("JWT_PRIVATE_KEY_PATH", "./keys/private.pem"),
		JWTPublicKeyPath:  getEnv("JWT_PUBLIC_KEY_PATH", "./keys/public.pem"),
		JWTIssuer:         getEnv("JWT_ISSUER", "microservices-platform"),
		JWTAudience:       getEnv("JWT_AUDIENCE", "microservices-api"),
		AccessTokenTTL:    15 * time.Minute,
		RefreshTokenTTL:   7 * 24 * time.Hour,
	}
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}
