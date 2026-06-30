package config

import (
	"os"
)

// Config holds all configuration for the billing service
type Config struct {
	ServiceName        string
	GRPCPort           string
	HTTPPort           string
	DatabaseURL        string
	RedisURL           string
	NatsURL            string
	JaegerEndpoint     string
	StripeSecretKey    string
	StripeWebhookSecret string
}

// Load loads configuration from environment variables
func Load() *Config {
	return &Config{
		ServiceName:        getEnv("SERVICE_NAME", "billing-service"),
		GRPCPort:           getEnv("GRPC_PORT", "9090"),
		HTTPPort:           getEnv("HTTP_PORT", "8080"),
		DatabaseURL:        getEnv("DATABASE_URL", "postgres://billingservice:billingservice_pass@localhost:5434/billing?sslmode=disable"),
		RedisURL:           getEnv("REDIS_URL", "redis://localhost:6379"),
		NatsURL:            getEnv("NATS_URL", "nats://localhost:4222"),
		JaegerEndpoint:     getEnv("JAEGER_ENDPOINT", "http://localhost:14268/api/traces"),
		StripeSecretKey:    getEnv("STRIPE_SECRET_KEY", ""),
		StripeWebhookSecret: getEnv("STRIPE_WEBHOOK_SECRET", ""),
	}
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}
