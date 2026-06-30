package billing

import (
	"github.com/mlai/microservice-platform/pkg/config"
)

// Config holds billing service configuration
type Config struct {
	Service  config.ServiceConfig  `mapstructure:"service"`
	Database config.DatabaseConfig `mapstructure:"database"`
	Redis    config.RedisConfig    `mapstructure:"redis"`
	NATS     config.NATSConfig     `mapstructure:"nats"`
	Tracing  config.TracingConfig  `mapstructure:"tracing"`
	Metrics  config.MetricsConfig  `mapstructure:"metrics"`
	OPA      config.OPAConfig      `mapstructure:"opa"`

	// Stripe configuration
	Stripe StripeConfig `mapstructure:"stripe"`
}

// StripeConfig holds Stripe API configuration
type StripeConfig struct {
	SecretKey      string `mapstructure:"secret_key"`
	WebhookSecret  string `mapstructure:"webhook_secret"`
	PublishableKey string `mapstructure:"publishable_key"`
}

// DefaultConfig returns default configuration
func DefaultConfig() *Config {
	return &Config{
		Service: config.ServiceConfig{
			Name:     "billing-service",
			GRPCPort: 9092,
			HTTPPort: 8092,
			LogLevel: "info",
		},
		Database: config.DatabaseConfig{
			Host:     "localhost",
			Port:     5433,
			User:     "postgres",
			Password: "postgres",
			Database: "billing",
			SSLMode:  "disable",
			MaxConns: 25,
			MinConns: 5,
		},
		Stripe: StripeConfig{
			SecretKey: "sk_test_xxx", // Use test key for development
		},
	}
}
