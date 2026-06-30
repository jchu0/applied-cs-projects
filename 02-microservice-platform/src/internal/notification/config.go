package notification

import (
	"github.com/mlai/microservice-platform/pkg/config"
)

// Config holds notification service configuration
type Config struct {
	Service  config.ServiceConfig  `mapstructure:"service"`
	Database config.DatabaseConfig `mapstructure:"database"`
	Redis    config.RedisConfig    `mapstructure:"redis"`
	NATS     config.NATSConfig     `mapstructure:"nats"`
	Tracing  config.TracingConfig  `mapstructure:"tracing"`
	Metrics  config.MetricsConfig  `mapstructure:"metrics"`

	// Email provider configuration
	Email EmailConfig `mapstructure:"email"`
}

// EmailConfig holds email provider settings
type EmailConfig struct {
	Provider   string `mapstructure:"provider"` // sendgrid, ses, smtp
	APIKey     string `mapstructure:"api_key"`
	FromEmail  string `mapstructure:"from_email"`
	FromName   string `mapstructure:"from_name"`
	SMTPHost   string `mapstructure:"smtp_host"`
	SMTPPort   int    `mapstructure:"smtp_port"`
	SMTPUser   string `mapstructure:"smtp_user"`
	SMTPPass   string `mapstructure:"smtp_pass"`
}

// DefaultConfig returns default configuration
func DefaultConfig() *Config {
	return &Config{
		Service: config.ServiceConfig{
			Name:     "notification-service",
			GRPCPort: 9093,
			HTTPPort: 8093,
			LogLevel: "info",
		},
		Database: config.DatabaseConfig{
			Host:     "localhost",
			Port:     5434,
			User:     "postgres",
			Password: "postgres",
			Database: "notifications",
			SSLMode:  "disable",
			MaxConns: 25,
			MinConns: 5,
		},
		Email: EmailConfig{
			Provider:  "smtp",
			FromEmail: "noreply@example.com",
			FromName:  "Microservice Platform",
			SMTPHost:  "localhost",
			SMTPPort:  1025, // Mailhog default
		},
	}
}
