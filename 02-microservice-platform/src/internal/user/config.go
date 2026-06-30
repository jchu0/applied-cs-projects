package user

import (
	"github.com/mlai/microservice-platform/pkg/config"
)

// Config holds user service configuration
type Config struct {
	Service  config.ServiceConfig  `mapstructure:"service"`
	Database config.DatabaseConfig `mapstructure:"database"`
	Redis    config.RedisConfig    `mapstructure:"redis"`
	NATS     config.NATSConfig     `mapstructure:"nats"`
	Tracing  config.TracingConfig  `mapstructure:"tracing"`
	Metrics  config.MetricsConfig  `mapstructure:"metrics"`
	OPA      config.OPAConfig      `mapstructure:"opa"`
}

// DefaultConfig returns default configuration
func DefaultConfig() *Config {
	return &Config{
		Service: config.ServiceConfig{
			Name:     "user-service",
			GRPCPort: 9090,
			HTTPPort: 8090,
			LogLevel: "info",
		},
		Database: config.DatabaseConfig{
			Host:     "localhost",
			Port:     5432,
			User:     "postgres",
			Password: "postgres",
			Database: "users",
			SSLMode:  "disable",
			MaxConns: 25,
			MinConns: 5,
		},
	}
}
