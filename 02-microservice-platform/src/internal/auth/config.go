package auth

import (
	"github.com/mlai/microservice-platform/pkg/config"
)

// Config holds auth service configuration
type Config struct {
	Service  config.ServiceConfig  `mapstructure:"service"`
	Database config.DatabaseConfig `mapstructure:"database"`
	Redis    config.RedisConfig    `mapstructure:"redis"`
	JWT      config.JWTConfig      `mapstructure:"jwt"`
	NATS     config.NATSConfig     `mapstructure:"nats"`
	Tracing  config.TracingConfig  `mapstructure:"tracing"`
	Metrics  config.MetricsConfig  `mapstructure:"metrics"`
	OPA      config.OPAConfig      `mapstructure:"opa"`

	// Service endpoints
	UserServiceAddr string `mapstructure:"user_service_addr"`
}

// DefaultConfig returns default configuration
func DefaultConfig() *Config {
	return &Config{
		Service: config.ServiceConfig{
			Name:     "auth-service",
			GRPCPort: 9091,
			HTTPPort: 8091,
			LogLevel: "info",
		},
		Database: config.DatabaseConfig{
			Host:     "localhost",
			Port:     5432,
			User:     "postgres",
			Password: "postgres",
			Database: "auth",
			SSLMode:  "disable",
			MaxConns: 25,
			MinConns: 5,
		},
		Redis: config.RedisConfig{
			Host: "localhost",
			Port: 6379,
		},
		JWT: config.JWTConfig{
			Issuer:          "microservice-platform",
			Audience:        "microservice-platform",
			AccessTokenTTL:  900,  // 15 minutes
			RefreshTokenTTL: 604800, // 7 days
		},
		UserServiceAddr: "localhost:9090",
	}
}
