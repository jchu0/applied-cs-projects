package gateway

import (
	"time"

	"github.com/mlai/microservice-platform/pkg/config"
)

// Config holds API gateway configuration
type Config struct {
	Service config.ServiceConfig `mapstructure:"service"`
	Redis   config.RedisConfig   `mapstructure:"redis"`
	Tracing config.TracingConfig `mapstructure:"tracing"`
	Metrics config.MetricsConfig `mapstructure:"metrics"`

	// Service endpoints
	Services ServicesConfig `mapstructure:"services"`

	// JWT configuration
	JWT JWTConfig `mapstructure:"jwt"`

	// Rate limiting
	RateLimit RateLimitConfig `mapstructure:"rate_limit"`

	// CORS
	CORS CORSConfig `mapstructure:"cors"`
}

// ServicesConfig holds backend service addresses
type ServicesConfig struct {
	UserService         string `mapstructure:"user_service"`
	AuthService         string `mapstructure:"auth_service"`
	BillingService      string `mapstructure:"billing_service"`
	NotificationService string `mapstructure:"notification_service"`
}

// JWTConfig holds JWT validation settings
type JWTConfig struct {
	PublicKeyPath string   `mapstructure:"public_key_path"`
	Issuer        string   `mapstructure:"issuer"`
	Audience      string   `mapstructure:"audience"`
	PublicRoutes  []string `mapstructure:"public_routes"`
}

// RateLimitConfig holds rate limiting settings
type RateLimitConfig struct {
	Enabled        bool          `mapstructure:"enabled"`
	RequestsPerMin int           `mapstructure:"requests_per_min"`
	BurstSize      int           `mapstructure:"burst_size"`
	WindowDuration time.Duration `mapstructure:"window_duration"`
}

// CORSConfig holds CORS settings
type CORSConfig struct {
	AllowedOrigins   []string `mapstructure:"allowed_origins"`
	AllowedMethods   []string `mapstructure:"allowed_methods"`
	AllowedHeaders   []string `mapstructure:"allowed_headers"`
	ExposedHeaders   []string `mapstructure:"exposed_headers"`
	AllowCredentials bool     `mapstructure:"allow_credentials"`
	MaxAge           int      `mapstructure:"max_age"`
}

// DefaultConfig returns default gateway configuration
func DefaultConfig() *Config {
	return &Config{
		Service: config.ServiceConfig{
			Name:     "api-gateway",
			GRPCPort: 9080,
			HTTPPort: 8080,
			LogLevel: "info",
		},
		Services: ServicesConfig{
			UserService:         "localhost:9090",
			AuthService:         "localhost:9091",
			BillingService:      "localhost:9092",
			NotificationService: "localhost:9093",
		},
		JWT: JWTConfig{
			Issuer:   "microservice-platform",
			Audience: "microservice-platform",
			PublicRoutes: []string{
				"/api/v1/auth/login",
				"/api/v1/auth/register",
				"/api/v1/auth/refresh",
				"/api/v1/health",
			},
		},
		RateLimit: RateLimitConfig{
			Enabled:        true,
			RequestsPerMin: 100,
			BurstSize:      20,
			WindowDuration: time.Minute,
		},
		CORS: CORSConfig{
			AllowedOrigins:   []string{"*"},
			AllowedMethods:   []string{"GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"},
			AllowedHeaders:   []string{"Authorization", "Content-Type", "X-Request-ID", "X-Tenant-ID"},
			ExposedHeaders:   []string{"X-Request-ID", "X-RateLimit-Remaining"},
			AllowCredentials: true,
			MaxAge:           86400,
		},
	}
}
