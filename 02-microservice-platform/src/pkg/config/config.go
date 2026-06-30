package config

import (
	"fmt"
	"strings"
	"time"

	"github.com/spf13/viper"
)

// ServiceConfig contains common service configuration
type ServiceConfig struct {
	Name        string        `mapstructure:"name"`
	Environment string        `mapstructure:"environment"`
	GRPCPort    int           `mapstructure:"grpc_port"`
	HTTPPort    int           `mapstructure:"http_port"`
	LogLevel    string        `mapstructure:"log_level"`
	Timeout     time.Duration `mapstructure:"timeout"`
}

// DatabaseConfig contains database connection settings
type DatabaseConfig struct {
	Host         string        `mapstructure:"host"`
	Port         int           `mapstructure:"port"`
	User         string        `mapstructure:"user"`
	Password     string        `mapstructure:"password"`
	Database     string        `mapstructure:"database"`
	SSLMode      string        `mapstructure:"ssl_mode"`
	MaxConns     int           `mapstructure:"max_conns"`
	MinConns     int           `mapstructure:"min_conns"`
	MaxConnLife  time.Duration `mapstructure:"max_conn_lifetime"`
	MaxConnIdle  time.Duration `mapstructure:"max_conn_idle_time"`
}

// DSN returns the PostgreSQL connection string
func (c *DatabaseConfig) DSN() string {
	return fmt.Sprintf(
		"host=%s port=%d user=%s password=%s dbname=%s sslmode=%s",
		c.Host, c.Port, c.User, c.Password, c.Database, c.SSLMode,
	)
}

// RedisConfig contains Redis connection settings
type RedisConfig struct {
	Host     string `mapstructure:"host"`
	Port     int    `mapstructure:"port"`
	Password string `mapstructure:"password"`
	DB       int    `mapstructure:"db"`
}

// Addr returns the Redis address
func (c *RedisConfig) Addr() string {
	return fmt.Sprintf("%s:%d", c.Host, c.Port)
}

// NATSConfig contains NATS connection settings
type NATSConfig struct {
	URL       string `mapstructure:"url"`
	ClusterID string `mapstructure:"cluster_id"`
	ClientID  string `mapstructure:"client_id"`
}

// JWTConfig contains JWT settings
type JWTConfig struct {
	PrivateKeyPath   string        `mapstructure:"private_key_path"`
	PublicKeyPath    string        `mapstructure:"public_key_path"`
	Issuer           string        `mapstructure:"issuer"`
	Audience         string        `mapstructure:"audience"`
	AccessTokenTTL   time.Duration `mapstructure:"access_token_ttl"`
	RefreshTokenTTL  time.Duration `mapstructure:"refresh_token_ttl"`
}

// TracingConfig contains OpenTelemetry tracing settings
type TracingConfig struct {
	Enabled      bool    `mapstructure:"enabled"`
	JaegerURL    string  `mapstructure:"jaeger_url"`
	SamplingRate float64 `mapstructure:"sampling_rate"`
}

// MetricsConfig contains Prometheus metrics settings
type MetricsConfig struct {
	Enabled bool `mapstructure:"enabled"`
	Port    int  `mapstructure:"port"`
	Path    string `mapstructure:"path"`
}

// OPAConfig contains Open Policy Agent settings
type OPAConfig struct {
	PolicyDir   string `mapstructure:"policy_dir"`
	ReloadEvery string `mapstructure:"reload_every"`
}

// Load loads configuration from file and environment variables
func Load(configPath string, cfg interface{}) error {
	v := viper.New()

	// Set defaults
	v.SetDefault("service.environment", "development")
	v.SetDefault("service.log_level", "info")
	v.SetDefault("service.timeout", "30s")

	v.SetDefault("database.host", "localhost")
	v.SetDefault("database.port", 5432)
	v.SetDefault("database.ssl_mode", "disable")
	v.SetDefault("database.max_conns", 25)
	v.SetDefault("database.min_conns", 5)
	v.SetDefault("database.max_conn_lifetime", "5m")
	v.SetDefault("database.max_conn_idle_time", "1m")

	v.SetDefault("redis.host", "localhost")
	v.SetDefault("redis.port", 6379)
	v.SetDefault("redis.db", 0)

	v.SetDefault("nats.url", "nats://localhost:4222")

	v.SetDefault("jwt.issuer", "microservice-platform")
	v.SetDefault("jwt.audience", "microservice-platform")
	v.SetDefault("jwt.access_token_ttl", "15m")
	v.SetDefault("jwt.refresh_token_ttl", "168h") // 7 days

	v.SetDefault("tracing.enabled", true)
	v.SetDefault("tracing.sampling_rate", 1.0)

	v.SetDefault("metrics.enabled", true)
	v.SetDefault("metrics.path", "/metrics")

	// Read from config file
	if configPath != "" {
		v.SetConfigFile(configPath)
		if err := v.ReadInConfig(); err != nil {
			return fmt.Errorf("failed to read config file: %w", err)
		}
	}

	// Read from environment variables
	v.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))
	v.AutomaticEnv()

	// Unmarshal to struct
	if err := v.Unmarshal(cfg); err != nil {
		return fmt.Errorf("failed to unmarshal config: %w", err)
	}

	return nil
}
