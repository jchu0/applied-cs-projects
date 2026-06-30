package common

import (
	"os"
	"testing"
	"time"
)

func TestLoadConfig(t *testing.T) {
	// Test with defaults
	cfg := LoadConfig("test-service")

	if cfg.GRPCPort != "9090" {
		t.Errorf("expected default GRPC port 9090, got %s", cfg.GRPCPort)
	}
	if cfg.HTTPPort != "8080" {
		t.Errorf("expected default HTTP port 8080, got %s", cfg.HTTPPort)
	}
	if cfg.Environment != "development" {
		t.Errorf("expected default environment development, got %s", cfg.Environment)
	}
	if cfg.ServiceName != "test-service" {
		t.Errorf("expected service name test-service, got %s", cfg.ServiceName)
	}
	if cfg.AccessTokenExpiry != 15*time.Minute {
		t.Errorf("expected default access token expiry 15m, got %v", cfg.AccessTokenExpiry)
	}
}

func TestLoadConfig_WithEnvVars(t *testing.T) {
	// Set environment variables
	os.Setenv("GRPC_PORT", "8081")
	os.Setenv("HTTP_PORT", "8082")
	os.Setenv("ENVIRONMENT", "production")
	defer func() {
		os.Unsetenv("GRPC_PORT")
		os.Unsetenv("HTTP_PORT")
		os.Unsetenv("ENVIRONMENT")
	}()

	cfg := LoadConfig("test-service")

	if cfg.GRPCPort != "8081" {
		t.Errorf("expected GRPC port 8081, got %s", cfg.GRPCPort)
	}
	if cfg.HTTPPort != "8082" {
		t.Errorf("expected HTTP port 8082, got %s", cfg.HTTPPort)
	}
	if cfg.Environment != "production" {
		t.Errorf("expected environment production, got %s", cfg.Environment)
	}
}

func TestGetEnv(t *testing.T) {
	// Test with no env var set
	result := getEnv("NONEXISTENT_VAR", "default")
	if result != "default" {
		t.Errorf("expected default, got %s", result)
	}

	// Test with env var set
	os.Setenv("TEST_VAR", "custom")
	defer os.Unsetenv("TEST_VAR")

	result = getEnv("TEST_VAR", "default")
	if result != "custom" {
		t.Errorf("expected custom, got %s", result)
	}
}

func TestGetDurationEnv(t *testing.T) {
	// Test with no env var set
	result := getDurationEnv("NONEXISTENT_DURATION", 30*time.Minute)
	if result != 30*time.Minute {
		t.Errorf("expected 30m, got %v", result)
	}

	// Test with valid duration
	os.Setenv("TEST_DURATION", "60")
	defer os.Unsetenv("TEST_DURATION")

	result = getDurationEnv("TEST_DURATION", 30*time.Minute)
	if result != 60*time.Minute {
		t.Errorf("expected 60m, got %v", result)
	}
}

func TestNewLogger(t *testing.T) {
	logger := NewLogger("test-service")
	if logger == nil {
		t.Error("expected logger to be created")
	}

	// Test that logger can be used
	logger.Info("test message", "key", "value")
	logger.WithTenant("tenant-1").Info("tenant message")
	logger.WithUser("user-1").Info("user message")
}

func TestLogger_WithContext(t *testing.T) {
	logger := NewLogger("test-service")

	// Create context with values
	// Note: This is a simplified test, actual context values would be set differently

	contextLogger := logger.WithTenant("tenant-1").WithUser("user-1")
	if contextLogger == nil {
		t.Error("expected context logger to be created")
	}
}
