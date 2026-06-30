package main

import (
	"context"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/mlai/microservice-platform/internal/gateway"
	"github.com/mlai/microservice-platform/pkg/config"
	"github.com/mlai/microservice-platform/pkg/logging"
	"github.com/redis/go-redis/v9"
)

func main() {
	configPath := flag.String("config", "", "Path to config file")
	flag.Parse()

	cfg := gateway.DefaultConfig()
	if err := config.Load(*configPath, cfg); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load config: %v\n", err)
		os.Exit(1)
	}

	logger, err := logging.NewLogger(cfg.Service.LogLevel, cfg.Service.Name)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create logger: %v\n", err)
		os.Exit(1)
	}
	defer logger.Sync()

	logger.Info("starting API gateway")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Connect to Redis for rate limiting
	redisClient := redis.NewClient(&redis.Options{
		Addr:     cfg.Redis.Addr(),
		Password: cfg.Redis.Password,
		DB:       cfg.Redis.DB,
	})
	defer redisClient.Close()

	if err := redisClient.Ping(ctx).Err(); err != nil {
		logger.Warn("Redis not available, rate limiting disabled")
		cfg.RateLimit.Enabled = false
	} else {
		logger.Info("connected to Redis")
	}

	// Create components
	router := gateway.NewRouter(cfg.Services, logger)
	rateLimiter := gateway.NewRateLimiter(redisClient, cfg.RateLimit)

	// JWT validator (optional if no public key configured)
	var jwtValidator *gateway.JWTValidator
	if cfg.JWT.PublicKeyPath != "" {
		jwtValidator, err = gateway.NewJWTValidator(cfg.JWT)
		if err != nil {
			logger.Warn("JWT validation disabled", logging.L().WithError(err).Logger.Sugar().Desugar().Core())
		} else {
			logger.Info("JWT validation enabled")
		}
	}

	// Build middleware chain
	var handler http.Handler = router

	// Apply middleware (in reverse order)
	if jwtValidator != nil {
		handler = jwtValidator.Middleware(handler)
	}
	handler = rateLimiter.Middleware(handler)
	handler = gateway.CORSMiddleware(cfg.CORS, handler)
	handler = gateway.LoggingMiddleware(logger, handler)
	handler = gateway.RecoveryMiddleware(logger, handler)

	// Create HTTP server
	server := &http.Server{
		Addr:         fmt.Sprintf(":%d", cfg.Service.HTTPPort),
		Handler:      handler,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	// Start server
	go func() {
		logger.Info("server started", logging.L().WithField("addr", server.Addr).Logger.Sugar().Desugar().Core())
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Fatal("server failed")
		}
	}()

	// Handle shutdown
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh

	logger.Info("shutting down server...")

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer shutdownCancel()

	if err := server.Shutdown(shutdownCtx); err != nil {
		logger.Error("server shutdown failed")
	}

	cancel()
	logger.Info("server stopped")
}
