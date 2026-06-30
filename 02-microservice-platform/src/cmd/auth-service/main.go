package main

import (
	"context"
	"flag"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/mlai/microservice-platform/internal/auth"
	"github.com/mlai/microservice-platform/internal/opa"
	"github.com/mlai/microservice-platform/pkg/audit"
	"github.com/mlai/microservice-platform/pkg/config"
	"github.com/mlai/microservice-platform/pkg/logging"
	"github.com/mlai/microservice-platform/pkg/metrics"
	"github.com/mlai/microservice-platform/pkg/middleware"
	"github.com/mlai/microservice-platform/pkg/resilience"
	authv1 "github.com/mlai/microservice-platform/pkg/pb/auth/v1"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/redis/go-redis/v9"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/reflection"
)

func main() {
	// Parse flags
	configPath := flag.String("config", "", "Path to config file")
	flag.Parse()

	// Load configuration
	cfg := auth.DefaultConfig()
	if err := config.Load(*configPath, cfg); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load config: %v\n", err)
		os.Exit(1)
	}

	// Initialize logger
	logger, err := logging.NewLogger(cfg.Service.LogLevel, cfg.Service.Name)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create logger: %v\n", err)
		os.Exit(1)
	}
	defer logger.Sync()

	logger.Info("starting auth service")

	// Create context
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Connect to Redis
	redisClient := redis.NewClient(&redis.Options{
		Addr:     cfg.Redis.Addr(),
		Password: cfg.Redis.Password,
		DB:       cfg.Redis.DB,
	})
	defer redisClient.Close()

	// Test Redis connection
	if err := redisClient.Ping(ctx).Err(); err != nil {
		logger.Fatal("failed to connect to Redis")
	}
	logger.Info("connected to Redis")

	// Initialize circuit breaker registry for outgoing calls
	cbRegistry := resilience.NewRegistry(resilience.Config{
		Name:        "auth-service",
		MaxFailures: 5,
		Timeout:     30 * time.Second,
		HalfOpenMax: 3,
	})

	// Connect to User Service with circuit breaker
	userConn, err := grpc.Dial(
		cfg.UserServiceAddr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithUnaryInterceptor(resilience.UnaryClientInterceptor(cbRegistry)),
		grpc.WithStreamInterceptor(resilience.StreamClientInterceptor(cbRegistry)),
	)
	if err != nil {
		logger.Fatal("failed to connect to user service")
	}
	defer userConn.Close()
	logger.Info("connected to user service with circuit breaker")

	// Initialize audit logging
	var auditLogger *audit.AuditLogger
	dbDSN := cfg.Database.DSN()
	if dbDSN != "" {
		dbPool, err := pgxpool.New(ctx, dbDSN)
		if err != nil {
			logger.Warn("failed to connect to database for audit logging, continuing without audit")
		} else {
			defer dbPool.Close()
			auditStore := audit.NewPostgresStore(dbPool)
			if err := auditStore.CreateTable(ctx); err != nil {
				logger.Warn("failed to create audit table")
			}
			auditLogger = audit.NewAuditLogger(audit.Config{
				ServiceName: cfg.Service.Name,
				Store:       auditStore,
			})
			logger.Info("audit logging initialized")
		}
	}

	// Initialize OPA policy engine
	var policyEngine *opa.PolicyEngine
	if cfg.OPA.PolicyDir != "" {
		policyEngine, err = opa.NewPolicyEngine(opa.Config{
			PolicyDir:   cfg.OPA.PolicyDir,
			ReloadEvery: 5 * time.Minute,
		})
		if err != nil {
			logger.Warn("failed to initialize OPA policy engine, continuing without authorization")
		} else {
			defer policyEngine.Stop()
			logger.Info("OPA policy engine initialized")
		}
	}

	// Initialize metrics
	var m *metrics.Metrics
	if cfg.Metrics.Enabled {
		m = metrics.New(metrics.Config{
			ServiceName: cfg.Service.Name,
			Namespace:   "microservice",
			Subsystem:   "auth",
		})
		logger.Info("metrics initialized")

		// Start metrics HTTP server
		metricsPort := cfg.Metrics.Port
		if metricsPort == 0 {
			metricsPort = cfg.Service.HTTPPort
		}
		metricsPath := cfg.Metrics.Path
		if metricsPath == "" {
			metricsPath = "/metrics"
		}

		go func() {
			mux := http.NewServeMux()
			mux.Handle(metricsPath, m.Handler())
			metricsAddr := fmt.Sprintf(":%d", metricsPort)
			logger.Info("starting metrics server")
			if err := http.ListenAndServe(metricsAddr, mux); err != nil && err != http.ErrServerClosed {
				logger.Error("metrics server error")
			}
		}()
	}

	// Create JWT Manager
	jwtManager, err := auth.NewJWTManager(
		cfg.JWT.PrivateKeyPath,
		cfg.JWT.PublicKeyPath,
		cfg.JWT.Issuer,
		cfg.JWT.Audience,
		cfg.JWT.AccessTokenTTL,
		cfg.JWT.RefreshTokenTTL,
	)
	if err != nil {
		logger.Fatal("failed to create JWT manager")
	}

	// Create session repository
	sessionRepo := auth.NewSessionRepository(redisClient, 7*24*time.Hour)

	// Create service
	svc := auth.NewService(jwtManager, sessionRepo, userConn, logger)

	// Build gRPC interceptors
	unaryInterceptors := []grpc.UnaryServerInterceptor{
		middleware.RecoveryInterceptor(logger),
		middleware.UnaryServerInterceptors(logger),
	}
	streamInterceptors := []grpc.StreamServerInterceptor{
		middleware.StreamServerInterceptors(logger),
	}

	// Add metrics interceptors if enabled
	if m != nil {
		unaryInterceptors = append(unaryInterceptors, m.UnaryServerInterceptor())
		streamInterceptors = append(streamInterceptors, m.StreamServerInterceptor())
	}

	// Add audit logging interceptor
	if auditLogger != nil {
		unaryInterceptors = append(unaryInterceptors, audit.UnaryServerInterceptor(auditLogger))
		logger.Info("audit logging interceptor added")
	}

	// Add OPA authorization interceptor
	if policyEngine != nil {
		unaryInterceptors = append(unaryInterceptors, opa.GRPCUnaryInterceptor(policyEngine, opa.DefaultGRPCSubjectExtractor()))
		streamInterceptors = append(streamInterceptors, opa.GRPCStreamInterceptor(policyEngine, opa.DefaultGRPCSubjectExtractor()))
		logger.Info("OPA authorization interceptor added")
	}

	// Create gRPC server
	grpcServer := grpc.NewServer(
		grpc.ChainUnaryInterceptor(unaryInterceptors...),
		grpc.ChainStreamInterceptor(streamInterceptors...),
	)

	// Register services
	authServer := auth.NewServer(svc)
	authv1.RegisterAuthServiceServer(grpcServer, authServer)

	// Register health check
	healthServer := health.NewServer()
	grpc_health_v1.RegisterHealthServer(grpcServer, healthServer)
	healthServer.SetServingStatus("auth.v1.AuthService", grpc_health_v1.HealthCheckResponse_SERVING)

	// Enable reflection for development
	reflection.Register(grpcServer)

	// Start listening
	addr := fmt.Sprintf(":%d", cfg.Service.GRPCPort)
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		logger.Fatal("failed to listen")
	}

	// Handle shutdown gracefully
	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh

		logger.Info("shutting down server...")
		healthServer.SetServingStatus("auth.v1.AuthService", grpc_health_v1.HealthCheckResponse_NOT_SERVING)
		grpcServer.GracefulStop()
		cancel()
	}()

	logger.Info("server started")

	// Start serving
	if err := grpcServer.Serve(listener); err != nil {
		logger.Fatal("failed to serve")
	}
}
