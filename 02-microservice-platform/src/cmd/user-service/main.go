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

	"github.com/mlai/microservice-platform/internal/opa"
	"github.com/mlai/microservice-platform/internal/user"
	"github.com/mlai/microservice-platform/pkg/audit"
	"github.com/mlai/microservice-platform/pkg/config"
	"github.com/mlai/microservice-platform/pkg/database"
	"github.com/mlai/microservice-platform/pkg/logging"
	"github.com/mlai/microservice-platform/pkg/metrics"
	"github.com/mlai/microservice-platform/pkg/middleware"
	userv1 "github.com/mlai/microservice-platform/pkg/pb/user/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/reflection"
)

func main() {
	// Parse flags
	configPath := flag.String("config", "", "Path to config file")
	flag.Parse()

	// Load configuration
	cfg := user.DefaultConfig()
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

	logger.Info("starting user service",
		logging.L().WithField("port", cfg.Service.GRPCPort).Logger.Sugar().Desugar().Core(),
	)

	// Create context
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Connect to database
	db, err := database.NewPostgresDB(ctx, &cfg.Database)
	if err != nil {
		logger.Fatal("failed to connect to database", logging.L().WithError(err).Logger.Sugar().Desugar().Core())
	}
	defer db.Close()

	logger.Info("connected to database")

	// Initialize audit logging using existing database pool
	auditStore := audit.NewPostgresStore(db.Pool())
	if err := auditStore.CreateTable(ctx); err != nil {
		logger.Warn("failed to create audit table, continuing without audit")
	}
	auditLogger := audit.NewAuditLogger(audit.Config{
		ServiceName: cfg.Service.Name,
		Store:       auditStore,
	})
	logger.Info("audit logging initialized")

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
			Subsystem:   "user",
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
			logger.Info("starting metrics server", logging.L().WithField("address", metricsAddr).Logger.Sugar().Desugar().Core())
			if err := http.ListenAndServe(metricsAddr, mux); err != nil && err != http.ErrServerClosed {
				logger.Error("metrics server error", logging.L().WithError(err).Logger.Sugar().Desugar().Core())
			}
		}()
	}

	// Create repository and service
	repo := user.NewRepository(db.Pool())
	svc := user.NewService(repo, logger)

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
	userServer := user.NewServer(svc)
	userv1.RegisterUserServiceServer(grpcServer, userServer)

	// Register health check
	healthServer := health.NewServer()
	grpc_health_v1.RegisterHealthServer(grpcServer, healthServer)
	healthServer.SetServingStatus("user.v1.UserService", grpc_health_v1.HealthCheckResponse_SERVING)

	// Enable reflection for development
	reflection.Register(grpcServer)

	// Start listening
	addr := fmt.Sprintf(":%d", cfg.Service.GRPCPort)
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		logger.Fatal("failed to listen", logging.L().WithError(err).Logger.Sugar().Desugar().Core())
	}

	// Handle shutdown gracefully
	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh

		logger.Info("shutting down server...")
		healthServer.SetServingStatus("user.v1.UserService", grpc_health_v1.HealthCheckResponse_NOT_SERVING)
		grpcServer.GracefulStop()
		cancel()
	}()

	logger.Info("server started", logging.L().WithField("address", addr).Logger.Sugar().Desugar().Core())

	// Start serving
	if err := grpcServer.Serve(listener); err != nil {
		logger.Fatal("failed to serve", logging.L().WithError(err).Logger.Sugar().Desugar().Core())
	}
}
