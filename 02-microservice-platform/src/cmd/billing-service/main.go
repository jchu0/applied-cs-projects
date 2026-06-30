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

	"github.com/mlai/microservice-platform/internal/billing"
	"github.com/mlai/microservice-platform/internal/opa"
	"github.com/mlai/microservice-platform/pkg/audit"
	"github.com/mlai/microservice-platform/pkg/config"
	"github.com/mlai/microservice-platform/pkg/database"
	"github.com/mlai/microservice-platform/pkg/events"
	"github.com/mlai/microservice-platform/pkg/logging"
	"github.com/mlai/microservice-platform/pkg/metrics"
	"github.com/mlai/microservice-platform/pkg/middleware"
	"github.com/mlai/microservice-platform/pkg/resilience"
	billingv1 "github.com/mlai/microservice-platform/pkg/pb/billing/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/reflection"
)

func main() {
	configPath := flag.String("config", "", "Path to config file")
	flag.Parse()

	cfg := billing.DefaultConfig()
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

	logger.Info("starting billing service")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Connect to database
	db, err := database.NewPostgresDB(ctx, &cfg.Database)
	if err != nil {
		logger.Fatal("failed to connect to database")
	}
	defer db.Close()
	logger.Info("connected to database")

	// Initialize circuit breaker registry for external API calls (Stripe)
	cbRegistry := resilience.NewRegistry(resilience.Config{
		Name:        "billing-service",
		MaxFailures: 5,
		Timeout:     30 * time.Second,
		HalfOpenMax: 3,
	})
	_ = cbRegistry // Used for wrapping external calls

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

	// Create event publisher
	var publisher *events.Publisher
	if cfg.NATS.URL != "" {
		publisher, err = events.NewPublisher(cfg.NATS.URL, cfg.Service.Name)
		if err != nil {
			logger.Warn("failed to create event publisher")
		} else {
			defer publisher.Close()
			logger.Info("connected to NATS")
		}
	}

	// Initialize metrics
	var m *metrics.Metrics
	if cfg.Metrics.Enabled {
		m = metrics.New(metrics.Config{
			ServiceName: cfg.Service.Name,
			Namespace:   "microservice",
			Subsystem:   "billing",
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

	// Create repository and service
	repo := billing.NewRepository(db.Pool())
	svc := billing.NewService(repo, publisher, logger, cfg.Stripe.SecretKey)

	// Build gRPC interceptors
	unaryInterceptors := []grpc.UnaryServerInterceptor{
		middleware.RecoveryInterceptor(logger),
		middleware.UnaryServerInterceptors(logger),
	}

	// Add metrics interceptor if enabled
	if m != nil {
		unaryInterceptors = append(unaryInterceptors, m.UnaryServerInterceptor())
	}

	// Add audit logging interceptor
	if auditLogger != nil {
		unaryInterceptors = append(unaryInterceptors, audit.UnaryServerInterceptor(auditLogger))
		logger.Info("audit logging interceptor added")
	}

	// Add OPA authorization interceptor
	if policyEngine != nil {
		unaryInterceptors = append(unaryInterceptors, opa.GRPCUnaryInterceptor(policyEngine, opa.DefaultGRPCSubjectExtractor()))
		logger.Info("OPA authorization interceptor added")
	}

	// Create gRPC server
	grpcServer := grpc.NewServer(
		grpc.ChainUnaryInterceptor(unaryInterceptors...),
	)

	// Register services
	billingServer := billing.NewServer(svc)
	billingv1.RegisterBillingServiceServer(grpcServer, billingServer)

	// Register health check
	healthServer := health.NewServer()
	grpc_health_v1.RegisterHealthServer(grpcServer, healthServer)
	healthServer.SetServingStatus("billing.v1.BillingService", grpc_health_v1.HealthCheckResponse_SERVING)

	reflection.Register(grpcServer)

	// Start listening
	addr := fmt.Sprintf(":%d", cfg.Service.GRPCPort)
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		logger.Fatal("failed to listen")
	}

	// Handle shutdown
	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh
		logger.Info("shutting down server...")
		healthServer.SetServingStatus("billing.v1.BillingService", grpc_health_v1.HealthCheckResponse_NOT_SERVING)
		grpcServer.GracefulStop()
		cancel()
	}()

	logger.Info("server started")

	if err := grpcServer.Serve(listener); err != nil {
		logger.Fatal("failed to serve")
	}
}
