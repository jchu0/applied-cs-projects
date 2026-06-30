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

	"github.com/mlai/microservice-platform/internal/notification"
	"github.com/mlai/microservice-platform/pkg/config"
	"github.com/mlai/microservice-platform/pkg/database"
	"github.com/mlai/microservice-platform/pkg/events"
	"github.com/mlai/microservice-platform/pkg/logging"
	"github.com/mlai/microservice-platform/pkg/metrics"
	"github.com/mlai/microservice-platform/pkg/middleware"
	notifv1 "github.com/mlai/microservice-platform/pkg/pb/notification/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/reflection"
)

func main() {
	configPath := flag.String("config", "", "Path to config file")
	flag.Parse()

	cfg := notification.DefaultConfig()
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

	logger.Info("starting notification service")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Connect to database
	db, err := database.NewPostgresDB(ctx, &cfg.Database)
	if err != nil {
		logger.Fatal("failed to connect to database")
	}
	defer db.Close()
	logger.Info("connected to database")

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

	// Create email sender
	sender := notification.NewSMTPSender(
		cfg.Email.SMTPHost,
		cfg.Email.SMTPPort,
		cfg.Email.SMTPUser,
		cfg.Email.SMTPPass,
		cfg.Email.FromEmail,
		cfg.Email.FromName,
	)

	// Initialize metrics
	var m *metrics.Metrics
	if cfg.Metrics.Enabled {
		m = metrics.New(metrics.Config{
			ServiceName: cfg.Service.Name,
			Namespace:   "microservice",
			Subsystem:   "notification",
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
	repo := notification.NewRepository(db.Pool())
	svc := notification.NewService(repo, publisher, sender, logger)

	// Build gRPC interceptors
	unaryInterceptors := []grpc.UnaryServerInterceptor{
		middleware.RecoveryInterceptor(logger),
		middleware.UnaryServerInterceptors(logger),
	}

	// Add metrics interceptor if enabled
	if m != nil {
		unaryInterceptors = append(unaryInterceptors, m.UnaryServerInterceptor())
	}

	// Create gRPC server
	grpcServer := grpc.NewServer(
		grpc.ChainUnaryInterceptor(unaryInterceptors...),
	)

	// Register services
	notifServer := notification.NewServer(svc)
	notifv1.RegisterNotificationServiceServer(grpcServer, notifServer)

	// Register health check
	healthServer := health.NewServer()
	grpc_health_v1.RegisterHealthServer(grpcServer, healthServer)
	healthServer.SetServingStatus("notification.v1.NotificationService", grpc_health_v1.HealthCheckResponse_SERVING)

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
		healthServer.SetServingStatus("notification.v1.NotificationService", grpc_health_v1.HealthCheckResponse_NOT_SERVING)
		grpcServer.GracefulStop()
		cancel()
	}()

	logger.Info("server started")

	if err := grpcServer.Serve(listener); err != nil {
		logger.Fatal("failed to serve")
	}
}
