package main

import (
	"context"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"

	"github.com/mlai/microservice-platform/internal/common"
	"github.com/mlai/microservice-platform/internal/user"
	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/reflection"
)

func main() {
	// Load configuration
	cfg := common.LoadConfig("users")
	logger := common.NewLogger("user-service")

	logger.Info("starting user service",
		"grpc_port", cfg.GRPCPort,
		"environment", cfg.Environment,
	)

	// Initialize database
	ctx := context.Background()
	db, err := common.NewDatabase(ctx, cfg.DatabaseURL)
	if err != nil {
		log.Fatalf("failed to connect to database: %v", err)
	}
	defer db.Close()

	logger.Info("connected to database")

	// Initialize repository and service
	repo := user.NewRepository(db.Pool)
	svc := user.NewService(repo, logger)

	// Create gRPC server
	grpcServer := grpc.NewServer(
		grpc.UnaryInterceptor(loggingInterceptor(logger)),
	)

	// Register user service (placeholder - would use generated pb)
	_ = svc // Service would be registered here with generated protobuf

	// Register health service
	healthServer := health.NewServer()
	grpc_health_v1.RegisterHealthServer(grpcServer, healthServer)
	healthServer.SetServingStatus("user.v1.UserService", grpc_health_v1.HealthCheckResponse_SERVING)

	// Enable reflection for development
	if cfg.Environment != "production" {
		reflection.Register(grpcServer)
	}

	// Start gRPC server
	listener, err := net.Listen("tcp", ":"+cfg.GRPCPort)
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	// Graceful shutdown
	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh

		logger.Info("shutting down user service")
		healthServer.SetServingStatus("user.v1.UserService", grpc_health_v1.HealthCheckResponse_NOT_SERVING)
		grpcServer.GracefulStop()
	}()

	logger.Info("user service started", "address", listener.Addr().String())

	if err := grpcServer.Serve(listener); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}

// loggingInterceptor logs gRPC requests
func loggingInterceptor(logger *common.Logger) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		logger.Info("grpc request", "method", info.FullMethod)

		resp, err := handler(ctx, req)

		if err != nil {
			logger.Error("grpc error", "method", info.FullMethod, "error", err)
		}

		return resp, err
	}
}

func init() {
	// Set default timezone
	if tz := os.Getenv("TZ"); tz == "" {
		os.Setenv("TZ", "UTC")
	}
}

// Placeholder for service registration
func registerUserService(s *grpc.Server, svc *user.Service) {
	// This would be:
	// pb.RegisterUserServiceServer(s, svc)
	fmt.Println("User service registered (placeholder)")
}
