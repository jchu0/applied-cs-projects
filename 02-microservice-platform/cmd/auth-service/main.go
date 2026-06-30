package main

import (
	"context"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"

	"github.com/mlai/microservice-platform/internal/auth"
	"github.com/mlai/microservice-platform/internal/common"
	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/reflection"
)

func main() {
	// Load configuration
	cfg := common.LoadConfig("auth")
	logger := common.NewLogger("auth-service")

	logger.Info("starting auth service",
		"grpc_port", cfg.GRPCPort,
		"environment", cfg.Environment,
	)

	// Initialize Redis session store
	sessionRepo, err := auth.NewSessionRepository(cfg.RedisURL)
	if err != nil {
		log.Fatalf("failed to connect to Redis: %v", err)
	}
	defer sessionRepo.Close()

	logger.Info("connected to Redis")

	// Initialize JWT manager
	jwtManager, err := auth.NewJWTManager(
		nil, nil, // Use generated keys for development
		cfg.AccessTokenExpiry,
		cfg.RefreshTokenExpiry,
	)
	if err != nil {
		log.Fatalf("failed to initialize JWT manager: %v", err)
	}

	logger.Info("JWT manager initialized")

	// Initialize user service client (placeholder)
	userClient := &mockUserClient{}

	// Initialize auth service
	svc := auth.NewService(sessionRepo, jwtManager, userClient, logger)

	// Create gRPC server
	grpcServer := grpc.NewServer(
		grpc.UnaryInterceptor(loggingInterceptor(logger)),
	)

	// Register auth service (placeholder - would use generated pb)
	_ = svc // Service would be registered here with generated protobuf

	// Register health service
	healthServer := health.NewServer()
	grpc_health_v1.RegisterHealthServer(grpcServer, healthServer)
	healthServer.SetServingStatus("auth.v1.AuthService", grpc_health_v1.HealthCheckResponse_SERVING)

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

		logger.Info("shutting down auth service")
		healthServer.SetServingStatus("auth.v1.AuthService", grpc_health_v1.HealthCheckResponse_NOT_SERVING)
		grpcServer.GracefulStop()
	}()

	logger.Info("auth service started", "address", listener.Addr().String())

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

// mockUserClient is a placeholder for the real user service client
type mockUserClient struct{}

func (c *mockUserClient) GetUserByEmail(ctx context.Context, email, tenantID string) (*auth.UserInfo, error) {
	// Placeholder implementation
	return nil, fmt.Errorf("not implemented")
}

func (c *mockUserClient) GetUserByID(ctx context.Context, id, tenantID string) (*auth.UserInfo, error) {
	// Placeholder implementation
	return nil, fmt.Errorf("not implemented")
}

func (c *mockUserClient) CreateUser(ctx context.Context, req *auth.CreateUserReq) (*auth.UserInfo, error) {
	// Placeholder implementation
	return nil, fmt.Errorf("not implemented")
}

func init() {
	// Set default timezone
	if tz := os.Getenv("TZ"); tz == "" {
		os.Setenv("TZ", "UTC")
	}
}
