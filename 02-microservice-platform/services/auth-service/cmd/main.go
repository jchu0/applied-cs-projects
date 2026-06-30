package main

import (
	"context"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"

	"github.com/project/microservices/auth-service/internal/config"
	"github.com/project/microservices/auth-service/internal/handlers"
	"github.com/project/microservices/auth-service/internal/jwt"
	"github.com/project/microservices/auth-service/internal/repository"
	"github.com/redis/go-redis/v9"
	"go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/jaeger"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.21.0"
	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/reflection"
)

func main() {
	// Load configuration
	cfg := config.Load()

	// Initialize tracer
	shutdown, err := initTracer(cfg)
	if err != nil {
		log.Fatalf("Failed to initialize tracer: %v", err)
	}
	defer shutdown()

	// Connect to Redis
	redisOpts, err := redis.ParseURL(cfg.RedisURL)
	if err != nil {
		log.Fatalf("Failed to parse Redis URL: %v", err)
	}
	redisClient := redis.NewClient(redisOpts)

	// Verify Redis connection
	if err := redisClient.Ping(context.Background()).Err(); err != nil {
		log.Fatalf("Failed to connect to Redis: %v", err)
	}
	log.Println("Connected to Redis")
	defer redisClient.Close()

	// Initialize JWT service
	jwtService, err := jwt.NewService(cfg)
	if err != nil {
		log.Fatalf("Failed to initialize JWT service: %v", err)
	}

	// Create repositories
	sessionRepo := repository.NewSessionRepository(redisClient)

	// Create user service client
	// In production, this would be a gRPC client to the user service
	userClient := NewMockUserClient()

	// Create gRPC server with OpenTelemetry interceptors
	grpcServer := grpc.NewServer(
		grpc.UnaryInterceptor(otelgrpc.UnaryServerInterceptor()),
		grpc.StreamInterceptor(otelgrpc.StreamServerInterceptor()),
	)

	// Register services
	authService := handlers.NewAuthServiceServer(cfg, jwtService, sessionRepo, userClient)
	// Note: In production, you'd register the generated protobuf service here
	// authpb.RegisterAuthServiceServer(grpcServer, authService)
	_ = authService // Placeholder until proto generation

	// Register health check
	healthServer := health.NewServer()
	grpc_health_v1.RegisterHealthServer(grpcServer, healthServer)
	healthServer.SetServingStatus("auth-service", grpc_health_v1.HealthCheckResponse_SERVING)

	// Enable reflection for development
	reflection.Register(grpcServer)

	// Start gRPC server
	listener, err := net.Listen("tcp", fmt.Sprintf(":%s", cfg.GRPCPort))
	if err != nil {
		log.Fatalf("Failed to listen on port %s: %v", cfg.GRPCPort, err)
	}

	// Graceful shutdown
	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh

		log.Println("Shutting down gracefully...")
		healthServer.SetServingStatus("auth-service", grpc_health_v1.HealthCheckResponse_NOT_SERVING)
		grpcServer.GracefulStop()
	}()

	log.Printf("Auth service starting on port %s", cfg.GRPCPort)
	if err := grpcServer.Serve(listener); err != nil {
		log.Fatalf("Failed to serve: %v", err)
	}
}

func initTracer(cfg *config.Config) (func(), error) {
	exporter, err := jaeger.New(jaeger.WithCollectorEndpoint(
		jaeger.WithEndpoint(cfg.JaegerEndpoint),
	))
	if err != nil {
		return nil, fmt.Errorf("failed to create Jaeger exporter: %w", err)
	}

	tp := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exporter),
		sdktrace.WithResource(resource.NewWithAttributes(
			semconv.SchemaURL,
			semconv.ServiceNameKey.String(cfg.ServiceName),
		)),
	)

	otel.SetTracerProvider(tp)

	return func() {
		if err := tp.Shutdown(context.Background()); err != nil {
			log.Printf("Error shutting down tracer provider: %v", err)
		}
	}, nil
}

// MockUserClient is a temporary mock for user service client
// In production, this would be replaced with a real gRPC client
type MockUserClient struct{}

func NewMockUserClient() *MockUserClient {
	return &MockUserClient{}
}

func (c *MockUserClient) GetUserByEmail(ctx context.Context, email, tenantID string) (*handlers.UserData, error) {
	// This would call the actual user service
	return nil, handlers.ErrUserNotFound
}

func (c *MockUserClient) CreateUser(ctx context.Context, input *handlers.CreateUserInput) (*handlers.UserData, error) {
	// This would call the actual user service
	return nil, fmt.Errorf("not implemented")
}
