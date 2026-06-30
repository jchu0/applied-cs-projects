package middleware

import (
	"context"
	"time"

	"github.com/google/uuid"
	"github.com/mlai/microservice-platform/pkg/logging"
	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

// UnaryServerInterceptors returns a chain of unary server interceptors
func UnaryServerInterceptors(logger *logging.Logger) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		// Chain: logging -> tenant -> recovery -> handler
		ctx = enrichContext(ctx)
		ctx = logger.ToContext(ctx)

		start := time.Now()
		resp, err := handler(ctx, req)
		duration := time.Since(start)

		// Log request
		code := codes.OK
		if err != nil {
			code = status.Code(err)
		}

		logger.WithContext(ctx).Info("grpc request",
			zap.String("method", info.FullMethod),
			zap.String("code", code.String()),
			zap.Duration("duration", duration),
			zap.Bool("error", err != nil),
		)

		return resp, err
	}
}

// StreamServerInterceptors returns a chain of stream server interceptors
func StreamServerInterceptors(logger *logging.Logger) grpc.StreamServerInterceptor {
	return func(srv interface{}, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
		start := time.Now()
		err := handler(srv, ss)
		duration := time.Since(start)

		code := codes.OK
		if err != nil {
			code = status.Code(err)
		}

		logger.Info("grpc stream",
			zap.String("method", info.FullMethod),
			zap.String("code", code.String()),
			zap.Duration("duration", duration),
			zap.Bool("error", err != nil),
		)

		return err
	}
}

// enrichContext adds request ID and extracts metadata from incoming context
func enrichContext(ctx context.Context) context.Context {
	// Generate request ID if not present
	md, ok := metadata.FromIncomingContext(ctx)
	if !ok {
		md = metadata.New(nil)
	}

	requestID := ""
	if vals := md.Get("x-request-id"); len(vals) > 0 {
		requestID = vals[0]
	} else {
		requestID = uuid.New().String()
	}
	ctx = context.WithValue(ctx, logging.RequestIDKey, requestID)

	// Extract trace ID
	if vals := md.Get("x-trace-id"); len(vals) > 0 {
		ctx = context.WithValue(ctx, logging.TraceIDKey, vals[0])
	}

	// Extract tenant ID
	if vals := md.Get("x-tenant-id"); len(vals) > 0 {
		ctx = context.WithValue(ctx, logging.TenantIDKey, vals[0])
	}

	// Extract user ID
	if vals := md.Get("x-user-id"); len(vals) > 0 {
		ctx = context.WithValue(ctx, logging.UserIDKey, vals[0])
	}

	return ctx
}

// TenantContextInterceptor extracts and validates tenant context
func TenantContextInterceptor() grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		md, ok := metadata.FromIncomingContext(ctx)
		if !ok {
			return nil, status.Error(codes.InvalidArgument, "missing metadata")
		}

		tenantID := md.Get("x-tenant-id")
		if len(tenantID) == 0 {
			return nil, status.Error(codes.InvalidArgument, "missing tenant ID")
		}

		ctx = context.WithValue(ctx, logging.TenantIDKey, tenantID[0])
		return handler(ctx, req)
	}
}

// RecoveryInterceptor recovers from panics
func RecoveryInterceptor(logger *logging.Logger) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (resp interface{}, err error) {
		defer func() {
			if r := recover(); r != nil {
				logger.WithContext(ctx).Error("panic recovered",
					zap.Any("panic", r),
					zap.String("method", info.FullMethod),
				)
				err = status.Error(codes.Internal, "internal server error")
			}
		}()

		return handler(ctx, req)
	}
}

// AuthInterceptor validates JWT tokens
func AuthInterceptor(validator TokenValidator, publicMethods map[string]bool) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		// Skip auth for public methods
		if publicMethods[info.FullMethod] {
			return handler(ctx, req)
		}

		md, ok := metadata.FromIncomingContext(ctx)
		if !ok {
			return nil, status.Error(codes.Unauthenticated, "missing metadata")
		}

		authHeader := md.Get("authorization")
		if len(authHeader) == 0 {
			return nil, status.Error(codes.Unauthenticated, "missing authorization header")
		}

		// Validate token
		claims, err := validator.ValidateToken(ctx, authHeader[0])
		if err != nil {
			return nil, status.Error(codes.Unauthenticated, "invalid token")
		}

		// Add claims to context
		ctx = context.WithValue(ctx, logging.UserIDKey, claims.UserID)
		ctx = context.WithValue(ctx, logging.TenantIDKey, claims.TenantID)

		return handler(ctx, req)
	}
}

// TokenValidator interface for token validation
type TokenValidator interface {
	ValidateToken(ctx context.Context, token string) (*TokenClaims, error)
}

// TokenClaims represents validated JWT claims
type TokenClaims struct {
	UserID      string
	TenantID    string
	Email       string
	Roles       []string
	Permissions []string
	SessionID   string
}
