package resilience

import (
	"context"

	"google.golang.org/grpc"
)

// UnaryClientInterceptor returns a gRPC unary client interceptor with circuit breaker protection.
// It uses the full method name to get/create a circuit breaker from the registry.
func UnaryClientInterceptor(registry *CircuitBreakerRegistry) grpc.UnaryClientInterceptor {
	return func(
		ctx context.Context,
		method string,
		req, reply interface{},
		cc *grpc.ClientConn,
		invoker grpc.UnaryInvoker,
		opts ...grpc.CallOption,
	) error {
		cb := registry.Get(method)
		return cb.Execute(func() error {
			return invoker(ctx, method, req, reply, cc, opts...)
		})
	}
}

// StreamClientInterceptor returns a gRPC stream client interceptor with circuit breaker protection.
func StreamClientInterceptor(registry *CircuitBreakerRegistry) grpc.StreamClientInterceptor {
	return func(
		ctx context.Context,
		desc *grpc.StreamDesc,
		cc *grpc.ClientConn,
		method string,
		streamer grpc.Streamer,
		opts ...grpc.CallOption,
	) (grpc.ClientStream, error) {
		cb := registry.Get(method)
		var stream grpc.ClientStream
		err := cb.Execute(func() error {
			var innerErr error
			stream, innerErr = streamer(ctx, desc, cc, method, opts...)
			return innerErr
		})
		return stream, err
	}
}

// UnaryServerInterceptor returns a gRPC unary server interceptor with circuit breaker protection.
// This is useful for protecting downstream services from overwhelming the server.
func UnaryServerInterceptor(registry *CircuitBreakerRegistry) grpc.UnaryServerInterceptor {
	return func(
		ctx context.Context,
		req interface{},
		info *grpc.UnaryServerInfo,
		handler grpc.UnaryHandler,
	) (interface{}, error) {
		cb := registry.Get(info.FullMethod)
		var resp interface{}
		err := cb.Execute(func() error {
			var innerErr error
			resp, innerErr = handler(ctx, req)
			return innerErr
		})
		return resp, err
	}
}

// StreamServerInterceptor returns a gRPC stream server interceptor with circuit breaker protection.
func StreamServerInterceptor(registry *CircuitBreakerRegistry) grpc.StreamServerInterceptor {
	return func(
		srv interface{},
		ss grpc.ServerStream,
		info *grpc.StreamServerInfo,
		handler grpc.StreamHandler,
	) error {
		cb := registry.Get(info.FullMethod)
		return cb.Execute(func() error {
			return handler(srv, ss)
		})
	}
}
