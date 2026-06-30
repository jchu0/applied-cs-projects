package metrics

import (
	"context"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/status"
)

// UnaryServerInterceptor returns a gRPC unary server interceptor for metrics
func (m *Metrics) UnaryServerInterceptor() grpc.UnaryServerInterceptor {
	return func(
		ctx context.Context,
		req interface{},
		info *grpc.UnaryServerInfo,
		handler grpc.UnaryHandler,
	) (interface{}, error) {
		start := time.Now()

		resp, err := handler(ctx, req)

		duration := time.Since(start)
		code := status.Code(err).String()

		m.RecordGRPCRequest(info.FullMethod, code, duration)

		return resp, err
	}
}

// StreamServerInterceptor returns a gRPC stream server interceptor for metrics
func (m *Metrics) StreamServerInterceptor() grpc.StreamServerInterceptor {
	return func(
		srv interface{},
		ss grpc.ServerStream,
		info *grpc.StreamServerInfo,
		handler grpc.StreamHandler,
	) error {
		start := time.Now()

		err := handler(srv, ss)

		duration := time.Since(start)
		code := status.Code(err).String()

		m.RecordGRPCRequest(info.FullMethod, code, duration)

		return err
	}
}

// UnaryClientInterceptor returns a gRPC unary client interceptor for metrics
func (m *Metrics) UnaryClientInterceptor() grpc.UnaryClientInterceptor {
	return func(
		ctx context.Context,
		method string,
		req, reply interface{},
		cc *grpc.ClientConn,
		invoker grpc.UnaryInvoker,
		opts ...grpc.CallOption,
	) error {
		start := time.Now()

		err := invoker(ctx, method, req, reply, cc, opts...)

		duration := time.Since(start)
		code := status.Code(err).String()

		m.RecordGRPCRequest(method, code, duration)

		return err
	}
}

// StreamClientInterceptor returns a gRPC stream client interceptor for metrics
func (m *Metrics) StreamClientInterceptor() grpc.StreamClientInterceptor {
	return func(
		ctx context.Context,
		desc *grpc.StreamDesc,
		cc *grpc.ClientConn,
		method string,
		streamer grpc.Streamer,
		opts ...grpc.CallOption,
	) (grpc.ClientStream, error) {
		start := time.Now()

		stream, err := streamer(ctx, desc, cc, method, opts...)

		duration := time.Since(start)
		code := status.Code(err).String()

		m.RecordGRPCRequest(method, code, duration)

		return stream, err
	}
}
