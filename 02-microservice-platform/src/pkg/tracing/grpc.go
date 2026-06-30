package tracing

import (
	"context"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

// UnaryServerInterceptor returns a gRPC unary server interceptor for tracing
func UnaryServerInterceptor(serviceName string) grpc.UnaryServerInterceptor {
	tracer := otel.Tracer(serviceName)

	return func(
		ctx context.Context,
		req interface{},
		info *grpc.UnaryServerInfo,
		handler grpc.UnaryHandler,
	) (interface{}, error) {
		// Extract trace context from metadata
		md, ok := metadata.FromIncomingContext(ctx)
		if ok {
			ctx = otel.GetTextMapPropagator().Extract(ctx, &metadataCarrier{md: md})
		}

		// Start span
		ctx, span := tracer.Start(
			ctx,
			info.FullMethod,
			trace.WithSpanKind(trace.SpanKindServer),
			trace.WithAttributes(
				attribute.String("rpc.system", "grpc"),
				attribute.String("rpc.service", serviceName),
				attribute.String("rpc.method", info.FullMethod),
			),
		)
		defer span.End()

		// Handle request
		resp, err := handler(ctx, req)

		// Record result
		if err != nil {
			s, _ := status.FromError(err)
			span.SetStatus(codes.Error, s.Message())
			span.SetAttributes(
				attribute.String("rpc.grpc.status_code", s.Code().String()),
			)
			span.RecordError(err)
		} else {
			span.SetStatus(codes.Ok, "")
			span.SetAttributes(
				attribute.String("rpc.grpc.status_code", "OK"),
			)
		}

		return resp, err
	}
}

// StreamServerInterceptor returns a gRPC stream server interceptor for tracing
func StreamServerInterceptor(serviceName string) grpc.StreamServerInterceptor {
	tracer := otel.Tracer(serviceName)

	return func(
		srv interface{},
		ss grpc.ServerStream,
		info *grpc.StreamServerInfo,
		handler grpc.StreamHandler,
	) error {
		ctx := ss.Context()

		// Extract trace context from metadata
		md, ok := metadata.FromIncomingContext(ctx)
		if ok {
			ctx = otel.GetTextMapPropagator().Extract(ctx, &metadataCarrier{md: md})
		}

		// Start span
		ctx, span := tracer.Start(
			ctx,
			info.FullMethod,
			trace.WithSpanKind(trace.SpanKindServer),
			trace.WithAttributes(
				attribute.String("rpc.system", "grpc"),
				attribute.String("rpc.service", serviceName),
				attribute.String("rpc.method", info.FullMethod),
				attribute.Bool("rpc.grpc.is_client_stream", info.IsClientStream),
				attribute.Bool("rpc.grpc.is_server_stream", info.IsServerStream),
			),
		)
		defer span.End()

		// Wrap stream with traced context
		wrapped := &tracedServerStream{
			ServerStream: ss,
			ctx:          ctx,
		}

		// Handle request
		err := handler(srv, wrapped)

		// Record result
		if err != nil {
			s, _ := status.FromError(err)
			span.SetStatus(codes.Error, s.Message())
			span.RecordError(err)
		} else {
			span.SetStatus(codes.Ok, "")
		}

		return err
	}
}

// UnaryClientInterceptor returns a gRPC unary client interceptor for tracing
func UnaryClientInterceptor(serviceName string) grpc.UnaryClientInterceptor {
	tracer := otel.Tracer(serviceName)

	return func(
		ctx context.Context,
		method string,
		req, reply interface{},
		cc *grpc.ClientConn,
		invoker grpc.UnaryInvoker,
		opts ...grpc.CallOption,
	) error {
		// Start span
		ctx, span := tracer.Start(
			ctx,
			method,
			trace.WithSpanKind(trace.SpanKindClient),
			trace.WithAttributes(
				attribute.String("rpc.system", "grpc"),
				attribute.String("rpc.service", serviceName),
				attribute.String("rpc.method", method),
			),
		)
		defer span.End()

		// Inject trace context into metadata
		md, ok := metadata.FromOutgoingContext(ctx)
		if !ok {
			md = metadata.MD{}
		}
		otel.GetTextMapPropagator().Inject(ctx, &metadataCarrier{md: md})
		ctx = metadata.NewOutgoingContext(ctx, md)

		// Make the call
		err := invoker(ctx, method, req, reply, cc, opts...)

		// Record result
		if err != nil {
			s, _ := status.FromError(err)
			span.SetStatus(codes.Error, s.Message())
			span.RecordError(err)
		} else {
			span.SetStatus(codes.Ok, "")
		}

		return err
	}
}

// tracedServerStream wraps grpc.ServerStream with traced context
type tracedServerStream struct {
	grpc.ServerStream
	ctx context.Context
}

func (s *tracedServerStream) Context() context.Context {
	return s.ctx
}

// metadataCarrier adapts metadata.MD to TextMapCarrier
type metadataCarrier struct {
	md metadata.MD
}

func (c *metadataCarrier) Get(key string) string {
	values := c.md.Get(key)
	if len(values) == 0 {
		return ""
	}
	return values[0]
}

func (c *metadataCarrier) Set(key, value string) {
	c.md.Set(key, value)
}

func (c *metadataCarrier) Keys() []string {
	keys := make([]string, 0, len(c.md))
	for k := range c.md {
		keys = append(keys, k)
	}
	return keys
}
