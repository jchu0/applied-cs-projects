// Package opa provides middleware for authorization using Open Policy Agent.
package opa

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

// HTTPMiddleware wraps an HTTP handler with OPA authorization.
func HTTPMiddleware(pe *PolicyEngine, extractSubject SubjectExtractor) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			subject, err := extractSubject(r)
			if err != nil {
				http.Error(w, "Unauthorized", http.StatusUnauthorized)
				return
			}

			input := AuthzInput{
				Subject: subject,
				Action:  methodToAction(r.Method),
				Resource: Resource{
					Type: pathToResourceType(r.URL.Path),
					ID:   extractResourceID(r.URL.Path),
				},
				Context: AuthzCtx{
					IP:        getClientIP(r),
					UserAgent: r.UserAgent(),
					Time:      time.Now(),
					Headers:   extractHeaders(r),
				},
			}

			result, err := pe.Authorize(r.Context(), input)
			if err != nil {
				http.Error(w, "Authorization error", http.StatusInternalServerError)
				return
			}

			if !result.Allow {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusForbidden)
				json.NewEncoder(w).Encode(map[string]interface{}{
					"error":   "Forbidden",
					"reasons": result.Deny,
				})
				return
			}

			next.ServeHTTP(w, r)
		})
	}
}

// SubjectExtractor extracts the subject from an HTTP request.
type SubjectExtractor func(*http.Request) (Subject, error)

// DefaultSubjectExtractor creates a subject extractor that uses JWT claims.
func DefaultSubjectExtractor(claimsKey string) SubjectExtractor {
	return func(r *http.Request) (Subject, error) {
		claims, ok := r.Context().Value(claimsKey).(map[string]interface{})
		if !ok {
			return Subject{}, nil
		}

		subject := Subject{}

		if userID, ok := claims["sub"].(string); ok {
			subject.UserID = userID
		}
		if tenantID, ok := claims["tenant_id"].(string); ok {
			subject.TenantID = tenantID
		}
		if roles, ok := claims["roles"].([]interface{}); ok {
			for _, role := range roles {
				if s, ok := role.(string); ok {
					subject.Roles = append(subject.Roles, s)
				}
			}
		}
		if groups, ok := claims["groups"].([]interface{}); ok {
			for _, group := range groups {
				if s, ok := group.(string); ok {
					subject.Groups = append(subject.Groups, s)
				}
			}
		}

		return subject, nil
	}
}

// GRPCUnaryInterceptor creates a gRPC unary interceptor for OPA authorization.
func GRPCUnaryInterceptor(pe *PolicyEngine, extractSubject GRPCSubjectExtractor) grpc.UnaryServerInterceptor {
	return func(
		ctx context.Context,
		req interface{},
		info *grpc.UnaryServerInfo,
		handler grpc.UnaryHandler,
	) (interface{}, error) {
		subject, err := extractSubject(ctx)
		if err != nil {
			return nil, status.Error(codes.Unauthenticated, "failed to extract subject")
		}

		input := AuthzInput{
			Subject: subject,
			Action:  grpcMethodToAction(info.FullMethod),
			Resource: Resource{
				Type: grpcMethodToResourceType(info.FullMethod),
			},
			Context: AuthzCtx{
				Time: time.Now(),
			},
		}

		// Extract resource from request if available
		if resourceReq, ok := req.(interface{ GetResourceId() string }); ok {
			input.Resource.ID = resourceReq.GetResourceId()
		}
		if tenantReq, ok := req.(interface{ GetTenantId() string }); ok {
			input.Resource.Tenant = tenantReq.GetTenantId()
		}

		result, err := pe.Authorize(ctx, input)
		if err != nil {
			return nil, status.Error(codes.Internal, "authorization error")
		}

		if !result.Allow {
			return nil, status.Error(codes.PermissionDenied, strings.Join(result.Deny, "; "))
		}

		return handler(ctx, req)
	}
}

// GRPCStreamInterceptor creates a gRPC stream interceptor for OPA authorization.
func GRPCStreamInterceptor(pe *PolicyEngine, extractSubject GRPCSubjectExtractor) grpc.StreamServerInterceptor {
	return func(
		srv interface{},
		ss grpc.ServerStream,
		info *grpc.StreamServerInfo,
		handler grpc.StreamHandler,
	) error {
		ctx := ss.Context()
		subject, err := extractSubject(ctx)
		if err != nil {
			return status.Error(codes.Unauthenticated, "failed to extract subject")
		}

		input := AuthzInput{
			Subject: subject,
			Action:  grpcMethodToAction(info.FullMethod),
			Resource: Resource{
				Type: grpcMethodToResourceType(info.FullMethod),
			},
			Context: AuthzCtx{
				Time: time.Now(),
			},
		}

		result, err := pe.Authorize(ctx, input)
		if err != nil {
			return status.Error(codes.Internal, "authorization error")
		}

		if !result.Allow {
			return status.Error(codes.PermissionDenied, strings.Join(result.Deny, "; "))
		}

		return handler(srv, ss)
	}
}

// GRPCSubjectExtractor extracts subject from gRPC context.
type GRPCSubjectExtractor func(context.Context) (Subject, error)

// DefaultGRPCSubjectExtractor creates a default subject extractor from metadata.
func DefaultGRPCSubjectExtractor() GRPCSubjectExtractor {
	return func(ctx context.Context) (Subject, error) {
		md, ok := metadata.FromIncomingContext(ctx)
		if !ok {
			return Subject{}, nil
		}

		subject := Subject{}

		if userIDs := md.Get("x-user-id"); len(userIDs) > 0 {
			subject.UserID = userIDs[0]
		}
		if tenantIDs := md.Get("x-tenant-id"); len(tenantIDs) > 0 {
			subject.TenantID = tenantIDs[0]
		}
		if roles := md.Get("x-roles"); len(roles) > 0 {
			subject.Roles = strings.Split(roles[0], ",")
		}

		return subject, nil
	}
}

// Helper functions

func methodToAction(method string) string {
	switch method {
	case http.MethodGet, http.MethodHead:
		return "read"
	case http.MethodPost:
		return "create"
	case http.MethodPut, http.MethodPatch:
		return "update"
	case http.MethodDelete:
		return "delete"
	default:
		return "unknown"
	}
}

func pathToResourceType(path string) string {
	parts := strings.Split(strings.Trim(path, "/"), "/")
	if len(parts) > 0 {
		// Strip API version prefix
		if strings.HasPrefix(parts[0], "v") && len(parts) > 1 {
			return parts[1]
		}
		return parts[0]
	}
	return "unknown"
}

func extractResourceID(path string) string {
	parts := strings.Split(strings.Trim(path, "/"), "/")
	if len(parts) >= 2 {
		return parts[len(parts)-1]
	}
	return ""
}

func grpcMethodToAction(method string) string {
	parts := strings.Split(method, "/")
	if len(parts) < 2 {
		return "unknown"
	}

	methodName := parts[len(parts)-1]
	switch {
	case strings.HasPrefix(methodName, "Get"), strings.HasPrefix(methodName, "List"):
		return "read"
	case strings.HasPrefix(methodName, "Create"):
		return "create"
	case strings.HasPrefix(methodName, "Update"):
		return "update"
	case strings.HasPrefix(methodName, "Delete"):
		return "delete"
	default:
		return strings.ToLower(methodName)
	}
}

func grpcMethodToResourceType(method string) string {
	parts := strings.Split(method, "/")
	if len(parts) >= 2 {
		// Extract service name
		servicePart := parts[len(parts)-2]
		// Convert ServiceName to resource_type
		return strings.ToLower(strings.TrimSuffix(servicePart, "Service"))
	}
	return "unknown"
}

func getClientIP(r *http.Request) string {
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		parts := strings.Split(xff, ",")
		return strings.TrimSpace(parts[0])
	}
	if xri := r.Header.Get("X-Real-IP"); xri != "" {
		return xri
	}
	return r.RemoteAddr
}

func extractHeaders(r *http.Request) map[string]string {
	headers := make(map[string]string)
	for _, key := range []string{
		"X-Request-ID",
		"X-Correlation-ID",
		"X-Tenant-ID",
		"Authorization",
	} {
		if val := r.Header.Get(key); val != "" {
			headers[key] = val
		}
	}
	return headers
}
