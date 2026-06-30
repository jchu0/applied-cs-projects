package gateway

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/mlai/microservice-platform/pkg/logging"
	"go.uber.org/zap"
)

// Router handles request routing to backend services
type Router struct {
	services map[string]*httputil.ReverseProxy
	logger   *logging.Logger
}

// NewRouter creates a new router
func NewRouter(config ServicesConfig, logger *logging.Logger) *Router {
	router := &Router{
		services: make(map[string]*httputil.ReverseProxy),
		logger:   logger,
	}

	// Create reverse proxies for each service
	serviceURLs := map[string]string{
		"user":         config.UserService,
		"auth":         config.AuthService,
		"billing":      config.BillingService,
		"notification": config.NotificationService,
	}

	for name, addr := range serviceURLs {
		if addr == "" {
			continue
		}

		target, err := url.Parse("http://" + addr)
		if err != nil {
			logger.Error("failed to parse service URL",
				zap.String("service", name),
				zap.Error(err),
			)
			continue
		}

		proxy := httputil.NewSingleHostReverseProxy(target)
		proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
			logger.Error("proxy error",
				zap.String("service", name),
				zap.Error(err),
			)
			http.Error(w, "Service unavailable", http.StatusServiceUnavailable)
		}

		router.services[name] = proxy
	}

	return router
}

// ServeHTTP implements http.Handler
func (r *Router) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	// Add request ID if not present
	requestID := req.Header.Get("X-Request-ID")
	if requestID == "" {
		requestID = uuid.New().String()
		req.Header.Set("X-Request-ID", requestID)
	}
	w.Header().Set("X-Request-ID", requestID)

	// Log request
	start := time.Now()
	defer func() {
		r.logger.Info("request",
			zap.String("method", req.Method),
			zap.String("path", req.URL.Path),
			zap.String("request_id", requestID),
			zap.Duration("duration", time.Since(start)),
		)
	}()

	// Route based on path
	path := req.URL.Path

	switch {
	case strings.HasPrefix(path, "/api/v1/users"):
		r.proxyTo("user", w, req)
	case strings.HasPrefix(path, "/api/v1/auth"):
		r.proxyTo("auth", w, req)
	case strings.HasPrefix(path, "/api/v1/billing"):
		r.proxyTo("billing", w, req)
	case strings.HasPrefix(path, "/api/v1/notifications"):
		r.proxyTo("notification", w, req)
	case path == "/api/v1/health" || path == "/health":
		r.healthCheck(w, req)
	default:
		http.NotFound(w, req)
	}
}

// proxyTo proxies the request to the specified service
func (r *Router) proxyTo(service string, w http.ResponseWriter, req *http.Request) {
	proxy, ok := r.services[service]
	if !ok {
		r.logger.Error("service not found", zap.String("service", service))
		http.Error(w, "Service not found", http.StatusNotFound)
		return
	}

	// Strip API prefix for backend services
	// /api/v1/users/123 -> /users/123
	req.URL.Path = strings.TrimPrefix(req.URL.Path, "/api/v1")

	proxy.ServeHTTP(w, req)
}

// healthCheck returns gateway health status
func (r *Router) healthCheck(w http.ResponseWriter, req *http.Request) {
	health := map[string]interface{}{
		"status":    "healthy",
		"timestamp": time.Now().UTC().Format(time.RFC3339),
		"services":  make(map[string]string),
	}

	// Check each service
	for name := range r.services {
		health["services"].(map[string]string)[name] = "unknown"
		// TODO: Add actual health checks to services
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(health)
}

// LoggingMiddleware logs all requests
func LoggingMiddleware(logger *logging.Logger, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		// Wrap response writer to capture status code
		wrapped := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}

		next.ServeHTTP(wrapped, r)

		logger.Info("http request",
			zap.String("method", r.Method),
			zap.String("path", r.URL.Path),
			zap.Int("status", wrapped.statusCode),
			zap.Duration("duration", time.Since(start)),
			zap.String("remote_addr", r.RemoteAddr),
			zap.String("user_agent", r.UserAgent()),
		)
	})
}

// responseWriter wraps http.ResponseWriter to capture status code
type responseWriter struct {
	http.ResponseWriter
	statusCode int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}

// CORSMiddleware handles CORS
func CORSMiddleware(config CORSConfig, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")

		// Check if origin is allowed
		allowed := false
		for _, o := range config.AllowedOrigins {
			if o == "*" || o == origin {
				allowed = true
				break
			}
		}

		if allowed {
			w.Header().Set("Access-Control-Allow-Origin", origin)
			w.Header().Set("Access-Control-Allow-Methods", strings.Join(config.AllowedMethods, ", "))
			w.Header().Set("Access-Control-Allow-Headers", strings.Join(config.AllowedHeaders, ", "))
			w.Header().Set("Access-Control-Expose-Headers", strings.Join(config.ExposedHeaders, ", "))
			if config.AllowCredentials {
				w.Header().Set("Access-Control-Allow-Credentials", "true")
			}
			w.Header().Set("Access-Control-Max-Age", string(rune(config.MaxAge)))
		}

		// Handle preflight
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// RecoveryMiddleware recovers from panics
func RecoveryMiddleware(logger *logging.Logger, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if err := recover(); err != nil {
				logger.Error("panic recovered",
					zap.Any("error", err),
					zap.String("path", r.URL.Path),
				)
				http.Error(w, "Internal server error", http.StatusInternalServerError)
			}
		}()

		next.ServeHTTP(w, r)
	})
}

// gRPC to REST translation helper
func translateGRPCToREST(grpcResp []byte) ([]byte, error) {
	// For now, just pass through
	// In production, would translate protobuf to JSON
	return grpcResp, nil
}

// readBody reads and returns the request body
func readBody(r *http.Request) ([]byte, error) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, err
	}
	r.Body.Close()
	return body, nil
}
