package gateway

import (
	"context"
	"crypto/rsa"
	"crypto/x509"
	"encoding/pem"
	"errors"
	"net/http"
	"os"
	"strings"

	"github.com/golang-jwt/jwt/v5"
)

// JWTValidator validates JWT tokens
type JWTValidator struct {
	publicKey    *rsa.PublicKey
	issuer       string
	audience     string
	publicRoutes map[string]bool
}

// TokenClaims represents JWT claims
type TokenClaims struct {
	jwt.RegisteredClaims
	UserID      string   `json:"uid"`
	TenantID    string   `json:"tid"`
	Email       string   `json:"email"`
	Roles       []string `json:"roles"`
	Permissions []string `json:"perms"`
	SessionID   string   `json:"sid"`
	TokenType   string   `json:"type"`
}

// NewJWTValidator creates a new JWT validator
func NewJWTValidator(config JWTConfig) (*JWTValidator, error) {
	var publicKey *rsa.PublicKey

	if config.PublicKeyPath != "" {
		key, err := loadPublicKey(config.PublicKeyPath)
		if err != nil {
			return nil, err
		}
		publicKey = key
	}

	// Build public routes map
	publicRoutes := make(map[string]bool)
	for _, route := range config.PublicRoutes {
		publicRoutes[route] = true
	}

	return &JWTValidator{
		publicKey:    publicKey,
		issuer:       config.Issuer,
		audience:     config.Audience,
		publicRoutes: publicRoutes,
	}, nil
}

// ValidateToken validates a JWT token
func (v *JWTValidator) ValidateToken(tokenString string) (*TokenClaims, error) {
	if v.publicKey == nil {
		return nil, errors.New("public key not configured")
	}

	// Remove Bearer prefix
	tokenString = strings.TrimPrefix(tokenString, "Bearer ")
	tokenString = strings.TrimPrefix(tokenString, "bearer ")

	token, err := jwt.ParseWithClaims(tokenString, &TokenClaims{}, func(token *jwt.Token) (interface{}, error) {
		if _, ok := token.Method.(*jwt.SigningMethodRSA); !ok {
			return nil, errors.New("unexpected signing method")
		}
		return v.publicKey, nil
	})

	if err != nil {
		return nil, err
	}

	claims, ok := token.Claims.(*TokenClaims)
	if !ok || !token.Valid {
		return nil, errors.New("invalid token")
	}

	// Validate token type
	if claims.TokenType != "access" {
		return nil, errors.New("not an access token")
	}

	return claims, nil
}

// IsPublicRoute checks if a route is public
func (v *JWTValidator) IsPublicRoute(path string) bool {
	// Check exact match
	if v.publicRoutes[path] {
		return true
	}

	// Check prefix match for patterns like /api/v1/auth/*
	for route := range v.publicRoutes {
		if strings.HasSuffix(route, "*") {
			prefix := strings.TrimSuffix(route, "*")
			if strings.HasPrefix(path, prefix) {
				return true
			}
		}
	}

	return false
}

// Middleware returns HTTP middleware for JWT validation
func (v *JWTValidator) Middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Skip auth for public routes
		if v.IsPublicRoute(r.URL.Path) {
			next.ServeHTTP(w, r)
			return
		}

		// Get authorization header
		authHeader := r.Header.Get("Authorization")
		if authHeader == "" {
			http.Error(w, "Missing authorization header", http.StatusUnauthorized)
			return
		}

		// Validate token
		claims, err := v.ValidateToken(authHeader)
		if err != nil {
			http.Error(w, "Invalid token", http.StatusUnauthorized)
			return
		}

		// Add claims to request context
		ctx := r.Context()
		ctx = context.WithValue(ctx, "user_id", claims.UserID)
		ctx = context.WithValue(ctx, "tenant_id", claims.TenantID)
		ctx = context.WithValue(ctx, "email", claims.Email)
		ctx = context.WithValue(ctx, "roles", claims.Roles)
		ctx = context.WithValue(ctx, "permissions", claims.Permissions)

		// Also set headers for downstream services
		r.Header.Set("X-User-ID", claims.UserID)
		r.Header.Set("X-Tenant-ID", claims.TenantID)
		r.Header.Set("X-User-Email", claims.Email)

		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// loadPublicKey loads a public key from file
func loadPublicKey(path string) (*rsa.PublicKey, error) {
	keyData, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	block, _ := pem.Decode(keyData)
	if block == nil {
		return nil, errors.New("failed to decode PEM block")
	}

	pub, err := x509.ParsePKIXPublicKey(block.Bytes)
	if err != nil {
		return nil, err
	}

	rsaPub, ok := pub.(*rsa.PublicKey)
	if !ok {
		return nil, errors.New("not an RSA public key")
	}

	return rsaPub, nil
}
