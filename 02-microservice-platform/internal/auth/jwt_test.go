package auth

import (
	"testing"
	"time"
)

func TestJWTManager_GenerateAndValidate(t *testing.T) {
	// Create JWT manager with generated keys
	manager, err := NewJWTManager(nil, nil, 15*time.Minute, 7*24*time.Hour)
	if err != nil {
		t.Fatalf("failed to create JWT manager: %v", err)
	}

	// Test data
	userID := "user-123"
	tenantID := "tenant-456"
	email := "test@example.com"
	sessionID := "session-789"
	roles := []string{"admin", "user"}
	permissions := []string{"read", "write"}

	// Generate token pair
	tokenPair, err := manager.GenerateTokenPair(userID, tenantID, email, sessionID, roles, permissions)
	if err != nil {
		t.Fatalf("failed to generate token pair: %v", err)
	}

	if tokenPair.AccessToken == "" {
		t.Error("access token is empty")
	}
	if tokenPair.RefreshToken == "" {
		t.Error("refresh token is empty")
	}
	if tokenPair.ExpiresIn != int64((15 * time.Minute).Seconds()) {
		t.Errorf("expected expires_in %d, got %d", int64((15*time.Minute).Seconds()), tokenPair.ExpiresIn)
	}

	// Validate access token
	claims, err := manager.ValidateToken(tokenPair.AccessToken)
	if err != nil {
		t.Fatalf("failed to validate access token: %v", err)
	}

	// Check claims
	if claims.UserID != userID {
		t.Errorf("expected user_id %s, got %s", userID, claims.UserID)
	}
	if claims.TenantID != tenantID {
		t.Errorf("expected tenant_id %s, got %s", tenantID, claims.TenantID)
	}
	if claims.Email != email {
		t.Errorf("expected email %s, got %s", email, claims.Email)
	}
	if claims.SessionID != sessionID {
		t.Errorf("expected session_id %s, got %s", sessionID, claims.SessionID)
	}
	if claims.TokenType != "access" {
		t.Errorf("expected token_type access, got %s", claims.TokenType)
	}
	if len(claims.Roles) != len(roles) {
		t.Errorf("expected %d roles, got %d", len(roles), len(claims.Roles))
	}
	if len(claims.Permissions) != len(permissions) {
		t.Errorf("expected %d permissions, got %d", len(permissions), len(claims.Permissions))
	}

	// Validate refresh token
	refreshClaims, err := manager.ValidateToken(tokenPair.RefreshToken)
	if err != nil {
		t.Fatalf("failed to validate refresh token: %v", err)
	}

	if refreshClaims.TokenType != "refresh" {
		t.Errorf("expected token_type refresh, got %s", refreshClaims.TokenType)
	}
}

func TestJWTManager_InvalidToken(t *testing.T) {
	manager, err := NewJWTManager(nil, nil, 15*time.Minute, 7*24*time.Hour)
	if err != nil {
		t.Fatalf("failed to create JWT manager: %v", err)
	}

	// Test invalid token
	_, err = manager.ValidateToken("invalid-token")
	if err == nil {
		t.Error("expected error for invalid token")
	}
	if err != ErrInvalidToken {
		t.Errorf("expected ErrInvalidToken, got %v", err)
	}
}

func TestJWTManager_ExpiredToken(t *testing.T) {
	// Create manager with very short expiry
	manager, err := NewJWTManager(nil, nil, 1*time.Millisecond, 7*24*time.Hour)
	if err != nil {
		t.Fatalf("failed to create JWT manager: %v", err)
	}

	// Generate token
	tokenPair, err := manager.GenerateTokenPair("user", "tenant", "email", "session", nil, nil)
	if err != nil {
		t.Fatalf("failed to generate token: %v", err)
	}

	// Wait for token to expire
	time.Sleep(10 * time.Millisecond)

	// Validate expired token
	_, err = manager.ValidateToken(tokenPair.AccessToken)
	if err == nil {
		t.Error("expected error for expired token")
	}
	if err != ErrExpiredToken {
		t.Errorf("expected ErrExpiredToken, got %v", err)
	}
}

func TestJWTManager_RefreshAccessToken(t *testing.T) {
	manager, err := NewJWTManager(nil, nil, 15*time.Minute, 7*24*time.Hour)
	if err != nil {
		t.Fatalf("failed to create JWT manager: %v", err)
	}

	// Generate initial tokens
	initialPair, err := manager.GenerateTokenPair("user", "tenant", "email", "session", []string{"user"}, []string{"read"})
	if err != nil {
		t.Fatalf("failed to generate initial tokens: %v", err)
	}

	// Refresh using refresh token
	newPair, err := manager.RefreshAccessToken(initialPair.RefreshToken, []string{"admin"}, []string{"read", "write"}, "email")
	if err != nil {
		t.Fatalf("failed to refresh token: %v", err)
	}

	// Validate new access token
	claims, err := manager.ValidateToken(newPair.AccessToken)
	if err != nil {
		t.Fatalf("failed to validate new access token: %v", err)
	}

	// Check updated roles and permissions
	if len(claims.Roles) != 1 || claims.Roles[0] != "admin" {
		t.Errorf("expected roles [admin], got %v", claims.Roles)
	}
	if len(claims.Permissions) != 2 {
		t.Errorf("expected 2 permissions, got %d", len(claims.Permissions))
	}
}

func TestJWTManager_RefreshWithAccessToken(t *testing.T) {
	manager, err := NewJWTManager(nil, nil, 15*time.Minute, 7*24*time.Hour)
	if err != nil {
		t.Fatalf("failed to create JWT manager: %v", err)
	}

	// Generate tokens
	tokenPair, err := manager.GenerateTokenPair("user", "tenant", "email", "session", nil, nil)
	if err != nil {
		t.Fatalf("failed to generate tokens: %v", err)
	}

	// Try to refresh using access token (should fail)
	_, err = manager.RefreshAccessToken(tokenPair.AccessToken, nil, nil, "email")
	if err == nil {
		t.Error("expected error when refreshing with access token")
	}
}

func BenchmarkJWTManager_GenerateTokenPair(b *testing.B) {
	manager, _ := NewJWTManager(nil, nil, 15*time.Minute, 7*24*time.Hour)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = manager.GenerateTokenPair("user", "tenant", "email", "session", []string{"admin"}, []string{"read"})
	}
}

func BenchmarkJWTManager_ValidateToken(b *testing.B) {
	manager, _ := NewJWTManager(nil, nil, 15*time.Minute, 7*24*time.Hour)
	tokenPair, _ := manager.GenerateTokenPair("user", "tenant", "email", "session", []string{"admin"}, []string{"read"})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = manager.ValidateToken(tokenPair.AccessToken)
	}
}
