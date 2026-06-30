package auth

import (
	"crypto/rand"
	"crypto/rsa"
	"errors"
	"fmt"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"
)

var (
	ErrInvalidToken = errors.New("invalid token")
	ErrExpiredToken = errors.New("token has expired")
)

// TokenClaims represents the JWT claims
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

// TokenPair contains access and refresh tokens
type TokenPair struct {
	AccessToken  string
	RefreshToken string
	ExpiresIn    int64
}

// JWTManager handles JWT token operations
type JWTManager struct {
	privateKey         *rsa.PrivateKey
	publicKey          *rsa.PublicKey
	accessTokenExpiry  time.Duration
	refreshTokenExpiry time.Duration
	issuer             string
	audience           []string
}

// NewJWTManager creates a new JWT manager
func NewJWTManager(privateKeyPEM, publicKeyPEM []byte, accessExpiry, refreshExpiry time.Duration) (*JWTManager, error) {
	var privateKey *rsa.PrivateKey
	var publicKey *rsa.PublicKey
	var err error

	if len(privateKeyPEM) > 0 && len(publicKeyPEM) > 0 {
		// Parse provided keys
		privateKey, err = jwt.ParseRSAPrivateKeyFromPEM(privateKeyPEM)
		if err != nil {
			return nil, fmt.Errorf("failed to parse private key: %w", err)
		}

		publicKey, err = jwt.ParseRSAPublicKeyFromPEM(publicKeyPEM)
		if err != nil {
			return nil, fmt.Errorf("failed to parse public key: %w", err)
		}
	} else {
		// Generate new key pair for development
		privateKey, err = rsa.GenerateKey(rand.Reader, 2048)
		if err != nil {
			return nil, fmt.Errorf("failed to generate RSA key: %w", err)
		}
		publicKey = &privateKey.PublicKey
	}

	return &JWTManager{
		privateKey:         privateKey,
		publicKey:          publicKey,
		accessTokenExpiry:  accessExpiry,
		refreshTokenExpiry: refreshExpiry,
		issuer:             "microservice-platform",
		audience:           []string{"microservice-platform-api"},
	}, nil
}

// GenerateTokenPair creates a new access/refresh token pair
func (m *JWTManager) GenerateTokenPair(userID, tenantID, email, sessionID string, roles, permissions []string) (*TokenPair, error) {
	now := time.Now()

	// Create access token
	accessClaims := TokenClaims{
		RegisteredClaims: jwt.RegisteredClaims{
			Issuer:    m.issuer,
			Subject:   userID,
			Audience:  m.audience,
			ExpiresAt: jwt.NewNumericDate(now.Add(m.accessTokenExpiry)),
			IssuedAt:  jwt.NewNumericDate(now),
			NotBefore: jwt.NewNumericDate(now),
			ID:        uuid.New().String(),
		},
		UserID:      userID,
		TenantID:    tenantID,
		Email:       email,
		Roles:       roles,
		Permissions: permissions,
		SessionID:   sessionID,
		TokenType:   "access",
	}

	accessToken := jwt.NewWithClaims(jwt.SigningMethodRS256, accessClaims)
	accessTokenString, err := accessToken.SignedString(m.privateKey)
	if err != nil {
		return nil, fmt.Errorf("failed to sign access token: %w", err)
	}

	// Create refresh token
	refreshClaims := TokenClaims{
		RegisteredClaims: jwt.RegisteredClaims{
			Issuer:    m.issuer,
			Subject:   userID,
			ExpiresAt: jwt.NewNumericDate(now.Add(m.refreshTokenExpiry)),
			IssuedAt:  jwt.NewNumericDate(now),
			ID:        uuid.New().String(),
		},
		UserID:    userID,
		TenantID:  tenantID,
		SessionID: sessionID,
		TokenType: "refresh",
	}

	refreshToken := jwt.NewWithClaims(jwt.SigningMethodRS256, refreshClaims)
	refreshTokenString, err := refreshToken.SignedString(m.privateKey)
	if err != nil {
		return nil, fmt.Errorf("failed to sign refresh token: %w", err)
	}

	return &TokenPair{
		AccessToken:  accessTokenString,
		RefreshToken: refreshTokenString,
		ExpiresIn:    int64(m.accessTokenExpiry.Seconds()),
	}, nil
}

// ValidateToken validates a token and returns the claims
func (m *JWTManager) ValidateToken(tokenString string) (*TokenClaims, error) {
	token, err := jwt.ParseWithClaims(tokenString, &TokenClaims{}, func(token *jwt.Token) (interface{}, error) {
		if _, ok := token.Method.(*jwt.SigningMethodRSA); !ok {
			return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
		}
		return m.publicKey, nil
	})

	if err != nil {
		if errors.Is(err, jwt.ErrTokenExpired) {
			return nil, ErrExpiredToken
		}
		return nil, ErrInvalidToken
	}

	claims, ok := token.Claims.(*TokenClaims)
	if !ok || !token.Valid {
		return nil, ErrInvalidToken
	}

	return claims, nil
}

// RefreshAccessToken creates a new access token from a valid refresh token
func (m *JWTManager) RefreshAccessToken(refreshTokenString string, roles, permissions []string, email string) (*TokenPair, error) {
	claims, err := m.ValidateToken(refreshTokenString)
	if err != nil {
		return nil, err
	}

	if claims.TokenType != "refresh" {
		return nil, ErrInvalidToken
	}

	return m.GenerateTokenPair(claims.UserID, claims.TenantID, email, claims.SessionID, roles, permissions)
}

// GetPublicKey returns the public key for token verification
func (m *JWTManager) GetPublicKey() *rsa.PublicKey {
	return m.publicKey
}
