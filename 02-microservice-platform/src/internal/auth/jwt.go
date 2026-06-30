package auth

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/pem"
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"
)

// TokenClaims represents JWT claims
type TokenClaims struct {
	jwt.RegisteredClaims
	UserID      string   `json:"uid"`
	TenantID    string   `json:"tid"`
	Email       string   `json:"email"`
	Roles       []string `json:"roles"`
	Permissions []string `json:"perms"`
	SessionID   string   `json:"sid"`
	TokenType   string   `json:"type"` // "access" or "refresh"
}

// TokenPair represents an access/refresh token pair
type TokenPair struct {
	AccessToken  string
	RefreshToken string
	ExpiresIn    int64 // seconds until access token expires
}

// JWTManager handles JWT operations
type JWTManager struct {
	privateKey      *rsa.PrivateKey
	publicKey       *rsa.PublicKey
	issuer          string
	audience        string
	accessTokenTTL  time.Duration
	refreshTokenTTL time.Duration
}

// NewJWTManager creates a new JWT manager
func NewJWTManager(privateKeyPath, publicKeyPath, issuer, audience string, accessTTL, refreshTTL time.Duration) (*JWTManager, error) {
	manager := &JWTManager{
		issuer:          issuer,
		audience:        audience,
		accessTokenTTL:  accessTTL,
		refreshTokenTTL: refreshTTL,
	}

	// Load or generate keys
	if privateKeyPath != "" && publicKeyPath != "" {
		// Load from files
		privateKey, err := loadPrivateKey(privateKeyPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load private key: %w", err)
		}
		manager.privateKey = privateKey

		publicKey, err := loadPublicKey(publicKeyPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load public key: %w", err)
		}
		manager.publicKey = publicKey
	} else {
		// Generate keys (for development)
		privateKey, err := rsa.GenerateKey(rand.Reader, 2048)
		if err != nil {
			return nil, fmt.Errorf("failed to generate private key: %w", err)
		}
		manager.privateKey = privateKey
		manager.publicKey = &privateKey.PublicKey
	}

	return manager, nil
}

// GenerateTokenPair generates access and refresh tokens
func (m *JWTManager) GenerateTokenPair(userID, tenantID, email, sessionID string, roles, permissions []string) (*TokenPair, error) {
	now := time.Now()

	// Access token
	accessClaims := TokenClaims{
		RegisteredClaims: jwt.RegisteredClaims{
			Issuer:    m.issuer,
			Subject:   userID,
			Audience:  []string{m.audience},
			ExpiresAt: jwt.NewNumericDate(now.Add(m.accessTokenTTL)),
			IssuedAt:  jwt.NewNumericDate(now),
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

	accessToken, err := jwt.NewWithClaims(jwt.SigningMethodRS256, accessClaims).SignedString(m.privateKey)
	if err != nil {
		return nil, fmt.Errorf("failed to sign access token: %w", err)
	}

	// Refresh token
	refreshClaims := TokenClaims{
		RegisteredClaims: jwt.RegisteredClaims{
			Issuer:    m.issuer,
			Subject:   userID,
			Audience:  []string{m.audience},
			ExpiresAt: jwt.NewNumericDate(now.Add(m.refreshTokenTTL)),
			IssuedAt:  jwt.NewNumericDate(now),
			ID:        uuid.New().String(),
		},
		UserID:    userID,
		TenantID:  tenantID,
		SessionID: sessionID,
		TokenType: "refresh",
	}

	refreshToken, err := jwt.NewWithClaims(jwt.SigningMethodRS256, refreshClaims).SignedString(m.privateKey)
	if err != nil {
		return nil, fmt.Errorf("failed to sign refresh token: %w", err)
	}

	return &TokenPair{
		AccessToken:  accessToken,
		RefreshToken: refreshToken,
		ExpiresIn:    int64(m.accessTokenTTL.Seconds()),
	}, nil
}

// ValidateToken validates a JWT token and returns claims
func (m *JWTManager) ValidateToken(tokenString string) (*TokenClaims, error) {
	token, err := jwt.ParseWithClaims(tokenString, &TokenClaims{}, func(token *jwt.Token) (interface{}, error) {
		if _, ok := token.Method.(*jwt.SigningMethodRSA); !ok {
			return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
		}
		return m.publicKey, nil
	})

	if err != nil {
		return nil, fmt.Errorf("failed to parse token: %w", err)
	}

	claims, ok := token.Claims.(*TokenClaims)
	if !ok || !token.Valid {
		return nil, errors.New("invalid token claims")
	}

	return claims, nil
}

// ValidateAccessToken validates an access token
func (m *JWTManager) ValidateAccessToken(tokenString string) (*TokenClaims, error) {
	claims, err := m.ValidateToken(tokenString)
	if err != nil {
		return nil, err
	}

	if claims.TokenType != "access" {
		return nil, errors.New("not an access token")
	}

	return claims, nil
}

// ValidateRefreshToken validates a refresh token
func (m *JWTManager) ValidateRefreshToken(tokenString string) (*TokenClaims, error) {
	claims, err := m.ValidateToken(tokenString)
	if err != nil {
		return nil, err
	}

	if claims.TokenType != "refresh" {
		return nil, errors.New("not a refresh token")
	}

	return claims, nil
}

// GetPublicKey returns the public key in PEM format
func (m *JWTManager) GetPublicKey() ([]byte, error) {
	pubKeyBytes, err := x509.MarshalPKIXPublicKey(m.publicKey)
	if err != nil {
		return nil, err
	}

	pubKeyPEM := pem.EncodeToMemory(&pem.Block{
		Type:  "PUBLIC KEY",
		Bytes: pubKeyBytes,
	})

	return pubKeyPEM, nil
}

// loadPrivateKey loads a private key from file
func loadPrivateKey(path string) (*rsa.PrivateKey, error) {
	keyData, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	block, _ := pem.Decode(keyData)
	if block == nil {
		return nil, errors.New("failed to decode PEM block")
	}

	return x509.ParsePKCS1PrivateKey(block.Bytes)
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
