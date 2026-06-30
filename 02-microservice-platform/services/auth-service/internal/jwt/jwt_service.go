package jwt

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
	"github.com/project/microservices/auth-service/internal/config"
	"github.com/project/microservices/auth-service/internal/models"
)

var (
	ErrInvalidToken     = errors.New("invalid token")
	ErrExpiredToken     = errors.New("token has expired")
	ErrInvalidTokenType = errors.New("invalid token type")
)

// Service handles JWT token operations
type Service struct {
	privateKey      *rsa.PrivateKey
	publicKey       *rsa.PublicKey
	issuer          string
	audience        string
	accessTokenTTL  time.Duration
	refreshTokenTTL time.Duration
}

// Claims represents the JWT claims structure
type Claims struct {
	jwt.RegisteredClaims
	UserID      string   `json:"uid"`
	TenantID    string   `json:"tid"`
	Email       string   `json:"email"`
	Roles       []string `json:"roles"`
	Permissions []string `json:"perms"`
	SessionID   string   `json:"sid"`
	TokenType   string   `json:"type"`
}

// NewService creates a new JWT service
func NewService(cfg *config.Config) (*Service, error) {
	var privateKey *rsa.PrivateKey
	var publicKey *rsa.PublicKey
	var err error

	// Try to load keys from files
	if _, err := os.Stat(cfg.JWTPrivateKeyPath); err == nil {
		privateKey, err = loadPrivateKey(cfg.JWTPrivateKeyPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load private key: %w", err)
		}

		publicKey, err = loadPublicKey(cfg.JWTPublicKeyPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load public key: %w", err)
		}
	} else {
		// Generate keys for development
		privateKey, err = rsa.GenerateKey(rand.Reader, 2048)
		if err != nil {
			return nil, fmt.Errorf("failed to generate RSA key: %w", err)
		}
		publicKey = &privateKey.PublicKey
	}

	return &Service{
		privateKey:      privateKey,
		publicKey:       publicKey,
		issuer:          cfg.JWTIssuer,
		audience:        cfg.JWTAudience,
		accessTokenTTL:  cfg.AccessTokenTTL,
		refreshTokenTTL: cfg.RefreshTokenTTL,
	}, nil
}

// GenerateTokenPair generates access and refresh tokens
func (s *Service) GenerateTokenPair(user *models.UserInfo, tenantID, sessionID string, permissions []string) (*models.TokenPair, error) {
	now := time.Now()

	// Generate access token
	accessClaims := Claims{
		RegisteredClaims: jwt.RegisteredClaims{
			Issuer:    s.issuer,
			Subject:   user.ID,
			Audience:  []string{s.audience},
			ExpiresAt: jwt.NewNumericDate(now.Add(s.accessTokenTTL)),
			IssuedAt:  jwt.NewNumericDate(now),
			NotBefore: jwt.NewNumericDate(now),
			ID:        uuid.New().String(),
		},
		UserID:      user.ID,
		TenantID:    tenantID,
		Email:       user.Email,
		Roles:       user.Roles,
		Permissions: permissions,
		SessionID:   sessionID,
		TokenType:   "access",
	}

	accessToken := jwt.NewWithClaims(jwt.SigningMethodRS256, accessClaims)
	accessTokenString, err := accessToken.SignedString(s.privateKey)
	if err != nil {
		return nil, fmt.Errorf("failed to sign access token: %w", err)
	}

	// Generate refresh token
	refreshClaims := Claims{
		RegisteredClaims: jwt.RegisteredClaims{
			Issuer:    s.issuer,
			Subject:   user.ID,
			ExpiresAt: jwt.NewNumericDate(now.Add(s.refreshTokenTTL)),
			IssuedAt:  jwt.NewNumericDate(now),
			ID:        uuid.New().String(),
		},
		UserID:    user.ID,
		TenantID:  tenantID,
		SessionID: sessionID,
		TokenType: "refresh",
	}

	refreshToken := jwt.NewWithClaims(jwt.SigningMethodRS256, refreshClaims)
	refreshTokenString, err := refreshToken.SignedString(s.privateKey)
	if err != nil {
		return nil, fmt.Errorf("failed to sign refresh token: %w", err)
	}

	return &models.TokenPair{
		AccessToken:  accessTokenString,
		RefreshToken: refreshTokenString,
		ExpiresIn:    int64(s.accessTokenTTL.Seconds()),
		TokenType:    "Bearer",
	}, nil
}

// ValidateToken validates a token and returns claims
func (s *Service) ValidateToken(tokenString string) (*Claims, error) {
	token, err := jwt.ParseWithClaims(tokenString, &Claims{}, func(token *jwt.Token) (interface{}, error) {
		if _, ok := token.Method.(*jwt.SigningMethodRSA); !ok {
			return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
		}
		return s.publicKey, nil
	})

	if err != nil {
		if errors.Is(err, jwt.ErrTokenExpired) {
			return nil, ErrExpiredToken
		}
		return nil, ErrInvalidToken
	}

	claims, ok := token.Claims.(*Claims)
	if !ok || !token.Valid {
		return nil, ErrInvalidToken
	}

	return claims, nil
}

// ValidateAccessToken validates an access token
func (s *Service) ValidateAccessToken(tokenString string) (*Claims, error) {
	claims, err := s.ValidateToken(tokenString)
	if err != nil {
		return nil, err
	}

	if claims.TokenType != "access" {
		return nil, ErrInvalidTokenType
	}

	return claims, nil
}

// ValidateRefreshToken validates a refresh token
func (s *Service) ValidateRefreshToken(tokenString string) (*Claims, error) {
	claims, err := s.ValidateToken(tokenString)
	if err != nil {
		return nil, err
	}

	if claims.TokenType != "refresh" {
		return nil, ErrInvalidTokenType
	}

	return claims, nil
}

// GetPublicKey returns the public key for external validation
func (s *Service) GetPublicKey() *rsa.PublicKey {
	return s.publicKey
}

// loadPrivateKey loads an RSA private key from a PEM file
func loadPrivateKey(path string) (*rsa.PrivateKey, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	block, _ := pem.Decode(data)
	if block == nil {
		return nil, errors.New("failed to decode PEM block")
	}

	key, err := x509.ParsePKCS1PrivateKey(block.Bytes)
	if err != nil {
		// Try PKCS8 format
		keyInterface, err := x509.ParsePKCS8PrivateKey(block.Bytes)
		if err != nil {
			return nil, err
		}
		var ok bool
		key, ok = keyInterface.(*rsa.PrivateKey)
		if !ok {
			return nil, errors.New("not an RSA private key")
		}
	}

	return key, nil
}

// loadPublicKey loads an RSA public key from a PEM file
func loadPublicKey(path string) (*rsa.PublicKey, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	block, _ := pem.Decode(data)
	if block == nil {
		return nil, errors.New("failed to decode PEM block")
	}

	keyInterface, err := x509.ParsePKIXPublicKey(block.Bytes)
	if err != nil {
		return nil, err
	}

	key, ok := keyInterface.(*rsa.PublicKey)
	if !ok {
		return nil, errors.New("not an RSA public key")
	}

	return key, nil
}
