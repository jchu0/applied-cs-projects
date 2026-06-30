package auth

import (
	"context"
	"errors"
	"time"

	"github.com/google/uuid"
	"github.com/mlai/microservice-platform/internal/common"
	"golang.org/x/crypto/bcrypt"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

var (
	ErrInvalidCredentials = errors.New("invalid credentials")
	ErrUserNotFound       = errors.New("user not found")
	ErrSessionNotFound    = errors.New("session not found")
)

// Service implements the auth gRPC service
type Service struct {
	sessionRepo *SessionRepository
	jwtManager  *JWTManager
	userClient  UserServiceClient
	logger      *common.Logger
}

// UserServiceClient defines the interface for user service calls
type UserServiceClient interface {
	GetUserByEmail(ctx context.Context, email, tenantID string) (*UserInfo, error)
	GetUserByID(ctx context.Context, id, tenantID string) (*UserInfo, error)
	CreateUser(ctx context.Context, req *CreateUserReq) (*UserInfo, error)
}

// UserInfo contains user information from user service
type UserInfo struct {
	ID           string
	TenantID     string
	Email        string
	PasswordHash string
	FirstName    string
	LastName     string
	Roles        []string
	Permissions  []string
	Status       string
}

// CreateUserReq contains user creation parameters
type CreateUserReq struct {
	TenantID  string
	Email     string
	Password  string
	FirstName string
	LastName  string
}

// NewService creates a new auth service
func NewService(sessionRepo *SessionRepository, jwtManager *JWTManager, userClient UserServiceClient, logger *common.Logger) *Service {
	return &Service{
		sessionRepo: sessionRepo,
		jwtManager:  jwtManager,
		userClient:  userClient,
		logger:      logger,
	}
}

// Login authenticates a user and returns tokens
func (s *Service) Login(ctx context.Context, req *LoginRequest) (*LoginResponse, error) {
	// Validate request
	if req.Email == "" {
		return nil, status.Error(codes.InvalidArgument, "email is required")
	}
	if req.Password == "" {
		return nil, status.Error(codes.InvalidArgument, "password is required")
	}
	if req.TenantID == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}

	// Get user from user service
	user, err := s.userClient.GetUserByEmail(ctx, req.Email, req.TenantID)
	if err != nil {
		if errors.Is(err, ErrUserNotFound) {
			return nil, status.Error(codes.Unauthenticated, "invalid credentials")
		}
		s.logger.Error("failed to get user", "error", err)
		return nil, status.Error(codes.Internal, "authentication failed")
	}

	// Check if user is active
	if user.Status != "active" {
		return nil, status.Error(codes.PermissionDenied, "user account is not active")
	}

	// Verify password
	err = bcrypt.CompareHashAndPassword([]byte(user.PasswordHash), []byte(req.Password))
	if err != nil {
		s.logger.WithTenant(req.TenantID).Warn("failed login attempt", "email", req.Email)
		return nil, status.Error(codes.Unauthenticated, "invalid credentials")
	}

	// Create session
	session := &Session{
		ID:           uuid.New().String(),
		UserID:       user.ID,
		TenantID:     user.TenantID,
		IPAddress:    req.IPAddress,
		UserAgent:    req.UserAgent,
		CreatedAt:    time.Now(),
		ExpiresAt:    time.Now().Add(7 * 24 * time.Hour),
		LastActivity: time.Now(),
	}

	if err := s.sessionRepo.Create(ctx, session); err != nil {
		s.logger.Error("failed to create session", "error", err)
		return nil, status.Error(codes.Internal, "failed to create session")
	}

	// Generate tokens
	tokenPair, err := s.jwtManager.GenerateTokenPair(
		user.ID, user.TenantID, user.Email, session.ID,
		user.Roles, user.Permissions,
	)
	if err != nil {
		s.logger.Error("failed to generate tokens", "error", err)
		return nil, status.Error(codes.Internal, "failed to generate tokens")
	}

	s.logger.WithTenant(req.TenantID).Info("user logged in", "user_id", user.ID)

	return &LoginResponse{
		AccessToken:  tokenPair.AccessToken,
		RefreshToken: tokenPair.RefreshToken,
		ExpiresIn:    tokenPair.ExpiresIn,
		TokenType:    "Bearer",
		User: &AuthUserInfo{
			ID:          user.ID,
			Email:       user.Email,
			FirstName:   user.FirstName,
			LastName:    user.LastName,
			Roles:       user.Roles,
			Permissions: user.Permissions,
		},
	}, nil
}

// Logout invalidates a session
func (s *Service) Logout(ctx context.Context, req *LogoutRequest) (*LogoutResponse, error) {
	// Validate access token to get session ID
	claims, err := s.jwtManager.ValidateToken(req.AccessToken)
	if err != nil {
		// Token might be expired, but we should still try to logout
		s.logger.Warn("logout with invalid access token", "error", err)
	}

	if claims != nil && claims.SessionID != "" {
		if err := s.sessionRepo.Delete(ctx, claims.SessionID); err != nil {
			s.logger.Error("failed to delete session", "error", err)
		}
	}

	s.logger.Info("user logged out")

	return &LogoutResponse{Success: true}, nil
}

// RefreshToken generates new tokens from a refresh token
func (s *Service) RefreshToken(ctx context.Context, req *RefreshTokenRequest) (*RefreshTokenResponse, error) {
	if req.RefreshToken == "" {
		return nil, status.Error(codes.InvalidArgument, "refresh_token is required")
	}

	// Validate refresh token
	claims, err := s.jwtManager.ValidateToken(req.RefreshToken)
	if err != nil {
		if errors.Is(err, ErrExpiredToken) {
			return nil, status.Error(codes.Unauthenticated, "refresh token expired")
		}
		return nil, status.Error(codes.Unauthenticated, "invalid refresh token")
	}

	// Check session still exists
	session, err := s.sessionRepo.GetByID(ctx, claims.SessionID)
	if err != nil {
		return nil, status.Error(codes.Unauthenticated, "session not found")
	}

	// Get user info for new tokens
	user, err := s.userClient.GetUserByID(ctx, claims.UserID, claims.TenantID)
	if err != nil {
		return nil, status.Error(codes.Internal, "failed to get user info")
	}

	// Update session activity
	session.LastActivity = time.Now()
	if err := s.sessionRepo.Update(ctx, session); err != nil {
		s.logger.Error("failed to update session", "error", err)
	}

	// Generate new tokens
	tokenPair, err := s.jwtManager.GenerateTokenPair(
		user.ID, user.TenantID, user.Email, session.ID,
		user.Roles, user.Permissions,
	)
	if err != nil {
		return nil, status.Error(codes.Internal, "failed to generate tokens")
	}

	return &RefreshTokenResponse{
		AccessToken:  tokenPair.AccessToken,
		RefreshToken: tokenPair.RefreshToken,
		ExpiresIn:    tokenPair.ExpiresIn,
		TokenType:    "Bearer",
	}, nil
}

// ValidateToken validates an access token
func (s *Service) ValidateToken(ctx context.Context, req *ValidateTokenRequest) (*ValidateTokenResponse, error) {
	if req.Token == "" {
		return nil, status.Error(codes.InvalidArgument, "token is required")
	}

	claims, err := s.jwtManager.ValidateToken(req.Token)
	if err != nil {
		return &ValidateTokenResponse{
			Valid: false,
		}, nil
	}

	return &ValidateTokenResponse{
		Valid: true,
		Claims: &TokenClaimsResponse{
			UserID:      claims.UserID,
			TenantID:    claims.TenantID,
			Email:       claims.Email,
			Roles:       claims.Roles,
			Permissions: claims.Permissions,
			SessionID:   claims.SessionID,
			IssuedAt:    claims.IssuedAt.Unix(),
			ExpiresAt:   claims.ExpiresAt.Unix(),
		},
	}, nil
}

// Register creates a new user and returns tokens
func (s *Service) Register(ctx context.Context, req *RegisterRequest) (*RegisterResponse, error) {
	// Validate request
	if req.Email == "" {
		return nil, status.Error(codes.InvalidArgument, "email is required")
	}
	if req.Password == "" {
		return nil, status.Error(codes.InvalidArgument, "password is required")
	}
	if req.TenantID == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}

	// Create user via user service
	user, err := s.userClient.CreateUser(ctx, &CreateUserReq{
		TenantID:  req.TenantID,
		Email:     req.Email,
		Password:  req.Password,
		FirstName: req.FirstName,
		LastName:  req.LastName,
	})
	if err != nil {
		return nil, err
	}

	// Create session
	session := &Session{
		ID:           uuid.New().String(),
		UserID:       user.ID,
		TenantID:     user.TenantID,
		CreatedAt:    time.Now(),
		ExpiresAt:    time.Now().Add(7 * 24 * time.Hour),
		LastActivity: time.Now(),
	}

	if err := s.sessionRepo.Create(ctx, session); err != nil {
		s.logger.Error("failed to create session", "error", err)
		return nil, status.Error(codes.Internal, "failed to create session")
	}

	// Generate tokens
	tokenPair, err := s.jwtManager.GenerateTokenPair(
		user.ID, user.TenantID, user.Email, session.ID,
		user.Roles, user.Permissions,
	)
	if err != nil {
		return nil, status.Error(codes.Internal, "failed to generate tokens")
	}

	s.logger.WithTenant(req.TenantID).Info("user registered", "user_id", user.ID)

	return &RegisterResponse{
		AccessToken:  tokenPair.AccessToken,
		RefreshToken: tokenPair.RefreshToken,
		ExpiresIn:    tokenPair.ExpiresIn,
		TokenType:    "Bearer",
		User: &AuthUserInfo{
			ID:          user.ID,
			Email:       user.Email,
			FirstName:   user.FirstName,
			LastName:    user.LastName,
			Roles:       user.Roles,
			Permissions: user.Permissions,
		},
	}, nil
}

// Request/Response types
type LoginRequest struct {
	Email     string
	Password  string
	TenantID  string
	IPAddress string
	UserAgent string
}

type LoginResponse struct {
	AccessToken  string
	RefreshToken string
	ExpiresIn    int64
	TokenType    string
	User         *AuthUserInfo
}

type AuthUserInfo struct {
	ID          string
	Email       string
	FirstName   string
	LastName    string
	Roles       []string
	Permissions []string
}

type LogoutRequest struct {
	AccessToken  string
	RefreshToken string
}

type LogoutResponse struct {
	Success bool
}

type RefreshTokenRequest struct {
	RefreshToken string
}

type RefreshTokenResponse struct {
	AccessToken  string
	RefreshToken string
	ExpiresIn    int64
	TokenType    string
}

type ValidateTokenRequest struct {
	Token string
}

type ValidateTokenResponse struct {
	Valid  bool
	Claims *TokenClaimsResponse
}

type TokenClaimsResponse struct {
	UserID      string
	TenantID    string
	Email       string
	Roles       []string
	Permissions []string
	SessionID   string
	IssuedAt    int64
	ExpiresAt   int64
}

type RegisterRequest struct {
	Email     string
	Password  string
	TenantID  string
	FirstName string
	LastName  string
}

type RegisterResponse struct {
	AccessToken  string
	RefreshToken string
	ExpiresIn    int64
	TokenType    string
	User         *AuthUserInfo
}
