package handlers

import (
	"context"
	"errors"

	"github.com/project/microservices/auth-service/internal/config"
	"github.com/project/microservices/auth-service/internal/jwt"
	"github.com/project/microservices/auth-service/internal/models"
	"github.com/project/microservices/auth-service/internal/repository"
	"golang.org/x/crypto/bcrypt"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

var (
	ErrInvalidCredentials = errors.New("invalid credentials")
	ErrUserNotFound       = errors.New("user not found")
)

// AuthServiceServer implements the gRPC AuthService
type AuthServiceServer struct {
	cfg         *config.Config
	jwtService  *jwt.Service
	sessionRepo *repository.SessionRepository
	userClient  UserServiceClient
	UnimplementedAuthServiceServer
}

// UnimplementedAuthServiceServer is embedded for forward compatibility
type UnimplementedAuthServiceServer struct{}

// UserServiceClient is an interface for the user service client
type UserServiceClient interface {
	GetUserByEmail(ctx context.Context, email, tenantID string) (*UserData, error)
	CreateUser(ctx context.Context, input *CreateUserInput) (*UserData, error)
}

// UserData holds user data from user service
type UserData struct {
	ID           string
	TenantID     string
	Email        string
	PasswordHash string
	FirstName    string
	LastName     string
	AvatarURL    string
	Roles        []string
	Permissions  []string
}

// CreateUserInput holds input for creating a user
type CreateUserInput struct {
	TenantID  string
	Email     string
	Password  string
	FirstName string
	LastName  string
	Metadata  map[string]string
}

// NewAuthServiceServer creates a new auth service server
func NewAuthServiceServer(cfg *config.Config, jwtService *jwt.Service, sessionRepo *repository.SessionRepository, userClient UserServiceClient) *AuthServiceServer {
	return &AuthServiceServer{
		cfg:         cfg,
		jwtService:  jwtService,
		sessionRepo: sessionRepo,
		userClient:  userClient,
	}
}

// Login authenticates a user and returns tokens
func (s *AuthServiceServer) Login(ctx context.Context, req *LoginRequest) (*LoginResponse, error) {
	if req.Email == "" {
		return nil, status.Error(codes.InvalidArgument, "email is required")
	}
	if req.Password == "" {
		return nil, status.Error(codes.InvalidArgument, "password is required")
	}
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}

	// Get user from user service
	user, err := s.userClient.GetUserByEmail(ctx, req.Email, req.TenantId)
	if err != nil {
		if errors.Is(err, ErrUserNotFound) {
			return nil, status.Error(codes.Unauthenticated, "invalid credentials")
		}
		return nil, status.Errorf(codes.Internal, "failed to get user: %v", err)
	}

	// Verify password
	if err := bcrypt.CompareHashAndPassword([]byte(user.PasswordHash), []byte(req.Password)); err != nil {
		return nil, status.Error(codes.Unauthenticated, "invalid credentials")
	}

	// Create session
	session, err := s.sessionRepo.Create(ctx, user.ID, user.TenantID, "", "", s.cfg.RefreshTokenTTL)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to create session: %v", err)
	}

	// Generate tokens
	userInfo := &models.UserInfo{
		ID:        user.ID,
		Email:     user.Email,
		FirstName: user.FirstName,
		LastName:  user.LastName,
		AvatarURL: user.AvatarURL,
		Roles:     user.Roles,
	}

	tokens, err := s.jwtService.GenerateTokenPair(userInfo, user.TenantID, session.ID, user.Permissions)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to generate tokens: %v", err)
	}

	return &LoginResponse{
		Tokens: &TokenResponse{
			AccessToken:  tokens.AccessToken,
			RefreshToken: tokens.RefreshToken,
			ExpiresIn:    tokens.ExpiresIn,
			TokenType:    tokens.TokenType,
		},
		User: &UserInfo{
			Id:        user.ID,
			Email:     user.Email,
			FirstName: user.FirstName,
			LastName:  user.LastName,
			AvatarUrl: user.AvatarURL,
			Roles:     user.Roles,
		},
		MfaRequired: false,
	}, nil
}

// Logout invalidates the current session
func (s *AuthServiceServer) Logout(ctx context.Context, req *LogoutRequest) (*LogoutResponse, error) {
	if req.SessionId == "" && req.AccessToken == "" {
		return nil, status.Error(codes.InvalidArgument, "session_id or access_token is required")
	}

	var sessionID string
	if req.SessionId != "" {
		sessionID = req.SessionId
	} else {
		// Extract session ID from access token
		claims, err := s.jwtService.ValidateAccessToken(req.AccessToken)
		if err != nil {
			return nil, status.Error(codes.Unauthenticated, "invalid token")
		}
		sessionID = claims.SessionID
	}

	if err := s.sessionRepo.Revoke(ctx, sessionID); err != nil {
		return nil, status.Errorf(codes.Internal, "failed to revoke session: %v", err)
	}

	return &LogoutResponse{Success: true}, nil
}

// RefreshToken exchanges a refresh token for new tokens
func (s *AuthServiceServer) RefreshToken(ctx context.Context, req *RefreshTokenRequest) (*TokenResponse, error) {
	if req.RefreshToken == "" {
		return nil, status.Error(codes.InvalidArgument, "refresh_token is required")
	}

	// Validate refresh token
	claims, err := s.jwtService.ValidateRefreshToken(req.RefreshToken)
	if err != nil {
		if errors.Is(err, jwt.ErrExpiredToken) {
			return nil, status.Error(codes.Unauthenticated, "refresh token expired")
		}
		return nil, status.Error(codes.Unauthenticated, "invalid refresh token")
	}

	// Verify session is still valid
	session, err := s.sessionRepo.Get(ctx, claims.SessionID)
	if err != nil {
		if errors.Is(err, repository.ErrSessionRevoked) {
			return nil, status.Error(codes.Unauthenticated, "session has been revoked")
		}
		return nil, status.Error(codes.Unauthenticated, "session not found")
	}

	// Get user info
	user, err := s.userClient.GetUserByEmail(ctx, claims.Email, claims.TenantID)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get user: %v", err)
	}

	// Update session activity
	if err := s.sessionRepo.UpdateActivity(ctx, session.ID); err != nil {
		// Non-fatal, continue
	}

	// Generate new tokens
	userInfo := &models.UserInfo{
		ID:        user.ID,
		Email:     user.Email,
		FirstName: user.FirstName,
		LastName:  user.LastName,
		AvatarURL: user.AvatarURL,
		Roles:     user.Roles,
	}

	tokens, err := s.jwtService.GenerateTokenPair(userInfo, user.TenantID, session.ID, user.Permissions)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to generate tokens: %v", err)
	}

	return &TokenResponse{
		AccessToken:  tokens.AccessToken,
		RefreshToken: tokens.RefreshToken,
		ExpiresIn:    tokens.ExpiresIn,
		TokenType:    tokens.TokenType,
	}, nil
}

// ValidateToken validates an access token and returns claims
func (s *AuthServiceServer) ValidateToken(ctx context.Context, req *ValidateTokenRequest) (*ValidateTokenResponse, error) {
	if req.Token == "" {
		return nil, status.Error(codes.InvalidArgument, "token is required")
	}

	claims, err := s.jwtService.ValidateAccessToken(req.Token)
	if err != nil {
		return &ValidateTokenResponse{
			Valid:        false,
			ErrorMessage: err.Error(),
		}, nil
	}

	// Verify session is still valid
	_, err = s.sessionRepo.Get(ctx, claims.SessionID)
	if err != nil {
		return &ValidateTokenResponse{
			Valid:        false,
			ErrorMessage: "session expired or revoked",
		}, nil
	}

	return &ValidateTokenResponse{
		Valid: true,
		Claims: &TokenClaims{
			UserId:      claims.UserID,
			TenantId:    claims.TenantID,
			Email:       claims.Email,
			Roles:       claims.Roles,
			Permissions: claims.Permissions,
			SessionId:   claims.SessionID,
			TokenType:   claims.TokenType,
			IssuedAt:    claims.IssuedAt.Unix(),
			ExpiresAt:   claims.ExpiresAt.Unix(),
		},
	}, nil
}

// Register creates a new user account
func (s *AuthServiceServer) Register(ctx context.Context, req *RegisterRequest) (*RegisterResponse, error) {
	if req.Email == "" {
		return nil, status.Error(codes.InvalidArgument, "email is required")
	}
	if req.Password == "" {
		return nil, status.Error(codes.InvalidArgument, "password is required")
	}
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}

	// Create user via user service
	user, err := s.userClient.CreateUser(ctx, &CreateUserInput{
		TenantID:  req.TenantId,
		Email:     req.Email,
		Password:  req.Password,
		FirstName: req.FirstName,
		LastName:  req.LastName,
		Metadata:  req.Metadata,
	})
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to create user: %v", err)
	}

	// Create session
	session, err := s.sessionRepo.Create(ctx, user.ID, user.TenantID, "", "", s.cfg.RefreshTokenTTL)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to create session: %v", err)
	}

	// Generate tokens
	userInfo := &models.UserInfo{
		ID:        user.ID,
		Email:     user.Email,
		FirstName: user.FirstName,
		LastName:  user.LastName,
		AvatarURL: user.AvatarURL,
		Roles:     user.Roles,
	}

	tokens, err := s.jwtService.GenerateTokenPair(userInfo, user.TenantID, session.ID, user.Permissions)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to generate tokens: %v", err)
	}

	return &RegisterResponse{
		Tokens: &TokenResponse{
			AccessToken:  tokens.AccessToken,
			RefreshToken: tokens.RefreshToken,
			ExpiresIn:    tokens.ExpiresIn,
			TokenType:    tokens.TokenType,
		},
		User: &UserInfo{
			Id:        user.ID,
			Email:     user.Email,
			FirstName: user.FirstName,
			LastName:  user.LastName,
			AvatarUrl: user.AvatarURL,
			Roles:     user.Roles,
		},
	}, nil
}

// Proto message types (simplified - would normally be generated)
type LoginRequest struct {
	Email    string
	Password string
	TenantId string
	Metadata map[string]string
}

type LoginResponse struct {
	Tokens      *TokenResponse
	User        *UserInfo
	MfaRequired bool
	MfaToken    string
}

type LogoutRequest struct {
	AccessToken string
	SessionId   string
}

type LogoutResponse struct {
	Success bool
}

type RefreshTokenRequest struct {
	RefreshToken string
}

type TokenResponse struct {
	AccessToken  string
	RefreshToken string
	ExpiresIn    int64
	TokenType    string
	Scopes       []string
}

type ValidateTokenRequest struct {
	Token string
}

type ValidateTokenResponse struct {
	Valid        bool
	Claims       *TokenClaims
	ErrorMessage string
}

type TokenClaims struct {
	UserId      string
	TenantId    string
	Email       string
	Roles       []string
	Permissions []string
	SessionId   string
	TokenType   string
	IssuedAt    int64
	ExpiresAt   int64
}

type UserInfo struct {
	Id        string
	Email     string
	FirstName string
	LastName  string
	AvatarUrl string
	Roles     []string
}

type RegisterRequest struct {
	Email     string
	Password  string
	TenantId  string
	FirstName string
	LastName  string
	Metadata  map[string]string
}

type RegisterResponse struct {
	Tokens *TokenResponse
	User   *UserInfo
}
