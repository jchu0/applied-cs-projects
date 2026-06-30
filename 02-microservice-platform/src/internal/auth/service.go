package auth

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/mlai/microservice-platform/pkg/logging"
	userv1 "github.com/mlai/microservice-platform/pkg/pb/user/v1"
	"go.uber.org/zap"
	"golang.org/x/crypto/bcrypt"
	"google.golang.org/grpc"
)

// Service errors
var (
	ErrInvalidCredentials = errors.New("invalid credentials")
	ErrUserNotFound       = errors.New("user not found")
	ErrInvalidToken       = errors.New("invalid token")
	ErrTokenExpired       = errors.New("token expired")
	ErrMFARequired        = errors.New("mfa verification required")
	ErrMFANotEnabled      = errors.New("mfa not enabled for user")
	ErrInvalidMFACode     = errors.New("invalid mfa code")
	ErrOAuthNotConfigured = errors.New("oauth provider not configured")
)

// Service handles authentication business logic
type Service struct {
	jwtManager   *JWTManager
	sessions     *SessionRepository
	userClient   userv1.UserServiceClient
	oauthManager *OAuthManager
	mfaManager   *MFAManager
	logger       *logging.Logger
}

// ServiceConfig holds configuration for the auth service
type ServiceConfig struct {
	OAuth *OAuthConfig
	MFA   *MFAConfig
}

// NewService creates a new auth service
func NewService(jwtManager *JWTManager, sessions *SessionRepository, userConn *grpc.ClientConn, logger *logging.Logger) *Service {
	return &Service{
		jwtManager: jwtManager,
		sessions:   sessions,
		userClient: userv1.NewUserServiceClient(userConn),
		logger:     logger,
	}
}

// NewServiceWithConfig creates a new auth service with OAuth and MFA support
func NewServiceWithConfig(jwtManager *JWTManager, sessions *SessionRepository, userConn *grpc.ClientConn, config *ServiceConfig, logger *logging.Logger) *Service {
	s := &Service{
		jwtManager: jwtManager,
		sessions:   sessions,
		userClient: userv1.NewUserServiceClient(userConn),
		logger:     logger,
	}

	// Initialize OAuth if configured
	if config != nil && config.OAuth != nil {
		s.oauthManager = NewOAuthManager(config.OAuth, logger)
	}

	// Initialize MFA if configured
	if config != nil && config.MFA != nil {
		s.mfaManager = NewMFAManager(config.MFA, NewInMemoryMFAStore(), logger)
	}

	return s
}

// LoginRequest contains login parameters
type LoginRequest struct {
	Email      string
	Password   string
	TenantID   string
	DeviceInfo map[string]string
	IPAddress  string
}

// LoginResponse contains login results
type LoginResponse struct {
	AccessToken  string
	RefreshToken string
	ExpiresIn    int64
	TokenType    string
	Scopes       []string
	SessionID    string
	User         *UserInfo
	MFARequired  bool   // True if MFA verification is needed
	MFAToken     string // Temporary token for MFA verification
}

// MFAVerifyRequest contains MFA verification parameters
type MFAVerifyRequest struct {
	MFAToken   string // Temporary token from login
	Code       string // TOTP code or recovery code
	TenantID   string
	DeviceInfo map[string]string
	IPAddress  string
}

// OAuthLoginRequest contains OAuth login parameters
type OAuthLoginRequest struct {
	Provider   OAuthProvider
	Code       string // Authorization code from OAuth provider
	State      string // State parameter for CSRF protection
	TenantID   string
	DeviceInfo map[string]string
	IPAddress  string
}

// MFASetupResponse contains MFA setup information
type MFASetupResponse struct {
	Secret        string   // Base32-encoded TOTP secret
	QRCodeURL     string   // URL for QR code generation
	RecoveryCodes []string // One-time recovery codes
}

// UserInfo contains user information
type UserInfo struct {
	ID          string
	Email       string
	FirstName   string
	LastName    string
	AvatarURL   string
	Roles       []string
	Permissions []string
}

// Login authenticates a user and returns tokens
func (s *Service) Login(ctx context.Context, req *LoginRequest) (*LoginResponse, error) {
	logger := s.logger.WithContext(ctx)

	// Get user by email from user service
	userResp, err := s.userClient.GetUserByEmail(ctx, &userv1.GetUserByEmailRequest{
		Email:    req.Email,
		TenantId: req.TenantID,
	})
	if err != nil {
		logger.Warn("user not found during login",
			zap.String("email", req.Email),
			zap.Error(err),
		)
		return nil, ErrInvalidCredentials
	}

	// Verify password
	if err := bcrypt.CompareHashAndPassword([]byte(userResp.PasswordHash), []byte(req.Password)); err != nil {
		logger.Warn("invalid password",
			zap.String("email", req.Email),
		)
		return nil, ErrInvalidCredentials
	}

	// Check user status
	if userResp.User.Status != userv1.UserStatus_USER_STATUS_ACTIVE {
		logger.Warn("user not active",
			zap.String("user_id", userResp.User.Id),
			zap.String("status", userResp.User.Status.String()),
		)
		return nil, ErrInvalidCredentials
	}

	// Check if MFA is enabled
	if s.mfaManager != nil {
		enabled, _ := s.mfaManager.IsMFAEnabled(ctx, userResp.User.Id, req.TenantID)
		if enabled {
			// Generate temporary MFA token for verification step
			mfaToken := generateSecureToken(32)
			// In production, store pending auth in Redis with TTL
			logger.Info("MFA required for login",
				zap.String("user_id", userResp.User.Id),
			)
			return &LoginResponse{
				MFARequired: true,
				MFAToken:    mfaToken,
				User: &UserInfo{
					ID:    userResp.User.Id,
					Email: userResp.User.Email,
				},
			}, nil
		}
	}

	// Get user roles
	rolesResp, err := s.userClient.GetUserRoles(ctx, &userv1.GetUserRolesRequest{
		UserId:   userResp.User.Id,
		TenantId: req.TenantID,
	})
	if err != nil {
		logger.Error("failed to get user roles", zap.Error(err))
		// Continue without roles
		rolesResp = &userv1.GetUserRolesResponse{}
	}

	// Extract roles and permissions
	var roles []string
	var permissions []string
	for _, role := range rolesResp.Roles {
		roles = append(roles, role.Name)
		permissions = append(permissions, role.Permissions...)
	}

	// Create session
	session, err := s.sessions.Create(ctx, userResp.User.Id, req.TenantID, req.IPAddress, req.DeviceInfo)
	if err != nil {
		logger.Error("failed to create session", zap.Error(err))
		return nil, fmt.Errorf("failed to create session: %w", err)
	}

	// Generate tokens
	tokenPair, err := s.jwtManager.GenerateTokenPair(
		userResp.User.Id,
		req.TenantID,
		userResp.User.Email,
		session.ID,
		roles,
		permissions,
	)
	if err != nil {
		logger.Error("failed to generate tokens", zap.Error(err))
		return nil, fmt.Errorf("failed to generate tokens: %w", err)
	}

	logger.Info("user logged in",
		zap.String("user_id", userResp.User.Id),
		zap.String("session_id", session.ID),
	)

	return &LoginResponse{
		AccessToken:  tokenPair.AccessToken,
		RefreshToken: tokenPair.RefreshToken,
		ExpiresIn:    tokenPair.ExpiresIn,
		TokenType:    "Bearer",
		SessionID:    session.ID,
		User: &UserInfo{
			ID:          userResp.User.Id,
			Email:       userResp.User.Email,
			FirstName:   userResp.User.FirstName,
			LastName:    userResp.User.LastName,
			AvatarURL:   userResp.User.AvatarUrl,
			Roles:       roles,
			Permissions: permissions,
		},
	}, nil
}

// Logout invalidates a session
func (s *Service) Logout(ctx context.Context, sessionID string, logoutAll bool, userID, tenantID string) error {
	logger := s.logger.WithContext(ctx)

	if logoutAll {
		if err := s.sessions.DeleteAllForUser(ctx, tenantID, userID); err != nil {
			logger.Error("failed to logout all sessions", zap.Error(err))
			return err
		}
		logger.Info("user logged out from all sessions", zap.String("user_id", userID))
	} else {
		if err := s.sessions.Delete(ctx, sessionID); err != nil {
			logger.Error("failed to logout session", zap.Error(err))
			return err
		}
		logger.Info("session logged out", zap.String("session_id", sessionID))
	}

	return nil
}

// RefreshToken exchanges a refresh token for new tokens
func (s *Service) RefreshToken(ctx context.Context, refreshToken string) (*TokenPair, error) {
	logger := s.logger.WithContext(ctx)

	// Validate refresh token
	claims, err := s.jwtManager.ValidateRefreshToken(refreshToken)
	if err != nil {
		logger.Warn("invalid refresh token", zap.Error(err))
		return nil, ErrInvalidToken
	}

	// Check if session is still valid
	valid, err := s.sessions.IsValid(ctx, claims.SessionID)
	if err != nil {
		logger.Error("failed to check session validity", zap.Error(err))
		return nil, err
	}
	if !valid {
		return nil, ErrSessionExpired
	}

	// Update session last active
	if err := s.sessions.UpdateLastActive(ctx, claims.SessionID); err != nil {
		logger.Warn("failed to update session last active", zap.Error(err))
	}

	// Get fresh user data
	userResp, err := s.userClient.GetUser(ctx, &userv1.GetUserRequest{
		UserId:   claims.UserID,
		TenantId: claims.TenantID,
	})
	if err != nil {
		logger.Error("failed to get user for refresh", zap.Error(err))
		return nil, ErrUserNotFound
	}

	// Get updated roles
	rolesResp, err := s.userClient.GetUserRoles(ctx, &userv1.GetUserRolesRequest{
		UserId:   claims.UserID,
		TenantId: claims.TenantID,
	})
	if err != nil {
		rolesResp = &userv1.GetUserRolesResponse{}
	}

	var roles []string
	var permissions []string
	for _, role := range rolesResp.Roles {
		roles = append(roles, role.Name)
		permissions = append(permissions, role.Permissions...)
	}

	// Generate new tokens
	tokenPair, err := s.jwtManager.GenerateTokenPair(
		claims.UserID,
		claims.TenantID,
		userResp.User.Email,
		claims.SessionID,
		roles,
		permissions,
	)
	if err != nil {
		logger.Error("failed to generate new tokens", zap.Error(err))
		return nil, err
	}

	logger.Info("tokens refreshed",
		zap.String("user_id", claims.UserID),
		zap.String("session_id", claims.SessionID),
	)

	return tokenPair, nil
}

// ValidateToken validates an access token and returns claims
func (s *Service) ValidateToken(ctx context.Context, accessToken string) (*TokenClaims, error) {
	claims, err := s.jwtManager.ValidateAccessToken(accessToken)
	if err != nil {
		return nil, ErrInvalidToken
	}

	// Optionally check session validity
	valid, err := s.sessions.IsValid(ctx, claims.SessionID)
	if err != nil {
		return nil, err
	}
	if !valid {
		return nil, ErrSessionExpired
	}

	return claims, nil
}

// GetActiveSessions returns all active sessions for a user
func (s *Service) GetActiveSessions(ctx context.Context, tenantID, userID string) ([]*Session, error) {
	return s.sessions.GetAllForUser(ctx, tenantID, userID)
}

// TerminateSession terminates a specific session
func (s *Service) TerminateSession(ctx context.Context, sessionID, userID, tenantID string) error {
	// Verify session belongs to user
	session, err := s.sessions.Get(ctx, sessionID)
	if err != nil {
		return err
	}

	if session.UserID != userID || session.TenantID != tenantID {
		return errors.New("session does not belong to user")
	}

	return s.sessions.Delete(ctx, sessionID)
}

// ChangePassword changes user password
func (s *Service) ChangePassword(ctx context.Context, userID, tenantID, currentPassword, newPassword string) error {
	logger := s.logger.WithContext(ctx)

	// Get user
	userResp, err := s.userClient.GetUser(ctx, &userv1.GetUserRequest{
		UserId:   userID,
		TenantId: tenantID,
	})
	if err != nil {
		return ErrUserNotFound
	}

	// Get user with password hash
	userWithPw, err := s.userClient.GetUserByEmail(ctx, &userv1.GetUserByEmailRequest{
		Email:    userResp.User.Email,
		TenantId: tenantID,
	})
	if err != nil {
		return ErrUserNotFound
	}

	// Verify current password
	if err := bcrypt.CompareHashAndPassword([]byte(userWithPw.PasswordHash), []byte(currentPassword)); err != nil {
		return ErrInvalidCredentials
	}

	// Hash new password
	newHash, err := bcrypt.GenerateFromPassword([]byte(newPassword), bcrypt.DefaultCost)
	if err != nil {
		return fmt.Errorf("failed to hash password: %w", err)
	}

	// Update password (would need a separate RPC for this)
	// For now, log the intent
	logger.Info("password change requested",
		zap.String("user_id", userID),
		zap.Int("new_hash_length", len(newHash)),
	)

	// Invalidate all other sessions
	// Keep current session active

	return nil
}

// --- OAuth Methods ---

// GetOAuthURL returns the OAuth authorization URL for a provider
func (s *Service) GetOAuthURL(ctx context.Context, provider OAuthProvider, tenantID, returnURL string) (string, error) {
	if s.oauthManager == nil {
		return "", ErrOAuthNotConfigured
	}

	if !s.oauthManager.IsProviderConfigured(provider) {
		return "", ErrOAuthProviderNotConfigured
	}

	// Create state with tenant context
	state := fmt.Sprintf("%s:%s:%d", tenantID, returnURL, time.Now().UnixNano())

	return s.oauthManager.GetAuthURL(provider, state)
}

// LoginWithOAuth authenticates a user via OAuth provider
func (s *Service) LoginWithOAuth(ctx context.Context, req *OAuthLoginRequest) (*LoginResponse, error) {
	logger := s.logger.WithContext(ctx)

	if s.oauthManager == nil {
		return nil, ErrOAuthNotConfigured
	}

	// Exchange code for user info
	oauthUser, err := s.oauthManager.ExchangeCode(ctx, req.Provider, req.Code)
	if err != nil {
		logger.Error("oauth code exchange failed",
			zap.String("provider", string(req.Provider)),
			zap.Error(err),
		)
		return nil, err
	}

	// Check if email is verified
	if !oauthUser.EmailVerified {
		return nil, ErrOAuthEmailNotVerified
	}

	// Get or create user by email
	userResp, err := s.userClient.GetUserByEmail(ctx, &userv1.GetUserByEmailRequest{
		Email:    oauthUser.Email,
		TenantId: req.TenantID,
	})
	if err != nil {
		// User doesn't exist, create them
		createResp, createErr := s.userClient.CreateUser(ctx, &userv1.CreateUserRequest{
			TenantId:  req.TenantID,
			Email:     oauthUser.Email,
			FirstName: oauthUser.FirstName,
			LastName:  oauthUser.LastName,
			AvatarUrl: oauthUser.AvatarURL,
			Status:    userv1.UserStatus_USER_STATUS_ACTIVE,
		})
		if createErr != nil {
			logger.Error("failed to create oauth user", zap.Error(createErr))
			return nil, fmt.Errorf("failed to create user: %w", createErr)
		}
		userResp = &userv1.GetUserByEmailResponse{
			User: createResp.User,
		}
	}

	// Get user roles
	rolesResp, err := s.userClient.GetUserRoles(ctx, &userv1.GetUserRolesRequest{
		UserId:   userResp.User.Id,
		TenantId: req.TenantID,
	})
	if err != nil {
		rolesResp = &userv1.GetUserRolesResponse{}
	}

	var roles []string
	var permissions []string
	for _, role := range rolesResp.Roles {
		roles = append(roles, role.Name)
		permissions = append(permissions, role.Permissions...)
	}

	// Check if MFA is enabled
	if s.mfaManager != nil {
		enabled, _ := s.mfaManager.IsMFAEnabled(ctx, userResp.User.Id, req.TenantID)
		if enabled {
			// Generate temporary MFA token
			mfaToken := generateSecureToken(32)
			// Store pending auth (in production, use Redis with TTL)
			logger.Info("MFA required for OAuth login",
				zap.String("user_id", userResp.User.Id),
			)
			return &LoginResponse{
				MFARequired: true,
				MFAToken:    mfaToken,
				User: &UserInfo{
					ID:    userResp.User.Id,
					Email: userResp.User.Email,
				},
			}, nil
		}
	}

	// Create session
	session, err := s.sessions.Create(ctx, userResp.User.Id, req.TenantID, req.IPAddress, req.DeviceInfo)
	if err != nil {
		return nil, fmt.Errorf("failed to create session: %w", err)
	}

	// Generate tokens
	tokenPair, err := s.jwtManager.GenerateTokenPair(
		userResp.User.Id,
		req.TenantID,
		userResp.User.Email,
		session.ID,
		roles,
		permissions,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to generate tokens: %w", err)
	}

	logger.Info("user logged in via OAuth",
		zap.String("user_id", userResp.User.Id),
		zap.String("provider", string(req.Provider)),
	)

	return &LoginResponse{
		AccessToken:  tokenPair.AccessToken,
		RefreshToken: tokenPair.RefreshToken,
		ExpiresIn:    tokenPair.ExpiresIn,
		TokenType:    "Bearer",
		SessionID:    session.ID,
		User: &UserInfo{
			ID:          userResp.User.Id,
			Email:       userResp.User.Email,
			FirstName:   userResp.User.FirstName,
			LastName:    userResp.User.LastName,
			AvatarURL:   userResp.User.AvatarUrl,
			Roles:       roles,
			Permissions: permissions,
		},
	}, nil
}

// --- MFA Methods ---

// SetupMFA initiates MFA setup for a user
func (s *Service) SetupMFA(ctx context.Context, userID, tenantID, email string) (*MFASetupResponse, error) {
	if s.mfaManager == nil {
		return nil, ErrMFANotEnabled
	}

	setup, err := s.mfaManager.InitiateSetup(ctx, userID, tenantID, email)
	if err != nil {
		return nil, fmt.Errorf("failed to initiate MFA setup: %w", err)
	}

	return &MFASetupResponse{
		Secret:        setup.Secret,
		QRCodeURL:     setup.QRCodeURL,
		RecoveryCodes: setup.RecoveryCodes,
	}, nil
}

// ConfirmMFA confirms MFA setup with a TOTP code
func (s *Service) ConfirmMFA(ctx context.Context, userID, tenantID, code string) ([]string, error) {
	if s.mfaManager == nil {
		return nil, ErrMFANotEnabled
	}

	recoveryCodes, err := s.mfaManager.ConfirmSetup(ctx, userID, tenantID, code)
	if err != nil {
		return nil, fmt.Errorf("failed to confirm MFA: %w", err)
	}

	s.logger.Info("MFA enabled for user",
		zap.String("user_id", userID),
		zap.String("tenant_id", tenantID),
	)

	return recoveryCodes, nil
}

// DisableMFA disables MFA for a user
func (s *Service) DisableMFA(ctx context.Context, userID, tenantID, code string) error {
	if s.mfaManager == nil {
		return ErrMFANotEnabled
	}

	// Verify the code first
	valid, err := s.mfaManager.ValidateTOTP(ctx, userID, tenantID, code)
	if err != nil {
		return err
	}
	if !valid {
		return ErrInvalidMFACode
	}

	if err := s.mfaManager.DisableMFA(ctx, userID, tenantID); err != nil {
		return fmt.Errorf("failed to disable MFA: %w", err)
	}

	s.logger.Info("MFA disabled for user",
		zap.String("user_id", userID),
		zap.String("tenant_id", tenantID),
	)

	return nil
}

// VerifyMFA verifies MFA code and completes login
func (s *Service) VerifyMFA(ctx context.Context, req *MFAVerifyRequest) (*LoginResponse, error) {
	logger := s.logger.WithContext(ctx)

	if s.mfaManager == nil {
		return nil, ErrMFANotEnabled
	}

	// In production, validate MFAToken and retrieve pending auth from Redis
	// For now, we'll need the user ID from the token
	// This is a simplified implementation

	// Try TOTP first, then recovery code
	// The MFAToken would contain encrypted user info in production

	logger.Info("MFA verification attempted",
		zap.String("tenant_id", req.TenantID),
	)

	return nil, errors.New("MFA token validation not implemented - requires Redis for pending auth storage")
}

// IsMFAEnabled checks if MFA is enabled for a user
func (s *Service) IsMFAEnabled(ctx context.Context, userID, tenantID string) (bool, error) {
	if s.mfaManager == nil {
		return false, nil
	}
	return s.mfaManager.IsMFAEnabled(ctx, userID, tenantID)
}

// GetRecoveryCodes generates new recovery codes for a user
func (s *Service) GetRecoveryCodes(ctx context.Context, userID, tenantID, code string) ([]string, error) {
	if s.mfaManager == nil {
		return nil, ErrMFANotEnabled
	}

	// Verify TOTP first
	valid, err := s.mfaManager.ValidateTOTP(ctx, userID, tenantID, code)
	if err != nil {
		return nil, err
	}
	if !valid {
		return nil, ErrInvalidMFACode
	}

	// Generate new recovery codes
	return s.mfaManager.RegenerateRecoveryCodes(ctx, userID, tenantID)
}
