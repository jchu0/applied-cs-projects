package auth

import (
	"context"
	"errors"
	"strings"

	authv1 "github.com/mlai/microservice-platform/pkg/pb/auth/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// Server implements the AuthService gRPC server
type Server struct {
	authv1.UnimplementedAuthServiceServer
	service *Service
}

// NewServer creates a new gRPC server
func NewServer(service *Service) *Server {
	return &Server{service: service}
}

// Login authenticates a user and returns tokens
func (s *Server) Login(ctx context.Context, req *authv1.LoginRequest) (*authv1.LoginResponse, error) {
	if req.Email == "" {
		return nil, status.Error(codes.InvalidArgument, "email is required")
	}
	if req.Password == "" {
		return nil, status.Error(codes.InvalidArgument, "password is required")
	}
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}

	resp, err := s.service.Login(ctx, &LoginRequest{
		Email:      req.Email,
		Password:   req.Password,
		TenantID:   req.TenantId,
		DeviceInfo: req.DeviceInfo,
	})
	if err != nil {
		if errors.Is(err, ErrInvalidCredentials) {
			return nil, status.Error(codes.Unauthenticated, "invalid credentials")
		}
		return nil, status.Error(codes.Internal, "login failed")
	}

	return &authv1.LoginResponse{
		AccessToken:  resp.AccessToken,
		RefreshToken: resp.RefreshToken,
		ExpiresIn:    resp.ExpiresIn,
		TokenType:    resp.TokenType,
		Scopes:       resp.Scopes,
		SessionId:    resp.SessionID,
		User: &authv1.User{
			Id:          resp.User.ID,
			Email:       resp.User.Email,
			FirstName:   resp.User.FirstName,
			LastName:    resp.User.LastName,
			AvatarUrl:   resp.User.AvatarURL,
			Roles:       resp.User.Roles,
			Permissions: resp.User.Permissions,
		},
	}, nil
}

// Logout invalidates the current session
func (s *Server) Logout(ctx context.Context, req *authv1.LogoutRequest) (*authv1.LogoutResponse, error) {
	// Extract user info from token
	claims, err := s.extractClaims(req.AccessToken)
	if err != nil {
		return nil, status.Error(codes.Unauthenticated, "invalid token")
	}

	sessionID := req.SessionId
	if sessionID == "" {
		sessionID = claims.SessionID
	}

	err = s.service.Logout(ctx, sessionID, req.LogoutAllSessions, claims.UserID, claims.TenantID)
	if err != nil {
		return nil, status.Error(codes.Internal, "logout failed")
	}

	return &authv1.LogoutResponse{
		Success: true,
	}, nil
}

// RefreshToken exchanges a refresh token for new tokens
func (s *Server) RefreshToken(ctx context.Context, req *authv1.RefreshTokenRequest) (*authv1.RefreshTokenResponse, error) {
	if req.RefreshToken == "" {
		return nil, status.Error(codes.InvalidArgument, "refresh_token is required")
	}

	tokenPair, err := s.service.RefreshToken(ctx, req.RefreshToken)
	if err != nil {
		if errors.Is(err, ErrInvalidToken) {
			return nil, status.Error(codes.Unauthenticated, "invalid refresh token")
		}
		if errors.Is(err, ErrSessionExpired) {
			return nil, status.Error(codes.Unauthenticated, "session expired")
		}
		return nil, status.Error(codes.Internal, "refresh failed")
	}

	return &authv1.RefreshTokenResponse{
		AccessToken:  tokenPair.AccessToken,
		RefreshToken: tokenPair.RefreshToken,
		ExpiresIn:    tokenPair.ExpiresIn,
		TokenType:    "Bearer",
	}, nil
}

// ValidateToken validates an access token and returns claims
func (s *Server) ValidateToken(ctx context.Context, req *authv1.ValidateTokenRequest) (*authv1.ValidateTokenResponse, error) {
	if req.AccessToken == "" {
		return nil, status.Error(codes.InvalidArgument, "access_token is required")
	}

	claims, err := s.service.ValidateToken(ctx, req.AccessToken)
	if err != nil {
		if errors.Is(err, ErrInvalidToken) || errors.Is(err, ErrSessionExpired) {
			return &authv1.ValidateTokenResponse{
				Valid: false,
			}, nil
		}
		return nil, status.Error(codes.Internal, "validation failed")
	}

	return &authv1.ValidateTokenResponse{
		Valid: true,
		Claims: &authv1.TokenClaims{
			UserId:      claims.UserID,
			TenantId:    claims.TenantID,
			Email:       claims.Email,
			Roles:       claims.Roles,
			Permissions: claims.Permissions,
			SessionId:   claims.SessionID,
			ExpiresAt:   timestamppb.New(claims.ExpiresAt.Time),
			IssuedAt:    timestamppb.New(claims.IssuedAt.Time),
		},
	}, nil
}

// RevokeToken revokes a specific token
func (s *Server) RevokeToken(ctx context.Context, req *authv1.RevokeTokenRequest) (*authv1.RevokeTokenResponse, error) {
	// For now, revoking a token invalidates the session
	claims, err := s.extractClaims(req.Token)
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, "invalid token")
	}

	err = s.service.Logout(ctx, claims.SessionID, false, claims.UserID, claims.TenantID)
	if err != nil {
		return nil, status.Error(codes.Internal, "revoke failed")
	}

	return &authv1.RevokeTokenResponse{
		Success: true,
	}, nil
}

// ChangePassword changes the user's password
func (s *Server) ChangePassword(ctx context.Context, req *authv1.ChangePasswordRequest) (*authv1.ChangePasswordResponse, error) {
	if req.UserId == "" {
		return nil, status.Error(codes.InvalidArgument, "user_id is required")
	}
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}
	if req.CurrentPassword == "" {
		return nil, status.Error(codes.InvalidArgument, "current_password is required")
	}
	if req.NewPassword == "" {
		return nil, status.Error(codes.InvalidArgument, "new_password is required")
	}

	err := s.service.ChangePassword(ctx, req.UserId, req.TenantId, req.CurrentPassword, req.NewPassword)
	if err != nil {
		if errors.Is(err, ErrInvalidCredentials) {
			return nil, status.Error(codes.Unauthenticated, "invalid current password")
		}
		if errors.Is(err, ErrUserNotFound) {
			return nil, status.Error(codes.NotFound, "user not found")
		}
		return nil, status.Error(codes.Internal, "password change failed")
	}

	return &authv1.ChangePasswordResponse{
		Success: true,
	}, nil
}

// ResetPassword initiates password reset
func (s *Server) ResetPassword(ctx context.Context, req *authv1.ResetPasswordRequest) (*authv1.ResetPasswordResponse, error) {
	// TODO: Implement password reset with email
	return &authv1.ResetPasswordResponse{
		Success: true,
		Message: "If the account exists, a reset email has been sent",
	}, nil
}

// ConfirmResetPassword completes password reset with token
func (s *Server) ConfirmResetPassword(ctx context.Context, req *authv1.ConfirmResetPasswordRequest) (*authv1.ConfirmResetPasswordResponse, error) {
	// TODO: Implement password reset confirmation
	return &authv1.ConfirmResetPasswordResponse{
		Success: true,
	}, nil
}

// GetActiveSessions returns all active sessions for a user
func (s *Server) GetActiveSessions(ctx context.Context, req *authv1.GetActiveSessionsRequest) (*authv1.GetActiveSessionsResponse, error) {
	if req.UserId == "" {
		return nil, status.Error(codes.InvalidArgument, "user_id is required")
	}
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}

	sessions, err := s.service.GetActiveSessions(ctx, req.TenantId, req.UserId)
	if err != nil {
		return nil, status.Error(codes.Internal, "failed to get sessions")
	}

	// Convert to proto
	protoSessions := make([]*authv1.Session, len(sessions))
	for i, sess := range sessions {
		deviceInfo := ""
		if ua, ok := sess.DeviceInfo["user_agent"]; ok {
			deviceInfo = ua
		}

		protoSessions[i] = &authv1.Session{
			Id:           sess.ID,
			UserId:       sess.UserID,
			DeviceInfo:   deviceInfo,
			IpAddress:    sess.IPAddress,
			CreatedAt:    timestamppb.New(sess.CreatedAt),
			LastActiveAt: timestamppb.New(sess.LastActiveAt),
			ExpiresAt:    timestamppb.New(sess.ExpiresAt),
		}
	}

	return &authv1.GetActiveSessionsResponse{
		Sessions: protoSessions,
	}, nil
}

// TerminateSession terminates a specific session
func (s *Server) TerminateSession(ctx context.Context, req *authv1.TerminateSessionRequest) (*authv1.TerminateSessionResponse, error) {
	if req.SessionId == "" {
		return nil, status.Error(codes.InvalidArgument, "session_id is required")
	}
	if req.UserId == "" {
		return nil, status.Error(codes.InvalidArgument, "user_id is required")
	}
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}

	err := s.service.TerminateSession(ctx, req.SessionId, req.UserId, req.TenantId)
	if err != nil {
		if errors.Is(err, ErrSessionNotFound) {
			return nil, status.Error(codes.NotFound, "session not found")
		}
		return nil, status.Error(codes.Internal, "failed to terminate session")
	}

	return &authv1.TerminateSessionResponse{
		Success: true,
	}, nil
}

// extractClaims extracts claims from a token string (with or without Bearer prefix)
func (s *Server) extractClaims(token string) (*TokenClaims, error) {
	// Remove "Bearer " prefix if present
	token = strings.TrimPrefix(token, "Bearer ")
	token = strings.TrimPrefix(token, "bearer ")

	return s.service.jwtManager.ValidateToken(token)
}
