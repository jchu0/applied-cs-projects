package user

import (
	"context"
	"fmt"

	"github.com/mlai/microservice-platform/pkg/logging"
	"go.uber.org/zap"
	"golang.org/x/crypto/bcrypt"
)

// Service handles user business logic
type Service struct {
	repo   *Repository
	logger *logging.Logger
}

// NewService creates a new user service
func NewService(repo *Repository, logger *logging.Logger) *Service {
	return &Service{
		repo:   repo,
		logger: logger,
	}
}

// CreateUser creates a new user with hashed password
func (s *Service) CreateUser(ctx context.Context, tenantID, email, password, firstName, lastName string, metadata map[string]string) (*User, error) {
	logger := s.logger.WithContext(ctx)

	// Hash password
	passwordHash, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		logger.Error("failed to hash password", zap.Error(err))
		return nil, fmt.Errorf("failed to hash password: %w", err)
	}

	user := &User{
		TenantID:      tenantID,
		Email:         email,
		EmailVerified: false,
		PasswordHash:  string(passwordHash),
		FirstName:     firstName,
		LastName:      lastName,
		Status:        "active",
		Metadata:      metadata,
	}

	if err := s.repo.Create(ctx, user); err != nil {
		logger.Error("failed to create user",
			zap.String("email", email),
			zap.Error(err),
		)
		return nil, err
	}

	logger.Info("user created",
		zap.String("user_id", user.ID),
		zap.String("email", email),
	)

	// Clear password hash before returning
	user.PasswordHash = ""
	return user, nil
}

// GetUser retrieves a user by ID
func (s *Service) GetUser(ctx context.Context, tenantID, userID string) (*User, error) {
	user, err := s.repo.GetByID(ctx, tenantID, userID)
	if err != nil {
		return nil, err
	}

	// Clear password hash before returning
	user.PasswordHash = ""
	return user, nil
}

// GetUserByEmail retrieves a user by email (includes password hash for auth)
func (s *Service) GetUserByEmail(ctx context.Context, tenantID, email string) (*User, error) {
	return s.repo.GetByEmail(ctx, tenantID, email)
}

// UpdateUser updates user information
func (s *Service) UpdateUser(ctx context.Context, user *User) (*User, error) {
	logger := s.logger.WithContext(ctx)

	if err := s.repo.Update(ctx, user); err != nil {
		logger.Error("failed to update user",
			zap.String("user_id", user.ID),
			zap.Error(err),
		)
		return nil, err
	}

	logger.Info("user updated", zap.String("user_id", user.ID))

	// Retrieve updated user
	return s.GetUser(ctx, user.TenantID, user.ID)
}

// DeleteUser soft-deletes a user
func (s *Service) DeleteUser(ctx context.Context, tenantID, userID string) error {
	logger := s.logger.WithContext(ctx)

	if err := s.repo.Delete(ctx, tenantID, userID); err != nil {
		logger.Error("failed to delete user",
			zap.String("user_id", userID),
			zap.Error(err),
		)
		return err
	}

	logger.Info("user deleted", zap.String("user_id", userID))
	return nil
}

// ListUsers lists users with pagination
func (s *Service) ListUsers(ctx context.Context, tenantID string, opts ListOptions) ([]*User, int, error) {
	// Set defaults
	if opts.Limit <= 0 {
		opts.Limit = 20
	}
	if opts.Limit > 100 {
		opts.Limit = 100
	}

	users, total, err := s.repo.List(ctx, tenantID, opts)
	if err != nil {
		return nil, 0, err
	}

	// Clear password hashes
	for _, u := range users {
		u.PasswordHash = ""
	}

	return users, total, nil
}

// AssignRole assigns a role to a user
func (s *Service) AssignRole(ctx context.Context, userID, roleID, tenantID, grantedBy string) error {
	logger := s.logger.WithContext(ctx)

	if err := s.repo.AssignRole(ctx, userID, roleID, tenantID, grantedBy); err != nil {
		logger.Error("failed to assign role",
			zap.String("user_id", userID),
			zap.String("role_id", roleID),
			zap.Error(err),
		)
		return err
	}

	logger.Info("role assigned",
		zap.String("user_id", userID),
		zap.String("role_id", roleID),
	)
	return nil
}

// RevokeRole removes a role from a user
func (s *Service) RevokeRole(ctx context.Context, userID, roleID, tenantID string) error {
	logger := s.logger.WithContext(ctx)

	if err := s.repo.RevokeRole(ctx, userID, roleID, tenantID); err != nil {
		logger.Error("failed to revoke role",
			zap.String("user_id", userID),
			zap.String("role_id", roleID),
			zap.Error(err),
		)
		return err
	}

	logger.Info("role revoked",
		zap.String("user_id", userID),
		zap.String("role_id", roleID),
	)
	return nil
}

// GetUserRoles retrieves all roles for a user
func (s *Service) GetUserRoles(ctx context.Context, tenantID, userID string) ([]*Role, error) {
	return s.repo.GetUserRoles(ctx, tenantID, userID)
}

// VerifyPassword verifies if the password matches the stored hash
func (s *Service) VerifyPassword(hashedPassword, password string) bool {
	err := bcrypt.CompareHashAndPassword([]byte(hashedPassword), []byte(password))
	return err == nil
}
