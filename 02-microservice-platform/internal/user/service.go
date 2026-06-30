package user

import (
	"context"
	"errors"

	"github.com/mlai/microservice-platform/internal/common"
	"golang.org/x/crypto/bcrypt"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// Service implements the user gRPC service
type Service struct {
	repo   *Repository
	logger *common.Logger
}

// NewService creates a new user service
func NewService(repo *Repository, logger *common.Logger) *Service {
	return &Service{
		repo:   repo,
		logger: logger,
	}
}

// CreateUser creates a new user
func (s *Service) CreateUser(ctx context.Context, req *CreateUserRequest) (*CreateUserResponse, error) {
	// Validate request
	if req.Email == "" {
		return nil, status.Error(codes.InvalidArgument, "email is required")
	}
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}
	if req.Password == "" {
		return nil, status.Error(codes.InvalidArgument, "password is required")
	}

	// Hash password
	hashedPassword, err := bcrypt.GenerateFromPassword([]byte(req.Password), bcrypt.DefaultCost)
	if err != nil {
		s.logger.Error("failed to hash password", "error", err)
		return nil, status.Error(codes.Internal, "failed to process password")
	}

	// Create user
	user := &User{
		TenantID:     req.TenantId,
		Email:        req.Email,
		PasswordHash: string(hashedPassword),
		FirstName:    req.FirstName,
		LastName:     req.LastName,
		Metadata:     req.Metadata,
	}

	createdUser, err := s.repo.Create(ctx, user)
	if err != nil {
		if errors.Is(err, ErrUserAlreadyExists) {
			return nil, status.Error(codes.AlreadyExists, "user already exists")
		}
		s.logger.Error("failed to create user", "error", err)
		return nil, status.Error(codes.Internal, "failed to create user")
	}

	s.logger.WithTenant(req.TenantId).Info("user created", "user_id", createdUser.ID)

	return &CreateUserResponse{
		User: userToProto(createdUser),
	}, nil
}

// GetUser retrieves a user by ID
func (s *Service) GetUser(ctx context.Context, req *GetUserRequest) (*GetUserResponse, error) {
	if req.Id == "" {
		return nil, status.Error(codes.InvalidArgument, "id is required")
	}
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}

	user, err := s.repo.GetByID(ctx, req.Id, req.TenantId)
	if err != nil {
		if errors.Is(err, ErrUserNotFound) {
			return nil, status.Error(codes.NotFound, "user not found")
		}
		s.logger.Error("failed to get user", "error", err)
		return nil, status.Error(codes.Internal, "failed to get user")
	}

	// Get roles
	roles, err := s.repo.GetRoles(ctx, user.ID)
	if err != nil {
		s.logger.Error("failed to get user roles", "error", err)
	}
	user.Roles = roles

	return &GetUserResponse{
		User: userToProto(user),
	}, nil
}

// UpdateUser updates a user
func (s *Service) UpdateUser(ctx context.Context, req *UpdateUserRequest) (*UpdateUserResponse, error) {
	if req.Id == "" {
		return nil, status.Error(codes.InvalidArgument, "id is required")
	}
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}

	// Get existing user
	existingUser, err := s.repo.GetByID(ctx, req.Id, req.TenantId)
	if err != nil {
		if errors.Is(err, ErrUserNotFound) {
			return nil, status.Error(codes.NotFound, "user not found")
		}
		return nil, status.Error(codes.Internal, "failed to get user")
	}

	// Apply updates
	if req.FirstName != nil {
		existingUser.FirstName = *req.FirstName
	}
	if req.LastName != nil {
		existingUser.LastName = *req.LastName
	}
	if req.AvatarUrl != nil {
		existingUser.AvatarURL = *req.AvatarUrl
	}
	if req.Status != nil {
		existingUser.Status = *req.Status
	}
	if req.Metadata != nil {
		existingUser.Metadata = req.Metadata
	}

	updatedUser, err := s.repo.Update(ctx, existingUser)
	if err != nil {
		s.logger.Error("failed to update user", "error", err)
		return nil, status.Error(codes.Internal, "failed to update user")
	}

	s.logger.WithTenant(req.TenantId).Info("user updated", "user_id", updatedUser.ID)

	return &UpdateUserResponse{
		User: userToProto(updatedUser),
	}, nil
}

// DeleteUser deletes a user
func (s *Service) DeleteUser(ctx context.Context, req *DeleteUserRequest) (*DeleteUserResponse, error) {
	if req.Id == "" {
		return nil, status.Error(codes.InvalidArgument, "id is required")
	}
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}

	err := s.repo.Delete(ctx, req.Id, req.TenantId)
	if err != nil {
		if errors.Is(err, ErrUserNotFound) {
			return nil, status.Error(codes.NotFound, "user not found")
		}
		s.logger.Error("failed to delete user", "error", err)
		return nil, status.Error(codes.Internal, "failed to delete user")
	}

	s.logger.WithTenant(req.TenantId).Info("user deleted", "user_id", req.Id)

	return &DeleteUserResponse{
		Success: true,
	}, nil
}

// ListUsers lists users with pagination
func (s *Service) ListUsers(ctx context.Context, req *ListUsersRequest) (*ListUsersResponse, error) {
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}

	pageSize := int(req.Pagination.PageSize)
	if pageSize <= 0 {
		pageSize = 20
	}
	if pageSize > 100 {
		pageSize = 100
	}

	users, nextPageToken, totalCount, err := s.repo.List(
		ctx, req.TenantId, pageSize, req.Pagination.PageToken, req.StatusFilter,
	)
	if err != nil {
		s.logger.Error("failed to list users", "error", err)
		return nil, status.Error(codes.Internal, "failed to list users")
	}

	protoUsers := make([]*UserProto, len(users))
	for i, user := range users {
		protoUsers[i] = userToProto(user)
	}

	return &ListUsersResponse{
		Users: protoUsers,
		Pagination: &PaginationResponse{
			NextPageToken: nextPageToken,
			TotalCount:    int32(totalCount),
		},
	}, nil
}

// GetUserByEmail retrieves a user by email
func (s *Service) GetUserByEmail(ctx context.Context, req *GetUserByEmailRequest) (*GetUserByEmailResponse, error) {
	if req.Email == "" {
		return nil, status.Error(codes.InvalidArgument, "email is required")
	}
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}

	user, err := s.repo.GetByEmail(ctx, req.Email, req.TenantId)
	if err != nil {
		if errors.Is(err, ErrUserNotFound) {
			return nil, status.Error(codes.NotFound, "user not found")
		}
		s.logger.Error("failed to get user by email", "error", err)
		return nil, status.Error(codes.Internal, "failed to get user")
	}

	return &GetUserByEmailResponse{
		User: userToProto(user),
	}, nil
}

// UserProto represents the protobuf User message
type UserProto struct {
	Id            string
	TenantId      string
	Email         string
	EmailVerified bool
	FirstName     string
	LastName      string
	AvatarUrl     string
	Status        string
	Metadata      map[string]string
	Roles         []string
	CreatedAt     *timestamppb.Timestamp
	UpdatedAt     *timestamppb.Timestamp
}

// Request/Response types (these would normally come from generated protobuf)
type CreateUserRequest struct {
	TenantId  string
	Email     string
	Password  string
	FirstName string
	LastName  string
	Metadata  map[string]string
}

type CreateUserResponse struct {
	User *UserProto
}

type GetUserRequest struct {
	Id       string
	TenantId string
}

type GetUserResponse struct {
	User *UserProto
}

type UpdateUserRequest struct {
	Id        string
	TenantId  string
	FirstName *string
	LastName  *string
	AvatarUrl *string
	Status    *string
	Metadata  map[string]string
}

type UpdateUserResponse struct {
	User *UserProto
}

type DeleteUserRequest struct {
	Id       string
	TenantId string
}

type DeleteUserResponse struct {
	Success bool
}

type ListUsersRequest struct {
	TenantId     string
	Pagination   *PaginationRequest
	StatusFilter string
}

type ListUsersResponse struct {
	Users      []*UserProto
	Pagination *PaginationResponse
}

type GetUserByEmailRequest struct {
	Email    string
	TenantId string
}

type GetUserByEmailResponse struct {
	User *UserProto
}

type PaginationRequest struct {
	PageSize  int32
	PageToken string
}

type PaginationResponse struct {
	NextPageToken string
	TotalCount    int32
}

// userToProto converts a User to protobuf format
func userToProto(user *User) *UserProto {
	return &UserProto{
		Id:            user.ID,
		TenantId:      user.TenantID,
		Email:         user.Email,
		EmailVerified: user.EmailVerified,
		FirstName:     user.FirstName,
		LastName:      user.LastName,
		AvatarUrl:     user.AvatarURL,
		Status:        user.Status,
		Metadata:      user.Metadata,
		Roles:         user.Roles,
		CreatedAt:     timestamppb.New(user.CreatedAt),
		UpdatedAt:     timestamppb.New(user.UpdatedAt),
	}
}
