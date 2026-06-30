package handlers

import (
	"context"
	"errors"

	"github.com/project/microservices/user-service/internal/models"
	"github.com/project/microservices/user-service/internal/repository"
	"golang.org/x/crypto/bcrypt"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// UserServiceServer implements the gRPC UserService
type UserServiceServer struct {
	repo *repository.UserRepository
	UnimplementedUserServiceServer
}

// UnimplementedUserServiceServer is embedded for forward compatibility
type UnimplementedUserServiceServer struct{}

// NewUserServiceServer creates a new user service server
func NewUserServiceServer(repo *repository.UserRepository) *UserServiceServer {
	return &UserServiceServer{
		repo: repo,
	}
}

// CreateUser creates a new user
func (s *UserServiceServer) CreateUser(ctx context.Context, req *CreateUserRequest) (*CreateUserResponse, error) {
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
	passwordHash, err := bcrypt.GenerateFromPassword([]byte(req.Password), bcrypt.DefaultCost)
	if err != nil {
		return nil, status.Error(codes.Internal, "failed to hash password")
	}

	input := &models.CreateUserInput{
		TenantID:  req.TenantId,
		Email:     req.Email,
		Password:  req.Password,
		FirstName: req.FirstName,
		LastName:  req.LastName,
		Metadata:  req.Metadata,
	}

	user, err := s.repo.Create(ctx, input, string(passwordHash))
	if err != nil {
		if errors.Is(err, repository.ErrUserExists) {
			return nil, status.Error(codes.AlreadyExists, "user already exists")
		}
		return nil, status.Errorf(codes.Internal, "failed to create user: %v", err)
	}

	return &CreateUserResponse{
		User: userToProto(user),
	}, nil
}

// GetUser retrieves a user by ID
func (s *UserServiceServer) GetUser(ctx context.Context, req *GetUserRequest) (*GetUserResponse, error) {
	if req.UserId == "" {
		return nil, status.Error(codes.InvalidArgument, "user_id is required")
	}
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}

	user, err := s.repo.GetByID(ctx, req.UserId, req.TenantId)
	if err != nil {
		if errors.Is(err, repository.ErrUserNotFound) {
			return nil, status.Error(codes.NotFound, "user not found")
		}
		return nil, status.Errorf(codes.Internal, "failed to get user: %v", err)
	}

	return &GetUserResponse{
		User: userToProto(user),
	}, nil
}

// UpdateUser updates an existing user
func (s *UserServiceServer) UpdateUser(ctx context.Context, req *UpdateUserRequest) (*UpdateUserResponse, error) {
	if req.UserId == "" {
		return nil, status.Error(codes.InvalidArgument, "user_id is required")
	}
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}

	input := &models.UpdateUserInput{
		UserID:   req.UserId,
		TenantID: req.TenantId,
		Metadata: req.Metadata,
	}

	if req.FirstName != nil {
		input.FirstName = req.FirstName
	}
	if req.LastName != nil {
		input.LastName = req.LastName
	}
	if req.AvatarUrl != nil {
		input.AvatarURL = req.AvatarUrl
	}
	if req.Status != nil {
		userStatus := models.UserStatus(*req.Status)
		input.Status = &userStatus
	}

	user, err := s.repo.Update(ctx, input)
	if err != nil {
		if errors.Is(err, repository.ErrUserNotFound) {
			return nil, status.Error(codes.NotFound, "user not found")
		}
		return nil, status.Errorf(codes.Internal, "failed to update user: %v", err)
	}

	return &UpdateUserResponse{
		User: userToProto(user),
	}, nil
}

// DeleteUser deletes a user
func (s *UserServiceServer) DeleteUser(ctx context.Context, req *DeleteUserRequest) (*DeleteUserResponse, error) {
	if req.UserId == "" {
		return nil, status.Error(codes.InvalidArgument, "user_id is required")
	}
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}

	err := s.repo.Delete(ctx, req.UserId, req.TenantId)
	if err != nil {
		if errors.Is(err, repository.ErrUserNotFound) {
			return nil, status.Error(codes.NotFound, "user not found")
		}
		return nil, status.Errorf(codes.Internal, "failed to delete user: %v", err)
	}

	return &DeleteUserResponse{
		Success: true,
	}, nil
}

// ListUsers lists users with pagination
func (s *UserServiceServer) ListUsers(ctx context.Context, req *ListUsersRequest) (*ListUsersResponse, error) {
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}

	var pageSize int32
	var pageToken string
	if req.Pagination != nil {
		pageSize = req.Pagination.PageSize
		pageToken = req.Pagination.PageToken
	}

	input := &models.ListUsersInput{
		TenantID:     req.TenantId,
		PageSize:     int(pageSize),
		PageToken:    pageToken,
		StatusFilter: req.StatusFilter,
		SearchQuery:  req.SearchQuery,
	}

	output, err := s.repo.List(ctx, input)
	if err != nil {
		if errors.Is(err, repository.ErrInvalidPageToken) {
			return nil, status.Error(codes.InvalidArgument, "invalid page token")
		}
		return nil, status.Errorf(codes.Internal, "failed to list users: %v", err)
	}

	users := make([]*User, len(output.Users))
	for i, u := range output.Users {
		users[i] = userToProto(u)
	}

	return &ListUsersResponse{
		Users: users,
		Pagination: &PaginationResponse{
			NextPageToken: output.NextPageToken,
			TotalCount:    int32(output.TotalCount),
		},
	}, nil
}

// GetUserByEmail retrieves a user by email (for auth service)
func (s *UserServiceServer) GetUserByEmail(ctx context.Context, req *GetUserByEmailRequest) (*GetUserByEmailResponse, error) {
	if req.Email == "" {
		return nil, status.Error(codes.InvalidArgument, "email is required")
	}
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}

	user, err := s.repo.GetByEmail(ctx, req.Email, req.TenantId)
	if err != nil {
		if errors.Is(err, repository.ErrUserNotFound) {
			return nil, status.Error(codes.NotFound, "user not found")
		}
		return nil, status.Errorf(codes.Internal, "failed to get user: %v", err)
	}

	return &GetUserByEmailResponse{
		User:         userToProto(user),
		PasswordHash: user.PasswordHash,
	}, nil
}

// userToProto converts a model user to proto user
func userToProto(u *models.User) *User {
	return &User{
		Id:            u.ID,
		TenantId:      u.TenantID,
		Email:         u.Email,
		EmailVerified: u.EmailVerified,
		FirstName:     u.FirstName,
		LastName:      u.LastName,
		AvatarUrl:     u.AvatarURL,
		Status:        string(u.Status),
		Metadata:      u.Metadata,
		Roles:         u.Roles,
		CreatedAt:     u.CreatedAt.Format("2006-01-02T15:04:05Z07:00"),
		UpdatedAt:     u.UpdatedAt.Format("2006-01-02T15:04:05Z07:00"),
	}
}

// Proto message types (simplified - would normally be generated)
type User struct {
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
	CreatedAt     string
	UpdatedAt     string
}

type CreateUserRequest struct {
	TenantId  string
	Email     string
	Password  string
	FirstName string
	LastName  string
	Metadata  map[string]string
}

type CreateUserResponse struct {
	User *User
}

type GetUserRequest struct {
	UserId   string
	TenantId string
}

type GetUserResponse struct {
	User *User
}

type UpdateUserRequest struct {
	UserId    string
	TenantId  string
	FirstName *string
	LastName  *string
	AvatarUrl *string
	Status    *string
	Metadata  map[string]string
}

type UpdateUserResponse struct {
	User *User
}

type DeleteUserRequest struct {
	UserId   string
	TenantId string
}

type DeleteUserResponse struct {
	Success bool
}

type PaginationRequest struct {
	PageSize  int32
	PageToken string
}

type PaginationResponse struct {
	NextPageToken string
	TotalCount    int32
}

type ListUsersRequest struct {
	TenantId     string
	Pagination   *PaginationRequest
	StatusFilter string
	SearchQuery  string
}

type ListUsersResponse struct {
	Users      []*User
	Pagination *PaginationResponse
}

type GetUserByEmailRequest struct {
	Email    string
	TenantId string
}

type GetUserByEmailResponse struct {
	User         *User
	PasswordHash string
}
