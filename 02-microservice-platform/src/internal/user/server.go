package user

import (
	"context"
	"errors"

	userv1 "github.com/mlai/microservice-platform/pkg/pb/user/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// Server implements the UserService gRPC server
type Server struct {
	userv1.UnimplementedUserServiceServer
	service *Service
}

// NewServer creates a new gRPC server
func NewServer(service *Service) *Server {
	return &Server{service: service}
}

// CreateUser creates a new user
func (s *Server) CreateUser(ctx context.Context, req *userv1.CreateUserRequest) (*userv1.CreateUserResponse, error) {
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}
	if req.Email == "" {
		return nil, status.Error(codes.InvalidArgument, "email is required")
	}
	if req.Password == "" {
		return nil, status.Error(codes.InvalidArgument, "password is required")
	}

	user, err := s.service.CreateUser(
		ctx,
		req.TenantId,
		req.Email,
		req.Password,
		req.FirstName,
		req.LastName,
		req.Metadata,
	)
	if err != nil {
		if errors.Is(err, ErrEmailExists) {
			return nil, status.Error(codes.AlreadyExists, "email already exists")
		}
		return nil, status.Error(codes.Internal, "failed to create user")
	}

	return &userv1.CreateUserResponse{
		User: userToProto(user),
	}, nil
}

// GetUser retrieves a user by ID
func (s *Server) GetUser(ctx context.Context, req *userv1.GetUserRequest) (*userv1.GetUserResponse, error) {
	if req.UserId == "" {
		return nil, status.Error(codes.InvalidArgument, "user_id is required")
	}
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}

	user, err := s.service.GetUser(ctx, req.TenantId, req.UserId)
	if err != nil {
		if errors.Is(err, ErrUserNotFound) {
			return nil, status.Error(codes.NotFound, "user not found")
		}
		return nil, status.Error(codes.Internal, "failed to get user")
	}

	return &userv1.GetUserResponse{
		User: userToProto(user),
	}, nil
}

// GetUserByEmail retrieves a user by email
func (s *Server) GetUserByEmail(ctx context.Context, req *userv1.GetUserByEmailRequest) (*userv1.GetUserByEmailResponse, error) {
	if req.Email == "" {
		return nil, status.Error(codes.InvalidArgument, "email is required")
	}
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}

	user, err := s.service.GetUserByEmail(ctx, req.TenantId, req.Email)
	if err != nil {
		if errors.Is(err, ErrUserNotFound) {
			return nil, status.Error(codes.NotFound, "user not found")
		}
		return nil, status.Error(codes.Internal, "failed to get user")
	}

	return &userv1.GetUserByEmailResponse{
		User:         userToProto(user),
		PasswordHash: user.PasswordHash,
	}, nil
}

// UpdateUser updates user information
func (s *Server) UpdateUser(ctx context.Context, req *userv1.UpdateUserRequest) (*userv1.UpdateUserResponse, error) {
	if req.UserId == "" {
		return nil, status.Error(codes.InvalidArgument, "user_id is required")
	}
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}

	// Get existing user
	existing, err := s.service.GetUser(ctx, req.TenantId, req.UserId)
	if err != nil {
		if errors.Is(err, ErrUserNotFound) {
			return nil, status.Error(codes.NotFound, "user not found")
		}
		return nil, status.Error(codes.Internal, "failed to get user")
	}

	// Apply updates
	if req.Email != "" {
		existing.Email = req.Email
	}
	if req.FirstName != "" {
		existing.FirstName = req.FirstName
	}
	if req.LastName != "" {
		existing.LastName = req.LastName
	}
	if req.AvatarUrl != "" {
		existing.AvatarURL = req.AvatarUrl
	}
	if req.Status != userv1.UserStatus_USER_STATUS_UNSPECIFIED {
		existing.Status = statusToString(req.Status)
	}
	if req.Metadata != nil {
		existing.Metadata = req.Metadata
	}

	user, err := s.service.UpdateUser(ctx, existing)
	if err != nil {
		if errors.Is(err, ErrEmailExists) {
			return nil, status.Error(codes.AlreadyExists, "email already exists")
		}
		return nil, status.Error(codes.Internal, "failed to update user")
	}

	return &userv1.UpdateUserResponse{
		User: userToProto(user),
	}, nil
}

// DeleteUser soft-deletes a user
func (s *Server) DeleteUser(ctx context.Context, req *userv1.DeleteUserRequest) (*userv1.DeleteUserResponse, error) {
	if req.UserId == "" {
		return nil, status.Error(codes.InvalidArgument, "user_id is required")
	}
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}

	err := s.service.DeleteUser(ctx, req.TenantId, req.UserId)
	if err != nil {
		if errors.Is(err, ErrUserNotFound) {
			return nil, status.Error(codes.NotFound, "user not found")
		}
		return nil, status.Error(codes.Internal, "failed to delete user")
	}

	return &userv1.DeleteUserResponse{
		Success: true,
	}, nil
}

// ListUsers lists users with pagination
func (s *Server) ListUsers(ctx context.Context, req *userv1.ListUsersRequest) (*userv1.ListUsersResponse, error) {
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}

	// Parse pagination
	pageSize := int32(20)
	offset := 0
	if req.Pagination != nil {
		if req.Pagination.PageSize > 0 {
			pageSize = req.Pagination.PageSize
		}
		// Page token is the offset
		if req.Pagination.PageToken != "" {
			// Parse offset from token
			// For simplicity, token is just the offset as string
		}
	}

	opts := ListOptions{
		Limit:        int(pageSize),
		Offset:       offset,
		StatusFilter: statusToString(req.StatusFilter),
		Search:       req.Search,
	}

	users, total, err := s.service.ListUsers(ctx, req.TenantId, opts)
	if err != nil {
		return nil, status.Error(codes.Internal, "failed to list users")
	}

	// Convert to proto
	protoUsers := make([]*userv1.User, len(users))
	for i, u := range users {
		protoUsers[i] = userToProto(u)
	}

	// Build pagination response
	hasMore := offset+len(users) < total
	nextToken := ""
	if hasMore {
		nextToken = "" // Would encode next offset
	}

	return &userv1.ListUsersResponse{
		Users: protoUsers,
		Pagination: &userv1.ListUsersResponse_PaginationResponse{
			NextPageToken: nextToken,
			TotalCount:    int32(total),
			HasMore:       hasMore,
		},
	}, nil
}

// AssignRole assigns a role to a user
func (s *Server) AssignRole(ctx context.Context, req *userv1.AssignRoleRequest) (*userv1.AssignRoleResponse, error) {
	if req.UserId == "" {
		return nil, status.Error(codes.InvalidArgument, "user_id is required")
	}
	if req.RoleId == "" {
		return nil, status.Error(codes.InvalidArgument, "role_id is required")
	}
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}

	err := s.service.AssignRole(ctx, req.UserId, req.RoleId, req.TenantId, req.GrantedBy)
	if err != nil {
		if errors.Is(err, ErrUserNotFound) {
			return nil, status.Error(codes.NotFound, "user or role not found")
		}
		if errors.Is(err, ErrRoleAlreadyAssigned) {
			return nil, status.Error(codes.AlreadyExists, "role already assigned")
		}
		return nil, status.Error(codes.Internal, "failed to assign role")
	}

	return &userv1.AssignRoleResponse{
		Success: true,
	}, nil
}

// RevokeRole removes a role from a user
func (s *Server) RevokeRole(ctx context.Context, req *userv1.RevokeRoleRequest) (*userv1.RevokeRoleResponse, error) {
	if req.UserId == "" {
		return nil, status.Error(codes.InvalidArgument, "user_id is required")
	}
	if req.RoleId == "" {
		return nil, status.Error(codes.InvalidArgument, "role_id is required")
	}
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}

	err := s.service.RevokeRole(ctx, req.UserId, req.RoleId, req.TenantId)
	if err != nil {
		if errors.Is(err, ErrRoleNotFound) {
			return nil, status.Error(codes.NotFound, "role assignment not found")
		}
		return nil, status.Error(codes.Internal, "failed to revoke role")
	}

	return &userv1.RevokeRoleResponse{
		Success: true,
	}, nil
}

// GetUserRoles gets all roles for a user
func (s *Server) GetUserRoles(ctx context.Context, req *userv1.GetUserRolesRequest) (*userv1.GetUserRolesResponse, error) {
	if req.UserId == "" {
		return nil, status.Error(codes.InvalidArgument, "user_id is required")
	}
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}

	roles, err := s.service.GetUserRoles(ctx, req.TenantId, req.UserId)
	if err != nil {
		return nil, status.Error(codes.Internal, "failed to get user roles")
	}

	// Convert to proto
	protoRoles := make([]*userv1.Role, len(roles))
	for i, r := range roles {
		protoRoles[i] = &userv1.Role{
			Id:          r.ID,
			TenantId:    r.TenantID,
			Name:        r.Name,
			Description: r.Description,
			Permissions: r.Permissions,
			CreatedAt:   timestamppb.New(r.CreatedAt),
		}
	}

	return &userv1.GetUserRolesResponse{
		Roles: protoRoles,
	}, nil
}

// Helper functions

func userToProto(u *User) *userv1.User {
	return &userv1.User{
		Id:            u.ID,
		TenantId:      u.TenantID,
		Email:         u.Email,
		EmailVerified: u.EmailVerified,
		FirstName:     u.FirstName,
		LastName:      u.LastName,
		AvatarUrl:     u.AvatarURL,
		Status:        stringToStatus(u.Status),
		Metadata:      u.Metadata,
		CreatedAt:     timestamppb.New(u.CreatedAt),
		UpdatedAt:     timestamppb.New(u.UpdatedAt),
	}
}

func stringToStatus(s string) userv1.UserStatus {
	switch s {
	case "active":
		return userv1.UserStatus_USER_STATUS_ACTIVE
	case "inactive":
		return userv1.UserStatus_USER_STATUS_INACTIVE
	case "suspended":
		return userv1.UserStatus_USER_STATUS_SUSPENDED
	case "deleted":
		return userv1.UserStatus_USER_STATUS_DELETED
	default:
		return userv1.UserStatus_USER_STATUS_UNSPECIFIED
	}
}

func statusToString(s userv1.UserStatus) string {
	switch s {
	case userv1.UserStatus_USER_STATUS_ACTIVE:
		return "active"
	case userv1.UserStatus_USER_STATUS_INACTIVE:
		return "inactive"
	case userv1.UserStatus_USER_STATUS_SUSPENDED:
		return "suspended"
	case userv1.UserStatus_USER_STATUS_DELETED:
		return "deleted"
	default:
		return ""
	}
}
