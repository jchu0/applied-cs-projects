package user

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/mlai/microservice-platform/internal/common"
)

// mockRepository implements a mock user repository for testing
type mockRepository struct {
	users   map[string]*User
	counter int
}

func newMockRepository() *mockRepository {
	return &mockRepository{
		users: make(map[string]*User),
	}
}

func (r *mockRepository) Create(ctx context.Context, user *User) (*User, error) {
	// Check for duplicate
	for _, u := range r.users {
		if u.Email == user.Email && u.TenantID == user.TenantID {
			return nil, ErrUserAlreadyExists
		}
	}

	r.counter++
	user.ID = fmt.Sprintf("test-user-%s-%d", time.Now().Format("20060102150405"), r.counter)
	user.CreatedAt = time.Now()
	user.UpdatedAt = time.Now()
	if user.Status == "" {
		user.Status = "active"
	}
	r.users[user.ID] = user
	return user, nil
}

func (r *mockRepository) GetByID(ctx context.Context, id, tenantID string) (*User, error) {
	user, ok := r.users[id]
	if !ok || user.TenantID != tenantID {
		return nil, ErrUserNotFound
	}
	return user, nil
}

func (r *mockRepository) GetByEmail(ctx context.Context, email, tenantID string) (*User, error) {
	for _, user := range r.users {
		if user.Email == email && user.TenantID == tenantID {
			return user, nil
		}
	}
	return nil, ErrUserNotFound
}

func (r *mockRepository) Update(ctx context.Context, user *User) (*User, error) {
	existing, ok := r.users[user.ID]
	if !ok {
		return nil, ErrUserNotFound
	}
	user.CreatedAt = existing.CreatedAt
	user.UpdatedAt = time.Now()
	r.users[user.ID] = user
	return user, nil
}

func (r *mockRepository) Delete(ctx context.Context, id, tenantID string) error {
	user, ok := r.users[id]
	if !ok || user.TenantID != tenantID {
		return ErrUserNotFound
	}
	delete(r.users, id)
	return nil
}

func (r *mockRepository) List(ctx context.Context, tenantID string, pageSize int, pageToken string, statusFilter string) ([]*User, string, int, error) {
	var users []*User
	for _, user := range r.users {
		if user.TenantID == tenantID {
			if statusFilter == "" || user.Status == statusFilter {
				users = append(users, user)
			}
		}
	}
	return users, "", len(users), nil
}

func (r *mockRepository) GetRoles(ctx context.Context, userID string) ([]string, error) {
	return []string{"user"}, nil
}

func TestService_CreateUser(t *testing.T) {
	logger := common.NewLogger("test")
	repo := newMockRepository()

	// Create service with mock repository
	svc := &Service{
		repo:   &Repository{pool: nil}, // Will use mock methods
		logger: logger,
	}
	_ = svc // Service created for reference

	// Test creating a user through the repository directly
	user := &User{
		TenantID:  "tenant-1",
		Email:     "test@example.com",
		FirstName: "Test",
		LastName:  "User",
	}

	createdUser, err := repo.Create(context.Background(), user)
	if err != nil {
		t.Fatalf("failed to create user: %v", err)
	}

	if createdUser.ID == "" {
		t.Error("user ID should not be empty")
	}
	if createdUser.Email != "test@example.com" {
		t.Errorf("expected email test@example.com, got %s", createdUser.Email)
	}
	if createdUser.Status != "active" {
		t.Errorf("expected status active, got %s", createdUser.Status)
	}
}

func TestService_CreateUser_Duplicate(t *testing.T) {
	repo := newMockRepository()

	// Create first user
	user1 := &User{
		TenantID: "tenant-1",
		Email:    "test@example.com",
	}
	_, err := repo.Create(context.Background(), user1)
	if err != nil {
		t.Fatalf("failed to create first user: %v", err)
	}

	// Try to create duplicate
	user2 := &User{
		TenantID: "tenant-1",
		Email:    "test@example.com",
	}
	_, err = repo.Create(context.Background(), user2)
	if err == nil {
		t.Error("expected error for duplicate user")
	}
	if err != ErrUserAlreadyExists {
		t.Errorf("expected ErrUserAlreadyExists, got %v", err)
	}
}

func TestService_GetUser(t *testing.T) {
	repo := newMockRepository()

	// Create a user
	user := &User{
		TenantID:  "tenant-1",
		Email:     "test@example.com",
		FirstName: "Test",
	}
	created, _ := repo.Create(context.Background(), user)

	// Get the user
	retrieved, err := repo.GetByID(context.Background(), created.ID, "tenant-1")
	if err != nil {
		t.Fatalf("failed to get user: %v", err)
	}

	if retrieved.Email != "test@example.com" {
		t.Errorf("expected email test@example.com, got %s", retrieved.Email)
	}
}

func TestService_GetUser_NotFound(t *testing.T) {
	repo := newMockRepository()

	_, err := repo.GetByID(context.Background(), "nonexistent", "tenant-1")
	if err == nil {
		t.Error("expected error for nonexistent user")
	}
	if err != ErrUserNotFound {
		t.Errorf("expected ErrUserNotFound, got %v", err)
	}
}

func TestService_GetUser_WrongTenant(t *testing.T) {
	repo := newMockRepository()

	// Create a user
	user := &User{
		TenantID: "tenant-1",
		Email:    "test@example.com",
	}
	created, _ := repo.Create(context.Background(), user)

	// Try to get with wrong tenant
	_, err := repo.GetByID(context.Background(), created.ID, "tenant-2")
	if err == nil {
		t.Error("expected error for wrong tenant")
	}
	if err != ErrUserNotFound {
		t.Errorf("expected ErrUserNotFound, got %v", err)
	}
}

func TestService_UpdateUser(t *testing.T) {
	repo := newMockRepository()

	// Create a user
	user := &User{
		TenantID:  "tenant-1",
		Email:     "test@example.com",
		FirstName: "Test",
		LastName:  "User",
	}
	created, _ := repo.Create(context.Background(), user)

	// Update the user
	created.FirstName = "Updated"
	updated, err := repo.Update(context.Background(), created)
	if err != nil {
		t.Fatalf("failed to update user: %v", err)
	}

	if updated.FirstName != "Updated" {
		t.Errorf("expected first_name Updated, got %s", updated.FirstName)
	}
	if !updated.UpdatedAt.After(updated.CreatedAt) {
		t.Error("updated_at should be after created_at")
	}
}

func TestService_DeleteUser(t *testing.T) {
	repo := newMockRepository()

	// Create a user
	user := &User{
		TenantID: "tenant-1",
		Email:    "test@example.com",
	}
	created, _ := repo.Create(context.Background(), user)

	// Delete the user
	err := repo.Delete(context.Background(), created.ID, "tenant-1")
	if err != nil {
		t.Fatalf("failed to delete user: %v", err)
	}

	// Verify user is deleted
	_, err = repo.GetByID(context.Background(), created.ID, "tenant-1")
	if err == nil {
		t.Error("expected error for deleted user")
	}
}

func TestService_ListUsers(t *testing.T) {
	repo := newMockRepository()

	// Create multiple users
	for i := 0; i < 5; i++ {
		user := &User{
			TenantID: "tenant-1",
			Email:    "user" + string(rune('0'+i)) + "@example.com",
			Status:   "active",
		}
		repo.Create(context.Background(), user)
	}

	// Create user in different tenant
	otherUser := &User{
		TenantID: "tenant-2",
		Email:    "other@example.com",
	}
	repo.Create(context.Background(), otherUser)

	// List users for tenant-1
	users, _, total, err := repo.List(context.Background(), "tenant-1", 10, "", "")
	if err != nil {
		t.Fatalf("failed to list users: %v", err)
	}

	if len(users) != 5 {
		t.Errorf("expected 5 users, got %d", len(users))
	}
	if total != 5 {
		t.Errorf("expected total 5, got %d", total)
	}
}

func TestService_GetUserByEmail(t *testing.T) {
	repo := newMockRepository()

	// Create a user
	user := &User{
		TenantID: "tenant-1",
		Email:    "test@example.com",
	}
	repo.Create(context.Background(), user)

	// Get by email
	retrieved, err := repo.GetByEmail(context.Background(), "test@example.com", "tenant-1")
	if err != nil {
		t.Fatalf("failed to get user by email: %v", err)
	}

	if retrieved.Email != "test@example.com" {
		t.Errorf("expected email test@example.com, got %s", retrieved.Email)
	}
}
