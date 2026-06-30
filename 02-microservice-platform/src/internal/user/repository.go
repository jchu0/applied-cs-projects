package user

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Repository errors
var (
	ErrUserNotFound      = errors.New("user not found")
	ErrEmailExists       = errors.New("email already exists")
	ErrRoleNotFound      = errors.New("role not found")
	ErrRoleAlreadyAssigned = errors.New("role already assigned")
)

// User represents a user entity
type User struct {
	ID            string
	TenantID      string
	Email         string
	EmailVerified bool
	PasswordHash  string
	FirstName     string
	LastName      string
	AvatarURL     string
	Status        string
	Metadata      map[string]string
	CreatedAt     time.Time
	UpdatedAt     time.Time
}

// Role represents a user role
type Role struct {
	ID          string
	TenantID    string
	Name        string
	Description string
	Permissions []string
	CreatedAt   time.Time
}

// UserRole represents user-role assignment
type UserRole struct {
	UserID    string
	RoleID    string
	GrantedAt time.Time
	GrantedBy string
}

// Repository handles user data access
type Repository struct {
	pool *pgxpool.Pool
}

// NewRepository creates a new user repository
func NewRepository(pool *pgxpool.Pool) *Repository {
	return &Repository{pool: pool}
}

// Create creates a new user
func (r *Repository) Create(ctx context.Context, user *User) error {
	user.ID = uuid.New().String()
	user.CreatedAt = time.Now().UTC()
	user.UpdatedAt = user.CreatedAt

	query := `
		INSERT INTO users (
			id, tenant_id, email, email_verified, password_hash,
			first_name, last_name, avatar_url, status, metadata,
			created_at, updated_at
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
	`

	_, err := r.pool.Exec(ctx, query,
		user.ID, user.TenantID, user.Email, user.EmailVerified, user.PasswordHash,
		user.FirstName, user.LastName, user.AvatarURL, user.Status, user.Metadata,
		user.CreatedAt, user.UpdatedAt,
	)
	if err != nil {
		if isUniqueViolation(err) {
			return ErrEmailExists
		}
		return fmt.Errorf("failed to create user: %w", err)
	}

	return nil
}

// GetByID retrieves a user by ID
func (r *Repository) GetByID(ctx context.Context, tenantID, userID string) (*User, error) {
	query := `
		SELECT id, tenant_id, email, email_verified, password_hash,
			   first_name, last_name, avatar_url, status, metadata,
			   created_at, updated_at
		FROM users
		WHERE id = $1 AND tenant_id = $2 AND status != 'deleted'
	`

	user := &User{}
	err := r.pool.QueryRow(ctx, query, userID, tenantID).Scan(
		&user.ID, &user.TenantID, &user.Email, &user.EmailVerified, &user.PasswordHash,
		&user.FirstName, &user.LastName, &user.AvatarURL, &user.Status, &user.Metadata,
		&user.CreatedAt, &user.UpdatedAt,
	)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, ErrUserNotFound
		}
		return nil, fmt.Errorf("failed to get user: %w", err)
	}

	return user, nil
}

// GetByEmail retrieves a user by email
func (r *Repository) GetByEmail(ctx context.Context, tenantID, email string) (*User, error) {
	query := `
		SELECT id, tenant_id, email, email_verified, password_hash,
			   first_name, last_name, avatar_url, status, metadata,
			   created_at, updated_at
		FROM users
		WHERE email = $1 AND tenant_id = $2 AND status != 'deleted'
	`

	user := &User{}
	err := r.pool.QueryRow(ctx, query, email, tenantID).Scan(
		&user.ID, &user.TenantID, &user.Email, &user.EmailVerified, &user.PasswordHash,
		&user.FirstName, &user.LastName, &user.AvatarURL, &user.Status, &user.Metadata,
		&user.CreatedAt, &user.UpdatedAt,
	)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, ErrUserNotFound
		}
		return nil, fmt.Errorf("failed to get user by email: %w", err)
	}

	return user, nil
}

// Update updates user information
func (r *Repository) Update(ctx context.Context, user *User) error {
	user.UpdatedAt = time.Now().UTC()

	query := `
		UPDATE users SET
			email = $3,
			first_name = $4,
			last_name = $5,
			avatar_url = $6,
			status = $7,
			metadata = $8,
			updated_at = $9
		WHERE id = $1 AND tenant_id = $2
	`

	result, err := r.pool.Exec(ctx, query,
		user.ID, user.TenantID, user.Email,
		user.FirstName, user.LastName, user.AvatarURL,
		user.Status, user.Metadata, user.UpdatedAt,
	)
	if err != nil {
		if isUniqueViolation(err) {
			return ErrEmailExists
		}
		return fmt.Errorf("failed to update user: %w", err)
	}

	if result.RowsAffected() == 0 {
		return ErrUserNotFound
	}

	return nil
}

// Delete soft-deletes a user
func (r *Repository) Delete(ctx context.Context, tenantID, userID string) error {
	query := `
		UPDATE users SET
			status = 'deleted',
			updated_at = $3
		WHERE id = $1 AND tenant_id = $2
	`

	result, err := r.pool.Exec(ctx, query, userID, tenantID, time.Now().UTC())
	if err != nil {
		return fmt.Errorf("failed to delete user: %w", err)
	}

	if result.RowsAffected() == 0 {
		return ErrUserNotFound
	}

	return nil
}

// List retrieves users with pagination
func (r *Repository) List(ctx context.Context, tenantID string, opts ListOptions) ([]*User, int, error) {
	// Build query
	baseQuery := `
		FROM users
		WHERE tenant_id = $1 AND status != 'deleted'
	`
	args := []interface{}{tenantID}
	argIdx := 2

	// Status filter
	if opts.StatusFilter != "" {
		baseQuery += fmt.Sprintf(" AND status = $%d", argIdx)
		args = append(args, opts.StatusFilter)
		argIdx++
	}

	// Search filter
	if opts.Search != "" {
		baseQuery += fmt.Sprintf(" AND (email ILIKE $%d OR first_name ILIKE $%d OR last_name ILIKE $%d)", argIdx, argIdx, argIdx)
		args = append(args, "%"+opts.Search+"%")
		argIdx++
	}

	// Count total
	var total int
	countQuery := "SELECT COUNT(*) " + baseQuery
	err := r.pool.QueryRow(ctx, countQuery, args...).Scan(&total)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to count users: %w", err)
	}

	// Get users
	selectQuery := `
		SELECT id, tenant_id, email, email_verified, password_hash,
			   first_name, last_name, avatar_url, status, metadata,
			   created_at, updated_at
	` + baseQuery + fmt.Sprintf(" ORDER BY created_at DESC LIMIT $%d OFFSET $%d", argIdx, argIdx+1)
	args = append(args, opts.Limit, opts.Offset)

	rows, err := r.pool.Query(ctx, selectQuery, args...)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to list users: %w", err)
	}
	defer rows.Close()

	var users []*User
	for rows.Next() {
		user := &User{}
		err := rows.Scan(
			&user.ID, &user.TenantID, &user.Email, &user.EmailVerified, &user.PasswordHash,
			&user.FirstName, &user.LastName, &user.AvatarURL, &user.Status, &user.Metadata,
			&user.CreatedAt, &user.UpdatedAt,
		)
		if err != nil {
			return nil, 0, fmt.Errorf("failed to scan user: %w", err)
		}
		users = append(users, user)
	}

	return users, total, nil
}

// ListOptions for list queries
type ListOptions struct {
	Limit        int
	Offset       int
	StatusFilter string
	Search       string
}

// AssignRole assigns a role to a user
func (r *Repository) AssignRole(ctx context.Context, userID, roleID, tenantID, grantedBy string) error {
	query := `
		INSERT INTO user_roles (user_id, role_id, granted_at, granted_by)
		SELECT $1, $2, $3, $4
		WHERE EXISTS (SELECT 1 FROM users WHERE id = $1 AND tenant_id = $5)
		  AND EXISTS (SELECT 1 FROM roles WHERE id = $2 AND tenant_id = $5)
	`

	result, err := r.pool.Exec(ctx, query, userID, roleID, time.Now().UTC(), grantedBy, tenantID)
	if err != nil {
		if isUniqueViolation(err) {
			return ErrRoleAlreadyAssigned
		}
		return fmt.Errorf("failed to assign role: %w", err)
	}

	if result.RowsAffected() == 0 {
		return ErrUserNotFound
	}

	return nil
}

// RevokeRole removes a role from a user
func (r *Repository) RevokeRole(ctx context.Context, userID, roleID, tenantID string) error {
	query := `
		DELETE FROM user_roles
		WHERE user_id = $1 AND role_id = $2
		  AND EXISTS (SELECT 1 FROM users WHERE id = $1 AND tenant_id = $3)
	`

	result, err := r.pool.Exec(ctx, query, userID, roleID, tenantID)
	if err != nil {
		return fmt.Errorf("failed to revoke role: %w", err)
	}

	if result.RowsAffected() == 0 {
		return ErrRoleNotFound
	}

	return nil
}

// GetUserRoles retrieves all roles for a user
func (r *Repository) GetUserRoles(ctx context.Context, tenantID, userID string) ([]*Role, error) {
	query := `
		SELECT r.id, r.tenant_id, r.name, r.description, r.permissions, r.created_at
		FROM roles r
		JOIN user_roles ur ON r.id = ur.role_id
		WHERE ur.user_id = $1 AND r.tenant_id = $2
		ORDER BY r.name
	`

	rows, err := r.pool.Query(ctx, query, userID, tenantID)
	if err != nil {
		return nil, fmt.Errorf("failed to get user roles: %w", err)
	}
	defer rows.Close()

	var roles []*Role
	for rows.Next() {
		role := &Role{}
		err := rows.Scan(
			&role.ID, &role.TenantID, &role.Name, &role.Description,
			&role.Permissions, &role.CreatedAt,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan role: %w", err)
		}
		roles = append(roles, role)
	}

	return roles, nil
}

// isUniqueViolation checks if error is a unique constraint violation
func isUniqueViolation(err error) bool {
	// PostgreSQL unique violation error code
	return err != nil && err.Error() != "" &&
		(contains(err.Error(), "23505") || contains(err.Error(), "unique constraint"))
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && containsHelper(s, substr))
}

func containsHelper(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
