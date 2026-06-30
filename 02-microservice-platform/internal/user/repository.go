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

var (
	ErrUserNotFound      = errors.New("user not found")
	ErrUserAlreadyExists = errors.New("user already exists")
)

// User represents a user in the database
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
	Roles         []string
	CreatedAt     time.Time
	UpdatedAt     time.Time
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
func (r *Repository) Create(ctx context.Context, user *User) (*User, error) {
	user.ID = uuid.New().String()
	user.CreatedAt = time.Now()
	user.UpdatedAt = time.Now()

	if user.Status == "" {
		user.Status = "active"
	}

	query := `
		INSERT INTO users (id, tenant_id, email, email_verified, password_hash,
			first_name, last_name, avatar_url, status, metadata, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
		RETURNING id, tenant_id, email, email_verified, first_name, last_name,
			avatar_url, status, metadata, created_at, updated_at
	`

	err := r.pool.QueryRow(ctx, query,
		user.ID, user.TenantID, user.Email, user.EmailVerified, user.PasswordHash,
		user.FirstName, user.LastName, user.AvatarURL, user.Status, user.Metadata,
		user.CreatedAt, user.UpdatedAt,
	).Scan(
		&user.ID, &user.TenantID, &user.Email, &user.EmailVerified,
		&user.FirstName, &user.LastName, &user.AvatarURL, &user.Status,
		&user.Metadata, &user.CreatedAt, &user.UpdatedAt,
	)

	if err != nil {
		if err.Error() == "ERROR: duplicate key value violates unique constraint" {
			return nil, ErrUserAlreadyExists
		}
		return nil, fmt.Errorf("failed to create user: %w", err)
	}

	return user, nil
}

// GetByID retrieves a user by ID
func (r *Repository) GetByID(ctx context.Context, id, tenantID string) (*User, error) {
	query := `
		SELECT id, tenant_id, email, email_verified, password_hash, first_name,
			last_name, avatar_url, status, metadata, created_at, updated_at
		FROM users
		WHERE id = $1 AND tenant_id = $2
	`

	var user User
	err := r.pool.QueryRow(ctx, query, id, tenantID).Scan(
		&user.ID, &user.TenantID, &user.Email, &user.EmailVerified, &user.PasswordHash,
		&user.FirstName, &user.LastName, &user.AvatarURL, &user.Status,
		&user.Metadata, &user.CreatedAt, &user.UpdatedAt,
	)

	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, ErrUserNotFound
		}
		return nil, fmt.Errorf("failed to get user: %w", err)
	}

	return &user, nil
}

// GetByEmail retrieves a user by email
func (r *Repository) GetByEmail(ctx context.Context, email, tenantID string) (*User, error) {
	query := `
		SELECT id, tenant_id, email, email_verified, password_hash, first_name,
			last_name, avatar_url, status, metadata, created_at, updated_at
		FROM users
		WHERE email = $1 AND tenant_id = $2
	`

	var user User
	err := r.pool.QueryRow(ctx, query, email, tenantID).Scan(
		&user.ID, &user.TenantID, &user.Email, &user.EmailVerified, &user.PasswordHash,
		&user.FirstName, &user.LastName, &user.AvatarURL, &user.Status,
		&user.Metadata, &user.CreatedAt, &user.UpdatedAt,
	)

	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, ErrUserNotFound
		}
		return nil, fmt.Errorf("failed to get user by email: %w", err)
	}

	return &user, nil
}

// Update updates a user
func (r *Repository) Update(ctx context.Context, user *User) (*User, error) {
	user.UpdatedAt = time.Now()

	query := `
		UPDATE users
		SET first_name = $1, last_name = $2, avatar_url = $3, status = $4,
			metadata = $5, updated_at = $6
		WHERE id = $7 AND tenant_id = $8
		RETURNING id, tenant_id, email, email_verified, first_name, last_name,
			avatar_url, status, metadata, created_at, updated_at
	`

	err := r.pool.QueryRow(ctx, query,
		user.FirstName, user.LastName, user.AvatarURL, user.Status,
		user.Metadata, user.UpdatedAt, user.ID, user.TenantID,
	).Scan(
		&user.ID, &user.TenantID, &user.Email, &user.EmailVerified,
		&user.FirstName, &user.LastName, &user.AvatarURL, &user.Status,
		&user.Metadata, &user.CreatedAt, &user.UpdatedAt,
	)

	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, ErrUserNotFound
		}
		return nil, fmt.Errorf("failed to update user: %w", err)
	}

	return user, nil
}

// Delete deletes a user
func (r *Repository) Delete(ctx context.Context, id, tenantID string) error {
	query := `DELETE FROM users WHERE id = $1 AND tenant_id = $2`

	result, err := r.pool.Exec(ctx, query, id, tenantID)
	if err != nil {
		return fmt.Errorf("failed to delete user: %w", err)
	}

	if result.RowsAffected() == 0 {
		return ErrUserNotFound
	}

	return nil
}

// List lists users with pagination
func (r *Repository) List(ctx context.Context, tenantID string, pageSize int, pageToken string, statusFilter string) ([]*User, string, int, error) {
	var users []*User
	var args []interface{}
	argIndex := 1

	query := `
		SELECT id, tenant_id, email, email_verified, first_name, last_name,
			avatar_url, status, metadata, created_at, updated_at
		FROM users
		WHERE tenant_id = $1
	`
	args = append(args, tenantID)
	argIndex++

	if statusFilter != "" {
		query += fmt.Sprintf(" AND status = $%d", argIndex)
		args = append(args, statusFilter)
		argIndex++
	}

	if pageToken != "" {
		query += fmt.Sprintf(" AND id > $%d", argIndex)
		args = append(args, pageToken)
		argIndex++
	}

	query += " ORDER BY id LIMIT $" + fmt.Sprint(argIndex)
	args = append(args, pageSize+1) // Fetch one extra to determine if there's a next page

	rows, err := r.pool.Query(ctx, query, args...)
	if err != nil {
		return nil, "", 0, fmt.Errorf("failed to list users: %w", err)
	}
	defer rows.Close()

	for rows.Next() {
		var user User
		err := rows.Scan(
			&user.ID, &user.TenantID, &user.Email, &user.EmailVerified,
			&user.FirstName, &user.LastName, &user.AvatarURL, &user.Status,
			&user.Metadata, &user.CreatedAt, &user.UpdatedAt,
		)
		if err != nil {
			return nil, "", 0, fmt.Errorf("failed to scan user: %w", err)
		}
		users = append(users, &user)
	}

	// Determine next page token
	var nextPageToken string
	if len(users) > pageSize {
		nextPageToken = users[pageSize-1].ID
		users = users[:pageSize]
	}

	// Get total count
	countQuery := `SELECT COUNT(*) FROM users WHERE tenant_id = $1`
	var totalCount int
	r.pool.QueryRow(ctx, countQuery, tenantID).Scan(&totalCount)

	return users, nextPageToken, totalCount, nil
}

// GetRoles retrieves roles for a user
func (r *Repository) GetRoles(ctx context.Context, userID string) ([]string, error) {
	query := `
		SELECT r.name
		FROM roles r
		JOIN user_roles ur ON r.id = ur.role_id
		WHERE ur.user_id = $1
	`

	rows, err := r.pool.Query(ctx, query, userID)
	if err != nil {
		return nil, fmt.Errorf("failed to get user roles: %w", err)
	}
	defer rows.Close()

	var roles []string
	for rows.Next() {
		var role string
		if err := rows.Scan(&role); err != nil {
			return nil, fmt.Errorf("failed to scan role: %w", err)
		}
		roles = append(roles, role)
	}

	return roles, nil
}
