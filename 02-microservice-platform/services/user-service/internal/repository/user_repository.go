package repository

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/project/microservices/user-service/internal/models"
)

var (
	ErrUserNotFound     = errors.New("user not found")
	ErrUserExists       = errors.New("user already exists")
	ErrInvalidPageToken = errors.New("invalid page token")
)

// UserRepository handles database operations for users
type UserRepository struct {
	db *pgxpool.Pool
}

// NewUserRepository creates a new user repository
func NewUserRepository(db *pgxpool.Pool) *UserRepository {
	return &UserRepository{db: db}
}

// Create creates a new user
func (r *UserRepository) Create(ctx context.Context, input *models.CreateUserInput, passwordHash string) (*models.User, error) {
	id := uuid.New().String()
	now := time.Now().UTC()

	metadataJSON, err := json.Marshal(input.Metadata)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal metadata: %w", err)
	}

	query := `
		INSERT INTO users (id, tenant_id, email, password_hash, first_name, last_name, status, metadata, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
		RETURNING id, tenant_id, email, email_verified, first_name, last_name, avatar_url, status, metadata, created_at, updated_at
	`

	var user models.User
	var metadataBytes []byte

	err = r.db.QueryRow(ctx, query,
		id,
		input.TenantID,
		input.Email,
		passwordHash,
		input.FirstName,
		input.LastName,
		models.UserStatusActive,
		metadataJSON,
		now,
		now,
	).Scan(
		&user.ID,
		&user.TenantID,
		&user.Email,
		&user.EmailVerified,
		&user.FirstName,
		&user.LastName,
		&user.AvatarURL,
		&user.Status,
		&metadataBytes,
		&user.CreatedAt,
		&user.UpdatedAt,
	)

	if err != nil {
		if isDuplicateKeyError(err) {
			return nil, ErrUserExists
		}
		return nil, fmt.Errorf("failed to create user: %w", err)
	}

	if err := json.Unmarshal(metadataBytes, &user.Metadata); err != nil {
		user.Metadata = make(map[string]string)
	}

	// Get user roles
	user.Roles, _ = r.getUserRoles(ctx, user.ID)

	return &user, nil
}

// GetByID retrieves a user by ID
func (r *UserRepository) GetByID(ctx context.Context, userID, tenantID string) (*models.User, error) {
	query := `
		SELECT id, tenant_id, email, email_verified, first_name, last_name, avatar_url, status, metadata, created_at, updated_at
		FROM users
		WHERE id = $1 AND tenant_id = $2
	`

	var user models.User
	var metadataBytes []byte

	err := r.db.QueryRow(ctx, query, userID, tenantID).Scan(
		&user.ID,
		&user.TenantID,
		&user.Email,
		&user.EmailVerified,
		&user.FirstName,
		&user.LastName,
		&user.AvatarURL,
		&user.Status,
		&metadataBytes,
		&user.CreatedAt,
		&user.UpdatedAt,
	)

	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, ErrUserNotFound
		}
		return nil, fmt.Errorf("failed to get user: %w", err)
	}

	if err := json.Unmarshal(metadataBytes, &user.Metadata); err != nil {
		user.Metadata = make(map[string]string)
	}

	user.Roles, _ = r.getUserRoles(ctx, user.ID)

	return &user, nil
}

// GetByEmail retrieves a user by email (includes password hash for auth)
func (r *UserRepository) GetByEmail(ctx context.Context, email, tenantID string) (*models.User, error) {
	query := `
		SELECT id, tenant_id, email, email_verified, password_hash, first_name, last_name, avatar_url, status, metadata, created_at, updated_at
		FROM users
		WHERE email = $1 AND tenant_id = $2
	`

	var user models.User
	var metadataBytes []byte

	err := r.db.QueryRow(ctx, query, email, tenantID).Scan(
		&user.ID,
		&user.TenantID,
		&user.Email,
		&user.EmailVerified,
		&user.PasswordHash,
		&user.FirstName,
		&user.LastName,
		&user.AvatarURL,
		&user.Status,
		&metadataBytes,
		&user.CreatedAt,
		&user.UpdatedAt,
	)

	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, ErrUserNotFound
		}
		return nil, fmt.Errorf("failed to get user by email: %w", err)
	}

	if err := json.Unmarshal(metadataBytes, &user.Metadata); err != nil {
		user.Metadata = make(map[string]string)
	}

	user.Roles, _ = r.getUserRoles(ctx, user.ID)

	return &user, nil
}

// Update updates an existing user
func (r *UserRepository) Update(ctx context.Context, input *models.UpdateUserInput) (*models.User, error) {
	// Build dynamic update query
	query := "UPDATE users SET updated_at = $1"
	args := []interface{}{time.Now().UTC()}
	argIndex := 2

	if input.FirstName != nil {
		query += fmt.Sprintf(", first_name = $%d", argIndex)
		args = append(args, *input.FirstName)
		argIndex++
	}

	if input.LastName != nil {
		query += fmt.Sprintf(", last_name = $%d", argIndex)
		args = append(args, *input.LastName)
		argIndex++
	}

	if input.AvatarURL != nil {
		query += fmt.Sprintf(", avatar_url = $%d", argIndex)
		args = append(args, *input.AvatarURL)
		argIndex++
	}

	if input.Status != nil {
		query += fmt.Sprintf(", status = $%d", argIndex)
		args = append(args, *input.Status)
		argIndex++
	}

	if len(input.Metadata) > 0 {
		metadataJSON, err := json.Marshal(input.Metadata)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal metadata: %w", err)
		}
		query += fmt.Sprintf(", metadata = metadata || $%d", argIndex)
		args = append(args, metadataJSON)
		argIndex++
	}

	query += fmt.Sprintf(" WHERE id = $%d AND tenant_id = $%d", argIndex, argIndex+1)
	args = append(args, input.UserID, input.TenantID)

	query += " RETURNING id, tenant_id, email, email_verified, first_name, last_name, avatar_url, status, metadata, created_at, updated_at"

	var user models.User
	var metadataBytes []byte

	err := r.db.QueryRow(ctx, query, args...).Scan(
		&user.ID,
		&user.TenantID,
		&user.Email,
		&user.EmailVerified,
		&user.FirstName,
		&user.LastName,
		&user.AvatarURL,
		&user.Status,
		&metadataBytes,
		&user.CreatedAt,
		&user.UpdatedAt,
	)

	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, ErrUserNotFound
		}
		return nil, fmt.Errorf("failed to update user: %w", err)
	}

	if err := json.Unmarshal(metadataBytes, &user.Metadata); err != nil {
		user.Metadata = make(map[string]string)
	}

	user.Roles, _ = r.getUserRoles(ctx, user.ID)

	return &user, nil
}

// Delete deletes a user
func (r *UserRepository) Delete(ctx context.Context, userID, tenantID string) error {
	query := `DELETE FROM users WHERE id = $1 AND tenant_id = $2`

	result, err := r.db.Exec(ctx, query, userID, tenantID)
	if err != nil {
		return fmt.Errorf("failed to delete user: %w", err)
	}

	if result.RowsAffected() == 0 {
		return ErrUserNotFound
	}

	return nil
}

// List lists users with pagination
func (r *UserRepository) List(ctx context.Context, input *models.ListUsersInput) (*models.ListUsersOutput, error) {
	// Parse page token
	var offset int
	if input.PageToken != "" {
		tokenBytes, err := base64.StdEncoding.DecodeString(input.PageToken)
		if err != nil {
			return nil, ErrInvalidPageToken
		}
		var tokenData struct {
			Offset int `json:"offset"`
		}
		if err := json.Unmarshal(tokenBytes, &tokenData); err != nil {
			return nil, ErrInvalidPageToken
		}
		offset = tokenData.Offset
	}

	pageSize := input.PageSize
	if pageSize <= 0 {
		pageSize = 20
	}
	if pageSize > 100 {
		pageSize = 100
	}

	// Build query
	query := `
		SELECT id, tenant_id, email, email_verified, first_name, last_name, avatar_url, status, metadata, created_at, updated_at
		FROM users
		WHERE tenant_id = $1
	`
	countQuery := `SELECT COUNT(*) FROM users WHERE tenant_id = $1`
	args := []interface{}{input.TenantID}
	countArgs := []interface{}{input.TenantID}
	argIndex := 2

	if input.StatusFilter != "" {
		query += fmt.Sprintf(" AND status = $%d", argIndex)
		countQuery += fmt.Sprintf(" AND status = $%d", argIndex)
		args = append(args, input.StatusFilter)
		countArgs = append(countArgs, input.StatusFilter)
		argIndex++
	}

	if input.SearchQuery != "" {
		query += fmt.Sprintf(" AND (email ILIKE $%d OR first_name ILIKE $%d OR last_name ILIKE $%d)", argIndex, argIndex, argIndex)
		countQuery += fmt.Sprintf(" AND (email ILIKE $%d OR first_name ILIKE $%d OR last_name ILIKE $%d)", argIndex, argIndex, argIndex)
		searchPattern := "%" + input.SearchQuery + "%"
		args = append(args, searchPattern)
		countArgs = append(countArgs, searchPattern)
		argIndex++
	}

	query += " ORDER BY created_at DESC"
	query += fmt.Sprintf(" LIMIT $%d OFFSET $%d", argIndex, argIndex+1)
	args = append(args, pageSize+1, offset) // Fetch one extra to determine if there's a next page

	// Get total count
	var totalCount int
	err := r.db.QueryRow(ctx, countQuery, countArgs...).Scan(&totalCount)
	if err != nil {
		return nil, fmt.Errorf("failed to count users: %w", err)
	}

	// Get users
	rows, err := r.db.Query(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to list users: %w", err)
	}
	defer rows.Close()

	var users []*models.User
	for rows.Next() {
		var user models.User
		var metadataBytes []byte

		err := rows.Scan(
			&user.ID,
			&user.TenantID,
			&user.Email,
			&user.EmailVerified,
			&user.FirstName,
			&user.LastName,
			&user.AvatarURL,
			&user.Status,
			&metadataBytes,
			&user.CreatedAt,
			&user.UpdatedAt,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan user: %w", err)
		}

		if err := json.Unmarshal(metadataBytes, &user.Metadata); err != nil {
			user.Metadata = make(map[string]string)
		}

		user.Roles, _ = r.getUserRoles(ctx, user.ID)
		users = append(users, &user)
	}

	// Determine next page token
	var nextPageToken string
	if len(users) > pageSize {
		users = users[:pageSize]
		tokenData := struct {
			Offset int `json:"offset"`
		}{
			Offset: offset + pageSize,
		}
		tokenBytes, _ := json.Marshal(tokenData)
		nextPageToken = base64.StdEncoding.EncodeToString(tokenBytes)
	}

	return &models.ListUsersOutput{
		Users:         users,
		NextPageToken: nextPageToken,
		TotalCount:    totalCount,
	}, nil
}

// getUserRoles retrieves roles for a user
func (r *UserRepository) getUserRoles(ctx context.Context, userID string) ([]string, error) {
	query := `
		SELECT r.name
		FROM roles r
		JOIN user_roles ur ON ur.role_id = r.id
		WHERE ur.user_id = $1
	`

	rows, err := r.db.Query(ctx, query, userID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var roles []string
	for rows.Next() {
		var role string
		if err := rows.Scan(&role); err != nil {
			return nil, err
		}
		roles = append(roles, role)
	}

	return roles, nil
}

// isDuplicateKeyError checks if the error is a duplicate key violation
func isDuplicateKeyError(err error) bool {
	return err != nil && (err.Error() == "ERROR: duplicate key value violates unique constraint \"users_tenant_id_email_key\" (SQLSTATE 23505)" ||
		contains(err.Error(), "duplicate key") ||
		contains(err.Error(), "23505"))
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
