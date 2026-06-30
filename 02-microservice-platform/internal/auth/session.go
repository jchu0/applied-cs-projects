package auth

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"
)

// Session represents a user session
type Session struct {
	ID           string    `json:"id"`
	UserID       string    `json:"user_id"`
	TenantID     string    `json:"tenant_id"`
	IPAddress    string    `json:"ip_address"`
	UserAgent    string    `json:"user_agent"`
	CreatedAt    time.Time `json:"created_at"`
	ExpiresAt    time.Time `json:"expires_at"`
	LastActivity time.Time `json:"last_activity"`
}

// SessionRepository handles session storage in Redis
type SessionRepository struct {
	client *redis.Client
	prefix string
}

// NewSessionRepository creates a new session repository
func NewSessionRepository(redisURL string) (*SessionRepository, error) {
	opt, err := redis.ParseURL(redisURL)
	if err != nil {
		return nil, fmt.Errorf("failed to parse redis URL: %w", err)
	}

	client := redis.NewClient(opt)

	// Test connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := client.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("failed to connect to redis: %w", err)
	}

	return &SessionRepository{
		client: client,
		prefix: "session:",
	}, nil
}

// Create stores a new session
func (r *SessionRepository) Create(ctx context.Context, session *Session) error {
	data, err := json.Marshal(session)
	if err != nil {
		return fmt.Errorf("failed to marshal session: %w", err)
	}

	key := r.prefix + session.ID
	ttl := time.Until(session.ExpiresAt)

	if err := r.client.Set(ctx, key, data, ttl).Err(); err != nil {
		return fmt.Errorf("failed to store session: %w", err)
	}

	// Also index by user ID for listing sessions
	userKey := r.prefix + "user:" + session.UserID
	r.client.SAdd(ctx, userKey, session.ID)
	r.client.Expire(ctx, userKey, ttl)

	return nil
}

// GetByID retrieves a session by ID
func (r *SessionRepository) GetByID(ctx context.Context, id string) (*Session, error) {
	key := r.prefix + id

	data, err := r.client.Get(ctx, key).Bytes()
	if err != nil {
		if errors.Is(err, redis.Nil) {
			return nil, ErrSessionNotFound
		}
		return nil, fmt.Errorf("failed to get session: %w", err)
	}

	var session Session
	if err := json.Unmarshal(data, &session); err != nil {
		return nil, fmt.Errorf("failed to unmarshal session: %w", err)
	}

	return &session, nil
}

// Update updates a session
func (r *SessionRepository) Update(ctx context.Context, session *Session) error {
	data, err := json.Marshal(session)
	if err != nil {
		return fmt.Errorf("failed to marshal session: %w", err)
	}

	key := r.prefix + session.ID
	ttl := time.Until(session.ExpiresAt)

	if err := r.client.Set(ctx, key, data, ttl).Err(); err != nil {
		return fmt.Errorf("failed to update session: %w", err)
	}

	return nil
}

// Delete removes a session
func (r *SessionRepository) Delete(ctx context.Context, id string) error {
	// Get session first to remove from user index
	session, err := r.GetByID(ctx, id)
	if err != nil {
		if errors.Is(err, ErrSessionNotFound) {
			return nil // Already deleted
		}
		return err
	}

	key := r.prefix + id
	if err := r.client.Del(ctx, key).Err(); err != nil {
		return fmt.Errorf("failed to delete session: %w", err)
	}

	// Remove from user index
	userKey := r.prefix + "user:" + session.UserID
	r.client.SRem(ctx, userKey, id)

	return nil
}

// GetUserSessions retrieves all sessions for a user
func (r *SessionRepository) GetUserSessions(ctx context.Context, userID string) ([]*Session, error) {
	userKey := r.prefix + "user:" + userID

	sessionIDs, err := r.client.SMembers(ctx, userKey).Result()
	if err != nil {
		return nil, fmt.Errorf("failed to get user sessions: %w", err)
	}

	var sessions []*Session
	for _, id := range sessionIDs {
		session, err := r.GetByID(ctx, id)
		if err != nil {
			continue // Skip invalid sessions
		}
		sessions = append(sessions, session)
	}

	return sessions, nil
}

// DeleteUserSessions removes all sessions for a user
func (r *SessionRepository) DeleteUserSessions(ctx context.Context, userID string) error {
	sessions, err := r.GetUserSessions(ctx, userID)
	if err != nil {
		return err
	}

	for _, session := range sessions {
		if err := r.Delete(ctx, session.ID); err != nil {
			return err
		}
	}

	return nil
}

// Close closes the Redis connection
func (r *SessionRepository) Close() error {
	return r.client.Close()
}

// HealthCheck verifies Redis connectivity
func (r *SessionRepository) HealthCheck(ctx context.Context) error {
	return r.client.Ping(ctx).Err()
}
