package auth

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/redis/go-redis/v9"
)

// Session represents a user session
type Session struct {
	ID           string            `json:"id"`
	UserID       string            `json:"user_id"`
	TenantID     string            `json:"tenant_id"`
	DeviceInfo   map[string]string `json:"device_info"`
	IPAddress    string            `json:"ip_address"`
	CreatedAt    time.Time         `json:"created_at"`
	LastActiveAt time.Time         `json:"last_active_at"`
	ExpiresAt    time.Time         `json:"expires_at"`
}

// SessionRepository handles session storage in Redis
type SessionRepository struct {
	client *redis.Client
	ttl    time.Duration
}

// NewSessionRepository creates a new session repository
func NewSessionRepository(client *redis.Client, ttl time.Duration) *SessionRepository {
	return &SessionRepository{
		client: client,
		ttl:    ttl,
	}
}

// Create creates a new session
func (r *SessionRepository) Create(ctx context.Context, userID, tenantID, ipAddress string, deviceInfo map[string]string) (*Session, error) {
	session := &Session{
		ID:           uuid.New().String(),
		UserID:       userID,
		TenantID:     tenantID,
		DeviceInfo:   deviceInfo,
		IPAddress:    ipAddress,
		CreatedAt:    time.Now().UTC(),
		LastActiveAt: time.Now().UTC(),
		ExpiresAt:    time.Now().UTC().Add(r.ttl),
	}

	data, err := json.Marshal(session)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal session: %w", err)
	}

	// Store session by ID
	sessionKey := fmt.Sprintf("session:%s", session.ID)
	if err := r.client.Set(ctx, sessionKey, data, r.ttl).Err(); err != nil {
		return nil, fmt.Errorf("failed to store session: %w", err)
	}

	// Add to user's session set
	userSessionsKey := fmt.Sprintf("user_sessions:%s:%s", tenantID, userID)
	if err := r.client.SAdd(ctx, userSessionsKey, session.ID).Err(); err != nil {
		return nil, fmt.Errorf("failed to add to user sessions: %w", err)
	}

	return session, nil
}

// Get retrieves a session by ID
func (r *SessionRepository) Get(ctx context.Context, sessionID string) (*Session, error) {
	sessionKey := fmt.Sprintf("session:%s", sessionID)
	data, err := r.client.Get(ctx, sessionKey).Bytes()
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

// UpdateLastActive updates the last active timestamp
func (r *SessionRepository) UpdateLastActive(ctx context.Context, sessionID string) error {
	session, err := r.Get(ctx, sessionID)
	if err != nil {
		return err
	}

	session.LastActiveAt = time.Now().UTC()

	data, err := json.Marshal(session)
	if err != nil {
		return fmt.Errorf("failed to marshal session: %w", err)
	}

	// Calculate remaining TTL
	remaining := time.Until(session.ExpiresAt)
	if remaining <= 0 {
		remaining = r.ttl
		session.ExpiresAt = time.Now().UTC().Add(r.ttl)
	}

	sessionKey := fmt.Sprintf("session:%s", sessionID)
	if err := r.client.Set(ctx, sessionKey, data, remaining).Err(); err != nil {
		return fmt.Errorf("failed to update session: %w", err)
	}

	return nil
}

// Delete removes a session
func (r *SessionRepository) Delete(ctx context.Context, sessionID string) error {
	// Get session first to remove from user set
	session, err := r.Get(ctx, sessionID)
	if err != nil {
		if errors.Is(err, ErrSessionNotFound) {
			return nil // Already deleted
		}
		return err
	}

	// Delete session
	sessionKey := fmt.Sprintf("session:%s", sessionID)
	if err := r.client.Del(ctx, sessionKey).Err(); err != nil {
		return fmt.Errorf("failed to delete session: %w", err)
	}

	// Remove from user's session set
	userSessionsKey := fmt.Sprintf("user_sessions:%s:%s", session.TenantID, session.UserID)
	if err := r.client.SRem(ctx, userSessionsKey, sessionID).Err(); err != nil {
		return fmt.Errorf("failed to remove from user sessions: %w", err)
	}

	return nil
}

// DeleteAllForUser removes all sessions for a user
func (r *SessionRepository) DeleteAllForUser(ctx context.Context, tenantID, userID string) error {
	userSessionsKey := fmt.Sprintf("user_sessions:%s:%s", tenantID, userID)

	// Get all session IDs
	sessionIDs, err := r.client.SMembers(ctx, userSessionsKey).Result()
	if err != nil {
		return fmt.Errorf("failed to get user sessions: %w", err)
	}

	// Delete each session
	for _, sessionID := range sessionIDs {
		sessionKey := fmt.Sprintf("session:%s", sessionID)
		r.client.Del(ctx, sessionKey)
	}

	// Delete the set
	if err := r.client.Del(ctx, userSessionsKey).Err(); err != nil {
		return fmt.Errorf("failed to delete user sessions set: %w", err)
	}

	return nil
}

// GetAllForUser retrieves all sessions for a user
func (r *SessionRepository) GetAllForUser(ctx context.Context, tenantID, userID string) ([]*Session, error) {
	userSessionsKey := fmt.Sprintf("user_sessions:%s:%s", tenantID, userID)

	// Get all session IDs
	sessionIDs, err := r.client.SMembers(ctx, userSessionsKey).Result()
	if err != nil {
		return nil, fmt.Errorf("failed to get user sessions: %w", err)
	}

	var sessions []*Session
	for _, sessionID := range sessionIDs {
		session, err := r.Get(ctx, sessionID)
		if err != nil {
			if errors.Is(err, ErrSessionNotFound) {
				// Session expired, remove from set
				r.client.SRem(ctx, userSessionsKey, sessionID)
				continue
			}
			return nil, err
		}
		sessions = append(sessions, session)
	}

	return sessions, nil
}

// IsValid checks if a session is still valid
func (r *SessionRepository) IsValid(ctx context.Context, sessionID string) (bool, error) {
	session, err := r.Get(ctx, sessionID)
	if err != nil {
		if errors.Is(err, ErrSessionNotFound) {
			return false, nil
		}
		return false, err
	}

	return time.Now().Before(session.ExpiresAt), nil
}

// Errors
var (
	ErrSessionNotFound = errors.New("session not found")
	ErrSessionExpired  = errors.New("session expired")
)
