package repository

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/project/microservices/auth-service/internal/models"
	"github.com/redis/go-redis/v9"
)

var (
	ErrSessionNotFound = errors.New("session not found")
	ErrSessionRevoked  = errors.New("session has been revoked")
)

// SessionRepository handles session storage in Redis
type SessionRepository struct {
	redis *redis.Client
}

// NewSessionRepository creates a new session repository
func NewSessionRepository(redisClient *redis.Client) *SessionRepository {
	return &SessionRepository{redis: redisClient}
}

// Create creates a new session
func (r *SessionRepository) Create(ctx context.Context, userID, tenantID, ipAddress, userAgent string, ttl time.Duration) (*models.Session, error) {
	session := &models.Session{
		ID:           uuid.New().String(),
		UserID:       userID,
		TenantID:     tenantID,
		IPAddress:    ipAddress,
		UserAgent:    userAgent,
		CreatedAt:    time.Now().UTC(),
		LastActivity: time.Now().UTC(),
		ExpiresAt:    time.Now().UTC().Add(ttl),
		Revoked:      false,
	}

	data, err := json.Marshal(session)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal session: %w", err)
	}

	key := r.sessionKey(session.ID)
	if err := r.redis.Set(ctx, key, data, ttl).Err(); err != nil {
		return nil, fmt.Errorf("failed to store session: %w", err)
	}

	// Add to user's session set
	userSessionsKey := r.userSessionsKey(userID)
	if err := r.redis.SAdd(ctx, userSessionsKey, session.ID).Err(); err != nil {
		return nil, fmt.Errorf("failed to add session to user set: %w", err)
	}

	return session, nil
}

// Get retrieves a session by ID
func (r *SessionRepository) Get(ctx context.Context, sessionID string) (*models.Session, error) {
	key := r.sessionKey(sessionID)
	data, err := r.redis.Get(ctx, key).Bytes()
	if err != nil {
		if errors.Is(err, redis.Nil) {
			return nil, ErrSessionNotFound
		}
		return nil, fmt.Errorf("failed to get session: %w", err)
	}

	var session models.Session
	if err := json.Unmarshal(data, &session); err != nil {
		return nil, fmt.Errorf("failed to unmarshal session: %w", err)
	}

	if session.Revoked {
		return nil, ErrSessionRevoked
	}

	return &session, nil
}

// UpdateActivity updates the last activity time for a session
func (r *SessionRepository) UpdateActivity(ctx context.Context, sessionID string) error {
	session, err := r.Get(ctx, sessionID)
	if err != nil {
		return err
	}

	session.LastActivity = time.Now().UTC()

	data, err := json.Marshal(session)
	if err != nil {
		return fmt.Errorf("failed to marshal session: %w", err)
	}

	key := r.sessionKey(sessionID)
	ttl := time.Until(session.ExpiresAt)
	if ttl <= 0 {
		return ErrSessionNotFound
	}

	if err := r.redis.Set(ctx, key, data, ttl).Err(); err != nil {
		return fmt.Errorf("failed to update session: %w", err)
	}

	return nil
}

// Revoke revokes a session
func (r *SessionRepository) Revoke(ctx context.Context, sessionID string) error {
	session, err := r.Get(ctx, sessionID)
	if err != nil {
		if errors.Is(err, ErrSessionNotFound) {
			return nil // Already deleted
		}
		return err
	}

	session.Revoked = true

	data, err := json.Marshal(session)
	if err != nil {
		return fmt.Errorf("failed to marshal session: %w", err)
	}

	key := r.sessionKey(sessionID)
	// Keep revoked session for a short time for audit purposes
	if err := r.redis.Set(ctx, key, data, 24*time.Hour).Err(); err != nil {
		return fmt.Errorf("failed to revoke session: %w", err)
	}

	return nil
}

// RevokeAllUserSessions revokes all sessions for a user
func (r *SessionRepository) RevokeAllUserSessions(ctx context.Context, userID string) error {
	userSessionsKey := r.userSessionsKey(userID)
	sessionIDs, err := r.redis.SMembers(ctx, userSessionsKey).Result()
	if err != nil {
		return fmt.Errorf("failed to get user sessions: %w", err)
	}

	for _, sessionID := range sessionIDs {
		if err := r.Revoke(ctx, sessionID); err != nil {
			// Log error but continue revoking other sessions
			continue
		}
	}

	// Clear the user sessions set
	if err := r.redis.Del(ctx, userSessionsKey).Err(); err != nil {
		return fmt.Errorf("failed to delete user sessions set: %w", err)
	}

	return nil
}

// GetUserSessions retrieves all active sessions for a user
func (r *SessionRepository) GetUserSessions(ctx context.Context, userID string) ([]*models.Session, error) {
	userSessionsKey := r.userSessionsKey(userID)
	sessionIDs, err := r.redis.SMembers(ctx, userSessionsKey).Result()
	if err != nil {
		return nil, fmt.Errorf("failed to get user sessions: %w", err)
	}

	var sessions []*models.Session
	for _, sessionID := range sessionIDs {
		session, err := r.Get(ctx, sessionID)
		if err != nil {
			if errors.Is(err, ErrSessionNotFound) || errors.Is(err, ErrSessionRevoked) {
				// Clean up stale reference
				r.redis.SRem(ctx, userSessionsKey, sessionID)
				continue
			}
			continue
		}
		sessions = append(sessions, session)
	}

	return sessions, nil
}

func (r *SessionRepository) sessionKey(sessionID string) string {
	return fmt.Sprintf("session:%s", sessionID)
}

func (r *SessionRepository) userSessionsKey(userID string) string {
	return fmt.Sprintf("user_sessions:%s", userID)
}
