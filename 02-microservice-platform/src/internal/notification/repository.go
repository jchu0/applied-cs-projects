package notification

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
	ErrTemplateNotFound     = errors.New("template not found")
	ErrNotificationNotFound = errors.New("notification not found")
)

// Template represents a notification template
type Template struct {
	ID              string
	TenantID        string
	Name            string
	Description     string
	Channel         string
	SubjectTemplate string
	ContentTemplate string
	Variables       []string
	IsActive        bool
	CreatedAt       time.Time
	UpdatedAt       time.Time
}

// Notification represents a notification
type Notification struct {
	ID           string
	TenantID     string
	RecipientID  string
	TemplateID   string
	Channel      string
	Subject      string
	Content      string
	Status       string
	Priority     int
	Variables    map[string]string
	Metadata     map[string]string
	ScheduledAt  *time.Time
	SentAt       *time.Time
	DeliveredAt  *time.Time
	ErrorMessage string
	CreatedAt    time.Time
}

// NotificationPreferences represents user notification preferences
type NotificationPreferences struct {
	ID                     string
	TenantID               string
	UserID                 string
	EmailEnabled           bool
	SMSEnabled             bool
	PushEnabled            bool
	InAppEnabled           bool
	UnsubscribedCategories []string
	ChannelPreferences     map[string]bool
	UpdatedAt              time.Time
}

// Repository handles notification data access
type Repository struct {
	pool *pgxpool.Pool
}

// NewRepository creates a new notification repository
func NewRepository(pool *pgxpool.Pool) *Repository {
	return &Repository{pool: pool}
}

// CreateTemplate creates a new template
func (r *Repository) CreateTemplate(ctx context.Context, tmpl *Template) error {
	tmpl.ID = uuid.New().String()
	tmpl.CreatedAt = time.Now().UTC()
	tmpl.UpdatedAt = tmpl.CreatedAt

	query := `
		INSERT INTO templates (
			id, tenant_id, name, description, channel, subject_template,
			content_template, variables, is_active, created_at, updated_at
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
	`

	_, err := r.pool.Exec(ctx, query,
		tmpl.ID, tmpl.TenantID, tmpl.Name, tmpl.Description, tmpl.Channel,
		tmpl.SubjectTemplate, tmpl.ContentTemplate, tmpl.Variables,
		tmpl.IsActive, tmpl.CreatedAt, tmpl.UpdatedAt,
	)
	if err != nil {
		return fmt.Errorf("failed to create template: %w", err)
	}

	return nil
}

// GetTemplateByID retrieves a template by ID
func (r *Repository) GetTemplateByID(ctx context.Context, tenantID, templateID string) (*Template, error) {
	query := `
		SELECT id, tenant_id, name, description, channel, subject_template,
			   content_template, variables, is_active, created_at, updated_at
		FROM templates
		WHERE id = $1 AND tenant_id = $2
	`

	tmpl := &Template{}
	err := r.pool.QueryRow(ctx, query, templateID, tenantID).Scan(
		&tmpl.ID, &tmpl.TenantID, &tmpl.Name, &tmpl.Description, &tmpl.Channel,
		&tmpl.SubjectTemplate, &tmpl.ContentTemplate, &tmpl.Variables,
		&tmpl.IsActive, &tmpl.CreatedAt, &tmpl.UpdatedAt,
	)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, ErrTemplateNotFound
		}
		return nil, fmt.Errorf("failed to get template: %w", err)
	}

	return tmpl, nil
}

// GetTemplateByName retrieves a template by name and channel
func (r *Repository) GetTemplateByName(ctx context.Context, tenantID, name, channel string) (*Template, error) {
	query := `
		SELECT id, tenant_id, name, description, channel, subject_template,
			   content_template, variables, is_active, created_at, updated_at
		FROM templates
		WHERE tenant_id = $1 AND name = $2 AND channel = $3 AND is_active = true
	`

	tmpl := &Template{}
	err := r.pool.QueryRow(ctx, query, tenantID, name, channel).Scan(
		&tmpl.ID, &tmpl.TenantID, &tmpl.Name, &tmpl.Description, &tmpl.Channel,
		&tmpl.SubjectTemplate, &tmpl.ContentTemplate, &tmpl.Variables,
		&tmpl.IsActive, &tmpl.CreatedAt, &tmpl.UpdatedAt,
	)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, ErrTemplateNotFound
		}
		return nil, fmt.Errorf("failed to get template: %w", err)
	}

	return tmpl, nil
}

// ListTemplates lists templates
func (r *Repository) ListTemplates(ctx context.Context, tenantID string, channel string, activeOnly bool, limit, offset int) ([]*Template, int, error) {
	// Build query
	baseQuery := `FROM templates WHERE tenant_id = $1`
	args := []interface{}{tenantID}
	argIdx := 2

	if channel != "" {
		baseQuery += fmt.Sprintf(" AND channel = $%d", argIdx)
		args = append(args, channel)
		argIdx++
	}

	if activeOnly {
		baseQuery += " AND is_active = true"
	}

	// Count
	var total int
	err := r.pool.QueryRow(ctx, "SELECT COUNT(*) "+baseQuery, args...).Scan(&total)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to count templates: %w", err)
	}

	// Get templates
	query := `
		SELECT id, tenant_id, name, description, channel, subject_template,
			   content_template, variables, is_active, created_at, updated_at
	` + baseQuery + fmt.Sprintf(" ORDER BY name LIMIT $%d OFFSET $%d", argIdx, argIdx+1)
	args = append(args, limit, offset)

	rows, err := r.pool.Query(ctx, query, args...)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to list templates: %w", err)
	}
	defer rows.Close()

	var templates []*Template
	for rows.Next() {
		tmpl := &Template{}
		err := rows.Scan(
			&tmpl.ID, &tmpl.TenantID, &tmpl.Name, &tmpl.Description, &tmpl.Channel,
			&tmpl.SubjectTemplate, &tmpl.ContentTemplate, &tmpl.Variables,
			&tmpl.IsActive, &tmpl.CreatedAt, &tmpl.UpdatedAt,
		)
		if err != nil {
			return nil, 0, fmt.Errorf("failed to scan template: %w", err)
		}
		templates = append(templates, tmpl)
	}

	return templates, total, nil
}

// CreateNotification creates a new notification
func (r *Repository) CreateNotification(ctx context.Context, notif *Notification) error {
	notif.ID = uuid.New().String()
	notif.CreatedAt = time.Now().UTC()

	query := `
		INSERT INTO notifications (
			id, tenant_id, recipient_id, template_id, channel, subject, content,
			status, priority, variables, metadata, scheduled_at, sent_at,
			delivered_at, error_message, created_at
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
	`

	_, err := r.pool.Exec(ctx, query,
		notif.ID, notif.TenantID, notif.RecipientID, notif.TemplateID, notif.Channel,
		notif.Subject, notif.Content, notif.Status, notif.Priority,
		notif.Variables, notif.Metadata, notif.ScheduledAt, notif.SentAt,
		notif.DeliveredAt, notif.ErrorMessage, notif.CreatedAt,
	)
	if err != nil {
		return fmt.Errorf("failed to create notification: %w", err)
	}

	return nil
}

// GetNotificationByID retrieves a notification by ID
func (r *Repository) GetNotificationByID(ctx context.Context, tenantID, notifID string) (*Notification, error) {
	query := `
		SELECT id, tenant_id, recipient_id, template_id, channel, subject, content,
			   status, priority, variables, metadata, scheduled_at, sent_at,
			   delivered_at, error_message, created_at
		FROM notifications
		WHERE id = $1 AND tenant_id = $2
	`

	notif := &Notification{}
	err := r.pool.QueryRow(ctx, query, notifID, tenantID).Scan(
		&notif.ID, &notif.TenantID, &notif.RecipientID, &notif.TemplateID, &notif.Channel,
		&notif.Subject, &notif.Content, &notif.Status, &notif.Priority,
		&notif.Variables, &notif.Metadata, &notif.ScheduledAt, &notif.SentAt,
		&notif.DeliveredAt, &notif.ErrorMessage, &notif.CreatedAt,
	)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, ErrNotificationNotFound
		}
		return nil, fmt.Errorf("failed to get notification: %w", err)
	}

	return notif, nil
}

// UpdateNotificationStatus updates notification status
func (r *Repository) UpdateNotificationStatus(ctx context.Context, notifID, status, errorMsg string, sentAt, deliveredAt *time.Time) error {
	query := `
		UPDATE notifications SET
			status = $2, error_message = $3, sent_at = $4, delivered_at = $5
		WHERE id = $1
	`

	result, err := r.pool.Exec(ctx, query, notifID, status, errorMsg, sentAt, deliveredAt)
	if err != nil {
		return fmt.Errorf("failed to update notification status: %w", err)
	}

	if result.RowsAffected() == 0 {
		return ErrNotificationNotFound
	}

	return nil
}

// ListNotifications lists notifications for a recipient
func (r *Repository) ListNotifications(ctx context.Context, tenantID, recipientID string, limit, offset int) ([]*Notification, int, error) {
	// Count
	var total int
	countQuery := `SELECT COUNT(*) FROM notifications WHERE tenant_id = $1 AND recipient_id = $2`
	err := r.pool.QueryRow(ctx, countQuery, tenantID, recipientID).Scan(&total)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to count notifications: %w", err)
	}

	// Get notifications
	query := `
		SELECT id, tenant_id, recipient_id, template_id, channel, subject, content,
			   status, priority, variables, metadata, scheduled_at, sent_at,
			   delivered_at, error_message, created_at
		FROM notifications
		WHERE tenant_id = $1 AND recipient_id = $2
		ORDER BY created_at DESC
		LIMIT $3 OFFSET $4
	`

	rows, err := r.pool.Query(ctx, query, tenantID, recipientID, limit, offset)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to list notifications: %w", err)
	}
	defer rows.Close()

	var notifications []*Notification
	for rows.Next() {
		notif := &Notification{}
		err := rows.Scan(
			&notif.ID, &notif.TenantID, &notif.RecipientID, &notif.TemplateID, &notif.Channel,
			&notif.Subject, &notif.Content, &notif.Status, &notif.Priority,
			&notif.Variables, &notif.Metadata, &notif.ScheduledAt, &notif.SentAt,
			&notif.DeliveredAt, &notif.ErrorMessage, &notif.CreatedAt,
		)
		if err != nil {
			return nil, 0, fmt.Errorf("failed to scan notification: %w", err)
		}
		notifications = append(notifications, notif)
	}

	return notifications, total, nil
}

// GetPreferences retrieves user notification preferences
func (r *Repository) GetPreferences(ctx context.Context, tenantID, userID string) (*NotificationPreferences, error) {
	query := `
		SELECT id, tenant_id, user_id, email_enabled, sms_enabled, push_enabled,
			   in_app_enabled, unsubscribed_categories, channel_preferences, updated_at
		FROM notification_preferences
		WHERE tenant_id = $1 AND user_id = $2
	`

	prefs := &NotificationPreferences{}
	err := r.pool.QueryRow(ctx, query, tenantID, userID).Scan(
		&prefs.ID, &prefs.TenantID, &prefs.UserID, &prefs.EmailEnabled,
		&prefs.SMSEnabled, &prefs.PushEnabled, &prefs.InAppEnabled,
		&prefs.UnsubscribedCategories, &prefs.ChannelPreferences, &prefs.UpdatedAt,
	)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			// Return default preferences
			return &NotificationPreferences{
				TenantID:     tenantID,
				UserID:       userID,
				EmailEnabled: true,
				SMSEnabled:   true,
				PushEnabled:  true,
				InAppEnabled: true,
			}, nil
		}
		return nil, fmt.Errorf("failed to get preferences: %w", err)
	}

	return prefs, nil
}

// UpdatePreferences updates user notification preferences
func (r *Repository) UpdatePreferences(ctx context.Context, prefs *NotificationPreferences) error {
	prefs.UpdatedAt = time.Now().UTC()

	query := `
		INSERT INTO notification_preferences (
			id, tenant_id, user_id, email_enabled, sms_enabled, push_enabled,
			in_app_enabled, unsubscribed_categories, channel_preferences, updated_at
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
		ON CONFLICT (tenant_id, user_id) DO UPDATE SET
			email_enabled = $4, sms_enabled = $5, push_enabled = $6,
			in_app_enabled = $7, unsubscribed_categories = $8,
			channel_preferences = $9, updated_at = $10
	`

	if prefs.ID == "" {
		prefs.ID = uuid.New().String()
	}

	_, err := r.pool.Exec(ctx, query,
		prefs.ID, prefs.TenantID, prefs.UserID, prefs.EmailEnabled,
		prefs.SMSEnabled, prefs.PushEnabled, prefs.InAppEnabled,
		prefs.UnsubscribedCategories, prefs.ChannelPreferences, prefs.UpdatedAt,
	)
	if err != nil {
		return fmt.Errorf("failed to update preferences: %w", err)
	}

	return nil
}
