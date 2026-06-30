package notification

import (
	"bytes"
	"context"
	"fmt"
	"strings"
	"text/template"
	"time"

	"github.com/mlai/microservice-platform/pkg/events"
	"github.com/mlai/microservice-platform/pkg/logging"
	"go.uber.org/zap"
)

// Service handles notification business logic
type Service struct {
	repo      *Repository
	publisher *events.Publisher
	sender    EmailSender
	logger    *logging.Logger
}

// EmailSender interface for email sending
type EmailSender interface {
	Send(ctx context.Context, to, subject, body string) error
}

// NewService creates a new notification service
func NewService(repo *Repository, publisher *events.Publisher, sender EmailSender, logger *logging.Logger) *Service {
	return &Service{
		repo:      repo,
		publisher: publisher,
		sender:    sender,
		logger:    logger,
	}
}

// SendNotificationRequest contains notification parameters
type SendNotificationRequest struct {
	TenantID    string
	RecipientID string
	Channel     string
	TemplateID  string
	Subject     string
	Content     string
	Variables   map[string]string
	Priority    int
	ScheduledAt *time.Time
	Metadata    map[string]string
}

// SendNotification sends a notification
func (s *Service) SendNotification(ctx context.Context, req *SendNotificationRequest) (*Notification, error) {
	logger := s.logger.WithContext(ctx)

	// Check preferences
	prefs, err := s.repo.GetPreferences(ctx, req.TenantID, req.RecipientID)
	if err != nil {
		logger.Warn("failed to get preferences, using defaults", zap.Error(err))
	}

	if !s.isChannelEnabled(prefs, req.Channel) {
		return &Notification{Status: "opted_out"}, nil
	}

	// Render content
	subject := req.Subject
	content := req.Content

	if req.TemplateID != "" {
		tmpl, err := s.repo.GetTemplateByID(ctx, req.TenantID, req.TemplateID)
		if err != nil {
			return nil, err
		}

		subject, err = s.renderTemplate(tmpl.SubjectTemplate, req.Variables)
		if err != nil {
			return nil, fmt.Errorf("failed to render subject: %w", err)
		}

		content, err = s.renderTemplate(tmpl.ContentTemplate, req.Variables)
		if err != nil {
			return nil, fmt.Errorf("failed to render content: %w", err)
		}
	}

	// Create notification record
	notif := &Notification{
		TenantID:    req.TenantID,
		RecipientID: req.RecipientID,
		TemplateID:  req.TemplateID,
		Channel:     req.Channel,
		Subject:     subject,
		Content:     content,
		Status:      "pending",
		Priority:    req.Priority,
		Variables:   req.Variables,
		Metadata:    req.Metadata,
		ScheduledAt: req.ScheduledAt,
	}

	if err := s.repo.CreateNotification(ctx, notif); err != nil {
		return nil, err
	}

	// Send immediately if not scheduled
	if req.ScheduledAt == nil || req.ScheduledAt.Before(time.Now()) {
		go s.deliver(context.Background(), notif)
	}

	logger.Info("notification created",
		zap.String("notification_id", notif.ID),
		zap.String("channel", req.Channel),
	)

	return notif, nil
}

// deliver sends the notification through the appropriate channel
func (s *Service) deliver(ctx context.Context, notif *Notification) {
	var err error
	sentAt := time.Now().UTC()

	switch notif.Channel {
	case "email":
		// Get recipient email (would need to call user service)
		recipientEmail := notif.Metadata["email"]
		if recipientEmail == "" {
			recipientEmail = notif.RecipientID + "@example.com" // Fallback
		}
		err = s.sender.Send(ctx, recipientEmail, notif.Subject, notif.Content)

	case "sms", "push", "webhook":
		// TODO: Implement other channels
		err = nil

	default:
		err = fmt.Errorf("unsupported channel: %s", notif.Channel)
	}

	// Update status
	status := "sent"
	errorMsg := ""
	if err != nil {
		status = "failed"
		errorMsg = err.Error()
		s.logger.Error("notification delivery failed",
			zap.String("notification_id", notif.ID),
			zap.Error(err),
		)
	}

	s.repo.UpdateNotificationStatus(ctx, notif.ID, status, errorMsg, &sentAt, nil)

	// Publish event
	if s.publisher != nil {
		eventType := events.EventNotificationSent
		if status == "failed" {
			eventType = events.EventNotificationFailed
		}
		s.publisher.Publish(ctx, eventType, notif.TenantID, map[string]string{
			"notification_id": notif.ID,
			"channel":         notif.Channel,
			"recipient_id":    notif.RecipientID,
			"status":          status,
		}, nil)
	}
}

// GetNotification retrieves a notification
func (s *Service) GetNotification(ctx context.Context, tenantID, notificationID string) (*Notification, error) {
	return s.repo.GetNotificationByID(ctx, tenantID, notificationID)
}

// ListNotifications lists notifications for a recipient
func (s *Service) ListNotifications(ctx context.Context, tenantID, recipientID string, limit, offset int) ([]*Notification, int, error) {
	if limit <= 0 {
		limit = 20
	}
	if limit > 100 {
		limit = 100
	}
	return s.repo.ListNotifications(ctx, tenantID, recipientID, limit, offset)
}

// CreateTemplate creates a notification template
func (s *Service) CreateTemplate(ctx context.Context, tmpl *Template) error {
	return s.repo.CreateTemplate(ctx, tmpl)
}

// GetTemplate retrieves a template
func (s *Service) GetTemplate(ctx context.Context, tenantID, templateID string) (*Template, error) {
	return s.repo.GetTemplateByID(ctx, tenantID, templateID)
}

// ListTemplates lists templates
func (s *Service) ListTemplates(ctx context.Context, tenantID, channel string, activeOnly bool, limit, offset int) ([]*Template, int, error) {
	if limit <= 0 {
		limit = 20
	}
	if limit > 100 {
		limit = 100
	}
	return s.repo.ListTemplates(ctx, tenantID, channel, activeOnly, limit, offset)
}

// GetPreferences retrieves user preferences
func (s *Service) GetPreferences(ctx context.Context, tenantID, userID string) (*NotificationPreferences, error) {
	return s.repo.GetPreferences(ctx, tenantID, userID)
}

// UpdatePreferences updates user preferences
func (s *Service) UpdatePreferences(ctx context.Context, prefs *NotificationPreferences) error {
	return s.repo.UpdatePreferences(ctx, prefs)
}

// renderTemplate renders a template with variables
func (s *Service) renderTemplate(tmplStr string, vars map[string]string) (string, error) {
	t, err := template.New("notification").Parse(tmplStr)
	if err != nil {
		return "", err
	}

	var buf bytes.Buffer
	if err := t.Execute(&buf, vars); err != nil {
		return "", err
	}

	return buf.String(), nil
}

// isChannelEnabled checks if a channel is enabled for the user
func (s *Service) isChannelEnabled(prefs *NotificationPreferences, channel string) bool {
	if prefs == nil {
		return true
	}

	switch strings.ToLower(channel) {
	case "email":
		return prefs.EmailEnabled
	case "sms":
		return prefs.SMSEnabled
	case "push":
		return prefs.PushEnabled
	case "in_app":
		return prefs.InAppEnabled
	default:
		return true
	}
}

// SMTPSender implements EmailSender using SMTP
type SMTPSender struct {
	host     string
	port     int
	username string
	password string
	from     string
	fromName string
}

// NewSMTPSender creates a new SMTP sender
func NewSMTPSender(host string, port int, username, password, from, fromName string) *SMTPSender {
	return &SMTPSender{
		host:     host,
		port:     port,
		username: username,
		password: password,
		from:     from,
		fromName: fromName,
	}
}

// Send sends an email via SMTP
func (s *SMTPSender) Send(ctx context.Context, to, subject, body string) error {
	// Simple implementation - in production use proper SMTP library
	// This is a placeholder that logs the email
	fmt.Printf("EMAIL TO: %s\nSUBJECT: %s\nBODY: %s\n", to, subject, body)
	return nil
}
