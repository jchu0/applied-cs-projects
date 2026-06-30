package events

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/nats-io/nats.go"
)

// Event represents a domain event
type Event struct {
	ID        string                 `json:"event_id"`
	Type      string                 `json:"event_type"`
	TenantID  string                 `json:"tenant_id"`
	Timestamp time.Time              `json:"timestamp"`
	Version   string                 `json:"version"`
	Data      interface{}            `json:"data"`
	Metadata  map[string]string      `json:"metadata"`
}

// Publisher publishes events to NATS
type Publisher struct {
	conn    *nats.Conn
	js      nats.JetStreamContext
	service string
}

// NewPublisher creates a new event publisher
func NewPublisher(url, service string) (*Publisher, error) {
	conn, err := nats.Connect(url,
		nats.RetryOnFailedConnect(true),
		nats.MaxReconnects(10),
		nats.ReconnectWait(time.Second),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to NATS: %w", err)
	}

	js, err := conn.JetStream()
	if err != nil {
		conn.Close()
		return nil, fmt.Errorf("failed to get JetStream context: %w", err)
	}

	// Create streams for different domains
	streams := []struct {
		name     string
		subjects []string
	}{
		{"USERS", []string{"user.>"}},
		{"AUTH", []string{"auth.>"}},
		{"BILLING", []string{"billing.>"}},
		{"NOTIFICATIONS", []string{"notification.>"}},
	}

	for _, s := range streams {
		_, err := js.AddStream(&nats.StreamConfig{
			Name:      s.name,
			Subjects:  s.subjects,
			Retention: nats.LimitsPolicy,
			MaxAge:    7 * 24 * time.Hour,
			Storage:   nats.FileStorage,
			Replicas:  1,
		})
		if err != nil && err != nats.ErrStreamNameAlreadyInUse {
			// Ignore if stream already exists
		}
	}

	return &Publisher{
		conn:    conn,
		js:      js,
		service: service,
	}, nil
}

// Publish publishes an event
func (p *Publisher) Publish(ctx context.Context, eventType string, tenantID string, data interface{}, metadata map[string]string) error {
	if metadata == nil {
		metadata = make(map[string]string)
	}
	metadata["source_service"] = p.service

	event := Event{
		ID:        uuid.New().String(),
		Type:      eventType,
		TenantID:  tenantID,
		Timestamp: time.Now().UTC(),
		Version:   "1.0",
		Data:      data,
		Metadata:  metadata,
	}

	payload, err := json.Marshal(event)
	if err != nil {
		return fmt.Errorf("failed to marshal event: %w", err)
	}

	_, err = p.js.Publish(eventType, payload)
	if err != nil {
		return fmt.Errorf("failed to publish event: %w", err)
	}

	return nil
}

// Close closes the connection
func (p *Publisher) Close() {
	if p.conn != nil {
		p.conn.Close()
	}
}

// Event types
const (
	// User events
	EventUserCreated  = "user.created"
	EventUserUpdated  = "user.updated"
	EventUserDeleted  = "user.deleted"
	EventRoleAssigned = "user.role.assigned"
	EventRoleRevoked  = "user.role.revoked"

	// Auth events
	EventUserLoggedIn    = "auth.login"
	EventUserLoggedOut   = "auth.logout"
	EventTokenRefreshed  = "auth.token.refreshed"
	EventPasswordChanged = "auth.password.changed"
	EventSessionCreated  = "auth.session.created"
	EventSessionExpired  = "auth.session.expired"

	// Billing events
	EventCustomerCreated      = "billing.customer.created"
	EventSubscriptionCreated  = "billing.subscription.created"
	EventSubscriptionUpdated  = "billing.subscription.updated"
	EventSubscriptionCanceled = "billing.subscription.canceled"
	EventPaymentSucceeded     = "billing.payment.succeeded"
	EventPaymentFailed        = "billing.payment.failed"
	EventInvoiceCreated       = "billing.invoice.created"
	EventInvoicePaid          = "billing.invoice.paid"

	// Notification events
	EventNotificationSent      = "notification.sent"
	EventNotificationDelivered = "notification.delivered"
	EventNotificationFailed    = "notification.failed"
	EventNotificationOpened    = "notification.opened"
)
