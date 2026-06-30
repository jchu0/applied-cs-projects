package events

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"github.com/google/uuid"
	"github.com/nats-io/nats.go"
)

// Event types
const (
	// User events
	EventUserCreated  = "user.created"
	EventUserUpdated  = "user.updated"
	EventUserDeleted  = "user.deleted"

	// Auth events
	EventUserLoggedIn    = "auth.login"
	EventUserLoggedOut   = "auth.logout"
	EventTokenRefreshed  = "auth.token_refreshed"
	EventUserRegistered  = "auth.registered"

	// Billing events
	EventSubscriptionCreated  = "billing.subscription.created"
	EventSubscriptionUpdated  = "billing.subscription.updated"
	EventSubscriptionCanceled = "billing.subscription.canceled"
	EventPaymentSucceeded     = "billing.payment.succeeded"
	EventPaymentFailed        = "billing.payment.failed"
	EventInvoiceCreated       = "billing.invoice.created"
	EventInvoicePaid          = "billing.invoice.paid"
)

// Event represents a domain event
type Event struct {
	ID            string                 `json:"event_id"`
	Type          string                 `json:"event_type"`
	TenantID      string                 `json:"tenant_id"`
	Timestamp     time.Time              `json:"timestamp"`
	Version       string                 `json:"version"`
	Data          map[string]interface{} `json:"data"`
	Metadata      map[string]string      `json:"metadata"`
	SourceService string                 `json:"source_service"`
	CorrelationID string                 `json:"correlation_id"`
}

// NewEvent creates a new event
func NewEvent(eventType, tenantID, sourceService string, data map[string]interface{}) *Event {
	return &Event{
		ID:            uuid.New().String(),
		Type:          eventType,
		TenantID:      tenantID,
		Timestamp:     time.Now().UTC(),
		Version:       "1.0",
		Data:          data,
		Metadata:      make(map[string]string),
		SourceService: sourceService,
		CorrelationID: uuid.New().String(),
	}
}

// EventBus handles event publishing and subscribing
type EventBus struct {
	conn          *nats.Conn
	js            nats.JetStreamContext
	serviceName   string
	subscriptions []*nats.Subscription
}

// NewEventBus creates a new event bus connected to NATS
func NewEventBus(natsURL, serviceName string) (*EventBus, error) {
	// Connect to NATS
	nc, err := nats.Connect(natsURL,
		nats.RetryOnFailedConnect(true),
		nats.MaxReconnects(10),
		nats.ReconnectWait(time.Second),
		nats.DisconnectErrHandler(func(nc *nats.Conn, err error) {
			log.Printf("NATS disconnected: %v", err)
		}),
		nats.ReconnectHandler(func(nc *nats.Conn) {
			log.Printf("NATS reconnected to %s", nc.ConnectedUrl())
		}),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to NATS: %w", err)
	}

	// Create JetStream context
	js, err := nc.JetStream()
	if err != nil {
		nc.Close()
		return nil, fmt.Errorf("failed to create JetStream context: %w", err)
	}

	// Ensure stream exists
	if err := ensureStream(js); err != nil {
		nc.Close()
		return nil, fmt.Errorf("failed to ensure stream: %w", err)
	}

	return &EventBus{
		conn:        nc,
		js:          js,
		serviceName: serviceName,
	}, nil
}

// ensureStream creates the events stream if it doesn't exist
func ensureStream(js nats.JetStreamContext) error {
	streamName := "EVENTS"

	// Check if stream exists
	_, err := js.StreamInfo(streamName)
	if err == nil {
		return nil // Stream exists
	}

	// Create stream
	_, err = js.AddStream(&nats.StreamConfig{
		Name:        streamName,
		Description: "Microservices domain events",
		Subjects:    []string{"events.>"},
		Retention:   nats.LimitsPolicy,
		MaxAge:      7 * 24 * time.Hour, // Keep events for 7 days
		Storage:     nats.FileStorage,
		Replicas:    1,
		Discard:     nats.DiscardOld,
	})
	if err != nil {
		return fmt.Errorf("failed to create stream: %w", err)
	}

	log.Printf("Created JetStream stream: %s", streamName)
	return nil
}

// Publish publishes an event to NATS JetStream
func (eb *EventBus) Publish(ctx context.Context, event *Event) error {
	// Set source service
	event.SourceService = eb.serviceName

	// Serialize event
	data, err := json.Marshal(event)
	if err != nil {
		return fmt.Errorf("failed to marshal event: %w", err)
	}

	// Publish to JetStream
	subject := fmt.Sprintf("events.%s", event.Type)
	_, err = eb.js.Publish(subject, data)
	if err != nil {
		return fmt.Errorf("failed to publish event: %w", err)
	}

	log.Printf("Published event: %s (id=%s, tenant=%s)", event.Type, event.ID, event.TenantID)
	return nil
}

// Subscribe subscribes to events matching the given pattern
func (eb *EventBus) Subscribe(pattern string, handler func(*Event) error) error {
	subject := fmt.Sprintf("events.%s", pattern)

	// Create durable consumer name from service name and pattern
	consumerName := fmt.Sprintf("%s_%s", eb.serviceName, sanitizeConsumerName(pattern))

	sub, err := eb.js.Subscribe(subject, func(msg *nats.Msg) {
		var event Event
		if err := json.Unmarshal(msg.Data, &event); err != nil {
			log.Printf("Failed to unmarshal event: %v", err)
			msg.Nak()
			return
		}

		if err := handler(&event); err != nil {
			log.Printf("Failed to handle event %s: %v", event.ID, err)
			msg.Nak()
			return
		}

		msg.Ack()
	}, nats.Durable(consumerName), nats.ManualAck())

	if err != nil {
		return fmt.Errorf("failed to subscribe to %s: %w", subject, err)
	}

	eb.subscriptions = append(eb.subscriptions, sub)
	log.Printf("Subscribed to events: %s", subject)
	return nil
}

// Close closes the event bus connection
func (eb *EventBus) Close() error {
	for _, sub := range eb.subscriptions {
		if err := sub.Unsubscribe(); err != nil {
			log.Printf("Failed to unsubscribe: %v", err)
		}
	}
	eb.conn.Close()
	return nil
}

// sanitizeConsumerName removes special characters from consumer name
func sanitizeConsumerName(name string) string {
	result := ""
	for _, c := range name {
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' {
			result += string(c)
		} else if c == '.' || c == '*' || c == '>' {
			result += "_"
		}
	}
	return result
}

// Helper functions for creating common events

// UserCreatedEvent creates a user.created event
func UserCreatedEvent(tenantID, userID, email string) *Event {
	return NewEvent(EventUserCreated, tenantID, "", map[string]interface{}{
		"user_id": userID,
		"email":   email,
	})
}

// UserUpdatedEvent creates a user.updated event
func UserUpdatedEvent(tenantID, userID string, changes map[string]interface{}) *Event {
	return NewEvent(EventUserUpdated, tenantID, "", map[string]interface{}{
		"user_id": userID,
		"changes": changes,
	})
}

// UserDeletedEvent creates a user.deleted event
func UserDeletedEvent(tenantID, userID string) *Event {
	return NewEvent(EventUserDeleted, tenantID, "", map[string]interface{}{
		"user_id": userID,
	})
}

// LoginEvent creates an auth.login event
func LoginEvent(tenantID, userID, sessionID, ipAddress string) *Event {
	return NewEvent(EventUserLoggedIn, tenantID, "", map[string]interface{}{
		"user_id":    userID,
		"session_id": sessionID,
		"ip_address": ipAddress,
	})
}

// LogoutEvent creates an auth.logout event
func LogoutEvent(tenantID, userID, sessionID string) *Event {
	return NewEvent(EventUserLoggedOut, tenantID, "", map[string]interface{}{
		"user_id":    userID,
		"session_id": sessionID,
	})
}

// SubscriptionCreatedEvent creates a billing.subscription.created event
func SubscriptionCreatedEvent(tenantID, subscriptionID, planID string) *Event {
	return NewEvent(EventSubscriptionCreated, tenantID, "", map[string]interface{}{
		"subscription_id": subscriptionID,
		"plan_id":         planID,
	})
}

// SubscriptionCanceledEvent creates a billing.subscription.canceled event
func SubscriptionCanceledEvent(tenantID, subscriptionID, reason string) *Event {
	return NewEvent(EventSubscriptionCanceled, tenantID, "", map[string]interface{}{
		"subscription_id": subscriptionID,
		"reason":          reason,
	})
}

// PaymentSucceededEvent creates a billing.payment.succeeded event
func PaymentSucceededEvent(tenantID, invoiceID string, amount int64, currency string) *Event {
	return NewEvent(EventPaymentSucceeded, tenantID, "", map[string]interface{}{
		"invoice_id": invoiceID,
		"amount":     amount,
		"currency":   currency,
	})
}

// InvoicePaidEvent creates a billing.invoice.paid event
func InvoicePaidEvent(tenantID, invoiceID string, amount int64) *Event {
	return NewEvent(EventInvoicePaid, tenantID, "", map[string]interface{}{
		"invoice_id": invoiceID,
		"amount":     amount,
	})
}
