package events

import (
	"context"
	"encoding/json"
	"log"
	"time"

	"github.com/google/uuid"
	"github.com/nats-io/nats.go"
)

// Publisher handles event publishing for the auth service
type Publisher struct {
	conn *nats.Conn
	js   nats.JetStreamContext
}

// NewPublisher creates a new event publisher
func NewPublisher(natsURL string) (*Publisher, error) {
	nc, err := nats.Connect(natsURL)
	if err != nil {
		return nil, err
	}

	js, err := nc.JetStream()
	if err != nil {
		nc.Close()
		return nil, err
	}

	// Ensure stream exists
	_, err = js.StreamInfo("EVENTS")
	if err != nil {
		_, err = js.AddStream(&nats.StreamConfig{
			Name:     "EVENTS",
			Subjects: []string{"events.>"},
		})
		if err != nil {
			nc.Close()
			return nil, err
		}
	}

	return &Publisher{conn: nc, js: js}, nil
}

// PublishUserLoggedIn publishes an auth.login event
func (p *Publisher) PublishUserLoggedIn(ctx context.Context, tenantID, userID, sessionID, email string) {
	event := map[string]interface{}{
		"event_id":       uuid.New().String(),
		"event_type":     "auth.login",
		"tenant_id":      tenantID,
		"timestamp":      time.Now().UTC().Format(time.RFC3339),
		"source_service": "auth-service",
		"data": map[string]interface{}{
			"user_id":    userID,
			"session_id": sessionID,
			"email":      email,
		},
	}
	p.publish("events.auth.login", event)
}

// PublishUserLoggedOut publishes an auth.logout event
func (p *Publisher) PublishUserLoggedOut(ctx context.Context, tenantID, userID, sessionID string) {
	event := map[string]interface{}{
		"event_id":       uuid.New().String(),
		"event_type":     "auth.logout",
		"tenant_id":      tenantID,
		"timestamp":      time.Now().UTC().Format(time.RFC3339),
		"source_service": "auth-service",
		"data": map[string]interface{}{
			"user_id":    userID,
			"session_id": sessionID,
		},
	}
	p.publish("events.auth.logout", event)
}

// PublishUserRegistered publishes an auth.registered event
func (p *Publisher) PublishUserRegistered(ctx context.Context, tenantID, userID, email string) {
	event := map[string]interface{}{
		"event_id":       uuid.New().String(),
		"event_type":     "auth.registered",
		"tenant_id":      tenantID,
		"timestamp":      time.Now().UTC().Format(time.RFC3339),
		"source_service": "auth-service",
		"data": map[string]interface{}{
			"user_id": userID,
			"email":   email,
		},
	}
	p.publish("events.auth.registered", event)
}

// PublishTokenRefreshed publishes an auth.token_refreshed event
func (p *Publisher) PublishTokenRefreshed(ctx context.Context, tenantID, userID, sessionID string) {
	event := map[string]interface{}{
		"event_id":       uuid.New().String(),
		"event_type":     "auth.token_refreshed",
		"tenant_id":      tenantID,
		"timestamp":      time.Now().UTC().Format(time.RFC3339),
		"source_service": "auth-service",
		"data": map[string]interface{}{
			"user_id":    userID,
			"session_id": sessionID,
		},
	}
	p.publish("events.auth.token_refreshed", event)
}

func (p *Publisher) publish(subject string, event map[string]interface{}) {
	data, err := json.Marshal(event)
	if err != nil {
		log.Printf("Failed to marshal event: %v", err)
		return
	}

	if _, err := p.js.Publish(subject, data); err != nil {
		log.Printf("Failed to publish event to %s: %v", subject, err)
	} else {
		log.Printf("Published event: %s", subject)
	}
}

// Close closes the publisher connection
func (p *Publisher) Close() {
	p.conn.Close()
}
