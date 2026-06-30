package events

import (
	"context"
	"encoding/json"
	"log"
	"time"

	"github.com/google/uuid"
	"github.com/nats-io/nats.go"
)

// Publisher handles event publishing for the user service
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

// PublishUserCreated publishes a user.created event
func (p *Publisher) PublishUserCreated(ctx context.Context, tenantID, userID, email string) {
	event := map[string]interface{}{
		"event_id":       uuid.New().String(),
		"event_type":     "user.created",
		"tenant_id":      tenantID,
		"timestamp":      time.Now().UTC().Format(time.RFC3339),
		"source_service": "user-service",
		"data": map[string]interface{}{
			"user_id": userID,
			"email":   email,
		},
	}
	p.publish("events.user.created", event)
}

// PublishUserUpdated publishes a user.updated event
func (p *Publisher) PublishUserUpdated(ctx context.Context, tenantID, userID string) {
	event := map[string]interface{}{
		"event_id":       uuid.New().String(),
		"event_type":     "user.updated",
		"tenant_id":      tenantID,
		"timestamp":      time.Now().UTC().Format(time.RFC3339),
		"source_service": "user-service",
		"data": map[string]interface{}{
			"user_id": userID,
		},
	}
	p.publish("events.user.updated", event)
}

// PublishUserDeleted publishes a user.deleted event
func (p *Publisher) PublishUserDeleted(ctx context.Context, tenantID, userID string) {
	event := map[string]interface{}{
		"event_id":       uuid.New().String(),
		"event_type":     "user.deleted",
		"tenant_id":      tenantID,
		"timestamp":      time.Now().UTC().Format(time.RFC3339),
		"source_service": "user-service",
		"data": map[string]interface{}{
			"user_id": userID,
		},
	}
	p.publish("events.user.deleted", event)
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
