package events

import (
	"context"
	"encoding/json"
	"log"
	"time"

	"github.com/google/uuid"
	"github.com/nats-io/nats.go"
)

// Publisher handles event publishing for the billing service
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

// PublishSubscriptionCreated publishes a billing.subscription.created event
func (p *Publisher) PublishSubscriptionCreated(ctx context.Context, tenantID, subscriptionID, planID string) {
	event := map[string]interface{}{
		"event_id":       uuid.New().String(),
		"event_type":     "billing.subscription.created",
		"tenant_id":      tenantID,
		"timestamp":      time.Now().UTC().Format(time.RFC3339),
		"source_service": "billing-service",
		"data": map[string]interface{}{
			"subscription_id": subscriptionID,
			"plan_id":         planID,
		},
	}
	p.publish("events.billing.subscription.created", event)
}

// PublishSubscriptionUpdated publishes a billing.subscription.updated event
func (p *Publisher) PublishSubscriptionUpdated(ctx context.Context, tenantID, subscriptionID, newPlanID string) {
	event := map[string]interface{}{
		"event_id":       uuid.New().String(),
		"event_type":     "billing.subscription.updated",
		"tenant_id":      tenantID,
		"timestamp":      time.Now().UTC().Format(time.RFC3339),
		"source_service": "billing-service",
		"data": map[string]interface{}{
			"subscription_id": subscriptionID,
			"new_plan_id":     newPlanID,
		},
	}
	p.publish("events.billing.subscription.updated", event)
}

// PublishSubscriptionCanceled publishes a billing.subscription.canceled event
func (p *Publisher) PublishSubscriptionCanceled(ctx context.Context, tenantID, subscriptionID, reason string) {
	event := map[string]interface{}{
		"event_id":       uuid.New().String(),
		"event_type":     "billing.subscription.canceled",
		"tenant_id":      tenantID,
		"timestamp":      time.Now().UTC().Format(time.RFC3339),
		"source_service": "billing-service",
		"data": map[string]interface{}{
			"subscription_id": subscriptionID,
			"reason":          reason,
		},
	}
	p.publish("events.billing.subscription.canceled", event)
}

// PublishPaymentSucceeded publishes a billing.payment.succeeded event
func (p *Publisher) PublishPaymentSucceeded(ctx context.Context, tenantID, invoiceID string, amount int64, currency string) {
	event := map[string]interface{}{
		"event_id":       uuid.New().String(),
		"event_type":     "billing.payment.succeeded",
		"tenant_id":      tenantID,
		"timestamp":      time.Now().UTC().Format(time.RFC3339),
		"source_service": "billing-service",
		"data": map[string]interface{}{
			"invoice_id": invoiceID,
			"amount":     amount,
			"currency":   currency,
		},
	}
	p.publish("events.billing.payment.succeeded", event)
}

// PublishPaymentFailed publishes a billing.payment.failed event
func (p *Publisher) PublishPaymentFailed(ctx context.Context, tenantID, invoiceID, errorMessage string) {
	event := map[string]interface{}{
		"event_id":       uuid.New().String(),
		"event_type":     "billing.payment.failed",
		"tenant_id":      tenantID,
		"timestamp":      time.Now().UTC().Format(time.RFC3339),
		"source_service": "billing-service",
		"data": map[string]interface{}{
			"invoice_id":    invoiceID,
			"error_message": errorMessage,
		},
	}
	p.publish("events.billing.payment.failed", event)
}

// PublishInvoicePaid publishes a billing.invoice.paid event
func (p *Publisher) PublishInvoicePaid(ctx context.Context, tenantID, invoiceID string, amount int64) {
	event := map[string]interface{}{
		"event_id":       uuid.New().String(),
		"event_type":     "billing.invoice.paid",
		"tenant_id":      tenantID,
		"timestamp":      time.Now().UTC().Format(time.RFC3339),
		"source_service": "billing-service",
		"data": map[string]interface{}{
			"invoice_id": invoiceID,
			"amount":     amount,
		},
	}
	p.publish("events.billing.invoice.paid", event)
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
