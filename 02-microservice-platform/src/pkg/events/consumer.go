package events

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/nats-io/nats.go"
)

// Handler processes an event
type Handler func(ctx context.Context, event *Event) error

// Consumer consumes events from NATS
type Consumer struct {
	conn     *nats.Conn
	js       nats.JetStreamContext
	handlers map[string]Handler
	subs     []*nats.Subscription
}

// NewConsumer creates a new event consumer
func NewConsumer(url string) (*Consumer, error) {
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

	return &Consumer{
		conn:     conn,
		js:       js,
		handlers: make(map[string]Handler),
		subs:     make([]*nats.Subscription, 0),
	}, nil
}

// Subscribe subscribes to an event type
func (c *Consumer) Subscribe(eventType string, handler Handler) error {
	c.handlers[eventType] = handler

	// Create durable consumer
	consumerName := fmt.Sprintf("consumer-%s", eventType)

	sub, err := c.js.Subscribe(eventType, func(msg *nats.Msg) {
		var event Event
		if err := json.Unmarshal(msg.Data, &event); err != nil {
			msg.Nak()
			return
		}

		ctx := context.Background()
		if err := handler(ctx, &event); err != nil {
			msg.Nak()
			return
		}

		msg.Ack()
	}, nats.Durable(consumerName), nats.ManualAck())

	if err != nil {
		return fmt.Errorf("failed to subscribe to %s: %w", eventType, err)
	}

	c.subs = append(c.subs, sub)
	return nil
}

// SubscribePattern subscribes to events matching a pattern
func (c *Consumer) SubscribePattern(pattern string, consumerName string, handler Handler) error {
	sub, err := c.js.Subscribe(pattern, func(msg *nats.Msg) {
		var event Event
		if err := json.Unmarshal(msg.Data, &event); err != nil {
			msg.Nak()
			return
		}

		ctx := context.Background()
		if err := handler(ctx, &event); err != nil {
			msg.Nak()
			return
		}

		msg.Ack()
	}, nats.Durable(consumerName), nats.ManualAck())

	if err != nil {
		return fmt.Errorf("failed to subscribe to %s: %w", pattern, err)
	}

	c.subs = append(c.subs, sub)
	return nil
}

// Close closes all subscriptions and the connection
func (c *Consumer) Close() {
	for _, sub := range c.subs {
		sub.Unsubscribe()
	}
	if c.conn != nil {
		c.conn.Close()
	}
}
