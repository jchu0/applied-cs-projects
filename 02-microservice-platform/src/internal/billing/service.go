package billing

import (
	"context"
	"fmt"
	"time"

	"github.com/mlai/microservice-platform/pkg/events"
	"github.com/mlai/microservice-platform/pkg/logging"
	"github.com/stripe/stripe-go/v76"
	"github.com/stripe/stripe-go/v76/customer"
	"github.com/stripe/stripe-go/v76/paymentmethod"
	"github.com/stripe/stripe-go/v76/subscription"
	"go.uber.org/zap"
)

// Service handles billing business logic
type Service struct {
	repo      *Repository
	publisher *events.Publisher
	logger    *logging.Logger
}

// NewService creates a new billing service
func NewService(repo *Repository, publisher *events.Publisher, logger *logging.Logger, stripeKey string) *Service {
	stripe.Key = stripeKey
	return &Service{
		repo:      repo,
		publisher: publisher,
		logger:    logger,
	}
}

// CreateCustomerRequest contains customer creation parameters
type CreateCustomerRequest struct {
	TenantID       string
	UserID         string
	Email          string
	Name           string
	Phone          string
	AddressLine1   string
	AddressLine2   string
	AddressCity    string
	AddressState   string
	AddressPostal  string
	AddressCountry string
	Metadata       map[string]string
}

// CreateCustomer creates a customer in Stripe and locally
func (s *Service) CreateCustomer(ctx context.Context, req *CreateCustomerRequest) (*Customer, error) {
	logger := s.logger.WithContext(ctx)

	// Create customer in Stripe
	params := &stripe.CustomerParams{
		Email: stripe.String(req.Email),
		Name:  stripe.String(req.Name),
		Phone: stripe.String(req.Phone),
		Address: &stripe.AddressParams{
			Line1:      stripe.String(req.AddressLine1),
			Line2:      stripe.String(req.AddressLine2),
			City:       stripe.String(req.AddressCity),
			State:      stripe.String(req.AddressState),
			PostalCode: stripe.String(req.AddressPostal),
			Country:    stripe.String(req.AddressCountry),
		},
		Metadata: map[string]string{
			"tenant_id": req.TenantID,
			"user_id":   req.UserID,
		},
	}

	stripeCustomer, err := customer.New(params)
	if err != nil {
		logger.Error("failed to create Stripe customer", zap.Error(err))
		return nil, fmt.Errorf("failed to create Stripe customer: %w", err)
	}

	// Create local customer
	cust := &Customer{
		TenantID:         req.TenantID,
		UserID:           req.UserID,
		StripeCustomerID: stripeCustomer.ID,
		Email:            req.Email,
		Name:             req.Name,
		Phone:            req.Phone,
		AddressLine1:     req.AddressLine1,
		AddressLine2:     req.AddressLine2,
		AddressCity:      req.AddressCity,
		AddressState:     req.AddressState,
		AddressPostal:    req.AddressPostal,
		AddressCountry:   req.AddressCountry,
		Metadata:         req.Metadata,
	}

	if err := s.repo.CreateCustomer(ctx, cust); err != nil {
		logger.Error("failed to create local customer", zap.Error(err))
		return nil, err
	}

	// Publish event
	if s.publisher != nil {
		s.publisher.Publish(ctx, events.EventCustomerCreated, req.TenantID, map[string]string{
			"customer_id": cust.ID,
			"user_id":     req.UserID,
			"email":       req.Email,
		}, nil)
	}

	logger.Info("customer created",
		zap.String("customer_id", cust.ID),
		zap.String("stripe_customer_id", stripeCustomer.ID),
	)

	return cust, nil
}

// GetCustomer retrieves a customer
func (s *Service) GetCustomer(ctx context.Context, tenantID, customerID string) (*Customer, error) {
	return s.repo.GetCustomerByID(ctx, tenantID, customerID)
}

// GetOrCreateCustomer gets existing customer or creates new one
func (s *Service) GetOrCreateCustomer(ctx context.Context, req *CreateCustomerRequest) (*Customer, error) {
	// Try to get existing customer
	cust, err := s.repo.GetCustomerByUserID(ctx, req.TenantID, req.UserID)
	if err == nil {
		return cust, nil
	}
	if err != ErrCustomerNotFound {
		return nil, err
	}

	// Create new customer
	return s.CreateCustomer(ctx, req)
}

// CreateSubscriptionRequest contains subscription creation parameters
type CreateSubscriptionRequest struct {
	TenantID        string
	CustomerID      string
	PriceID         string
	Quantity        int64
	PaymentMethodID string
	Metadata        map[string]string
}

// CreateSubscription creates a subscription in Stripe and locally
func (s *Service) CreateSubscription(ctx context.Context, req *CreateSubscriptionRequest) (*Subscription, string, error) {
	logger := s.logger.WithContext(ctx)

	// Get customer
	cust, err := s.repo.GetCustomerByID(ctx, req.TenantID, req.CustomerID)
	if err != nil {
		return nil, "", err
	}

	// Create subscription in Stripe
	quantity := req.Quantity
	if quantity == 0 {
		quantity = 1
	}

	params := &stripe.SubscriptionParams{
		Customer: stripe.String(cust.StripeCustomerID),
		Items: []*stripe.SubscriptionItemsParams{
			{
				Price:    stripe.String(req.PriceID),
				Quantity: stripe.Int64(quantity),
			},
		},
		PaymentBehavior: stripe.String("default_incomplete"),
		PaymentSettings: &stripe.SubscriptionPaymentSettingsParams{
			SaveDefaultPaymentMethod: stripe.String("on_subscription"),
		},
		Metadata: map[string]string{
			"tenant_id":   req.TenantID,
			"customer_id": req.CustomerID,
		},
	}

	if req.PaymentMethodID != "" {
		params.DefaultPaymentMethod = stripe.String(req.PaymentMethodID)
	}

	stripeSub, err := subscription.New(params)
	if err != nil {
		logger.Error("failed to create Stripe subscription", zap.Error(err))
		return nil, "", fmt.Errorf("failed to create Stripe subscription: %w", err)
	}

	// Create local subscription
	sub := &Subscription{
		TenantID:             req.TenantID,
		CustomerID:           req.CustomerID,
		StripeSubscriptionID: stripeSub.ID,
		Status:               string(stripeSub.Status),
		PriceID:              req.PriceID,
		Quantity:             quantity,
		CurrentPeriodStart:   time.Unix(stripeSub.CurrentPeriodStart, 0),
		CurrentPeriodEnd:     time.Unix(stripeSub.CurrentPeriodEnd, 0),
		Metadata:             req.Metadata,
	}

	if err := s.repo.CreateSubscription(ctx, sub); err != nil {
		logger.Error("failed to create local subscription", zap.Error(err))
		return nil, "", err
	}

	// Get client secret for payment confirmation if needed
	var clientSecret string
	if stripeSub.LatestInvoice != nil && stripeSub.LatestInvoice.PaymentIntent != nil {
		clientSecret = stripeSub.LatestInvoice.PaymentIntent.ClientSecret
	}

	// Publish event
	if s.publisher != nil {
		s.publisher.Publish(ctx, events.EventSubscriptionCreated, req.TenantID, map[string]string{
			"subscription_id": sub.ID,
			"customer_id":     req.CustomerID,
			"price_id":        req.PriceID,
			"status":          sub.Status,
		}, nil)
	}

	logger.Info("subscription created",
		zap.String("subscription_id", sub.ID),
		zap.String("stripe_subscription_id", stripeSub.ID),
	)

	return sub, clientSecret, nil
}

// GetSubscription retrieves a subscription
func (s *Service) GetSubscription(ctx context.Context, tenantID, subscriptionID string) (*Subscription, error) {
	return s.repo.GetSubscriptionByID(ctx, tenantID, subscriptionID)
}

// CancelSubscription cancels a subscription
func (s *Service) CancelSubscription(ctx context.Context, tenantID, subscriptionID string, cancelAtPeriodEnd bool, reason string) (*Subscription, error) {
	logger := s.logger.WithContext(ctx)

	// Get subscription
	sub, err := s.repo.GetSubscriptionByID(ctx, tenantID, subscriptionID)
	if err != nil {
		return nil, err
	}

	// Cancel in Stripe
	params := &stripe.SubscriptionParams{
		CancelAtPeriodEnd: stripe.Bool(cancelAtPeriodEnd),
	}

	if reason != "" {
		params.Metadata = map[string]string{"cancel_reason": reason}
	}

	stripeSub, err := subscription.Update(sub.StripeSubscriptionID, params)
	if err != nil {
		logger.Error("failed to cancel Stripe subscription", zap.Error(err))
		return nil, fmt.Errorf("failed to cancel subscription: %w", err)
	}

	// Update local subscription
	now := time.Now().UTC()
	sub.Status = string(stripeSub.Status)
	sub.CancelAtPeriodEnd = cancelAtPeriodEnd
	if cancelAtPeriodEnd {
		cancelAt := time.Unix(stripeSub.CurrentPeriodEnd, 0)
		sub.CancelAt = &cancelAt
	} else {
		sub.CanceledAt = &now
	}

	if err := s.repo.UpdateSubscription(ctx, sub); err != nil {
		return nil, err
	}

	// Publish event
	if s.publisher != nil {
		s.publisher.Publish(ctx, events.EventSubscriptionCanceled, tenantID, map[string]string{
			"subscription_id":     sub.ID,
			"customer_id":         sub.CustomerID,
			"cancel_at_period_end": fmt.Sprintf("%v", cancelAtPeriodEnd),
			"reason":              reason,
		}, nil)
	}

	logger.Info("subscription canceled",
		zap.String("subscription_id", sub.ID),
		zap.Bool("cancel_at_period_end", cancelAtPeriodEnd),
	)

	return sub, nil
}

// ListSubscriptions lists subscriptions for a customer
func (s *Service) ListSubscriptions(ctx context.Context, tenantID, customerID string, limit, offset int) ([]*Subscription, int, error) {
	if limit <= 0 {
		limit = 20
	}
	if limit > 100 {
		limit = 100
	}
	return s.repo.ListSubscriptions(ctx, tenantID, customerID, limit, offset)
}

// AddPaymentMethod adds a payment method to a customer
func (s *Service) AddPaymentMethod(ctx context.Context, tenantID, customerID, paymentMethodID string, setDefault bool) error {
	logger := s.logger.WithContext(ctx)

	// Get customer
	cust, err := s.repo.GetCustomerByID(ctx, tenantID, customerID)
	if err != nil {
		return err
	}

	// Attach payment method to customer in Stripe
	params := &stripe.PaymentMethodAttachParams{
		Customer: stripe.String(cust.StripeCustomerID),
	}

	_, err = paymentmethod.Attach(paymentMethodID, params)
	if err != nil {
		logger.Error("failed to attach payment method", zap.Error(err))
		return fmt.Errorf("failed to attach payment method: %w", err)
	}

	// Set as default if requested
	if setDefault {
		updateParams := &stripe.CustomerParams{
			InvoiceSettings: &stripe.CustomerInvoiceSettingsParams{
				DefaultPaymentMethod: stripe.String(paymentMethodID),
			},
		}
		_, err = customer.Update(cust.StripeCustomerID, updateParams)
		if err != nil {
			logger.Warn("failed to set default payment method", zap.Error(err))
		}
	}

	logger.Info("payment method added",
		zap.String("customer_id", customerID),
		zap.String("payment_method_id", paymentMethodID),
	)

	return nil
}

// ListInvoices lists invoices for a customer
func (s *Service) ListInvoices(ctx context.Context, tenantID, customerID string, limit, offset int) ([]*Invoice, int, error) {
	if limit <= 0 {
		limit = 20
	}
	if limit > 100 {
		limit = 100
	}
	return s.repo.ListInvoices(ctx, tenantID, customerID, limit, offset)
}
