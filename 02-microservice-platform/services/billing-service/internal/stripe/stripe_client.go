package stripe

import (
	"context"
	"fmt"
	"time"

	"github.com/project/microservices/billing-service/internal/models"
	"github.com/stripe/stripe-go/v76"
	"github.com/stripe/stripe-go/v76/customer"
	"github.com/stripe/stripe-go/v76/invoice"
	"github.com/stripe/stripe-go/v76/paymentmethod"
	"github.com/stripe/stripe-go/v76/subscription"
)

// Client wraps the Stripe API client
type Client struct {
	secretKey string
}

// NewClient creates a new Stripe client
func NewClient(secretKey string) *Client {
	stripe.Key = secretKey
	return &Client{secretKey: secretKey}
}

// CreateCustomer creates a new Stripe customer
func (c *Client) CreateCustomer(ctx context.Context, email, name string, metadata map[string]string) (*stripe.Customer, error) {
	params := &stripe.CustomerParams{
		Email: stripe.String(email),
		Name:  stripe.String(name),
	}

	if metadata != nil {
		params.Metadata = metadata
	}

	cust, err := customer.New(params)
	if err != nil {
		return nil, fmt.Errorf("failed to create Stripe customer: %w", err)
	}

	return cust, nil
}

// GetCustomer retrieves a Stripe customer
func (c *Client) GetCustomer(ctx context.Context, customerID string) (*stripe.Customer, error) {
	cust, err := customer.Get(customerID, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to get Stripe customer: %w", err)
	}
	return cust, nil
}

// CreateSubscription creates a new Stripe subscription
func (c *Client) CreateSubscription(ctx context.Context, input *models.CreateSubscriptionInput, stripeCustomerID string) (*stripe.Subscription, error) {
	params := &stripe.SubscriptionParams{
		Customer: stripe.String(stripeCustomerID),
		Items: []*stripe.SubscriptionItemsParams{
			{Price: stripe.String(input.PriceID)},
		},
		PaymentBehavior: stripe.String("default_incomplete"),
		PaymentSettings: &stripe.SubscriptionPaymentSettingsParams{
			SaveDefaultPaymentMethod: stripe.String("on_subscription"),
		},
	}

	if input.PaymentMethodID != "" {
		params.DefaultPaymentMethod = stripe.String(input.PaymentMethodID)
	}

	if input.Metadata != nil {
		params.Metadata = input.Metadata
	}

	// Add tenant and user IDs to metadata
	if params.Metadata == nil {
		params.Metadata = make(map[string]string)
	}
	params.Metadata["tenant_id"] = input.TenantID
	params.Metadata["user_id"] = input.UserID

	sub, err := subscription.New(params)
	if err != nil {
		return nil, fmt.Errorf("failed to create Stripe subscription: %w", err)
	}

	return sub, nil
}

// GetSubscription retrieves a Stripe subscription
func (c *Client) GetSubscription(ctx context.Context, subscriptionID string) (*stripe.Subscription, error) {
	sub, err := subscription.Get(subscriptionID, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to get Stripe subscription: %w", err)
	}
	return sub, nil
}

// UpdateSubscription updates a Stripe subscription (change plan)
func (c *Client) UpdateSubscription(ctx context.Context, subscriptionID, newPriceID string, prorate bool) (*stripe.Subscription, error) {
	// First get the current subscription to find the item ID
	sub, err := subscription.Get(subscriptionID, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to get subscription: %w", err)
	}

	if len(sub.Items.Data) == 0 {
		return nil, fmt.Errorf("subscription has no items")
	}

	itemID := sub.Items.Data[0].ID

	params := &stripe.SubscriptionParams{
		Items: []*stripe.SubscriptionItemsParams{
			{
				ID:    stripe.String(itemID),
				Price: stripe.String(newPriceID),
			},
		},
	}

	if prorate {
		params.ProrationBehavior = stripe.String("create_prorations")
	} else {
		params.ProrationBehavior = stripe.String("none")
	}

	updatedSub, err := subscription.Update(subscriptionID, params)
	if err != nil {
		return nil, fmt.Errorf("failed to update Stripe subscription: %w", err)
	}

	return updatedSub, nil
}

// CancelSubscription cancels a Stripe subscription
func (c *Client) CancelSubscription(ctx context.Context, subscriptionID string, cancelImmediately bool) (*stripe.Subscription, error) {
	if cancelImmediately {
		sub, err := subscription.Cancel(subscriptionID, nil)
		if err != nil {
			return nil, fmt.Errorf("failed to cancel Stripe subscription: %w", err)
		}
		return sub, nil
	}

	// Cancel at period end
	params := &stripe.SubscriptionParams{
		CancelAtPeriodEnd: stripe.Bool(true),
	}

	sub, err := subscription.Update(subscriptionID, params)
	if err != nil {
		return nil, fmt.Errorf("failed to update Stripe subscription: %w", err)
	}

	return sub, nil
}

// AttachPaymentMethod attaches a payment method to a customer
func (c *Client) AttachPaymentMethod(ctx context.Context, paymentMethodID, customerID string) (*stripe.PaymentMethod, error) {
	params := &stripe.PaymentMethodAttachParams{
		Customer: stripe.String(customerID),
	}

	pm, err := paymentmethod.Attach(paymentMethodID, params)
	if err != nil {
		return nil, fmt.Errorf("failed to attach payment method: %w", err)
	}

	return pm, nil
}

// SetDefaultPaymentMethod sets the default payment method for a customer
func (c *Client) SetDefaultPaymentMethod(ctx context.Context, customerID, paymentMethodID string) error {
	params := &stripe.CustomerParams{
		InvoiceSettings: &stripe.CustomerInvoiceSettingsParams{
			DefaultPaymentMethod: stripe.String(paymentMethodID),
		},
	}

	_, err := customer.Update(customerID, params)
	if err != nil {
		return fmt.Errorf("failed to set default payment method: %w", err)
	}

	return nil
}

// ListInvoices lists invoices for a customer
func (c *Client) ListInvoices(ctx context.Context, customerID string, limit int64) ([]*stripe.Invoice, error) {
	params := &stripe.InvoiceListParams{
		Customer: stripe.String(customerID),
	}
	params.Limit = stripe.Int64(limit)

	var invoices []*stripe.Invoice
	iter := invoice.List(params)

	for iter.Next() {
		invoices = append(invoices, iter.Invoice())
	}

	if err := iter.Err(); err != nil {
		return nil, fmt.Errorf("failed to list invoices: %w", err)
	}

	return invoices, nil
}

// SubscriptionToModel converts a Stripe subscription to our model
func SubscriptionToModel(stripeSub *stripe.Subscription, tenantID string) *models.Subscription {
	return &models.Subscription{
		TenantID:             tenantID,
		StripeSubscriptionID: stripeSub.ID,
		StripeCustomerID:     stripeSub.Customer.ID,
		PlanID:               stripeSub.Items.Data[0].Price.ID,
		Status:               models.SubscriptionStatus(stripeSub.Status),
		CurrentPeriodStart:   time.Unix(stripeSub.CurrentPeriodStart, 0),
		CurrentPeriodEnd:     time.Unix(stripeSub.CurrentPeriodEnd, 0),
		CancelAtPeriodEnd:    stripeSub.CancelAtPeriodEnd,
		Metadata:             stripeSub.Metadata,
	}
}

// InvoiceToModel converts a Stripe invoice to our model
func InvoiceToModel(stripeInv *stripe.Invoice, tenantID string) *models.Invoice {
	var dueDate time.Time
	if stripeInv.DueDate > 0 {
		dueDate = time.Unix(stripeInv.DueDate, 0)
	}

	return &models.Invoice{
		TenantID:        tenantID,
		StripeInvoiceID: stripeInv.ID,
		Status:          string(stripeInv.Status),
		AmountDue:       stripeInv.AmountDue,
		AmountPaid:      stripeInv.AmountPaid,
		Currency:        string(stripeInv.Currency),
		InvoiceURL:      stripeInv.HostedInvoiceURL,
		InvoicePDF:      stripeInv.InvoicePDF,
		DueDate:         dueDate,
	}
}

// PaymentMethodToModel converts a Stripe payment method to our model
func PaymentMethodToModel(pm *stripe.PaymentMethod, isDefault bool) *models.PaymentMethod {
	result := &models.PaymentMethod{
		ID:        pm.ID,
		Type:      string(pm.Type),
		IsDefault: isDefault,
	}

	if pm.Card != nil {
		result.LastFour = pm.Card.Last4
		result.Brand = string(pm.Card.Brand)
		result.ExpMonth = int32(pm.Card.ExpMonth)
		result.ExpYear = int32(pm.Card.ExpYear)
	}

	return result
}
