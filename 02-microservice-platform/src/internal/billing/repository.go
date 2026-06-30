package billing

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
	ErrCustomerNotFound     = errors.New("customer not found")
	ErrSubscriptionNotFound = errors.New("subscription not found")
	ErrInvoiceNotFound      = errors.New("invoice not found")
	ErrPaymentMethodNotFound = errors.New("payment method not found")
)

// Customer represents a billing customer
type Customer struct {
	ID               string
	TenantID         string
	UserID           string
	StripeCustomerID string
	Email            string
	Name             string
	Phone            string
	AddressLine1     string
	AddressLine2     string
	AddressCity      string
	AddressState     string
	AddressPostal    string
	AddressCountry   string
	Metadata         map[string]string
	CreatedAt        time.Time
	UpdatedAt        time.Time
}

// Subscription represents a billing subscription
type Subscription struct {
	ID                   string
	TenantID             string
	CustomerID           string
	StripeSubscriptionID string
	Status               string
	PriceID              string
	PlanName             string
	Quantity             int64
	CurrentPeriodStart   time.Time
	CurrentPeriodEnd     time.Time
	CancelAt             *time.Time
	CancelAtPeriodEnd    bool
	CanceledAt           *time.Time
	Metadata             map[string]string
	CreatedAt            time.Time
	UpdatedAt            time.Time
}

// PaymentMethod represents a payment method
type PaymentMethod struct {
	ID                    string
	CustomerID            string
	StripePaymentMethodID string
	Type                  string
	CardBrand             string
	CardLast4             string
	CardExpMonth          int
	CardExpYear           int
	IsDefault             bool
	CreatedAt             time.Time
}

// Invoice represents an invoice
type Invoice struct {
	ID               string
	TenantID         string
	CustomerID       string
	SubscriptionID   string
	StripeInvoiceID  string
	Status           string
	AmountDue        int64
	AmountPaid       int64
	AmountRemaining  int64
	Currency         string
	InvoicePDF       string
	HostedInvoiceURL string
	PeriodStart      time.Time
	PeriodEnd        time.Time
	DueDate          *time.Time
	PaidAt           *time.Time
	CreatedAt        time.Time
}

// Repository handles billing data access
type Repository struct {
	pool *pgxpool.Pool
}

// NewRepository creates a new billing repository
func NewRepository(pool *pgxpool.Pool) *Repository {
	return &Repository{pool: pool}
}

// CreateCustomer creates a new customer
func (r *Repository) CreateCustomer(ctx context.Context, customer *Customer) error {
	customer.ID = uuid.New().String()
	customer.CreatedAt = time.Now().UTC()
	customer.UpdatedAt = customer.CreatedAt

	query := `
		INSERT INTO customers (
			id, tenant_id, user_id, stripe_customer_id, email, name, phone,
			address_line1, address_line2, address_city, address_state,
			address_postal_code, address_country, metadata, created_at, updated_at
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
	`

	_, err := r.pool.Exec(ctx, query,
		customer.ID, customer.TenantID, customer.UserID, customer.StripeCustomerID,
		customer.Email, customer.Name, customer.Phone,
		customer.AddressLine1, customer.AddressLine2, customer.AddressCity,
		customer.AddressState, customer.AddressPostal, customer.AddressCountry,
		customer.Metadata, customer.CreatedAt, customer.UpdatedAt,
	)
	if err != nil {
		return fmt.Errorf("failed to create customer: %w", err)
	}

	return nil
}

// GetCustomerByID retrieves a customer by ID
func (r *Repository) GetCustomerByID(ctx context.Context, tenantID, customerID string) (*Customer, error) {
	query := `
		SELECT id, tenant_id, user_id, stripe_customer_id, email, name, phone,
			   address_line1, address_line2, address_city, address_state,
			   address_postal_code, address_country, metadata, created_at, updated_at
		FROM customers
		WHERE id = $1 AND tenant_id = $2
	`

	customer := &Customer{}
	err := r.pool.QueryRow(ctx, query, customerID, tenantID).Scan(
		&customer.ID, &customer.TenantID, &customer.UserID, &customer.StripeCustomerID,
		&customer.Email, &customer.Name, &customer.Phone,
		&customer.AddressLine1, &customer.AddressLine2, &customer.AddressCity,
		&customer.AddressState, &customer.AddressPostal, &customer.AddressCountry,
		&customer.Metadata, &customer.CreatedAt, &customer.UpdatedAt,
	)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, ErrCustomerNotFound
		}
		return nil, fmt.Errorf("failed to get customer: %w", err)
	}

	return customer, nil
}

// GetCustomerByUserID retrieves a customer by user ID
func (r *Repository) GetCustomerByUserID(ctx context.Context, tenantID, userID string) (*Customer, error) {
	query := `
		SELECT id, tenant_id, user_id, stripe_customer_id, email, name, phone,
			   address_line1, address_line2, address_city, address_state,
			   address_postal_code, address_country, metadata, created_at, updated_at
		FROM customers
		WHERE user_id = $1 AND tenant_id = $2
	`

	customer := &Customer{}
	err := r.pool.QueryRow(ctx, query, userID, tenantID).Scan(
		&customer.ID, &customer.TenantID, &customer.UserID, &customer.StripeCustomerID,
		&customer.Email, &customer.Name, &customer.Phone,
		&customer.AddressLine1, &customer.AddressLine2, &customer.AddressCity,
		&customer.AddressState, &customer.AddressPostal, &customer.AddressCountry,
		&customer.Metadata, &customer.CreatedAt, &customer.UpdatedAt,
	)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, ErrCustomerNotFound
		}
		return nil, fmt.Errorf("failed to get customer: %w", err)
	}

	return customer, nil
}

// CreateSubscription creates a new subscription
func (r *Repository) CreateSubscription(ctx context.Context, sub *Subscription) error {
	sub.ID = uuid.New().String()
	sub.CreatedAt = time.Now().UTC()
	sub.UpdatedAt = sub.CreatedAt

	query := `
		INSERT INTO subscriptions (
			id, tenant_id, customer_id, stripe_subscription_id, status, price_id,
			plan_name, quantity, current_period_start, current_period_end,
			cancel_at, cancel_at_period_end, canceled_at, metadata, created_at, updated_at
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
	`

	_, err := r.pool.Exec(ctx, query,
		sub.ID, sub.TenantID, sub.CustomerID, sub.StripeSubscriptionID, sub.Status,
		sub.PriceID, sub.PlanName, sub.Quantity,
		sub.CurrentPeriodStart, sub.CurrentPeriodEnd,
		sub.CancelAt, sub.CancelAtPeriodEnd, sub.CanceledAt,
		sub.Metadata, sub.CreatedAt, sub.UpdatedAt,
	)
	if err != nil {
		return fmt.Errorf("failed to create subscription: %w", err)
	}

	return nil
}

// GetSubscriptionByID retrieves a subscription by ID
func (r *Repository) GetSubscriptionByID(ctx context.Context, tenantID, subID string) (*Subscription, error) {
	query := `
		SELECT id, tenant_id, customer_id, stripe_subscription_id, status, price_id,
			   plan_name, quantity, current_period_start, current_period_end,
			   cancel_at, cancel_at_period_end, canceled_at, metadata, created_at, updated_at
		FROM subscriptions
		WHERE id = $1 AND tenant_id = $2
	`

	sub := &Subscription{}
	err := r.pool.QueryRow(ctx, query, subID, tenantID).Scan(
		&sub.ID, &sub.TenantID, &sub.CustomerID, &sub.StripeSubscriptionID, &sub.Status,
		&sub.PriceID, &sub.PlanName, &sub.Quantity,
		&sub.CurrentPeriodStart, &sub.CurrentPeriodEnd,
		&sub.CancelAt, &sub.CancelAtPeriodEnd, &sub.CanceledAt,
		&sub.Metadata, &sub.CreatedAt, &sub.UpdatedAt,
	)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, ErrSubscriptionNotFound
		}
		return nil, fmt.Errorf("failed to get subscription: %w", err)
	}

	return sub, nil
}

// UpdateSubscription updates a subscription
func (r *Repository) UpdateSubscription(ctx context.Context, sub *Subscription) error {
	sub.UpdatedAt = time.Now().UTC()

	query := `
		UPDATE subscriptions SET
			status = $3, price_id = $4, plan_name = $5, quantity = $6,
			current_period_start = $7, current_period_end = $8,
			cancel_at = $9, cancel_at_period_end = $10, canceled_at = $11,
			updated_at = $12
		WHERE id = $1 AND tenant_id = $2
	`

	result, err := r.pool.Exec(ctx, query,
		sub.ID, sub.TenantID, sub.Status, sub.PriceID, sub.PlanName, sub.Quantity,
		sub.CurrentPeriodStart, sub.CurrentPeriodEnd,
		sub.CancelAt, sub.CancelAtPeriodEnd, sub.CanceledAt, sub.UpdatedAt,
	)
	if err != nil {
		return fmt.Errorf("failed to update subscription: %w", err)
	}

	if result.RowsAffected() == 0 {
		return ErrSubscriptionNotFound
	}

	return nil
}

// ListSubscriptions lists subscriptions for a customer
func (r *Repository) ListSubscriptions(ctx context.Context, tenantID, customerID string, limit, offset int) ([]*Subscription, int, error) {
	// Count total
	var total int
	countQuery := `SELECT COUNT(*) FROM subscriptions WHERE tenant_id = $1 AND customer_id = $2`
	err := r.pool.QueryRow(ctx, countQuery, tenantID, customerID).Scan(&total)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to count subscriptions: %w", err)
	}

	// Get subscriptions
	query := `
		SELECT id, tenant_id, customer_id, stripe_subscription_id, status, price_id,
			   plan_name, quantity, current_period_start, current_period_end,
			   cancel_at, cancel_at_period_end, canceled_at, metadata, created_at, updated_at
		FROM subscriptions
		WHERE tenant_id = $1 AND customer_id = $2
		ORDER BY created_at DESC
		LIMIT $3 OFFSET $4
	`

	rows, err := r.pool.Query(ctx, query, tenantID, customerID, limit, offset)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to list subscriptions: %w", err)
	}
	defer rows.Close()

	var subs []*Subscription
	for rows.Next() {
		sub := &Subscription{}
		err := rows.Scan(
			&sub.ID, &sub.TenantID, &sub.CustomerID, &sub.StripeSubscriptionID, &sub.Status,
			&sub.PriceID, &sub.PlanName, &sub.Quantity,
			&sub.CurrentPeriodStart, &sub.CurrentPeriodEnd,
			&sub.CancelAt, &sub.CancelAtPeriodEnd, &sub.CanceledAt,
			&sub.Metadata, &sub.CreatedAt, &sub.UpdatedAt,
		)
		if err != nil {
			return nil, 0, fmt.Errorf("failed to scan subscription: %w", err)
		}
		subs = append(subs, sub)
	}

	return subs, total, nil
}

// CreateInvoice creates a new invoice
func (r *Repository) CreateInvoice(ctx context.Context, invoice *Invoice) error {
	invoice.ID = uuid.New().String()
	invoice.CreatedAt = time.Now().UTC()

	query := `
		INSERT INTO invoices (
			id, tenant_id, customer_id, subscription_id, stripe_invoice_id, status,
			amount_due, amount_paid, amount_remaining, currency, invoice_pdf,
			hosted_invoice_url, period_start, period_end, due_date, paid_at, created_at
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
	`

	_, err := r.pool.Exec(ctx, query,
		invoice.ID, invoice.TenantID, invoice.CustomerID, invoice.SubscriptionID,
		invoice.StripeInvoiceID, invoice.Status,
		invoice.AmountDue, invoice.AmountPaid, invoice.AmountRemaining, invoice.Currency,
		invoice.InvoicePDF, invoice.HostedInvoiceURL,
		invoice.PeriodStart, invoice.PeriodEnd, invoice.DueDate, invoice.PaidAt, invoice.CreatedAt,
	)
	if err != nil {
		return fmt.Errorf("failed to create invoice: %w", err)
	}

	return nil
}

// ListInvoices lists invoices for a customer
func (r *Repository) ListInvoices(ctx context.Context, tenantID, customerID string, limit, offset int) ([]*Invoice, int, error) {
	var total int
	countQuery := `SELECT COUNT(*) FROM invoices WHERE tenant_id = $1 AND customer_id = $2`
	err := r.pool.QueryRow(ctx, countQuery, tenantID, customerID).Scan(&total)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to count invoices: %w", err)
	}

	query := `
		SELECT id, tenant_id, customer_id, subscription_id, stripe_invoice_id, status,
			   amount_due, amount_paid, amount_remaining, currency, invoice_pdf,
			   hosted_invoice_url, period_start, period_end, due_date, paid_at, created_at
		FROM invoices
		WHERE tenant_id = $1 AND customer_id = $2
		ORDER BY created_at DESC
		LIMIT $3 OFFSET $4
	`

	rows, err := r.pool.Query(ctx, query, tenantID, customerID, limit, offset)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to list invoices: %w", err)
	}
	defer rows.Close()

	var invoices []*Invoice
	for rows.Next() {
		inv := &Invoice{}
		err := rows.Scan(
			&inv.ID, &inv.TenantID, &inv.CustomerID, &inv.SubscriptionID,
			&inv.StripeInvoiceID, &inv.Status,
			&inv.AmountDue, &inv.AmountPaid, &inv.AmountRemaining, &inv.Currency,
			&inv.InvoicePDF, &inv.HostedInvoiceURL,
			&inv.PeriodStart, &inv.PeriodEnd, &inv.DueDate, &inv.PaidAt, &inv.CreatedAt,
		)
		if err != nil {
			return nil, 0, fmt.Errorf("failed to scan invoice: %w", err)
		}
		invoices = append(invoices, inv)
	}

	return invoices, total, nil
}
