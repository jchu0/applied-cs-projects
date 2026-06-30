package repository

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/project/microservices/billing-service/internal/models"
)

var (
	ErrSubscriptionNotFound = errors.New("subscription not found")
	ErrCustomerNotFound     = errors.New("customer not found")
	ErrPlanNotFound         = errors.New("plan not found")
)

// BillingRepository handles database operations for billing
type BillingRepository struct {
	db *pgxpool.Pool
}

// NewBillingRepository creates a new billing repository
func NewBillingRepository(db *pgxpool.Pool) *BillingRepository {
	return &BillingRepository{db: db}
}

// CreateCustomer creates a new customer
func (r *BillingRepository) CreateCustomer(ctx context.Context, customer *models.Customer) error {
	customer.ID = uuid.New().String()
	customer.CreatedAt = time.Now().UTC()
	customer.UpdatedAt = time.Now().UTC()

	metadataJSON, err := json.Marshal(customer.Metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}

	query := `
		INSERT INTO customers (id, tenant_id, stripe_customer_id, email, name, metadata, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
	`

	_, err = r.db.Exec(ctx, query,
		customer.ID,
		customer.TenantID,
		customer.StripeCustomerID,
		customer.Email,
		customer.Name,
		metadataJSON,
		customer.CreatedAt,
		customer.UpdatedAt,
	)

	if err != nil {
		return fmt.Errorf("failed to create customer: %w", err)
	}

	return nil
}

// GetCustomerByTenantID retrieves a customer by tenant ID
func (r *BillingRepository) GetCustomerByTenantID(ctx context.Context, tenantID string) (*models.Customer, error) {
	query := `
		SELECT id, tenant_id, stripe_customer_id, email, name, metadata, created_at, updated_at
		FROM customers
		WHERE tenant_id = $1
	`

	var customer models.Customer
	var metadataBytes []byte

	err := r.db.QueryRow(ctx, query, tenantID).Scan(
		&customer.ID,
		&customer.TenantID,
		&customer.StripeCustomerID,
		&customer.Email,
		&customer.Name,
		&metadataBytes,
		&customer.CreatedAt,
		&customer.UpdatedAt,
	)

	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, ErrCustomerNotFound
		}
		return nil, fmt.Errorf("failed to get customer: %w", err)
	}

	if err := json.Unmarshal(metadataBytes, &customer.Metadata); err != nil {
		customer.Metadata = make(map[string]string)
	}

	return &customer, nil
}

// CreateSubscription creates a new subscription
func (r *BillingRepository) CreateSubscription(ctx context.Context, sub *models.Subscription) error {
	sub.ID = uuid.New().String()
	sub.CreatedAt = time.Now().UTC()
	sub.UpdatedAt = time.Now().UTC()

	metadataJSON, err := json.Marshal(sub.Metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}

	query := `
		INSERT INTO subscriptions (id, tenant_id, stripe_subscription_id, stripe_customer_id, plan_id, status,
			current_period_start, current_period_end, cancel_at_period_end, metadata, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
	`

	_, err = r.db.Exec(ctx, query,
		sub.ID,
		sub.TenantID,
		sub.StripeSubscriptionID,
		sub.StripeCustomerID,
		sub.PlanID,
		sub.Status,
		sub.CurrentPeriodStart,
		sub.CurrentPeriodEnd,
		sub.CancelAtPeriodEnd,
		metadataJSON,
		sub.CreatedAt,
		sub.UpdatedAt,
	)

	if err != nil {
		return fmt.Errorf("failed to create subscription: %w", err)
	}

	return nil
}

// GetSubscription retrieves a subscription by ID
func (r *BillingRepository) GetSubscription(ctx context.Context, subscriptionID, tenantID string) (*models.Subscription, error) {
	query := `
		SELECT id, tenant_id, stripe_subscription_id, stripe_customer_id, plan_id, status,
			current_period_start, current_period_end, cancel_at_period_end, metadata, created_at, updated_at
		FROM subscriptions
		WHERE id = $1 AND tenant_id = $2
	`

	var sub models.Subscription
	var metadataBytes []byte

	err := r.db.QueryRow(ctx, query, subscriptionID, tenantID).Scan(
		&sub.ID,
		&sub.TenantID,
		&sub.StripeSubscriptionID,
		&sub.StripeCustomerID,
		&sub.PlanID,
		&sub.Status,
		&sub.CurrentPeriodStart,
		&sub.CurrentPeriodEnd,
		&sub.CancelAtPeriodEnd,
		&metadataBytes,
		&sub.CreatedAt,
		&sub.UpdatedAt,
	)

	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, ErrSubscriptionNotFound
		}
		return nil, fmt.Errorf("failed to get subscription: %w", err)
	}

	if err := json.Unmarshal(metadataBytes, &sub.Metadata); err != nil {
		sub.Metadata = make(map[string]string)
	}

	return &sub, nil
}

// UpdateSubscription updates a subscription
func (r *BillingRepository) UpdateSubscription(ctx context.Context, sub *models.Subscription) error {
	sub.UpdatedAt = time.Now().UTC()

	metadataJSON, err := json.Marshal(sub.Metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}

	query := `
		UPDATE subscriptions
		SET status = $1, plan_id = $2, current_period_start = $3, current_period_end = $4,
			cancel_at_period_end = $5, metadata = $6, updated_at = $7
		WHERE id = $8 AND tenant_id = $9
	`

	result, err := r.db.Exec(ctx, query,
		sub.Status,
		sub.PlanID,
		sub.CurrentPeriodStart,
		sub.CurrentPeriodEnd,
		sub.CancelAtPeriodEnd,
		metadataJSON,
		sub.UpdatedAt,
		sub.ID,
		sub.TenantID,
	)

	if err != nil {
		return fmt.Errorf("failed to update subscription: %w", err)
	}

	if result.RowsAffected() == 0 {
		return ErrSubscriptionNotFound
	}

	return nil
}

// ListSubscriptions lists subscriptions for a tenant
func (r *BillingRepository) ListSubscriptions(ctx context.Context, tenantID string, statusFilter string, limit, offset int) ([]*models.Subscription, int, error) {
	// Count query
	countQuery := `SELECT COUNT(*) FROM subscriptions WHERE tenant_id = $1`
	countArgs := []interface{}{tenantID}

	if statusFilter != "" {
		countQuery += ` AND status = $2`
		countArgs = append(countArgs, statusFilter)
	}

	var totalCount int
	if err := r.db.QueryRow(ctx, countQuery, countArgs...).Scan(&totalCount); err != nil {
		return nil, 0, fmt.Errorf("failed to count subscriptions: %w", err)
	}

	// List query
	query := `
		SELECT id, tenant_id, stripe_subscription_id, stripe_customer_id, plan_id, status,
			current_period_start, current_period_end, cancel_at_period_end, metadata, created_at, updated_at
		FROM subscriptions
		WHERE tenant_id = $1
	`
	args := []interface{}{tenantID}
	argIndex := 2

	if statusFilter != "" {
		query += fmt.Sprintf(` AND status = $%d`, argIndex)
		args = append(args, statusFilter)
		argIndex++
	}

	query += ` ORDER BY created_at DESC`
	query += fmt.Sprintf(` LIMIT $%d OFFSET $%d`, argIndex, argIndex+1)
	args = append(args, limit, offset)

	rows, err := r.db.Query(ctx, query, args...)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to list subscriptions: %w", err)
	}
	defer rows.Close()

	var subscriptions []*models.Subscription
	for rows.Next() {
		var sub models.Subscription
		var metadataBytes []byte

		err := rows.Scan(
			&sub.ID,
			&sub.TenantID,
			&sub.StripeSubscriptionID,
			&sub.StripeCustomerID,
			&sub.PlanID,
			&sub.Status,
			&sub.CurrentPeriodStart,
			&sub.CurrentPeriodEnd,
			&sub.CancelAtPeriodEnd,
			&metadataBytes,
			&sub.CreatedAt,
			&sub.UpdatedAt,
		)
		if err != nil {
			return nil, 0, fmt.Errorf("failed to scan subscription: %w", err)
		}

		if err := json.Unmarshal(metadataBytes, &sub.Metadata); err != nil {
			sub.Metadata = make(map[string]string)
		}

		subscriptions = append(subscriptions, &sub)
	}

	return subscriptions, totalCount, nil
}

// CreateInvoice creates an invoice record
func (r *BillingRepository) CreateInvoice(ctx context.Context, invoice *models.Invoice) error {
	invoice.ID = uuid.New().String()
	invoice.CreatedAt = time.Now().UTC()

	query := `
		INSERT INTO invoices (id, tenant_id, stripe_invoice_id, status, amount_due, amount_paid,
			currency, invoice_url, invoice_pdf, due_date, created_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
	`

	_, err := r.db.Exec(ctx, query,
		invoice.ID,
		invoice.TenantID,
		invoice.StripeInvoiceID,
		invoice.Status,
		invoice.AmountDue,
		invoice.AmountPaid,
		invoice.Currency,
		invoice.InvoiceURL,
		invoice.InvoicePDF,
		invoice.DueDate,
		invoice.CreatedAt,
	)

	if err != nil {
		return fmt.Errorf("failed to create invoice: %w", err)
	}

	return nil
}

// ListInvoices lists invoices for a tenant
func (r *BillingRepository) ListInvoices(ctx context.Context, tenantID string, limit, offset int) ([]*models.Invoice, int, error) {
	// Count
	var totalCount int
	if err := r.db.QueryRow(ctx, `SELECT COUNT(*) FROM invoices WHERE tenant_id = $1`, tenantID).Scan(&totalCount); err != nil {
		return nil, 0, fmt.Errorf("failed to count invoices: %w", err)
	}

	query := `
		SELECT id, tenant_id, stripe_invoice_id, status, amount_due, amount_paid,
			currency, invoice_url, invoice_pdf, due_date, created_at
		FROM invoices
		WHERE tenant_id = $1
		ORDER BY created_at DESC
		LIMIT $2 OFFSET $3
	`

	rows, err := r.db.Query(ctx, query, tenantID, limit, offset)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to list invoices: %w", err)
	}
	defer rows.Close()

	var invoices []*models.Invoice
	for rows.Next() {
		var inv models.Invoice
		err := rows.Scan(
			&inv.ID,
			&inv.TenantID,
			&inv.StripeInvoiceID,
			&inv.Status,
			&inv.AmountDue,
			&inv.AmountPaid,
			&inv.Currency,
			&inv.InvoiceURL,
			&inv.InvoicePDF,
			&inv.DueDate,
			&inv.CreatedAt,
		)
		if err != nil {
			return nil, 0, fmt.Errorf("failed to scan invoice: %w", err)
		}
		invoices = append(invoices, &inv)
	}

	return invoices, totalCount, nil
}

// RecordUsage records usage for metered billing
func (r *BillingRepository) RecordUsage(ctx context.Context, tenantID, metricName string, quantity int64) error {
	query := `
		INSERT INTO usage_records (id, tenant_id, metric_name, quantity, timestamp)
		VALUES ($1, $2, $3, $4, $5)
	`

	_, err := r.db.Exec(ctx, query,
		uuid.New().String(),
		tenantID,
		metricName,
		quantity,
		time.Now().UTC(),
	)

	if err != nil {
		return fmt.Errorf("failed to record usage: %w", err)
	}

	return nil
}

// GetUsage retrieves usage records for a tenant
func (r *BillingRepository) GetUsage(ctx context.Context, tenantID, metricName string, startDate, endDate time.Time) ([]*models.UsageRecord, int64, error) {
	query := `
		SELECT id, tenant_id, metric_name, quantity, timestamp
		FROM usage_records
		WHERE tenant_id = $1 AND metric_name = $2 AND timestamp >= $3 AND timestamp <= $4
		ORDER BY timestamp ASC
	`

	rows, err := r.db.Query(ctx, query, tenantID, metricName, startDate, endDate)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to get usage: %w", err)
	}
	defer rows.Close()

	var records []*models.UsageRecord
	var totalUsage int64

	for rows.Next() {
		var record models.UsageRecord
		err := rows.Scan(
			&record.ID,
			&record.TenantID,
			&record.MetricName,
			&record.Quantity,
			&record.Timestamp,
		)
		if err != nil {
			return nil, 0, fmt.Errorf("failed to scan usage record: %w", err)
		}
		records = append(records, &record)
		totalUsage += record.Quantity
	}

	return records, totalUsage, nil
}
