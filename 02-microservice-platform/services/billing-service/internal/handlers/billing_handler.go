package handlers

import (
	"context"
	"errors"

	"github.com/project/microservices/billing-service/internal/models"
	"github.com/project/microservices/billing-service/internal/repository"
	stripeClient "github.com/project/microservices/billing-service/internal/stripe"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// BillingServiceServer implements the gRPC BillingService
type BillingServiceServer struct {
	repo   *repository.BillingRepository
	stripe *stripeClient.Client
	UnimplementedBillingServiceServer
}

// UnimplementedBillingServiceServer is embedded for forward compatibility
type UnimplementedBillingServiceServer struct{}

// NewBillingServiceServer creates a new billing service server
func NewBillingServiceServer(repo *repository.BillingRepository, stripeClient *stripeClient.Client) *BillingServiceServer {
	return &BillingServiceServer{
		repo:   repo,
		stripe: stripeClient,
	}
}

// CreateSubscription creates a new subscription
func (s *BillingServiceServer) CreateSubscription(ctx context.Context, req *CreateSubscriptionRequest) (*CreateSubscriptionResponse, error) {
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}
	if req.PriceId == "" {
		return nil, status.Error(codes.InvalidArgument, "price_id is required")
	}

	// Get or create customer
	customer, err := s.repo.GetCustomerByTenantID(ctx, req.TenantId)
	if err != nil {
		if errors.Is(err, repository.ErrCustomerNotFound) {
			// Create new customer in Stripe
			stripeCust, err := s.stripe.CreateCustomer(ctx, "", "", map[string]string{
				"tenant_id": req.TenantId,
				"user_id":   req.UserId,
			})
			if err != nil {
				return nil, status.Errorf(codes.Internal, "failed to create Stripe customer: %v", err)
			}

			customer = &models.Customer{
				TenantID:         req.TenantId,
				StripeCustomerID: stripeCust.ID,
				Metadata:         map[string]string{"user_id": req.UserId},
			}

			if err := s.repo.CreateCustomer(ctx, customer); err != nil {
				return nil, status.Errorf(codes.Internal, "failed to create customer: %v", err)
			}
		} else {
			return nil, status.Errorf(codes.Internal, "failed to get customer: %v", err)
		}
	}

	// Create subscription in Stripe
	input := &models.CreateSubscriptionInput{
		TenantID:        req.TenantId,
		UserID:          req.UserId,
		PriceID:         req.PriceId,
		PaymentMethodID: req.PaymentMethodId,
		Metadata:        req.Metadata,
	}

	stripeSub, err := s.stripe.CreateSubscription(ctx, input, customer.StripeCustomerID)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to create Stripe subscription: %v", err)
	}

	// Store subscription locally
	sub := stripeClient.SubscriptionToModel(stripeSub, req.TenantId)
	if err := s.repo.CreateSubscription(ctx, sub); err != nil {
		return nil, status.Errorf(codes.Internal, "failed to store subscription: %v", err)
	}

	// Get client secret for payment confirmation if needed
	var clientSecret string
	if stripeSub.LatestInvoice != nil && stripeSub.LatestInvoice.PaymentIntent != nil {
		clientSecret = stripeSub.LatestInvoice.PaymentIntent.ClientSecret
	}

	return &CreateSubscriptionResponse{
		Subscription: subscriptionToProto(sub),
		ClientSecret: clientSecret,
	}, nil
}

// GetSubscription retrieves subscription details
func (s *BillingServiceServer) GetSubscription(ctx context.Context, req *GetSubscriptionRequest) (*GetSubscriptionResponse, error) {
	if req.SubscriptionId == "" {
		return nil, status.Error(codes.InvalidArgument, "subscription_id is required")
	}
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}

	sub, err := s.repo.GetSubscription(ctx, req.SubscriptionId, req.TenantId)
	if err != nil {
		if errors.Is(err, repository.ErrSubscriptionNotFound) {
			return nil, status.Error(codes.NotFound, "subscription not found")
		}
		return nil, status.Errorf(codes.Internal, "failed to get subscription: %v", err)
	}

	return &GetSubscriptionResponse{
		Subscription: subscriptionToProto(sub),
	}, nil
}

// UpdateSubscription updates a subscription (change plan)
func (s *BillingServiceServer) UpdateSubscription(ctx context.Context, req *UpdateSubscriptionRequest) (*UpdateSubscriptionResponse, error) {
	if req.SubscriptionId == "" {
		return nil, status.Error(codes.InvalidArgument, "subscription_id is required")
	}
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}
	if req.NewPriceId == "" {
		return nil, status.Error(codes.InvalidArgument, "new_price_id is required")
	}

	// Get current subscription
	sub, err := s.repo.GetSubscription(ctx, req.SubscriptionId, req.TenantId)
	if err != nil {
		if errors.Is(err, repository.ErrSubscriptionNotFound) {
			return nil, status.Error(codes.NotFound, "subscription not found")
		}
		return nil, status.Errorf(codes.Internal, "failed to get subscription: %v", err)
	}

	// Update in Stripe
	stripeSub, err := s.stripe.UpdateSubscription(ctx, sub.StripeSubscriptionID, req.NewPriceId, req.Prorate)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to update Stripe subscription: %v", err)
	}

	// Update local record
	updatedSub := stripeClient.SubscriptionToModel(stripeSub, req.TenantId)
	updatedSub.ID = sub.ID
	if err := s.repo.UpdateSubscription(ctx, updatedSub); err != nil {
		return nil, status.Errorf(codes.Internal, "failed to update subscription: %v", err)
	}

	return &UpdateSubscriptionResponse{
		Subscription: subscriptionToProto(updatedSub),
	}, nil
}

// CancelSubscription cancels a subscription
func (s *BillingServiceServer) CancelSubscription(ctx context.Context, req *CancelSubscriptionRequest) (*CancelSubscriptionResponse, error) {
	if req.SubscriptionId == "" {
		return nil, status.Error(codes.InvalidArgument, "subscription_id is required")
	}
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}

	// Get current subscription
	sub, err := s.repo.GetSubscription(ctx, req.SubscriptionId, req.TenantId)
	if err != nil {
		if errors.Is(err, repository.ErrSubscriptionNotFound) {
			return nil, status.Error(codes.NotFound, "subscription not found")
		}
		return nil, status.Errorf(codes.Internal, "failed to get subscription: %v", err)
	}

	// Cancel in Stripe
	stripeSub, err := s.stripe.CancelSubscription(ctx, sub.StripeSubscriptionID, req.CancelImmediately)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to cancel Stripe subscription: %v", err)
	}

	// Update local record
	canceledSub := stripeClient.SubscriptionToModel(stripeSub, req.TenantId)
	canceledSub.ID = sub.ID
	if err := s.repo.UpdateSubscription(ctx, canceledSub); err != nil {
		return nil, status.Errorf(codes.Internal, "failed to update subscription: %v", err)
	}

	return &CancelSubscriptionResponse{
		Subscription: subscriptionToProto(canceledSub),
	}, nil
}

// ListSubscriptions lists subscriptions for a tenant
func (s *BillingServiceServer) ListSubscriptions(ctx context.Context, req *ListSubscriptionsRequest) (*ListSubscriptionsResponse, error) {
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}

	pageSize := 20
	offset := 0
	if req.Pagination != nil {
		if req.Pagination.PageSize > 0 {
			pageSize = int(req.Pagination.PageSize)
		}
	}

	subscriptions, totalCount, err := s.repo.ListSubscriptions(ctx, req.TenantId, req.StatusFilter, pageSize, offset)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to list subscriptions: %v", err)
	}

	protoSubs := make([]*Subscription, len(subscriptions))
	for i, sub := range subscriptions {
		protoSubs[i] = subscriptionToProto(sub)
	}

	return &ListSubscriptionsResponse{
		Subscriptions: protoSubs,
		Pagination: &PaginationResponse{
			TotalCount: int32(totalCount),
		},
	}, nil
}

// CreatePaymentMethod adds a payment method
func (s *BillingServiceServer) CreatePaymentMethod(ctx context.Context, req *CreatePaymentMethodRequest) (*CreatePaymentMethodResponse, error) {
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}
	if req.PaymentMethodId == "" {
		return nil, status.Error(codes.InvalidArgument, "payment_method_id is required")
	}

	// Get customer
	customer, err := s.repo.GetCustomerByTenantID(ctx, req.TenantId)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get customer: %v", err)
	}

	// Attach payment method
	pm, err := s.stripe.AttachPaymentMethod(ctx, req.PaymentMethodId, customer.StripeCustomerID)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to attach payment method: %v", err)
	}

	// Set as default if requested
	if req.SetDefault {
		if err := s.stripe.SetDefaultPaymentMethod(ctx, customer.StripeCustomerID, req.PaymentMethodId); err != nil {
			return nil, status.Errorf(codes.Internal, "failed to set default payment method: %v", err)
		}
	}

	return &CreatePaymentMethodResponse{
		PaymentMethod: &PaymentMethod{
			Id:        pm.ID,
			Type:      string(pm.Type),
			LastFour:  pm.Card.Last4,
			Brand:     string(pm.Card.Brand),
			ExpMonth:  int32(pm.Card.ExpMonth),
			ExpYear:   int32(pm.Card.ExpYear),
			IsDefault: req.SetDefault,
		},
	}, nil
}

// ListInvoices lists invoices for a tenant
func (s *BillingServiceServer) ListInvoices(ctx context.Context, req *ListInvoicesRequest) (*ListInvoicesResponse, error) {
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}

	pageSize := 20
	if req.Pagination != nil && req.Pagination.PageSize > 0 {
		pageSize = int(req.Pagination.PageSize)
	}

	invoices, totalCount, err := s.repo.ListInvoices(ctx, req.TenantId, pageSize, 0)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to list invoices: %v", err)
	}

	protoInvoices := make([]*Invoice, len(invoices))
	for i, inv := range invoices {
		protoInvoices[i] = invoiceToProto(inv)
	}

	return &ListInvoicesResponse{
		Invoices: protoInvoices,
		Pagination: &PaginationResponse{
			TotalCount: int32(totalCount),
		},
	}, nil
}

// GetUsage retrieves usage metrics
func (s *BillingServiceServer) GetUsage(ctx context.Context, req *GetUsageRequest) (*GetUsageResponse, error) {
	// Implementation for usage-based billing
	return nil, status.Error(codes.Unimplemented, "not implemented")
}

// Helper functions
func subscriptionToProto(sub *models.Subscription) *Subscription {
	return &Subscription{
		Id:                   sub.ID,
		TenantId:             sub.TenantID,
		StripeSubscriptionId: sub.StripeSubscriptionID,
		StripeCustomerId:     sub.StripeCustomerID,
		PlanId:               sub.PlanID,
		Status:               string(sub.Status),
		CurrentPeriodStart:   sub.CurrentPeriodStart.Format("2006-01-02T15:04:05Z07:00"),
		CurrentPeriodEnd:     sub.CurrentPeriodEnd.Format("2006-01-02T15:04:05Z07:00"),
		CancelAtPeriodEnd:    sub.CancelAtPeriodEnd,
		CreatedAt:            sub.CreatedAt.Format("2006-01-02T15:04:05Z07:00"),
		UpdatedAt:            sub.UpdatedAt.Format("2006-01-02T15:04:05Z07:00"),
		Metadata:             sub.Metadata,
	}
}

func invoiceToProto(inv *models.Invoice) *Invoice {
	return &Invoice{
		Id:              inv.ID,
		TenantId:        inv.TenantID,
		StripeInvoiceId: inv.StripeInvoiceID,
		Status:          inv.Status,
		AmountDue:       inv.AmountDue,
		AmountPaid:      inv.AmountPaid,
		Currency:        inv.Currency,
		InvoiceUrl:      inv.InvoiceURL,
		InvoicePdf:      inv.InvoicePDF,
		DueDate:         inv.DueDate.Format("2006-01-02T15:04:05Z07:00"),
		CreatedAt:       inv.CreatedAt.Format("2006-01-02T15:04:05Z07:00"),
	}
}

// Proto message types (simplified - would normally be generated)
type Subscription struct {
	Id                   string
	TenantId             string
	StripeSubscriptionId string
	StripeCustomerId     string
	PlanId               string
	Status               string
	CurrentPeriodStart   string
	CurrentPeriodEnd     string
	CancelAtPeriodEnd    bool
	CreatedAt            string
	UpdatedAt            string
	Metadata             map[string]string
}

type Invoice struct {
	Id              string
	TenantId        string
	StripeInvoiceId string
	Status          string
	AmountDue       int64
	AmountPaid      int64
	Currency        string
	InvoiceUrl      string
	InvoicePdf      string
	DueDate         string
	CreatedAt       string
}

type PaymentMethod struct {
	Id        string
	Type      string
	LastFour  string
	Brand     string
	ExpMonth  int32
	ExpYear   int32
	IsDefault bool
}

type PaginationRequest struct {
	PageSize  int32
	PageToken string
}

type PaginationResponse struct {
	NextPageToken string
	TotalCount    int32
}

type CreateSubscriptionRequest struct {
	TenantId        string
	UserId          string
	PriceId         string
	PaymentMethodId string
	Metadata        map[string]string
}

type CreateSubscriptionResponse struct {
	Subscription *Subscription
	ClientSecret string
}

type GetSubscriptionRequest struct {
	SubscriptionId string
	TenantId       string
}

type GetSubscriptionResponse struct {
	Subscription *Subscription
}

type UpdateSubscriptionRequest struct {
	SubscriptionId string
	TenantId       string
	NewPriceId     string
	Prorate        bool
}

type UpdateSubscriptionResponse struct {
	Subscription *Subscription
}

type CancelSubscriptionRequest struct {
	SubscriptionId    string
	TenantId          string
	CancelImmediately bool
	Reason            string
}

type CancelSubscriptionResponse struct {
	Subscription *Subscription
}

type ListSubscriptionsRequest struct {
	TenantId     string
	Pagination   *PaginationRequest
	StatusFilter string
}

type ListSubscriptionsResponse struct {
	Subscriptions []*Subscription
	Pagination    *PaginationResponse
}

type CreatePaymentMethodRequest struct {
	TenantId        string
	PaymentMethodId string
	SetDefault      bool
}

type CreatePaymentMethodResponse struct {
	PaymentMethod *PaymentMethod
}

type ListInvoicesRequest struct {
	TenantId   string
	Pagination *PaginationRequest
}

type ListInvoicesResponse struct {
	Invoices   []*Invoice
	Pagination *PaginationResponse
}

type GetUsageRequest struct {
	TenantId   string
	MetricName string
	StartDate  string
	EndDate    string
}

type GetUsageResponse struct {
	Records    []*UsageRecord
	TotalUsage int64
}

type UsageRecord struct {
	Date     string
	Quantity int64
}
