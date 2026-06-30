package billing

import (
	"context"
	"errors"

	billingv1 "github.com/mlai/microservice-platform/pkg/pb/billing/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// Server implements the BillingService gRPC server
type Server struct {
	billingv1.UnimplementedBillingServiceServer
	service *Service
}

// NewServer creates a new gRPC server
func NewServer(service *Service) *Server {
	return &Server{service: service}
}

// CreateCustomer creates a billing customer
func (s *Server) CreateCustomer(ctx context.Context, req *billingv1.CreateCustomerRequest) (*billingv1.CreateCustomerResponse, error) {
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}
	if req.Email == "" {
		return nil, status.Error(codes.InvalidArgument, "email is required")
	}

	var addr *billingv1.Address
	if req.Address != nil {
		addr = req.Address
	} else {
		addr = &billingv1.Address{}
	}

	cust, err := s.service.CreateCustomer(ctx, &CreateCustomerRequest{
		TenantID:       req.TenantId,
		UserID:         req.UserId,
		Email:          req.Email,
		Name:           req.Name,
		Phone:          req.Phone,
		AddressLine1:   addr.Line1,
		AddressLine2:   addr.Line2,
		AddressCity:    addr.City,
		AddressState:   addr.State,
		AddressPostal:  addr.PostalCode,
		AddressCountry: addr.Country,
		Metadata:       req.Metadata,
	})
	if err != nil {
		return nil, status.Error(codes.Internal, "failed to create customer")
	}

	return &billingv1.CreateCustomerResponse{
		Customer: customerToProto(cust),
	}, nil
}

// GetCustomer retrieves customer billing info
func (s *Server) GetCustomer(ctx context.Context, req *billingv1.GetCustomerRequest) (*billingv1.GetCustomerResponse, error) {
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}
	if req.CustomerId == "" {
		return nil, status.Error(codes.InvalidArgument, "customer_id is required")
	}

	cust, err := s.service.GetCustomer(ctx, req.TenantId, req.CustomerId)
	if err != nil {
		if errors.Is(err, ErrCustomerNotFound) {
			return nil, status.Error(codes.NotFound, "customer not found")
		}
		return nil, status.Error(codes.Internal, "failed to get customer")
	}

	return &billingv1.GetCustomerResponse{
		Customer: customerToProto(cust),
	}, nil
}

// CreateSubscription creates a new subscription
func (s *Server) CreateSubscription(ctx context.Context, req *billingv1.CreateSubscriptionRequest) (*billingv1.CreateSubscriptionResponse, error) {
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}
	if req.CustomerId == "" {
		return nil, status.Error(codes.InvalidArgument, "customer_id is required")
	}
	if req.PriceId == "" {
		return nil, status.Error(codes.InvalidArgument, "price_id is required")
	}

	sub, clientSecret, err := s.service.CreateSubscription(ctx, &CreateSubscriptionRequest{
		TenantID:        req.TenantId,
		CustomerID:      req.CustomerId,
		PriceID:         req.PriceId,
		Quantity:        req.Quantity,
		PaymentMethodID: req.PaymentMethodId,
		Metadata:        req.Metadata,
	})
	if err != nil {
		if errors.Is(err, ErrCustomerNotFound) {
			return nil, status.Error(codes.NotFound, "customer not found")
		}
		return nil, status.Error(codes.Internal, "failed to create subscription")
	}

	return &billingv1.CreateSubscriptionResponse{
		Subscription: subscriptionToProto(sub),
		ClientSecret: clientSecret,
	}, nil
}

// GetSubscription retrieves subscription details
func (s *Server) GetSubscription(ctx context.Context, req *billingv1.GetSubscriptionRequest) (*billingv1.GetSubscriptionResponse, error) {
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}
	if req.SubscriptionId == "" {
		return nil, status.Error(codes.InvalidArgument, "subscription_id is required")
	}

	sub, err := s.service.GetSubscription(ctx, req.TenantId, req.SubscriptionId)
	if err != nil {
		if errors.Is(err, ErrSubscriptionNotFound) {
			return nil, status.Error(codes.NotFound, "subscription not found")
		}
		return nil, status.Error(codes.Internal, "failed to get subscription")
	}

	return &billingv1.GetSubscriptionResponse{
		Subscription: subscriptionToProto(sub),
	}, nil
}

// CancelSubscription cancels a subscription
func (s *Server) CancelSubscription(ctx context.Context, req *billingv1.CancelSubscriptionRequest) (*billingv1.CancelSubscriptionResponse, error) {
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}
	if req.SubscriptionId == "" {
		return nil, status.Error(codes.InvalidArgument, "subscription_id is required")
	}

	sub, err := s.service.CancelSubscription(ctx, req.TenantId, req.SubscriptionId, req.CancelAtPeriodEnd, req.Reason)
	if err != nil {
		if errors.Is(err, ErrSubscriptionNotFound) {
			return nil, status.Error(codes.NotFound, "subscription not found")
		}
		return nil, status.Error(codes.Internal, "failed to cancel subscription")
	}

	return &billingv1.CancelSubscriptionResponse{
		Subscription: subscriptionToProto(sub),
	}, nil
}

// ListSubscriptions lists all subscriptions for a customer
func (s *Server) ListSubscriptions(ctx context.Context, req *billingv1.ListSubscriptionsRequest) (*billingv1.ListSubscriptionsResponse, error) {
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}
	if req.CustomerId == "" {
		return nil, status.Error(codes.InvalidArgument, "customer_id is required")
	}

	pageSize := 20
	offset := 0
	if req.Pagination != nil && req.Pagination.PageSize > 0 {
		pageSize = int(req.Pagination.PageSize)
	}

	subs, total, err := s.service.ListSubscriptions(ctx, req.TenantId, req.CustomerId, pageSize, offset)
	if err != nil {
		return nil, status.Error(codes.Internal, "failed to list subscriptions")
	}

	protoSubs := make([]*billingv1.Subscription, len(subs))
	for i, sub := range subs {
		protoSubs[i] = subscriptionToProto(sub)
	}

	return &billingv1.ListSubscriptionsResponse{
		Subscriptions: protoSubs,
		Pagination: &billingv1.ListSubscriptionsResponse_PaginationResponse{
			TotalCount: int32(total),
			HasMore:    offset+len(subs) < total,
		},
	}, nil
}

// CreatePaymentMethod adds a payment method
func (s *Server) CreatePaymentMethod(ctx context.Context, req *billingv1.CreatePaymentMethodRequest) (*billingv1.CreatePaymentMethodResponse, error) {
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}
	if req.CustomerId == "" {
		return nil, status.Error(codes.InvalidArgument, "customer_id is required")
	}
	if req.PaymentMethodId == "" {
		return nil, status.Error(codes.InvalidArgument, "payment_method_id is required")
	}

	err := s.service.AddPaymentMethod(ctx, req.TenantId, req.CustomerId, req.PaymentMethodId, req.SetAsDefault)
	if err != nil {
		if errors.Is(err, ErrCustomerNotFound) {
			return nil, status.Error(codes.NotFound, "customer not found")
		}
		return nil, status.Error(codes.Internal, "failed to add payment method")
	}

	return &billingv1.CreatePaymentMethodResponse{
		PaymentMethod: &billingv1.PaymentMethod{
			Id:                     req.PaymentMethodId,
			StripePaymentMethodId: req.PaymentMethodId,
			IsDefault:              req.SetAsDefault,
		},
	}, nil
}

// ListInvoices lists customer invoices
func (s *Server) ListInvoices(ctx context.Context, req *billingv1.ListInvoicesRequest) (*billingv1.ListInvoicesResponse, error) {
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}
	if req.CustomerId == "" {
		return nil, status.Error(codes.InvalidArgument, "customer_id is required")
	}

	pageSize := 20
	offset := 0
	if req.Pagination != nil && req.Pagination.PageSize > 0 {
		pageSize = int(req.Pagination.PageSize)
	}

	invoices, total, err := s.service.ListInvoices(ctx, req.TenantId, req.CustomerId, pageSize, offset)
	if err != nil {
		return nil, status.Error(codes.Internal, "failed to list invoices")
	}

	protoInvoices := make([]*billingv1.Invoice, len(invoices))
	for i, inv := range invoices {
		protoInvoices[i] = invoiceToProto(inv)
	}

	return &billingv1.ListInvoicesResponse{
		Invoices: protoInvoices,
		Pagination: &billingv1.ListInvoicesResponse_PaginationResponse{
			TotalCount: int32(total),
			HasMore:    offset+len(invoices) < total,
		},
	}, nil
}

// Helper functions

func customerToProto(c *Customer) *billingv1.Customer {
	return &billingv1.Customer{
		Id:               c.ID,
		TenantId:         c.TenantID,
		StripeCustomerId: c.StripeCustomerID,
		Email:            c.Email,
		Name:             c.Name,
		Phone:            c.Phone,
		Address: &billingv1.Address{
			Line1:      c.AddressLine1,
			Line2:      c.AddressLine2,
			City:       c.AddressCity,
			State:      c.AddressState,
			PostalCode: c.AddressPostal,
			Country:    c.AddressCountry,
		},
		Metadata:  c.Metadata,
		CreatedAt: timestamppb.New(c.CreatedAt),
	}
}

func subscriptionToProto(s *Subscription) *billingv1.Subscription {
	proto := &billingv1.Subscription{
		Id:                     s.ID,
		TenantId:               s.TenantID,
		CustomerId:             s.CustomerID,
		StripeSubscriptionId:   s.StripeSubscriptionID,
		Status:                 stringToSubStatus(s.Status),
		PriceId:                s.PriceID,
		PlanName:               s.PlanName,
		Quantity:               s.Quantity,
		CurrentPeriodStart:     timestamppb.New(s.CurrentPeriodStart),
		CurrentPeriodEnd:       timestamppb.New(s.CurrentPeriodEnd),
		CancelAtPeriodEnd:      s.CancelAtPeriodEnd,
		CreatedAt:              timestamppb.New(s.CreatedAt),
	}

	if s.CancelAt != nil {
		proto.CancelAt = timestamppb.New(*s.CancelAt)
	}

	return proto
}

func invoiceToProto(i *Invoice) *billingv1.Invoice {
	proto := &billingv1.Invoice{
		Id:               i.ID,
		StripeInvoiceId: i.StripeInvoiceID,
		CustomerId:       i.CustomerID,
		SubscriptionId:   i.SubscriptionID,
		Status:           stringToInvStatus(i.Status),
		AmountDue:        i.AmountDue,
		AmountPaid:       i.AmountPaid,
		AmountRemaining:  i.AmountRemaining,
		Currency:         i.Currency,
		InvoicePdf:       i.InvoicePDF,
		HostedInvoiceUrl: i.HostedInvoiceURL,
		PeriodStart:      timestamppb.New(i.PeriodStart),
		PeriodEnd:        timestamppb.New(i.PeriodEnd),
		CreatedAt:        timestamppb.New(i.CreatedAt),
	}

	if i.DueDate != nil {
		proto.DueDate = timestamppb.New(*i.DueDate)
	}

	return proto
}

func stringToSubStatus(s string) billingv1.SubscriptionStatus {
	switch s {
	case "active":
		return billingv1.SubscriptionStatus_SUBSCRIPTION_STATUS_ACTIVE
	case "past_due":
		return billingv1.SubscriptionStatus_SUBSCRIPTION_STATUS_PAST_DUE
	case "unpaid":
		return billingv1.SubscriptionStatus_SUBSCRIPTION_STATUS_UNPAID
	case "canceled":
		return billingv1.SubscriptionStatus_SUBSCRIPTION_STATUS_CANCELED
	case "incomplete":
		return billingv1.SubscriptionStatus_SUBSCRIPTION_STATUS_INCOMPLETE
	case "trialing":
		return billingv1.SubscriptionStatus_SUBSCRIPTION_STATUS_TRIALING
	default:
		return billingv1.SubscriptionStatus_SUBSCRIPTION_STATUS_UNSPECIFIED
	}
}

func stringToInvStatus(s string) billingv1.InvoiceStatus {
	switch s {
	case "draft":
		return billingv1.InvoiceStatus_INVOICE_STATUS_DRAFT
	case "open":
		return billingv1.InvoiceStatus_INVOICE_STATUS_OPEN
	case "paid":
		return billingv1.InvoiceStatus_INVOICE_STATUS_PAID
	case "uncollectible":
		return billingv1.InvoiceStatus_INVOICE_STATUS_UNCOLLECTIBLE
	case "void":
		return billingv1.InvoiceStatus_INVOICE_STATUS_VOID
	default:
		return billingv1.InvoiceStatus_INVOICE_STATUS_UNSPECIFIED
	}
}
