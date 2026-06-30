package notification

import (
	"context"
	"errors"

	notifv1 "github.com/mlai/microservice-platform/pkg/pb/notification/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// Server implements the NotificationService gRPC server
type Server struct {
	notifv1.UnimplementedNotificationServiceServer
	service *Service
}

// NewServer creates a new gRPC server
func NewServer(service *Service) *Server {
	return &Server{service: service}
}

// SendNotification sends a notification
func (s *Server) SendNotification(ctx context.Context, req *notifv1.SendNotificationRequest) (*notifv1.SendNotificationResponse, error) {
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}
	if req.RecipientId == "" {
		return nil, status.Error(codes.InvalidArgument, "recipient_id is required")
	}

	sendReq := &SendNotificationRequest{
		TenantID:    req.TenantId,
		RecipientID: req.RecipientId,
		Channel:     channelToString(req.Channel),
		Variables:   req.Variables,
		Priority:    int(req.Priority),
		Metadata:    req.Metadata,
	}

	// Handle content source
	switch cs := req.ContentSource.(type) {
	case *notifv1.SendNotificationRequest_TemplateId:
		sendReq.TemplateID = cs.TemplateId
	case *notifv1.SendNotificationRequest_Direct:
		sendReq.Subject = cs.Direct.Subject
		sendReq.Content = cs.Direct.Body
	}

	if req.ScheduledAt != nil {
		t := req.ScheduledAt.AsTime()
		sendReq.ScheduledAt = &t
	}

	notif, err := s.service.SendNotification(ctx, sendReq)
	if err != nil {
		return nil, status.Error(codes.Internal, "failed to send notification")
	}

	return &notifv1.SendNotificationResponse{
		NotificationId: notif.ID,
		Status:         stringToDeliveryStatus(notif.Status),
	}, nil
}

// GetNotification retrieves notification status
func (s *Server) GetNotification(ctx context.Context, req *notifv1.GetNotificationRequest) (*notifv1.GetNotificationResponse, error) {
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}
	if req.NotificationId == "" {
		return nil, status.Error(codes.InvalidArgument, "notification_id is required")
	}

	notif, err := s.service.GetNotification(ctx, req.TenantId, req.NotificationId)
	if err != nil {
		if errors.Is(err, ErrNotificationNotFound) {
			return nil, status.Error(codes.NotFound, "notification not found")
		}
		return nil, status.Error(codes.Internal, "failed to get notification")
	}

	return &notifv1.GetNotificationResponse{
		Notification: notificationToProto(notif),
	}, nil
}

// ListNotifications lists notifications for a recipient
func (s *Server) ListNotifications(ctx context.Context, req *notifv1.ListNotificationsRequest) (*notifv1.ListNotificationsResponse, error) {
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}
	if req.RecipientId == "" {
		return nil, status.Error(codes.InvalidArgument, "recipient_id is required")
	}

	pageSize := 20
	offset := 0
	if req.Pagination != nil && req.Pagination.PageSize > 0 {
		pageSize = int(req.Pagination.PageSize)
	}

	notifications, total, err := s.service.ListNotifications(ctx, req.TenantId, req.RecipientId, pageSize, offset)
	if err != nil {
		return nil, status.Error(codes.Internal, "failed to list notifications")
	}

	protoNotifs := make([]*notifv1.Notification, len(notifications))
	for i, n := range notifications {
		protoNotifs[i] = notificationToProto(n)
	}

	return &notifv1.ListNotificationsResponse{
		Notifications: protoNotifs,
		Pagination: &notifv1.ListNotificationsResponse_PaginationResponse{
			TotalCount: int32(total),
			HasMore:    offset+len(notifications) < total,
		},
	}, nil
}

// CreateTemplate creates a notification template
func (s *Server) CreateTemplate(ctx context.Context, req *notifv1.CreateTemplateRequest) (*notifv1.CreateTemplateResponse, error) {
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}
	if req.Name == "" {
		return nil, status.Error(codes.InvalidArgument, "name is required")
	}

	tmpl := &Template{
		TenantID:        req.TenantId,
		Name:            req.Name,
		Description:     req.Description,
		Channel:         channelToString(req.Channel),
		SubjectTemplate: req.SubjectTemplate,
		ContentTemplate: req.ContentTemplate,
		Variables:       req.Variables,
		IsActive:        true,
	}

	if err := s.service.CreateTemplate(ctx, tmpl); err != nil {
		return nil, status.Error(codes.Internal, "failed to create template")
	}

	return &notifv1.CreateTemplateResponse{
		Template: templateToProto(tmpl),
	}, nil
}

// GetTemplate retrieves a template
func (s *Server) GetTemplate(ctx context.Context, req *notifv1.GetTemplateRequest) (*notifv1.GetTemplateResponse, error) {
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}
	if req.TemplateId == "" {
		return nil, status.Error(codes.InvalidArgument, "template_id is required")
	}

	tmpl, err := s.service.GetTemplate(ctx, req.TenantId, req.TemplateId)
	if err != nil {
		if errors.Is(err, ErrTemplateNotFound) {
			return nil, status.Error(codes.NotFound, "template not found")
		}
		return nil, status.Error(codes.Internal, "failed to get template")
	}

	return &notifv1.GetTemplateResponse{
		Template: templateToProto(tmpl),
	}, nil
}

// ListTemplates lists all templates
func (s *Server) ListTemplates(ctx context.Context, req *notifv1.ListTemplatesRequest) (*notifv1.ListTemplatesResponse, error) {
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}

	pageSize := 20
	offset := 0
	if req.Pagination != nil && req.Pagination.PageSize > 0 {
		pageSize = int(req.Pagination.PageSize)
	}

	templates, total, err := s.service.ListTemplates(ctx, req.TenantId, channelToString(req.ChannelFilter), req.ActiveOnly, pageSize, offset)
	if err != nil {
		return nil, status.Error(codes.Internal, "failed to list templates")
	}

	protoTemplates := make([]*notifv1.Template, len(templates))
	for i, t := range templates {
		protoTemplates[i] = templateToProto(t)
	}

	return &notifv1.ListTemplatesResponse{
		Templates: protoTemplates,
		Pagination: &notifv1.ListTemplatesResponse_PaginationResponse{
			TotalCount: int32(total),
			HasMore:    offset+len(templates) < total,
		},
	}, nil
}

// GetPreferences retrieves user notification preferences
func (s *Server) GetPreferences(ctx context.Context, req *notifv1.GetPreferencesRequest) (*notifv1.GetPreferencesResponse, error) {
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}
	if req.UserId == "" {
		return nil, status.Error(codes.InvalidArgument, "user_id is required")
	}

	prefs, err := s.service.GetPreferences(ctx, req.TenantId, req.UserId)
	if err != nil {
		return nil, status.Error(codes.Internal, "failed to get preferences")
	}

	return &notifv1.GetPreferencesResponse{
		Preferences: preferencesToProto(prefs),
	}, nil
}

// UpdatePreferences updates user notification preferences
func (s *Server) UpdatePreferences(ctx context.Context, req *notifv1.UpdatePreferencesRequest) (*notifv1.UpdatePreferencesResponse, error) {
	if req.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id is required")
	}
	if req.UserId == "" {
		return nil, status.Error(codes.InvalidArgument, "user_id is required")
	}

	prefs := &NotificationPreferences{
		TenantID:               req.TenantId,
		UserID:                 req.UserId,
		EmailEnabled:           req.EmailEnabled,
		SMSEnabled:             req.SmsEnabled,
		PushEnabled:            req.PushEnabled,
		InAppEnabled:           req.InAppEnabled,
		UnsubscribedCategories: req.UnsubscribedCategories,
	}

	if err := s.service.UpdatePreferences(ctx, prefs); err != nil {
		return nil, status.Error(codes.Internal, "failed to update preferences")
	}

	return &notifv1.UpdatePreferencesResponse{
		Preferences: preferencesToProto(prefs),
	}, nil
}

// Helper functions

func notificationToProto(n *Notification) *notifv1.Notification {
	proto := &notifv1.Notification{
		Id:           n.ID,
		TenantId:     n.TenantID,
		RecipientId:  n.RecipientID,
		TemplateId:   n.TemplateID,
		Channel:      stringToChannel(n.Channel),
		Subject:      n.Subject,
		Content:      n.Content,
		Status:       stringToDeliveryStatus(n.Status),
		Priority:     notifv1.Priority(n.Priority),
		Variables:    n.Variables,
		Metadata:     n.Metadata,
		ErrorMessage: n.ErrorMessage,
		CreatedAt:    timestamppb.New(n.CreatedAt),
	}

	if n.ScheduledAt != nil {
		proto.ScheduledAt = timestamppb.New(*n.ScheduledAt)
	}
	if n.SentAt != nil {
		proto.SentAt = timestamppb.New(*n.SentAt)
	}
	if n.DeliveredAt != nil {
		proto.DeliveredAt = timestamppb.New(*n.DeliveredAt)
	}

	return proto
}

func templateToProto(t *Template) *notifv1.Template {
	return &notifv1.Template{
		Id:              t.ID,
		TenantId:        t.TenantID,
		Name:            t.Name,
		Description:     t.Description,
		Channel:         stringToChannel(t.Channel),
		SubjectTemplate: t.SubjectTemplate,
		ContentTemplate: t.ContentTemplate,
		Variables:       t.Variables,
		IsActive:        t.IsActive,
		CreatedAt:       timestamppb.New(t.CreatedAt),
		UpdatedAt:       timestamppb.New(t.UpdatedAt),
	}
}

func preferencesToProto(p *NotificationPreferences) *notifv1.NotificationPreferences {
	return &notifv1.NotificationPreferences{
		UserId:                 p.UserID,
		TenantId:               p.TenantID,
		EmailEnabled:           p.EmailEnabled,
		SmsEnabled:             p.SMSEnabled,
		PushEnabled:            p.PushEnabled,
		InAppEnabled:           p.InAppEnabled,
		UnsubscribedCategories: p.UnsubscribedCategories,
		UpdatedAt:              timestamppb.New(p.UpdatedAt),
	}
}

func channelToString(c notifv1.Channel) string {
	switch c {
	case notifv1.Channel_CHANNEL_EMAIL:
		return "email"
	case notifv1.Channel_CHANNEL_SMS:
		return "sms"
	case notifv1.Channel_CHANNEL_PUSH:
		return "push"
	case notifv1.Channel_CHANNEL_WEBHOOK:
		return "webhook"
	case notifv1.Channel_CHANNEL_IN_APP:
		return "in_app"
	default:
		return "email"
	}
}

func stringToChannel(s string) notifv1.Channel {
	switch s {
	case "email":
		return notifv1.Channel_CHANNEL_EMAIL
	case "sms":
		return notifv1.Channel_CHANNEL_SMS
	case "push":
		return notifv1.Channel_CHANNEL_PUSH
	case "webhook":
		return notifv1.Channel_CHANNEL_WEBHOOK
	case "in_app":
		return notifv1.Channel_CHANNEL_IN_APP
	default:
		return notifv1.Channel_CHANNEL_UNSPECIFIED
	}
}

func stringToDeliveryStatus(s string) notifv1.DeliveryStatus {
	switch s {
	case "pending":
		return notifv1.DeliveryStatus_DELIVERY_STATUS_PENDING
	case "sent":
		return notifv1.DeliveryStatus_DELIVERY_STATUS_SENT
	case "delivered":
		return notifv1.DeliveryStatus_DELIVERY_STATUS_DELIVERED
	case "failed":
		return notifv1.DeliveryStatus_DELIVERY_STATUS_FAILED
	case "bounced":
		return notifv1.DeliveryStatus_DELIVERY_STATUS_BOUNCED
	case "opened":
		return notifv1.DeliveryStatus_DELIVERY_STATUS_OPENED
	case "clicked":
		return notifv1.DeliveryStatus_DELIVERY_STATUS_CLICKED
	default:
		return notifv1.DeliveryStatus_DELIVERY_STATUS_UNSPECIFIED
	}
}
