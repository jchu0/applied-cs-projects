# Project 2: Microservice Platform (Auth + Billing + Notifications + Users)

## Staff-Level Design Document

**Complexity:** ⭐⭐⭐⭐⭐ (Expert)
**Timeline:** 10-12 weeks
**Languages:** Go (primary), TypeScript (gateway), Python (notifications)

> **Concepts covered:** [§05 Authentication (JWT/OIDC/MFA)](../../05-cross-cutting-concerns/security/authentication/authentication.md) · [§05 Secrets management](../../05-cross-cutting-concerns/security/secrets-management/secrets-management.md) · [§01 Go concurrency / gRPC](../../01-software-engineering/go/) · [§07 Kubernetes](../../07-infrastructure/kubernetes/kubernetes-guide.md). Pairs with [Project 13 (service mesh — the mTLS / traffic layer)](../13-service-mesh/) and [Project 05 (SaaS — full-stack consumer)](../05-saas-web-platform/). Map: [`CONCEPT_TO_PROJECT_MAP.md`](../CONCEPT_TO_PROJECT_MAP.md).

---

## What This Project Teaches

### Core Concepts
- **Service decomposition** - Domain-driven design, bounded contexts, service boundaries
- **Inter-service communication** - gRPC, protobuf IDLs, async messaging patterns
- **API gateway patterns** - Rate limiting, authentication, request routing, aggregation
- **Distributed observability** - Distributed tracing, metrics aggregation, log correlation
- **Multi-tenant architecture** - Data isolation, tenant-aware routing, resource quotas
- **Event-driven architecture** - Event sourcing, CQRS, eventual consistency
- **Security patterns** - JWT/OIDC, mTLS, zero-trust networking

### Industry Relevance
This mirrors how companies like Stripe, Auth0, and Twilio build their platforms. Understanding microservice patterns is essential for scaling organizations and systems independently.

---

## High-Level Architecture

```
                                    ┌─────────────────┐
                                    │   Load Balancer │
                                    └────────┬────────┘
                                             │
                                    ┌────────▼────────┐
                                    │   API Gateway   │
                                    │  (Kong/Envoy)   │
                                    └────────┬────────┘
                                             │
                 ┌───────────────────────────┼───────────────────────────┐
                 │                           │                           │
        ┌────────▼────────┐        ┌─────────▼────────┐        ┌────────▼────────┐
        │   Auth Service  │        │   User Service   │        │ Billing Service │
        │    (gRPC/REST)  │        │     (gRPC)       │        │   (gRPC/REST)   │
        └────────┬────────┘        └─────────┬────────┘        └────────┬────────┘
                 │                           │                           │
                 │                  ┌────────▼────────┐                   │
                 │                  │    Postgres     │                   │
                 │                  │   (per-service) │                   │
                 │                  └─────────────────┘                   │
                 │                                                        │
        ┌────────▼────────┐                                    ┌─────────▼────────┐
        │     Redis       │                                    │     Stripe       │
        │  (sessions/rate)│                                    │   (external)     │
        └─────────────────┘                                    └──────────────────┘
                                             │
                                    ┌────────▼────────┐
                                    │  Message Broker │
                                    │ (NATS/RabbitMQ) │
                                    └────────┬────────┘
                                             │
                                    ┌────────▼────────┐
                                    │  Notification   │
                                    │    Service      │
                                    └─────────────────┘
```

### Service Breakdown

#### 1. API Gateway
**Responsibilities:**
- Request routing and load balancing
- Authentication/authorization
- Rate limiting and throttling
- Request/response transformation
- API versioning
- Circuit breaking

**Configuration:**
```yaml
# kong.yaml or envoy.yaml equivalent
services:
  - name: auth-service
    url: http://auth:8080
    routes:
      - paths: ["/api/v1/auth/*"]
    plugins:
      - name: rate-limiting
        config:
          minute: 100
          policy: redis
      - name: jwt
        config:
          claims_to_verify: ["exp", "nbf"]

  - name: user-service
    url: grpc://users:9090
    routes:
      - paths: ["/api/v1/users/*"]
    plugins:
      - name: grpc-gateway
      - name: request-transformer
```

#### 2. Auth Service
**Responsibilities:**
- User authentication (password, OAuth, SSO)
- JWT token issuance and validation
- OIDC provider implementation
- Session management
- MFA/2FA support

**API Surface:**
```protobuf
syntax = "proto3";
package auth.v1;

service AuthService {
  rpc Login(LoginRequest) returns (LoginResponse);
  rpc Logout(LogoutRequest) returns (LogoutResponse);
  rpc RefreshToken(RefreshRequest) returns (TokenResponse);
  rpc ValidateToken(ValidateRequest) returns (ValidateResponse);
  rpc InitiateMFA(MFARequest) returns (MFAChallenge);
  rpc VerifyMFA(MFAVerifyRequest) returns (TokenResponse);
}

message LoginRequest {
  string email = 1;
  string password = 2;
  string tenant_id = 3;
  map<string, string> metadata = 4;
}

message TokenResponse {
  string access_token = 1;
  string refresh_token = 2;
  int64 expires_in = 3;
  string token_type = 4;
  repeated string scopes = 5;
}
```

#### 3. User Service
**Responsibilities:**
- User CRUD operations
- Profile management
- Role and permission assignment
- User search and filtering
- Tenant user management

**Data Model:**
```sql
-- Users table with multi-tenant support
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    email VARCHAR(255) NOT NULL,
    email_verified BOOLEAN DEFAULT FALSE,
    password_hash VARCHAR(255),
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    avatar_url TEXT,
    status VARCHAR(20) DEFAULT 'active',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(tenant_id, email)
);

-- Row-level security for multi-tenancy
ALTER TABLE users ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation ON users
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

-- Roles and permissions
CREATE TABLE roles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    name VARCHAR(100) NOT NULL,
    permissions JSONB NOT NULL DEFAULT '[]',
    UNIQUE(tenant_id, name)
);

CREATE TABLE user_roles (
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    role_id UUID REFERENCES roles(id) ON DELETE CASCADE,
    granted_at TIMESTAMPTZ DEFAULT NOW(),
    granted_by UUID REFERENCES users(id),
    PRIMARY KEY (user_id, role_id)
);
```

#### 4. Billing Service
**Responsibilities:**
- Subscription management
- Payment processing (Stripe integration)
- Invoice generation
- Usage metering
- Plan and pricing management

**Stripe Integration:**
```go
type BillingService struct {
    stripe     *stripe.Client
    db         *sql.DB
    eventBus   EventBus
}

func (s *BillingService) CreateSubscription(ctx context.Context, req *CreateSubscriptionRequest) (*Subscription, error) {
    // Get or create Stripe customer
    customer, err := s.getOrCreateCustomer(ctx, req.TenantID, req.UserID)
    if err != nil {
        return nil, fmt.Errorf("customer lookup failed: %w", err)
    }

    // Create subscription in Stripe
    params := &stripe.SubscriptionParams{
        Customer: stripe.String(customer.ID),
        Items: []*stripe.SubscriptionItemsParams{
            {Price: stripe.String(req.PriceID)},
        },
        PaymentBehavior:      stripe.String("default_incomplete"),
        PaymentSettings: &stripe.SubscriptionPaymentSettingsParams{
            SaveDefaultPaymentMethod: stripe.String("on_subscription"),
        },
        Metadata: map[string]string{
            "tenant_id": req.TenantID,
            "user_id":   req.UserID,
        },
    }

    stripeSub, err := subscription.New(params)
    if err != nil {
        return nil, fmt.Errorf("stripe subscription failed: %w", err)
    }

    // Store locally
    sub := &Subscription{
        ID:               uuid.New().String(),
        TenantID:         req.TenantID,
        StripeSubID:      stripeSub.ID,
        Status:           string(stripeSub.Status),
        CurrentPeriodEnd: time.Unix(stripeSub.CurrentPeriodEnd, 0),
    }

    if err := s.db.CreateSubscription(ctx, sub); err != nil {
        return nil, err
    }

    // Emit event
    s.eventBus.Publish("billing.subscription.created", sub)

    return sub, nil
}
```

#### 5. Notification Service
**Responsibilities:**
- Multi-channel delivery (email, SMS, push, webhook)
- Template management
- Delivery tracking
- Preference management
- Batch/bulk sending

**Architecture:**
```python
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
import asyncio

class Channel(Enum):
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"
    IN_APP = "in_app"

@dataclass
class Notification:
    id: str
    tenant_id: str
    recipient_id: str
    template_id: str
    channel: Channel
    variables: Dict[str, str]
    priority: int = 5
    scheduled_at: Optional[datetime] = None
    metadata: Dict = None

class NotificationService:
    def __init__(self):
        self.providers = {
            Channel.EMAIL: SendGridProvider(),
            Channel.SMS: TwilioProvider(),
            Channel.PUSH: FCMProvider(),
            Channel.WEBHOOK: WebhookProvider(),
        }
        self.template_engine = JinjaTemplateEngine()

    async def send(self, notification: Notification) -> DeliveryResult:
        # Check user preferences
        if not await self.check_preferences(notification):
            return DeliveryResult(status="opted_out")

        # Render template
        content = await self.template_engine.render(
            notification.template_id,
            notification.variables
        )

        # Get provider and send
        provider = self.providers[notification.channel]
        result = await provider.send(notification.recipient_id, content)

        # Track delivery
        await self.track_delivery(notification.id, result)

        return result
```

---

## Core Internals

### Inter-Service Communication

#### gRPC Service Mesh
```protobuf
// Common types shared across services
syntax = "proto3";
package common.v1;

message TenantContext {
    string tenant_id = 1;
    string user_id = 2;
    string trace_id = 3;
    string request_id = 4;
    map<string, string> metadata = 5;
}

message PaginationRequest {
    int32 page_size = 1;
    string page_token = 2;
}

message PaginationResponse {
    string next_page_token = 1;
    int32 total_count = 2;
}
```

#### Event Bus Schema
```json
{
  "event_id": "evt_123456789",
  "event_type": "user.created",
  "tenant_id": "tenant_abc",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0",
  "data": {
    "user_id": "usr_xyz",
    "email": "user@example.com",
    "source": "registration"
  },
  "metadata": {
    "trace_id": "trace_abc123",
    "source_service": "user-service",
    "correlation_id": "corr_789"
  }
}
```

### Authentication Flow

```
┌────────┐     ┌─────────┐     ┌──────────┐     ┌───────────┐
│ Client │────▶│ Gateway │────▶│   Auth   │────▶│   User    │
└───┬────┘     └────┬────┘     └────┬─────┘     └─────┬─────┘
    │               │               │                  │
    │  1. Login     │               │                  │
    │──────────────▶│               │                  │
    │               │ 2. Forward    │                  │
    │               │──────────────▶│                  │
    │               │               │ 3. Verify user   │
    │               │               │─────────────────▶│
    │               │               │                  │
    │               │               │ 4. User data     │
    │               │               │◀─────────────────│
    │               │               │                  │
    │               │ 5. JWT tokens │                  │
    │               │◀──────────────│                  │
    │  6. Tokens    │               │                  │
    │◀──────────────│               │                  │
    │               │               │                  │
    │  7. API call  │               │                  │
    │  + JWT        │               │                  │
    │──────────────▶│               │                  │
    │               │ 8. Validate   │                  │
    │               │──────────────▶│                  │
    │               │               │                  │
    │               │ 9. Claims     │                  │
    │               │◀──────────────│                  │
    │               │               │                  │
```

### JWT Token Structure
```go
type TokenClaims struct {
    jwt.RegisteredClaims
    UserID      string   `json:"uid"`
    TenantID    string   `json:"tid"`
    Email       string   `json:"email"`
    Roles       []string `json:"roles"`
    Permissions []string `json:"perms"`
    SessionID   string   `json:"sid"`
    TokenType   string   `json:"type"` // access, refresh
}

func (s *AuthService) GenerateTokenPair(user *User, session *Session) (*TokenPair, error) {
    now := time.Now()

    // Access token (short-lived)
    accessClaims := TokenClaims{
        RegisteredClaims: jwt.RegisteredClaims{
            Issuer:    s.issuer,
            Subject:   user.ID,
            Audience:  []string{s.audience},
            ExpiresAt: jwt.NewNumericDate(now.Add(15 * time.Minute)),
            IssuedAt:  jwt.NewNumericDate(now),
            ID:        uuid.New().String(),
        },
        UserID:      user.ID,
        TenantID:    user.TenantID,
        Email:       user.Email,
        Roles:       user.Roles,
        Permissions: user.Permissions,
        SessionID:   session.ID,
        TokenType:   "access",
    }

    accessToken, err := jwt.NewWithClaims(jwt.SigningMethodRS256, accessClaims).
        SignedString(s.privateKey)
    if err != nil {
        return nil, err
    }

    // Refresh token (long-lived)
    refreshClaims := TokenClaims{
        RegisteredClaims: jwt.RegisteredClaims{
            ExpiresAt: jwt.NewNumericDate(now.Add(7 * 24 * time.Hour)),
            IssuedAt:  jwt.NewNumericDate(now),
            ID:        uuid.New().String(),
        },
        UserID:    user.ID,
        TenantID:  user.TenantID,
        SessionID: session.ID,
        TokenType: "refresh",
    }

    refreshToken, err := jwt.NewWithClaims(jwt.SigningMethodRS256, refreshClaims).
        SignedString(s.privateKey)

    return &TokenPair{
        AccessToken:  accessToken,
        RefreshToken: refreshToken,
        ExpiresIn:    900, // 15 minutes
    }, nil
}
```

### Multi-Tenant Data Isolation

```go
// Middleware to inject tenant context
func TenantContextMiddleware() grpc.UnaryServerInterceptor {
    return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
        md, ok := metadata.FromIncomingContext(ctx)
        if !ok {
            return nil, status.Error(codes.Unauthenticated, "missing metadata")
        }

        tenantID := md.Get("x-tenant-id")
        if len(tenantID) == 0 {
            return nil, status.Error(codes.InvalidArgument, "missing tenant ID")
        }

        // Add to context
        ctx = context.WithValue(ctx, TenantIDKey, tenantID[0])

        // Set PostgreSQL session variable for RLS
        db := GetDB(ctx)
        _, err := db.ExecContext(ctx,
            "SET app.current_tenant = $1", tenantID[0])
        if err != nil {
            return nil, status.Error(codes.Internal, "failed to set tenant context")
        }

        return handler(ctx, req)
    }
}

// Repository with tenant isolation
type UserRepository struct {
    db *sql.DB
}

func (r *UserRepository) FindByID(ctx context.Context, id string) (*User, error) {
    tenantID := ctx.Value(TenantIDKey).(string)

    query := `
        SELECT id, email, first_name, last_name, status
        FROM users
        WHERE id = $1 AND tenant_id = $2
    `
    // RLS provides additional safety layer
    row := r.db.QueryRowContext(ctx, query, id, tenantID)
    // ...
}
```

---

## Enterprise Features

### 1. Audit Logging

```go
type AuditEvent struct {
    ID          string                 `json:"id"`
    Timestamp   time.Time              `json:"timestamp"`
    TenantID    string                 `json:"tenant_id"`
    ActorID     string                 `json:"actor_id"`
    ActorType   string                 `json:"actor_type"` // user, service, system
    Action      string                 `json:"action"`
    Resource    string                 `json:"resource"`
    ResourceID  string                 `json:"resource_id"`
    Status      string                 `json:"status"` // success, failure
    IPAddress   string                 `json:"ip_address"`
    UserAgent   string                 `json:"user_agent"`
    Changes     map[string]interface{} `json:"changes,omitempty"`
    Metadata    map[string]string      `json:"metadata"`
}

type AuditLogger struct {
    writer AuditWriter // Elasticsearch, S3, etc.
}

func (l *AuditLogger) Log(ctx context.Context, event AuditEvent) error {
    // Enrich with context
    event.ID = uuid.New().String()
    event.Timestamp = time.Now().UTC()

    if traceID := ctx.Value(TraceIDKey); traceID != nil {
        event.Metadata["trace_id"] = traceID.(string)
    }

    // Async write
    return l.writer.Write(event)
}

// Usage in service
func (s *UserService) UpdateUser(ctx context.Context, req *UpdateUserRequest) (*User, error) {
    oldUser, _ := s.repo.FindByID(ctx, req.UserID)

    user, err := s.repo.Update(ctx, req)
    if err != nil {
        s.audit.Log(ctx, AuditEvent{
            Action:     "user.update",
            Resource:   "user",
            ResourceID: req.UserID,
            Status:     "failure",
            Metadata:   map[string]string{"error": err.Error()},
        })
        return nil, err
    }

    s.audit.Log(ctx, AuditEvent{
        Action:     "user.update",
        Resource:   "user",
        ResourceID: user.ID,
        Status:     "success",
        Changes:    computeDiff(oldUser, user),
    })

    return user, nil
}
```

### 2. Rate Limiting

```go
type RateLimiter struct {
    redis   *redis.Client
    configs map[string]RateLimitConfig
}

type RateLimitConfig struct {
    Requests int           // Number of requests
    Window   time.Duration // Time window
    Burst    int           // Burst allowance
}

func (r *RateLimiter) Allow(ctx context.Context, key string, config RateLimitConfig) (bool, error) {
    // Sliding window rate limiting using Redis
    now := time.Now().UnixMicro()
    windowStart := now - config.Window.Microseconds()

    pipe := r.redis.Pipeline()

    // Remove old entries
    pipe.ZRemRangeByScore(ctx, key, "0", fmt.Sprintf("%d", windowStart))

    // Count current entries
    countCmd := pipe.ZCard(ctx, key)

    // Add current request
    pipe.ZAdd(ctx, key, redis.Z{Score: float64(now), Member: now})

    // Set expiry
    pipe.Expire(ctx, key, config.Window)

    _, err := pipe.Exec(ctx)
    if err != nil {
        return false, err
    }

    count := countCmd.Val()
    return count < int64(config.Requests), nil
}

// Middleware
func RateLimitMiddleware(limiter *RateLimiter) gin.HandlerFunc {
    return func(c *gin.Context) {
        // Build key: tenant + user + endpoint
        key := fmt.Sprintf("ratelimit:%s:%s:%s",
            c.GetString("tenant_id"),
            c.GetString("user_id"),
            c.FullPath())

        allowed, err := limiter.Allow(c.Request.Context(), key, RateLimitConfig{
            Requests: 100,
            Window:   time.Minute,
        })

        if err != nil || !allowed {
            c.Header("X-RateLimit-Remaining", "0")
            c.Header("Retry-After", "60")
            c.AbortWithStatusJSON(429, gin.H{"error": "rate limit exceeded"})
            return
        }

        c.Next()
    }
}
```

### 3. Circuit Breaking

```go
type CircuitBreaker struct {
    name          string
    maxFailures   int
    timeout       time.Duration
    halfOpenMax   int

    mu            sync.Mutex
    state         CircuitState
    failures      int
    successes     int
    lastFailure   time.Time
}

type CircuitState int

const (
    StateClosed CircuitState = iota
    StateOpen
    StateHalfOpen
)

func (cb *CircuitBreaker) Execute(fn func() error) error {
    cb.mu.Lock()

    if cb.state == StateOpen {
        if time.Since(cb.lastFailure) > cb.timeout {
            cb.state = StateHalfOpen
            cb.successes = 0
        } else {
            cb.mu.Unlock()
            return ErrCircuitOpen
        }
    }

    cb.mu.Unlock()

    err := fn()

    cb.mu.Lock()
    defer cb.mu.Unlock()

    if err != nil {
        cb.failures++
        cb.lastFailure = time.Now()

        if cb.state == StateHalfOpen || cb.failures >= cb.maxFailures {
            cb.state = StateOpen
        }
        return err
    }

    if cb.state == StateHalfOpen {
        cb.successes++
        if cb.successes >= cb.halfOpenMax {
            cb.state = StateClosed
            cb.failures = 0
        }
    } else {
        cb.failures = 0
    }

    return nil
}

// Usage with gRPC client
func (c *BillingClient) CreateSubscription(ctx context.Context, req *CreateSubRequest) (*Subscription, error) {
    var resp *Subscription

    err := c.breaker.Execute(func() error {
        var err error
        resp, err = c.client.CreateSubscription(ctx, req)
        return err
    })

    if errors.Is(err, ErrCircuitOpen) {
        // Fallback behavior
        return nil, status.Error(codes.Unavailable, "billing service temporarily unavailable")
    }

    return resp, err
}
```

### 4. Canary Deployments

```yaml
# Kubernetes canary deployment with Istio
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: user-service
spec:
  hosts:
    - user-service
  http:
    - match:
        - headers:
            x-canary:
              exact: "true"
      route:
        - destination:
            host: user-service
            subset: canary
    - route:
        - destination:
            host: user-service
            subset: stable
          weight: 95
        - destination:
            host: user-service
            subset: canary
          weight: 5
---
apiVersion: networking.istio.io/v1alpha3
kind: DestinationRule
metadata:
  name: user-service
spec:
  host: user-service
  subsets:
    - name: stable
      labels:
        version: v1
    - name: canary
      labels:
        version: v2
```

### 5. Distributed Tracing

```go
import (
    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/trace"
    "go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc"
)

func initTracer() func() {
    exporter, _ := jaeger.New(jaeger.WithCollectorEndpoint(
        jaeger.WithEndpoint("http://jaeger:14268/api/traces"),
    ))

    tp := sdktrace.NewTracerProvider(
        sdktrace.WithBatcher(exporter),
        sdktrace.WithResource(resource.NewWithAttributes(
            semconv.SchemaURL,
            semconv.ServiceNameKey.String("auth-service"),
        )),
    )

    otel.SetTracerProvider(tp)

    return func() { tp.Shutdown(context.Background()) }
}

// gRPC server with tracing
func NewGRPCServer() *grpc.Server {
    return grpc.NewServer(
        grpc.UnaryInterceptor(otelgrpc.UnaryServerInterceptor()),
        grpc.StreamInterceptor(otelgrpc.StreamServerInterceptor()),
    )
}

// Manual span creation
func (s *AuthService) ValidateToken(ctx context.Context, req *ValidateRequest) (*ValidateResponse, error) {
    ctx, span := otel.Tracer("auth").Start(ctx, "ValidateToken")
    defer span.End()

    span.SetAttributes(
        attribute.String("token.type", "access"),
        attribute.String("tenant.id", req.TenantID),
    )

    claims, err := s.parseToken(ctx, req.Token)
    if err != nil {
        span.RecordError(err)
        span.SetStatus(codes.Error, err.Error())
        return nil, err
    }

    span.SetAttributes(attribute.String("user.id", claims.UserID))

    return &ValidateResponse{Valid: true, Claims: claims}, nil
}
```

---

## Performance Considerations

### Service Communication
- **Connection pooling:** Maintain gRPC connection pools to each service
- **Request coalescing:** Batch multiple requests to same service
- **Caching:** Cache frequently accessed data (user profiles, permissions)
- **Async processing:** Use message queue for non-critical operations

### Database Optimization
- **Read replicas:** Route read queries to replicas
- **Connection pooling:** Use PgBouncer for connection management
- **Partitioning:** Partition large tables by tenant_id
- **Indexing strategy:** Composite indexes for tenant + common queries

### Latency Targets
| Operation | Target P50 | Target P99 |
|-----------|------------|------------|
| Token validation | <5ms | <20ms |
| User lookup | <10ms | <50ms |
| Create subscription | <500ms | <2s |
| Send notification | <100ms | <500ms |

---

## Stretch Goals

### 1. Multi-Region Failover

```go
type RegionalRouter struct {
    regions    []Region
    primary    string
    healthChecker HealthChecker
}

type Region struct {
    Name     string
    Endpoint string
    Weight   int
    Healthy  bool
}

func (r *RegionalRouter) Route(ctx context.Context) (*Region, error) {
    // Check primary region health
    primary := r.getRegion(r.primary)
    if primary.Healthy {
        return primary, nil
    }

    // Failover to secondary
    for _, region := range r.regions {
        if region.Name != r.primary && region.Healthy {
            log.Warn("failing over to secondary region",
                "from", r.primary,
                "to", region.Name)
            return &region, nil
        }
    }

    return nil, ErrNoHealthyRegion
}

// DNS-based failover with Route53/Cloud DNS
// Active-passive or active-active configuration
```

### 2. OPA Policy Engine

```rego
# policy.rego
package authz

default allow = false

# Allow users to read their own profile
allow {
    input.action == "read"
    input.resource == "user"
    input.resource_id == input.user.id
}

# Allow admins to read any user in their tenant
allow {
    input.action == "read"
    input.resource == "user"
    "admin" in input.user.roles
    input.resource_tenant == input.user.tenant_id
}

# Billing permissions
allow {
    input.action == "create"
    input.resource == "subscription"
    "billing_admin" in input.user.roles
}

# Check specific permissions
allow {
    required_permission := sprintf("%s:%s", [input.resource, input.action])
    required_permission in input.user.permissions
}
```

```go
// OPA client integration
type OPAAuthorizer struct {
    client *opa.Client
}

func (a *OPAAuthorizer) Authorize(ctx context.Context, req AuthzRequest) (bool, error) {
    input := map[string]interface{}{
        "action":          req.Action,
        "resource":        req.Resource,
        "resource_id":     req.ResourceID,
        "resource_tenant": req.ResourceTenant,
        "user": map[string]interface{}{
            "id":          req.User.ID,
            "tenant_id":   req.User.TenantID,
            "roles":       req.User.Roles,
            "permissions": req.User.Permissions,
        },
    }

    result, err := a.client.Query(ctx, "data.authz.allow", input)
    if err != nil {
        return false, err
    }

    return result.Allowed, nil
}
```

---

## Testing Strategy

### Unit Tests
- Token generation/validation
- Rate limiting logic
- Circuit breaker state transitions
- Policy evaluation

### Integration Tests
- End-to-end auth flows
- Service-to-service communication
- Event publishing/consumption
- Database operations with RLS

### Contract Tests
- Protobuf schema compatibility
- API versioning
- Event schema validation

### Load Tests
- Concurrent authentication requests
- Subscription creation under load
- Notification throughput
- Rate limiter accuracy

### Security Tests
- SQL injection attempts
- Token manipulation
- Tenant isolation verification
- Authorization bypass attempts

---

## Monitoring & Alerting

### Key Metrics
```yaml
metrics:
  # Service health
  - name: grpc_server_handled_total
    type: counter
    labels: [service, method, code]

  - name: grpc_server_handling_seconds
    type: histogram
    labels: [service, method]

  # Business metrics
  - name: auth_login_total
    type: counter
    labels: [tenant_id, result]

  - name: billing_subscription_total
    type: counter
    labels: [tenant_id, plan, status]

  - name: notification_sent_total
    type: counter
    labels: [channel, status]

  # System metrics
  - name: rate_limit_exceeded_total
    type: counter
    labels: [tenant_id, endpoint]

  - name: circuit_breaker_state
    type: gauge
    labels: [service, state]
```

### Alerts
- Service error rate > 1% for 5 minutes
- P99 latency > SLA threshold
- Circuit breaker open
- Rate limit saturation > 80%
- Failed login spike (potential attack)

---

## Implementation Phases

### Phase 1: Foundation (Week 1-3)
- [ ] Set up project structure and build system
- [ ] Define protobuf schemas for all services
- [ ] Implement basic User Service CRUD
- [ ] Implement basic Auth Service (login, JWT)
- [ ] Set up PostgreSQL with migrations

### Phase 2: Core Services (Week 4-6)
- [ ] Complete Auth Service (refresh, logout, sessions)
- [ ] Implement Billing Service with Stripe
- [ ] Build Notification Service (email channel)
- [ ] Set up message broker (NATS/RabbitMQ)
- [ ] Implement event publishing

### Phase 3: API Gateway (Week 7-8)
- [ ] Deploy Kong or Envoy
- [ ] Configure routing rules
- [ ] Implement rate limiting
- [ ] Add JWT validation at gateway
- [ ] Set up gRPC-gateway for REST

### Phase 4: Enterprise (Week 9-10)
- [ ] Implement audit logging
- [ ] Add circuit breakers
- [ ] Set up distributed tracing
- [ ] Implement multi-tenant isolation
- [ ] Add comprehensive monitoring

### Phase 5: Polish & Scale (Week 11-12)
- [ ] Kubernetes deployment configs
- [ ] Canary deployment setup
- [ ] Load testing and optimization
- [ ] Security hardening
- [ ] Documentation

---

## References

- [Microservices Patterns](https://microservices.io/patterns/)
- [gRPC Best Practices](https://grpc.io/docs/guides/)
- [Stripe API Design](https://stripe.com/docs/api)
- [Open Policy Agent](https://www.openpolicyagent.org/)
- [Kong Gateway](https://docs.konghq.com/)
- [OpenTelemetry](https://opentelemetry.io/docs/)
