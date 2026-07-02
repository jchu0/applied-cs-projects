# SaaS Web Platform - API Documentation

## Table of Contents
- [Overview](#overview)
- [Authentication](#authentication)
- [API Endpoints](#api-endpoints)
- [Request/Response Format](#requestresponse-format)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Webhooks](#webhooks)
- [Client Examples](#client-examples)

## Overview

The SaaS Platform API is a RESTful API that provides programmatic access to all platform features. Built with Django REST Framework, it follows REST principles and uses JSON for request and response payloads.

### Base URLs
- **Production**: `https://api.saas-platform.com`
- **Staging**: `https://staging-api.saas-platform.com`
- **Development**: `http://localhost:8000`

### API Versioning
The API uses URL-based versioning. The current version is v1.
```
https://api.saas-platform.com/api/v1/
```

## Authentication

### JWT Authentication

The API uses JWT (JSON Web Tokens) for authentication. To authenticate, you need to obtain an access token by registering or logging in.

#### Login
```http
POST /api/v1/auth/login/
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "YourPassword123!"
}
```

**Response:**
```json
{
  "user": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "email": "user@example.com",
    "email_verified": false,
    "first_name": "John",
    "last_name": "Doe",
    "full_name": "John Doe",
    "avatar_url": null,
    "created_at": "2024-01-15T10:30:00Z",
    "last_login_at": "2024-01-15T09:00:00Z"
  },
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

The access token is valid for 1 hour; the refresh token for 7 days.

#### Using the Token
Include the access token in the Authorization header for authenticated requests:
```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

#### Refresh Token
```http
POST /api/v1/auth/token/refresh/
Content-Type: application/json

{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer"
}
```

## API Endpoints

All endpoints are mounted under `/api/v1/`. The route groups are:

| Prefix | App | Purpose |
|--------|-----|---------|
| `/api/v1/` | core | Health/readiness probes |
| `/api/v1/auth/` | users | Registration, login, profile, password |
| `/api/v1/tenants/` | tenants | Workspaces and membership |
| `/api/v1/billing/` | billing | Plans, subscriptions, invoices, Stripe |
| `/api/v1/admin/` | admin_dashboard | Platform admin (staff only) |
| `/api/v1/scheduler/` | scheduler | Scheduled tasks, queues, cron schedules |
| `/api/v1/resources/` | resources | Compute nodes, GPUs, allocations, quotas |
| `/api/v1/training/` | training | Training jobs, runs, experiments, sweeps |

### Core / Health

#### Health Check
```http
GET /api/v1/health/
```

**Response:**
```json
{
  "status": "ok",
  "database": "healthy"
}
```

#### Readiness Check
```http
GET /api/v1/ready/
```

**Response:**
```json
{
  "status": "ready"
}
```

### Authentication & User Management

All auth endpoints are mounted under `/api/v1/auth/`.

#### Register
```http
POST /api/v1/auth/register/
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "YourPassword123!",
  "first_name": "John",
  "last_name": "Doe"
}
```
Returns `201 Created` with the same `{ user, access_token, refresh_token }` shape as login.

#### Login
```http
POST /api/v1/auth/login/
```
See [Authentication](#authentication).

#### Logout
```http
POST /api/v1/auth/logout/
Authorization: Bearer {access_token}
```
JWTs are stateless; the client should discard the token. Returns `{"message": "Logged out successfully"}`.

#### Get / Update Current User
```http
GET   /api/v1/auth/me/
PATCH /api/v1/auth/me/
Authorization: Bearer {access_token}
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "email": "user@example.com",
  "email_verified": false,
  "first_name": "John",
  "last_name": "Doe",
  "full_name": "John Doe",
  "avatar_url": null,
  "created_at": "2024-01-15T10:30:00Z",
  "last_login_at": "2024-01-15T09:00:00Z"
}
```
`PATCH` accepts writable fields (`first_name`, `last_name`, `avatar_url`); `id`, `email`, `email_verified`, `created_at`, and `last_login_at` are read-only.

#### Profile (alias)
```http
GET   /api/v1/auth/profile/
PATCH /api/v1/auth/profile/
Authorization: Bearer {access_token}
```
Same behavior and payload as `/auth/me/`.

#### Change Password
```http
POST /api/v1/auth/password/change/
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "current_password": "OldPassword123!",
  "new_password": "NewPassword123!"
}
```

#### Refresh Token
```http
POST /api/v1/auth/token/refresh/
```
See [Authentication](#authentication).

#### Request Password Reset
```http
POST /api/v1/auth/password/reset/
Content-Type: application/json

{ "email": "user@example.com" }
```
Always returns a generic success message so account existence is not revealed.

#### Confirm Password Reset
```http
POST /api/v1/auth/password/reset/confirm/
Content-Type: application/json

{ "token": "<reset-token>", "new_password": "NewPassword123!" }
```

### Tenant Management

Mounted under `/api/v1/tenants/`. Tenants are the workspace/organization concept for the platform.

#### List / Create Tenants
```http
GET  /api/v1/tenants/
POST /api/v1/tenants/
Authorization: Bearer {access_token}
```
`GET` lists the tenants the current user belongs to. `POST` creates a tenant (creator becomes `owner`):
```json
{ "name": "Acme Corp", "slug": "acme-corp" }
```

#### Get / Update / Delete Tenant
```http
GET    /api/v1/tenants/{tenant_id}/
PATCH  /api/v1/tenants/{tenant_id}/
DELETE /api/v1/tenants/{tenant_id}/
Authorization: Bearer {access_token}
```
`tenant_id` is a UUID. `PATCH` requires `owner` or `admin`; `DELETE` requires `owner`.

#### List Members / Invite Member
```http
GET  /api/v1/tenants/{tenant_id}/members/
POST /api/v1/tenants/{tenant_id}/members/
Authorization: Bearer {access_token}
```
`POST` (owner/admin only) creates an invitation:
```json
{ "email": "newmember@example.com", "role": "member" }
```

#### Accept Invitation
```http
POST /api/v1/tenants/invitations/{token}/accept/
Authorization: Bearer {access_token}
```
Consumes an invitation token and adds the current user to the tenant.

### Subscription & Billing

Mounted under `/api/v1/billing/`.

#### List Available Plans
```http
GET /api/v1/billing/plans/
```
Returns the list of active plans (serialized `Plan` objects).

#### Get / Create / Cancel Subscription
```http
GET    /api/v1/billing/tenants/{tenant_id}/subscription/
POST   /api/v1/billing/tenants/{tenant_id}/subscription/
DELETE /api/v1/billing/tenants/{tenant_id}/subscription/
Authorization: Bearer {access_token}
```
`POST` creates or updates the subscription (owner/admin only):
```json
{ "plan_id": "<plan-uuid>", "billing_interval": "monthly", "payment_method_id": "pm_card_visa" }
```
`DELETE` cancels the subscription (owner only).

#### List Invoices
```http
GET /api/v1/billing/tenants/{tenant_id}/invoices/
Authorization: Bearer {access_token}
```

#### Payment Methods
```http
GET    /api/v1/billing/tenants/{tenant_id}/payment-methods/
POST   /api/v1/billing/tenants/{tenant_id}/payment-methods/
Authorization: Bearer {access_token}
```
`POST` (owner/admin) returns a Stripe SetupIntent for attaching a new card.

#### Checkout Session
```http
POST /api/v1/billing/tenants/{tenant_id}/checkout/
Authorization: Bearer {access_token}
Content-Type: application/json

{ "plan_id": "<plan-uuid>", "billing_interval": "monthly" }
```
Returns `{ "checkout_url": "https://checkout.stripe.com/..." }`.

#### Billing Portal
```http
POST /api/v1/billing/tenants/{tenant_id}/portal/
Authorization: Bearer {access_token}
```
Returns `{ "portal_url": "https://billing.stripe.com/..." }`.

#### Stripe Webhook
```http
POST /api/v1/billing/webhooks/stripe/
Stripe-Signature: {signature}
```
Public endpoint that verifies the Stripe signature and processes billing events.

### Admin Dashboard (staff only)

Mounted under `/api/v1/admin/`. All endpoints require an admin (staff) user.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/admin/` | Dashboard summary (user/tenant/subscription counts, recent activity) |
| GET | `/api/v1/admin/stats/` | Aggregate stats (users, tenants, subscriptions, revenue/MRR) |
| GET | `/api/v1/admin/users/` | Paginated user list (filters: `email`, `is_active`, `page`, `per_page`) |
| GET | `/api/v1/admin/tenants/` | Paginated tenant list (filters: `name`, `is_active`, `page`, `per_page`) |
| GET | `/api/v1/admin/audit-logs/` | Paginated audit log (filters: `action`, `resource_type`, `user_id`, `tenant_id`) |
| GET | `/api/v1/admin/charts/growth/` | Daily user signups and tenant creations (`days` query param) |

### Scheduler

Mounted under `/api/v1/scheduler/` via DRF routers. Each resource supports the standard list/retrieve/create/update/delete verbs plus the custom actions below.

**Scheduled Tasks** — `/api/v1/scheduler/tasks/`
- `POST /tasks/{id}/run/` — trigger a task immediately
- `POST /tasks/{id}/cancel/` — cancel a task
- `POST /tasks/{id}/retry/` — retry a failed task
- `GET  /tasks/{id}/executions/` — execution history for a task
- `GET  /tasks/{id}/status/` — current status
- `GET  /tasks/stats/` — task statistics
- `GET  /tasks/upcoming/` — upcoming scheduled tasks

**Task Executions** (read-only) — `/api/v1/scheduler/executions/`

**Task Queues** — `/api/v1/scheduler/queues/`
- `GET  /queues/stats/` — queue statistics
- `GET  /queues/workers/` — worker status
- `POST /queues/{id}/purge/` — purge a queue

**Cron Schedules** — `/api/v1/scheduler/cron-schedules/`
- `GET /cron-schedules/presets/` — common cron presets

### Resources

Mounted under `/api/v1/resources/` via DRF routers.

**Compute Nodes** (read-only) — `/api/v1/resources/nodes/`
- `GET /nodes/regions/` — available regions
- `GET /nodes/gpu_types/` — available GPU types
- `GET /nodes/{id}/gpus/` — GPUs on a node

**GPUs** (read-only) — `/api/v1/resources/gpus/`
- `GET /gpus/available/` — currently available GPUs
- `GET /gpus/availability/` — availability summary

**Allocations** — `/api/v1/resources/allocations/`
- `POST /allocations/{id}/release/` — release an allocation
- `POST /allocations/{id}/extend/` — extend an allocation
- `GET  /allocations/active/` — active allocations
- `GET  /allocations/stats/` — allocation statistics

**Quotas** — `/api/v1/resources/quotas/`
- `GET  /quotas/current/` — current quota usage
- `POST /quotas/{id}/reset_usage/` — reset usage counters

**Reservations** — `/api/v1/resources/reservations/`
- `POST /reservations/{id}/cancel/` — cancel a reservation
- `POST /reservations/{id}/activate/` — activate a reservation
- `GET  /reservations/upcoming/` — upcoming reservations

### Training

Mounted under `/api/v1/training/` via DRF routers.

**Training Jobs** — `/api/v1/training/jobs/`
- `POST /jobs/{id}/submit/` — submit a job to the queue
- `POST /jobs/{id}/cancel/` — cancel a job
- `POST /jobs/{id}/retry/` — retry a job
- `POST /jobs/{id}/update_progress/` — report progress
- `POST /jobs/{id}/checkpoint/` — record a checkpoint
- `GET  /jobs/{id}/runs/` — runs for a job
- `GET  /jobs/{id}/artifacts/` — artifacts for a job
- `GET  /jobs/{id}/logs/` — job logs
- `GET  /jobs/running/`, `/jobs/queued/`, `/jobs/stats/` — collection views

**Training Runs** (read-only) — `/api/v1/training/runs/`

**Experiments** — `/api/v1/training/experiments/`
- `GET /experiments/{id}/jobs/` — jobs in an experiment
- `GET /experiments/{id}/sweeps/` — sweeps in an experiment

**Hyperparameter Sweeps** — `/api/v1/training/sweeps/`
- `POST /sweeps/{id}/start/` — start a sweep
- `POST /sweeps/{id}/stop/` — stop a sweep
- `GET  /sweeps/{id}/next_params/` — next hyperparameters to try
- `POST /sweeps/{id}/report_trial/` — report a trial result

**Model Artifacts** (read-only) — `/api/v1/training/artifacts/`
- `GET /artifacts/best/` — best artifact by metric

## Request/Response Format

### Request Headers
```http
Content-Type: application/json
Accept: application/json
Authorization: Bearer {access_token}
X-Request-ID: {uuid}  # Optional, for tracking
```

### Standard Response Format
```json
{
  "data": { },
  "meta": {
    "request_id": "req_550e8400e29b41d4a716446655440000",
    "timestamp": "2024-01-15T10:30:00Z",
    "version": "v1"
  }
}
```

### Pagination
```json
{
  "count": 100,
  "next": "https://api.saas-platform.com/api/v1/resources/?page=2",
  "previous": null,
  "page": 1,
  "page_size": 20,
  "total_pages": 5,
  "results": []
}
```

### Filtering
```http
GET /api/v1/resources/?status=active&created_after=2024-01-01&tags=backend,api
```

### Sorting
```http
GET /api/v1/resources/?sort_by=created_at&order=desc
```

## Error Handling

### Error Response Format
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Validation failed",
    "details": [
      {
        "field": "email",
        "message": "Invalid email format"
      }
    ],
    "request_id": "req_550e8400e29b41d4a716446655440000",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### Error Codes

| Status Code | Error Code | Description |
|------------|-----------|-------------|
| 400 | BAD_REQUEST | Invalid request format |
| 400 | VALIDATION_ERROR | Request validation failed |
| 401 | UNAUTHORIZED | Authentication required |
| 401 | TOKEN_EXPIRED | Access token has expired |
| 403 | FORBIDDEN | Insufficient permissions |
| 404 | NOT_FOUND | Resource not found |
| 409 | CONFLICT | Resource already exists |
| 422 | UNPROCESSABLE_ENTITY | Business logic error |
| 429 | RATE_LIMITED | Too many requests |
| 500 | INTERNAL_ERROR | Server error |
| 503 | SERVICE_UNAVAILABLE | Service temporarily unavailable |

### Error Handling Examples

#### Python
```python
import requests

try:
    response = requests.get(
        'https://api.saas-platform.com/api/v1/tenants/',
        headers={'Authorization': f'Bearer {token}'}
    )
    response.raise_for_status()
    data = response.json()
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 401:
        # Refresh token and retry
        refresh_token()
    elif e.response.status_code == 429:
        # Handle rate limiting
        retry_after = e.response.headers.get('Retry-After', 60)
        time.sleep(int(retry_after))
    else:
        error_data = e.response.json()
        print(f"Error: {error_data['error']['message']}")
```

#### JavaScript
```javascript
try {
  const response = await fetch('https://api.saas-platform.com/api/v1/tenants/', {
    headers: {
      'Authorization': `Bearer ${token}`
    }
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error.message);
  }

  const data = await response.json();
} catch (error) {
  if (error.response?.status === 401) {
    // Refresh token and retry
    await refreshToken();
  } else if (error.response?.status === 429) {
    // Handle rate limiting
    const retryAfter = error.response.headers.get('Retry-After') || 60;
    await sleep(retryAfter * 1000);
  } else {
    console.error('API Error:', error.message);
  }
}
```

## Rate Limiting

### Rate Limit Headers
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642252800
Retry-After: 60
```

### Rate Limits by Endpoint

Auth endpoints use DRF's `ScopedRateThrottle`. The scopes below are applied per view; other authenticated endpoints fall back to the default throttle.

| Endpoint Pattern | Throttle Scope | Notes |
|-----------------|----------------|-------|
| POST /api/v1/auth/register/ | `auth_register` | Registration |
| POST /api/v1/auth/login/ | `auth_login` | Login |
| POST /api/v1/auth/token/refresh/ | `auth_token_refresh` | Token refresh |
| POST /api/v1/auth/password/reset/ | `auth_password_reset` | Password reset request/confirm |
| Default | (DRF default) | All other authenticated endpoints |

### Handling Rate Limits
```python
def api_request_with_retry(url, headers, max_retries=3):
    for attempt in range(max_retries):
        response = requests.get(url, headers=headers)

        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            print(f"Rate limited. Waiting {retry_after} seconds...")
            time.sleep(retry_after)
            continue

        return response

    raise Exception("Max retries exceeded")
```

## Webhooks

The platform exposes a single inbound webhook: the Stripe billing webhook. It is
consumed by the platform (not registered by API clients) and is used to keep
subscription, invoice, and payment state in sync with Stripe.

### Stripe Webhook Endpoint
```http
POST /api/v1/billing/webhooks/stripe/
Stripe-Signature: {signature}
Content-Type: application/json

{ ...standard Stripe event payload... }
```
The endpoint is public (no bearer token) but verifies the `Stripe-Signature`
header against the configured webhook signing secret before processing. Relevant
Stripe events (e.g. `checkout.session.completed`, `customer.subscription.updated`,
`invoice.paid`, `invoice.payment_failed`) update the corresponding
`Subscription`, `Invoice`, and `PaymentMethod` records.

Configure this URL as a webhook destination in the Stripe Dashboard and set the
signing secret via the `STRIPE_WEBHOOK_SECRET` setting.

## Client Examples

The API is a plain JSON REST API authenticated with a JWT bearer token; any HTTP
client works. The examples below use real endpoints.

### Python (requests)
```python
import requests

BASE = 'http://localhost:8000/api/v1'

# Login to obtain tokens
resp = requests.post(f'{BASE}/auth/login/', json={
    'email': 'user@example.com',
    'password': 'YourPassword123!',
})
tokens = resp.json()
headers = {'Authorization': f"Bearer {tokens['access_token']}"}

# Current user
me = requests.get(f'{BASE}/auth/me/', headers=headers).json()

# List the tenants I belong to
tenants = requests.get(f'{BASE}/tenants/', headers=headers).json()

# Create a tenant
tenant = requests.post(f'{BASE}/tenants/', headers=headers, json={
    'name': 'Acme Corp', 'slug': 'acme-corp',
}).json()

# List available plans
plans = requests.get(f'{BASE}/billing/plans/', headers=headers).json()

# List training jobs
jobs = requests.get(f'{BASE}/training/jobs/', headers=headers).json()
```

### cURL Examples
```bash
BASE=http://localhost:8000/api/v1

# Login
curl -X POST $BASE/auth/login/ \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"YourPassword123!"}'

# Current user
curl -X GET $BASE/auth/me/ \
  -H "Authorization: Bearer ${ACCESS_TOKEN}"

# Create a tenant
curl -X POST $BASE/tenants/ \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"name":"Acme Corp","slug":"acme-corp"}'

# Get a tenant's subscription
curl -X GET $BASE/billing/tenants/${TENANT_ID}/subscription/ \
  -H "Authorization: Bearer ${ACCESS_TOKEN}"

# Submit a training job
curl -X POST $BASE/training/jobs/${JOB_ID}/submit/ \
  -H "Authorization: Bearer ${ACCESS_TOKEN}"
```

## Support

### Changelog
View API changelog: https://developers.saas-platform.com/changelog

### Deprecation Policy
- Deprecated features are marked with `X-Deprecated` header
- Minimum 6 months notice before removing features
- Migration guides provided for breaking changes