# SaaS Web Platform - API Documentation

## Table of Contents
- [Overview](#overview)
- [Authentication](#authentication)
- [API Endpoints](#api-endpoints)
- [Request/Response Format](#requestresponse-format)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Webhooks](#webhooks)
- [SDK Examples](#sdk-examples)

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

The API uses JWT (JSON Web Tokens) for authentication. To authenticate, you need to obtain an access token by logging in.

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
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "email": "user@example.com",
    "username": "johndoe",
    "first_name": "John",
    "last_name": "Doe"
  },
  "expires_in": 900
}
```

#### Using the Token
Include the access token in the Authorization header for authenticated requests:
```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

#### Refresh Token
```http
POST /api/v1/auth/refresh/
Content-Type: application/json

{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

### API Key Authentication

For server-to-server communication, you can use API keys.

#### Generate API Key
```http
POST /api/v1/keys/generate/
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "name": "Production Server",
  "permissions": ["read", "write"],
  "expires_at": "2025-12-31T23:59:59Z"
}
```

**Response:**
```json
{
  "id": "key_550e8400e29b41d4a716446655440000",
  "key": "sk_live_EXAMPLEONLYnotARealKey00",
  "name": "Production Server",
  "created_at": "2024-01-15T10:30:00Z",
  "expires_at": "2025-12-31T23:59:59Z"
}
```

#### Using API Key
```http
X-API-Key: sk_live_EXAMPLEONLYnotARealKey00
```

## API Endpoints

### User Management

#### Get Current User
```http
GET /api/v1/users/me/
Authorization: Bearer {access_token}
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "email": "user@example.com",
  "username": "johndoe",
  "first_name": "John",
  "last_name": "Doe",
  "avatar_url": "https://cdn.saas-platform.com/avatars/johndoe.jpg",
  "organization": {
    "id": "org_123",
    "name": "Acme Corp",
    "role": "admin"
  },
  "created_at": "2023-01-15T10:30:00Z",
  "last_login": "2024-01-15T09:00:00Z"
}
```

#### Update User Profile
```http
PATCH /api/v1/users/me/
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "first_name": "Jane",
  "last_name": "Smith",
  "bio": "Software Engineer",
  "timezone": "America/New_York"
}
```

#### List Users (Admin Only)
```http
GET /api/v1/users/
Authorization: Bearer {access_token}

Query Parameters:
- page: integer (default: 1)
- page_size: integer (default: 20, max: 100)
- search: string (search by name or email)
- organization_id: string (filter by organization)
- role: string (filter by role)
```

### Organization Management

#### Create Organization
```http
POST /api/v1/organizations/
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "name": "Acme Corp",
  "slug": "acme-corp",
  "description": "Leading software company",
  "website": "https://acme-corp.com",
  "size": "50-100",
  "industry": "technology"
}
```

#### Get Organization
```http
GET /api/v1/organizations/{org_id}/
Authorization: Bearer {access_token}
```

#### Update Organization
```http
PATCH /api/v1/organizations/{org_id}/
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "name": "Acme Corporation",
  "description": "Updated description"
}
```

#### Invite Members
```http
POST /api/v1/organizations/{org_id}/invitations/
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "email": "newmember@example.com",
  "role": "editor",
  "message": "Welcome to our team!"
}
```

### Project Management

#### List Projects
```http
GET /api/v1/projects/
Authorization: Bearer {access_token}

Query Parameters:
- organization_id: string
- status: string (active, archived, deleted)
- created_after: datetime
- created_before: datetime
- tags: array of strings
- sort_by: string (created_at, updated_at, name)
- order: string (asc, desc)
```

**Response:**
```json
{
  "count": 42,
  "next": "https://api.saas-platform.com/api/v1/projects/?page=2",
  "previous": null,
  "results": [
    {
      "id": "proj_550e8400e29b41d4a716446655440000",
      "name": "Website Redesign",
      "slug": "website-redesign",
      "description": "Complete redesign of company website",
      "status": "active",
      "visibility": "private",
      "organization": {
        "id": "org_123",
        "name": "Acme Corp"
      },
      "owner": {
        "id": "user_456",
        "username": "johndoe",
        "email": "john@example.com"
      },
      "members_count": 5,
      "tags": ["design", "frontend"],
      "settings": {
        "allow_comments": true,
        "require_approval": false
      },
      "created_at": "2024-01-01T10:00:00Z",
      "updated_at": "2024-01-15T14:30:00Z"
    }
  ]
}
```

#### Create Project
```http
POST /api/v1/projects/
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "name": "New Project",
  "description": "Project description",
  "organization_id": "org_123",
  "visibility": "private",
  "tags": ["backend", "api"],
  "settings": {
    "allow_comments": true,
    "require_approval": true,
    "default_branch": "main"
  }
}
```

#### Get Project Details
```http
GET /api/v1/projects/{project_id}/
Authorization: Bearer {access_token}
```

#### Update Project
```http
PATCH /api/v1/projects/{project_id}/
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "name": "Updated Project Name",
  "status": "archived"
}
```

#### Delete Project
```http
DELETE /api/v1/projects/{project_id}/
Authorization: Bearer {access_token}
```

### File Management

#### Upload File
```http
POST /api/v1/files/upload/
Authorization: Bearer {access_token}
Content-Type: multipart/form-data

file: (binary)
project_id: proj_123
folder_path: /documents
description: "Project documentation"
```

**Response:**
```json
{
  "id": "file_789",
  "name": "document.pdf",
  "size": 2048576,
  "mime_type": "application/pdf",
  "url": "https://cdn.saas-platform.com/files/document.pdf",
  "thumbnail_url": "https://cdn.saas-platform.com/thumbnails/document.jpg",
  "project_id": "proj_123",
  "folder_path": "/documents",
  "uploaded_by": {
    "id": "user_456",
    "username": "johndoe"
  },
  "created_at": "2024-01-15T10:00:00Z"
}
```

#### List Files
```http
GET /api/v1/files/
Authorization: Bearer {access_token}

Query Parameters:
- project_id: string
- folder_path: string
- mime_type: string
- search: string
```

#### Download File
```http
GET /api/v1/files/{file_id}/download/
Authorization: Bearer {access_token}
```

#### Delete File
```http
DELETE /api/v1/files/{file_id}/
Authorization: Bearer {access_token}
```

### Subscription & Billing

#### Get Current Subscription
```http
GET /api/v1/subscriptions/current/
Authorization: Bearer {access_token}
```

**Response:**
```json
{
  "id": "sub_123",
  "plan": {
    "id": "plan_pro",
    "name": "Professional",
    "price": 29.99,
    "currency": "USD",
    "interval": "month",
    "features": {
      "users": 5,
      "storage": 10240,
      "api_calls": 10000,
      "custom_domain": true
    }
  },
  "status": "active",
  "current_period_start": "2024-01-01T00:00:00Z",
  "current_period_end": "2024-02-01T00:00:00Z",
  "cancel_at_period_end": false,
  "usage": {
    "users": {
      "used": 3,
      "limit": 5
    },
    "storage": {
      "used": 2048,
      "limit": 10240
    },
    "api_calls": {
      "used": 5423,
      "limit": 10000
    }
  }
}
```

#### List Available Plans
```http
GET /api/v1/plans/
```

**Response:**
```json
{
  "plans": [
    {
      "id": "plan_free",
      "name": "Free",
      "price": 0,
      "currency": "USD",
      "interval": "month",
      "features": {
        "users": 1,
        "storage": 1024,
        "api_calls": 1000,
        "projects": 3
      }
    },
    {
      "id": "plan_pro",
      "name": "Professional",
      "price": 29.99,
      "currency": "USD",
      "interval": "month",
      "features": {
        "users": 5,
        "storage": 10240,
        "api_calls": 10000,
        "projects": 20,
        "custom_domain": true
      }
    },
    {
      "id": "plan_enterprise",
      "name": "Enterprise",
      "price": 99.99,
      "currency": "USD",
      "interval": "month",
      "features": {
        "users": -1,
        "storage": 102400,
        "api_calls": -1,
        "projects": -1,
        "custom_domain": true,
        "sso": true,
        "priority_support": true
      }
    }
  ]
}
```

#### Subscribe to Plan
```http
POST /api/v1/subscriptions/subscribe/
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "plan_id": "plan_pro",
  "payment_method": "pm_card_visa"
}
```

#### Cancel Subscription
```http
POST /api/v1/subscriptions/{subscription_id}/cancel/
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "reason": "too_expensive",
  "feedback": "Optional feedback message",
  "cancel_immediately": false
}
```

### Analytics

#### Get Usage Analytics
```http
GET /api/v1/analytics/usage/
Authorization: Bearer {access_token}

Query Parameters:
- start_date: date (YYYY-MM-DD)
- end_date: date (YYYY-MM-DD)
- metric: string (api_calls, storage, users, projects)
- granularity: string (hour, day, week, month)
```

**Response:**
```json
{
  "metric": "api_calls",
  "period": {
    "start": "2024-01-01",
    "end": "2024-01-31"
  },
  "granularity": "day",
  "data": [
    {
      "date": "2024-01-01",
      "value": 1234
    },
    {
      "date": "2024-01-02",
      "value": 1456
    }
  ],
  "summary": {
    "total": 45678,
    "average": 1473,
    "peak": 2345,
    "peak_date": "2024-01-15"
  }
}
```

#### Get Project Analytics
```http
GET /api/v1/analytics/projects/{project_id}/
Authorization: Bearer {access_token}
```

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
        'https://api.saas-platform.com/api/v1/projects/',
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
  const response = await fetch('https://api.saas-platform.com/api/v1/projects/', {
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

| Endpoint Pattern | Limit | Window |
|-----------------|-------|---------|
| /api/v1/auth/* | 5 | 1 minute |
| /api/v1/projects/* | 100 | 1 hour |
| /api/v1/files/upload/ | 10 | 1 minute |
| /api/v1/analytics/* | 50 | 1 hour |
| Default | 1000 | 1 hour |

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

### Webhook Events

| Event | Description | Payload |
|-------|-------------|---------|
| project.created | New project created | Project object |
| project.updated | Project updated | Project object with changes |
| project.deleted | Project deleted | Project ID |
| user.created | New user registered | User object |
| subscription.created | New subscription | Subscription object |
| subscription.updated | Subscription changed | Subscription object |
| subscription.canceled | Subscription canceled | Subscription object |
| payment.succeeded | Payment successful | Payment object |
| payment.failed | Payment failed | Payment object with error |

### Registering Webhooks
```http
POST /api/v1/webhooks/
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "url": "https://your-app.com/webhook",
  "events": ["project.created", "project.updated"],
  "secret": "your-webhook-secret"
}
```

### Webhook Payload
```json
{
  "id": "evt_550e8400e29b41d4a716446655440000",
  "type": "project.created",
  "created": "2024-01-15T10:30:00Z",
  "data": {
    "object": {
      "id": "proj_123",
      "name": "New Project",
      "created_at": "2024-01-15T10:30:00Z"
    }
  }
}
```

### Webhook Security
```python
import hmac
import hashlib

def verify_webhook(payload, signature, secret):
    expected_sig = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(expected_sig, signature)

# In your webhook handler
@app.route('/webhook', methods=['POST'])
def handle_webhook():
    signature = request.headers.get('X-Webhook-Signature')

    if not verify_webhook(request.data, signature, WEBHOOK_SECRET):
        return {'error': 'Invalid signature'}, 401

    event = request.json
    # Process webhook event
    return {'status': 'success'}, 200
```

## SDK Examples

### Python SDK
```python
from saas_platform import Client

# Initialize client
client = Client(api_key='sk_live_...')

# Get current user
user = client.users.me()

# List projects
projects = client.projects.list(
    organization_id='org_123',
    status='active'
)

# Create project
new_project = client.projects.create(
    name='New Project',
    description='Description',
    organization_id='org_123'
)

# Upload file
with open('document.pdf', 'rb') as f:
    file = client.files.upload(
        file=f,
        project_id='proj_123',
        folder_path='/documents'
    )

# Subscribe to plan
subscription = client.subscriptions.create(
    plan_id='plan_pro',
    payment_method='pm_card_visa'
)
```

### JavaScript/TypeScript SDK
```typescript
import { SaaSPlatformClient } from '@saas-platform/sdk';

// Initialize client
const client = new SaaSPlatformClient({
  apiKey: 'sk_live_...',
});

// Get current user
const user = await client.users.me();

// List projects
const projects = await client.projects.list({
  organizationId: 'org_123',
  status: 'active',
});

// Create project
const newProject = await client.projects.create({
  name: 'New Project',
  description: 'Description',
  organizationId: 'org_123',
});

// Upload file
const file = await client.files.upload({
  file: fileInput.files[0],
  projectId: 'proj_123',
  folderPath: '/documents',
});

// Subscribe to plan
const subscription = await client.subscriptions.create({
  planId: 'plan_pro',
  paymentMethod: 'pm_card_visa',
});
```

### cURL Examples
```bash
# Get current user
curl -X GET https://api.saas-platform.com/api/v1/users/me/ \
  -H "Authorization: Bearer ${ACCESS_TOKEN}"

# Create project
curl -X POST https://api.saas-platform.com/api/v1/projects/ \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "New Project",
    "description": "Project description",
    "organization_id": "org_123"
  }'

# Upload file
curl -X POST https://api.saas-platform.com/api/v1/files/upload/ \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -F "file=@document.pdf" \
  -F "project_id=proj_123" \
  -F "folder_path=/documents"
```

## GraphQL Alternative (Beta)

```graphql
# Query example
query GetProjectsAndUser {
  me {
    id
    username
    email
    organization {
      id
      name
      projects(status: ACTIVE, first: 10) {
        edges {
          node {
            id
            name
            description
            membersCount
            createdAt
          }
        }
        pageInfo {
          hasNextPage
          endCursor
        }
      }
    }
  }
}

# Mutation example
mutation CreateProject($input: CreateProjectInput!) {
  createProject(input: $input) {
    project {
      id
      name
      slug
    }
    errors {
      field
      message
    }
  }
}
```

## API Testing

### Postman Collection
Download our Postman collection: [SaaS Platform API.postman_collection.json](https://api.saas-platform.com/postman)

### API Playground
Interactive API documentation: https://api.saas-platform.com/playground

### Test Environment
- Base URL: https://sandbox-api.saas-platform.com
- Test API Key: `sk_test_EXAMPLEONLYnotARealKey00`
- Rate Limits: 10x higher than production

## Support

### Contact
- Email: api-support@saas-platform.com
- Developer Portal: https://developers.saas-platform.com
- Status Page: https://status.saas-platform.com

### Changelog
View API changelog: https://developers.saas-platform.com/changelog

### Deprecation Policy
- Deprecated features are marked with `X-Deprecated` header
- Minimum 6 months notice before removing features
- Migration guides provided for breaking changes