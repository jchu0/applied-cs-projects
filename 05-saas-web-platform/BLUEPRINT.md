# Project 4: Full SaaS Web Platform (Next.js + Django/Go)

## Staff-Level Design Document

**Complexity:** ⭐⭐⭐⭐⭐ (Expert)
**Timeline:** 12-16 weeks
**Languages:** TypeScript (frontend), Python/Go (backend), SQL

> **Concepts covered:** [§05 Authentication (OAuth/OIDC/MFA)](../../05-cross-cutting-concerns/security/authentication/authentication.md) · [§05 Secrets management](../../05-cross-cutting-concerns/security/secrets-management/secrets-management.md) · [§05 Observability](../../05-cross-cutting-concerns/observability/) · [§05 CI/CD](../../05-cross-cutting-concerns/ci-cd/) · [§07 Docker + Kubernetes](../../07-infrastructure/). Pairs with [Project 02 (microservice backend)](../02-microservice-platform/) and [Project 13 (service mesh)](../13-service-mesh/). Map: [`CONCEPT_TO_PROJECT_MAP.md`](../CONCEPT_TO_PROJECT_MAP.md).

---

## What This Project Teaches

### Core Concepts
- **End-to-end product engineering** - Full stack development from UI to database
- **Authentication patterns** - OAuth, OIDC, session management, MFA
- **Authorization models** - RBAC, ABAC, permissions systems
- **Subscription billing** - Stripe integration, metering, invoicing
- **Multi-tenancy** - Data isolation, tenant configuration, white-labeling
- **DevOps practices** - CI/CD, containerization, Kubernetes deployment
- **Infrastructure as Code** - Terraform, GitOps, secret management
- **Production operations** - Monitoring, logging, alerting, incident response

### Industry Relevance
This is how companies like Vercel, Linear, and Notion build their platforms. Understanding full-stack SaaS development is essential for founding startups or leading engineering teams.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Production Environment                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐       │
│  │   CDN        │    │  Load        │    │   WAF/DDoS      │       │
│  │ (CloudFront) │────│  Balancer    │────│   Protection    │       │
│  └──────┬───────┘    └──────┬───────┘    └─────────────────┘       │
│         │                   │                                        │
│  ┌──────▼───────┐    ┌──────▼───────┐                               │
│  │   Next.js    │    │   Backend    │                               │
│  │   Frontend   │────│   API        │                               │
│  │   (Vercel)   │    │  (K8s/ECS)   │                               │
│  └──────────────┘    └──────┬───────┘                               │
│                             │                                        │
│         ┌───────────────────┼───────────────────┐                   │
│         │                   │                   │                   │
│  ┌──────▼──────┐    ┌───────▼──────┐   ┌───────▼──────┐            │
│  │  PostgreSQL │    │    Redis     │   │  S3/Blob     │            │
│  │  (Primary)  │    │   (Cache)    │   │  Storage     │            │
│  └──────┬──────┘    └──────────────┘   └──────────────┘            │
│         │                                                            │
│  ┌──────▼──────┐                                                    │
│  │  Read       │                                                    │
│  │  Replicas   │                                                    │
│  └─────────────┘                                                    │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  External Services: Stripe, SendGrid, Auth0 (optional)   │       │
│  └──────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Breakdown

#### 1. Frontend (Next.js + Tailwind)
**Responsibilities:**
- Server-side rendering (SSR)
- Static site generation (SSG)
- Client-side interactivity
- Authentication UI
- Admin dashboards

**Tech Stack:**
- Next.js 14 (App Router)
- TypeScript
- Tailwind CSS
- Radix UI / shadcn/ui
- React Query / SWR
- Zustand (state management)

**Project Structure:**
```
frontend/
├── app/
│   ├── (auth)/
│   │   ├── login/
│   │   ├── signup/
│   │   └── forgot-password/
│   ├── (dashboard)/
│   │   ├── layout.tsx
│   │   ├── page.tsx
│   │   ├── settings/
│   │   ├── billing/
│   │   └── team/
│   ├── (marketing)/
│   │   ├── page.tsx
│   │   ├── pricing/
│   │   └── features/
│   └── api/
│       └── webhooks/
├── components/
│   ├── ui/
│   ├── forms/
│   └── layouts/
├── lib/
│   ├── api.ts
│   ├── auth.ts
│   └── utils.ts
└── middleware.ts
```

#### 2. Backend API (Django REST / Go + gRPC)

**Django Version:**
```python
# Project structure
backend/
├── config/
│   ├── settings/
│   │   ├── base.py
│   │   ├── development.py
│   │   └── production.py
│   ├── urls.py
│   └── wsgi.py
├── apps/
│   ├── users/
│   │   ├── models.py
│   │   ├── serializers.py
│   │   ├── views.py
│   │   └── permissions.py
│   ├── tenants/
│   ├── billing/
│   └── core/
├── utils/
└── manage.py
```

**Go Version:**
```go
// Project structure
backend/
├── cmd/
│   └── server/
│       └── main.go
├── internal/
│   ├── api/
│   │   ├── handlers/
│   │   ├── middleware/
│   │   └── routes.go
│   ├── domain/
│   │   ├── user/
│   │   ├── tenant/
│   │   └── billing/
│   ├── repository/
│   └── service/
├── pkg/
│   ├── auth/
│   ├── database/
│   └── logger/
└── go.mod
```

#### 3. Database Schema

```sql
-- Core multi-tenant schema

-- Tenants (organizations)
CREATE TABLE tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL,
    domain VARCHAR(255) UNIQUE,
    plan VARCHAR(50) NOT NULL DEFAULT 'free',
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Users
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    email_verified BOOLEAN DEFAULT FALSE,
    password_hash VARCHAR(255),
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    avatar_url TEXT,
    is_superuser BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_login_at TIMESTAMPTZ
);

-- Tenant memberships (users can belong to multiple tenants)
CREATE TABLE tenant_memberships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL DEFAULT 'member',
    permissions JSONB DEFAULT '[]',
    invited_by UUID REFERENCES users(id),
    joined_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(tenant_id, user_id)
);

-- Roles
CREATE TABLE roles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    permissions JSONB NOT NULL DEFAULT '[]',
    is_system BOOLEAN DEFAULT FALSE,
    UNIQUE(tenant_id, name)
);

-- API Keys
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    key_hash VARCHAR(255) NOT NULL,
    prefix VARCHAR(10) NOT NULL,
    scopes JSONB DEFAULT '[]',
    last_used_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Audit Log
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenants(id),
    user_id UUID REFERENCES users(id),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100),
    resource_id UUID,
    ip_address INET,
    user_agent TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes
CREATE INDEX idx_tenant_memberships_user ON tenant_memberships(user_id);
CREATE INDEX idx_tenant_memberships_tenant ON tenant_memberships(tenant_id);
CREATE INDEX idx_audit_logs_tenant ON audit_logs(tenant_id, created_at);
CREATE INDEX idx_api_keys_prefix ON api_keys(prefix);
```

---

## Core Internals

### Authentication System

#### Session-Based Auth Flow
```typescript
// middleware.ts
import { NextRequest, NextResponse } from 'next/server';
import { getSession } from '@/lib/auth';

export async function middleware(request: NextRequest) {
  const session = await getSession(request);
  const isAuthPage = request.nextUrl.pathname.startsWith('/login');
  const isProtectedRoute = request.nextUrl.pathname.startsWith('/dashboard');

  if (isProtectedRoute && !session) {
    const url = new URL('/login', request.url);
    url.searchParams.set('callbackUrl', request.nextUrl.pathname);
    return NextResponse.redirect(url);
  }

  if (isAuthPage && session) {
    return NextResponse.redirect(new URL('/dashboard', request.url));
  }

  // Add tenant context to headers
  if (session?.tenantId) {
    const headers = new Headers(request.headers);
    headers.set('x-tenant-id', session.tenantId);
    return NextResponse.next({ headers });
  }

  return NextResponse.next();
}

export const config = {
  matcher: ['/((?!api|_next/static|_next/image|favicon.ico).*)'],
};
```

```python
# Django auth backend
from django.contrib.auth.backends import BaseBackend
from django.contrib.auth import get_user_model
from apps.tenants.models import TenantMembership

User = get_user_model()

class TenantAuthBackend(BaseBackend):
    def authenticate(self, request, email=None, password=None, tenant_slug=None):
        try:
            user = User.objects.get(email=email)
            if user.check_password(password):
                if tenant_slug:
                    # Verify user belongs to tenant
                    membership = TenantMembership.objects.filter(
                        user=user,
                        tenant__slug=tenant_slug
                    ).first()
                    if not membership:
                        return None
                return user
        except User.DoesNotExist:
            return None

    def get_user(self, user_id):
        try:
            return User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return None
```

### RBAC Permission System

```python
# permissions.py
from functools import wraps
from rest_framework import permissions

class Permission:
    """Permission definition"""
    def __init__(self, resource: str, action: str):
        self.resource = resource
        self.action = action

    def __str__(self):
        return f"{self.resource}:{self.action}"

# Define permissions
class Permissions:
    # User permissions
    USER_READ = Permission("user", "read")
    USER_WRITE = Permission("user", "write")
    USER_DELETE = Permission("user", "delete")
    USER_INVITE = Permission("user", "invite")

    # Billing permissions
    BILLING_READ = Permission("billing", "read")
    BILLING_WRITE = Permission("billing", "write")

    # Settings permissions
    SETTINGS_READ = Permission("settings", "read")
    SETTINGS_WRITE = Permission("settings", "write")

# Role definitions
ROLE_PERMISSIONS = {
    "owner": [
        Permissions.USER_READ, Permissions.USER_WRITE,
        Permissions.USER_DELETE, Permissions.USER_INVITE,
        Permissions.BILLING_READ, Permissions.BILLING_WRITE,
        Permissions.SETTINGS_READ, Permissions.SETTINGS_WRITE,
    ],
    "admin": [
        Permissions.USER_READ, Permissions.USER_WRITE,
        Permissions.USER_INVITE,
        Permissions.SETTINGS_READ, Permissions.SETTINGS_WRITE,
    ],
    "member": [
        Permissions.USER_READ,
        Permissions.SETTINGS_READ,
    ],
}

class HasPermission(permissions.BasePermission):
    def __init__(self, required_permission: Permission):
        self.required_permission = required_permission

    def has_permission(self, request, view):
        if not request.user.is_authenticated:
            return False

        # Get user's membership in current tenant
        tenant_id = request.headers.get('X-Tenant-ID')
        membership = request.user.memberships.filter(
            tenant_id=tenant_id
        ).first()

        if not membership:
            return False

        # Check role permissions
        role_perms = ROLE_PERMISSIONS.get(membership.role, [])
        if self.required_permission in role_perms:
            return True

        # Check custom permissions
        if str(self.required_permission) in membership.permissions:
            return True

        return False

# Usage in views
class UserViewSet(viewsets.ModelViewSet):
    permission_classes = [IsAuthenticated]

    def get_permissions(self):
        if self.action == 'create':
            return [HasPermission(Permissions.USER_INVITE)]
        if self.action in ['update', 'partial_update']:
            return [HasPermission(Permissions.USER_WRITE)]
        if self.action == 'destroy':
            return [HasPermission(Permissions.USER_DELETE)]
        return [HasPermission(Permissions.USER_READ)]
```

### Billing Integration

```typescript
// lib/billing.ts
import Stripe from 'stripe';

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY!, {
  apiVersion: '2023-10-16',
});

export async function createCheckoutSession(
  tenantId: string,
  priceId: string,
  userId: string
): Promise<Stripe.Checkout.Session> {
  // Get or create Stripe customer
  const tenant = await db.tenant.findUnique({ where: { id: tenantId } });
  let customerId = tenant?.stripeCustomerId;

  if (!customerId) {
    const customer = await stripe.customers.create({
      metadata: { tenantId },
    });
    customerId = customer.id;
    await db.tenant.update({
      where: { id: tenantId },
      data: { stripeCustomerId: customerId },
    });
  }

  // Create checkout session
  const session = await stripe.checkout.sessions.create({
    customer: customerId,
    payment_method_types: ['card'],
    line_items: [{ price: priceId, quantity: 1 }],
    mode: 'subscription',
    success_url: `${process.env.NEXT_PUBLIC_APP_URL}/billing?success=true`,
    cancel_url: `${process.env.NEXT_PUBLIC_APP_URL}/billing?canceled=true`,
    metadata: {
      tenantId,
      userId,
    },
  });

  return session;
}

export async function createBillingPortalSession(
  tenantId: string
): Promise<Stripe.BillingPortal.Session> {
  const tenant = await db.tenant.findUnique({ where: { id: tenantId } });

  if (!tenant?.stripeCustomerId) {
    throw new Error('No billing account found');
  }

  return stripe.billingPortal.sessions.create({
    customer: tenant.stripeCustomerId,
    return_url: `${process.env.NEXT_PUBLIC_APP_URL}/billing`,
  });
}

// Webhook handler
export async function handleStripeWebhook(
  event: Stripe.Event
): Promise<void> {
  switch (event.type) {
    case 'checkout.session.completed': {
      const session = event.data.object as Stripe.Checkout.Session;
      const tenantId = session.metadata?.tenantId;

      if (tenantId && session.subscription) {
        await db.tenant.update({
          where: { id: tenantId },
          data: {
            stripeSubscriptionId: session.subscription as string,
            plan: 'pro', // Determine from price
          },
        });
      }
      break;
    }

    case 'customer.subscription.updated':
    case 'customer.subscription.deleted': {
      const subscription = event.data.object as Stripe.Subscription;
      const tenantId = subscription.metadata?.tenantId;

      if (tenantId) {
        await db.tenant.update({
          where: { id: tenantId },
          data: {
            plan: subscription.status === 'active' ? 'pro' : 'free',
          },
        });
      }
      break;
    }

    case 'invoice.payment_failed': {
      const invoice = event.data.object as Stripe.Invoice;
      // Send notification to tenant
      await sendPaymentFailedEmail(invoice);
      break;
    }
  }
}
```

### Multi-Tenant Data Access

```python
# Django middleware for tenant context
class TenantMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Extract tenant from subdomain or header
        tenant_slug = self.get_tenant_from_request(request)

        if tenant_slug:
            try:
                tenant = Tenant.objects.get(slug=tenant_slug)
                request.tenant = tenant

                # Set connection schema for PostgreSQL schemas approach
                # connection.set_tenant(tenant)
            except Tenant.DoesNotExist:
                return HttpResponseNotFound("Tenant not found")
        else:
            request.tenant = None

        response = self.get_response(request)
        return response

    def get_tenant_from_request(self, request):
        # Check header first (for API calls)
        tenant_id = request.headers.get('X-Tenant-ID')
        if tenant_id:
            return tenant_id

        # Check subdomain
        host = request.get_host().split(':')[0]
        parts = host.split('.')

        if len(parts) > 2:  # subdomain.domain.tld
            return parts[0]

        return None


# Base manager with tenant filtering
class TenantManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset()

    def for_tenant(self, tenant):
        return self.get_queryset().filter(tenant=tenant)


# Base model
class TenantModel(models.Model):
    tenant = models.ForeignKey(
        'tenants.Tenant',
        on_delete=models.CASCADE,
        related_name='+'
    )

    objects = TenantManager()

    class Meta:
        abstract = True


# View mixin
class TenantViewMixin:
    def get_queryset(self):
        return super().get_queryset().filter(
            tenant=self.request.tenant
        )

    def perform_create(self, serializer):
        serializer.save(tenant=self.request.tenant)
```

---

## Enterprise Features

### 1. Infrastructure as Code (Terraform)

```hcl
# main.tf
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket         = "terraform-state-saas"
    key            = "prod/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-locks"
  }
}

# VPC
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"

  name = "saas-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["us-east-1a", "us-east-1b", "us-east-1c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  enable_nat_gateway = true
  single_nat_gateway = false

  tags = {
    Environment = "production"
  }
}

# RDS PostgreSQL
module "rds" {
  source = "terraform-aws-modules/rds/aws"

  identifier = "saas-db"

  engine               = "postgres"
  engine_version       = "15"
  family               = "postgres15"
  major_engine_version = "15"
  instance_class       = "db.r6g.large"

  allocated_storage     = 100
  max_allocated_storage = 500

  db_name  = "saas"
  username = "admin"
  port     = 5432

  multi_az               = true
  db_subnet_group_name   = module.vpc.database_subnet_group
  vpc_security_group_ids = [module.security_group.security_group_id]

  backup_retention_period = 30
  skip_final_snapshot     = false
  deletion_protection     = true

  performance_insights_enabled = true
  monitoring_interval         = 60
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"

  cluster_name    = "saas-cluster"
  cluster_version = "1.28"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  eks_managed_node_groups = {
    general = {
      desired_size = 3
      min_size     = 2
      max_size     = 10

      instance_types = ["t3.large"]
      capacity_type  = "ON_DEMAND"
    }
  }

  cluster_addons = {
    coredns    = { most_recent = true }
    kube-proxy = { most_recent = true }
    vpc-cni    = { most_recent = true }
  }
}

# ElastiCache Redis
resource "aws_elasticache_cluster" "redis" {
  cluster_id           = "saas-redis"
  engine               = "redis"
  node_type            = "cache.r6g.large"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  port                 = 6379
  subnet_group_name    = aws_elasticache_subnet_group.redis.name
}
```

### 2. Secret Management

```yaml
# Kubernetes External Secrets
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: saas-secrets
spec:
  refreshInterval: 1h
  secretStoreRef:
    kind: SecretStore
    name: aws-secrets-manager
  target:
    name: saas-secrets
    creationPolicy: Owner
  data:
    - secretKey: DATABASE_URL
      remoteRef:
        key: prod/saas/database
        property: url
    - secretKey: STRIPE_SECRET_KEY
      remoteRef:
        key: prod/saas/stripe
        property: secret_key
    - secretKey: JWT_SECRET
      remoteRef:
        key: prod/saas/auth
        property: jwt_secret
```

```python
# Django settings with secret loading
import boto3
from botocore.exceptions import ClientError
import json

def get_secret(secret_name):
    client = boto3.client('secretsmanager')
    try:
        response = client.get_secret_value(SecretId=secret_name)
        return json.loads(response['SecretString'])
    except ClientError as e:
        raise e

# Load secrets in production
if ENVIRONMENT == 'production':
    secrets = get_secret('prod/saas/django')
    SECRET_KEY = secrets['secret_key']
    DATABASE_URL = secrets['database_url']
else:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret')
```

### 3. Logging with Redaction

```python
# logging_utils.py
import re
import logging

class SensitiveDataFilter(logging.Filter):
    """Filter to redact sensitive data from logs"""

    PATTERNS = [
        (r'password["\']?\s*[:=]\s*["\']?[\w@#$%^&*]+', 'password=***REDACTED***'),
        (r'api[_-]?key["\']?\s*[:=]\s*["\']?[\w-]+', 'api_key=***REDACTED***'),
        (r'token["\']?\s*[:=]\s*["\']?[\w.-]+', 'token=***REDACTED***'),
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '***EMAIL***'),
        (r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '***CARD***'),
        (r'sk_live_[\w]+', '***STRIPE_KEY***'),
        (r'whsec_[\w]+', '***WEBHOOK_SECRET***'),
    ]

    def filter(self, record):
        if isinstance(record.msg, str):
            for pattern, replacement in self.PATTERNS:
                record.msg = re.sub(pattern, replacement, record.msg, flags=re.IGNORECASE)

        if record.args:
            record.args = self._redact_args(record.args)

        return True

    def _redact_args(self, args):
        if isinstance(args, dict):
            return {k: self._redact_value(v) for k, v in args.items()}
        elif isinstance(args, (list, tuple)):
            return type(args)(self._redact_value(v) for v in args)
        return args

    def _redact_value(self, value):
        if isinstance(value, str):
            for pattern, replacement in self.PATTERNS:
                value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)
        return value

# Configure logging
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'filters': {
        'sensitive_data': {
            '()': SensitiveDataFilter,
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'filters': ['sensitive_data'],
            'formatter': 'json',
        },
    },
    'formatters': {
        'json': {
            '()': 'pythonjsonlogger.jsonlogger.JsonFormatter',
            'format': '%(asctime)s %(levelname)s %(name)s %(message)s'
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
}
```

### 4. CI/CD Pipeline

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

env:
  AWS_REGION: us-east-1
  ECR_REPOSITORY: saas-backend
  EKS_CLUSTER: saas-cluster

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run tests
        run: pytest --cov=apps --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    outputs:
      image: ${{ steps.build.outputs.image }}

    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}

      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build and push image
        id: build
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
          echo "image=$ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG" >> $GITHUB_OUTPUT

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}

      - name: Update kubeconfig
        run: aws eks update-kubeconfig --name $EKS_CLUSTER --region $AWS_REGION

      - name: Deploy to EKS
        run: |
          helm upgrade --install saas-backend ./helm/backend \
            --set image.repository=${{ needs.build-and-push.outputs.image }} \
            --set image.tag=${{ github.sha }} \
            --wait

      - name: Run migrations
        run: |
          kubectl exec -it deploy/saas-backend -- python manage.py migrate

      - name: Notify Slack
        if: always()
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          fields: repo,commit,author,action,workflow
```

### 5. Admin Dashboard

```typescript
// app/(dashboard)/admin/page.tsx
import { Suspense } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { getAdminMetrics } from '@/lib/admin';

export default async function AdminDashboard() {
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">Admin Dashboard</h1>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Suspense fallback={<MetricCardSkeleton />}>
          <MetricCards />
        </Suspense>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Recent Signups</CardTitle>
          </CardHeader>
          <CardContent>
            <Suspense fallback={<TableSkeleton />}>
              <RecentSignups />
            </Suspense>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Revenue</CardTitle>
          </CardHeader>
          <CardContent>
            <Suspense fallback={<ChartSkeleton />}>
              <RevenueChart />
            </Suspense>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Recent Activity</CardTitle>
        </CardHeader>
        <CardContent>
          <Suspense fallback={<TableSkeleton />}>
            <ActivityFeed />
          </Suspense>
        </CardContent>
      </Card>
    </div>
  );
}

async function MetricCards() {
  const metrics = await getAdminMetrics();

  return (
    <>
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium">Total Users</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{metrics.totalUsers}</div>
          <p className="text-xs text-muted-foreground">
            +{metrics.newUsersThisWeek} this week
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium">Active Tenants</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{metrics.activeTenants}</div>
          <p className="text-xs text-muted-foreground">
            {metrics.trialTenants} in trial
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium">MRR</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">
            ${metrics.mrr.toLocaleString()}
          </div>
          <p className="text-xs text-muted-foreground">
            +{metrics.mrrGrowth}% from last month
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium">Churn Rate</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{metrics.churnRate}%</div>
          <p className="text-xs text-muted-foreground">
            {metrics.churnCount} cancellations
          </p>
        </CardContent>
      </Card>
    </>
  );
}
```

---

## Performance Considerations

### Frontend
- **Code splitting:** Dynamic imports for routes and heavy components
- **Image optimization:** Next.js Image component, WebP/AVIF
- **Edge caching:** CDN caching for static assets
- **Prefetching:** Link prefetching for navigation

### Backend
- **Database indexing:** Composite indexes for common queries
- **Query optimization:** N+1 prevention, select_related/prefetch_related
- **Caching:** Redis for sessions, API responses, computed data
- **Connection pooling:** PgBouncer for database connections

### API Performance Targets
| Endpoint | Target P50 | Target P99 |
|----------|------------|------------|
| Login | <100ms | <500ms |
| Dashboard load | <200ms | <1s |
| List users | <50ms | <200ms |
| Create resource | <100ms | <500ms |

---

## Stretch Goals

### 1. Full Production Deployment Pipeline

```yaml
# ArgoCD Application
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: saas-platform
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/org/saas-platform
    targetRevision: HEAD
    path: k8s/overlays/production
  destination:
    server: https://kubernetes.default.svc
    namespace: production
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
```

### 2. Multi-Region Replication

```hcl
# Terraform for multi-region
module "primary_region" {
  source = "./modules/region"
  region = "us-east-1"
  role   = "primary"
}

module "secondary_region" {
  source = "./modules/region"
  region = "eu-west-1"
  role   = "secondary"

  primary_db_arn = module.primary_region.db_arn
}

# Route53 health checks and failover
resource "aws_route53_health_check" "primary" {
  fqdn              = module.primary_region.endpoint
  port              = 443
  type              = "HTTPS"
  resource_path     = "/health"
  failure_threshold = 3
  request_interval  = 30
}

resource "aws_route53_record" "api" {
  zone_id = aws_route53_zone.main.zone_id
  name    = "api"
  type    = "A"

  set_identifier = "primary"
  health_check_id = aws_route53_health_check.primary.id

  failover_routing_policy {
    type = "PRIMARY"
  }

  alias {
    name                   = module.primary_region.alb_dns
    zone_id                = module.primary_region.alb_zone_id
    evaluate_target_health = true
  }
}
```

---

## Testing Strategy

### Unit Tests
- React component testing (Jest, React Testing Library)
- API endpoint testing
- Business logic testing
- Utility function testing

### Integration Tests
- API integration tests
- Database integration tests
- External service mocking (Stripe, SendGrid)

### E2E Tests
- Critical user flows (Playwright/Cypress)
- Authentication flows
- Billing flows
- Admin operations

### Performance Tests
- Load testing (k6, Locust)
- Database query performance
- API endpoint benchmarks

---

## Implementation Phases

### Phase 1: Foundation (Week 1-3)
- [ ] Next.js project setup with TypeScript
- [ ] Django/Go backend setup
- [ ] Database schema and migrations
- [ ] Basic authentication (email/password)
- [ ] Session management

### Phase 2: Core Features (Week 4-6)
- [ ] User management (CRUD)
- [ ] Team/tenant management
- [ ] RBAC permissions
- [ ] Basic dashboard UI
- [ ] Settings pages

### Phase 3: Billing (Week 7-8)
- [ ] Stripe integration
- [ ] Subscription management
- [ ] Checkout flow
- [ ] Billing portal
- [ ] Usage metering (optional)

### Phase 4: Admin & Polish (Week 9-10)
- [ ] Admin dashboard
- [ ] Audit logging
- [ ] API key management
- [ ] Email notifications
- [ ] Comprehensive error handling

### Phase 5: DevOps (Week 11-13)
- [ ] Docker configuration
- [ ] Kubernetes manifests
- [ ] CI/CD pipeline
- [ ] Terraform infrastructure
- [ ] Monitoring and alerting

### Phase 6: Production (Week 14-16)
- [ ] Security hardening
- [ ] Performance optimization
- [ ] Load testing
- [ ] Documentation
- [ ] Launch checklist

---

## References

- [Next.js Documentation](https://nextjs.org/docs)
- [Django REST Framework](https://www.django-rest-framework.org/)
- [Stripe Documentation](https://stripe.com/docs)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [SaaS Boilerplate Patterns](https://github.com/topics/saas-boilerplate)
