# SaaS Web Platform - Deployment Guide

## Table of Contents
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [CI/CD Pipeline](#cicd-pipeline)
- [Monitoring & Logging](#monitoring--logging)
- [Security Checklist](#security-checklist)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended) or macOS
- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 20GB minimum free space

### Software Requirements
- Docker 24.0+
- Docker Compose 2.20+
- Node.js 18+ and npm 9+
- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- Git 2.30+

### Cloud Requirements (for production)
- AWS/GCP/Azure account
- Kubernetes cluster (EKS/GKE/AKS)
- Domain name with DNS management
- SSL certificates (or use Let's Encrypt)

## Environment Setup

### 1. Clone Repository
```bash
git clone https://github.com/your-org/saas-platform.git
cd saas-platform
```

### 2. Environment Variables

Create environment files from templates:
```bash
cp .env.example .env
cp frontend/.env.local.example frontend/.env.local
cp backend/.env.example backend/.env
```

#### Backend Environment Variables (backend/.env)
```bash
# Django Settings
SECRET_KEY=your-secret-key-here
DEBUG=False
ALLOWED_HOSTS=api.yourdomain.com,localhost

# Database
DATABASE_URL=postgresql://user:password@db:5432/saas_platform
REDIS_URL=redis://redis:6379/0

# AWS Services
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_STORAGE_BUCKET_NAME=saas-platform-files
AWS_REGION=us-east-1

# Email
EMAIL_BACKEND=django.core.mail.backends.smtp.EmailBackend
EMAIL_HOST=smtp.sendgrid.net
EMAIL_PORT=587
EMAIL_HOST_USER=apikey
EMAIL_HOST_PASSWORD=your-sendgrid-api-key
DEFAULT_FROM_EMAIL=noreply@yourdomain.com

# Stripe
STRIPE_PUBLIC_KEY=pk_live_...
STRIPE_SECRET_KEY=sk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...

# OAuth
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-secret
GITHUB_CLIENT_ID=your-github-client-id
GITHUB_CLIENT_SECRET=your-github-secret

# Sentry (optional)
SENTRY_DSN=https://...@sentry.io/...

# Celery
CELERY_BROKER_URL=redis://redis:6379/1
CELERY_RESULT_BACKEND=redis://redis:6379/2
```

#### Frontend Environment Variables (frontend/.env.local)
```bash
# API Configuration
NEXT_PUBLIC_API_URL=https://api.yourdomain.com
NEXT_PUBLIC_WS_URL=wss://api.yourdomain.com/ws

# Public Keys
NEXT_PUBLIC_STRIPE_PUBLIC_KEY=pk_live_...
NEXT_PUBLIC_GOOGLE_CLIENT_ID=your-google-client-id
NEXT_PUBLIC_GITHUB_CLIENT_ID=your-github-client-id

# Analytics (optional)
NEXT_PUBLIC_GA_MEASUREMENT_ID=G-XXXXXXXXXX
NEXT_PUBLIC_MIXPANEL_TOKEN=your-mixpanel-token

# Feature Flags
NEXT_PUBLIC_ENABLE_OAUTH=true
NEXT_PUBLIC_ENABLE_ANALYTICS=true
```

## Local Development

### 1. Using Docker Compose

Start all services:
```bash
docker-compose up -d
```

Initialize database:
```bash
docker-compose exec backend python manage.py migrate
docker-compose exec backend python manage.py createsuperuser
docker-compose exec backend python manage.py collectstatic --noinput
```

Load sample data (optional):
```bash
docker-compose exec backend python manage.py loaddata fixtures/sample_data.json
```

Access services:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- Admin Panel: http://localhost:8000/admin
- Redis Commander: http://localhost:8081
- PostgreSQL: localhost:5432

### 2. Without Docker

#### Backend Setup
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize database
python manage.py migrate
python manage.py createsuperuser
python manage.py collectstatic

# Start development server
python manage.py runserver

# In another terminal, start Celery worker
celery -A config worker -l info

# Start Celery beat (for scheduled tasks)
celery -A config beat -l info
```

#### Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
npm start
```

### 3. Database Setup

Create database and user:
```sql
CREATE DATABASE saas_platform;
CREATE USER saas_user WITH PASSWORD 'your-password';
GRANT ALL PRIVILEGES ON DATABASE saas_platform TO saas_user;
ALTER DATABASE saas_platform OWNER TO saas_user;
```

Run migrations:
```bash
python manage.py migrate
```

Create indexes for performance:
```sql
-- User queries
CREATE INDEX idx_users_email ON auth_user(email);
CREATE INDEX idx_users_organization ON organization_membership(user_id, organization_id);

-- Project queries
CREATE INDEX idx_projects_org ON projects_project(organization_id, status);
CREATE INDEX idx_projects_owner ON projects_project(owner_id);

-- Subscription queries
CREATE INDEX idx_subscriptions_user ON subscriptions_subscription(user_id, status);
CREATE INDEX idx_subscriptions_org ON subscriptions_subscription(organization_id);
```

## Docker Deployment

### 1. Build Images

```bash
# Build all images
docker-compose -f docker-compose.prod.yml build

# Or build individually
docker build -t saas-platform/backend:latest ./backend
docker build -t saas-platform/frontend:latest ./frontend
```

### 2. Push to Registry

```bash
# Tag images
docker tag saas-platform/backend:latest your-registry/saas-platform/backend:latest
docker tag saas-platform/frontend:latest your-registry/saas-platform/frontend:latest

# Push to registry
docker push your-registry/saas-platform/backend:latest
docker push your-registry/saas-platform/frontend:latest
```

### 3. Deploy with Docker Compose

```bash
# Production deployment
docker-compose -f docker-compose.prod.yml up -d

# Scale services
docker-compose -f docker-compose.prod.yml up -d --scale backend=3 --scale worker=2
```

### 4. Docker Swarm Deployment

Initialize swarm:
```bash
docker swarm init
```

Deploy stack:
```bash
docker stack deploy -c docker-stack.yml saas-platform
```

Scale services:
```bash
docker service scale saas-platform_backend=3 saas-platform_worker=2
```

## Cloud Deployment

### AWS Deployment

#### 1. Setup Infrastructure with Terraform

```hcl
# terraform/main.tf
provider "aws" {
  region = var.aws_region
}

module "vpc" {
  source = "./modules/vpc"
  cidr_block = "10.0.0.0/16"
  availability_zones = ["us-east-1a", "us-east-1b"]
}

module "eks" {
  source = "./modules/eks"
  vpc_id = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnet_ids
  cluster_name = "saas-platform"
  node_groups = {
    main = {
      desired_capacity = 3
      max_capacity     = 10
      min_capacity     = 2
      instance_types   = ["t3.medium"]
    }
  }
}

module "rds" {
  source = "./modules/rds"
  vpc_id = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnet_ids
  engine = "postgres"
  engine_version = "15.3"
  instance_class = "db.t3.medium"
  allocated_storage = 100
  database_name = "saas_platform"
}

module "elasticache" {
  source = "./modules/elasticache"
  vpc_id = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnet_ids
  node_type = "cache.t3.micro"
  num_cache_nodes = 2
}
```

Apply Terraform:
```bash
cd terraform
terraform init
terraform plan
terraform apply
```

#### 2. Deploy to EKS

```bash
# Configure kubectl
aws eks update-kubeconfig --region us-east-1 --name saas-platform

# Create namespace
kubectl create namespace saas-platform

# Create secrets
kubectl create secret generic backend-secrets \
  --from-env-file=backend/.env \
  -n saas-platform

# Apply Kubernetes manifests
kubectl apply -f k8s/ -n saas-platform

# Check deployment status
kubectl get pods -n saas-platform
kubectl get services -n saas-platform
```

#### 3. Setup Load Balancer

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: saas-platform-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - api.yourdomain.com
    - app.yourdomain.com
    secretName: saas-platform-tls
  rules:
  - host: api.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: backend-service
            port:
              number: 8000
  - host: app.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend-service
            port:
              number: 3000
```

### Google Cloud Platform Deployment

#### 1. Setup with gcloud CLI

```bash
# Set project
gcloud config set project your-project-id

# Create GKE cluster
gcloud container clusters create saas-platform \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type n1-standard-2 \
  --enable-autoscaling \
  --min-nodes 2 \
  --max-nodes 10

# Get credentials
gcloud container clusters get-credentials saas-platform --zone us-central1-a

# Create Cloud SQL instance
gcloud sql instances create saas-platform-db \
  --database-version=POSTGRES_15 \
  --tier=db-n1-standard-2 \
  --region=us-central1

# Create Memorystore Redis instance
gcloud redis instances create saas-platform-cache \
  --size=1 \
  --region=us-central1 \
  --redis-version=redis_7_0
```

#### 2. Deploy Application

```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/your-project-id/saas-platform-backend ./backend
gcloud builds submit --tag gcr.io/your-project-id/saas-platform-frontend ./frontend

# Deploy to GKE
kubectl apply -f k8s/gcp/
```

### Azure Deployment

```bash
# Create resource group
az group create --name saas-platform-rg --location eastus

# Create AKS cluster
az aks create \
  --resource-group saas-platform-rg \
  --name saas-platform-aks \
  --node-count 3 \
  --enable-addons monitoring \
  --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group saas-platform-rg --name saas-platform-aks

# Create Azure Database for PostgreSQL
az postgres server create \
  --resource-group saas-platform-rg \
  --name saas-platform-db \
  --sku-name B_Gen5_2 \
  --version 15

# Create Azure Cache for Redis
az redis create \
  --resource-group saas-platform-rg \
  --name saas-platform-cache \
  --sku Standard \
  --vm-size c1
```

## CI/CD Pipeline

### GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]
  workflow_dispatch:

env:
  AWS_REGION: us-east-1
  ECR_REPOSITORY: saas-platform

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Setup Node
        uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Run Backend Tests
        run: |
          cd backend
          pip install -r requirements-test.txt
          pytest --cov=apps --cov-report=xml

      - name: Run Frontend Tests
        run: |
          cd frontend
          npm ci
          npm test -- --coverage

      - name: SonarCloud Scan
        uses: SonarSource/sonarcloud-github-action@master
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}

      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v1

      - name: Build and push Backend
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY/backend:$IMAGE_TAG ./backend
          docker push $ECR_REGISTRY/$ECR_REPOSITORY/backend:$IMAGE_TAG

      - name: Build and push Frontend
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY/frontend:$IMAGE_TAG ./frontend
          docker push $ECR_REGISTRY/$ECR_REPOSITORY/frontend:$IMAGE_TAG

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}

      - name: Update kubeconfig
        run: |
          aws eks update-kubeconfig --region ${{ env.AWS_REGION }} --name saas-platform

      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/backend backend=${{ steps.login-ecr.outputs.registry }}/${{ env.ECR_REPOSITORY }}/backend:${{ github.sha }} -n saas-platform
          kubectl set image deployment/frontend frontend=${{ steps.login-ecr.outputs.registry }}/${{ env.ECR_REPOSITORY }}/frontend:${{ github.sha }} -n saas-platform
          kubectl rollout status deployment/backend -n saas-platform
          kubectl rollout status deployment/frontend -n saas-platform

      - name: Run smoke tests
        run: |
          ./scripts/smoke-tests.sh

      - name: Notify Slack
        if: always()
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {
              "text": "Deployment ${{ job.status }}: ${{ github.event.head_commit.message }}",
              "blocks": [
                {
                  "type": "section",
                  "text": {
                    "type": "mrkdwn",
                    "text": "*Deployment ${{ job.status }}*\nCommit: `${{ github.sha }}`\nAuthor: ${{ github.actor }}\nMessage: ${{ github.event.head_commit.message }}"
                  }
                }
              ]
            }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
```

### GitLab CI/CD

```yaml
# .gitlab-ci.yml
stages:
  - test
  - build
  - deploy

variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: ""

test:backend:
  stage: test
  image: python:3.11
  script:
    - cd backend
    - pip install -r requirements-test.txt
    - pytest --cov=apps --cov-report=xml
    - coverage report
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: backend/coverage.xml

test:frontend:
  stage: test
  image: node:18
  script:
    - cd frontend
    - npm ci
    - npm test -- --coverage
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: frontend/coverage/cobertura-coverage.xml

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
    - docker build -t $CI_REGISTRY_IMAGE/backend:$CI_COMMIT_SHA ./backend
    - docker build -t $CI_REGISTRY_IMAGE/frontend:$CI_COMMIT_SHA ./frontend
    - docker push $CI_REGISTRY_IMAGE/backend:$CI_COMMIT_SHA
    - docker push $CI_REGISTRY_IMAGE/frontend:$CI_COMMIT_SHA

deploy:production:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl config set-cluster k8s --server="$KUBE_URL" --insecure-skip-tls-verify=true
    - kubectl config set-credentials admin --token="$KUBE_TOKEN"
    - kubectl config set-context default --cluster=k8s --user=admin
    - kubectl config use-context default
    - kubectl set image deployment/backend backend=$CI_REGISTRY_IMAGE/backend:$CI_COMMIT_SHA -n saas-platform
    - kubectl set image deployment/frontend frontend=$CI_REGISTRY_IMAGE/frontend:$CI_COMMIT_SHA -n saas-platform
    - kubectl rollout status deployment/backend -n saas-platform
    - kubectl rollout status deployment/frontend -n saas-platform
  environment:
    name: production
    url: https://app.yourdomain.com
  only:
    - main
```

## Monitoring & Logging

### Prometheus Setup

```yaml
# k8s/monitoring/prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
data:
  prometheus.yml: |
    global:
      scrape_interval: 30s
      evaluation_interval: 30s

    scrape_configs:
      - job_name: 'backend'
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_label_app]
            action: keep
            regex: backend

      - job_name: 'frontend'
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_label_app]
            action: keep
            regex: frontend

      - job_name: 'postgres'
        static_configs:
          - targets: ['postgres-exporter:9187']

      - job_name: 'redis'
        static_configs:
          - targets: ['redis-exporter:9121']
```

### Grafana Dashboards

Import these dashboard IDs:
- Django Application: 9528
- PostgreSQL: 9628
- Redis: 11835
- Kubernetes Cluster: 8588
- NGINX Ingress: 9614

### ELK Stack Setup

```yaml
# k8s/logging/filebeat-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: filebeat-config
data:
  filebeat.yml: |
    filebeat.inputs:
    - type: container
      paths:
        - /var/log/containers/*.log
      processors:
        - add_kubernetes_metadata:
            host: ${NODE_NAME}
            matchers:
            - logs_path:
                logs_path: "/var/log/containers/"

    output.elasticsearch:
      hosts: ['${ELASTICSEARCH_HOST:elasticsearch}:${ELASTICSEARCH_PORT:9200}']
      username: ${ELASTICSEARCH_USERNAME}
      password: ${ELASTICSEARCH_PASSWORD}
      index: "saas-platform-%{+yyyy.MM.dd}"

    setup.template.name: "saas-platform"
    setup.template.pattern: "saas-platform-*"
```

### Application Metrics

```python
# backend/apps/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
request_count = Counter('app_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
request_duration = Histogram('app_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
active_users = Gauge('app_active_users', 'Active users')
subscription_revenue = Gauge('app_subscription_revenue', 'Monthly recurring revenue')

# Middleware to collect metrics
class MetricsMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        start_time = time.time()

        response = self.get_response(request)

        duration = time.time() - start_time
        request_count.labels(
            method=request.method,
            endpoint=request.path,
            status=response.status_code
        ).inc()

        request_duration.labels(
            method=request.method,
            endpoint=request.path
        ).observe(duration)

        return response
```

### Health Checks

```python
# backend/apps/health/views.py
from django.http import JsonResponse
from django.db import connection
from django.core.cache import cache
import redis

def health_check(request):
    """Basic health check endpoint"""
    return JsonResponse({'status': 'ok'})

def readiness_check(request):
    """Comprehensive readiness check"""
    checks = {}

    # Database check
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
        checks['database'] = 'ok'
    except Exception as e:
        checks['database'] = f'error: {str(e)}'

    # Redis check
    try:
        cache.set('health_check', 'ok', 1)
        cache.get('health_check')
        checks['redis'] = 'ok'
    except Exception as e:
        checks['redis'] = f'error: {str(e)}'

    # External services check
    # Add checks for S3, Stripe, etc.

    status = all(v == 'ok' for v in checks.values())
    return JsonResponse({
        'status': 'ok' if status else 'error',
        'checks': checks
    }, status=200 if status else 503)
```

## Security Checklist

### Pre-Deployment Security Audit

- [ ] All secrets are stored in environment variables or secret management system
- [ ] SSL/TLS certificates are valid and properly configured
- [ ] CORS settings are restrictive and only allow trusted origins
- [ ] Rate limiting is enabled on all endpoints
- [ ] SQL injection protection is active
- [ ] XSS protection headers are set
- [ ] CSRF tokens are required for state-changing operations
- [ ] File upload restrictions are in place
- [ ] Authentication tokens have appropriate expiration times
- [ ] Password requirements meet security standards
- [ ] Two-factor authentication is available
- [ ] Audit logging is enabled for sensitive operations
- [ ] Backup encryption is enabled
- [ ] Network security groups/firewalls are properly configured
- [ ] Vulnerability scanning has been performed
- [ ] Dependencies are up to date and free from known vulnerabilities
- [ ] Docker images are scanned for vulnerabilities
- [ ] Kubernetes RBAC is properly configured
- [ ] Secrets are rotated regularly
- [ ] Monitoring and alerting are configured for security events

### Security Headers

```python
# backend/apps/security/middleware.py
class SecurityHeadersMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)

        # Security headers
        response['X-Content-Type-Options'] = 'nosniff'
        response['X-Frame-Options'] = 'DENY'
        response['X-XSS-Protection'] = '1; mode=block'
        response['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        response['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        response['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com"
        response['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'

        return response
```

## Troubleshooting

### Common Issues

#### 1. Database Connection Issues
```bash
# Check database connectivity
docker-compose exec backend python manage.py dbshell

# Reset database connections
docker-compose restart backend

# Check database logs
docker-compose logs -f db
```

#### 2. Redis Connection Issues
```bash
# Check Redis connectivity
docker-compose exec backend python manage.py shell
>>> from django.core.cache import cache
>>> cache.set('test', 'value')
>>> cache.get('test')

# Clear Redis cache
docker-compose exec redis redis-cli FLUSHALL
```

#### 3. Static Files Not Loading
```bash
# Collect static files
docker-compose exec backend python manage.py collectstatic --noinput

# Check nginx configuration
docker-compose exec nginx nginx -t

# Verify static files directory
docker-compose exec backend ls -la /app/staticfiles/
```

#### 4. Celery Tasks Not Running
```bash
# Check Celery workers
docker-compose logs -f worker

# Restart Celery
docker-compose restart worker beat

# Monitor Celery tasks
docker-compose exec backend celery -A config inspect active
docker-compose exec backend celery -A config inspect registered
```

#### 5. Memory Issues
```bash
# Check memory usage
docker stats

# Increase memory limits in docker-compose.yml
services:
  backend:
    mem_limit: 512m
    mem_reservation: 256m

# For Kubernetes
kubectl top pods -n saas-platform
kubectl describe pod <pod-name> -n saas-platform
```

### Debug Mode

Enable debug mode for troubleshooting:
```bash
# backend/.env
DEBUG=True
LOG_LEVEL=DEBUG

# Restart services
docker-compose restart
```

### Logging

Check application logs:
```bash
# Docker logs
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f worker

# Kubernetes logs
kubectl logs -f deployment/backend -n saas-platform
kubectl logs -f deployment/frontend -n saas-platform

# Application logs (if using file logging)
tail -f backend/logs/django.log
tail -f backend/logs/celery.log
```

### Performance Debugging

```python
# Enable Django Debug Toolbar (development only)
INSTALLED_APPS += ['debug_toolbar']
MIDDLEWARE += ['debug_toolbar.middleware.DebugToolbarMiddleware']
INTERNAL_IPS = ['127.0.0.1', 'localhost']

# Enable SQL query logging
LOGGING = {
    'version': 1,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        'django.db.backends': {
            'handlers': ['console'],
            'level': 'DEBUG',
        },
    },
}
```

### Rollback Procedure

```bash
# Kubernetes rollback
kubectl rollout undo deployment/backend -n saas-platform
kubectl rollout undo deployment/frontend -n saas-platform

# Docker Swarm rollback
docker service rollback saas-platform_backend
docker service rollback saas-platform_frontend

# Database migration rollback
docker-compose exec backend python manage.py migrate app_name migration_name
```
