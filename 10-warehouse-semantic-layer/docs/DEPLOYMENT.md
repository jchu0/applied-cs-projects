# Deployment Guide

## Overview

This guide covers deploying the Warehouse Semantic Layer in various environments, from local development to production cloud deployments.

## Prerequisites

### System Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended for production)
- 10GB disk space for application and logs
- Network access to target data warehouse

### Required Services

- Data warehouse (Snowflake, BigQuery, Redshift, or PostgreSQL)
- Redis (optional, for caching)
- PostgreSQL (optional, for metadata storage)

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-org/warehouse-semantic-layer.git
cd warehouse-semantic-layer
```

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

### 3. Environment Configuration

Create a `.env` file with your configuration:

```bash
# Application Settings
APP_ENV=production
APP_PORT=8080
APP_WORKERS=4
LOG_LEVEL=INFO

# Data Warehouse Configuration
WAREHOUSE_TYPE=snowflake
WAREHOUSE_ACCOUNT=your-account
WAREHOUSE_USER=semantic_layer_user
WAREHOUSE_PASSWORD=secure_password
WAREHOUSE_DATABASE=analytics
WAREHOUSE_SCHEMA=semantic_layer
WAREHOUSE_WAREHOUSE=compute_wh
WAREHOUSE_ROLE=semantic_layer_role

# Cache Configuration (Optional)
CACHE_ENABLED=true
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0
CACHE_TTL=3600

# Metadata Database (Optional)
METADATA_DB_URL=postgresql://user:password@localhost/semantic_layer

# Security
API_KEY_HEADER=X-API-Key
SECRET_KEY=your-secret-key-here
ALLOWED_ORIGINS=https://app.example.com

# Monitoring (Optional)
ENABLE_METRICS=true
METRICS_PORT=9090
DATADOG_API_KEY=
NEW_RELIC_LICENSE_KEY=
```

## Local Development

### Running Locally

```bash
# Activate virtual environment
source venv/bin/activate

# Run development server
python -m uvicorn semantic_layer.main:app --reload --port 8080

# Or use the CLI
python manage.py runserver
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=semantic_layer --cov-report=html

# Run specific test file
pytest tests/test_query_engine.py

# Run integration tests only
pytest tests/test_integration.py
```

## Docker Deployment

### Build Docker Image

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/semantic_layer ./semantic_layer
COPY config ./config

# Create non-root user
RUN useradd -m -u 1000 semantic && chown -R semantic:semantic /app
USER semantic

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8080/health')"

# Run application
CMD ["uvicorn", "semantic_layer.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### Build and Run

```bash
# Build image
docker build -t semantic-layer:latest .

# Run container
docker run -d \
  --name semantic-layer \
  -p 8080:8080 \
  --env-file .env \
  semantic-layer:latest

# View logs
docker logs -f semantic-layer
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    image: semantic-layer:latest
    ports:
      - "8080:8080"
    environment:
      - WAREHOUSE_TYPE=${WAREHOUSE_TYPE}
      - CACHE_ENABLED=true
      - REDIS_HOST=redis
    depends_on:
      - redis
      - postgres
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped

  postgres:
    image: postgres:14-alpine
    environment:
      - POSTGRES_DB=semantic_layer
      - POSTGRES_USER=semantic
      - POSTGRES_PASSWORD=secure_password
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api
    restart: unless-stopped

volumes:
  redis-data:
  postgres-data:
```

## Kubernetes Deployment

### Kubernetes Manifests

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: semantic-layer
  namespace: analytics
spec:
  replicas: 3
  selector:
    matchLabels:
      app: semantic-layer
  template:
    metadata:
      labels:
        app: semantic-layer
    spec:
      containers:
      - name: api
        image: semantic-layer:latest
        ports:
        - containerPort: 8080
        env:
        - name: WAREHOUSE_TYPE
          valueFrom:
            configMapKeyRef:
              name: semantic-config
              key: warehouse.type
        - name: WAREHOUSE_PASSWORD
          valueFrom:
            secretKeyRef:
              name: semantic-secrets
              key: warehouse.password
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: semantic-layer
  namespace: analytics
spec:
  selector:
    app: semantic-layer
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: semantic-layer
  namespace: analytics
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: semantic-layer
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Deploy to Kubernetes

```bash
# Create namespace
kubectl create namespace analytics

# Create config and secrets
kubectl create configmap semantic-config --from-env-file=.env -n analytics
kubectl create secret generic semantic-secrets \
  --from-literal=warehouse.password=$WAREHOUSE_PASSWORD \
  -n analytics

# Deploy application
kubectl apply -f deployment.yaml

# Check status
kubectl get pods -n analytics
kubectl get svc -n analytics

# View logs
kubectl logs -f deployment/semantic-layer -n analytics
```

## Cloud Deployments

### AWS Deployment

#### Using ECS Fargate

```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ECR_REGISTRY
docker tag semantic-layer:latest $ECR_REGISTRY/semantic-layer:latest
docker push $ECR_REGISTRY/semantic-layer:latest

# Create task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create service
aws ecs create-service \
  --cluster analytics \
  --service-name semantic-layer \
  --task-definition semantic-layer:1 \
  --desired-count 3 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

#### Using Lambda

```python
# serverless.yml
service: semantic-layer

provider:
  name: aws
  runtime: python3.10
  stage: ${opt:stage, 'prod'}
  region: us-east-1
  environment:
    WAREHOUSE_TYPE: snowflake
    WAREHOUSE_ACCOUNT: ${ssm:/semantic-layer/warehouse-account}

functions:
  api:
    handler: handler.main
    events:
      - http:
          path: /{proxy+}
          method: ANY
    timeout: 30
    memorySize: 1024

plugins:
  - serverless-python-requirements
  - serverless-api-gateway-caching
```

### Google Cloud Deployment

#### Using Cloud Run

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/$PROJECT_ID/semantic-layer

# Deploy to Cloud Run
gcloud run deploy semantic-layer \
  --image gcr.io/$PROJECT_ID/semantic-layer \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars WAREHOUSE_TYPE=bigquery \
  --min-instances 1 \
  --max-instances 10 \
  --memory 2Gi
```

### Azure Deployment

#### Using Container Instances

```bash
# Create resource group
az group create --name semantic-layer-rg --location eastus

# Create container instance
az container create \
  --resource-group semantic-layer-rg \
  --name semantic-layer \
  --image semantic-layer:latest \
  --dns-name-label semantic-layer \
  --ports 8080 \
  --environment-variables \
    WAREHOUSE_TYPE=synapse \
    CACHE_ENABLED=true \
  --secure-environment-variables \
    WAREHOUSE_PASSWORD=$WAREHOUSE_PASSWORD
```

## Production Configuration

### Performance Tuning

```python
# config/production.py
import os

# Application
WORKERS = int(os.getenv('APP_WORKERS', 4))
WORKER_CLASS = 'uvicorn.workers.UvicornWorker'
WORKER_CONNECTIONS = 1000
MAX_REQUESTS = 1000
MAX_REQUESTS_JITTER = 50
TIMEOUT = 120
KEEPALIVE = 5

# Database Connections
DB_POOL_SIZE = 20
DB_POOL_RECYCLE = 3600
DB_POOL_PRE_PING = True
DB_MAX_OVERFLOW = 40
DB_POOL_TIMEOUT = 30

# Cache Settings
CACHE_DEFAULT_TIMEOUT = 3600
CACHE_KEY_PREFIX = 'semantic:v1:'
CACHE_REDIS_POOL_SIZE = 50
CACHE_REDIS_POOL_MAX_CONNECTIONS = 100

# Rate Limiting
RATELIMIT_ENABLED = True
RATELIMIT_STORAGE_URL = 'redis://redis:6379'
RATELIMIT_STRATEGY = 'fixed-window'
RATELIMIT_DEFAULT = '100/minute'

# Monitoring
PROMETHEUS_ENABLED = True
STATSD_HOST = 'localhost'
STATSD_PORT = 8125
TRACING_ENABLED = True
JAEGER_AGENT_HOST = 'localhost'
JAEGER_AGENT_PORT = 6831
```

### Security Hardening

```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    server_name api.semantic-layer.example.com;

    # SSL Configuration
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Security Headers
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Rate Limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;

    # Proxy to application
    location / {
        proxy_pass http://semantic-layer:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

## Monitoring

### Health Checks

```python
# health_check.py
import sys
import requests
from datetime import datetime

def check_health(url):
    try:
        response = requests.get(f"{url}/health", timeout=5)
        data = response.json()

        if data['status'] != 'healthy':
            print(f"Unhealthy: {data}")
            sys.exit(1)

        if not data['warehouse']['connected']:
            print("Warehouse disconnected")
            sys.exit(1)

        print(f"Healthy at {datetime.now()}")
        sys.exit(0)

    except Exception as e:
        print(f"Health check failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_health("https://api.semantic-layer.example.com")
```

### Prometheus Metrics

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'semantic-layer'
    static_configs:
      - targets: ['semantic-layer:9090']
    metrics_path: '/metrics'
```

### Logging Configuration

```python
# logging.conf
[loggers]
keys=root,semantic_layer

[handlers]
keys=console,file,error_file

[formatters]
keys=detailed

[logger_root]
level=INFO
handlers=console,file

[logger_semantic_layer]
level=DEBUG
handlers=console,file,error_file
qualname=semantic_layer
propagate=0

[handler_console]
class=StreamHandler
level=INFO
formatter=detailed
args=(sys.stdout,)

[handler_file]
class=handlers.RotatingFileHandler
level=INFO
formatter=detailed
args=('/var/log/semantic-layer/app.log', 'a', 10485760, 10)

[handler_error_file]
class=handlers.RotatingFileHandler
level=ERROR
formatter=detailed
args=('/var/log/semantic-layer/error.log', 'a', 10485760, 10)

[formatter_detailed]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
datefmt=%Y-%m-%d %H:%M:%S
```

## Backup and Recovery

### Metric Definitions Backup

```bash
#!/bin/bash
# backup.sh

# Export metrics to YAML
curl -H "Authorization: Bearer $API_KEY" \
  https://api.semantic-layer.example.com/v1/export/metrics?format=yaml \
  > metrics-backup-$(date +%Y%m%d).yaml

# Backup to S3
aws s3 cp metrics-backup-*.yaml s3://backups/semantic-layer/

# Clean old backups
find . -name "metrics-backup-*.yaml" -mtime +30 -delete
```

### Disaster Recovery

```bash
#!/bin/bash
# restore.sh

# Download latest backup
aws s3 cp s3://backups/semantic-layer/metrics-backup-latest.yaml .

# Import metrics
curl -X POST \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d @metrics-backup-latest.yaml \
  https://api.semantic-layer.example.com/v1/import/metrics
```

## Troubleshooting

### Common Issues

1. **Connection timeout to warehouse**
   - Check network connectivity
   - Verify warehouse credentials
   - Increase connection timeout

2. **High memory usage**
   - Reduce worker count
   - Enable query result pagination
   - Increase cache TTL

3. **Slow queries**
   - Add warehouse-specific indexes
   - Use appropriate time grains
   - Limit dimension cardinality

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export SQL_ECHO=true

# Run with debug server
python -m semantic_layer.main --debug
```

## Maintenance

### Version Upgrades

```bash
# Backup current version
docker tag semantic-layer:latest semantic-layer:backup

# Deploy new version
docker pull semantic-layer:v2.0.0
docker tag semantic-layer:v2.0.0 semantic-layer:latest

# Rollback if needed
docker tag semantic-layer:backup semantic-layer:latest
```

### Database Migrations

```bash
# Run migrations
python manage.py migrate

# Create migration
python manage.py makemigrations

# Rollback migration
python manage.py migrate semantic_layer 0001_previous
```