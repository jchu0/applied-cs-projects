# RAG Baseline Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the RAG Baseline system in various environments, from development to production-scale deployments.

## Prerequisites

### System Requirements

- **OS**: Ubuntu 20.04+ / CentOS 8+ / macOS 12+ / Windows Server 2019+
- **CPU**: Minimum 4 cores, recommended 8+ cores
- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: 50GB+ SSD (depends on document volume)
- **GPU**: Optional, NVIDIA GPU with CUDA 11.0+ for acceleration

### Software Requirements

- Python 3.9+
- Docker 20.10+ (for containerized deployment)
- PostgreSQL 13+ or Redis 6+ (for caching)
- NGINX or Apache (for reverse proxy)

## Installation

### 1. Local Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/rag-baseline.git
cd rag-baseline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Initialize vector database
python scripts/init_db.py

# Run development server
uvicorn ragbaseline.api:app --reload --host 0.0.0.0 --port 8000
```

### 2. Docker Deployment

#### Single Container

```dockerfile
# Dockerfile is included in the project
docker build -t rag-baseline:latest .

# Run container
docker run -d \
  --name rag-baseline \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -e OPENAI_API_KEY=your-key \
  -e DATABASE_URL=postgresql://user:pass@host/db \
  rag-baseline:latest
```

#### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  rag-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - VECTOR_DB_URL=http://chroma:8000
      - REDIS_URL=redis://redis:6379
    depends_on:
      - chroma
      - redis
    volumes:
      - ./data:/app/data

  chroma:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - chroma_data:/chroma/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  chroma_data:
  redis_data:
```

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f rag-api

# Stop services
docker-compose down
```

### 3. Kubernetes Deployment

#### Namespace and ConfigMap

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: rag-system

---
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: rag-config
  namespace: rag-system
data:
  CHUNK_SIZE: "512"
  CHUNK_OVERLAP: "50"
  RETRIEVAL_K: "5"
  EMBEDDING_MODEL: "text-embedding-ada-002"
  LLM_MODEL: "gpt-3.5-turbo"
```

#### Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
  namespace: rag-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-api
  template:
    metadata:
      labels:
        app: rag-api
    spec:
      containers:
      - name: rag-api
        image: your-registry/rag-baseline:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: rag-config
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

#### Service and Ingress

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: rag-api-service
  namespace: rag-system
spec:
  selector:
    app: rag-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer

---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rag-api-ingress
  namespace: rag-system
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: rag.your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: rag-api-service
            port:
              number: 80
```

```bash
# Deploy to Kubernetes
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f secrets.yaml  # Create this with your secrets
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml

# Check deployment status
kubectl get all -n rag-system

# View logs
kubectl logs -f deployment/rag-api -n rag-system

# Scale deployment
kubectl scale deployment/rag-api --replicas=5 -n rag-system
```

### 4. Cloud Deployments

#### AWS Deployment

```bash
# Using AWS Copilot
copilot app init rag-baseline
copilot env init --name production
copilot svc init --name api

# Deploy
copilot svc deploy --name api --env production
```

#### AWS ECS Task Definition

```json
{
  "family": "rag-baseline",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "containerDefinitions": [
    {
      "name": "rag-api",
      "image": "your-ecr-repo/rag-baseline:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "ENVIRONMENT", "value": "production"}
      ],
      "secrets": [
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:openai-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/rag-baseline",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### Google Cloud Platform

```yaml
# app.yaml for App Engine
runtime: python39
instance_class: F4

env_variables:
  ENVIRONMENT: "production"

automatic_scaling:
  target_cpu_utilization: 0.65
  min_instances: 2
  max_instances: 10

handlers:
- url: /.*
  script: auto
  secure: always
```

```bash
# Deploy to GCP
gcloud app deploy

# Or use Cloud Run
gcloud run deploy rag-baseline \
  --image gcr.io/your-project/rag-baseline \
  --platform managed \
  --allow-unauthenticated \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2
```

#### Azure Deployment

```bash
# Using Azure Container Instances
az container create \
  --resource-group rag-rg \
  --name rag-baseline \
  --image your-acr.azurecr.io/rag-baseline:latest \
  --dns-name-label rag-baseline \
  --ports 8000 \
  --cpu 2 \
  --memory 4 \
  --environment-variables \
    ENVIRONMENT=production \
  --secure-environment-variables \
    OPENAI_API_KEY=$OPENAI_API_KEY
```

## Configuration

### Environment Variables

```bash
# Core Configuration
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=info

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_KEY_HEADER=X-API-Key

# Database Configuration
DATABASE_URL=postgresql://user:pass@localhost/ragdb
REDIS_URL=redis://localhost:6379/0

# Vector Store Configuration
VECTOR_STORE_TYPE=chroma  # chroma|qdrant|pinecone|weaviate
CHROMA_HOST=localhost
CHROMA_PORT=8000
CHROMA_COLLECTION=rag_documents

# Embedding Configuration
EMBEDDING_PROVIDER=openai  # openai|cohere|huggingface|local
EMBEDDING_MODEL=text-embedding-ada-002
EMBEDDING_DIMENSION=1536
EMBEDDING_BATCH_SIZE=100

# LLM Configuration
LLM_PROVIDER=openai  # openai|anthropic|cohere|local
LLM_MODEL=gpt-3.5-turbo
LLM_MAX_TOKENS=500
LLM_TEMPERATURE=0.7

# Retrieval Configuration
RETRIEVAL_K=5
RETRIEVAL_SCORE_THRESHOLD=0.7
RETRIEVAL_METHOD=hybrid  # vector|hybrid|mmr

# Caching Configuration
CACHE_ENABLED=true
CACHE_TTL=3600
CACHE_MAX_SIZE=1000

# Security Configuration
CORS_ORIGINS=["https://app.example.com"]
RATE_LIMIT=100
RATE_LIMIT_PERIOD=60

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
ENABLE_TRACING=true
TRACING_ENDPOINT=http://jaeger:14268/api/traces
```

### Production Configuration

```python
# config/production.py
class ProductionConfig:
    # Performance
    WORKER_CLASS = "uvicorn.workers.UvicornWorker"
    WORKER_CONNECTIONS = 1000
    KEEPALIVE = 5

    # Security
    SSL_REDIRECT = True
    SECURE_HEADERS = {
        "X-Frame-Options": "DENY",
        "X-Content-Type-Options": "nosniff",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains"
    }

    # Database
    DB_POOL_SIZE = 20
    DB_MAX_OVERFLOW = 40
    DB_POOL_TIMEOUT = 30

    # Caching
    CACHE_TYPE = "redis"
    CACHE_REDIS_URL = "redis://redis:6379/0"

    # Monitoring
    SENTRY_DSN = "https://your-sentry-dsn"
    APM_ENABLED = True
```

## Monitoring and Logging

### Prometheus Metrics

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
query_counter = Counter('rag_queries_total', 'Total RAG queries')
query_duration = Histogram('rag_query_duration_seconds', 'Query duration')
active_connections = Gauge('rag_active_connections', 'Active connections')
cache_hits = Counter('rag_cache_hits_total', 'Cache hits')
```

### Logging Configuration

```python
# logging_config.py
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        },
        'json': {
            'class': 'pythonjsonlogger.jsonlogger.JsonFormatter',
            'format': '%(asctime)s %(name)s %(levelname)s %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'json' if ENVIRONMENT == 'production' else 'default',
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/var/log/rag/app.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'json',
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console', 'file']
    }
}
```

### Health Checks

```python
# health.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
async def health_check():
    """Basic health check"""
    return {"status": "healthy"}

@router.get("/ready")
async def readiness_check():
    """Readiness check including dependencies"""
    checks = {
        "database": await check_database(),
        "vector_store": await check_vector_store(),
        "cache": await check_cache(),
    }

    if all(checks.values()):
        return {"status": "ready", "checks": checks}
    else:
        return {"status": "not_ready", "checks": checks}, 503
```

## Scaling Strategies

### Horizontal Scaling

```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-api-hpa
  namespace: rag-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-api
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

### Load Balancing

```nginx
# nginx.conf
upstream rag_backend {
    least_conn;
    server rag-api-1:8000 weight=1;
    server rag-api-2:8000 weight=1;
    server rag-api-3:8000 weight=1;
    keepalive 32;
}

server {
    listen 80;
    server_name rag.example.com;

    location / {
        proxy_pass http://rag_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_buffering off;
        proxy_request_buffering off;

        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

## Backup and Recovery

### Database Backup

```bash
#!/bin/bash
# backup.sh

# Backup PostgreSQL
pg_dump $DATABASE_URL > backup_$(date +%Y%m%d_%H%M%S).sql

# Backup vector store
python scripts/backup_vectors.py --output backup_vectors_$(date +%Y%m%d).pkl

# Upload to S3
aws s3 cp backup_*.sql s3://your-backup-bucket/rag/
aws s3 cp backup_*.pkl s3://your-backup-bucket/rag/
```

### Disaster Recovery

```python
# disaster_recovery.py
import asyncio
from ragbaseline import recovery

async def restore_from_backup(backup_date: str):
    """Restore system from backup"""

    # Download backups
    await recovery.download_backup(backup_date)

    # Restore database
    await recovery.restore_database()

    # Restore vector index
    await recovery.restore_vectors()

    # Verify restoration
    await recovery.verify_restoration()

    print(f"System restored from {backup_date}")
```

## Security Hardening

### SSL/TLS Configuration

```python
# ssl_config.py
import ssl

ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
ssl_context.load_cert_chain('path/to/cert.pem', 'path/to/key.pem')

# For production
ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
ssl_context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
```

### API Key Management

```python
# api_key_management.py
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify API key"""
    if not await is_valid_api_key(api_key):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
```bash
# Increase container memory
docker run -m 8g rag-baseline:latest

# Or adjust Python memory
export PYTHONMALLOC=malloc
```

2. **Slow Query Performance**
```python
# Enable query caching
CACHE_ENABLED=true
CACHE_TTL=3600

# Optimize retrieval
RETRIEVAL_K=3  # Reduce number of retrieved docs
```

3. **Connection Timeouts**
```python
# Increase timeouts
httpx.AsyncClient(timeout=30.0)

# Add retry logic
from tenacity import retry, stop_after_attempt

@retry(stop=stop_after_attempt(3))
async def query_with_retry():
    pass
```

### Debug Mode

```bash
# Enable debug logging
export DEBUG=true
export LOG_LEVEL=debug

# Run with verbose output
python -m ragbaseline.api --verbose

# Profile performance
python -m cProfile -o profile.stats ragbaseline/api.py
```

## Performance Tuning

### Database Optimization

```sql
-- Create indexes
CREATE INDEX idx_documents_metadata ON documents USING GIN (metadata);
CREATE INDEX idx_chunks_document_id ON chunks (document_id);
CREATE INDEX idx_embeddings_chunk_id ON embeddings (chunk_id);

-- Optimize queries
VACUUM ANALYZE documents;
VACUUM ANALYZE chunks;
```

### Caching Strategy

```python
# cache_config.py
CACHE_STRATEGIES = {
    "embeddings": {
        "ttl": 86400,  # 24 hours
        "max_size": 10000
    },
    "queries": {
        "ttl": 3600,  # 1 hour
        "max_size": 1000
    },
    "documents": {
        "ttl": 7200,  # 2 hours
        "max_size": 500
    }
}
```

## Maintenance

### Regular Tasks

```bash
# Daily maintenance
0 2 * * * /opt/rag/scripts/cleanup_logs.sh
0 3 * * * /opt/rag/scripts/optimize_db.sh
0 4 * * * /opt/rag/scripts/backup.sh

# Weekly maintenance
0 5 * * 0 /opt/rag/scripts/full_backup.sh
0 6 * * 0 /opt/rag/scripts/update_embeddings.sh
```

### Updates and Upgrades

```bash
# Rolling update on Kubernetes
kubectl set image deployment/rag-api rag-api=your-registry/rag-baseline:v2.0 -n rag-system
kubectl rollout status deployment/rag-api -n rag-system

# Rollback if needed
kubectl rollout undo deployment/rag-api -n rag-system
```
